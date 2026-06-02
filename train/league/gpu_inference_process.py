"""
Multiprocessing GPU Inference Service
====================================

Why this exists:
- `GPUInferenceServer` is thread-based (in-process queues), but league self-play uses
  multiprocessing worker processes.
- This module provides a simple MP queue protocol so workers can offload NN eval
  (policy logits for legal moves + value) to a single GPU-hosted process.

Protocol:
- Workers send:
    - legacy: (worker_id:int, request_id:str, fen:str)
    - fast:   (worker_id:int, request_id:str, features, legal_indices:list[int])
    - batch_fast: ("__BATCH__", worker_id:int, [request_ids], [(features, legal_indices), ...])
- Server replies (to that worker's response queue): (request_id:str, legal_logits:list[float], value:float)

Notes:
- The server loads ONE model on GPU and batches requests.
- Workers never load the model (saves RAM and avoids CPU model inference).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
import threading
from pathlib import Path
from queue import Empty
from typing import Any, Callable, Dict, List, Optional, Tuple

import chess
import torch
import torch.nn as nn

from .gpu_inference_server import GPUInferenceServer

logger = logging.getLogger(__name__)


def _should_flush_batch(pending_size: int, pending_since: Optional[float], batch_size: int, post_batch_wait_ms: int, now: Optional[float] = None) -> bool:
    """Return True when a pending inference batch should be flushed.

    Flush conditions:
    - reached target ``batch_size``
    - waited at least ``post_batch_wait_ms`` since first pending request
    """
    if pending_size <= 0:
        return False
    if pending_size >= batch_size:
        return True
    if pending_since is None:
        return False
    if now is None:
        now = time.time()
    return (now - pending_since) >= (post_batch_wait_ms / 1000.0)


class GPUInferenceClient:
    """Client used inside a self-play worker process.

    Supports both single-request mode and batched mode for higher GPU throughput.

    Batched mode accumulates requests until ``batch_flush_size`` is reached or
    ``collect_results()`` is called, then sends them all in one queue message so the
    server can do a single GPU forward pass instead of many small ones.
    """

    def __init__(
        self,
        worker_id: int,
        request_queue,
        response_queue,
        batch_flush_size: int = 8,       # accumulate this many before auto-flush
        collect_timeout_ms: int = 5000,   # ms to wait for each result in collect_results()
    ) -> None:
        self.worker_id = int(worker_id)
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._batch_flush_size = max(1, int(batch_flush_size))
        self._collect_timeout_ms = int(collect_timeout_ms)

        # Single-request mode (legacy)
        self._pending: Dict[str, Tuple[List[float], float]] = {}
        self._lock = threading.Lock()

        # Batched mode accumulators
        self._batch_ids: List[str] = []
        self._batch_features: List[Any] = []
        self._batch_indices: List[List[int]] = []
        self._batch_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Single-request API (legacy, unchanged behaviour)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        board: chess.Board,
        timeout_sec: float = 10.0,
    ) -> Tuple[Optional[List[float]], float]:
        """Return (legal_logits, value). legal_logits aligned to list(board.legal_moves)."""
        request_id = uuid.uuid4().hex
        fen = board.fen()
        self.request_queue.put((self.worker_id, request_id, fen))

        deadline = time.time() + float(timeout_sec)
        with self._lock:
            if request_id in self._pending:
                logits, value = self._pending.pop(request_id)
                return logits, value

            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None, 0.0

                try:
                    rid, logits, value = self.response_queue.get(timeout=remaining)
                except Empty:
                    return None, 0.0

                if rid == request_id:
                    return logits, float(value)

                # Another thread requested something else; store it.
                self._pending[rid] = (logits, float(value))

    def evaluate_features(
        self,
        features,
        legal_indices: List[int],
        timeout_sec: float = 10.0,
    ) -> Tuple[Optional[List[float]], float]:
        """Evaluate precomputed features on the GPU server (single-request mode)."""
        request_id = uuid.uuid4().hex
        self.request_queue.put((self.worker_id, request_id, features, list(legal_indices)))

        deadline = time.time() + float(timeout_sec)
        with self._lock:
            if request_id in self._pending:
                logits, value = self._pending.pop(request_id)
                if logits is None:
                    return None, 0.0
                return logits, value

            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None, 0.0

                try:
                    rid, logits, value = self.response_queue.get(timeout=remaining)
                except Empty:
                    return None, 0.0

                if rid == "__ERROR__":
                    return None, 0.0

                if rid == request_id:
                    if logits is None:
                        return None, 0.0
                    return logits, float(value)

                self._pending[rid] = (logits, float(value))

    # ------------------------------------------------------------------
    # Batched API — accumulates requests then flushes in one message
    # ------------------------------------------------------------------

    def evaluate_features_batched(
        self,
        features,
        legal_indices: List[int],
    ) -> str:
        """Queue a single board evaluation in batch mode.

        Returns the request_id so the caller can later retrieve the result via
        :meth:`collect_results`.  The request is sent to the server immediately
        once ``batch_flush_size`` boards have been accumulated, or when
        :meth:`flush` / :meth:`collect_results` is called.
        """
        request_id = uuid.uuid4().hex

        with self._batch_lock:
            self._batch_ids.append(request_id)
            self._batch_features.append(features)
            self._batch_indices.append(list(legal_indices))

            if len(self._batch_ids) >= self._batch_flush_size:
                self._send_batch_unlocked()

        return request_id

    def flush(self) -> None:
        """Send any accumulated batched requests immediately."""
        with self._batch_lock:
            if self._batch_ids:
                self._send_batch_unlocked()

    def collect_results(
        self,
        request_ids: List[str],
    ) -> Dict[str, Tuple[Optional[List[float]], float]]:
        """Collect results for a list of batched request IDs.

        Blocks until all requested results arrive (up to ``collect_timeout_ms``
        per result).  Results that time out get ``(None, 0.0)``.

        Returns:
            dict mapping request_id -> (legal_logits, value)
        """
        timeout_sec = self._collect_timeout_ms / 1000.0
        results: Dict[str, Tuple[Optional[List[float]], float]] = {}
        pending = set(request_ids)

        deadline = time.time() + timeout_sec * max(1, len(pending))

        with self._lock:
            # Check already-arrived single-request results first
            self._drain_pending_into(results, pending)

            while pending:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break

                rid, logits, value = self._recv_or_break(remaining)
                if rid is None:
                    break

                if rid == "__ERROR__":
                    self._fill_errors(results, pending)
                    break

                # Store for later retrieval (single-request path handles cross-thread)
                self._pending[rid] = (logits, float(value))
                if rid in pending:
                    log = logits if logits is not None else [0.0]
                    results[rid] = (log, float(value))
                    pending.discard(rid)

        # Fill missing with defaults
        for rid in request_ids:
            if rid not in results:
                results[rid] = (None, 0.0)

        return results

    def _drain_pending_into(
        self,
        results: Dict[str, Tuple[Optional[List[float]], float]],
        pending: set,
    ) -> None:
        """Move any already-arrived single-request results into *results*."""
        for rid in list(pending):
            if rid in self._pending:
                logits, value = self._pending.pop(rid)
                results[rid] = (logits, value)
                pending.discard(rid)

    def _recv_or_break(self, remaining: float) -> Tuple[Any, Any, Any]:
        """Try to receive from the response queue; return ``(None, None, None)`` on timeout."""
        try:
            rid, logits, value = self.response_queue.get(timeout=remaining)
            return rid, logits, value
        except Empty:
            return None, None, None

    def _fill_errors(self, results: Dict[str, Tuple[Optional[List[float]], float]], pending: set) -> None:
        """Mark all still-pending request IDs as errors."""
        for r in pending:
            results[r] = (None, 0.0)
        pending.clear()

    def get_batch_size(self) -> int:
        """Return the number of accumulated (unsent) batched requests."""
        with self._batch_lock:
            return len(self._batch_ids)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_batch_unlocked(self) -> None:
        """Send accumulated batch — caller must hold ``_batch_lock``."""
        if not self._batch_ids:
            return

        ids = list(self._batch_ids)
        feats = [f for f in self._batch_features]
        idxs = [list(i) for i in self._batch_indices]
        self._batch_ids.clear()
        self._batch_features.clear()
        self._batch_indices.clear()

        try:
            self.request_queue.put(("__BATCH__", self.worker_id, ids, feats, idxs))
        except Exception as e:
            logger.warning(f"Worker {self.worker_id}: failed to send batch: {e}")


def gpu_inference_server_main(
    model_state_dict: Dict[str, torch.Tensor],
    model_constructor: Callable[..., nn.Module],
    model_config: Dict[str, Any],
    device: str,
    request_queue,
    response_queues: List,
    batch_size: int = 32,
    post_batch_wait_ms: int = 10,
    variant: str = "unknown",
    stats_output_path: Optional[str] = None,
) -> None:
    """Entry point for the GPU inference server process."""
    try:
        torch.set_num_threads(1)

        # Build and load model
        model = model_constructor(**model_config)
        model.load_state_dict(model_state_dict, strict=False)

        server = GPUInferenceServer(model=model, device=device, batch_size=batch_size)

        pending: List[Tuple] = []
        pending_since: Optional[float] = None
        last_log = time.time()
        last_request_time = time.time()
        processed = 0
        flush_by_size = 0
        flush_by_wait = 0
        flush_by_batch_msg = 0
        stats_path = Path(stats_output_path) if stats_output_path else None

        def _write_stats_snapshot(force: bool = False) -> None:
            if stats_path is None:
                return
            snapshot = {
                "variant": variant,
                "timestamp": time.time(),
                "batch_size": batch_size,
                "post_batch_wait_ms": post_batch_wait_ms,
                "processed_evals": server.processed_evals,
                "total_batches": server.total_batches,
                "avg_batch_size": server.avg_batch_size,
                "flush_by_size": flush_by_size,
                "flush_by_wait": flush_by_wait,
                "flush_by_batch_msg": flush_by_batch_msg,
                "pending_size": len(pending),
                "last_request_age_sec": max(0.0, time.time() - last_request_time),
            }
            try:
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = stats_path.with_suffix(stats_path.suffix + ".tmp")
                tmp_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
                tmp_path.replace(stats_path)
            except Exception as write_err:
                if force:
                    logger.warning(f"{variant}: Failed to write GPU stats snapshot: {write_err}")

        logger.info(
            f"{variant}: GPU inference started (batch_size={batch_size}, post_wait_ms={post_batch_wait_ms})"
        )

        while True:
            wait_timeout = 0.01
            if pending and pending_since is not None:
                elapsed = time.time() - pending_since
                remaining = (post_batch_wait_ms / 1000.0) - elapsed
                wait_timeout = max(0.0005, min(0.01, remaining))

            # Block briefly waiting for requests
            try:
                item = request_queue.get(timeout=wait_timeout)
            except Empty:
                item = "__EMPTY__"
                if time.time() - last_request_time > 30.0:
                    logger.warning(f"{variant}: GPU inference idle for 30+ seconds (no requests)")
                    last_request_time = time.time()

            if item is None:
                break

            if item != "__EMPTY__":
                last_request_time = time.time()

                # Handle batch messages from client (client-side accumulated boards)
                if isinstance(item, tuple) and len(item) >= 5 and item[0] == "__BATCH__":
                    # (__BATCH__, worker_id, [request_ids], [(features, legal_indices), ...])
                    _, wid, req_ids, feats_list, idxs_list = item[:5]
                    for rid, features, legal_indices in zip(req_ids, feats_list, idxs_list):
                        pending.append((int(wid), str(rid), features, list(legal_indices)))
                    if not pending_since:
                        pending_since = time.time()
                    flush_by_batch_msg += 1
                else:
                    # legacy or fast single request
                    if not pending:
                        pending_since = time.time()
                    pending.append(item)

            now = time.time()
            should_flush = _should_flush_batch(
                pending_size=len(pending),
                pending_since=pending_since,
                batch_size=batch_size,
                post_batch_wait_ms=post_batch_wait_ms,
                now=now,
            )

            if not should_flush:
                continue

            if len(pending) >= batch_size:
                flush_by_size += 1
            else:
                flush_by_wait += 1

            # Flush batch
            batch = pending
            pending = []
            pending_since = None

            # Fast path: features already provided by workers (single or batched)
            if len(batch[0]) == 4:
                feats_list: List[torch.Tensor] = []
                indices_list: List[List[int]] = []
                meta: List[Tuple[int, str]] = []

                for worker_id, rid, features, legal_indices in batch:
                    meta.append((int(worker_id), str(rid)))
                    indices_list.append(list(legal_indices) if legal_indices is not None else [])

                    if isinstance(features, torch.Tensor):
                        feats_list.append(features.detach().cpu().float())
                    else:
                        feats_list.append(torch.tensor(features, dtype=torch.float32))

                feats = torch.stack(feats_list).to(server.device)

                with torch.no_grad():
                    policy_logits, values = server.model(feats)
                    values = values.squeeze(-1).detach().cpu().tolist()

                processed += len(meta)
                server.processed_evals += len(meta)
                server.total_batches += 1
                if server.total_batches > 0:
                    server.avg_batch_size = server.processed_evals / server.total_batches

                for i, (worker_id, rid) in enumerate(meta):
                    legal_indices = indices_list[i]
                    if not legal_indices:
                        legal_logits = [0.0]
                    else:
                        idx_tensor = torch.tensor(legal_indices, device=policy_logits.device, dtype=torch.long)
                        legal_logits = policy_logits[i, idx_tensor].detach().cpu().tolist()

                    try:
                        response_queues[int(worker_id)].put((rid, legal_logits, float(values[i])))
                    except Exception as e:
                        logger.warning(f"Failed to respond to worker={worker_id} rid={rid}: {e}")

            else:
                # Legacy path: FEN -> board -> server does feature extraction
                boards: List[chess.Board] = []
                for _worker_id, _rid, fen in batch:
                    try:
                        boards.append(chess.Board(fen))
                    except Exception:
                        boards.append(chess.Board())

                results = server.evaluate_batch(boards)
                processed += len(boards)

                for (worker_id, rid, _fen), (legal_logits, value) in zip(batch, results):
                    try:
                        out = (rid, legal_logits.tolist(), float(value))
                        response_queues[int(worker_id)].put(out)
                    except Exception as e:
                        logger.warning(f"Failed to respond to worker={worker_id} rid={rid}: {e}")

            if processed >= 1000 and (time.time() - last_log) > 10.0:
                last_log = time.time()
                try:
                    logger.info(
                        f"{variant}: GPU stats: evals={server.processed_evals}, batches={server.total_batches}, "
                        f"avg_batch={server.avg_batch_size:.1f}, flush_size={flush_by_size}, "
                        f"flush_wait={flush_by_wait}, flush_batch_msg={flush_by_batch_msg}"
                    )
                except Exception:
                    pass
                _write_stats_snapshot(force=True)
                processed = 0

        # --- exit cleanup (after while loop breaks) ---

        # Drain any pending (best effort) with neutral outputs
        for item in pending:
            try:
                if isinstance(item, tuple) and len(item) >= 3:
                    worker_id = int(item[0])
                    rid = str(item[1])
                    response_queues[int(worker_id)].put((rid, [0.0], 0.0))
            except Exception:
                pass

        _write_stats_snapshot(force=True)

        logger.info(f"{variant}: GPU inference process stopped")
    except Exception as gpu_process_error:
        logger.error(f"GPU inference process crashed: {gpu_process_error}", exc_info=True)
        try:
            for q in response_queues:
                q.put(("__ERROR__", [0.0], 0.0))
        except Exception:
            pass
        raise
