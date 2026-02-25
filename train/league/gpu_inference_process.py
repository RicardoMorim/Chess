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
- Server replies (to that worker's response queue): (request_id:str, legal_logits:list[float], value:float)

Notes:
- The server loads ONE model on GPU and batches requests.
- Workers never load the model (saves RAM and avoids CPU model inference).
"""

from __future__ import annotations

import logging
import time
import uuid
import threading
from queue import Empty
from typing import Any, Callable, Dict, List, Optional, Tuple

import chess
import torch
import torch.nn as nn

from .gpu_inference_server import GPUInferenceServer

logger = logging.getLogger(__name__)


class GPUInferenceClient:
    """Client used inside a self-play worker process."""

    def __init__(
        self,
        worker_id: int,
        request_queue,
        response_queue,
    ) -> None:
        self.worker_id = int(worker_id)
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._pending: Dict[str, Tuple[List[float], float]] = {}
        self._lock = threading.Lock()

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
        """Evaluate precomputed features on the GPU server.

        Args:
            features: numpy array or torch tensor shaped [C, 8, 8]
            legal_indices: policy indices aligned to legal moves order
            timeout_sec: max wait time
        """
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
                    # GPU process crashed
                    return None, 0.0
                    
                if rid == request_id:
                    if logits is None:
                        return None, 0.0
                    return logits, float(value)

                self._pending[rid] = (logits, float(value))


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
) -> None:
    """Entry point for the GPU inference server process."""
    try:
        torch.set_num_threads(1)

        # Build and load model
        model = model_constructor(**model_config)
        model.load_state_dict(model_state_dict, strict=False)

        server = GPUInferenceServer(model=model, device=device, batch_size=batch_size)

        pending: List[Tuple] = []
        last_flush = time.time()
        last_log = time.time()
        last_request_time = time.time()
        processed = 0

        logger.info(
            f"{variant}: GPU inference started (batch_size={batch_size}, post_wait_ms={post_batch_wait_ms})"
        )

        while True:
            # Block briefly waiting for requests
            try:
                item = request_queue.get(timeout=0.01)
            except Empty:
                item = "__EMPTY__"
                # Heartbeat: warn if no requests for 30 seconds
                if time.time() - last_request_time > 30.0:
                    logger.warning(f"{variant}: GPU inference idle for 30+ seconds (no requests)")
                    last_request_time = time.time()

            if item is None:
                break

            if item != "__EMPTY__":
                last_request_time = time.time()
                # legacy: (worker_id, request_id, fen)
                # fast:   (worker_id, request_id, features, legal_indices)
                pending.append(item)

            should_flush = (
                len(pending) >= batch_size
                or (pending and (time.time() - last_flush) > (post_batch_wait_ms / 1000.0))
            )

            if not should_flush:
                continue

            # Flush batch
            batch = pending
            pending = []
            last_flush = time.time()

            # Fast path: features already provided by workers
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
                        f"{variant}: GPU stats: evals={server.processed_evals}, batches={server.total_batches}, avg_batch={server.avg_batch_size:.1f}"
                    )
                except Exception:
                    pass
                processed = 0

    # Drain any pending (best effort) with neutral outputs
        for item in pending:
            try:
                worker_id = int(item[0])
                rid = str(item[1])
                response_queues[int(worker_id)].put((rid, [0.0], 0.0))
            except Exception:
                pass

        logger.info(f"{variant}: GPU inference process stopped")
    except Exception as gpu_process_error:
        logger.error(f"GPU inference process crashed: {gpu_process_error}", exc_info=True)
        # Try to notify all workers of failure (best effort)
        try:
            for q in response_queues:
                q.put(("__ERROR__", [0.0], 0.0))
        except Exception:
            pass
        raise
