""" 
Multiprocessing GPU Inference Service
====================================

Why this exists:
- `GPUInferenceServer` is thread-based (in-process queues), but league self-play uses
  multiprocessing worker processes.
- This module provides a simple MP queue protocol so workers can offload NN eval
  (policy logits for legal moves + value) to a single GPU-hosted process.

Protocol:
- Workers send: (worker_id:int, request_id:str, fen:str)
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


def gpu_inference_server_main(
    model_state_dict: Dict[str, torch.Tensor],
    model_constructor: Callable[..., nn.Module],
    model_config: Dict[str, Any],
    device: str,
    request_queue,
    response_queues: List,
    batch_size: int = 32,
    post_batch_wait_ms: int = 10,
) -> None:
    """Entry point for the GPU inference server process."""
    torch.set_num_threads(1)

    # Build and load model
    model = model_constructor(**model_config)
    model.load_state_dict(model_state_dict, strict=False)

    server = GPUInferenceServer(model=model, device=device, batch_size=batch_size)

    pending: List[Tuple[int, str, str]] = []
    last_flush = time.time()
    last_log = time.time()
    processed = 0

    logger.info(
        f"GPU inference process started: device={device}, batch_size={batch_size}, post_wait_ms={post_batch_wait_ms}"
    )

    while True:
        # Block briefly waiting for requests
        try:
            item = request_queue.get(timeout=0.01)
        except Empty:
            item = "__EMPTY__"

        if item is None:
            break

        if item != "__EMPTY__":
            # (worker_id, request_id, fen)
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

        if processed >= 200 and (time.time() - last_log) > 2.0:
            last_log = time.time()
            try:
                logger.info(
                    f"GPU inference stats: processed_evals={server.processed_evals}, "
                    f"batches={server.total_batches}, avg_batch={server.avg_batch_size:.1f}"
                )
            except Exception:
                pass
            processed = 0

    # Drain any pending (best effort) with neutral outputs
    for worker_id, rid, _fen in pending:
        try:
            response_queues[int(worker_id)].put((rid, [0.0], 0.0))
        except Exception:
            pass

    logger.info("GPU inference process stopped")
