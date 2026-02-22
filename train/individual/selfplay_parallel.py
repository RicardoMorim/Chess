"""Parallel self-play generation for individual curriculum training.

Goal: Use CPU cores efficiently without oversubscribing threads.

Design:
- Multiprocessing (spawn-safe on Windows)
- Each worker loads the model on CPU and generates games via core.mcts.generate_mcts_game
- Workers send PGN strings back to the parent process (safe to pickle)
"""

from __future__ import annotations

import io
import os
import sys
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import chess.pgn

# Ensure train/ is on sys.path so `core` imports work in spawned workers.
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.models import create_model

logger = logging.getLogger(__name__)


def _worker_generate_pgn(
    variant: str,
    model_state_dict: dict,
    num_games: int,
    num_simulations: int,
    temperature: float,
    mcts_parallel_workers: int,
    result_queue: mp.Queue,
    worker_id: int,
) -> None:
    """Worker entry: generate `num_games` PGNs and push to result_queue."""
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        torch.set_num_threads(1)

        from core.mcts import generate_mcts_game

        model = create_model(variant=variant)
        model.load_state_dict(model_state_dict)
        model.to("cpu")
        model.eval()

        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)

        for game_idx in range(num_games):
            game = generate_mcts_game(
                model=model,
                device="cpu",
                temperature=temperature,
                num_simulations=num_simulations,
                parallel_workers=mcts_parallel_workers,
            )
            if game is None:
                continue
            pgn_str = game.accept(exporter)
            result_queue.put((worker_id, game_idx, pgn_str))

        result_queue.put((worker_id, None, None))  # sentinel

    except Exception as e:
        result_queue.put((worker_id, "error", str(e)))


def generate_games_parallel(
    *,
    variant: str,
    model: torch.nn.Module,
    num_games: int,
    num_simulations: int,
    temperature: float = 1.0,
    selfplay_workers: int = 8,
    mcts_parallel_workers: int = 1,
) -> List[chess.pgn.Game]:
    """Generate self-play games in parallel CPU worker processes.

    Returns a list of chess.pgn.Game objects.
    """
    model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return generate_games_parallel_from_state(
        variant=variant,
        model_state_dict=model_state,
        num_games=num_games,
        num_simulations=num_simulations,
        temperature=temperature,
        selfplay_workers=selfplay_workers,
        mcts_parallel_workers=mcts_parallel_workers,
    )


def generate_games_parallel_from_state(
    *,
    variant: str,
    model_state_dict: dict,
    num_games: int,
    num_simulations: int,
    temperature: float = 1.0,
    selfplay_workers: int = 8,
    mcts_parallel_workers: int = 1,
) -> List[chess.pgn.Game]:
    """Generate self-play games in parallel given a CPU state_dict snapshot.

    This is safe to call while the main process continues training the live model,
    because workers never touch the live model object.
    """
    if num_games <= 0:
        return []

    selfplay_workers = max(1, int(selfplay_workers))
    selfplay_workers = min(selfplay_workers, num_games)

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    processes = []

    base = num_games // selfplay_workers
    remainder = num_games % selfplay_workers

    for worker_id in range(selfplay_workers):
        worker_games = base + (1 if worker_id < remainder else 0)
        p = ctx.Process(
            target=_worker_generate_pgn,
            args=(
                variant,
                model_state_dict,
                worker_games,
                num_simulations,
                temperature,
                int(mcts_parallel_workers),
                queue,
                worker_id,
            ),
            name=f"individual_selfplay_{variant}_{worker_id}",
        )
        p.start()
        processes.append(p)

    finished_workers = 0
    pgn_strings: List[str] = []

    while finished_workers < selfplay_workers:
        worker_id, game_idx, payload = queue.get()
        if game_idx == "error":
            logger.error(f"Self-play worker {worker_id} error: {payload}")
            finished_workers += 1
            continue
        if game_idx is None and payload is None:
            finished_workers += 1
            continue
        if payload:
            pgn_strings.append(payload)

    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    games: List[chess.pgn.Game] = []
    for pgn_str in pgn_strings:
        try:
            game = chess.pgn.read_game(io.StringIO(pgn_str))
            if game is not None:
                games.append(game)
        except Exception:
            continue

    return games
