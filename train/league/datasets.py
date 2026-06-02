"""
Auxiliary Dataset Loader for League Training
=============================================

Wraps Lichess puzzles + Stockfish-labelled pro PGNs into lightweight in-memory
datasets sized for league training. Designed to be:

  - Cheap to construct (one-time cost, then `sample(batch_size)` per step).
  - Toggle-friendly (every source can be enabled/disabled independently).
  - Compatible with the existing ``board_to_tensor`` LRU cache and ``get_move_index``.

Public API:
  - ``AuxDataConfig``: dataclass-like settings for the three sources.
  - ``AuxDataLoader``: holds (optional) puzzle + progame datasets and provides
    ``sample_puzzle_batch(batch_size)`` and ``sample_progame_batch(batch_size)``.
"""

import os
import random
import logging
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch

from core.data import (
    PuzzleDataset,
    ProGameDataset,
    load_lichess_puzzles,
    expand_mate_sequences,
    discover_pgn_files,
)

logger = logging.getLogger(__name__)


class AuxDataConfig:
    """Configuration for the three auxiliary data sources."""

    def __init__(
        self,
        use_puzzle_injection: bool = True,
        use_pro_games: bool = True,
        use_stockfish_eval: bool = True,
        puzzles_csv: Optional[str] = None,
        progames_dir: Optional[str] = None,
        progame_sample_cap: int = 50_000,
        progame_pgn_file_limit: int = 0,
        cache_dir: str = "train/cache",
        stockfish_depth: int = 12,
        stockfish_threads: int = 2,
        stockfish_hash_mb: int = 256,
        stockfish_per_position_timeout_sec: float = 600.0,
        stockfish_num_workers: int = 0,
        puzzle_max_expand_depth: int = 4,
        model_type: str = "big",
    ):
        self.use_puzzle_injection = use_puzzle_injection
        self.use_pro_games = use_pro_games
        self.use_stockfish_eval = use_stockfish_eval
        self.puzzles_csv = puzzles_csv
        self.progames_dir = progames_dir
        self.progame_sample_cap = progame_sample_cap
        self.progame_pgn_file_limit = progame_pgn_file_limit
        self.cache_dir = cache_dir
        self.stockfish_depth = stockfish_depth
        self.stockfish_threads = stockfish_threads
        self.stockfish_hash_mb = stockfish_hash_mb
        self.stockfish_per_position_timeout_sec = stockfish_per_position_timeout_sec
        self.stockfish_num_workers = stockfish_num_workers
        self.puzzle_max_expand_depth = puzzle_max_expand_depth
        self.model_type = model_type

    def __repr__(self) -> str:
        return (
            f"AuxDataConfig(puzzles={self.use_puzzle_injection}, "
            f"progames={self.use_pro_games}, stockfish={self.use_stockfish_eval})"
        )


def _default_puzzles_csv() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / "chess_pgns" / "puzzles" / "lichess_db_puzzle.csv")


def _default_progames_dir() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / "chess_pgns" / "pros")


class AuxDataLoader:
    """Holds the puzzle + progame datasets and provides per-step batch sampling.

    Construction is lazy: datasets are only built if the corresponding toggle
    is enabled AND the source data exists. This keeps the no-toggle-off path
    cheap and side-effect free.
    """

    def __init__(self, config: AuxDataConfig):
        self.config = config
        self.puzzle_dataset: Optional[PuzzleDataset] = None
        self.progame_dataset: Optional[ProGameDataset] = None
        self._rng = random.Random(0xC0FFEE)

    def initialize(self) -> None:
        """Build the enabled datasets. Safe to call multiple times."""
        if self.config.use_puzzle_injection and self.puzzle_dataset is None:
            self._build_puzzle_dataset()
        if self.config.use_pro_games and self.progame_dataset is None:
            self._build_progame_dataset()

    def _build_puzzle_dataset(self) -> None:
        csv_path = self.config.puzzles_csv or _default_puzzles_csv()
        if not os.path.exists(csv_path):
            logger.warning(f"Puzzle CSV not found: {csv_path}; puzzle injection disabled")
            return
        logger.info(f"Loading Lichess puzzles from {csv_path}")
        puzzles = load_lichess_puzzles(csv_path, cache_dir=os.path.join(self.config.cache_dir, "puzzles"))
        puzzles = expand_mate_sequences(
            puzzles,
            max_expand_depth=self.config.puzzle_max_expand_depth,
            cache_dir=os.path.join(self.config.cache_dir, "puzzles"),
        )
        # The PuzzleDataset constructor can be slow for very large lists; cap input
        # to keep RAM bounded (the disk cache stays as the source of truth).
        if len(puzzles) > self.config.progame_sample_cap:
            logger.info(f"Capping puzzle input at {self.config.progame_sample_cap} (from {len(puzzles)})")
            self._rng.shuffle(puzzles)
            puzzles = puzzles[: self.config.progame_sample_cap]
        self.puzzle_dataset = PuzzleDataset(
            puzzles,
            model_type=self.config.model_type,
            cache_dir=os.path.join(self.config.cache_dir, "puzzle_tensors"),
        )

    def _build_progame_dataset(self) -> None:
        # Lazy import to avoid pulling Stockfish at module-import time
        from league.aux_phases import ProgameLabeller

        pro_dir = self.config.progames_dir or _default_progames_dir()
        if not os.path.isdir(pro_dir):
            logger.warning(f"Pro PGN dir not found: {pro_dir}; progames disabled")
            return
        pgn_files = discover_pgn_files(root_dir=str(Path(pro_dir).parent), subdirs=(Path(pro_dir).name,))
        if not pgn_files:
            logger.warning(f"No PGN files discovered under {pro_dir}")
            return

        # Optional cap on number of PGN files (0 = no cap, use all).
        if self.config.progame_pgn_file_limit > 0 and len(pgn_files) > self.config.progame_pgn_file_limit:
            logger.info(
                f"Capping pro-PGN file scan to {self.config.progame_pgn_file_limit} "
                f"of {len(pgn_files)} available files"
            )
            pgn_files = pgn_files[: self.config.progame_pgn_file_limit]

        if self.config.use_stockfish_eval:
            labeller = ProgameLabeller(
                depth=self.config.stockfish_depth,
                threads=self.config.stockfish_threads,
                hash_mb=self.config.stockfish_hash_mb,
                cache_dir=os.path.join(self.config.cache_dir, "labelled_pgns"),
                per_position_time_limit_sec=self.config.stockfish_per_position_timeout_sec,
                num_workers=self.config.stockfish_num_workers,
            )
            try:
                samples = labeller.label_pgns(pgn_files)
            finally:
                labeller.close()
        else:
            # Stockfish disabled: fall back to game outcome as the value target
            # (weak signal, but better than nothing when toggled off).
            samples = self._outcome_only_labels(pgn_files)

        if not samples:
            logger.warning("Pro PGN labelling produced 0 samples; progames disabled")
            return
        if len(samples) > self.config.progame_sample_cap:
            self._rng.shuffle(samples)
            samples = samples[: self.config.progame_sample_cap]
        self.progame_dataset = ProGameDataset(samples, model_type=self.config.model_type)
        logger.info(f"ProGameDataset ready: {len(self.progame_dataset)} samples")

    @staticmethod
    def _outcome_only_labels(pgn_files):
        """Label positions using the game result (1/0/-1 from white's POV).

        Used only when ``use_stockfish_eval`` is False. Each position gets
        the game's terminal outcome as its value target.
        """
        import chess.pgn
        samples = []
        for path in pgn_files:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    while True:
                        game = chess.pgn.read_game(fh)
                        if game is None:
                            break
                        result_str = game.headers.get("Result", "*")
                        if result_str not in ("1-0", "0-1", "1/2-1/2"):
                            continue
                        result = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}[result_str]
                        board = game.board()
                        for move in game.mainline_moves():
                            fen = board.fen()
                            samples.append((fen, move.uci(), result))
                            board.push(move)
            except Exception:
                continue
        return samples

    def is_ready(self) -> bool:
        """True if at least one dataset is available and has samples."""
        return self._puzzle_ready() or self._progame_ready()

    def _puzzle_ready(self) -> bool:
        return self.puzzle_dataset is not None and len(self.puzzle_dataset) > 0

    def _progame_ready(self) -> bool:
        return self.progame_dataset is not None and len(self.progame_dataset) > 0

    def sample_puzzle_batch(self, batch_size: int):
        """Return a (positions, policies, values) tuple as torch tensors on CPU.

        Returns None if puzzle injection is disabled or the dataset is empty.
        """
        if not self._puzzle_ready():
            return None
        n = len(self.puzzle_dataset)
        idxs = [self._rng.randrange(n) for _ in range(batch_size)]
        positions, policies, values = [], [], []
        for i in idxs:
            item = self.puzzle_dataset[i]
            pos, pol, val = item[0], item[1], item[2]
            positions.append(pos)
            policies.append(pol)
            values.append(val)
        pos_t = torch.stack(positions)
        pol_t = torch.stack(policies)
        val_t = torch.stack(values)
        return pos_t, pol_t, val_t

    def sample_progame_batch(self, batch_size: int):
        """Return a (positions, policies, values) tuple as torch tensors on CPU.

        Returns None if progame injection is disabled or the dataset is empty.
        """
        if not self._progame_ready():
            return None
        n = len(self.progame_dataset)
        idxs = [self._rng.randrange(n) for _ in range(batch_size)]
        positions, policies, values = [], [], []
        for i in idxs:
            pos, pol, val = self.progame_dataset[i]
            positions.append(pos)
            policies.append(pol)
            values.append(val)
        pos_t = torch.stack(positions)
        pol_t = torch.stack(policies)
        val_t = torch.stack(values)
        return pos_t, pol_t, val_t
