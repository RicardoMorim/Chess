"""
Auxiliary Training Phases: Stockfish labelling and benchmarking
===============================================================

This module provides:
  - `ProgameLabeller`: Uses Stockfish to assign value targets to positions from
    pro PGN files (centipawn eval -> [-1, 1] scalar).
  - `StockfishBenchmark`: Plays model vs Stockfish games for periodic ELO estimates.

Both helpers use a configurable Stockfish binary path; defaults to
``<repo>/stockfish/stockfish-windows-x86-64-avx2.exe`` on Windows.

Design notes:
  - All Stockfish subprocess lifecycles are explicit (open/close).
  - Labelling is cached on disk (pro-PGN FENs are stable; re-labelling is wasted work).
  - Benchmark uses the model's own policy + MCTS for move selection, so ELO is
    representative of actual play (not just raw NN eval).
"""

import os
import math
import time
import chess
import chess.engine
import chess.pgn
import logging
import multiprocessing as mp
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Tuple

logger = logging.getLogger(__name__)


def default_stockfish_path() -> str:
    """Locate Stockfish binary, preferring the repo's vendored copy on Windows."""
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "stockfish" / "stockfish-windows-x86-64-avx2.exe",
        repo_root / "stockfish" / "stockfish",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "stockfish"  # fall back to PATH lookup


def centipawns_to_value(cp: int, scale: float = 400.0) -> float:
    """Convert a centipawn eval to a [-1, 1] value target.

    Uses a tanh-like squash: small evals stay close to 0, large evals saturate.
    scale=400 means 4 pawns of advantage already gives ~0.76 (close to 1).
    """
    if cp is None:
        return 0.0
    # chess engine reports mate in N as +/- (mate_score - ply). Skip if mate.
    if abs(cp) > 30000:
        return 1.0 if cp > 0 else -1.0
    return math.tanh(cp / scale)


class ProgameLabeller:
    """Labels pro-PGN positions with Stockfish centipawn evaluations.

    Output is a list of (fen, move_uci, value_target) tuples (no category).
    The result is cached on disk as a pickle; re-labelling only happens when
    the cache file is missing or the source PGN hash changes.
    """

    def __init__(
        self,
        stockfish_path: Optional[str] = None,
        depth: int = 12,
        threads: int = 4,
        hash_mb: int = 512,
        cache_dir: str = "train/cache/labelled_pgns",
        positions_per_game_cap: int = 40,
        per_position_time_limit_sec: float = 600.0,
        num_workers: int = 0,
    ):
        self.stockfish_path = stockfish_path or default_stockfish_path()
        self.depth = depth
        self.threads = threads
        self.hash_mb = hash_mb
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.positions_per_game_cap = positions_per_game_cap
        self.per_position_time_limit_sec = per_position_time_limit_sec
        # Parallel labelling: spawn N Stockfish subprocesses, each with `threads`.
        # Default strategy: 2 workers x 4 threads = 8 threads total. This balances
        # inter-process overhead against per-process SMP scaling. Going beyond 2
        # workers tends to HURT throughput on most CPUs (4 x 2 threads was ~50%
        # slower than 2 x 4 threads in our benchmarks) because Stockfish's
        # in-process SMP scaling beats running many small instances.
        # 0 = auto-pick (2 workers, 4 threads each).
        if num_workers <= 0:
            num_workers = 2
        self.num_workers = num_workers
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def _get_engine(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            logger.info(f"Starting Stockfish: {self.stockfish_path}")
            self._engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            self._engine.configure({"Threads": self.threads, "Hash": self.hash_mb})
        return self._engine

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    def _cache_key(self, pgn_files: List[str]) -> str:
        import hashlib
        h = hashlib.md5()
        for f in sorted(pgn_files):
            try:
                st = os.stat(f)
                h.update(f.encode())
                h.update(str(st.st_size).encode())
                h.update(str(int(st.st_mtime)).encode())
            except OSError:
                continue
        h.update(f"d{self.depth}t{self.threads}c{self.positions_per_game_cap}".encode())
        return h.hexdigest()[:16]

    def label_pgns(self, pgn_files: List[str]) -> List[Tuple[str, str, float]]:
        """Label positions from the given PGN files.

        Uses ``num_workers`` parallel Stockfish subprocesses (one per worker),
        each with ``threads`` threads. PGNs are distributed round-robin across
        workers. Output is cached on disk by PGN set + depth, so a re-run is
        instant.

        Returns a list of (fen, move_uci, value_target) tuples.
        """
        import pickle

        if not pgn_files:
            return []

        cache_key = self._cache_key(pgn_files)
        cache_file = self.cache_dir / f"progame_labels_{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                logger.info(f"Loaded {len(cached)} pro-PGN labels from cache: {cache_file.name}")
                return cached
            except Exception as e:
                logger.warning(f"Pro-PGN cache load failed: {e}; rebuilding")

        total = len(pgn_files)
        num_workers = max(1, min(self.num_workers, total))
        logger.info(
            f"Labelling positions from {total} PGN file(s) at depth={self.depth} "
            f"(cap {self.positions_per_game_cap} positions/game, "
            f"{num_workers} parallel Stockfish workers x {self.threads} threads, "
            f"timeout {self.per_position_time_limit_sec:.0f}s/position)..."
        )

        # Quick upfront scan: count games by scanning for "[Event " headers
        # (much faster than chess.pgn.read_game for a 100MB+ collection).
        if total <= 200:
            logger.info("Scanning PGNs for game count (header scan)...")
            count_t0 = time.time()
            total_games = 0
            total_bytes = 0
            for p in pgn_files:
                try:
                    st = os.stat(p)
                    total_bytes += st.st_size
                    # Count "[Event " occurrences — every chess.pgn game has this header.
                    with open(p, "rb") as fh:
                        # 1MB chunks keep memory low
                        chunk_size = 1024 * 1024
                        while True:
                            chunk = fh.read(chunk_size)
                            if not chunk:
                                break
                            total_games += chunk.count(b"[Event ")
                except Exception:
                    continue
            elapsed_count = time.time() - count_t0
            est_positions = total_games * self.positions_per_game_cap
            logger.info(
                f"Scope: ~{total_games:,} games in {total} PGNs "
                f"({total_bytes / 1024 / 1024:.1f} MB on disk) — "
                f"up to ~{est_positions:,} positions to label "
                f"(header-scanned in {elapsed_count:.1f}s)"
            )

        # Distribute PGN files round-robin across workers.
        chunks: List[List[str]] = [[] for _ in range(num_workers)]
        for i, p in enumerate(pgn_files):
            chunks[i % num_workers].append(p)

        t0 = time.time()
        labelled: List[Tuple[str, str, float]] = []

        # Use spawn context (Windows-friendly, matches the rest of the codebase).
        ctx = mp.get_context("spawn")
        # Hash per worker is split so we don't blow past available RAM.
        hash_per_worker = max(64, self.hash_mb // num_workers)

        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers, mp_context=ctx
            ) as executor:
                futures = []
                for worker_id, chunk in enumerate(chunks):
                    futures.append(
                        executor.submit(
                            _label_chunk_worker,
                            chunk,
                            self.stockfish_path,
                            self.depth,
                            self.threads,
                            hash_per_worker,
                            self.positions_per_game_cap,
                            self.per_position_time_limit_sec,
                            worker_id,
                            num_workers,
                        )
                    )

                # Stream results as they complete; log per worker + heartbeat.
                last_log = t0
                total_processed = 0
                total_errors = 0
                done_workers = 0
                # We don't have a great way to peek at intermediate progress
                # from worker processes, so we log a main-process heartbeat
                # every 15s with the worker-done count + estimated wait time.
                pending = set(futures)
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending, timeout=15.0, return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    now = time.time()
                    elapsed = now - t0
                    if not done:
                        # Heartbeat: workers still running, show elapsed time
                        logger.info(
                            f"  ... labelling in progress: {done_workers}/{num_workers} workers done, "
                            f"~{total_processed:,} positions so far, "
                            f"elapsed {elapsed/60:.1f} min"
                        )
                        last_log = now
                        continue
                    for fut in done:
                        try:
                            chunk_samples, chunk_processed, chunk_errors = fut.result()
                        except Exception as e:
                            logger.error(f"Stockfish worker failed: {e}")
                            continue
                        labelled.extend(chunk_samples)
                        total_processed += chunk_processed
                        total_errors += chunk_errors
                        done_workers += 1
                        rate = total_processed / max(elapsed, 0.001)
                        logger.info(
                            f"  Stockfish labelling: worker {done_workers}/{num_workers} done "
                            f"({total_processed:,} positions, {rate:.1f} pos/s, errors={total_errors})"
                        )
                        last_log = now
        except KeyboardInterrupt:
            logger.warning(
                "KeyboardInterrupt during labelling; partial results discarded. "
                "Re-run will pick up from cache (none was written)."
            )
            return labelled

        elapsed = time.time() - t0
        rate = total_processed / max(elapsed, 0.001)
        logger.info(
            f"Labelled {total_processed} pro-PGN positions in {elapsed:.1f}s "
            f"({rate:.1f} pos/s, errors={total_errors})"
        )

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(labelled, f)
            logger.info(f"Cached pro-PGN labels to {cache_file.name}")
        except Exception as e:
            logger.warning(f"Failed to cache pro-PGN labels: {e}")

        return labelled


# Module-level worker function (must be picklable for spawn).
def _label_chunk_worker(
    pgn_paths: List[str],
    stockfish_path: str,
    depth: int,
    threads: int,
    hash_mb: int,
    positions_per_game_cap: int,
    per_position_time_limit_sec: float,
    worker_id: int,
    total_workers: int,
):
    """Worker process: initialize one Stockfish, process a chunk of PGNs.

    Returns (samples, processed_count, error_count).
    """
    import chess
    import chess.engine
    import chess.pgn
    import logging as _logging
    logger = _logging.getLogger(f"league.aux_phases.worker{worker_id}")
    if not logger.handlers:
        h = _logging.StreamHandler()
        h.setFormatter(_logging.Formatter(f"%(asctime)s [worker{worker_id}] %(message)s"))
        logger.addHandler(h)
        logger.setLevel(_logging.INFO)

    samples = []
    processed = 0
    errors = 0
    worker_start = time.time()
    worker_last_log = worker_start
    per_pos_limit = chess.engine.Limit(
        depth=depth, time=per_position_time_limit_sec,
    )

    engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Threads": threads, "Hash": hash_mb})
        logger.info(
            f"Worker {worker_id}/{total_workers}: {len(pgn_paths)} PGN file(s), "
            f"depth={depth}, threads={threads}, hash={hash_mb}MB"
        )
        for pgn_path in pgn_paths:
            file_name = os.path.basename(pgn_path)
            file_start = time.time()
            file_processed = 0
            file_games = 0
            try:
                with open(pgn_path, "r", encoding="utf-8", errors="ignore") as pgn_fh:
                    while True:
                        game = chess.pgn.read_game(pgn_fh)
                        if game is None:
                            break
                        file_games += 1
                        board = game.board()
                        moves = list(game.mainline_moves())
                        if not moves:
                            continue
                        step = max(1, len(moves) // positions_per_game_cap) if len(moves) > positions_per_game_cap else 1
                        for move_idx, move in enumerate(moves):
                            if move_idx % step != 0:
                                board.push(move)
                                continue
                            try:
                                fen = board.fen()
                                info = engine.analyse(board, per_pos_limit)
                                score = info.get("score")
                                cp = None
                                if score is not None:
                                    pov = score.relative
                                    cp = pov.score(mate_score=30000)
                                value = centipawns_to_value(cp or 0)
                                samples.append((fen, move.uci(), value))
                                processed += 1
                                file_processed += 1
                            except chess.engine.EngineTerminatedError:
                                logger.error("Stockfish died; aborting this worker")
                                return samples, processed, errors + 1
                            except Exception:
                                errors += 1
                            board.push(move)

                            # Intra-worker progress log (every ~5s or every 500 positions)
                            now = time.time()
                            if processed % 500 == 0 or (now - worker_last_log) >= 5.0:
                                rate = processed / max(now - worker_start, 0.001)
                                logger.info(
                                    f"  worker{worker_id}: {processed} pos "
                                    f"({rate:.1f} pos/s) [{file_name} +{file_processed}/{file_games}g]"
                                )
                                worker_last_log = now
                file_elapsed = time.time() - file_start
                logger.info(
                    f"  worker{worker_id}: {file_name} done — "
                    f"{file_games} games, +{file_processed} positions in {file_elapsed:.1f}s"
                )
            except Exception as e:
                logger.warning(f"Worker {worker_id}: failed to read {file_name}: {e}")
                continue
    except Exception as e:
        logger.error(f"Worker {worker_id} failed to start: {e}")
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass
    return samples, processed, errors


class StockfishBenchmark:
    """Plays model vs Stockfish games for periodic ELO estimation.

    The model selects moves via its current policy + light MCTS; Stockfish plays
    at the configured depth/limit. ELO is estimated from the score using the
    logistic formula (logistic_5 = 400 / ln(10) ≈ 173.7).
    """

    def __init__(
        self,
        stockfish_path: Optional[str] = None,
        depth: int = 15,
        threads: int = 1,
        hash_mb: int = 128,
        num_games: int = 20,
        time_limit_ms: int = 200,
    ):
        self.stockfish_path = stockfish_path or default_stockfish_path()
        self.depth = depth
        self.threads = threads
        self.hash_mb = hash_mb
        self.num_games = num_games
        self.time_limit_ms = time_limit_ms
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def _get_engine(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            self._engine.configure({"Threads": self.threads, "Hash": self.hash_mb})
        return self._engine

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    @staticmethod
    def score_to_elo_diff(score: float, logistic_5: float = 400.0 / math.log(10)) -> float:
        """Convert a score in [0, 1] to an approximate ELO difference vs the opponent.

        score=0.5 -> 0 ELO. score=0.75 -> +191 ELO. score=0.9 -> +382 ELO.
        """
        score = max(1e-6, min(1.0 - 1e-6, score))
        return logistic_5 * math.log10(score / (1.0 - score))

    def play_random_games(self, model_move_fn) -> Dict[str, Any]:
        """Play ``num_games`` model-vs-Stockfish games starting from the standard
        position (alternating colours each game).

        ``model_move_fn`` is a callable ``(board: chess.Board) -> chess.Move``.
        Returns a dict with win/draw/loss counts, score, and ELO estimate.
        """
        engine = self._get_engine()
        results = {"wins": 0, "draws": 0, "losses": 0, "games": 0}
        t0 = time.time()
        for i in range(self.num_games):
            board = chess.Board()
            model_is_white = (i % 2 == 0)
            try:
                while not board.is_game_over(claim_draw=True):
                    if (board.turn == chess.WHITE) == model_is_white:
                        move = model_move_fn(board)
                        if move is None or move not in board.legal_moves:
                            move = next(iter(board.legal_moves), None)
                            if move is None:
                                break
                        board.push(move)
                    else:
                        result = engine.play(
                            board,
                            chess.engine.Limit(time=self.time_limit_ms / 1000.0),
                        )
                        if result.move is None:
                            break
                        board.push(result.move)
                outcome = board.outcome(claim_draw=True)
                if outcome is None:
                    results["draws"] += 1
                elif outcome.winner is None:
                    results["draws"] += 1
                elif (outcome.winner == chess.WHITE) == model_is_white:
                    results["wins"] += 1
                else:
                    results["losses"] += 1
                results["games"] += 1
            except Exception as e:
                logger.warning(f"Benchmark game {i} failed: {e}")
                continue

        games = max(1, results["games"])
        score = (results["wins"] + 0.5 * results["draws"]) / games
        elo_diff = self.score_to_elo_diff(score)
        elapsed = time.time() - t0
        logger.info(
            f"Stockfish benchmark: {results['wins']}W-{results['draws']}D-{results['losses']}L "
            f"over {games} games in {elapsed:.1f}s (score={score:.2%}, elo_diff={elo_diff:+.0f})"
        )
        return {
            **results,
            "score": score,
            "elo_diff_estimate": elo_diff,
            "elapsed_seconds": elapsed,
        }
