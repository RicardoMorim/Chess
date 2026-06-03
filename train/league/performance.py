"""
Performance presets for LeagueTrainer (Fase 1).

Defines 3 named presets (``eco``, ``balanced``, ``boost``) that map to a
coherent set of knob values. Switching presets is equivalent to issuing a
batch ``set_knob`` call.

Why presets and not individual knobs?
  - Atomicity: you can't half-apply a mode change mid-round. A preset is
    a single logical change.
  - Safety: a preset is a known-good combination that has been smoke-tested.
    Allowing free-form knob edits invites footguns.
  - Discoverability: 3 buttons in the UI are easier than 20 sliders.

Auto-mode (optional): a watchdog thread that promotes/demotes presets based
on observed CPU usage. Default OFF; opt-in via ``set_auto_mode(True)``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .league_trainer import LeagueTrainer

logger = logging.getLogger(__name__)


# Knobs that each preset controls. These MUST be in LeagueTrainer._HOT_KNOBS.
PRESET_KNOBS: tuple = (
    "BATCH_SIZE",
    "TRAINING_STEPS_PER_ROUND",
    "GAMES_PER_WORKER_PER_ROUND",
    "MCTS_VISITS_SELFPLAY",
    "NUM_SELF_PLAY_WORKERS",
    "REPLAY_BUFFER_MAX_SIZE",
    "GPU_INFER_BATCH_SIZE",
    "STOCKFISH_BENCH_EVERY_N_ROUNDS",
    "PUZZLE_BATCHES_PER_GAME_BATCH",
    "SELF_PLAY_VARIANT_PARALLELISM",
)


@dataclass(frozen=True)
class PerformancePreset:
    """A coherent set of knob values."""

    name: str
    description: str
    batch_size: int
    training_steps_per_round: int
    games_per_worker_per_round: int
    mcts_visits_selfplay: int
    num_self_play_workers: int
    replay_buffer_max_size: int
    gpu_infer_batch_size: int
    stockfish_bench_every_n_rounds: int
    puzzle_batches_per_game_batch: int
    self_play_variant_parallelism: int

    def as_knob_dict(self) -> Dict[str, object]:
        """Map preset fields to the corresponding LeagueTrainer attribute names."""
        return {
            "BATCH_SIZE": self.batch_size,
            "TRAINING_STEPS_PER_ROUND": self.training_steps_per_round,
            "GAMES_PER_WORKER_PER_ROUND": self.games_per_worker_per_round,
            "MCTS_VISITS_SELFPLAY": self.mcts_visits_selfplay,
            "NUM_SELF_PLAY_WORKERS": self.num_self_play_workers,
            "REPLAY_BUFFER_MAX_SIZE": self.replay_buffer_max_size,
            "GPU_INFER_BATCH_SIZE": self.gpu_infer_batch_size,
            "STOCKFISH_BENCH_EVERY_N_ROUNDS": self.stockfish_bench_every_n_rounds,
            "PUZZLE_BATCHES_PER_GAME_BATCH": self.puzzle_batches_per_game_batch,
            "SELF_PLAY_VARIANT_PARALLELISM": self.self_play_variant_parallelism,
        }


PRESETS: Dict[str, PerformancePreset] = {
    "eco": PerformancePreset(
        name="eco",
        description=(
            "Light training for when you're using the PC. "
            "Small batch, few workers, short MCTS, frequent Stockfish benchmarks."
        ),
        batch_size=128,
        training_steps_per_round=25,
        games_per_worker_per_round=2,
        mcts_visits_selfplay=80,
        num_self_play_workers=3,
        replay_buffer_max_size=50_000,
        gpu_infer_batch_size=32,
        stockfish_bench_every_n_rounds=100,
        puzzle_batches_per_game_batch=0,
        self_play_variant_parallelism=2,
    ),
    "balanced": PerformancePreset(
        name="balanced",
        description="Default. Matches the original class-level constants.",
        batch_size=256,
        training_steps_per_round=50,
        games_per_worker_per_round=5,
        mcts_visits_selfplay=200,
        num_self_play_workers=6,
        replay_buffer_max_size=100_000,
        gpu_infer_batch_size=64,
        stockfish_bench_every_n_rounds=25,
        puzzle_batches_per_game_batch=1,
        self_play_variant_parallelism=3,
    ),
    "boost": PerformancePreset(
        name="boost",
        description=(
            "Overnight mode. 4x batch, 2x workers, 2x MCTS visits, "
            "2x buffer. Use when you're away from the PC."
        ),
        batch_size=1024,
        training_steps_per_round=100,
        games_per_worker_per_round=8,
        mcts_visits_selfplay=400,
        num_self_play_workers=12,
        replay_buffer_max_size=300_000,
        gpu_infer_batch_size=128,
        stockfish_bench_every_n_rounds=10,
        puzzle_batches_per_game_batch=2,
        self_play_variant_parallelism=3,
    ),
}


def list_preset_names() -> list:
    return list(PRESETS.keys())


def get_preset(name: str) -> PerformancePreset:
    if name not in PRESETS:
        raise KeyError(
            f"Unknown preset '{name}'. Choose from {list_preset_names()}."
        )
    return PRESETS[name]


# =============================================================================
# Auto-mode watchdog
# =============================================================================


@dataclass
class AutoModeConfig:
    """Rules for the auto-mode watchdog."""

    enabled: bool = False
    promote_cpu_pct: float = 15.0   # below this => upgrade preset
    demote_cpu_pct: float = 50.0    # above this => downgrade preset
    poll_interval_sec: float = 60.0
    min_seconds_between_changes: float = 300.0  # hysteresis (5 min)
    sample_interval_sec: float = 5.0  # how long psutil samples for


_PROMOTE_ORDER = ("eco", "balanced", "boost")
_DEMOTE_ORDER = tuple(reversed(_PROMOTE_ORDER))


def _next_in_order(current: str, order: tuple) -> Optional[str]:
    """Return the next preset in the order, or None if at the end."""
    try:
        idx = order.index(current)
    except ValueError:
        return None
    if idx + 1 >= len(order):
        return None
    return order[idx + 1]


class AutoModeController:
    """Background thread that promotes/demotes presets based on CPU usage.

    Designed to be lightweight (60s polling, 5s sample). Hysteresis prevents
    flapping (5 min minimum between changes). Re-entrant: can be enabled
    and disabled at any time; thread exits cleanly.
    """

    def __init__(self, trainer: "LeagueTrainer", config: Optional[AutoModeConfig] = None):
        self.trainer = trainer
        self.config = config or AutoModeConfig()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_change_time: float = 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="AutoMode", daemon=True)
        self._thread.start()
        logger.info(f"AutoMode started (poll={self.config.poll_interval_sec}s)")

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            self._thread = None
        logger.info("AutoMode stopped")

    def set_enabled(self, enabled: bool) -> None:
        was_enabled = self.config.enabled
        self.config.enabled = enabled
        if enabled and not was_enabled:
            self.start()
        elif not enabled and was_enabled:
            self.stop()

    def _loop(self) -> None:
        try:
            import psutil
        except ImportError:
            logger.warning("AutoMode: psutil not available, disabling")
            return

        while not self._stop.is_set():
            try:
                if self.config.enabled:
                    self._tick(psutil)
            except Exception as e:
                logger.debug(f"AutoMode tick failed (non-fatal): {e}")
            # Wait, but be interruptible
            self._stop.wait(self.config.poll_interval_sec)

    def _tick(self, psutil_mod) -> None:
        cpu_pct = psutil_mod.cpu_percent(interval=self.config.sample_interval_sec)
        now = time.time()
        if now - self._last_change_time < self.config.min_seconds_between_changes:
            return  # hysteresis

        current = getattr(self.trainer, "performance_mode", "balanced")
        if cpu_pct < self.config.promote_cpu_pct:
            nxt = _next_in_order(current, _PROMOTE_ORDER)
            if nxt is not None:
                self._change(current, nxt, cpu_pct)
        elif cpu_pct > self.config.demote_cpu_pct:
            nxt = _next_in_order(current, _DEMOTE_ORDER)
            if nxt is not None:
                self._change(current, nxt, cpu_pct)

    def _change(self, old: str, new: str, cpu_pct: float) -> None:
        logger.info(f"AutoMode: CPU={cpu_pct:.1f}%%, switching {old} -> {new}")
        self.trainer.set_mode(new)
        self._last_change_time = time.time()
