"""
League Training Monitoring System
==================================

Tracks metrics across all training components:
- Self-play statistics (games/sec, game length, outcomes)
- Replay buffer health (fill ratio, value distribution)
- Training dynamics (loss, learning rate, gradient norms)
- Model performance (vs checkpoints, strength trends)

Design: Minimal overhead, thread-safe, JSON-serializable for logging.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime
import threading
from typing import Tuple

try:
    import wandb
except Exception:
    wandb = None


class MetricsCollector:
    """
    Collects and aggregates metrics from all league training components.
    
    Thread-safe for concurrent updates from multiple workers.
    Provides periodic snapshots for logging and monitoring dashboards.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize metrics collector.
        
        Args:
            log_dir: Directory to store metric logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.lock = threading.Lock()
        self.start_time = time.time()
        
        # Metric storage
        self.metrics = defaultdict(list)  # metric_name -> list of (timestamp, value)
        self.counters = defaultdict(int)  # counter_name -> cumulative count
        self.gauges = {}  # gauge_name -> current value
        
        # Per-variant tracking
        self.variant_metrics = defaultdict(lambda: defaultdict(list))

        # Prevent unbounded memory growth for long-running sessions
        self.max_points_per_series = 50_000
        
        # Buffer for high-frequency events
        self.event_buffer = defaultdict(list)

        # Optional W&B run (lazy-initialized)
        self.wandb_run = None
        
        # Logger setup
        self.logger = logging.getLogger("LeagueMetrics")
        handler = logging.FileHandler(self.log_dir / "metrics.log")
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def enable_wandb(
        self,
        project: str = "chess-league",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        mode: str = "offline",
    ) -> bool:
        """Initialize a W&B run if the package is available.

        Returns True if a run was created, False otherwise.
        """
        if wandb is None:
            self.logger.info("W&B not available; skipping experiment tracking")
            return False
        if self.wandb_run is not None:
            return True

        try:
            self.wandb_run = wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                tags=tags,
                mode=mode,
                reinit=True,
            )
            return True
        except Exception as e:
            self.logger.warning(f"Could not initialize W&B: {e}")
            self.wandb_run = None
            return False

    def _flatten_summary(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        flat = {}

        def _walk(prefix: str, value: Any):
            if isinstance(value, dict):
                for key, child in value.items():
                    _walk(f"{prefix}{key}." if prefix else f"{key}.", child)
            else:
                flat[prefix[:-1] if prefix.endswith(".") else prefix] = value

        _walk("", summary)
        return flat

    def log_wandb_summary(self, summary: Optional[Dict[str, Any]] = None, step: Optional[int] = None) -> None:
        """Log a flattened metrics summary to W&B if enabled."""
        if self.wandb_run is None:
            return
        try:
            if summary is None:
                summary = self.get_summary()
            payload = self._flatten_summary(summary)
            self.wandb_run.log(payload, step=step)
        except Exception as e:
            self.logger.warning(f"W&B logging failed: {e}")

    def finish_wandb(self) -> None:
        """Finish the W&B run if one is active."""
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.finish()
        except Exception:
            pass
        finally:
            self.wandb_run = None
    
    def record_metric(
        self,
        name: str,
        value: float,
        variant: Optional[str] = None,
    ) -> None:
        """
        Record a metric value.
        
        Args:
            name: Metric name (e.g., "game_length", "loss")
            value: Metric value
            variant: Optional variant name for per-variant tracking
        """
        with self.lock:
            timestamp = time.time()
            series = self.metrics[name]
            series.append((timestamp, value))
            if len(series) > self.max_points_per_series:
                del series[:-self.max_points_per_series]
            
            if variant:
                v_series = self.variant_metrics[variant][name]
                v_series.append((timestamp, value))
                if len(v_series) > self.max_points_per_series:
                    del v_series[:-self.max_points_per_series]
    
    def record_counter(
        self,
        name: str,
        increment: int = 1,
        variant: Optional[str] = None,
    ) -> None:
        """
        Increment a counter.
        
        Args:
            name: Counter name
            increment: Amount to increment
            variant: Optional variant name
        """
        with self.lock:
            # Don't double-prefix if name already has variant prefix
            if variant and not name.startswith(f"{variant}_"):
                counter_key = f"{variant}_{name}"
            else:
                counter_key = name
            self.counters[counter_key] += increment
    
    def set_gauge(
        self,
        name: str,
        value: float,
        variant: Optional[str] = None,
    ) -> None:
        """
        Set a gauge value (current state).
        
        Args:
            name: Gauge name
            value: Gauge value
            variant: Optional variant name
        """
        with self.lock:
            # Don't double-prefix if name already has variant prefix
            if variant and not name.startswith(f"{variant}_"):
                gauge_key = f"{variant}_{name}"
            else:
                gauge_key = name
            self.gauges[gauge_key] = value
    
    def record_self_play_game(
        self,
        variant: str,
        game_length: int,
        outcome: str,  # "1-0", "0-1", "1/2-1/2"
    ) -> None:
        """
        Record statistics for a completed self-play game.
        
        Args:
            variant: Model variant
            game_length: Number of half-moves
            outcome: Game outcome
        """
        self.record_metric(f"{variant}/game_length", game_length, variant)
        
        if outcome == "1-0":
            self.record_metric(f"{variant}/white_wins", 1, variant)
        elif outcome == "0-1":
            self.record_metric(f"{variant}/black_wins", 1, variant)
        else:
            self.record_metric(f"{variant}/draws", 1, variant)
        
        self.record_counter(f"{variant}_games", 1, variant)
    
    def record_training_step(
        self,
        variant: str,
        loss: float,
        policy_loss: float,
        value_loss: float,
        learning_rate: float,
    ) -> None:
        """
        Record training step metrics.
        
        Args:
            variant: Model variant
            loss: Total loss
            policy_loss: Policy head loss
            value_loss: Value head loss
            learning_rate: Current learning rate
        """
        self.record_metric(f"{variant}/loss", loss, variant)
        self.record_metric(f"{variant}/policy_loss", policy_loss, variant)
        self.record_metric(f"{variant}/value_loss", value_loss, variant)
        self.set_gauge(f"{variant}_lr", learning_rate, variant)
        self.record_counter(f"{variant}_train_steps", 1, variant)
    
    def record_evaluation(
        self,
        variant: str,
        opponent: str,
        result: str,  # "win", "loss", "draw"
        elo_change: float = 0.0,
    ) -> None:
        """
        Record evaluation game result.
        
        Args:
            variant: Model variant being evaluated
            opponent: Opponent variant or description
            result: Game result
            elo_change: Estimated ELO change
        """
        self.record_metric(f"{variant}/eval_vs_{opponent}", 1 if result == "win" else 0, variant)
        self.record_metric(f"{variant}/elo_vs_{opponent}", elo_change, variant)
        self.record_counter(f"{variant}_evals", 1, variant)
    
    def record_buffer_stats(
        self,
        variant: str,
        buffer_size: int,
        capacity: int,
        value_mean: float,
        value_std: float,
    ) -> None:
        """
        Record replay buffer statistics.
        
        Args:
            variant: Model variant
            buffer_size: Current number of positions
            capacity: Buffer capacity
            value_mean: Mean value in buffer
            value_std: Standard deviation of values
        """
        fill_ratio = buffer_size / capacity if capacity > 0 else 0
        self.set_gauge(f"{variant}_buffer_size", buffer_size, variant)
        self.set_gauge(f"{variant}_buffer_fill", fill_ratio, variant)
        self.set_gauge(f"{variant}_buffer_value_mean", value_mean, variant)
        self.set_gauge(f"{variant}_buffer_value_std", value_std, variant)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get current metrics summary.
        
        Returns:
            Dict with keys: timestamp, elapsed_time, counters, gauges, variants_summary
        """
        with self.lock:
            elapsed = time.time() - self.start_time
            
            summary = {
                "timestamp": datetime.now().isoformat(),
                "elapsed_seconds": elapsed,
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
            }
            
            # Summarize by variant
            variants_summary = {}
            for variant in self.variant_metrics.keys():
                v_metrics = self.variant_metrics[variant]
                
                # Get most recent values for time series
                variant_summary = {
                    "games": self.counters.get(f"{variant}_games", 0),
                    "train_steps": self.counters.get(f"{variant}_train_steps", 0),
                    "buffer_size": self.gauges.get(f"{variant}_buffer_size", 0),
                    "buffer_fill_ratio": self.gauges.get(f"{variant}_buffer_fill", 0),
                }
                
                # Add recent losses if available
                if f"{variant}/loss" in v_metrics and v_metrics[f"{variant}/loss"]:
                    variant_summary["recent_loss"] = v_metrics[f"{variant}/loss"][-1][1]
                
                if f"{variant}/game_length" in v_metrics and v_metrics[f"{variant}/game_length"]:
                    lengths = [x[1] for x in v_metrics[f"{variant}/game_length"][-100:]]
                    variant_summary["avg_game_length"] = sum(lengths) / len(lengths) if lengths else 0
                
                variants_summary[variant] = variant_summary
            
            summary["variants"] = variants_summary
            
            return summary
    
    def save_checkpoint(self, name: str = None) -> str:
        """
        Save metrics snapshot to file.
        
        Args:
            name: Optional checkpoint name suffix
        
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if name:
            filename = f"metrics_{name}_{timestamp}.json"
        else:
            filename = f"metrics_{timestamp}.json"
        
        filepath = self.log_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(self.get_summary(), f, indent=2)
        
        self.logger.info(f"Saved metrics checkpoint: {filepath}")
        return str(filepath)
    
    def log_summary(self, prefix: str = "") -> None:
        """
        Log current metrics summary to logger.
        
        Args:
            prefix: Optional prefix for log messages
        """
        summary = self.get_summary()
        
        msg = f"{prefix} Metrics Summary (elapsed: {summary['elapsed_seconds']:.1f}s)\n"
        
        for variant, v_summary in summary["variants"].items():
            msg += f"\n{variant}:\n"
            for key, value in v_summary.items():
                if isinstance(value, float):
                    msg += f"  {key}: {value:.4f}\n"
                else:
                    msg += f"  {key}: {value}\n"
        
        self.logger.info(msg)
    
    def get_timeseries(self, metric_name: str) -> List[Dict[str, Any]]:
        """
        Get time series data for a metric.
        
        Args:
            metric_name: Name of metric to retrieve
        
        Returns:
            List of {"timestamp": ts, "value": val} dicts
        """
        with self.lock:
            if metric_name not in self.metrics:
                return []
            
            return [
                {"timestamp": ts, "value": val}
                for ts, val in self.metrics[metric_name]
            ]

    def get_variant_throughput(self, variant: str) -> Optional[float]:
        """
        Get the most recent throughput (games/min) for a variant.

        Args:
            variant: Model variant name

        Returns:
            Most recent games/min value, or None if not available
        """
        with self.lock:
            metric_name = f"{variant}_throughput"
            if metric_name not in self.metrics:
                return None

            points = self.metrics[metric_name]
            if not points:
                return None

            # Return the most recent value
            return points[-1][1]

    def get_recent_loss(self, variant: str) -> Optional[float]:
        """Get the most recent loss value for a variant, if any."""
        with self.lock:
            for name in (f"{variant}_loss", f"{variant}_train_loss", f"{variant}_policy_loss"):
                points = self.metrics.get(name)
                if points:
                    return points[-1][1]
            return None


class MetricsServer:
    """
    Simple HTTP server to expose metrics for monitoring dashboards.
    (Optional advanced monitoring - can skip if not needed)
    """
    pass
