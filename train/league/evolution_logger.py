"""Separate evolution report writer for league training.

This keeps a human-readable progress file in ``logs/`` that summarizes the
league evolution round by round, independent from the verbose technical logs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class EvolutionLogger:
    """Append round-by-round league progress to Markdown and JSONL files."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.markdown_path = self.log_dir / "league_evolution.md"
        self.jsonl_path = self.log_dir / "league_evolution.jsonl"

        self._ensure_header()

        # Track previous batch sizes for trend arrows (variant -> last avg_batch)
        self._prev_batches: Dict[str, float] = {}

    def _ensure_header(self) -> None:
        if self.markdown_path.exists():
            return

        header = (
            "# League evolution\n\n"
            "Snapshot of the three-model league progression.\n\n"
            "| Round | Time | Baseline score | Attack score | EST score | Baseline ELO | Attack ELO | EST ELO |\n"
            "|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        self.markdown_path.write_text(header, encoding="utf-8")

    def _fmt_float(self, value: Any, digits: int = 3) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except Exception:
            return "-"

    def _variant_stat(self, summary: Dict[str, Any], variant: str, key: str, default: Any = "-") -> Any:
        return summary.get("variants", {}).get(variant, {}).get(key, default)

    def _variant_gpu_stat(self, summary: Dict[str, Any], variant: str, key: str, default: Any = "-") -> Any:
        return summary.get("variants", {}).get(variant, {}).get(key, default)

    def append_round(
        self,
        round_num: int,
        summary: Dict[str, Any],
        round_time_seconds: Optional[float] = None,
        note: str = "",
    ) -> None:
        """Append a single round snapshot to the evolution logs."""
        timestamp = datetime.now().isoformat(timespec="seconds")

        record = {
            "timestamp": timestamp,
            "round": round_num,
            "round_time_seconds": round_time_seconds,
            "note": note,
            "summary": summary,
        }

        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        baseline_score = self._fmt_float(self._variant_stat(summary, "baseline", "fair_score"))
        attack_score = self._fmt_float(self._variant_stat(summary, "attack", "fair_score"))
        est_score = self._fmt_float(self._variant_stat(summary, "est", "fair_score"))

        baseline_elo = self._fmt_float(self._variant_stat(summary, "baseline", "fair_elo"))
        attack_elo = self._fmt_float(self._variant_stat(summary, "attack", "fair_elo"))
        est_elo = self._fmt_float(self._variant_stat(summary, "est", "fair_elo"))

        baseline_batch = self._fmt_float(self._variant_gpu_stat(summary, "baseline", "gpu_avg_batch"), 2)
        attack_batch = self._fmt_float(self._variant_gpu_stat(summary, "attack", "gpu_avg_batch"), 2)
        est_batch = self._fmt_float(self._variant_gpu_stat(summary, "est", "gpu_avg_batch"), 2)

        elapsed = self._fmt_float(round_time_seconds if round_time_seconds is not None else summary.get("elapsed_seconds", 0.0), 1)

        # Table row (8 core columns)
        line = (
            f"| {round_num} | {elapsed} | {baseline_score} | {attack_score} | {est_score} | "
            f"{baseline_elo} | {attack_elo} | {est_elo} |\n"
        )

        with self.markdown_path.open("a", encoding="utf-8") as handle:
            handle.write(line)

        # Dashboard summary below the table
        ranking = self._build_ranking(summary)
        batch_trend = self._build_batch_trend(summary)
        dashboard = (
            f"\n---\n"
            f"**R{round_num}** {ranking} | avg_batch: b={baseline_batch} a={attack_batch} e={est_batch}{batch_trend}\n"
        )

        with self.markdown_path.open("a", encoding="utf-8") as handle:
            handle.write(dashboard)

    def _build_ranking(self, summary: Dict[str, Any]) -> str:
        """Return a short ranking string like '1) baseline · 2) attack · 3) est'."""
        variants = summary.get("variants", {})
        ranked = []
        for variant in self._variant_order():
            elo_str = self._fmt_float(variants.get(variant, {}).get("fair_elo"))
            try:
                elo_val = float(elo_str) if elo_str != "-" else 0.0
            except Exception:
                elo_val = 0.0
            ranked.append((variant, elo_val))

        ranked.sort(key=lambda x: x[1], reverse=True)
        parts = []
        for i, (name, _) in enumerate(ranked):
            short = name[:3]
            parts.append(f"{i + 1}) {short}")
        return " · ".join(parts)

    def _build_batch_trend(self, summary: Dict[str, Any]) -> str:
        """Return trend arrows for avg_batch per variant, e.g. ' | b↑ a→ e↓'."""
        trends = []
        has_any_arrow = False
        for variant in self._variant_order():
            batch_str = self._fmt_float(
                self._variant_gpu_stat(summary, variant, "gpu_avg_batch"), 2
            )
            try:
                current = float(batch_str) if batch_str != "-" else 0.0
            except Exception:
                continue

            prev = self._prev_batches.get(variant)
            arrow = ""
            if prev is not None and current > 0 and prev > 0:
                diff_pct = (current - prev) / prev
                if abs(diff_pct) < 0.03:
                    arrow = "→"
                elif diff_pct > 0:
                    arrow = "↑"
                else:
                    arrow = "↓"

            self._prev_batches[variant] = current
            trends.append(arrow)
            if arrow not in ("", "→"):
                has_any_arrow = True

        if not has_any_arrow:
            return ""

        # Build compact label: only show arrows for variants that changed, or all with short labels
        parts = []
        for v, a in zip(self._variant_order(), trends):
            short = v[:3]
            parts.append(f"{short}{a}")
        return " | " + " ".join(parts)

    def _variant_order(self) -> list[str]:
        return ["baseline", "attack", "est"]

    def append_note(self, message: str) -> None:
        """Append a free-form note to the Markdown evolution log."""
        with self.markdown_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n- {message}\n")
