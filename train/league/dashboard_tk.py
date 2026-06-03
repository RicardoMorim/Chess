"""
Tkinter dashboard for the LeagueTrainer (Fase 3b).

A lightweight standalone client that polls the trainer's HTTP control plane
(at ``http://127.0.0.1:<port>``) and shows the same data as the browser
dashboard, but in a native Tk window.

Why a separate process?
  - Tkinter is single-threaded and not friendly to being embedded in the
    trainer's main thread.
  - Running in its own process means the dashboard can be started/stopped
    independently of training.
  - It's a consumer of the HTTP API only — never touches the trainer
    directly. This keeps the trainer's process clean.

Usage:
  python -m league.dashboard_tk
  python -m league.dashboard_tk --port 7861    # custom trainer port

The dashboard requires the trainer to be running (it just talks HTTP).
If the trainer isn't reachable, the dashboard shows a "connecting..."
status and retries every 2 seconds.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any, Dict, List, Optional
from urllib import request as urlrequest
from urllib.error import URLError

# Add project root to path so we can import from league package
_THIS = Path(__file__).resolve()
_TRAIN = _THIS.parent.parent
sys.path.insert(0, str(_TRAIN))

logger = logging.getLogger(__name__)


# =============================================================================
# HTTP client (synchronous; called from a worker thread)
# =============================================================================


class TrainerClient:
    """Minimal HTTP client to the trainer's control plane."""

    def __init__(self, base_url: str, timeout: float = 3.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.last_error: Optional[str] = None

    def _get(self, path: str) -> Optional[Dict[str, Any]]:
        try:
            with urlrequest.urlopen(self.base_url + path, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except (URLError, json.JSONDecodeError, OSError) as e:
            self.last_error = str(e)
            return None

    def _post(self, path: str, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            data = json.dumps(body).encode("utf-8")
            req = urlrequest.Request(
                self.base_url + path, data=data, method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urlrequest.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except (URLError, json.JSONDecodeError, OSError) as e:
            self.last_error = str(e)
            return None

    def get_status(self) -> Optional[Dict[str, Any]]:
        return self._get("/api/status")

    def get_knobs(self) -> Optional[Dict[str, Any]]:
        return self._get("/api/knobs")

    def get_modes(self) -> Optional[Dict[str, Any]]:
        return self._get("/api/modes")

    def get_checkpoints(self) -> Optional[List[Dict[str, Any]]]:
        return self._get("/api/checkpoints")

    def get_variants(self) -> Optional[List[Dict[str, Any]]]:
        return self._get("/api/variants")

    def set_mode(self, mode: str) -> bool:
        r = self._post("/api/mode", {"mode": mode})
        return bool(r and r.get("ok"))

    def set_knob(self, name: str, value: Any) -> bool:
        r = self._post("/api/knobs", {name: value})
        return bool(r and r.get(name))

    def set_auto_mode(self, enabled: bool) -> bool:
        r = self._post("/api/auto_mode", {"enabled": enabled})
        return bool(r and r.get("ok"))

    def set_paused(self, paused: bool) -> bool:
        r = self._post("/api/pause", {"paused": paused})
        return bool(r and r.get("ok"))


# =============================================================================
# Tkinter widgets
# =============================================================================


class ResourceBars(tk.Frame):
    """Three progress bars: VRAM, CPU, RAM."""

    def __init__(self, master):
        super().__init__(master)
        self._bars: Dict[str, ttk.Progressbar] = {}
        self._labels: Dict[str, tk.Label] = {}
        for name, color in [("vram", "#3fb950"), ("cpu", "#4a9eff"), ("ram", "#d29922")]:
            row = tk.Frame(self)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=name.upper(), width=6, anchor="w").pack(side=tk.LEFT)
            bar = ttk.Progressbar(row, length=200, mode="determinate", maximum=100)
            bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            lbl = tk.Label(row, text="--", width=14, anchor="e")
            lbl.pack(side=tk.LEFT)
            self._bars[name] = bar
            self._labels[name] = lbl

    def update(self, r: Dict[str, Any]) -> None:
        vram_pct = r.get("vram_pct")
        self._bars["vram"]["value"] = vram_pct if vram_pct is not None else 0
        used = r.get("vram_used_mb")
        total = r.get("vram_total_mb")
        self._labels["vram"].config(
            text=f"{used}/{total} MB" if used is not None and total is not None else "--"
        )
        cpu = r.get("cpu_pct")
        self._bars["cpu"]["value"] = cpu if cpu is not None else 0
        self._labels["cpu"].config(text=f"{cpu:.1f}%" if cpu is not None else "--")
        ram = r.get("ram_pct")
        self._bars["ram"]["value"] = ram if ram is not None else 0
        self._labels["ram"].config(text=f"{ram:.1f}%" if ram is not None else "--")


class VariantsPanel(tk.Frame):
    """Per-variant loss + buffer display."""

    def __init__(self, master, variants: List[str]):
        super().__init__(master)
        self._labels: Dict[str, Dict[str, tk.Label]] = {}
        for v in variants:
            box = tk.LabelFrame(self, text=v, padx=8, pady=4)
            box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
            loss_lbl = tk.Label(box, text="loss: --", font=("Consolas", 11))
            loss_lbl.pack(anchor="w")
            tput_lbl = tk.Label(box, text="tput: --", font=("Consolas", 11))
            tput_lbl.pack(anchor="w")
            buf_lbl = tk.Label(box, text="buf: --", font=("Consolas", 11))
            buf_lbl.pack(anchor="w")
            self._labels[v] = {"loss": loss_lbl, "tput": tput_lbl, "buf": buf_lbl}

    def update(self, status: Dict[str, Any]) -> None:
        losses = status.get("losses", {})
        tputs = status.get("throughput_gpm", {})
        bufs = status.get("buffers", {})
        for v, labels in self._labels.items():
            l = losses.get(v)
            labels["loss"].config(text=f"loss: {l:.3f}" if l is not None else "loss: --")
            t = tputs.get(v)
            labels["tput"].config(text=f"tput: {t:.1f}/min" if t is not None else "tput: --")
            b = bufs.get(v, {})
            size = b.get("size", 0)
            cap = b.get("capacity", 0)
            pct = b.get("fill_pct", 0)
            labels["buf"].config(text=f"buf: {size}/{cap} ({pct:.1f}%)")


class CheckpointTable(tk.Frame):
    """Simple listbox of checkpoints; double-click to spectate."""

    def __init__(self, master, on_select=None):
        super().__init__(master)
        self._on_select = on_select
        self.listbox = tk.Listbox(self, font=("Consolas", 10), height=8)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.listbox.yview)
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.bind("<Double-Button-1>", self._on_double)
        self._items: List[Dict[str, Any]] = []

    def set_items(self, items: List[Dict[str, Any]]) -> None:
        self._items = items
        self.listbox.delete(0, tk.END)
        for c in items[:30]:
            label = f"{c['variant']:<10} step {c['step']:<6} {c['size_mb']:>5.1f} MB"
            self.listbox.insert(tk.END, label)

    def _on_double(self, _evt):
        sel = self.listbox.curselection()
        if not sel:
            return
        ckpt = self._items[sel[0]]
        if self._on_select:
            self._on_select(ckpt)


# =============================================================================
# Main window
# =============================================================================


class DashboardApp:
    def __init__(self, client: TrainerClient):
        self.client = client
        self.root = tk.Tk()
        self.root.title("Chess Trainer Dashboard")
        self.root.geometry("780x560")
        self._build_ui()
        self._running = True
        # Polling thread
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def _build_ui(self) -> None:
        # Top: title + status
        top = tk.Frame(self.root, padx=10, pady=8)
        top.pack(fill=tk.X)
        tk.Label(top, text="Chess Trainer", font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT)
        self._status_label = tk.Label(top, text="  connecting...", fg="#888")
        self._status_label.pack(side=tk.LEFT, padx=12)
        self._round_label = tk.Label(top, text="", font=("Consolas", 10))
        self._round_label.pack(side=tk.RIGHT)

        # Mode buttons
        mode_frame = tk.LabelFrame(self.root, text="Performance mode", padx=8, pady=6)
        mode_frame.pack(fill=tk.X, padx=10, pady=4)
        self._mode_buttons: Dict[str, tk.Button] = {}
        for mode in ("eco", "balanced", "boost"):
            b = tk.Button(mode_frame, text=mode.upper(), width=10,
                          command=lambda m=mode: self._set_mode(m))
            b.pack(side=tk.LEFT, padx=4)
            self._mode_buttons[mode] = b

        # Auto-mode + pause
        ctrl_frame = tk.Frame(self.root, padx=10)
        ctrl_frame.pack(fill=tk.X, pady=4)
        self._auto_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl_frame, text="Auto-mode", variable=self._auto_var,
                       command=self._toggle_auto).pack(side=tk.LEFT, padx=4)
        self._pause_btn = tk.Button(ctrl_frame, text="⏸ Pause", width=10,
                                    command=self._toggle_pause)
        self._pause_btn.pack(side=tk.LEFT, padx=4)
        tk.Button(ctrl_frame, text="↻ Refresh", command=self._refresh_once).pack(side=tk.LEFT, padx=4)

        # Variants panel
        vp_frame = tk.LabelFrame(self.root, text="Variants", padx=6, pady=6)
        vp_frame.pack(fill=tk.X, padx=10, pady=4)
        self._variants_panel = VariantsPanel(vp_frame, ["baseline", "attack", "est"])
        self._variants_panel.pack(fill=tk.X)

        # Resources
        rb_frame = tk.LabelFrame(self.root, text="System resources", padx=6, pady=6)
        rb_frame.pack(fill=tk.X, padx=10, pady=4)
        self._resources = ResourceBars(rb_frame)
        self._resources.pack(fill=tk.X)

        # Checkpoints
        ck_frame = tk.LabelFrame(self.root, text="Checkpoints (double-click to spectate)", padx=6, pady=6)
        ck_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        self._ck_table = CheckpointTable(ck_frame, on_select=self._on_ckpt_select)
        self._ck_table.pack(fill=tk.BOTH, expand=True)

        # Footer
        self._footer = tk.Label(self.root, text="", font=("Consolas", 9), fg="#666")
        self._footer.pack(fill=tk.X, padx=10, pady=2)

    # ---- UI callbacks -------------------------------------------------------

    def _set_mode(self, mode: str) -> None:
        ok = self.client.set_mode(mode)
        if not ok:
            self._status_label.config(text=f"  mode change failed", fg="#f85149")

    def _toggle_auto(self) -> None:
        self.client.set_auto_mode(self._auto_var.get())

    def _toggle_pause(self) -> None:
        paused = "⏸ Pause" in self._pause_btn.cget("text")
        ok = self.client.set_paused(not paused)
        if ok:
            self._pause_btn.config(text="▶ Resume" if not paused else "⏸ Pause")

    def _refresh_once(self) -> None:
        self._poll_once()

    def _on_ckpt_select(self, ckpt: Dict[str, Any]) -> None:
        other_variant = "attack" if ckpt["variant"] == "baseline" else "baseline"
        # Build a match payload and POST it
        try:
            payload = {
                "type": "model",
                "white": f"{ckpt['variant']}_step_{ckpt['step']}",
                "black": other_variant,
                "visits": 100,
            }
            r = self.client._post("/api/matches", payload)
            if r and r.get("ok"):
                self._footer.config(text=f"Match #{r['match']['id']} queued — open browser dashboard to spectate")
        except Exception as e:
            self._footer.config(text=f"Queue error: {e}")

    # ---- Polling ------------------------------------------------------------

    def _poll_loop(self) -> None:
        # Initial fetch
        self._poll_once()
        # Then poll every 2s
        while self._running:
            time.sleep(2.0)
            if not self._running:
                break
            try:
                self._poll_once()
            except Exception as e:
                logger.debug(f"poll error: {e}")

    def _poll_once(self) -> None:
        status = self.client.get_status()
        if status is None:
            self.root.after(0, lambda: self._status_label.config(
                text=f"  offline ({self.client.last_error})", fg="#f85149"))
            return
        # Snapshot for thread safety
        modes = self.client.get_modes() or {}
        checkpoints = self.client.get_checkpoints() or []
        # Schedule UI update on main thread
        self.root.after(0, lambda: self._apply_status(status, modes, checkpoints))

    def _apply_status(self, s: Dict[str, Any], modes: Dict[str, Any],
                      ckpts: List[Dict[str, Any]]) -> None:
        self._status_label.config(text="  online", fg="#3fb950")
        self._round_label.config(text=(
            f"round {s.get('round', 0)} · "
            f"games {s.get('total_games', 0)} · "
            f"steps {s.get('total_training_steps', 0)}"
        ))
        # Mode buttons
        cur_mode = s.get("performance_mode", "balanced")
        for mode, btn in self._mode_buttons.items():
            btn.config(relief=tk.SUNKEN if mode == cur_mode else tk.RAISED)
        # Pause button
        self._pause_btn.config(text=("▶ Resume" if s.get("training_paused") else "⏸ Pause"))
        # Auto-mode (only if state changed externally)
        if s.get("auto_mode") != self._auto_var.get():
            self._auto_var.set(bool(s.get("auto_mode")))
        # Variants
        self._variants_panel.update(s)
        # Resources
        self._resources.update(s.get("resources", {}))
        # Checkpoints
        self._ck_table.set_items(ckpts)
        # Footer
        self._footer.config(text=f"last update: {time.strftime('%H:%M:%S')}")

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            self._running = False
            self._poll_thread.join(timeout=2.0)


# =============================================================================
# Entry point
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Tk dashboard for the chess trainer")
    parser.add_argument("--host", default="127.0.0.1", help="trainer host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="trainer control port (default 7860)")
    parser.add_argument("--poll-interval", type=float, default=2.0,
                        help="seconds between status polls (default 2)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    base = f"http://{args.host}:{args.port}"
    client = TrainerClient(base)
    app = DashboardApp(client)
    print(f"Dashboard connecting to {base} (poll={args.poll_interval}s)")
    app.run()


if __name__ == "__main__":
    main()
