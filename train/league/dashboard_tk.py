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
from tkinter import font as tkfont
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
# Theme
# =============================================================================


class Theme:
    """Dark theme matching the browser dashboard."""

    BG = "#0f1419"
    PANEL = "#1a2028"
    PANEL_2 = "#232b36"
    BORDER = "#2c3540"
    TEXT = "#e6edf3"
    MUTED = "#8b949e"
    ACCENT = "#4a9eff"
    ACCENT_2 = "#58a6ff"
    GOOD = "#3fb950"
    WARN = "#d29922"
    BAD = "#f85149"
    ECO = "#3fb950"
    BALANCED = "#4a9eff"
    BOOST = "#f85149"

    FONT_UI = ("Segoe UI", 10)
    FONT_BOLD = ("Segoe UI", 10, "bold")
    FONT_TITLE = ("Segoe UI", 14, "bold")
    FONT_H2 = ("Segoe UI", 10, "bold")
    FONT_MONO = ("Consolas", 10)
    FONT_MONO_BIG = ("Consolas", 13, "bold")


def apply_theme(root: tk.Tk) -> None:
    """Configure ttk styles for a dark, modern look."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    root.configure(bg=Theme.BG)

    style.configure(".",
                    background=Theme.BG,
                    foreground=Theme.TEXT,
                    fieldbackground=Theme.PANEL_2,
                    bordercolor=Theme.BORDER,
                    font=Theme.FONT_UI)
    style.configure("TFrame", background=Theme.BG)
    style.configure("Panel.TFrame", background=Theme.PANEL)
    style.configure("Panel2.TFrame", background=Theme.PANEL_2)
    style.configure("TLabel", background=Theme.BG, foreground=Theme.TEXT)
    style.configure("Panel.TLabel", background=Theme.PANEL, foreground=Theme.TEXT)
    style.configure("Muted.TLabel", background=Theme.BG, foreground=Theme.MUTED)
    style.configure("Panel.Muted.TLabel", background=Theme.PANEL, foreground=Theme.MUTED)
    style.configure("Title.TLabel", background=Theme.BG, foreground=Theme.TEXT,
                    font=Theme.FONT_TITLE)
    style.configure("H2.TLabel", background=Theme.PANEL, foreground=Theme.MUTED,
                    font=Theme.FONT_H2)
    style.configure("Mono.TLabel", background=Theme.PANEL, foreground=Theme.TEXT,
                    font=Theme.FONT_MONO)
    style.configure("MonoBig.TLabel", background=Theme.PANEL_2, foreground=Theme.TEXT,
                    font=Theme.FONT_MONO_BIG)
    style.configure("Good.TLabel", background=Theme.PANEL, foreground=Theme.GOOD)
    style.configure("Bad.TLabel", background=Theme.PANEL, foreground=Theme.BAD)
    style.configure("Muted.TLabelframe", background=Theme.PANEL, foreground=Theme.MUTED,
                    bordercolor=Theme.BORDER, relief="solid")
    style.configure("Muted.TLabelframe.Label", background=Theme.PANEL,
                    foreground=Theme.MUTED, font=Theme.FONT_H2)
    style.configure("TLabelframe", background=Theme.PANEL, foreground=Theme.TEXT,
                    bordercolor=Theme.BORDER)
    style.configure("TLabelframe.Label", background=Theme.PANEL,
                    foreground=Theme.MUTED, font=Theme.FONT_H2)
    style.configure("TButton",
                    background=Theme.PANEL_2, foreground=Theme.TEXT,
                    bordercolor=Theme.BORDER, focuscolor=Theme.ACCENT,
                    padding=(10, 5), relief="flat")
    style.map("TButton",
              background=[("active", Theme.BORDER), ("pressed", Theme.BORDER)],
              foreground=[("disabled", Theme.MUTED)])
    # Mode buttons
    for mode, color, dark in [
        ("ModeEco.TButton", Theme.ECO, True),
        ("ModeBalanced.TButton", Theme.BALANCED, True),
        ("ModeBoost.TButton", Theme.BOOST, False),
    ]:
        style.configure(mode, background=Theme.PANEL_2, foreground=Theme.MUTED)
        style.map(mode,
                  background=[("active", Theme.BORDER), ("pressed", color),
                              ("selected", color)],
                  foreground=[("selected", "#000" if dark else "#fff")])
    # Progressbar — three flavours
    style.configure("VRAM.Horizontal.TProgressbar", troughcolor=Theme.PANEL_2,
                    background=Theme.GOOD, bordercolor=Theme.BORDER, lightcolor=Theme.GOOD,
                    darkcolor=Theme.GOOD)
    style.configure("CPU.Horizontal.TProgressbar", troughcolor=Theme.PANEL_2,
                    background=Theme.ACCENT, bordercolor=Theme.BORDER,
                    lightcolor=Theme.ACCENT, darkcolor=Theme.ACCENT)
    style.configure("RAM.Horizontal.TProgressbar", troughcolor=Theme.PANEL_2,
                    background=Theme.WARN, bordercolor=Theme.BORDER,
                    lightcolor=Theme.WARN, darkcolor=Theme.WARN)
    # Treeview
    style.configure("Treeview",
                    background=Theme.PANEL_2, foreground=Theme.TEXT,
                    fieldbackground=Theme.PANEL_2, bordercolor=Theme.BORDER,
                    rowheight=24, font=Theme.FONT_MONO)
    style.configure("Treeview.Heading",
                    background=Theme.PANEL, foreground=Theme.MUTED,
                    relief="flat", font=Theme.FONT_H2, padding=(8, 6))
    style.map("Treeview",
              background=[("selected", Theme.ACCENT)],
              foreground=[("selected", "#000")])
    style.map("Treeview.Heading",
              background=[("active", Theme.BORDER)])
    # Scrollbar
    style.configure("Vertical.TScrollbar",
                    background=Theme.PANEL_2, troughcolor=Theme.PANEL,
                    bordercolor=Theme.BORDER, arrowcolor=Theme.MUTED)
    # Checkbutton
    style.configure("TCheckbutton", background=Theme.BG, foreground=Theme.TEXT,
                    focuscolor=Theme.BG, indicatorcolor=Theme.PANEL_2)
    style.map("TCheckbutton",
              background=[("active", Theme.BG)],
              indicatorcolor=[("selected", Theme.ACCENT)],
              foreground=[("active", Theme.TEXT), ("disabled", Theme.MUTED)])


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
# Widgets
# =============================================================================


class StatusBar(tk.Frame):
    """Top header bar: title, status dot, round, last update."""

    def __init__(self, master):
        super().__init__(master, bg=Theme.BG, height=56)
        self.pack_propagate(False)
        # Left: brand
        left = tk.Frame(self, bg=Theme.BG)
        left.pack(side=tk.LEFT, padx=14, pady=10)
        tk.Label(left, text="♞", font=("Segoe UI", 20), bg=Theme.BG,
                 fg=Theme.ACCENT).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(left, text="Chess Trainer", font=Theme.FONT_TITLE,
                 bg=Theme.BG, fg=Theme.TEXT).pack(side=tk.LEFT)
        # Status dot
        self._dot = tk.Canvas(left, width=12, height=12, bg=Theme.BG,
                              highlightthickness=0)
        self._dot.pack(side=tk.LEFT, padx=(12, 4))
        self._dot_id = self._dot.create_oval(2, 2, 10, 10, fill=Theme.MUTED, outline="")
        self._status_text = tk.Label(left, text="connecting...", font=Theme.FONT_UI,
                                     bg=Theme.BG, fg=Theme.MUTED)
        self._status_text.pack(side=tk.LEFT)
        # Right: round + games + steps
        right = tk.Frame(self, bg=Theme.BG)
        right.pack(side=tk.RIGHT, padx=14, pady=10)
        self._round_lbl = tk.Label(right, text="round -- · games -- · steps --",
                                   font=Theme.FONT_MONO, bg=Theme.BG, fg=Theme.MUTED)
        self._round_lbl.pack(side=tk.RIGHT)

    def set_status(self, online: bool, msg: str = "") -> None:
        if online:
            self._dot.itemconfig(self._dot_id, fill=Theme.GOOD)
            self._status_text.config(text="online" if not msg else f"online · {msg}",
                                     fg=Theme.GOOD)
        else:
            self._dot.itemconfig(self._dot_id, fill=Theme.BAD)
            self._status_text.config(text=f"offline · {msg}", fg=Theme.BAD)

    def set_round(self, round_n: int, games: int, steps: int) -> None:
        self._round_lbl.config(
            text=f"round {round_n}  ·  games {_fmt_int(games)}  ·  steps {_fmt_int(steps)}",
            fg=Theme.TEXT,
        )


class ModePanel(tk.LabelFrame):
    """Mode buttons + auto-mode + pause controls."""

    def __init__(self, master, on_mode, on_auto, on_pause):
        super().__init__(master, text="PERFORMANCE MODE", padx=12, pady=10,
                         bg=Theme.PANEL, fg=Theme.MUTED,
                         font=Theme.FONT_H2, bd=1, relief="solid",
                         highlightbackground=Theme.BORDER)
        self._on_mode = on_mode
        self._on_auto = on_auto
        self._on_pause = on_pause
        # Mode buttons (segmented control look)
        self._mode_frame = tk.Frame(self, bg=Theme.PANEL_2, bd=0,
                                     highlightthickness=1,
                                     highlightbackground=Theme.BORDER)
        self._mode_frame.pack(fill=tk.X, pady=(0, 10))
        self._mode_buttons: Dict[str, tk.Button] = {}
        for i, (mode, style, color) in enumerate([
            ("eco", "ModeEco.TButton", Theme.ECO),
            ("balanced", "ModeBalanced.TButton", Theme.BALANCED),
            ("boost", "ModeBoost.TButton", Theme.BOOST),
        ]):
            b = ttk.Button(self._mode_frame, text=mode.upper(), style=style,
                           command=lambda m=mode: self._on_mode(m))
            b.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2, pady=2)
            self._mode_buttons[mode] = b
        # Auto + pause row
        ctrl = tk.Frame(self, bg=Theme.PANEL)
        ctrl.pack(fill=tk.X)
        self._auto_var = tk.BooleanVar(value=False)
        self._auto_btn = ttk.Checkbutton(
            ctrl, text="Auto-mode (CPU-based)", variable=self._auto_var,
            command=self._on_auto_toggle, style="TCheckbutton")
        self._auto_btn.pack(side=tk.LEFT)
        self._pause_btn = ttk.Button(ctrl, text="⏸  Pause",
                                     command=self._on_pause)
        self._pause_btn.pack(side=tk.RIGHT)

    def _on_auto_toggle(self) -> None:
        self._on_auto(self._auto_var.get())

    def set_mode(self, mode: str) -> None:
        for m, b in self._mode_buttons.items():
            state = m == mode
            b.state(["selected"] if state else ["!selected"])

    def set_auto(self, enabled: bool) -> None:
        if self._auto_var.get() != enabled:
            self._auto_var.set(enabled)

    def set_paused(self, paused: bool) -> None:
        self._pause_btn.config(text="▶  Resume" if paused else "⏸  Pause")


class MetricCard(tk.Frame):
    """Single large metric tile."""

    def __init__(self, master, label: str, color: str = Theme.ACCENT):
        super().__init__(master, bg=Theme.PANEL, bd=1, relief="solid",
                         highlightbackground=Theme.BORDER)
        self._color = color
        tk.Label(self, text=label.upper(), font=("Segoe UI", 8, "bold"),
                 bg=Theme.PANEL, fg=Theme.MUTED, padx=10, pady=2).pack(anchor="w")
        self._value = tk.Label(self, text="--", font=("Consolas", 22, "bold"),
                                bg=Theme.PANEL, fg=color, padx=10, pady=4)
        self._value.pack(anchor="w")
        self._sub = tk.Label(self, text=" ", font=("Segoe UI", 9),
                             bg=Theme.PANEL, fg=Theme.MUTED, padx=10)
        self._sub.pack(anchor="w", pady=(0, 8))

    def set(self, value: str, sub: str = "") -> None:
        self._value.config(text=value)
        self._sub.config(text=sub or " ")


class VariantsPanel(tk.Frame):
    """Row of cards: per-variant loss / throughput / buffer."""

    def __init__(self, master):
        super().__init__(master, bg=Theme.BG)
        self._cards: Dict[str, Dict[str, tk.Label]] = {}
        for v in ("baseline", "attack", "est"):
            card = tk.Frame(self, bg=Theme.PANEL, bd=1, relief="solid",
                            highlightbackground=Theme.BORDER)
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
            # Header with color bar
            header = tk.Frame(card, bg=Theme.PANEL)
            header.pack(fill=tk.X, padx=0, pady=0)
            tk.Frame(header, bg=_variant_color(v), width=4, height=24).pack(
                side=tk.LEFT, fill=tk.Y)
            tk.Label(header, text=v.upper(), font=("Segoe UI", 10, "bold"),
                     bg=Theme.PANEL, fg=Theme.TEXT, padx=8).pack(side=tk.LEFT, pady=6)
            # Loss big
            loss_lbl = tk.Label(card, text="--", font=("Consolas", 24, "bold"),
                                bg=Theme.PANEL, fg=Theme.ACCENT_2, padx=10, pady=2)
            loss_lbl.pack(anchor="w")
            loss_sub = tk.Label(card, text="loss", font=("Segoe UI", 8, "bold"),
                                bg=Theme.PANEL, fg=Theme.MUTED, padx=10)
            loss_sub.pack(anchor="w")
            # Throughput
            tput_lbl = tk.Label(card, text="--", font=("Consolas", 13, "bold"),
                                bg=Theme.PANEL, fg=Theme.GOOD, padx=10, pady=0)
            tput_lbl.pack(anchor="w")
            tput_sub = tk.Label(card, text="games/min", font=("Segoe UI", 8, "bold"),
                                bg=Theme.PANEL, fg=Theme.MUTED, padx=10)
            tput_sub.pack(anchor="w")
            # Buffer bar
            bar_frame = tk.Frame(card, bg=Theme.PANEL)
            bar_frame.pack(fill=tk.X, padx=10, pady=(8, 2))
            bar = ttk.Progressbar(bar_frame, length=120, mode="determinate",
                                  maximum=100, style="VRAM.Horizontal.TProgressbar")
            bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
            buf_text = tk.Label(card, text="buf --/--", font=("Consolas", 9),
                                bg=Theme.PANEL, fg=Theme.MUTED, padx=10, pady=4)
            buf_text.pack(anchor="w")
            self._cards[v] = {"loss": loss_lbl, "tput": tput_lbl, "bar": bar, "buf": buf_text}

    def update(self, status: Dict[str, Any]) -> None:
        losses = status.get("losses", {}) or {}
        tputs = status.get("throughput_gpm", {}) or {}
        bufs = status.get("buffers", {}) or {}
        for v, w in self._cards.items():
            l = losses.get(v)
            w["loss"].config(text=f"{l:.3f}" if l is not None else "--")
            t = tputs.get(v)
            w["tput"].config(text=f"{t:.1f}" if t is not None else "--")
            b = bufs.get(v, {}) or {}
            size = b.get("size", 0)
            cap = b.get("capacity", 0)
            pct = b.get("fill_pct", 0) or 0
            w["bar"]["value"] = pct
            # Colour the bar by fill level
            style = ("RAM.Horizontal.TProgressbar" if pct > 80
                     else "CPU.Horizontal.TProgressbar" if pct > 40
                     else "VRAM.Horizontal.TProgressbar")
            w["bar"].config(style=style)
            w["buf"].config(text=f"buf {_fmt_int(size)}/{_fmt_int(cap)} ({pct:.0f}%)")


class ResourcesPanel(tk.LabelFrame):
    """Three resource bars (VRAM, CPU, RAM) with status text."""

    def __init__(self, master):
        super().__init__(master, text="SYSTEM", padx=12, pady=10,
                         bg=Theme.PANEL, fg=Theme.MUTED, font=Theme.FONT_H2,
                         bd=1, relief="solid", highlightbackground=Theme.BORDER)
        self._bars: Dict[str, ttk.Progressbar] = {}
        self._labels: Dict[str, tk.Label] = {}
        rows = [
            ("vram", "VRAM", "VRAM.Horizontal.TProgressbar", Theme.GOOD),
            ("cpu",  "CPU ", "CPU.Horizontal.TProgressbar", Theme.ACCENT),
            ("ram",  "RAM ", "RAM.Horizontal.TProgressbar", Theme.WARN),
        ]
        for key, label, style, color in rows:
            row = tk.Frame(self, bg=Theme.PANEL)
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=label, font=("Consolas", 9, "bold"),
                     bg=Theme.PANEL, fg=color, width=5, anchor="w").pack(side=tk.LEFT)
            bar = ttk.Progressbar(row, length=200, mode="determinate", maximum=100,
                                  style=style)
            bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
            lbl = tk.Label(row, text="--", font=("Consolas", 9),
                           bg=Theme.PANEL, fg=Theme.MUTED, width=14, anchor="e")
            lbl.pack(side=tk.LEFT)
            self._bars[key] = bar
            self._labels[key] = lbl

    def update(self, r: Dict[str, Any]) -> None:
        vram_pct = r.get("vram_pct")
        used = r.get("vram_used_mb")
        total = r.get("vram_total_mb")
        self._bars["vram"]["value"] = vram_pct if vram_pct is not None else 0
        self._labels["vram"].config(
            text=f"{used}/{total} MB" if used is not None and total is not None else "--"
        )
        cpu = r.get("cpu_pct")
        self._bars["cpu"]["value"] = cpu if cpu is not None else 0
        self._labels["cpu"].config(text=f"{cpu:.1f}%" if cpu is not None else "--")
        ram = r.get("ram_pct")
        self._bars["ram"]["value"] = ram if ram is not None else 0
        self._labels["ram"].config(text=f"{ram:.1f}%" if ram is not None else "--")


# Backwards-compatible alias for the old widget name (kept for tests).
ResourceBars = ResourcesPanel


class CheckpointTable(tk.LabelFrame):
    """Treeview-based checkpoint table with sort and double-click spectate."""

    def __init__(self, master, on_select=None):
        super().__init__(master, text="CHECKPOINTS  (double-click to spectate)",
                         padx=8, pady=8, bg=Theme.PANEL, fg=Theme.MUTED,
                         font=Theme.FONT_H2, bd=1, relief="solid",
                         highlightbackground=Theme.BORDER)
        self._on_select = on_select
        cols = ("variant", "step", "size", "age", "name")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=8,
                                 style="Treeview")
        self.tree.heading("variant", text="VARIANT", anchor="w",
                          command=lambda: self._sort("variant"))
        self.tree.heading("step", text="STEP", anchor="e",
                          command=lambda: self._sort("step", numeric=True))
        self.tree.heading("size", text="SIZE", anchor="e",
                          command=lambda: self._sort("size", numeric=True))
        self.tree.heading("age", text="AGE", anchor="e",
                          command=lambda: self._sort("age", numeric=True))
        self.tree.heading("name", text="FILE", anchor="w")
        self.tree.column("variant", width=90, anchor="w", stretch=False)
        self.tree.column("step", width=70, anchor="e", stretch=False)
        self.tree.column("size", width=80, anchor="e", stretch=False)
        self.tree.column("age", width=60, anchor="e", stretch=False)
        self.tree.column("name", width=200, anchor="w", stretch=True)
        self.tree.tag_configure("odd", background=Theme.PANEL_2)
        self.tree.tag_configure("even", background=Theme.PANEL)
        self.tree.bind("<Double-Button-1>", self._on_double)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview,
                           style="Vertical.TScrollbar")
        self.tree.config(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._items: List[Dict[str, Any]] = []
        self._sort_col: Optional[str] = None
        self._sort_reverse: bool = False

    def set_items(self, items: List[Dict[str, Any]]) -> None:
        self._items = items
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        for i, c in enumerate(items[:30]):
            tag = "odd" if i % 2 else "even"
            self.tree.insert("", tk.END, values=(
                c["variant"],
                c["step"],
                f"{c['size_mb']:.1f} MB",
                _format_age(c.get("mtime", 0)),
                c.get("name", ""),
            ), tags=(tag,))
        # Empty placeholder
        if not items:
            self.tree.insert("", tk.END,
                             values=("--", "--", "--", "--", "no checkpoints yet"),
                             tags=("even",))

    def _sort(self, col: str, numeric: bool = False) -> None:
        if not self._items:
            return
        if self._sort_col == col:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_col = col
            self._sort_reverse = False
        items = list(self._items)
        def keyfn(c):
            v = c.get(col)
            if v is None and col == "age":
                # age = now - mtime; smaller mtime = older; sort by mtime desc when reverse
                v = c.get("mtime", 0)
                return v
            return v
        items.sort(key=keyfn, reverse=self._sort_reverse)
        self.set_items(items)

    def _on_double(self, _evt):
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        if idx < len(self._items):
            ckpt = self._items[idx]
            if self._on_select:
                self._on_select(ckpt)


# =============================================================================
# Main window
# =============================================================================


class DashboardApp:
    def __init__(self, client: TrainerClient):
        self.client = client
        self.root = tk.Tk()
        self.root.title("♞ Chess Trainer Dashboard")
        self.root.geometry("1080x680")
        self.root.minsize(880, 580)
        apply_theme(self.root)
        self._build_ui()
        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def _build_ui(self) -> None:
        # Top status bar
        self._status_bar = StatusBar(self.root)
        self._status_bar.pack(fill=tk.X)

        # Body container
        body = tk.Frame(self.root, bg=Theme.BG)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # Top row: mode panel + summary metric cards
        top_row = tk.Frame(body, bg=Theme.BG)
        top_row.pack(fill=tk.X, pady=(0, 8))

        self._mode_panel = ModePanel(top_row,
                                     on_mode=self._on_mode,
                                     on_auto=self._on_auto,
                                     on_pause=self._on_pause)
        self._mode_panel.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        # Right: 3 summary metric cards
        cards_col = tk.Frame(top_row, bg=Theme.BG)
        cards_col.pack(side=tk.LEFT, fill=tk.Y)
        self._card_total = MetricCard(cards_col, "TOTAL GAMES", Theme.ACCENT_2)
        self._card_total.pack(fill=tk.X, pady=2)
        self._card_steps = MetricCard(cards_col, "TRAIN STEPS", Theme.GOOD)
        self._card_steps.pack(fill=tk.X, pady=2)
        self._card_state = MetricCard(cards_col, "STATE", Theme.WARN)
        self._card_state.pack(fill=tk.X, pady=2)

        # Variants row
        vp_frame = tk.LabelFrame(body, text="VARIANTS", padx=8, pady=8,
                                 bg=Theme.PANEL, fg=Theme.MUTED,
                                 font=Theme.FONT_H2, bd=1, relief="solid",
                                 highlightbackground=Theme.BORDER)
        vp_frame.pack(fill=tk.X, pady=(0, 8))
        self._variants_panel = VariantsPanel(vp_frame)
        self._variants_panel.pack(fill=tk.X)

        # Resources row
        rb_frame = tk.Frame(body, bg=Theme.BG)
        rb_frame.pack(fill=tk.X, pady=(0, 8))
        self._resources = ResourcesPanel(rb_frame)
        self._resources.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        # Help / actions
        actions = tk.LabelFrame(rb_frame, text="ACTIONS", padx=12, pady=10,
                                bg=Theme.PANEL, fg=Theme.MUTED,
                                font=Theme.FONT_H2, bd=1, relief="solid",
                                highlightbackground=Theme.BORDER)
        actions.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Button(actions, text="↻  Refresh now",
                   command=self._refresh_once).pack(fill=tk.X, pady=2)
        self._footer = tk.Label(actions, text="last update: --",
                                font=("Consolas", 9), bg=Theme.PANEL,
                                fg=Theme.MUTED, padx=4, pady=4)
        self._footer.pack(fill=tk.X)

        # Checkpoint table
        self._ck_table = CheckpointTable(body, on_select=self._on_ckpt_select)
        self._ck_table.pack(fill=tk.BOTH, expand=True)

    # ---- UI callbacks -------------------------------------------------------

    def _on_mode(self, mode: str) -> None:
        ok = self.client.set_mode(mode)
        if not ok:
            self._footer.config(text=f"mode change failed: {self.client.last_error}",
                                fg=Theme.BAD)
        else:
            self._footer.config(text=f"mode -> {mode}", fg=Theme.GOOD)
            self._mode_panel.set_mode(mode)

    def _on_auto(self, enabled: bool) -> None:
        ok = self.client.set_auto_mode(enabled)
        if not ok:
            self._footer.config(text=f"auto-mode failed: {self.client.last_error}",
                                fg=Theme.BAD)

    def _on_pause(self) -> None:
        cur = "Resume" in self._pause_btn.cget("text")
        target = not cur
        ok = self.client.set_paused(target)
        if ok:
            self._mode_panel.set_paused(target)
        else:
            self._footer.config(text=f"pause failed: {self.client.last_error}",
                                fg=Theme.BAD)

    def _refresh_once(self) -> None:
        # Run a poll on the worker thread for safety; the main thread
        # doesn't actually block (the worker just returns faster).
        threading.Thread(target=self._poll_once, daemon=True).start()

    def _on_ckpt_select(self, ckpt: Dict[str, Any]) -> None:
        other_variant = "attack" if ckpt["variant"] == "baseline" else "baseline"
        try:
            payload = {
                "type": "model",
                "white": f"{ckpt['variant']}_step_{ckpt['step']}",
                "black": other_variant,
                "visits": 100,
            }
            r = self.client._post("/api/matches", payload)
            if r and r.get("ok"):
                mid = r.get("match", {}).get("id", "?")
                self._footer.config(
                    text=f"match #{mid} queued — open browser dashboard to spectate",
                    fg=Theme.GOOD)
            else:
                self._footer.config(
                    text=f"queue failed: {r.get('error', self.client.last_error) if r else self.client.last_error}",
                    fg=Theme.BAD)
        except Exception as e:
            self._footer.config(text=f"queue error: {e}", fg=Theme.BAD)

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
            self.root.after(0, lambda: self._status_bar.set_status(
                False, self.client.last_error or "no response"))
            return
        modes = self.client.get_modes() or {}
        checkpoints = self.client.get_checkpoints() or []
        self.root.after(0, lambda: self._apply_status(status, modes, checkpoints))

    def _apply_status(self, s: Dict[str, Any], modes: Dict[str, Any],
                      ckpts: List[Dict[str, Any]]) -> None:
        self._status_bar.set_status(True)
        self._status_bar.set_round(
            s.get("round", 0),
            s.get("total_games", 0),
            s.get("total_training_steps", 0),
        )
        # Top cards
        self._card_total.set(_fmt_int(s.get("total_games")))
        self._card_steps.set(_fmt_int(s.get("total_training_steps")))
        state = "paused" if s.get("training_paused") else "running"
        if s.get("auto_mode"):
            state += " (auto)"
        self._card_state.set(state.upper(),
                             sub=s.get("performance_mode", "balanced"))
        # Mode panel
        self._mode_panel.set_mode(s.get("performance_mode", "balanced"))
        self._mode_panel.set_auto(bool(s.get("auto_mode")))
        self._mode_panel.set_paused(bool(s.get("training_paused")))
        # Variants
        self._variants_panel.update(s)
        # Resources
        self._resources.update(s.get("resources", {}))
        # Checkpoints
        self._ck_table.set_items(ckpts)
        # Footer
        self._footer.config(text=f"last update: {time.strftime('%H:%M:%S')}",
                            fg=Theme.MUTED)

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            self._running = False
            self._poll_thread.join(timeout=2.0)


# =============================================================================
# Helpers
# =============================================================================


def _fmt_int(n: Any) -> str:
    if n is None:
        return "--"
    try:
        return f"{int(round(float(n))):,}"
    except (ValueError, TypeError):
        return str(n)


def _format_age(mtime: float) -> str:
    if not mtime:
        return "--"
    age = max(0.0, time.time() - mtime)
    if age < 60:
        return f"{int(age)}s"
    if age < 3600:
        return f"{int(age / 60)}m"
    if age < 86400:
        return f"{int(age / 3600)}h"
    return f"{int(age / 86400)}d"


def _variant_color(v: str) -> str:
    return {
        "baseline": Theme.ACCENT_2,
        "attack": Theme.BOOST,
        "est": Theme.WARN,
    }.get(v, Theme.ACCENT)


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
