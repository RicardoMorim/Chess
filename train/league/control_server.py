"""
Control plane HTTP server for the LeagueTrainer (Fase 2).

A small stdlib HTTP server that runs as a daemon thread inside the trainer
process, exposing REST endpoints to inspect status, change modes, and queue
spectate matches. The server binds to 127.0.0.1 only (no LAN exposure by
default — security is enforced at the bind level, not the application level).

Why stdlib?
  - Zero new dependencies (Flask/FastAPI are great but we don't need them)
  - SSE is trivial with raw bytes on the wire
  - ~250 LOC for everything we need
  - Easy to audit (no hidden middleware)

Endpoints (JSON unless noted):
  GET  /api/status              -> snapshot of trainer state + resources
  GET  /api/checkpoints         -> list of saved checkpoints
  GET  /api/variants            -> list of model variants
  GET  /api/modes               -> available performance modes
  GET  /api/knobs               -> current values of hot-swappable knobs
  POST /api/mode                -> {"mode": "eco"|"balanced"|"boost"}
  POST /api/knobs               -> {"BATCH_SIZE": 1024, ...}
  POST /api/auto_mode           -> {"enabled": bool}
  POST /api/pause               -> {"paused": bool}
  POST /api/matches             -> {"type": "model"|"puzzle", ...}
  GET  /api/matches             -> list of recent matches
  GET  /api/matches/stream      -> SSE: live match events
  GET  /                        -> dashboard/index.html (served if available)

SSE is intentionally minimal: each event is a single line of JSON followed
by a blank line (the SSE wire format). Clients consume via EventSource.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that silently coerces non-serializable objects to strings.

    This is a defense-in-depth measure: the snapshot helpers should already
    produce JSON-safe values, but a stray MagicMock or torch.Tensor should
    not crash the HTTP server. It gets stringified instead.
    """

    def default(self, o):
        try:
            return str(o)
        except Exception:
            return f"<unserializable {type(o).__name__}>"

if TYPE_CHECKING:
    from .league_trainer import LeagueTrainer

logger = logging.getLogger(__name__)


# =============================================================================
# Match event bus (for SSE)
# =============================================================================


class MatchEventBus:
    """Thread-safe pub/sub for spectate match events.

    Multiple SSE clients can subscribe; each gets its own queue. Producers
    (SpectateSession callbacks) push events that get fanned out to all
    subscribers. Slow consumers (full queue) are dropped (with a warning)
    so a stuck client doesn't block training.
    """

    def __init__(self, max_queue_size: int = 256):
        self._subscribers: List[queue.Queue] = []
        self._lock = threading.Lock()
        self._max_queue_size = max_queue_size

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=self._max_queue_size)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def publish(self, event: Dict[str, Any]) -> None:
        """Fan out an event to all subscribers. Drop on backpressure."""
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                logger.debug("SSE subscriber queue full, dropping event")


# =============================================================================
# Snapshot helpers (pure functions over trainer state)
# =============================================================================


def _resource_snapshot() -> Dict[str, Any]:
    """CPU / RAM / VRAM usage for the dashboard. All values are best-effort."""
    snap: Dict[str, Any] = {
        "vram_used_mb": None,
        "vram_total_mb": None,
        "vram_pct": None,
        "cpu_pct": None,
        "ram_pct": None,
    }
    try:
        import psutil
        snap["cpu_pct"] = float(psutil.cpu_percent(interval=None))
        snap["ram_pct"] = float(psutil.virtual_memory().percent)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            snap["vram_used_mb"] = int(torch.cuda.memory_allocated() / 1024 / 1024)
            snap["vram_total_mb"] = int(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
            if snap["vram_total_mb"]:
                snap["vram_pct"] = round(100.0 * snap["vram_used_mb"] / snap["vram_total_mb"], 1)
    except Exception:
        pass
    return snap


def _safe_float(v: Any) -> Optional[float]:
    """Coerce a value to a JSON-safe float, returning None for anything weird."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _trainer_snapshot(trainer: "LeagueTrainer") -> Dict[str, Any]:
    """Status snapshot used by /api/status and the dashboard."""
    snap: Dict[str, Any] = {
        "round": int(getattr(trainer, "round", 0) or 0),
        "total_games": int(getattr(trainer, "total_games", 0) or 0),
        "total_training_steps": int(getattr(trainer, "total_training_steps", 0) or 0),
        "performance_mode": str(getattr(trainer, "performance_mode", "balanced")),
        "auto_mode": bool(getattr(trainer, "_auto_mode", None) and trainer._auto_mode.config.enabled),
        "training_paused": bool(getattr(trainer, "_training_pause_event", None)
                                and trainer._training_pause_event.is_set()),
        "variants": {},
        "losses": {},
        "throughput_gpm": {},
        "buffers": {},
        "resources": _resource_snapshot(),
    }

    metrics = getattr(trainer, "metrics", None)
    for variant in getattr(trainer, "VARIANTS", []):
        # Loss
        if metrics is not None and hasattr(metrics, "get_recent_loss"):
            snap["losses"][variant] = _safe_float(metrics.get_recent_loss(variant))
        # Throughput
        if metrics is not None and hasattr(metrics, "get_variant_throughput"):
            snap["throughput_gpm"][variant] = _safe_float(metrics.get_variant_throughput(variant))
        # Buffer fill
        buf = getattr(trainer, "buffers", {}).get(variant)
        if buf is not None and hasattr(buf, "get_stats"):
            stats = buf.get_stats()
            size = int(stats.get("size", 0) or 0)
            cap = int(stats.get("capacity", 0) or 0)
            snap["buffers"][variant] = {
                "size": size,
                "capacity": cap,
                "fill_pct": round(100.0 * size / max(1, cap), 1),
            }
    return snap


def _checkpoints_snapshot(trainer: "LeagueTrainer") -> List[Dict[str, Any]]:
    """List of saved checkpoints with metadata."""
    out: List[Dict[str, Any]] = []
    ckpt_dir: Optional[Path] = getattr(trainer, "checkpoint_dir", None)
    if ckpt_dir is None or not ckpt_dir.exists():
        return out
    for p in sorted(ckpt_dir.glob("*_step_*.pt")):
        name = p.stem  # e.g. baseline_step_35
        try:
            variant, _, step_str = name.partition("_step_")
            step = int(step_str)
        except ValueError:
            continue
        out.append({
            "variant": variant,
            "step": step,
            "path": str(p),
            "mtime": p.stat().st_mtime,
            "size_mb": round(p.stat().st_size / 1024 / 1024, 1),
        })
    return out


def _variants_snapshot(trainer: "LeagueTrainer") -> List[Dict[str, Any]]:
    """Per-variant metadata (channels, param count)."""
    out: List[Dict[str, Any]] = []
    for variant in getattr(trainer, "VARIANTS", []):
        model = getattr(trainer, "models", {}).get(variant)
        info: Dict[str, Any] = {"name": variant}
        if model is not None:
            try:
                conv = model.conv_in
                info["input_channels"] = int(conv.weight.shape[1])
                info["param_count"] = sum(p.numel() for p in model.parameters())
            except AttributeError:
                pass
        out.append(info)
    return out


# =============================================================================
# HTTP server
# =============================================================================


class ControlServer(threading.Thread):
    """Threaded HTTP server exposing trainer state.

    Stops cleanly when ``stop()`` is called (typically from LeagueTrainer
    shutdown, or test tearDown). Bound to 127.0.0.1 by default for safety.
    """

    DEFAULT_PORT = 7860
    DEFAULT_HOST = "127.0.0.1"

    def __init__(
        self,
        trainer: "LeagueTrainer",
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        dashboard_dir: Optional[Path] = None,
        max_match_history: int = 50,
    ):
        super().__init__(name="ControlServer", daemon=True)
        self.trainer = trainer
        self.host = host
        self.port = port
        self.dashboard_dir = Path(dashboard_dir) if dashboard_dir else None
        self.match_bus = MatchEventBus()
        self._stop_event = threading.Event()
        self._server: Optional[ThreadingHTTPServer] = None
        self._lock = threading.Lock()
        self._match_history: List[Dict[str, Any]] = []
        self._max_match_history = max_match_history
        self.actual_port: Optional[int] = None  # set after start

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        with self._lock:
            server = self._server
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
            try:
                server.server_close()
            except Exception:
                pass
        self.join(timeout=timeout)
        logger.info(f"ControlServer stopped (was on {self.host}:{self.actual_port})")

    def _find_free_port(self) -> int:
        """Find a free port if the requested one is taken."""
        if self.port != 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((self.host, self.port))
                    return self.port
                except OSError:
                    pass
        # Fall back to ephemeral
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, 0))
            return s.getsockname()[1]

    def run(self) -> None:
        try:
            port = self._find_free_port()
            handler_cls = self._make_handler()
            self._server = ThreadingHTTPServer((self.host, port), handler_cls)
            self.actual_port = port
            logger.info(f"ControlServer listening on http://{self.host}:{port}")
            self._server.serve_forever()
        except Exception as e:
            logger.error(f"ControlServer crashed: {e}", exc_info=True)

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            # Suppress default access log; we have our own logger.
            def log_message(self, format, *args):
                return

            def _set_json_headers(self, code: int = 200) -> None:
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

            def _send_json(self, payload: Any, code: int = 200) -> None:
                body = json.dumps(payload, cls=_SafeEncoder).encode("utf-8")
                self._set_json_headers(code)
                self.wfile.write(body)

            def _read_json(self) -> Optional[Dict[str, Any]]:
                try:
                    length = int(self.headers.get("Content-Length", "0") or "0")
                    if length <= 0:
                        return {}
                    raw = self.rfile.read(length)
                    return json.loads(raw.decode("utf-8"))
                except (ValueError, json.JSONDecodeError):
                    return None

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/api/status":
                    return self._send_json(_trainer_snapshot(server.trainer))
                if path == "/api/checkpoints":
                    return self._send_json(_checkpoints_snapshot(server.trainer))
                if path == "/api/variants":
                    return self._send_json(_variants_snapshot(server.trainer))
                if path == "/api/modes":
                    return self._send_json({
                        "available": server.trainer.list_available_modes(),
                        "current": server.trainer.get_mode(),
                    })
                if path == "/api/knobs":
                    return self._send_json(server.trainer.list_hot_knobs())
                if path == "/api/matches":
                    return self._send_json({
                        "history": list(server._match_history),
                    })
                if path.startswith("/api/matches/stream"):
                    return self._handle_sse()
                if path == "/" or path == "/index.html":
                    return self._serve_dashboard()
                if path.startswith("/dashboard/") or path.endswith((".js", ".css")):
                    return self._serve_dashboard()
                self._send_json({"error": "not found", "path": path}, code=404)

            def do_POST(self):
                path = self.path.split("?", 1)[0]
                payload = self._read_json()
                if payload is None:
                    return self._send_json({"error": "invalid JSON"}, code=400)
                if path == "/api/mode":
                    mode = payload.get("mode")
                    ok = bool(mode) and server.trainer.set_mode(mode)
                    return self._send_json({"ok": ok, "mode": server.trainer.get_mode()})
                if path == "/api/knobs":
                    return self._send_json(server.trainer.set_knobs(payload))
                if path == "/api/auto_mode":
                    enabled = bool(payload.get("enabled"))
                    server.trainer.set_auto_mode(enabled)
                    return self._send_json({"ok": True, "auto_mode": server.trainer.get_auto_mode()})
                if path == "/api/pause":
                    paused = bool(payload.get("paused"))
                    server._set_paused(paused)
                    return self._send_json({"ok": True, "paused": paused})
                if path == "/api/matches":
                    match = server._enqueue_match(payload)
                    return self._send_json(match, code=202 if match.get("ok") else 400)
                self._send_json({"error": "not found", "path": path}, code=404)

            def _handle_sse(self):
                """Stream match events via Server-Sent Events."""
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                sub = server.match_bus.subscribe()
                try:
                    # Send a comment line so the client knows the stream is live
                    self.wfile.write(b": connected\n\n")
                    self.wfile.flush()
                    while not server._stop_event.is_set():
                        try:
                            evt = sub.get(timeout=1.0)
                        except queue.Empty:
                            # Send keepalive comment every second
                            try:
                                self.wfile.write(b": keepalive\n\n")
                                self.wfile.flush()
                            except Exception:
                                break
                            continue
                        try:
                            data = json.dumps(evt).encode("utf-8")
                            self.wfile.write(b"data: " + data + b"\n\n")
                            self.wfile.flush()
                        except Exception:
                            break
                finally:
                    server.match_bus.unsubscribe(sub)

            def _serve_dashboard(self):
                # Try to serve dashboard/index.html from the configured dir
                if server.dashboard_dir is None or not server.dashboard_dir.exists():
                    body = (
                        b"<html><body><h1>Chess Trainer Control Server</h1>"
                        b"<p>Dashboard assets not built. API available at /api/*</p>"
                        b"<ul>"
                        b"<li>GET /api/status</li>"
                        b"<li>POST /api/mode</li>"
                        b"<li>POST /api/knobs</li>"
                        b"</ul></body></html>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                # Resolve relative file
                rel = self.path.lstrip("/").split("?", 1)[0]
                if rel in ("", "index.html"):
                    target = server.dashboard_dir / "index.html"
                else:
                    target = server.dashboard_dir / rel
                # Defense-in-depth: don't escape dashboard_dir
                try:
                    target = target.resolve()
                    if server.dashboard_dir.resolve() not in target.parents and target != server.dashboard_dir.resolve():
                        raise ValueError("path escape")
                except Exception:
                    self._send_json({"error": "bad path"}, code=400)
                    return
                if not target.is_file():
                    self._send_json({"error": "not found", "path": str(target)}, code=404)
                    return
                # Basic content-type sniffing
                ext = target.suffix.lower()
                ctype = {
                    ".html": "text/html; charset=utf-8",
                    ".js": "application/javascript; charset=utf-8",
                    ".css": "text/css; charset=utf-8",
                    ".json": "application/json; charset=utf-8",
                    ".svg": "image/svg+xml",
                }.get(ext, "application/octet-stream")
                try:
                    body = target.read_bytes()
                except OSError:
                    self._send_json({"error": "read failed"}, code=500)
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        return Handler

    # ---- trainer-side helpers ------------------------------------------------

    def _set_paused(self, paused: bool) -> None:
        """Toggle the training pause event. Lazily creates the Event."""
        trainer = self.trainer
        if not hasattr(trainer, "_training_pause_event") or trainer._training_pause_event is None:
            trainer._training_pause_event = threading.Event()
        if paused:
            trainer._training_pause_event.set()
        else:
            trainer._training_pause_event.clear()

    def _enqueue_match(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Queue a spectate match. Returns the queued match descriptor."""
        mtype = payload.get("type", "model")
        if mtype not in ("model", "puzzle"):
            return {"ok": False, "error": f"unknown match type '{mtype}'"}
        match = {
            "id": int(time.time() * 1000),
            "type": mtype,
            "created_at": time.time(),
            "status": "queued",
            "params": payload,
        }
        with self._lock:
            self._match_history.append(match)
            # Trim history
            if len(self._match_history) > self._max_match_history:
                self._match_history = self._match_history[-self._max_match_history:]
        # Hand off to trainer's spectate queue if it has one
        queue_obj = getattr(self.trainer, "_spectate_queue", None)
        if queue_obj is not None:
            try:
                queue_obj.put_nowait(match)
            except Exception as e:
                match["status"] = "queue_error"
                match["error"] = str(e)
        return {"ok": True, "match": match}
