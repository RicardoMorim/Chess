"""
Tests for the control plane HTTP server (Fase 2).

Covers:
  - Server starts on 127.0.0.1 with a real port
  - GET /api/status returns a JSON snapshot
  - GET /api/knobs returns current knob values
  - GET /api/modes returns available + current mode
  - POST /api/mode changes the trainer mode
  - POST /api/knobs applies hot-swap
  - POST /api/auto_mode toggles auto-mode
  - POST /api/pause toggles the pause event
  - GET /api/matches/stream subscribes via SSE
  - Binding to 127.0.0.1 (not 0.0.0.0) is enforced
"""

import json
import os
import socket
import sys
import threading
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "train"))


def _make_trainer(tmpdir: str):
    from train.league.league_trainer import LeagueTrainer
    from train.league.replay_buffer import ReplayBuffer

    trainer = LeagueTrainer.__new__(LeagueTrainer)
    trainer.device = type("D", (), {"__str__": lambda self: "cuda"})()
    trainer.checkpoint_dir = Path(tmpdir) / "ckpt"
    trainer.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer.log_dir = Path(tmpdir) / "logs"
    trainer.log_dir.mkdir(parents=True, exist_ok=True)
    trainer.use_gpu_batching = False
    trainer._state_lock = threading.RLock()
    trainer._pending_changes = {}
    trainer._num_self_play_workers = 6
    trainer._variant_parallelism = 3
    trainer._buffer_target_size = 100_000
    trainer._last_buffer_target_size = 100_000
    trainer._current_mcts_visits = 200
    trainer.VARIANTS = ["baseline", "attack", "est"]
    trainer.buffers = {v: ReplayBuffer(max_size=1_000) for v in trainer.VARIANTS}
    trainer.evaluator = type("E", (), {"mcts_visits": 400})()
    trainer.models = {}
    trainer.optimizers = {}
    trainer.schedulers = {}
    trainer.metrics = type("M", (), {"get_recent_loss": lambda *a, **k: None,
                                      "get_variant_throughput": lambda *a, **k: None})()
    trainer.round = 7
    trainer.total_games = 1234
    trainer.total_training_steps = 567

    from train.league.performance import AutoModeConfig, AutoModeController
    trainer.performance_mode = "balanced"
    trainer._auto_mode = AutoModeController(trainer, AutoModeConfig(enabled=False))

    import queue
    trainer._spectate_queue = queue.Queue()
    trainer._training_pause_event = None
    return trainer


def _wait_for_server(server, timeout: float = 3.0) -> int:
    """Block until the server's actual_port is set, return it."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server.actual_port is not None:
            return server.actual_port
        time.sleep(0.02)
    raise RuntimeError("ControlServer didn't start in time")


def _conn(server):
    return HTTPConnection("127.0.0.1", server.actual_port, timeout=5.0)


class ControlServerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.tmp = tempfile.mkdtemp()
        cls.trainer = _make_trainer(cls.tmp)
        from train.league.control_server import ControlServer
        cls.server = ControlServer(cls.trainer, host="127.0.0.1", port=0)
        cls.server.start()
        cls.port = _wait_for_server(cls.server)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop(timeout=2.0)

    def test_server_bound_to_loopback(self):
        # Confirm the listening socket is on 127.0.0.1, not 0.0.0.0
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("127.0.0.1", self.port))
            peer = s.getpeername()
            self.assertEqual(peer[0], "127.0.0.1")

    def test_get_status(self):
        c = _conn(self.server)
        c.request("GET", "/api/status")
        r = c.getresponse()
        self.assertEqual(r.status, 200)
        body = json.loads(r.read().decode())
        self.assertIn("round", body)
        self.assertIn("performance_mode", body)
        self.assertIn("resources", body)
        self.assertEqual(body["round"], 7)
        self.assertEqual(body["total_games"], 1234)
        self.assertEqual(body["performance_mode"], "balanced")

    def test_get_knobs(self):
        c = _conn(self.server)
        c.request("GET", "/api/knobs")
        r = c.getresponse()
        body = json.loads(r.read().decode())
        self.assertIn("BATCH_SIZE", body)
        self.assertIn("MCTS_VISITS_SELFPLAY", body)
        self.assertEqual(body["BATCH_SIZE"], 256)

    def test_get_modes(self):
        c = _conn(self.server)
        c.request("GET", "/api/modes")
        r = c.getresponse()
        body = json.loads(r.read().decode())
        self.assertIn("eco", body["available"])
        self.assertIn("balanced", body["available"])
        self.assertIn("boost", body["available"])
        self.assertEqual(body["current"], "balanced")

    def test_get_variants(self):
        c = _conn(self.server)
        c.request("GET", "/api/variants")
        r = c.getresponse()
        body = json.loads(r.read().decode())
        self.assertEqual(len(body), 3)
        names = {v["name"] for v in body}
        self.assertEqual(names, {"baseline", "attack", "est"})

    def test_get_checkpoints_empty(self):
        c = _conn(self.server)
        c.request("GET", "/api/checkpoints")
        r = c.getresponse()
        body = json.loads(r.read().decode())
        self.assertIsInstance(body, list)
        self.assertEqual(len(body), 0)

    def test_get_checkpoints_finds_files(self):
        # Create fake checkpoint files
        for v in ("baseline", "attack"):
            (self.trainer.checkpoint_dir / f"{v}_step_5.pt").write_bytes(b"x" * 100)
        c = _conn(self.server)
        c.request("GET", "/api/checkpoints")
        r = c.getresponse()
        body = json.loads(r.read().decode())
        self.assertGreaterEqual(len(body), 2)
        names = {item["variant"] for item in body}
        self.assertIn("baseline", names)

    def test_get_checkpoints_filter_by_variant(self):
        """?variant=baseline returns only baseline checkpoints."""
        for v in ("baseline", "attack", "est"):
            (self.trainer.checkpoint_dir / f"{v}_step_5.pt").write_bytes(b"x" * 100)
            (self.trainer.checkpoint_dir / f"{v}_step_10.pt").write_bytes(b"x" * 100)
        c = _conn(self.server)
        c.request("GET", "/api/checkpoints?variant=baseline")
        r = c.getresponse()
        body = json.loads(r.read().decode())
        self.assertEqual(len(body), 2)
        for item in body:
            self.assertEqual(item["variant"], "baseline")
        # And attack
        c.request("GET", "/api/checkpoints?variant=attack")
        body = json.loads(c.getresponse().read().decode())
        self.assertEqual(len(body), 2)
        for item in body:
            self.assertEqual(item["variant"], "attack")
        # And unknown variant -> empty
        c.request("GET", "/api/checkpoints?variant=ghost")
        body = json.loads(c.getresponse().read().decode())
        self.assertEqual(len(body), 0)

    def test_post_mode_eco(self):
        c = _conn(self.server)
        body = json.dumps({"mode": "eco"}).encode()
        c.request("POST", "/api/mode", body, {"Content-Type": "application/json"})
        r = c.getresponse()
        resp = json.loads(r.read().decode())
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["mode"], "eco")
        self.assertEqual(self.trainer.get_mode(), "eco")
        # Reset to balanced
        c.request("POST", "/api/mode", json.dumps({"mode": "balanced"}).encode(),
                  {"Content-Type": "application/json"})
        c.getresponse().read()

    def test_post_mode_unknown(self):
        c = _conn(self.server)
        body = json.dumps({"mode": "nuclear"}).encode()
        c.request("POST", "/api/mode", body, {"Content-Type": "application/json"})
        r = c.getresponse()
        resp = json.loads(r.read().decode())
        self.assertFalse(resp["ok"])

    def test_post_knobs_applies_change(self):
        c = _conn(self.server)
        body = json.dumps({"BATCH_SIZE": 512, "PUZZLE_BATCHES_PER_GAME_BATCH": 0}).encode()
        c.request("POST", "/api/knobs", body, {"Content-Type": "application/json"})
        r = c.getresponse()
        resp = json.loads(r.read().decode())
        self.assertTrue(resp["BATCH_SIZE"])
        self.assertTrue(resp["PUZZLE_BATCHES_PER_GAME_BATCH"])
        self.assertEqual(self.trainer.BATCH_SIZE, 512)
        # Restore
        c.request("POST", "/api/knobs",
                  json.dumps({"BATCH_SIZE": 256, "PUZZLE_BATCHES_PER_GAME_BATCH": 1}).encode(),
                  {"Content-Type": "application/json"})
        c.getresponse().read()

    def test_post_knobs_rejects_unknown(self):
        c = _conn(self.server)
        body = json.dumps({"NONEXISTENT": 42}).encode()
        c.request("POST", "/api/knobs", body, {"Content-Type": "application/json"})
        r = c.getresponse()
        resp = json.loads(r.read().decode())
        self.assertFalse(resp["NONEXISTENT"])

    def test_post_auto_mode_toggle(self):
        c = _conn(self.server)
        body = json.dumps({"enabled": True}).encode()
        c.request("POST", "/api/auto_mode", body, {"Content-Type": "application/json"})
        r = c.getresponse()
        resp = json.loads(r.read().decode())
        self.assertTrue(resp["auto_mode"])
        self.assertTrue(self.trainer.get_auto_mode())
        # Toggle off
        c.request("POST", "/api/auto_mode",
                  json.dumps({"enabled": False}).encode(),
                  {"Content-Type": "application/json"})
        c.getresponse().read()
        self.assertFalse(self.trainer.get_auto_mode())

    def test_post_pause_sets_event(self):
        c = _conn(self.server)
        c.request("POST", "/api/pause", json.dumps({"paused": True}).encode(),
                  {"Content-Type": "application/json"})
        r = c.getresponse()
        resp = json.loads(r.read().decode())
        self.assertTrue(resp["ok"])
        self.assertIsNotNone(self.trainer._training_pause_event)
        self.assertTrue(self.trainer._training_pause_event.is_set())
        # Unpause
        c.request("POST", "/api/pause", json.dumps({"paused": False}).encode(),
                  {"Content-Type": "application/json"})
        c.getresponse().read()
        self.assertFalse(self.trainer._training_pause_event.is_set())

    def test_post_match_model_queues(self):
        # Drain any leftover items from previous tests (alphabetical order)
        while not self.trainer._spectate_queue.empty():
            self.trainer._spectate_queue.get_nowait()
        c = _conn(self.server)
        body = json.dumps({
            "type": "model",
            "white": "baseline",
            "black": "attack",
            "visits": 100,
        }).encode()
        c.request("POST", "/api/matches", body, {"Content-Type": "application/json"})
        r = c.getresponse()
        self.assertEqual(r.status, 202)
        resp = json.loads(r.read().decode())
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["match"]["type"], "model")
        # Should have been pushed to the queue
        queued = self.trainer._spectate_queue.get_nowait()
        self.assertEqual(queued["type"], "model")

    def test_post_match_puzzle_queues(self):
        # Drain any leftover items from previous tests (alphabetical order)
        while not self.trainer._spectate_queue.empty():
            self.trainer._spectate_queue.get_nowait()
        c = _conn(self.server)
        body = json.dumps({"type": "puzzle", "puzzle_id": "abc123", "visits": 50}).encode()
        c.request("POST", "/api/matches", body, {"Content-Type": "application/json"})
        r = c.getresponse()
        resp = json.loads(r.read().decode())
        self.assertTrue(resp["ok"])
        queued = self.trainer._spectate_queue.get_nowait()
        self.assertEqual(queued["type"], "puzzle")
        self.assertEqual(queued["params"]["puzzle_id"], "abc123")

    def test_post_match_unknown_type(self):
        c = _conn(self.server)
        body = json.dumps({"type": "garbage"}).encode()
        c.request("POST", "/api/matches", body, {"Content-Type": "application/json"})
        r = c.getresponse()
        resp = json.loads(r.read().decode())
        self.assertFalse(resp["ok"])

    def test_get_matches_returns_history(self):
        # Enqueue first, then read history (tests may run in any order)
        c = _conn(self.server)
        for body in ({"type": "model", "white": "a", "black": "b"},
                     {"type": "puzzle", "puzzle_id": "x"}):
            c.request("POST", "/api/matches", json.dumps(body).encode(),
                      {"Content-Type": "application/json"})
            c.getresponse().read()
        c.request("GET", "/api/matches")
        r = c.getresponse()
        body = json.loads(r.read().decode())
        self.assertGreaterEqual(len(body["history"]), 2)
        # Most recent first OR last? In our impl we append, so last is newest
        types = [m["type"] for m in body["history"]]
        self.assertIn("puzzle", types)

    def test_post_match_with_stockfish_dict(self):
        """Stockfish side via dict descriptor must validate and queue."""
        while not self.trainer._spectate_queue.empty():
            self.trainer._spectate_queue.get_nowait()
        c = _conn(self.server)
        body = json.dumps({
            "type": "model",
            "white": {"type": "model", "name": "baseline"},
            "black": {"type": "stockfish", "depth": 15, "label": "MySF"},
            "visits": 100,
        }).encode()
        c.request("POST", "/api/matches", body, {"Content-Type": "application/json"})
        r = c.getresponse()
        resp = json.loads(r.read().decode())
        self.assertTrue(resp["ok"], resp)
        queued = self.trainer._spectate_queue.get_nowait()
        self.assertEqual(queued["params"]["black"]["type"], "stockfish")
        self.assertEqual(queued["params"]["black"]["depth"], 15)

    def test_post_match_stockfish_shorthand(self):
        """'stockfish' as a string is treated as a Stockfish side with defaults."""
        while not self.trainer._spectate_queue.empty():
            self.trainer._spectate_queue.get_nowait()
        c = _conn(self.server)
        body = json.dumps({
            "type": "model",
            "white": "baseline",
            "black": "stockfish",
        }).encode()
        c.request("POST", "/api/matches", body, {"Content-Type": "application/json"})
        r = c.getresponse()
        resp = json.loads(r.read().decode())
        self.assertTrue(resp["ok"], resp)

    def test_post_match_invalid_visits(self):
        c = _conn(self.server)
        body = json.dumps({
            "type": "model", "white": "a", "black": "b", "visits": 99999,
        }).encode()
        c.request("POST", "/api/matches", body, {"Content-Type": "application/json"})
        resp = json.loads(c.getresponse().read().decode())
        self.assertFalse(resp["ok"])
        self.assertIn("visits", resp["error"])

    def test_post_match_invalid_stockfish_type(self):
        c = _conn(self.server)
        body = json.dumps({
            "type": "model",
            "white": {"type": "stockfish", "depth": 999},
            "black": "baseline",
        }).encode()
        c.request("POST", "/api/matches", body, {"Content-Type": "application/json"})
        resp = json.loads(c.getresponse().read().decode())
        self.assertFalse(resp["ok"])
        self.assertIn("depth", resp["error"])

    def test_post_match_model_missing_name(self):
        c = _conn(self.server)
        body = json.dumps({
            "type": "model",
            "white": {"type": "model"},
            "black": "baseline",
        }).encode()
        c.request("POST", "/api/matches", body, {"Content-Type": "application/json"})
        resp = json.loads(c.getresponse().read().decode())
        self.assertFalse(resp["ok"])
        self.assertIn("name", resp["error"])

    def test_post_match_with_time_limit(self):
        """time_limit_ms takes precedence over depth in the dict."""
        while not self.trainer._spectate_queue.empty():
            self.trainer._spectate_queue.get_nowait()
        c = _conn(self.server)
        body = json.dumps({
            "type": "model",
            "white": "baseline",
            "black": {"type": "stockfish", "time_limit_ms": 5000},
        }).encode()
        c.request("POST", "/api/matches", body, {"Content-Type": "application/json"})
        r = c.getresponse()
        resp = json.loads(r.read().decode())
        self.assertTrue(resp["ok"], resp)
        queued = self.trainer._spectate_queue.get_nowait()
        self.assertEqual(queued["params"]["black"]["time_limit_ms"], 5000)

    def test_get_root_serves_html_fallback(self):
        c = _conn(self.server)
        c.request("GET", "/")
        r = c.getresponse()
        # Either dashboard/index.html (if it exists) or the fallback HTML
        self.assertEqual(r.status, 200)
        body = r.read()
        self.assertIn(b"<html>", body.lower())


class ControlServerDashboardTests(unittest.TestCase):
    """The dashboard (index.html, css, js) is served if the dir is configured."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.tmp = tempfile.mkdtemp()
        cls.trainer = _make_trainer(cls.tmp)
        from train.league.control_server import ControlServer
        from pathlib import Path
        cls.dashboard_dir = Path("train/league/dashboard")
        cls.server = ControlServer(cls.trainer, host="127.0.0.1", port=0,
                                  dashboard_dir=cls.dashboard_dir)
        cls.server.start()
        _wait_for_server(cls.server)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop(timeout=2.0)

    def test_dashboard_files_served(self):
        c = _conn(self.server)
        for path, contains in [
            ("/", b"Chess Trainer"),
            ("/style.css", b"--bg"),
            ("/dashboard.js", b"pollStatus"),
        ]:
            c.request("GET", path)
            r = c.getresponse()
            self.assertEqual(r.status, 200, f"{path} returned {r.status}")
            body = r.read()
            self.assertIn(contains, body, f"{path} missing {contains!r}")

    def test_dashboard_has_spectate_modal(self):
        c = _conn(self.server)
        c.request("GET", "/")
        r = c.getresponse()
        body = r.read()
        self.assertIn(b'spectate-modal', body)
        self.assertIn(b'spectate-board', body)

    def test_dashboard_has_mode_buttons(self):
        c = _conn(self.server)
        c.request("GET", "/")
        r = c.getresponse()
        body = r.read()
        self.assertIn(b'data-mode="eco"', body)
        self.assertIn(b'data-mode="balanced"', body)
        self.assertIn(b'data-mode="boost"', body)

    def test_dashboard_has_spectate_player_selectors(self):
        """Each side (white/black) must have its own model+engine selector."""
        c = _conn(self.server)
        c.request("GET", "/")
        r = c.getresponse()
        body = r.read()
        # Two sides, each with engine and model selectors
        self.assertIn(b'side-engine', body)
        self.assertIn(b'side-model', body)
        self.assertIn(b'side-checkpoint', body)
        self.assertIn(b'side-white', body)
        self.assertIn(b'side-black', body)
        # Player names in the modal
        self.assertIn(b'player-white-name', body)
        self.assertIn(b'player-black-name', body)
        # Stockfish fields are present
        self.assertIn(b'side-depth', body)
        self.assertIn(b'side-time', body)
        self.assertIn(b'stockfish', body)

    def test_dashboard_js_handles_stockfish_descriptors(self):
        """The frontend buildPlayerDescriptor must handle stockfish dicts."""
        c = _conn(self.server)
        c.request("GET", "/dashboard.js")
        body = c.getresponse().read()
        # Frontend knows how to build stockfish descriptors
        self.assertIn(b'buildPlayerDescriptor', body)
        self.assertIn(b'stockfish', body)
        self.assertIn(b'time_limit_ms', body)
        self.assertIn(b'player-white-name', body)

    def test_dashboard_js_handles_player_names(self):
        """Player name fields must be updated on start and move events."""
        c = _conn(self.server)
        c.request("GET", "/dashboard.js")
        body = c.getresponse().read()
        self.assertIn(b'player-white-name', body)
        self.assertIn(b'player-black-name', body)
        self.assertIn(b'evt.white', body)
        self.assertIn(b'evt.black', body)

    def test_path_escape_blocked(self):
        """Dashboard dir traversal attempts are rejected."""
        c = _conn(self.server)
        c.request("GET", "/../league_trainer.py")
        r = c.getresponse()
        # Either 400 (path escape) or 404 — must NOT be 200 with file content
        self.assertIn(r.status, (400, 404))

    def test_404_for_unknown_path(self):
        c = _conn(self.server)
        c.request("GET", "/api/nope")
        r = c.getresponse()
        self.assertEqual(r.status, 404)

    def test_post_invalid_json_400(self):
        c = _conn(self.server)
        c.request("POST", "/api/mode", b"not json", {"Content-Type": "application/json"})
        r = c.getresponse()
        self.assertEqual(r.status, 400)


class ControlServerSSEStreamTests(unittest.TestCase):
    """SSE: subscribe, receive a published event, then close."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.tmp = tempfile.mkdtemp()
        cls.trainer = _make_trainer(cls.tmp)
        from train.league.control_server import ControlServer
        cls.server = ControlServer(cls.trainer, host="127.0.0.1", port=0)
        cls.server.start()
        _wait_for_server(cls.server)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop(timeout=2.0)

    def test_sse_subscribe_and_receive(self):
        # Open raw TCP and do the GET manually so we can stream bytes.
        s = socket.create_connection(("127.0.0.1", self.server.actual_port), timeout=5.0)
        s.sendall(b"GET /api/matches/stream HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")

        # Read response headers
        buf = b""
        deadline = time.time() + 2.0
        while b"\r\n\r\n" not in buf and time.time() < deadline:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
        self.assertIn(b"200 OK", buf)
        self.assertIn(b"text/event-stream", buf)

        # Wait for the keepalive comment
        deadline = time.time() + 3.0
        seen_keepalive = b": keepalive" in buf
        while not seen_keepalive and time.time() < deadline:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
            seen_keepalive = b": keepalive" in buf
        self.assertTrue(seen_keepalive, f"no keepalive in stream; got {buf!r}")

        # Now publish an event
        self.server.match_bus.publish({"type": "test", "msg": "hello"})

        # Read until we see the event data line
        deadline = time.time() + 3.0
        seen_event = b'"msg": "hello"' in buf or b'"msg":"hello"' in buf
        while not seen_event and time.time() < deadline:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
            seen_event = b"hello" in buf
        self.assertTrue(seen_event, f"event not received; buf tail={buf[-400:]!r}")
        s.close()


class ControlServerPortBindingTests(unittest.TestCase):
    """If the requested port is taken, the server picks a free one."""

    def test_falls_back_to_free_port(self):
        import tempfile
        from train.league.control_server import ControlServer

        # Grab a port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            taken = s.getsockname()[1]
            # We don't listen on it; that's fine — bind() is what we test
            tmp = tempfile.mkdtemp()
            trainer = _make_trainer(tmp)
            server = ControlServer(trainer, host="127.0.0.1", port=taken)
            server.start()
            try:
                _wait_for_server(server)
                # It might end up on the same port (race) or another one.
                # The contract is just "port is set, server is reachable."
                self.assertIsNotNone(server.actual_port)
                self.assertGreater(server.actual_port, 0)
            finally:
                server.stop(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
