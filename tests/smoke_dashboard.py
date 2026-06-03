"""Smoke test for the dashboard served by the control server."""
import sys
import tempfile
import json
import time
import http.client
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, "train")
from league.control_server import ControlServer
from league.replay_buffer import ReplayBuffer


def main():
    tmp = tempfile.mkdtemp()
    ck = Path(tmp) / "ckpt"
    ck.mkdir()
    ld = Path(tmp) / "logs"
    ld.mkdir()

    t = MagicMock()
    t.checkpoint_dir = ck
    t.log_dir = ld
    t.round = 5
    t.total_games = 100
    t.total_training_steps = 50
    t.performance_mode = "balanced"
    t._auto_mode = None
    t._training_pause_event = None
    t.VARIANTS = ["baseline", "attack", "est"]
    t.buffers = {v: ReplayBuffer(max_size=100000) for v in t.VARIANTS}
    t._spectate_queue = __import__("queue").Queue()

    s = ControlServer(t, host="127.0.0.1", port=0,
                      dashboard_dir=Path("train/league/dashboard"))
    s.start()
    deadline = time.time() + 3
    while s.actual_port is None and time.time() < deadline:
        time.sleep(0.02)
    print(f"Server on port {s.actual_port}")

    def req(method, path, body=None):
        c = http.client.HTTPConnection("127.0.0.1", s.actual_port, timeout=3)
        headers = {"Content-Type": "application/json"} if body else {}
        c.request(method, path, body, headers)
        r = c.getresponse()
        return r, r.read()

    r, body = req("GET", "/")
    print(f"GET / -> {r.status} ({len(body)} bytes)")
    assert r.status == 200
    assert b"spectate-board" in body, "dashboard HTML missing spectate canvas"

    r, body = req("GET", "/style.css")
    print(f"GET /style.css -> {r.status} ({len(body)} bytes)")
    assert r.status == 200 and b"--bg" in body

    r, body = req("GET", "/dashboard.js")
    print(f"GET /dashboard.js -> {r.status} ({len(body)} bytes)")
    assert r.status == 200 and b"pollStatus" in body

    r, body = req("GET", "/api/status")
    print(f"GET /api/status -> {r.status} keys: {list(json.loads(body).keys())}")
    assert r.status == 200

    r, body = req("POST", "/api/mode", json.dumps({"mode": "boost"}))
    print(f"POST /api/mode -> {r.status} body: {body.decode()}")
    assert r.status == 200

    s.stop()
    print("OK")


if __name__ == "__main__":
    main()
