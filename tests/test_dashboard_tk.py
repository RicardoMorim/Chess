"""
Smoke tests for the Tkinter dashboard module (Fase 3b).

Tkinter is not friendly to unit testing on a headless CI, so we:
  - Verify the module imports cleanly
  - Verify the HTTP client wrapper constructs the right URLs / payloads
  - Mock HTTP and confirm widgets update without raising

The full GUI is exercised manually (Tkinter windows can't be tested in
headless mode without extra deps like Xvfb).
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "train"))


class DashboardTkImportTests(unittest.TestCase):

    def test_module_imports(self):
        from train.league import dashboard_tk
        self.assertTrue(hasattr(dashboard_tk, "TrainerClient"))
        self.assertTrue(hasattr(dashboard_tk, "DashboardApp"))
        self.assertTrue(hasattr(dashboard_tk, "ResourceBars"))
        self.assertTrue(hasattr(dashboard_tk, "VariantsPanel"))
        self.assertTrue(hasattr(dashboard_tk, "CheckpointTable"))

    def test_main_function_exists(self):
        from train.league.dashboard_tk import main
        self.assertTrue(callable(main))


class DashboardTkSmokeTests(unittest.TestCase):
    """Headless smoke tests — construct widgets in a hidden root."""

    def setUp(self):
        # Try to use the real Tk if there's a display; otherwise
        # skip with a clear message (CI is typically headless).
        try:
            import tkinter as tk
            self.root = tk.Tk()
            self.root.withdraw()
        except Exception as e:
            self.skipTest(f"no Tk display available: {e}")

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def test_widgets_construct(self):
        import tkinter as tk
        from train.league.dashboard_tk import (
            ResourceBars, VariantsPanel, CheckpointTable,
            StatusBar, ModePanel, ResourcesPanel, MetricCard, apply_theme,
        )
        apply_theme(self.root)
        # Each widget must construct without raising
        rb = ResourceBars(self.root); rb.pack()
        vp = VariantsPanel(self.root); vp.pack()
        ck = CheckpointTable(self.root); ck.pack()
        sb = StatusBar(self.root); sb.pack()
        mp = ModePanel(self.root, on_mode=lambda m: None, on_auto=lambda e: None,
                       on_pause=lambda: None); mp.pack()
        rp = ResourcesPanel(self.root); rp.pack()
        mc = MetricCard(self.root, "test", "#3fb950"); mc.pack()
        # Update with a fake status to exercise the codepaths
        vp.update({"losses": {"baseline": 0.42, "attack": 0.51, "est": 0.38},
                   "throughput_gpm": {"baseline": 10.0, "attack": 8.5, "est": 9.2},
                   "buffers": {"baseline": {"size": 1000, "capacity": 5000, "fill_pct": 20.0},
                               "attack": {"size": 4500, "capacity": 5000, "fill_pct": 90.0},
                               "est": {"size": 2500, "capacity": 5000, "fill_pct": 50.0}}})
        rb.update({"vram_pct": 45, "vram_used_mb": 7000, "vram_total_mb": 16000,
                   "cpu_pct": 32.0, "ram_pct": 60.0})
        rp.update({"vram_pct": 45, "vram_used_mb": 7000, "vram_total_mb": 16000,
                   "cpu_pct": 32.0, "ram_pct": 60.0})
        ck.set_items([{"variant": "baseline", "step": 35, "size_mb": 144.0,
                       "mtime": 1234567890.0, "name": "baseline_step_35.pt"}])
        mc.set("123", "sub")
        sb.set_status(True, "ok")
        sb.set_round(7, 210, 5000)
        mp.set_mode("boost")
        mp.set_auto(True)
        mp.set_paused(False)
        # Spin the event loop briefly
        self.root.update_idletasks()
        self.root.update()


class TrainerClientTests(unittest.TestCase):
    """The HTTP client wrapper that the dashboard uses."""

    def setUp(self):
        from train.league.dashboard_tk import TrainerClient
        self.client = TrainerClient("http://127.0.0.1:7860")

    def test_get_status_calls_correct_url(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__ = lambda s, *a: None
            mock_resp.read.return_value = json.dumps({"round": 5}).encode()
            mock_urlopen.return_value = mock_resp
            result = self.client.get_status()
            self.assertEqual(result, {"round": 5})
            mock_urlopen.assert_called_once()
            args, kwargs = mock_urlopen.call_args
            self.assertEqual(args[0], "http://127.0.0.1:7860/api/status")
            self.assertEqual(kwargs.get("timeout"), 3.0)

    def test_set_mode_posts_json(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__ = lambda s, *a: None
            mock_resp.read.return_value = json.dumps({"ok": True, "mode": "boost"}).encode()
            mock_urlopen.return_value = mock_resp
            ok = self.client.set_mode("boost")
            self.assertTrue(ok)
            args, kwargs = mock_urlopen.call_args
            req = args[0]
            self.assertEqual(req.full_url, "http://127.0.0.1:7860/api/mode")
            self.assertEqual(req.get_method(), "POST")
            self.assertEqual(req.data, json.dumps({"mode": "boost"}).encode())

    def test_set_paused(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__ = lambda s, *a: None
            mock_resp.read.return_value = json.dumps({"ok": True, "paused": True}).encode()
            mock_urlopen.return_value = mock_resp
            ok = self.client.set_paused(True)
            self.assertTrue(ok)

    def test_network_error_returns_none(self):
        with patch("urllib.request.urlopen", side_effect=OSError("no route")):
            result = self.client.get_status()
            self.assertIsNone(result)
            self.assertIn("no route", self.client.last_error)

    def test_set_knob_returns_bool_from_response(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__ = lambda s, *a: None
            mock_resp.read.return_value = json.dumps({"BATCH_SIZE": True}).encode()
            mock_urlopen.return_value = mock_resp
            ok = self.client.set_knob("BATCH_SIZE", 512)
            self.assertTrue(ok)

    def test_set_knob_false_for_unknown(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__ = lambda s, *a: None
            mock_resp.read.return_value = json.dumps({"BOGUS": False}).encode()
            mock_urlopen.return_value = mock_resp
            ok = self.client.set_knob("BOGUS", 1)
            self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
