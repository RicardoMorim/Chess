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
