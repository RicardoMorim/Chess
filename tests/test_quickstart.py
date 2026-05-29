import unittest

try:
    import pytorch_lightning  # noqa: F401
    from train import run_lightning_quickstart as quickstart
    HAS_PL = True
except Exception:
    HAS_PL = False


class SmokeQuickstartTests(unittest.TestCase):
    def test_quickstart_runs(self):
        if not HAS_PL:
            self.skipTest("pytorch_lightning not available in environment")
        # Run one-epoch quickstart; should complete without exception
        quickstart.main(["--epochs", "1", "--batch-size", "4", "--seed", "42"])


if __name__ == "__main__":
    unittest.main()
