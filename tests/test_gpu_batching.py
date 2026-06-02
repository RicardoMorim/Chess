import unittest

from train.league.gpu_inference_process import _should_flush_batch


class GPUBatchingTests(unittest.TestCase):
    def test_should_flush_when_batch_size_reached(self):
        self.assertTrue(
            _should_flush_batch(
                pending_size=64,
                pending_since=None,
                batch_size=64,
                post_batch_wait_ms=15,
                now=100.0,
            )
        )

    def test_should_not_flush_early_when_wait_not_elapsed(self):
        self.assertFalse(
            _should_flush_batch(
                pending_size=3,
                pending_since=100.0,
                batch_size=64,
                post_batch_wait_ms=15,
                now=100.010,
            )
        )

    def test_should_flush_after_wait_window(self):
        self.assertTrue(
            _should_flush_batch(
                pending_size=3,
                pending_since=100.0,
                batch_size=64,
                post_batch_wait_ms=15,
                now=100.020,
            )
        )


if __name__ == "__main__":
    unittest.main()
