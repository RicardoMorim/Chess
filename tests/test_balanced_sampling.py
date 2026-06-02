import unittest

import torch
from torch.utils.data import TensorDataset

from train.core.data import compute_source_sample_weights, create_balanced_concat_dataloader


class BalancedSamplingTests(unittest.TestCase):
    def test_compute_source_sample_weights(self):
        ds1 = TensorDataset(torch.zeros(2, 1), torch.zeros(2, dtype=torch.long))
        ds2 = TensorDataset(torch.zeros(4, 1), torch.zeros(4, dtype=torch.long))

        weights = compute_source_sample_weights([ds1, ds2], source_weights=[1.0, 3.0])

        self.assertEqual(len(weights), 6)
        self.assertEqual(weights[:2], [0.5, 0.5])
        self.assertEqual(weights[2:], [0.75, 0.75, 0.75, 0.75])

    def test_create_balanced_concat_dataloader(self):
        ds1 = TensorDataset(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long))
        ds2 = TensorDataset(torch.zeros(2, 1), torch.zeros(2, dtype=torch.long))

        loader = create_balanced_concat_dataloader(
            [ds1, ds2],
            batch_size=2,
            source_weights=[1.0, 2.0],
            num_workers=0,
            pin_memory=False,
        )

        self.assertEqual(len(loader.dataset), 3)
        self.assertEqual(len(loader), 2)
        batch = next(iter(loader))
        self.assertEqual(len(batch), 2)


if __name__ == "__main__":
    unittest.main()
