"""
Regression tests for the channel-mismatch and policy-shape fix in
``LeagueTrainer._train_one_step``.

The original bug:
  * ``ReplayBuffer.DEFAULT_POLICY_SIZE = 4096`` silently truncated the 4672-dim
    MCTS policies. The model's policy head outputs 4672. Training failed with
    "0D or 1D target tensor expected, multi-target not supported" once the
    channel fix was applied.
  * The buffer stored 22-channel positions but baseline/est models accept 18
    channels. Forward through the model raised
    "expected input[256, 22, 8, 8] to have 18 channels".

This file tests:
  1. ``_convert_pos_channels`` trims/pads the channel axis correctly.
  2. ``_soft_cross_entropy`` handles 2D soft targets and produces a finite
     scalar with gradients.
  3. An end-to-end training step on a tiny buffer+model works for both
     18-channel (baseline) and 22-channel (attack) variants.
  4. ``DEFAULT_POLICY_SIZE`` is now 4672 (not the legacy 4096).
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "train"))

from train.league.replay_buffer import ReplayBuffer, DEFAULT_POLICY_SIZE  # noqa: E402
from train.league.league_trainer import (  # noqa: E402
    _convert_pos_channels,
    _soft_cross_entropy,
)
from train.core.models import create_model  # noqa: E402


def _rand_pos(channels: int = 22, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((channels, 8, 8)).astype(np.float32)


def _rand_soft_policy(size: int = 4672, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.random(size).astype(np.float32)
    p /= p.sum()
    return p


class ConvertPosChannelsTests(unittest.TestCase):
    """``_convert_pos_channels`` trims/pads along the channel axis."""

    def test_same_channels_is_noop(self):
        x = torch.randn(4, 18, 8, 8)
        y = _convert_pos_channels(x, 18)
        self.assertEqual(y.shape, (4, 18, 8, 8))
        self.assertTrue(torch.equal(x, y))

    def test_trim_when_source_has_more_channels(self):
        """22-channel buffer data into 18-channel baseline/est model."""
        x = torch.randn(4, 22, 8, 8)
        y = _convert_pos_channels(x, 18)
        self.assertEqual(y.shape, (4, 18, 8, 8))
        # First 18 channels preserved exactly
        self.assertTrue(torch.equal(x[:, :18], y))

    def test_pad_when_source_has_fewer_channels(self):
        """18-channel data into 22-channel buffer (round-trip)."""
        x = torch.randn(4, 18, 8, 8)
        y = _convert_pos_channels(x, 22)
        self.assertEqual(y.shape, (4, 22, 8, 8))
        # First 18 channels preserved; last 4 are zero
        self.assertTrue(torch.equal(x, y[:, :18]))
        self.assertTrue(torch.equal(y[:, 18:], torch.zeros(4, 4, 8, 8)))

    def test_pad_does_not_mutate_input(self):
        x = torch.randn(2, 18, 8, 8)
        x_orig = x.clone()
        _convert_pos_channels(x, 22)
        self.assertTrue(torch.equal(x, x_orig))


class SoftCrossEntropyTests(unittest.TestCase):
    """``_soft_cross_entropy`` accepts 2D soft targets and produces a scalar."""

    def test_returns_scalar(self):
        logits = torch.randn(8, 4672, requires_grad=True)
        target = torch.softmax(torch.randn(8, 4672), dim=1)
        loss = _soft_cross_entropy(logits, target)
        self.assertEqual(loss.shape, ())
        self.assertTrue(torch.isfinite(loss).item())

    def test_gradients_flow(self):
        logits = torch.randn(4, 4672, requires_grad=True)
        target = torch.softmax(torch.randn(4, 4672), dim=1)
        loss = _soft_cross_entropy(logits, target)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_zero_loss_when_target_matches_softmax(self):
        """If target == softmax(logits), soft CE should be the entropy of target."""
        logits = torch.randn(4, 4672, requires_grad=True)
        target = torch.softmax(logits.detach(), dim=1)
        loss = _soft_cross_entropy(logits, target)
        # -sum(target * log(target)) = entropy of target
        expected = -(target * torch.log(target + 1e-12)).sum(dim=1).mean()
        self.assertAlmostEqual(loss.item(), expected.item(), places=4)

    def test_works_on_one_hot_targets(self):
        """One-hot distribution is a valid soft target — should match hard CE for that class."""
        torch.manual_seed(0)
        logits = torch.randn(8, 4672, requires_grad=True)
        idx = torch.randint(0, 4672, (8,))
        one_hot = torch.nn.functional.one_hot(idx, 4672).float()
        soft = _soft_cross_entropy(logits, one_hot)
        hard = torch.nn.functional.cross_entropy(logits, idx)
        self.assertAlmostEqual(soft.item(), hard.item(), places=4)


class DefaultPolicySizeTests(unittest.TestCase):
    """The default buffer policy size must match the model's policy head."""

    def test_default_is_4672(self):
        self.assertEqual(
            DEFAULT_POLICY_SIZE, 4672,
            f"DEFAULT_POLICY_SIZE is {DEFAULT_POLICY_SIZE}; expected 4672 "
            "(73 planes x 64 squares = AlphaZero move encoding used by the model).",
        )

    def test_buffer_default_matches_model(self):
        """Buffer created with defaults must have policy_size == model output."""
        buf = ReplayBuffer(max_size=10)
        self.assertEqual(buf.policy_size, 4672)
        # And the model's policy head outputs 73*64 = 4672
        model = create_model("attack")
        # Trigger lazy init by running a dummy forward
        model.eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 22, 8, 8))
        self.assertEqual(out[0].shape[-1], 4672)


class TrainOneStepEndToEndTests(unittest.TestCase):
    """End-to-end training step with a tiny buffer + real model.

    Catches the original "RuntimeError: ... expected input to have 18
    channels" and "0D or 1D target tensor expected, multi-target not
    supported" regressions.
    """

    def _make_trainer_minimal(self, variant: str, in_channels: int):
        """Build a minimal LeagueTrainer with one variant wired up.

        Heavy subsystems (self-play workers, evaluator, control server,
        spectator) are patched out. We only need: model, optimizer, buffer.
        """
        from unittest.mock import MagicMock
        from train.league.league_trainer import LeagueTrainer
        from torch.optim.lr_scheduler import LambdaLR

        trainer = LeagueTrainer.__new__(LeagueTrainer)
        # Real model on CPU for determinism
        model = create_model(variant, value_dropout=0.0)
        for p in model.parameters():
            p.requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Real (no-op) scheduler — _train_one_step calls .step() on it
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        trainer.models = {variant: model}
        trainer.optimizers = {variant: optimizer}
        trainer.schedulers = {variant: scheduler}
        # 22-channel buffer (the production default)
        trainer.buffers = {
            variant: ReplayBuffer(max_size=128, pos_channels=22, policy_size=4672)
        }
        trainer.device = torch.device("cpu")
        trainer.POLICY_LOSS_WEIGHT = 1.0
        trainer.VALUE_LOSS_WEIGHT = 1.0
        trainer.BATCH_SIZE = 4
        trainer.PUZZLE_BATCHES_PER_GAME_BATCH = 0
        trainer.PROGAME_BATCHES_PER_GAME_BATCH = 0
        trainer.VARIANTS = [variant]
        # Disable aux loader (no puzzles/PGNs in this test)
        trainer.aux_loader = MagicMock()
        trainer.aux_loader.sample_puzzle_batch.return_value = None
        trainer.aux_loader.sample_progame_batch.return_value = None
        # Metrics (callable no-op for the per-step return path)
        trainer.metrics = MagicMock()
        # Bookkeeping touched at the end of _train_one_step
        trainer.total_training_steps = 0
        return trainer
        return trainer

    def _fill_buffer(self, trainer, variant: str, n_games: int = 2):
        rng = np.random.default_rng(0)
        for _ in range(n_games):
            trajectory = []
            for _move in range(trainer.BATCH_SIZE):
                pos = rng.standard_normal((22, 8, 8)).astype(np.float32)
                pol = rng.random(4672).astype(np.float32)
                pol /= pol.sum()
                v = float(rng.uniform(-1.0, 1.0))
                trajectory.append((pos, pol, v))
            trainer.buffers[variant].add_game(trajectory)

    def test_baseline_18ch_with_22ch_buffer(self):
        """The original failure mode: 18ch model fed 22ch buffer data."""
        trainer = self._make_trainer_minimal("baseline", in_channels=18)
        self._fill_buffer(trainer, "baseline")
        # Must not raise
        loss = trainer._train_one_step("baseline")
        self.assertIsNotNone(loss)
        self.assertTrue(np.isfinite(loss))

    def test_attack_22ch_with_22ch_buffer(self):
        """The 22-channel attack model fed 22ch buffer data (no conversion needed)."""
        trainer = self._make_trainer_minimal("attack", in_channels=22)
        self._fill_buffer(trainer, "attack")
        loss = trainer._train_one_step("attack")
        self.assertIsNotNone(loss)
        self.assertTrue(np.isfinite(loss))

    def test_est_18ch_with_22ch_buffer(self):
        """The EST model (18ch early-split) also needs channel trimming."""
        trainer = self._make_trainer_minimal("est", in_channels=18)
        self._fill_buffer(trainer, "est")
        loss = trainer._train_one_step("est")
        self.assertIsNotNone(loss)
        self.assertTrue(np.isfinite(loss))

    def test_gradients_are_finite_after_step(self):
        """All parameter gradients should be finite after a training step."""
        trainer = self._make_trainer_minimal("baseline", in_channels=18)
        self._fill_buffer(trainer, "baseline")
        trainer._train_one_step("baseline")
        for name, p in trainer.models["baseline"].named_parameters():
            if p.grad is None:
                continue
            self.assertTrue(
                torch.isfinite(p.grad).all().item(),
                f"non-finite grad on {name}",
            )


if __name__ == "__main__":
    unittest.main()
