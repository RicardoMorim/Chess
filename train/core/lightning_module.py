"""PyTorch Lightning module wrapper for the Chess models.

This provides a compact LightningModule that re-uses existing loss
and optimizer factories from :mod:`train.core.training` and the model
factory in :mod:`train.core.models`.
"""
from __future__ import annotations

import typing

import torch

try:
    import pytorch_lightning as pl
except Exception as _pl_import_error:
    pl = None
    _PL_IMPORT_ERROR = _pl_import_error

from .models import create_model
from .training import PolicyLoss, ValueLoss, create_optimizer, create_scheduler, TRAIN_CONFIG


class ChessLightning(pl.LightningModule if pl is not None else object):
    """LightningModule wrapping the project's chess networks.

    Args:
        variant: Model variant name passed to ``create_model`` (e.g. 'baseline').
        lr_epochs: Number of epochs used to construct the scheduler.
        config: Optional training config; falls back to TRAIN_CONFIG.
        value_dropout: Dropout rate for value head (0.0 = no dropout).
    """
    def __init__(self, variant: str = 'baseline', lr_epochs: int = 10, config: typing.Optional[dict] = None, value_dropout: float = 0.0):
        if pl is None:
            raise ImportError(
                "pytorch_lightning is required to instantiate ChessLightning "
                f"(import error: {_PL_IMPORT_ERROR})"
            )
        super().__init__()
        self.save_hyperparameters()
        self.config = config or TRAIN_CONFIG
        self.model = create_model(variant, value_dropout=value_dropout)

        # Loss helpers
        self.policy_loss = PolicyLoss()
        self.value_loss = ValueLoss(use_huber=True)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, policy_targets, value_targets = batch
        policy_logits, value_pred = self.model(inputs)

        p_loss = self.policy_loss(policy_logits, policy_targets)
        v_loss = self.value_loss(value_pred, value_targets)

        loss = self.config.get('policy_weight', 1.0) * p_loss + self.config.get('value_weight', 1.0) * v_loss

        # Log scalars
        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train/policy_loss', p_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log('train/value_loss', v_loss, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, policy_targets, value_targets = batch
        policy_logits, value_pred = self.model(inputs)

        p_loss = self.policy_loss(policy_logits, policy_targets)
        v_loss = self.value_loss(value_pred, value_targets)

        loss = self.config.get('policy_weight', 1.0) * p_loss + self.config.get('value_weight', 1.0) * v_loss
        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.config)
        scheduler = create_scheduler(optimizer, num_epochs=self.hparams.lr_epochs, config=self.config)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
