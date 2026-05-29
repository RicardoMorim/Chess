"""Quickstart script to run a tiny Lightning training session for CI smoke tests.

Runs one epoch on a compact tactical dataset using the Lightning wrapper.
Intended for fast verification (CI smoke) rather than full training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train.core.repro import set_seed


def build_loaders(batch_size: int = 8):
    # Reuse the Optuna helper dataset builder logic without importing the whole script
    import chess
    from train.core.constants import TACTICAL_TEST_POSITIONS
    from train.core.data import board_to_tensor, get_move_index

    examples = []
    for category, items in TACTICAL_TEST_POSITIONS.items():
        for fen, move_uci in items:
            try:
                b = chess.Board(fen)
                m = chess.Move.from_uci(move_uci)
                if m not in b.legal_moves:
                    continue
                examples.append((b, m, 1.0))
                mb = b.mirror()
                mm = chess.Move(chess.square_mirror(m.from_square), chess.square_mirror(m.to_square), m.promotion)
                if mm in mb.legal_moves:
                    examples.append((mb, mm, 1.0))
            except Exception:
                continue

    if not examples:
        raise RuntimeError("No examples available for quickstart")

    # Convert to tensors
    inputs = []
    policies = []
    values = []
    for b, m, v in examples:
        inputs.append(board_to_tensor(b, move_number=b.fullmove_number, input_channels=18))
        policies.append(get_move_index(m))
        values.append(v)

    import numpy as np
    import torch

    x = torch.tensor(np.asarray(inputs), dtype=torch.float32)
    y_p = torch.tensor(policies, dtype=torch.long)
    y_v = torch.tensor(values, dtype=torch.float32)

    ds = torch.utils.data.TensorDataset(x, y_p, y_v)
    # Small split
    n = max(4, int(len(ds) * 0.8))
    train_ds = torch.utils.data.Subset(ds, list(range(0, n)))
    val_ds = torch.utils.data.Subset(ds, list(range(n, len(ds))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--variant", type=str, default="baseline")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="offline", choices=["offline", "online", "disabled"])
    args = parser.parse_args(argv)

    set_seed(args.seed)

    # Import Lightning lazily so the module can still be imported in lightweight smoke tests.
    import pytorch_lightning as pl
    from train.core.lightning_module import ChessLightning

    train_loader, val_loader = build_loaders(batch_size=args.batch_size)

    model = ChessLightning(variant=args.variant, lr_epochs=args.epochs)

    logger = None
    if args.wandb_project:
        try:
            from pytorch_lightning.loggers import WandbLogger

            logger = WandbLogger(
                project=args.wandb_project,
                name=args.wandb_run_name,
                mode=args.wandb_mode,
                log_model=False,
            )
        except Exception:
            logger = None

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="cpu",
        devices=1,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        logger=logger,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
