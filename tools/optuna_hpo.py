"""Optuna hyperparameter optimization helper for the chess training pipeline.

This script keeps the search space intentionally small and fast so it can be
used as a repeatable smoke test for training ideas.

It trains on a compact tactical dataset derived from `core.constants`
`TACTICAL_TEST_POSITIONS`, then evaluates on a held-out split.

Example usage:
  python tools/optuna_hpo.py --config tools/hpo_example.json
  python tools/optuna_hpo.py --trials 10 --epochs 4 --seed 42

If you want to persist results across runs, provide a SQLite storage URI in
the config or via `--storage`, for example:
  sqlite:///optuna_chess.db
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from functools import partial
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = REPO_ROOT / "train"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))


@dataclass(frozen=True)
class HPOConfig:
    seed: int = 42
    trials: int = 20
    epochs: int = 4
    batch_size_candidates: tuple[int, ...] = (4, 8, 16)
    variants: tuple[str, ...] = ("baseline", "attack", "est")
    learning_rate_min: float = 1e-4
    learning_rate_max: float = 3e-2
    weight_decay_min: float = 1e-6
    weight_decay_max: float = 1e-3
    policy_weight_min: float = 0.7
    policy_weight_max: float = 1.8
    value_weight_min: float = 0.5
    value_weight_max: float = 1.5
    storage: str | None = None
    study_name: str = "chess_tactical_hpo"


def set_global_seed(seed: int):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False


def load_config(path: str | None) -> HPOConfig:
    if not path:
        return HPOConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    base = HPOConfig()
    return HPOConfig(
        seed=int(raw.get("seed", base.seed)),
        trials=int(raw.get("trials", base.trials)),
        epochs=int(raw.get("epochs", base.epochs)),
        batch_size_candidates=tuple(raw.get("batch_size_candidates", base.batch_size_candidates)),
        variants=tuple(raw.get("variants", base.variants)),
        learning_rate_min=float(raw.get("learning_rate_min", base.learning_rate_min)),
        learning_rate_max=float(raw.get("learning_rate_max", base.learning_rate_max)),
        weight_decay_min=float(raw.get("weight_decay_min", base.weight_decay_min)),
        weight_decay_max=float(raw.get("weight_decay_max", base.weight_decay_max)),
        policy_weight_min=float(raw.get("policy_weight_min", base.policy_weight_min)),
        policy_weight_max=float(raw.get("policy_weight_max", base.policy_weight_max)),
        value_weight_min=float(raw.get("value_weight_min", base.value_weight_min)),
        value_weight_max=float(raw.get("value_weight_max", base.value_weight_max)),
        storage=raw.get("storage", base.storage),
        study_name=str(raw.get("study_name", base.study_name)),
    )


def build_examples():
    """Build a compact tactical dataset from the frozen tactical test positions.

    Returns raw examples as `(fen, move_uci, value_target)` tuples so the tensor
    conversion can be adapted to the selected model variant later.
    """
    import chess

    from core.constants import TACTICAL_TEST_POSITIONS

    category_value = {
        "mate_in_one": 1.0,
        "knight_fork": 0.9,
        "pin": 0.85,
        "discovered": 0.85,
        "skewer": 0.85,
        "endgame": 0.75,
    }

    examples = []
    for category, entries in TACTICAL_TEST_POSITIONS.items():
        for fen, move_uci in entries:
            try:
                board = chess.Board(fen)
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    continue

                value_target = category_value.get(category, 0.7)
                examples.append((board.fen(), move.uci(), value_target))

                mirrored_board = board.mirror()
                mirrored_move = chess.Move(
                    chess.square_mirror(move.from_square),
                    chess.square_mirror(move.to_square),
                    move.promotion,
                )
                if mirrored_move in mirrored_board.legal_moves:
                    examples.append((mirrored_board.fen(), mirrored_move.uci(), value_target))
            except Exception:
                continue

    if not examples:
        raise RuntimeError("No tactical examples could be built from TACTICAL_TEST_POSITIONS")

    return examples


def split_examples(examples, seed: int, train_ratio: float = 0.8):
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * train_ratio))
    train_examples = shuffled[:split_idx]
    valid_examples = shuffled[split_idx:] or shuffled[-1:]
    return train_examples, valid_examples


def make_loader(examples, batch_size: int, shuffle: bool, input_channels: int):
    if torch is None:
        raise RuntimeError("PyTorch is required for the HPO runner")

    import chess

    from core.data import board_to_tensor, get_move_index

    inputs = []
    policy_targets = []
    value_targets = []
    for fen, move_uci, value_target in examples:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        inputs.append(board_to_tensor(board, move_number=board.fullmove_number, input_channels=input_channels))
        policy_targets.append(get_move_index(move))
        value_targets.append(value_target)

    inputs = torch.tensor(np.asarray(inputs), dtype=torch.float32)
    policy_targets = torch.tensor(policy_targets, dtype=torch.long)
    value_targets = torch.tensor(value_targets, dtype=torch.float32)
    dataset = TensorDataset(inputs, policy_targets, value_targets)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def choose_optimizer(model, trial, learning_rate: float, weight_decay: float):
    optimizer_name = trial.suggest_categorical("optimizer", ["sgd", "adam"])
    if optimizer_name == "sgd":
        momentum = trial.suggest_float("momentum", 0.75, 0.95)
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=True,
        )

    betas = trial.suggest_categorical("adam_betas", [(0.9, 0.999), (0.9, 0.98)])
    return torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
    )


def suggest_trial_params(trial, config: HPOConfig):
    return {
        "variant": trial.suggest_categorical("variant", list(config.variants)),
        "batch_size": trial.suggest_categorical("batch_size", list(config.batch_size_candidates)),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            config.learning_rate_min,
            config.learning_rate_max,
            log=True,
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            config.weight_decay_min,
            config.weight_decay_max,
            log=True,
        ),
        "policy_weight": trial.suggest_float(
            "policy_weight",
            config.policy_weight_min,
            config.policy_weight_max,
        ),
        "value_weight": trial.suggest_float(
            "value_weight",
            config.value_weight_min,
            config.value_weight_max,
        ),
    }


def run_training_epoch(
    model,
    loader,
    optimizer,
    device,
    policy_loss_fn,
    value_loss_fn,
    policy_weight: float,
    value_weight: float,
    scaler,
):
    model.train()
    for inputs, policy_targets, value_targets in loader:
        inputs = inputs.to(device)
        policy_targets = policy_targets.to(device)
        value_targets = value_targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                policy_logits, value_pred = model(inputs)
                policy_loss = policy_loss_fn(policy_logits, policy_targets)
                value_loss = value_loss_fn(value_pred, value_targets)
                loss = policy_weight * policy_loss + value_weight * value_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            policy_logits, value_pred = model(inputs)
            policy_loss = policy_loss_fn(policy_logits, policy_targets)
            value_loss = value_loss_fn(value_pred, value_targets)
            loss = policy_weight * policy_loss + value_weight * value_loss
            loss.backward()
            optimizer.step()


def optuna_objective(trial, *, config: HPOConfig, train_examples, valid_examples):
    # Use Lightning Trainer for training within Optuna trials.
    from core.constants import MODEL_CONFIG
    from core.models import create_model
    from core.training import PolicyLoss, ValueLoss, TRAIN_CONFIG
    from train.core.lightning_module import ChessLightning
    from train.core.repro import set_seed

    # Per-trial suggestions
    params = suggest_trial_params(trial, config)
    variant = params["variant"]
    input_channels = MODEL_CONFIG[variant]["input_channels"]
    batch_size = params["batch_size"]

    # Build DataLoaders
    train_loader = make_loader(train_examples, batch_size=batch_size, shuffle=True, input_channels=input_channels)
    valid_loader = make_loader(valid_examples, batch_size=batch_size, shuffle=False, input_channels=input_channels)

    # Seed per-trial for determinism
    set_seed(config.seed + int(trial.number))

    # Build Lightning model and pass minimal config overrides
    lit_config = dict(TRAIN_CONFIG)
    lit_config.update({
        "policy_weight": params["policy_weight"],
        "value_weight": params["value_weight"],
    })

    model = ChessLightning(variant=variant, lr_epochs=config.epochs, config=lit_config)

    # Configure Trainer
    import pytorch_lightning as pl
    try:
        from optuna.integration import PyTorchLightningPruningCallback
        pruning_cb = PyTorchLightningPruningCallback(trial, monitor="val/loss")
        callbacks = [pruning_cb]
    except Exception:
        callbacks = []

    accelerator = "gpu" if torch and torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        logger=False,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        enable_progress_bar=False,
    )

    # Run training
    trainer.fit(model, train_loader, valid_loader)

    # Prefer Lightning's validation metrics if available
    try:
        val_results = trainer.validate(model, valid_loader, verbose=False)
        # val_results is a list of dicts; take first
        if val_results and isinstance(val_results, list):
            vr = val_results[0]
            # Our LightningModule logs 'val/loss'
            val_loss = vr.get("val/loss") or vr.get("val_loss") or float('inf')
        else:
            val_loss = float('inf')
    except Exception:
        # Fallback: evaluate with existing function
        policy_loss_fn = PolicyLoss()
        value_loss_fn = ValueLoss(use_huber=True)
        val_policy_loss, val_value_loss, val_policy_accuracy = evaluate(model, valid_loader, policy_loss_fn, value_loss_fn, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        val_loss = val_policy_loss + val_value_loss + (1.0 - val_policy_accuracy)

    # Report to Optuna and allow pruning logic to use PyTorchLightningPruningCallback
    trial.report(val_loss, 0)
    if trial.should_prune():
        raise __import__("optuna").TrialPruned()

    trial.set_user_attr("val_loss", float(val_loss))
    return float(val_loss)


def evaluate(model, loader, policy_loss_fn, value_loss_fn, device):
    model.eval()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_examples = 0
    correct = 0

    with torch.no_grad():
        for inputs, policy_targets, value_targets in loader:
            inputs = inputs.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)

            policy_logits, value_pred = model(inputs)
            policy_loss = policy_loss_fn(policy_logits, policy_targets)
            value_loss = value_loss_fn(value_pred, value_targets)

            total_policy_loss += float(policy_loss.item()) * inputs.size(0)
            total_value_loss += float(value_loss.item()) * inputs.size(0)
            total_examples += inputs.size(0)

            predictions = policy_logits.argmax(dim=1)
            correct += int((predictions == policy_targets).sum().item())

    avg_policy_loss = total_policy_loss / max(1, total_examples)
    avg_value_loss = total_value_loss / max(1, total_examples)
    policy_accuracy = correct / max(1, total_examples)
    return avg_policy_loss, avg_value_loss, policy_accuracy


def objective_factory(config: HPOConfig):
    examples = build_examples()
    train_examples, valid_examples = split_examples(examples, seed=config.seed)
    return partial(optuna_objective, config=config, train_examples=train_examples, valid_examples=valid_examples)


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna HPO for the chess neural network")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file")
    parser.add_argument("--seed", type=int, default=None, help="Override the seed")
    parser.add_argument("--trials", type=int, default=None, help="Override the trial count")
    parser.add_argument("--epochs", type=int, default=None, help="Override the training epochs per trial")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URI (e.g. sqlite:///optuna.db)")
    parser.add_argument("--study-name", type=str, default=None, help="Optuna study name")
    return parser.parse_args()


def main():
    if torch is None:
        raise SystemExit("PyTorch is required. Install dependencies with `pip install -r requirements.txt`.")

    try:
        import optuna
    except Exception as e:
        raise SystemExit(
            "Optuna is required. Install dependencies with `pip install -r requirements.txt`.\n"
            f"Import error: {e}"
        )

    args = parse_args()
    config = load_config(args.config)

    if args.seed is not None:
        config = HPOConfig(**{**config.__dict__, "seed": args.seed})
    if args.trials is not None:
        config = HPOConfig(**{**config.__dict__, "trials": args.trials})
    if args.epochs is not None:
        config = HPOConfig(**{**config.__dict__, "epochs": args.epochs})
    if args.storage is not None:
        config = HPOConfig(**{**config.__dict__, "storage": args.storage})
    if args.study_name is not None:
        config = HPOConfig(**{**config.__dict__, "study_name": args.study_name})

    # Use centralized reproducibility helper
    try:
        from train.core.repro import set_seed
    except Exception:
        # Fallback to local setter if centralized helper is unavailable
        from tools.optuna_hpo import set_global_seed as _local_set
        set_seed = _local_set

    set_seed(config.seed)
    objective = objective_factory(config)

    sampler = optuna.samplers.TPESampler(seed=config.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)
    study = optuna.create_study(
        study_name=config.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=config.storage,
        load_if_exists=bool(config.storage),
    )

    print("Starting Optuna study")
    print(f"  study_name: {config.study_name}")
    print("  direction:   minimize")
    print(f"  trials:      {config.trials}")
    print(f"  epochs:      {config.epochs}")
    print(f"  seed:        {config.seed}")
    if config.storage:
        print(f"  storage:     {config.storage}")

    study.optimize(objective, n_trials=config.trials)

    print("\nBest trial:")
    print(f"  value: {study.best_value:.6f}")
    print("  params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print("  metrics:")
    for key, value in study.best_trial.user_attrs.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
