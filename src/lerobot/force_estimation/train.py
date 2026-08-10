#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training loop for FACTR2's NEXT external-torque estimator."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .dataset import NextWindowDataset, load_episode, resample_uniform
from .next_model import NextTorqueEstimator

logger = logging.getLogger(__name__)


@dataclass
class NextTrainConfig:
    """Configuration for [`train_next`].

    Args:
        data_paths (`list[str]`):
            Paths to `.npz` free-motion logs from `collect_free_motion.py`. All must share the
            same `joint_names`/order.
        output_path (`str`):
            Where to save the trained checkpoint (`.pt`).
        history_length (`int`, *optional*, defaults to `50`):
            Number of past timesteps fed to the model at each prediction.
        resample_hz (`float`, *optional*, defaults to `100.0`):
            Uniform rate the raw (irregularly-timestamped) logs are resampled to before windowing.
        lstm_hidden_size (`int`, *optional*, defaults to `128`):
            Hidden size of the LSTM backbone.
        lstm_num_layers (`int`, *optional*, defaults to `2`):
            Number of stacked LSTM layers.
        mlp_hidden_size (`int`, *optional*, defaults to `256`):
            Hidden size of the MLP head.
        dropout (`float`, *optional*, defaults to `0.1`):
            Dropout probability applied in the LSTM (between layers) and the MLP head.
        batch_size (`int`, *optional*, defaults to `256`):
            Training/validation batch size.
        max_epochs (`int`, *optional*, defaults to `200`):
            Maximum number of training epochs before stopping regardless of validation loss.
        patience (`int`, *optional*, defaults to `15`):
            Epochs without validation-loss improvement before early stopping.
        lr (`float`, *optional*, defaults to `1e-3`):
            AdamW learning rate.
        weight_decay (`float`, *optional*, defaults to `1e-6`):
            AdamW weight decay.
        val_fraction (`float`, *optional*, defaults to `0.1`):
            Fraction of windows held out for validation/early stopping.
        seed (`int`, *optional*, defaults to `0`):
            Random seed for the train/validation split and model initialization.
    """

    data_paths: list[str] = field(default_factory=list)
    output_path: str = ""

    history_length: int = 50
    resample_hz: float = 100.0

    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    mlp_hidden_size: int = 256
    dropout: float = 0.1

    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-6
    val_fraction: float = 0.1
    seed: int = 0


def train_next(cfg: NextTrainConfig) -> NextTorqueEstimator:
    """Train a [`NextTorqueEstimator`] on logged free-motion episodes.

    Follows FACTR2's NEXT training recipe (arXiv:2606.12406, eq. 4): AdamW + L2 regression
    against measured (free-space) motor torque, with early stopping on a held-out validation
    split.

    Feature/target standardization (zero mean, unit std) is applied before feeding the model;
    the paper doesn't specify this, but it's necessary here since raw position/velocity/current
    register values live on very different numeric scales (see `README.md`). Stats are saved in
    the checkpoint and must be reapplied identically at inference time (handled by
    [`~force_estimation.OnlineExternalTorqueEstimator`]).

    Args:
        cfg (`NextTrainConfig`):
            Training configuration.

    Returns:
        `NextTorqueEstimator`: The trained model (best validation checkpoint restored).
    """
    if not cfg.data_paths:
        raise ValueError("cfg.data_paths must contain at least one free-motion log (.npz).")

    torch.manual_seed(cfg.seed)

    episodes = [resample_uniform(load_episode(p), cfg.resample_hz) for p in cfg.data_paths]
    joint_names = episodes[0].joint_names
    if any(ep.joint_names != joint_names for ep in episodes):
        raise ValueError("All data_paths must share the same joint_names/order.")
    num_joints = len(joint_names)

    dataset = NextWindowDataset(episodes, history_length=cfg.history_length)
    if len(dataset) < 10:
        raise ValueError(
            f"Only {len(dataset)} training windows found across {len(episodes)} episode(s); "
            "collect more free-motion data (aim for >= 10 minutes at ~100Hz)."
        )

    all_x = np.concatenate(dataset.features, axis=0)
    x_mean, x_std = all_x.mean(axis=0), all_x.std(axis=0) + 1e-6
    all_y = np.concatenate(dataset.targets, axis=0)
    y_mean, y_std = all_y.mean(axis=0), all_y.std(axis=0) + 1e-6

    val_size = max(1, int(len(dataset) * cfg.val_fraction))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(cfg.seed)
    )
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NextTorqueEstimator(
        num_joints=num_joints,
        history_length=cfg.history_length,
        lstm_hidden_size=cfg.lstm_hidden_size,
        lstm_num_layers=cfg.lstm_num_layers,
        mlp_hidden_size=cfg.mlp_hidden_size,
        dropout=cfg.dropout,
    ).to(device)

    x_mean_t = torch.tensor(x_mean, device=device)
    x_std_t = torch.tensor(x_std, device=device)
    y_mean_t = torch.tensor(y_mean, device=device)
    y_std_t = torch.tensor(y_std, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = (x - x_mean_t) / x_std_t
            y = (y - y_mean_t) / y_std_t
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                x = (x - x_mean_t) / x_std_t
                y = (y - y_mean_t) / y_std_t
                val_losses.append(loss_fn(model(x), y).item())
        val_loss = float(np.mean(val_losses))
        logger.info(f"epoch {epoch}: val_loss={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                logger.info(f"Early stopping at epoch {epoch} (best val_loss={best_val_loss:.5f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "joint_names": joint_names,
        "history_length": cfg.history_length,
        "lstm_hidden_size": cfg.lstm_hidden_size,
        "lstm_num_layers": cfg.lstm_num_layers,
        "mlp_hidden_size": cfg.mlp_hidden_size,
        "dropout": cfg.dropout,
        "resample_hz": cfg.resample_hz,
        # Stored as tensors (not numpy arrays) so the checkpoint only contains torch-native
        # types and can be loaded with the safer `torch.load(..., weights_only=True)` default.
        "x_mean": torch.from_numpy(x_mean),
        "x_std": torch.from_numpy(x_std),
        "y_mean": torch.from_numpy(y_mean),
        "y_std": torch.from_numpy(y_std),
    }
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    logger.info(f"Saved NEXT checkpoint to {output_path} (best val_loss={best_val_loss:.5f})")

    return model
