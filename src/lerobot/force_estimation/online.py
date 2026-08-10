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

"""Real-time external-torque estimation from a trained NEXT checkpoint."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from .next_model import NextTorqueEstimator


class OnlineExternalTorqueEstimator:
    """Streaming external-torque estimation from a trained [`NextTorqueEstimator`] checkpoint.

    Implements FACTR2's NEXT inference rule (arXiv:2606.12406, eq. 2): `tau_ext = tau_m -
    f_theta(x)`.

    Call [`~OnlineExternalTorqueEstimator.update`] once per control step with the latest
    per-joint readings. Returns `None` until `history_length` steps have been observed (the
    ring buffer needs to fill first), then a per-joint external-torque estimate on every
    subsequent call.

    Args:
        checkpoint_path (`str | Path`):
            Path to a checkpoint produced by [`~force_estimation.train_next`].
        device (`str`, *optional*):
            Torch device to run inference on. Defaults to CUDA if available, else CPU.

    **Attributes**:
        - **joint_names** (`list[str]`) -- Joint names, in the order expected by `update`'s dict
          arguments.
        - **history_length** (`int`) -- Number of past timesteps the model requires.
        - **resample_hz** (`float`) -- Uniform sampling rate the training data was resampled to
          (see [`~force_estimation.resample_uniform`]); replaying logged episodes for
          evaluation should resample to this same rate first.
    """

    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        """Load the checkpoint and build the model; see the class docstring for the parameters."""
        # weights_only=True: the checkpoint only contains tensors, the model's own state_dict,
        # and plain str/int/float metadata (see train.train_next) -- no arbitrary objects.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        self.joint_names: list[str] = list(checkpoint["joint_names"])
        self.history_length: int = checkpoint["history_length"]
        self.resample_hz: float = checkpoint["resample_hz"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = NextTorqueEstimator(
            num_joints=len(self.joint_names),
            history_length=self.history_length,
            lstm_hidden_size=checkpoint["lstm_hidden_size"],
            lstm_num_layers=checkpoint["lstm_num_layers"],
            mlp_hidden_size=checkpoint["mlp_hidden_size"],
            dropout=checkpoint["dropout"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.x_mean = checkpoint["x_mean"].to(device=self.device, dtype=torch.float32)
        self.x_std = checkpoint["x_std"].to(device=self.device, dtype=torch.float32)
        self.y_mean = checkpoint["y_mean"].to(device=self.device, dtype=torch.float32)
        self.y_std = checkpoint["y_std"].to(device=self.device, dtype=torch.float32)

        self._history: deque[np.ndarray] = deque(maxlen=self.history_length)

    def reset(self) -> None:
        """Clear the rolling history buffer (e.g. after a discontinuous jump in commanded pose)."""
        self._history.clear()

    @torch.no_grad()
    def update(
        self,
        q: dict[str, float],
        qdot: dict[str, float],
        goal_q: dict[str, float],
        current: dict[str, float],
    ) -> dict[str, float] | None:
        """Push one control step of joint state and get the latest external-torque estimate.

        Args:
            q (`dict[str, float]`):
                Present position per joint, keyed by bare joint name (same units as the
                training data, i.e. [`~robots.Robot.get_observation`]'s `f"{joint}.pos"` values).
            qdot (`dict[str, float]`):
                Present velocity per joint (raw `Present_Velocity` register units).
            goal_q (`dict[str, float]`):
                Commanded goal position per joint (same units as `q`).
            current (`dict[str, float]`):
                Present current/load per joint (raw `Present_Current` register units).

        Returns:
            `dict[str, float] | None`: Estimated external torque per joint (same raw-proxy
            units as `current`), or `None` if the history buffer has not yet filled (first
            `history_length` calls).
        """
        feat = np.concatenate(
            [
                np.array([q[j] for j in self.joint_names], dtype=np.float32),
                np.array([qdot[j] for j in self.joint_names], dtype=np.float32),
                np.array([goal_q[j] - q[j] for j in self.joint_names], dtype=np.float32),
            ]
        )
        self._history.append(feat)
        if len(self._history) < self.history_length:
            return None

        x = torch.from_numpy(np.stack(self._history, axis=0)).unsqueeze(0).to(self.device)
        x = (x - self.x_mean) / self.x_std
        pred_norm = self.model(x)[0]
        pred_free_space = pred_norm * self.y_std + self.y_mean

        current_arr = torch.tensor(
            [current[j] for j in self.joint_names], device=self.device, dtype=torch.float32
        )
        tau_ext = current_arr - pred_free_space
        return {joint: float(tau_ext[i]) for i, joint in enumerate(self.joint_names)}
