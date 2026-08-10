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

"""The NEXT (Neural External Torque Estimation) model from FACTR2 (arXiv:2606.12406)."""

import torch
from torch import nn


class NextTorqueEstimator(nn.Module):
    """LSTM+MLP free-space motor torque predictor, per FACTR2's NEXT (arXiv:2606.12406, Sec. 3).

    Given a history of `history_length` timesteps of per-joint `[q, qdot, delta_q_d]`
    (joint position, velocity, and commanded-minus-actual tracking error), predicts the
    motor torque that would be measured in free space (no external contact) at the final
    timestep. The external torque estimate is then the residual between this prediction
    and the actually measured motor torque (see [`~force_estimation.online.OnlineExternalTorqueEstimator`]).

    Architecture matches the paper: 2-layer LSTM (hidden=128) feeding a 2-layer MLP head
    (hidden=256), with dropout=0.1, using a stateless sliding-window formulation (no hidden
    state carried across calls).

    Args:
        num_joints (`int`):
            Number of joints jointly modeled (e.g. 5 for the OMX arm, gripper excluded).
        history_length (`int`, *optional*, defaults to `50`):
            Number of past timesteps fed to the LSTM at each prediction.
        lstm_hidden_size (`int`, *optional*, defaults to `128`):
            Hidden size of the LSTM backbone.
        lstm_num_layers (`int`, *optional*, defaults to `2`):
            Number of stacked LSTM layers.
        mlp_hidden_size (`int`, *optional*, defaults to `256`):
            Hidden size of the MLP head.
        dropout (`float`, *optional*, defaults to `0.1`):
            Dropout probability applied in the LSTM (between layers) and the MLP head.
    """

    def __init__(
        self,
        num_joints: int,
        history_length: int = 50,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        mlp_hidden_size: int = 256,
        dropout: float = 0.1,
    ):
        """Build the LSTM backbone and MLP head; see the class docstring for the parameters."""
        super().__init__()
        self.num_joints = num_joints
        self.history_length = history_length
        input_size = num_joints * 3  # [q, qdot, delta_q_d] per joint

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, num_joints),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict free-space motor torque from a window of joint-state history.

        Args:
            x (`torch.Tensor`):
                Shape `(batch, history_length, num_joints * 3)` feature history.

        Returns:
            `torch.Tensor`: Shape `(batch, num_joints)` predicted free-space torque (proxy units, see
            `README.md`).
        """
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.head(last_step)
