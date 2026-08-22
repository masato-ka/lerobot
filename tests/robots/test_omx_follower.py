#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Config/property-level tests for OmxFollower's force-estimation integration -- no hardware connection
needed (constructing `OmxFollower`/`DynamixelMotorsBus` doesn't open the serial port; that only happens on
`.connect()`)."""

from pathlib import Path

import torch

from lerobot.robots.omx_follower.config_omx_follower import (
    OmxFollowerConfig,
    OmxFollowerForceEstimationConfig,
)
from lerobot.robots.omx_follower.omx_follower import OmxFollower

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def _make_minimal_checkpoint(tmp_path: Path, history_length: int = 2) -> Path:
    """A tiny but structurally valid NEXT checkpoint -- just large enough for
    `OnlineExternalTorqueEstimator` to load and run, not meant to have learned anything."""
    from lerobot.force_estimation.next_model import NextTorqueEstimator

    num_joints = len(ARM_JOINTS)
    model = NextTorqueEstimator(
        num_joints=num_joints,
        history_length=history_length,
        lstm_hidden_size=4,
        lstm_num_layers=1,
        mlp_hidden_size=4,
        dropout=0.0,
    )
    checkpoint = {
        "joint_names": ARM_JOINTS,
        "history_length": history_length,
        "resample_hz": 100.0,
        "lstm_hidden_size": 4,
        "lstm_num_layers": 1,
        "mlp_hidden_size": 4,
        "dropout": 0.0,
        "model_state_dict": model.state_dict(),
        "x_mean": torch.zeros(3 * num_joints),
        "x_std": torch.ones(3 * num_joints),
        "y_mean": torch.zeros(num_joints),
        "y_std": torch.ones(num_joints),
    }
    checkpoint_path = tmp_path / "next_test.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def test_default_config_disables_force_estimation():
    config = OmxFollowerConfig(port="/dev/fake")
    assert config.force_estimation.checkpoint_path == ""

    follower = OmxFollower(config)
    assert not any(k.startswith("force.") for k in follower.observation_features)
    assert follower._force_estimator is None


def test_force_estimation_enabled_via_checkpoint_path(tmp_path):
    checkpoint_path = _make_minimal_checkpoint(tmp_path)
    config = OmxFollowerConfig(
        port="/dev/fake",
        force_estimation=OmxFollowerForceEstimationConfig(checkpoint_path=str(checkpoint_path)),
    )
    follower = OmxFollower(config)

    assert follower._force_estimator is not None
    expected_force_keys = {f"force.{j}": float for j in ARM_JOINTS}
    assert expected_force_keys.items() <= follower.observation_features.items()
    # Position keys (incl. gripper) must still be present alongside the new force keys.
    assert all(f"{j}.pos" in follower.observation_features for j in [*ARM_JOINTS, "gripper"])
