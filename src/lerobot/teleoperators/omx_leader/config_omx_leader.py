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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@dataclass
class OmxLeaderForceFeedbackConfig:
    """Gravity compensation + joint-limit barrier + velocity damping + force-feedback current injection
    for the omx_leader arm, ported from `examples/omx/bilateral_teleop/bilateral_teleop_demo.py` (see
    `lerobot.teleoperators.omx_leader.gravity_compensation`/`leader_safety` for the underlying math, reused
    unchanged here). Disabled by default (`urdf_path == ""`) -- every field below is inert unless
    `urdf_path` is set, which also switches the arm into Dynamixel Current Control Mode on connect.
    """

    # Path to `omx_l.urdf` (or an equivalent URDF using the same joint1..joint5 naming). Empty string
    # (the default) disables force feedback entirely -- `OmxLeader` stays plain position teleop.
    urdf_path: str = ""

    # Gravity-compensation gain (Nm -> mA conversion factor on top of KT_NM_PER_A), joint-limit barrier
    # P/D gains, velocity damping gain, and force-feedback gain -- global defaults matching
    # `bilateral_teleop_demo.py`'s CLI defaults. See `leader_safety.DEFAULT_JOINT_MODIFIER_OVERRIDES` for
    # why `modifier_overrides` defaults the way it does below.
    modifier: float = 0.09
    damping_gain: float = 0.05
    joint_limit_kp: float = 3.0
    joint_limit_kd: float = 0.0
    feedback_gain: float = 0.0
    current_limit_ma: int = 500
    feedback_limit_ma: int = 200

    # Per-joint overrides of the gains above (joint name -> value), applied on top of the global default
    # for that joint only. `None` (the default) means no per-joint overrides beyond
    # `leader_safety.DEFAULT_JOINT_MODIFIER_OVERRIDES` for `modifier`. From the CLI, draccus parses these
    # as a JSON string, e.g. --teleop.force_feedback.modifier_overrides='{"shoulder_lift": 0.1}'
    # (dotted-key syntax like `...modifier_overrides.shoulder_lift=0.1` is NOT supported).
    modifier_overrides: dict[str, float] | None = None
    damping_gain_overrides: dict[str, float] | None = None
    joint_limit_kp_overrides: dict[str, float] | None = None
    joint_limit_kd_overrides: dict[str, float] | None = None
    feedback_gain_overrides: dict[str, float] | None = None


@TeleoperatorConfig.register_subclass("omx_leader")
@dataclass
class OmxLeaderConfig(TeleoperatorConfig):
    # Port to connect to the arm
    port: str

    # Sets the arm in torque mode with the gripper motor set to this value. This makes it possible to squeeze
    # the gripper and have it spring back to an open position on its own.
    gripper_open_pos: float = 60.0

    # Force feedback (gravity comp + joint-limit barrier + damping + tau_ext-driven current injection).
    # Disabled by default -- see `OmxLeaderForceFeedbackConfig.urdf_path`.
    force_feedback: OmxLeaderForceFeedbackConfig = field(default_factory=OmxLeaderForceFeedbackConfig)
