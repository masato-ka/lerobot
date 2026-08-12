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

"""Shared control-loop pieces for driving the omx_leader arm in Dynamixel Current Control
Mode: entering/leaving Current Control Mode, gravity-torque unit conversion, a FACTR-style
soft joint-limit barrier, velocity damping, and per-joint gain resolution from CLI args.

Extracted from `examples/omx/gravity_compensation/gravity_comp_demo.py` once a second
consumer (`examples/omx/bilateral_teleop/bilateral_teleop_demo.py`) needed the same logic --
behavior and default values are unchanged from that script.

Does not modify `OmxLeader`/`OmxLeaderConfig`; this is a standalone helper, same as
`gravity_compensation.py`.
"""

from __future__ import annotations

import argparse

from lerobot.motors.dynamixel import OperatingMode

from .gravity_compensation import ARM_JOINTS
from .omx_leader import OmxLeader

# Nominal XL330 torque constant (Nm per A), derived from ROBOTIS's published stall
# torque/stall current figures (~0.35-0.38 Nm/A across the 3.7-6.0V range). XL330's
# Present_Current/Goal_Current is input-supply current rather than true phase current, so
# treat this as a starting point for gravity-comp gain tuning, not a precise calibration. Only
# the gravity term (OmxGravityModel's Nm output) goes through this conversion -- the
# joint-limit and damping gains below are defined directly in mA-per-unit, matching the
# pragmatic "empirical gain, not physically exact" approach used throughout this arm's tuning.
KT_NM_PER_A = 0.36

# Per-joint gravity-comp defaults confirmed on hardware: wrist_roll's gravity torque is close
# to zero at essentially every pose (see OmxGravityModel docstring/README) and applying any
# noticeable current there mostly just resists manual operation, so it stays uncompensated.
# shoulder_pan needs none either. shoulder_lift carries the most load and needs more than the
# --modifier default to avoid falling in some poses.
DEFAULT_JOINT_MODIFIER_OVERRIDES: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.1,
    "wrist_roll": 0.0,
}

# Safe range (normalized position units, per joint), measured on one leader unit with
# find_leader_joint_range.py. wrist_roll is a continuous-rotation joint (no meaningful
# mechanical limit within the normalized range), hence the near-full-range values. Re-measure
# and replace if running a different physical arm.
JOINT_LIMIT_RANGE: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-52.4, 50.9),
    "shoulder_lift": (-68.0, 48.2),
    "elbow_flex": (-59.3, 54.3),
    "wrist_flex": (-48.8, 50.3),
    "wrist_roll": (-100.0, 100.0),
}

# Subtracted from JOINT_LIMIT_RANGE so the barrier starts pushing back a bit before the
# configured limit rather than right at it.
JOINT_LIMIT_SAFETY_MARGIN = 5.0


def resolve_per_joint(
    args: argparse.Namespace,
    prefix: str,
    global_value: float,
    defaults: dict[str, float] | None = None,
) -> dict[str, float]:
    """Per-joint value = its `--{prefix}_<joint>` CLI override if given, else `defaults[joint]`
    if set, else `global_value`."""
    defaults = defaults or {}
    resolved = {}
    for joint in ARM_JOINTS:
        cli_override = getattr(args, f"{prefix}_{joint}")
        if cli_override is not None:
            resolved[joint] = cli_override
        elif joint in defaults:
            resolved[joint] = defaults[joint]
        else:
            resolved[joint] = global_value
    return resolved


def resolve_modifiers(args: argparse.Namespace) -> dict[str, float]:
    return resolve_per_joint(args, "modifier", args.modifier, DEFAULT_JOINT_MODIFIER_OVERRIDES)


def compute_joint_limit_torque(
    q: dict[str, float],
    qdot: dict[str, float],
    joint_limit_range: dict[str, tuple[float, float]],
    kp: dict[str, float],
    kd: dict[str, float],
) -> dict[str, float]:
    """FACTR-style soft joint-limit barrier: a repulsive current (mA-scale, see `KT_NM_PER_A`
    comment above) that grows as a joint approaches/exceeds its configured safe range, and is
    exactly zero once a joint is back inside the margin.
    """
    tau: dict[str, float] = {}
    for joint in ARM_JOINTS:
        lo, hi = joint_limit_range[joint]
        lo += JOINT_LIMIT_SAFETY_MARGIN
        hi -= JOINT_LIMIT_SAFETY_MARGIN
        if q[joint] > hi:
            tau[joint] = -kp[joint] * (q[joint] - hi) - kd[joint] * qdot[joint]
        elif q[joint] < lo:
            tau[joint] = -kp[joint] * (q[joint] - lo) - kd[joint] * qdot[joint]
        else:
            tau[joint] = 0.0
    return tau


def compute_damping_torque(qdot: dict[str, float], damping_gain: dict[str, float]) -> dict[str, float]:
    """`-damping_gain * qdot` per joint (mA-scale, see `KT_NM_PER_A` comment above)."""
    return {joint: -damping_gain[joint] * qdot[joint] for joint in ARM_JOINTS}


def enter_current_control_mode(leader: OmxLeader, current_limit_ma: int) -> None:
    # Scoped to ARM_JOINTS only -- confirmed on hardware that toggling the gripper's
    # Torque_Enable off then back on (even briefly, as the unscoped torque_disabled() used to do)
    # has a lasting side effect: in CURRENT_POSITION mode, the Dynamixel firmware appears to
    # re-lock Goal_Position to wherever Present_Position was at that exact moment, rather than
    # keeping the originally configured target (this is firmware behavior on the Torque_Enable
    # OFF->ON transition, not something DynamixelMotorsBus.enable_torque()/disable_torque() do
    # themselves -- they only ever write Torque_Enable). The gripper has no mechanical bias of
    # its own (confirmed: fully free/backdrivable when unpowered) -- the drift only happens
    # because of the torque toggle. This function never needs to touch the gripper's
    # Operating_Mode/Current_Limit at all, so there's no need to disable its torque either.
    with leader.bus.torque_disabled(ARM_JOINTS):
        for joint in ARM_JOINTS:
            leader.bus.write("Operating_Mode", joint, OperatingMode.CURRENT.value)
            leader.bus.write("Current_Limit", joint, current_limit_ma)


def restore_position_mode(leader: OmxLeader) -> None:
    # Leave torque disabled on exit (don't use torque_disabled(), which would re-enable it) --
    # this is cleanup, the arm should be safe to walk away from afterward.
    leader.bus.disable_torque()
    for joint in ARM_JOINTS:
        leader.bus.write("Operating_Mode", joint, OperatingMode.EXTENDED_POSITION.value)
