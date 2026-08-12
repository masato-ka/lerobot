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

"""Gravity-compensation torque for the omx_leader arm via Pinocchio RNEA.

Standalone helper: does not modify `OmxLeader`/`OmxLeaderConfig`. Callers (see
`examples/omx/gravity_compensation/`) read the leader's own present position and feed it
through `OmxGravityModel.compute_gravity_torque()`.

URDF source: ROBOTIS's official leader model, confirmed to match this LeRobot build's motor
IDs 1-5 (joint1..joint5, in order) plus the gripper (gripper_joint_1):
https://github.com/ROBOTIS-GIT/open_manipulator/blob/main/open_manipulator_description/urdf/omx_l/omx_l.urdf

UNVERIFIED SIGN CONVENTION: the mapping from LeRobot's normalized position (-100..100) to the
URDF's radian convention assumes each joint's raw-tick center (normalized 0) coincides with
the URDF's q=0 "assembly zero" (the MuJoCo port of the sibling follower model has an
all-zero home keyframe, consistent with this), and defaults every joint's sign to `+1`
(untested). Verify this against your own arm with
`examples/omx/gravity_compensation/preview_gravity_model.py` -- which disables motor torque
and only prints the computed values -- before ever enabling Current Control Mode.

KNOWN LIMITATION -- gravity-comp residual error at full extension: `--modifier` (see
`examples/omx/gravity_compensation/gravity_comp_demo.py`) is tuned empirically to feel
"comfortable," not calibrated to be physically exact. Confirmed on hardware
(`examples/omx/diagnose_pose.py`): for the same physical end-effector pose, the leader arm
settles at a measurably different joint configuration under active Current Control Mode than
when fully passive (as in stock `lerobot-teleop`) -- joint-limit-barrier engagement and
velocity damping were both ruled out (the barrier's margins weren't reached, and damping is ~0
once a pose is held still), leaving gravity-comp error as the steady-state cause. This residual
error is roughly constant in joint-angle terms, but its Cartesian consequence at the
end-effector scales with the arm's kinematic Jacobian -- negligible folded close to the base,
up to ~1-2cm of height error when reached out to full extension. Improving this further would
need a more precise gravity-comp calibration (e.g. better URDF mass/inertia parameters) rather
than a `--modifier` tweak; accepted as a known limitation for now.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:
    import pinocchio as pin

    _pinocchio_available = True
except ImportError:
    _pinocchio_available = False

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# LeRobot joint name -> URDF joint name. Confirmed against omx_l.urdf: joint1..joint5 form a
# single serial chain matching motor IDs 1-5 in omx_leader.py, in the same order.
URDF_JOINT_NAMES: dict[str, str] = {
    "shoulder_pan": "joint1",
    "shoulder_lift": "joint2",
    "elbow_flex": "joint3",
    "wrist_flex": "joint4",
    "wrist_roll": "joint5",
}


class OmxGravityModel:
    """Computes per-joint gravity-compensation torque for the omx_leader arm.

    Args:
        urdf_path (`str | Path`):
            Path to a local copy of `omx_l.urdf` (or an equivalent URDF using the same
            `joint1`..`joint5` naming).
        joint_sign (`dict[str, float]`, *optional*):
            Per-joint `+1.0`/`-1.0` multiplier applied when converting LeRobot's normalized
            position to the URDF's radian convention. Defaults to `+1.0` for every joint --
            unverified, see the module docstring.
        joint_offset_rad (`dict[str, float]`, *optional*):
            Per-joint radian offset added after the sign multiplication. Defaults to `0.0`.

    **Attributes**:
        - **model** (`pinocchio.Model`) -- The loaded (mesh-free) dynamics model.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        joint_sign: dict[str, float] | None = None,
        joint_offset_rad: dict[str, float] | None = None,
    ):
        if not _pinocchio_available:
            raise ImportError(
                "pinocchio is required for gravity compensation. Install it via "
                "`uv sync --extra kinematics` (pulls in pinocchio transitively through placo) "
                "or `pip install pin`."
            )
        # buildModelFromUrdf (not RobotWrapper.BuildFromURDF) loads only the dynamics model,
        # not visual/collision geometry -- so the URDF's `package://` mesh references never
        # need to resolve.
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()

        self.joint_sign = joint_sign or dict.fromkeys(ARM_JOINTS, 1.0)
        self.joint_offset_rad = joint_offset_rad or dict.fromkeys(ARM_JOINTS, 0.0)

        self._q_index: dict[str, int] = {}
        self._v_index: dict[str, int] = {}
        for joint in ARM_JOINTS:
            joint_id = self.model.getJointId(URDF_JOINT_NAMES[joint])
            self._q_index[joint] = self.model.joints[joint_id].idx_q
            self._v_index[joint] = self.model.joints[joint_id].idx_v

    def _normalized_to_radians(self, joint: str, norm_pct: float) -> float:
        return self.joint_sign[joint] * (norm_pct / 100.0) * math.pi + self.joint_offset_rad[joint]

    def compute_gravity_torque(self, q_lerobot: dict[str, float]) -> dict[str, float]:
        """Compute the gravity-compensation torque for a given leader arm pose.

        Args:
            q_lerobot (`dict[str, float]`):
                Present position per arm joint, keyed by bare joint name (`ARM_JOINTS`), in
                LeRobot's normalized `RANGE_M100_100` units (i.e. `Robot.get_observation()`'s
                `f"{joint}.pos"` values, without the `.pos` suffix). The gripper is not
                included -- it keeps its own current-based grasp-force control.

        Returns:
            `dict[str, float]`: Gravity-compensation torque in Nm per arm joint, using RNEA's
            sign convention for the mapped `joint_sign`/`joint_offset_rad` (see the class
            docstring for the caveat on this mapping being unverified by default).
        """
        q = pin.neutral(self.model)
        for joint in ARM_JOINTS:
            q[self._q_index[joint]] = self._normalized_to_radians(joint, q_lerobot[joint])

        zero_v = np.zeros(self.model.nv)
        tau = pin.rnea(self.model, self.data, q, zero_v, zero_v)

        return {joint: float(tau[self._v_index[joint]]) for joint in ARM_JOINTS}
