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

"""Config/property-level tests for OmxLeader's force-feedback integration -- no hardware connection
needed (constructing `OmxLeader`/`DynamixelMotorsBus` doesn't open the serial port; that only happens on
`.connect()`)."""

from pathlib import Path

import pytest

from lerobot.teleoperators.omx_leader.config_omx_leader import (
    OmxLeaderConfig,
    OmxLeaderForceFeedbackConfig,
)
from lerobot.teleoperators.omx_leader.omx_leader import OmxLeader

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

_MINIMAL_URDF = """<?xml version="1.0"?>
<robot name="omx_test">
  <link name="link0">
    <inertial><mass value="0.1"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
  </link>
  {joints_and_links}
</robot>
"""


def _make_minimal_urdf(tmp_path: Path) -> Path:
    """A minimal 5-revolute-joint serial chain named joint1..joint5 -- the only thing
    `OmxGravityModel` needs (it doesn't load meshes, just the dynamics model)."""
    parts = []
    for i in range(1, 6):
        parts.append(f"""
  <link name="link{i}">
    <inertial><mass value="0.1"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
  </link>
  <joint name="joint{i}" type="revolute">
    <parent link="link{i - 1}"/>
    <child link="link{i}"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
""")
    urdf_path = tmp_path / "omx_test.urdf"
    urdf_path.write_text(_MINIMAL_URDF.format(joints_and_links="".join(parts)))
    return urdf_path


def test_default_config_disables_force_feedback():
    config = OmxLeaderConfig(port="/dev/fake")
    assert config.force_feedback.urdf_path == ""
    assert config.force_feedback.feedback_gain == 0.0

    leader = OmxLeader(config)
    assert leader.feedback_features == {}
    assert leader.wants_continuous_feedback is False


def test_force_feedback_enabled_via_urdf_path(tmp_path):
    urdf_path = _make_minimal_urdf(tmp_path)
    config = OmxLeaderConfig(
        port="/dev/fake",
        force_feedback=OmxLeaderForceFeedbackConfig(urdf_path=str(urdf_path), feedback_gain=0.3),
    )
    leader = OmxLeader(config)

    assert leader.wants_continuous_feedback is True
    assert leader.feedback_features == dict.fromkeys([f"force.{j}" for j in ARM_JOINTS], float)
    assert leader._gravity_model is not None


def test_force_feedback_urdf_path_required_for_gravity_model():
    """Constructing with a non-empty urdf_path that doesn't exist should fail at construction time
    (not silently defer to first use), consistent with `OmxGravityModel.__init__` doing the URDF parse
    immediately."""
    config = OmxLeaderConfig(
        port="/dev/fake",
        force_feedback=OmxLeaderForceFeedbackConfig(urdf_path="/nonexistent/omx_l.urdf"),
    )
    with pytest.raises(Exception):  # noqa: B017 -- pinocchio raises its own exception type for a bad path
        OmxLeader(config)
