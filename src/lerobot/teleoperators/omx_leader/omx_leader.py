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

import logging
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DriveMode,
    DynamixelMotorsBus,
    OperatingMode,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_omx_leader import OmxLeaderConfig
from .gravity_compensation import ARM_JOINTS, OmxGravityModel
from .leader_safety import (
    DEFAULT_JOINT_MODIFIER_OVERRIDES,
    JOINT_LIMIT_RANGE,
    KT_NM_PER_A,
    compute_damping_torque,
    compute_feedback_torque,
    compute_joint_limit_torque,
    enter_current_control_mode,
    resolve_per_joint_from_config,
    restore_position_mode,
)

logger = logging.getLogger(__name__)


class OmxLeader(Teleoperator):
    """
    - [OMX](https://github.com/ROBOTIS-GIT/open_manipulator),
        expansion, developed by Woojin Wie and Junha Cha from [ROBOTIS](https://ai.robotis.com/)
    """

    config_class = OmxLeaderConfig
    name = "omx_leader"

    def __init__(self, config: OmxLeaderConfig):
        super().__init__(config)
        self.config = config
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "xl330-m288", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "xl330-m288", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "xl330-m288", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "xl330-m288", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "xl330-m288", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "xl330-m077", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        self._force_feedback_enabled = bool(config.force_feedback.urdf_path)
        self._gravity_model: OmxGravityModel | None = None
        if self._force_feedback_enabled:
            ff = config.force_feedback
            self._gravity_model = OmxGravityModel(ff.urdf_path)
            self._modifier = resolve_per_joint_from_config(
                ff.modifier, ff.modifier_overrides, DEFAULT_JOINT_MODIFIER_OVERRIDES
            )
            self._damping_gain = resolve_per_joint_from_config(ff.damping_gain, ff.damping_gain_overrides)
            self._joint_limit_kp = resolve_per_joint_from_config(ff.joint_limit_kp, ff.joint_limit_kp_overrides)
            self._joint_limit_kd = resolve_per_joint_from_config(ff.joint_limit_kd, ff.joint_limit_kd_overrides)
            self._feedback_gain = resolve_per_joint_from_config(ff.feedback_gain, ff.feedback_gain_overrides)

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        if self._force_feedback_enabled:
            return {f"force.{j}": float for j in ARM_JOINTS}
        return {}

    @property
    def wants_continuous_feedback(self) -> bool:
        return self._force_feedback_enabled

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()

        if self._force_feedback_enabled:
            enter_current_control_mode(self, self.config.force_feedback.current_limit_ma)

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        self.bus.disable_torque()
        logger.info(f"\nUsing factory default calibration values for {self}")
        logger.info(f"\nWriting default configuration of {self} to the motors")
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        for motor in self.bus.motors:
            if motor == "gripper":
                self.bus.write("Drive_Mode", motor, DriveMode.INVERTED.value)
            else:
                self.bus.write("Drive_Mode", motor, DriveMode.NON_INVERTED.value)
        drive_modes = {motor: 1 if motor == "gripper" else 0 for motor in self.bus.motors}

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=drive_modes[motor],
                homing_offset=0 if motor != "gripper" else 100,
                range_min=0,
                range_max=4095,
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            if motor != "gripper":
                # Use 'extended position mode' for all motors except gripper, because in joint mode the servos
                # can't rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while
                # assembling the arm, you could end up with a servo with a position 0 or 4095 at a crucial
                # point
                self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

            if motor == "gripper":
                self.bus.write("Drive_Mode", motor, DriveMode.INVERTED.value)
            else:
                self.bus.write("Drive_Mode", motor, DriveMode.NON_INVERTED.value)

        # Use 'position control current based' for gripper to be limited by the limit of the current.
        # For the follower gripper, it means it can grasp an object without forcing too much even tho,
        # its goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
        # For the leader gripper, it means we can use it as a physical trigger, since we can force with our finger
        # to make it move, and it will move back to its original target position when we release the force.
        self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)
        self.bus.write("Current_Limit", "gripper", 100)
        self.bus.write("Goal_Current", "gripper", 100)
        self.bus.write("Homing_Offset", "gripper", 100)
        # Set gripper's goal pos in current position mode so that we can use it as a trigger.
        self.bus.enable_torque("gripper")
        if self.is_calibrated:
            self.bus.write("Goal_Position", "gripper", self.config.gripper_open_pos)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Gravity comp + joint-limit barrier + velocity damping + force-feedback current injection, only
        when `force_feedback` is configured (a no-op otherwise, matching the previous behavior for anyone
        not opting in). Ported verbatim from
        `examples/omx/bilateral_teleop/bilateral_teleop_demo.py`'s per-tick combine/clip order -- see that
        script's history for why each step is ordered/scaled/clipped the way it is.

        Args:
            feedback (`dict[str, float]`): `force.<joint>` (the follower's NEXT `tau_ext` estimate) per
                `ARM_JOINTS`. Missing keys (e.g. force estimation not enabled on the follower) default to
                `0.0`, degrading gracefully to gravity+limit+damping only -- the same degradation already
                used while the estimator's history buffer is still filling.
        """
        if not self._force_feedback_enabled:
            return

        q = self.bus.sync_read("Present_Position")
        qdot = self.bus.sync_read("Present_Velocity")
        tau_ext = {joint: feedback.get(f"force.{joint}", 0.0) for joint in ARM_JOINTS}

        tau_g = self._gravity_model.compute_gravity_torque(q)
        tau_limit = compute_joint_limit_torque(q, qdot, JOINT_LIMIT_RANGE, self._joint_limit_kp, self._joint_limit_kd)
        tau_damping = compute_damping_torque(qdot, self._damping_gain)
        ff = self.config.force_feedback
        feedback_ma = compute_feedback_torque(tau_ext, self._feedback_gain, ff.feedback_limit_ma)

        goal_current_ma: dict[str, int] = {}
        for joint in ARM_JOINTS:
            gravity_ma = (tau_g[joint] / KT_NM_PER_A) * self._modifier[joint] * 1000.0
            total_ma = gravity_ma + tau_limit[joint] + tau_damping[joint] + feedback_ma[joint]
            total_ma = max(-ff.current_limit_ma, min(ff.current_limit_ma, total_ma))
            goal_current_ma[joint] = int(total_ma)

        self.bus.sync_write("Goal_Current", goal_current_ma)

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._force_feedback_enabled:
            restore_position_mode(self)
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
