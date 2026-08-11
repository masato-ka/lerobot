#!/usr/bin/env python3
"""Bilateral teleoperation for OMX: position teleop (leader -> follower) plus FACTR2-style
force feedback (follower's estimated external torque -> leader).

Connects `omx_follower` and `omx_leader` in one process (both on the same PC) and runs a
single combined loop:
  1. Read the leader's position and command it to the follower (ordinary position teleop).
  2. Read the follower's state and feed it through a trained NEXT checkpoint
     (`OnlineExternalTorqueEstimator`, see `examples/omx/force_sensing/`) to get `tau_ext`.
  3. Read the leader's own state and compute gravity compensation + joint-limit barrier +
     damping (same as `gravity_comp_demo.py`, via `leader_safety.py`/`gravity_compensation.py`).
  4. `tau_feedback = feedback_gain[joint] * tau_ext[joint]`, clipped to `--feedback_limit_ma`
     independently of the overall `--current_limit_ma`.
  5. Sum all leader torque terms, convert/clip to mA, write `Goal_Current`.

`--feedback_gain` defaults to `-0.2` (see below for why it's negative) -- this is a confirmed
working value on one unit, not a "disabled by default" safety fallback like the other gains
here. Pass `--feedback_gain 0.0` explicitly if you want the pre-feedback, `gravity_comp_demo.py`-
equivalent behavior (e.g. to isolate a regression) instead of the tuned default.

`tau_ext` uses the follower's raw per-joint sensing units (see
src/lerobot/force_estimation/README.md), not Nm, and its sign relative to "which direction the
leader should push back" is unverified in general -- `feedback_gain` is deliberately allowed to
be negative so a backwards joint can just have its sign flipped during tuning. Confirmed on
hardware: `tau_ext`'s sign is opposite the intuitive "push back the same way" direction, so a
*negative* `--feedback_gain` is what actually renders correctly here -- both leader and
follower have every arm joint's `Drive_Mode` set the same way (`NON_INVERTED`), so this flip is
expected to be uniform across joints rather than needing a different sign per joint, but verify
per joint if some feel backwards after the global flip.

Usage (run from repo root; showing the confirmed defaults explicitly -- they apply even if
omitted):
    python -m examples.omx.bilateral_teleop.bilateral_teleop_demo \\
        --follower_port /dev/ttyACM0 --follower_id omx_follower \\
        --leader_port /dev/ttyACM1 --leader_id omx_leader \\
        --urdf_path /path/to/omx_l.urdf --checkpoint checkpoints/omx_next.pt \\
        --modifier 0.09 --modifier_shoulder_lift 0.1 \\
        --modifier_shoulder_pan 0.0 --modifier_wrist_roll 0.0 \\
        --damping_gain 0.05 --joint_limit_kp 3 --joint_limit_kd 0 \\
        --feedback_gain -0.2
"""

import argparse
import logging
import time

from lerobot.force_estimation import OnlineExternalTorqueEstimator
from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig
from lerobot.teleoperators.omx_leader import OmxLeader, OmxLeaderConfig
from lerobot.teleoperators.omx_leader.gravity_compensation import ARM_JOINTS, OmxGravityModel
from lerobot.teleoperators.omx_leader.leader_safety import (
    JOINT_LIMIT_RANGE,
    KT_NM_PER_A,
    compute_damping_torque,
    compute_joint_limit_torque,
    enter_current_control_mode,
    resolve_modifiers,
    resolve_per_joint,
    restore_position_mode,
)

logger = logging.getLogger(__name__)


def add_leader_control_args(parser: argparse.ArgumentParser) -> None:
    """Same gravity/damping/joint-limit CLI surface as gravity_comp_demo.py, plus feedback."""
    parser.add_argument("--modifier", type=float, default=0.09, help="Default gravity-comp gain")
    parser.add_argument("--damping_gain", type=float, default=0.05, help="Default velocity damping gain")
    parser.add_argument(
        "--joint_limit_kp", type=float, default=3.0, help="Default joint-limit barrier P gain"
    )
    parser.add_argument(
        "--joint_limit_kd", type=float, default=0.0, help="Default joint-limit barrier D gain"
    )
    parser.add_argument("--feedback_gain", type=float, default=-0.2, help="Default force-feedback gain")
    for prefix in ("modifier", "damping_gain", "joint_limit_kp", "joint_limit_kd", "feedback_gain"):
        for joint in ARM_JOINTS:
            parser.add_argument(
                f"--{prefix}_{joint}",
                type=float,
                default=None,
                help=f"{prefix} override for {joint} (defaults to --{prefix})",
            )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--follower_port", default="/dev/ttyACM0")
    parser.add_argument("--follower_id", default="omx_follower")
    parser.add_argument("--leader_port", default="/dev/ttyACM1")
    parser.add_argument("--leader_id", default="omx_leader")
    parser.add_argument("--urdf_path", required=True, help="Path to a local copy of omx_l.urdf")
    parser.add_argument("--checkpoint", required=True, help="Path to a NEXT checkpoint from train_next.py")
    add_leader_control_args(parser)
    parser.add_argument("--current_limit_ma", type=int, default=500, help="Hard per-joint current cap")
    parser.add_argument(
        "--feedback_limit_ma", type=int, default=200, help="Hard per-joint cap on the feedback term alone"
    )
    parser.add_argument("--hz", type=float, default=50.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    modifiers = resolve_modifiers(args)
    damping_gains = resolve_per_joint(args, "damping_gain", args.damping_gain)
    joint_limit_kp = resolve_per_joint(args, "joint_limit_kp", args.joint_limit_kp)
    joint_limit_kd = resolve_per_joint(args, "joint_limit_kd", args.joint_limit_kd)
    feedback_gains = resolve_per_joint(args, "feedback_gain", args.feedback_gain)

    follower = OmxFollower(OmxFollowerConfig(port=args.follower_port, id=args.follower_id))
    leader = OmxLeader(OmxLeaderConfig(port=args.leader_port, id=args.leader_id))
    follower.connect(calibrate=True)
    leader.connect(calibrate=True)

    gravity_model = OmxGravityModel(args.urdf_path)
    estimator = OnlineExternalTorqueEstimator(args.checkpoint)

    dt = 1.0 / args.hz
    zero_tau_ext = dict.fromkeys(ARM_JOINTS, 0.0)
    try:
        enter_current_control_mode(leader, args.current_limit_ma)
        print(
            f"Current Control Mode enabled (current_limit={args.current_limit_ma}mA, "
            f"feedback_limit={args.feedback_limit_ma}mA).\n"
            f"feedback_gain: {' '.join(f'{j}={feedback_gains[j]}' for j in ARM_JOINTS)}\n"
            "Ctrl+C to stop.\n"
        )
        loop_count = 0
        rate_check_start = time.perf_counter()
        while True:
            loop_start = time.perf_counter()

            # 1. Leader position -> follower (position teleop). Includes the gripper --
            # ARM_JOINTS is arm-only (used for the force/safety terms below), but the
            # follower's gripper still needs to track the leader's trigger position.
            leader_pos = leader.bus.sync_read("Present_Position")
            leader_vel = leader.bus.sync_read("Present_Velocity")
            q_leader = {j: leader_pos[j] for j in ARM_JOINTS}
            qdot_leader = {j: leader_vel[j] for j in ARM_JOINTS}
            follower_action = {f"{j}.pos": q_leader[j] for j in ARM_JOINTS}
            follower_action["gripper.pos"] = leader_pos["gripper"]
            follower.send_action(follower_action)

            # 2. Follower state -> tau_ext (goal_q is what we just commanded above).
            follower_pos = follower.bus.sync_read("Present_Position")
            follower_vel = follower.bus.sync_read("Present_Velocity")
            follower_cur = follower.bus.sync_read("Present_Current")
            tau_ext = estimator.update(
                q={j: follower_pos[j] for j in ARM_JOINTS},
                qdot={j: follower_vel[j] for j in ARM_JOINTS},
                goal_q=q_leader,
                current={j: follower_cur[j] for j in ARM_JOINTS},
            )
            if tau_ext is None:  # history buffer still filling (first history_length steps)
                tau_ext = zero_tau_ext

            # 3. Leader gravity + joint-limit + damping (same as gravity_comp_demo.py).
            tau_g = gravity_model.compute_gravity_torque(q_leader)
            tau_limit_ma = compute_joint_limit_torque(
                q_leader, qdot_leader, JOINT_LIMIT_RANGE, joint_limit_kp, joint_limit_kd
            )
            tau_damping_ma = compute_damping_torque(qdot_leader, damping_gains)

            # 4-5. Combine, including the clipped force-feedback term, and write.
            goal_current_ma = {}
            for joint in ARM_JOINTS:
                gravity_ma = (tau_g[joint] / KT_NM_PER_A) * modifiers[joint] * 1000.0
                feedback_ma = feedback_gains[joint] * tau_ext[joint]
                feedback_ma = max(-args.feedback_limit_ma, min(args.feedback_limit_ma, feedback_ma))
                total_ma = gravity_ma + tau_limit_ma[joint] + tau_damping_ma[joint] + feedback_ma
                total_ma = max(-args.current_limit_ma, min(args.current_limit_ma, total_ma))
                goal_current_ma[joint] = int(total_ma)
            leader.bus.sync_write("Goal_Current", goal_current_ma)

            loop_count += 1
            elapsed_total = time.perf_counter() - rate_check_start
            if elapsed_total >= 1.0:
                # Scrolling (not overwritten) on purpose: push on the follower and scroll back
                # to see whether tau_ext actually moved, and by how much relative to the noise
                # floor (~15, see src/lerobot/force_estimation/README.md) and --feedback_limit_ma.
                actual_hz = loop_count / elapsed_total
                tau_ext_row = "  ".join(f"{j}={tau_ext[j]:+7.1f}" for j in ARM_JOINTS)
                current_row = "  ".join(f"{j}={goal_current_ma[j]:+5d}mA" for j in ARM_JOINTS)
                print(f"[{actual_hz:5.1f}Hz] tau_ext      {tau_ext_row}")
                print(f"          goal_current {current_row}")
                loop_count = 0
                rate_check_start = time.perf_counter()

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, dt - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        print()
        try:
            restore_position_mode(leader)
        finally:
            leader.disconnect()
            follower.disconnect()


if __name__ == "__main__":
    main()
