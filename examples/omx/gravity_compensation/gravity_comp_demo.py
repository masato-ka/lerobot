#!/usr/bin/env python3
"""Active gravity compensation for the omx_leader arm, with a joint-limit barrier and damping.

Switches the arm's 5 joints (gripper excluded) to Dynamixel Current Control Mode and
continuously commands `Goal_Current` from three combined terms:
  - RNEA-computed gravity torque (`OmxGravityModel`), scaled by a nominal XL330 torque
    constant and a tunable `--modifier` gain -- so the arm stays wherever it's placed instead
    of falling under its own weight.
  - A soft joint-limit barrier (FACTR-style): a repulsive current that grows as a joint
    approaches/exceeds `JOINT_LIMIT_RANGE`, zero well inside it. Necessary because Current
    Control Mode has no firmware position loop -- `Min/Max_Position_Limit` are NOT enforced in
    this mode, so without this term nothing stops a joint from being driven to its mechanical
    end stop.
  - A velocity damping term (`-damping_gain * qdot`), to curb oscillation.

SAFETY -- read before running:
  1. Run `preview_gravity_model.py` first and confirm the printed torques/angles look
     physically sensible for your arm. This script trusts that mapping blindly.
  2. `JOINT_LIMIT_RANGE` below was measured with `find_leader_joint_range.py` on one specific
     leader unit. If you're running a different physical arm, re-measure and replace it before
     relying on the barrier for real protection.
  3. Support the arm by hand before starting. `--modifier` defaults to a conservative 0.09
     (per-joint requirements vary a lot -- e.g. shoulder_lift typically needs noticeably more
     than the rest, wrist_roll needs close to none); increase gradually while feeling whether
     the arm gets lighter, rather than jumping to a large value. Same approach for
     `--damping_gain`/`--joint_limit_kp`/`--joint_limit_kd`: all default small, tune upward.
     Note that `--damping_gain` applies across the *entire* range (not just near the joint
     limits), so a value that's too high makes manual operation feel uniformly heavy rather
     than just damping motion near the limits -- if that happens, turn it down before touching
     the joint-limit gains. The joint-limit barrier itself can ring/oscillate right at the
     wall if `--joint_limit_kp` is too high for this loop's rate + Dynamixel bus latency;
     lower `kp` first, and only add `--joint_limit_kd` back in if the wall still feels too
     bouncy once `kp` is reasonable.
  4. `--current_limit_ma` is a hard per-joint ceiling independent of all the gains above, in
     case a gain/sign mapping is wrong. Keep it conservative until you've validated behavior.
  5. Ctrl+C (or any exception) always disables torque and restores the arm to
     EXTENDED_POSITION mode before exiting.

Per-joint tuning: every gain (`--modifier`, `--damping_gain`, `--joint_limit_kp`,
`--joint_limit_kd`) has a `--<name>_<joint>` override (e.g. `--modifier_shoulder_lift 0.15`)
that falls back to the global `--<name>` value when not given. `wrist_roll`'s `--modifier`
defaults to `0.0` (its gravity torque is close to zero at every pose, and any nonzero current
there tends to just get in the way of manual operation) -- pass `--modifier_wrist_roll`
explicitly if you want it compensated too.

Usage (run from repo root; defaults below are the values confirmed comfortable -- oscillation-
free at the joint limits, no perceptible extra weight during normal operation -- on one unit):
    python -m examples.omx.gravity_compensation.gravity_comp_demo \\
        --port /dev/ttyACM1 --robot_id omx_leader \\
        --urdf_path /path/to/omx_l.urdf \\
        --modifier 0.09 --modifier_shoulder_lift 0.15 \\
        --damping_gain 0.15 --joint_limit_kp 3 --joint_limit_kd 0
"""

import argparse
import logging
import time

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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", default="/dev/ttyACM1")
    parser.add_argument("--robot_id", default="omx_leader")
    parser.add_argument("--urdf_path", required=True, help="Path to a local copy of omx_l.urdf")
    parser.add_argument("--modifier", type=float, default=0.09, help="Default gravity-comp gain")
    parser.add_argument("--damping_gain", type=float, default=0.15, help="Default velocity damping gain")
    parser.add_argument(
        "--joint_limit_kp", type=float, default=3.0, help="Default joint-limit barrier P gain"
    )
    parser.add_argument(
        "--joint_limit_kd", type=float, default=0.0, help="Default joint-limit barrier D gain"
    )
    for prefix in ("modifier", "damping_gain", "joint_limit_kp", "joint_limit_kd"):
        for joint in ARM_JOINTS:
            parser.add_argument(
                f"--{prefix}_{joint}",
                type=float,
                default=None,
                help=f"{prefix} override for {joint} (defaults to --{prefix})",
            )
    parser.add_argument("--current_limit_ma", type=int, default=500, help="Hard per-joint current cap")
    parser.add_argument("--hz", type=float, default=50.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    modifiers = resolve_modifiers(args)
    damping_gains = resolve_per_joint(args, "damping_gain", args.damping_gain)
    joint_limit_kp = resolve_per_joint(args, "joint_limit_kp", args.joint_limit_kp)
    joint_limit_kd = resolve_per_joint(args, "joint_limit_kd", args.joint_limit_kd)

    leader = OmxLeader(OmxLeaderConfig(port=args.port, id=args.robot_id))
    leader.connect(calibrate=True)
    gravity_model = OmxGravityModel(args.urdf_path)

    dt = 1.0 / args.hz
    try:
        enter_current_control_mode(leader, args.current_limit_ma)
        modifier_row = "  ".join(f"{j}={modifiers[j]}" for j in ARM_JOINTS)
        limit_row = "  ".join(f"{j}={JOINT_LIMIT_RANGE[j]}" for j in ARM_JOINTS)
        print(
            f"Current Control Mode enabled (current_limit={args.current_limit_ma}mA).\n"
            f"modifiers: {modifier_row}\n"
            f"joint_limit_range: {limit_row}\n"
            "Ctrl+C to stop.\n"
        )
        while True:
            pos = leader.bus.sync_read("Present_Position")
            vel = leader.bus.sync_read("Present_Velocity")
            q_lerobot = {j: pos[j] for j in ARM_JOINTS}
            qdot_lerobot = {j: vel[j] for j in ARM_JOINTS}

            tau_g = gravity_model.compute_gravity_torque(q_lerobot)
            tau_limit_ma = compute_joint_limit_torque(
                q_lerobot, qdot_lerobot, JOINT_LIMIT_RANGE, joint_limit_kp, joint_limit_kd
            )
            tau_damping_ma = compute_damping_torque(qdot_lerobot, damping_gains)

            goal_current_ma = {}
            for joint in ARM_JOINTS:
                gravity_ma = (tau_g[joint] / KT_NM_PER_A) * modifiers[joint] * 1000.0
                total_ma = gravity_ma + tau_limit_ma[joint] + tau_damping_ma[joint]
                total_ma = max(-args.current_limit_ma, min(args.current_limit_ma, total_ma))
                goal_current_ma[joint] = int(total_ma)
            leader.bus.sync_write("Goal_Current", goal_current_ma)

            row = "  ".join(f"{j}={goal_current_ma[j]:+5d}mA" for j in ARM_JOINTS)
            print(f"\rgoal_current {row}", end="", flush=True)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        try:
            restore_position_mode(leader)
        finally:
            leader.disconnect()


if __name__ == "__main__":
    main()
