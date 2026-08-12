#!/usr/bin/env python3
"""Compare the leader's and follower's per-joint normalized positions, live, to investigate a
reported end-effector height discrepancy between our bilateral scripts and stock `lerobot-teleop`.

Report: with stock `lerobot-teleop`, posing the leader so its gripper touches the ground/table
makes the follower's gripper match. Doing the exact same pose with `bilateral_teleop_demo.py`/
`record_bilateral.py` leaves the follower's end-effector ~2cm higher. Confirmed by the user: no
felt extra resistance on the leader, the pose was held for several seconds before comparing (not
a one-shot glance), and the discrepancy only shows up when the arm is reached far forward
(`shoulder_pan > 0` and `shoulder_lift`/`elbow_flex < 0`) -- not when working close to the base.
That "fine near the base, worse at full extension" pattern is the signature of a small,
roughly-constant joint-angle error that gets amplified by the lever arm at extension, rather
than a joint-limit-barrier-engagement or simple loop-lag issue (both of which were considered
and are weakened, though not fully ruled out, by the "no felt resistance" / "held for seconds"
answers -- see the Phase 7 plan for the full reasoning).

Since the position-relay code itself was already confirmed identical between the bilateral
scripts and stock `lerobot-teleop` (same `sync_read("Present_Position")` ->
`send_action()` shape, established in an earlier investigation), this script isolates the one
real behavioral difference -- whether the leader's arm joints are passive (stock teleop) or
under active Current Control Mode (gravity comp + joint-limit barrier + damping, our scripts) --
and prints a live per-joint leader-vs-follower table so the two poses described above (near the
base vs. reached far forward) can be compared directly, with and without active control.

No NEXT checkpoint / force estimation is used here (this is about position tracking, not force
feedback), so `--checkpoint` is not required, unlike `bilateral_teleop_demo.py`.

Usage (run from repo root):
    # Passive relay (leader arm fully backdrivable, same as stock lerobot-teleop):
    python -m examples.omx.diagnose_pose \\
        --follower_port /dev/ttyACM0 --leader_port /dev/ttyACM1 \\
        --urdf_path /path/to/omx_l.urdf --passive_relay

    # Active control (same gravity comp/joint-limit/damping as bilateral_teleop_demo.py):
    python -m examples.omx.diagnose_pose \\
        --follower_port /dev/ttyACM0 --leader_port /dev/ttyACM1 \\
        --urdf_path /path/to/omx_l.urdf \\
        --modifier 0.09 --modifier_shoulder_lift 0.1 \\
        --modifier_shoulder_pan 0.0 --modifier_wrist_roll 0.0 \\
        --damping_gain 0.05 --joint_limit_kp 3 --joint_limit_kd 0

Hold each pose (near the base, and reached far forward with shoulder_pan > 0 /
shoulder_lift or elbow_flex < 0) for a few seconds under both modes and compare the printed
leader/follower/diff table.

CONCLUSION (confirmed on hardware): the `leader`/`follower`/`diff` table showed `diff` staying
small (roughly -1 to +1) in all four combinations (near base / extended, passive / active) --
the follower tracks whatever the leader is currently reporting accurately. The `leader` column
itself, however, differed by several normalized units between the passive and active runs for
the *same* physical pose (gripper resting on the same block), especially at `shoulder_lift`/
`elbow_flex`/`wrist_flex` in the extended pose. So the discrepancy is not a follower-tracking
problem -- it's that the leader itself settles at a measurably different joint configuration
under active Current Control Mode than fully passive, for what a human intends as the same
pose. Joint-limit barrier and velocity damping were both ruled out as the steady-state cause
(the observed joint values aren't near `JOINT_LIMIT_RANGE`'s margins, and `qdot ≈ 0` once a pose
is held still for a few seconds, so `compute_damping_torque()`'s output is ~0 at the moment of
comparison). That leaves gravity compensation -- the one term that stays nonzero at rest -- as
the leading explanation: `OmxGravityModel`'s `--modifier` gain is an empirically "comfortable"
value (see `gravity_comp_demo.py`'s SAFETY notes), not a precise physical calibration, and any
residual error compounds with the arm's kinematic Jacobian (the same joint-angle error maps to a
much larger Cartesian displacement when the arm is extended than folded near the base) plus the
fact that required gravity torque itself is larger when extended, so a proportional gain error
also has more to act on. Net effect confirmed on hardware: up to ~1-2cm of end-effector height
error at full extension, negligible near the base. Accepted as a known limitation rather than
pursued further (see `src/lerobot/teleoperators/omx_leader/gravity_compensation.py` and the
bilateral script docstrings for the user-facing note) -- improving it further would need a more
precise gravity-comp calibration (e.g. better URDF mass/inertia parameters), which is out of
scope for now.
"""

import argparse
import logging
import time

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

ALL_JOINTS = [*ARM_JOINTS, "gripper"]


def add_gravity_control_args(parser: argparse.ArgumentParser) -> None:
    """Same gravity/damping/joint-limit CLI surface as gravity_comp_demo.py, no force feedback."""
    parser.add_argument("--modifier", type=float, default=0.09, help="Default gravity-comp gain")
    parser.add_argument("--damping_gain", type=float, default=0.05, help="Default velocity damping gain")
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--follower_port", default="/dev/ttyACM0")
    parser.add_argument("--follower_id", default="omx_follower")
    parser.add_argument("--leader_port", default="/dev/ttyACM1")
    parser.add_argument("--leader_id", default="omx_leader")
    parser.add_argument("--urdf_path", required=True, help="Path to a local copy of omx_l.urdf")
    add_gravity_control_args(parser)
    parser.add_argument("--current_limit_ma", type=int, default=500, help="Hard per-joint current cap")
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument(
        "--passive_relay",
        action="store_true",
        help="Leave the leader's arm joints fully passive (torque off), matching stock "
        "lerobot-teleop, instead of enabling Current Control Mode (gravity comp + "
        "joint-limit barrier + damping)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    modifiers = resolve_modifiers(args)
    damping_gains = resolve_per_joint(args, "damping_gain", args.damping_gain)
    joint_limit_kp = resolve_per_joint(args, "joint_limit_kp", args.joint_limit_kp)
    joint_limit_kd = resolve_per_joint(args, "joint_limit_kd", args.joint_limit_kd)

    follower = OmxFollower(OmxFollowerConfig(port=args.follower_port, id=args.follower_id))
    leader = OmxLeader(OmxLeaderConfig(port=args.leader_port, id=args.leader_id))
    follower.connect(calibrate=True)
    leader.connect(calibrate=True)

    gravity_model = OmxGravityModel(args.urdf_path) if not args.passive_relay else None

    dt = 1.0 / args.hz
    active_control_entered = False
    try:
        if not args.passive_relay:
            enter_current_control_mode(leader, args.current_limit_ma)
            active_control_entered = True
            print(
                f"Current Control Mode enabled (current_limit={args.current_limit_ma}mA).\n"
                f"modifier: {' '.join(f'{j}={modifiers[j]}' for j in ARM_JOINTS)}\n"
            )
        else:
            print("Passive relay: leader arm joints left torque-off (same as stock lerobot-teleop).\n")
        print("Hold a pose steady for a few seconds and read the table below. Ctrl+C to stop.\n")

        loop_count = 0
        rate_check_start = time.perf_counter()
        while True:
            loop_start = time.perf_counter()

            leader_pos = leader.bus.sync_read("Present_Position")
            leader_vel = leader.bus.sync_read("Present_Velocity")
            q_leader = {j: leader_pos[j] for j in ARM_JOINTS}
            qdot_leader = {j: leader_vel[j] for j in ARM_JOINTS}
            follower_action = {f"{j}.pos": q_leader[j] for j in ARM_JOINTS}
            follower_action["gripper.pos"] = leader_pos["gripper"]
            follower.send_action(follower_action)

            follower_pos = follower.bus.sync_read("Present_Position")

            if not args.passive_relay:
                tau_g = gravity_model.compute_gravity_torque(q_leader)
                tau_limit_ma = compute_joint_limit_torque(
                    q_leader, qdot_leader, JOINT_LIMIT_RANGE, joint_limit_kp, joint_limit_kd
                )
                tau_damping_ma = compute_damping_torque(qdot_leader, damping_gains)
                goal_current_ma = {}
                for joint in ARM_JOINTS:
                    gravity_ma = (tau_g[joint] / KT_NM_PER_A) * modifiers[joint] * 1000.0
                    total_ma = gravity_ma + tau_limit_ma[joint] + tau_damping_ma[joint]
                    total_ma = max(-args.current_limit_ma, min(args.current_limit_ma, total_ma))
                    goal_current_ma[joint] = int(total_ma)
                leader.bus.sync_write("Goal_Current", goal_current_ma)

            loop_count += 1
            elapsed_total = time.perf_counter() - rate_check_start
            if elapsed_total >= 1.0:
                actual_hz = loop_count / elapsed_total
                print(f"[{actual_hz:5.1f}Hz] {'joint':<15}{'leader':>10}{'follower':>10}{'diff':>10}")
                for j in ALL_JOINTS:
                    diff = follower_pos[j] - leader_pos[j]
                    print(f"          {j:<15}{leader_pos[j]:>10.1f}{follower_pos[j]:>10.1f}{diff:>+10.1f}")
                print()
                loop_count = 0
                rate_check_start = time.perf_counter()

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, dt - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        print()
        try:
            if active_control_entered:
                restore_position_mode(leader)
        finally:
            leader.disconnect()
            follower.disconnect()


if __name__ == "__main__":
    main()
