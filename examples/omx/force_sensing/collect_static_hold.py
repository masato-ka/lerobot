#!/usr/bin/env python3
"""Log a static, held pose on the OMX follower for a fixed duration -- for A/B comparisons like
"does tau_ext detect a small added mass" that need a clean, motion-free baseline rather than the
continuous sweeps `collect_free_motion.py` produces.

Does not move the arm: like `demo_force_sensing.py`, it pins `goal_q` to whatever pose the
follower is already holding when it connects and keeps re-sending that same target for
`--duration_sec`, logging every step via `collect_free_motion.py`'s `FreeMotionLogger` (same
`.npz` schema, directly loadable by `load_episode()`/`evaluate_free_motion.py`).

Typical A/B workflow for a small-mass detection check:
    1. Position the follower at the pose you want to test (e.g. teleoperate it there, then stop
       the teleop script -- the follower holds its last commanded position on its own). If you
       need the gripper to grasp a test mass, use `--leader_port` instead (see below) rather than
       positioning by hand and then attaching the mass by hand -- hand contact with the arm or the
       mass right before logging shows up as noise in exactly the data meant to be contact-free.
    2. Log a baseline with nothing touching the arm:
           python -m examples.omx.force_sensing.collect_static_hold \\
               --port /dev/ttyACM0 --duration_sec 15 --output data/omx_static_hold/pose1_unloaded.npz
    3. Attach the test mass to the gripper without disturbing the pose, then log again:
           python -m examples.omx.force_sensing.collect_static_hold \\
               --port /dev/ttyACM0 --duration_sec 15 --output data/omx_static_hold/pose1_loaded.npz
    4. Repeat for each pose you want to test, then compare noise floors directly:
           python -m examples.omx.force_sensing.evaluate_free_motion \\
               --checkpoint checkpoints/omx_next.pt --data data/omx_static_hold/pose1_unloaded.npz
           python -m examples.omx.force_sensing.evaluate_free_motion \\
               --checkpoint checkpoints/omx_next.pt --data data/omx_static_hold/pose1_loaded.npz
       A clear detection shows up as the loaded run's per-joint mean shifting well beyond the
       unloaded run's std (see the current force-estimation investigation in TECHNICAL_REPORT_*.md
       for what magnitude to expect).

Teleoperated grasping (`--leader_port`), for the "loaded" side of an A/B test without touching
the follower or the test mass by hand:
    python -m examples.omx.force_sensing.collect_static_hold \\
        --port /dev/ttyACM0 --leader_port /dev/ttyACM1 \\
        --duration_sec 15 --output data/omx_static_hold/pose1_loaded.npz
This connects the leader too and relays it to the follower (position + gripper) so you can
teleoperate the follower into position and grasp the test mass normally. Press Ctrl+C once
you're done teleoperating -- rather than exiting, this switches into the same hold-and-log phase
as above, pinned at whatever pose (and grasp) you left the follower in. Deliberately done within
one continuous connection rather than as two separate processes (teleop, then a fresh
`collect_static_hold` run): `OmxFollower.connect()` unconditionally cycles Torque_Enable on every
motor (including the gripper) via `configure()`, and per the Torque_Enable OFF->ON firmware quirk
documented in `leader_safety.py` (a CURRENT_POSITION-mode motor's `Goal_Position` re-locks to
wherever `Present_Position` was at that instant), reconnecting after an object is already grasped
would likely weaken the grip -- it stops trying to close past the object and just holds the
current width instead. Keeping teleoperation and logging in one connection avoids ever
reconnecting after the grasp is established.

Paired unloaded/loaded logging (`--output_loaded`), to eliminate pose drift between an A/B pair
entirely (see `compare_pose_drift.py`: two separate `collect_static_hold` runs, even both using
`--leader_port`, do NOT reliably land on the same pose -- each is an independent teleop session):
    python -m examples.omx.force_sensing.collect_static_hold \\
        --port /dev/ttyACM0 --leader_port /dev/ttyACM1 \\
        --duration_sec 15 --output data/omx_static_hold/pose1_unloaded.npz \\
        --output_loaded data/omx_static_hold/pose1_loaded.npz
Requires `--leader_port`. Teleoperate into position (gripper empty) and Ctrl+C to pin the pose,
exactly as above; this pinned pose (the 5 arm joints only) is then held FIXED -- re-sent
unchanged, never re-read -- through both phases below, so the two logs share the identical
commanded pose by construction:
    1. Hold-and-log immediately with nothing grasped -> saved to `--output`.
    2. Switch to a gripper-only relay: the arm keeps holding the same pinned pose while the
       leader's gripper command is relayed live, so you can bring the test mass up to the
       (stationary) gripper and close it via the leader -- your hand only ever touches the test
       mass, never the arm or gripper. Ctrl+C once the grasp is set.
    3. Hold-and-log again at the same pinned pose (grasp now closed) -> saved to `--output_loaded`.
"""

import argparse
import logging
import time
from pathlib import Path

from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig
from lerobot.teleoperators.omx_leader import OmxLeader, OmxLeaderConfig
from lerobot.utils.robot_utils import precise_sleep

from .collect_free_motion import FreeMotionLogger, MotorStallError, _check_motor_health
from .common import ARM_JOINTS

logger = logging.getLogger(__name__)


def _teleop_relay_until_interrupt(leader: OmxLeader, robot: OmxFollower, hz: float) -> None:
    """Relay `leader.get_action()` (all 6 motors, gripper included) to `robot.send_action()`
    at `hz` until Ctrl+C -- lets you teleoperate the follower into position and grasp a test
    mass normally. Catches `KeyboardInterrupt` itself (rather than letting it propagate) so this
    is a phase switch, not a script exit: the caller continues into hold-and-log right after.
    """
    print("Teleoperate the follower into position and grasp as needed. Press Ctrl+C when ready to hold and log.")
    dt = 1.0 / hz
    try:
        while True:
            loop_start = time.perf_counter()
            robot.send_action(leader.get_action())
            precise_sleep(max(0.0, dt - (time.perf_counter() - loop_start)))
    except KeyboardInterrupt:
        print()


def _gripper_only_relay_until_interrupt(
    leader: OmxLeader, robot: OmxFollower, hold_pose: dict[str, float], hz: float
) -> None:
    """Like `_teleop_relay_until_interrupt()`, but only the leader's gripper command is relayed --
    the 5 arm joints keep re-sending the same `hold_pose` dict passed in (never re-read), so the
    arm cannot drift while you grasp the test mass. Lets an unloaded/loaded pair share the exact
    same commanded arm pose instead of relying on two separate teleop sessions landing on the
    same pose (which `compare_pose_drift.py` showed is not reliable).
    """
    print("Bring the test mass up to the (stationary) gripper and close it via the leader. Press Ctrl+C when the grasp is set.")
    dt = 1.0 / hz
    try:
        while True:
            loop_start = time.perf_counter()
            action = {f"{j}.pos": v for j, v in hold_pose.items()}
            action["gripper.pos"] = leader.get_action()["gripper.pos"]
            robot.send_action(action)
            precise_sleep(max(0.0, dt - (time.perf_counter() - loop_start)))
    except KeyboardInterrupt:
        print()


def hold_and_log(robot: OmxFollower, log: FreeMotionLogger, hold_pose: dict[str, float], duration_sec: float, hz: float) -> None:
    dt = 1.0 / hz
    end_time = time.perf_counter() + duration_sec
    last_print = 0.0
    while time.perf_counter() < end_time:
        loop_start = time.perf_counter()
        robot.send_action({f"{j}.pos": v for j, v in hold_pose.items()})
        log.log_step(robot, hold_pose)
        if loop_start - last_print > 0.5:
            remaining = end_time - loop_start
            print(f"\rHolding pose -- {remaining:4.1f}s remaining", end="", flush=True)
            last_print = loop_start
        precise_sleep(max(0.0, dt - (time.perf_counter() - loop_start)))
    _check_motor_health(robot)
    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--robot_id", default="omx_follower")
    parser.add_argument("--output", required=True, help="Output .npz path for the logged episode")
    parser.add_argument("--duration_sec", type=float, default=15.0, help="How long to hold and log")
    parser.add_argument("--hz", type=float, default=100.0)
    parser.add_argument(
        "--leader_port",
        default=None,
        help=(
            "If set, connects an omx_leader too and relays it to the follower (position + "
            "gripper) until Ctrl+C, so you can teleoperate the follower into position and grasp "
            "a test mass before logging, instead of positioning/attaching it by hand."
        ),
    )
    parser.add_argument("--leader_id", default="omx_leader")
    parser.add_argument("--teleop_hz", type=float, default=50.0)
    parser.add_argument(
        "--output_loaded",
        default=None,
        help=(
            "If set, logs a paired unloaded/loaded A/B hold in one continuous connection at the "
            "SAME pinned pose: --output gets the unloaded hold, then a gripper-only teleop relay "
            "lets you grasp the test mass without moving the arm, then this path gets the loaded "
            "hold. Requires --leader_port."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.output_loaded is not None and args.leader_port is None:
        parser.error("--output_loaded requires --leader_port (the gripper-only relay phase needs the leader).")

    robot = OmxFollower(OmxFollowerConfig(port=args.port, id=args.robot_id))
    robot.connect(calibrate=True)

    leader = None
    if args.leader_port is not None:
        leader = OmxLeader(OmxLeaderConfig(port=args.leader_port, id=args.leader_id))
        leader.connect(calibrate=True)

    log = FreeMotionLogger()
    log_loaded = FreeMotionLogger() if args.output_loaded is not None else None
    try:
        if leader is not None:
            _teleop_relay_until_interrupt(leader, robot, args.teleop_hz)
        obs = robot.get_observation()
        hold_pose = {j: obs[f"{j}.pos"] for j in ARM_JOINTS}
        print(f"Holding current pose for {args.duration_sec}s (Ctrl+C to stop early).")
        hold_and_log(robot, log, hold_pose, args.duration_sec, args.hz)
        if log_loaded is not None:
            _gripper_only_relay_until_interrupt(leader, robot, hold_pose, args.teleop_hz)
            print(f"Holding same pose for {args.duration_sec}s (Ctrl+C to stop early).")
            hold_and_log(robot, log_loaded, hold_pose, args.duration_sec, args.hz)
    except MotorStallError:
        logger.exception("Aborting: motor overload protection tripped.")
    except KeyboardInterrupt:
        print()
    finally:
        if leader is not None:
            leader.disconnect()
        robot.disconnect()
        if len(log.t) > 0:
            log.save(Path(args.output))
        else:
            logger.warning("No samples were logged for --output; nothing saved.")
        if log_loaded is not None:
            if len(log_loaded.t) > 0:
                log_loaded.save(Path(args.output_loaded))
            else:
                logger.warning("No samples were logged for --output_loaded; nothing saved.")


if __name__ == "__main__":
    main()
