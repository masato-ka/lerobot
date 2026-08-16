#!/usr/bin/env python3
"""Collect free-motion (contact-free) training data for NEXT by teleoperating the follower
directly, instead of `collect_free_motion.py`'s scripted per-axis/grid sweeps.

Motivation: `check_training_coverage.py` showed that `collect_free_motion.py`'s swept ranges
(`SHOULDER_LIFT_RANGE`, `safe_elbow_flex_range`) leave large regions of `shoulder_lift`/
`elbow_flex` space completely uncovered (0 training samples), and that even where a test pose
*is* covered, it may still not match how the arm actually moves during the real task -- a
scripted sweep has no notion of which joint configurations the task actually visits. Letting a
human teleoperate the follower through motions resembling the real task (moving around near
whatever pose the task actually spends its time in, e.g. near `--reset_environment`'s `HOME_POSE`
("zero")) directly collects the pose distribution that matters, without having to guess it or
widen scripted ranges into poses (like full extension) the task never actually reaches.

Like `collect_free_motion.py`, this data must stay CONTACT-FREE: keep the gripper empty and
don't push against anything while teleoperating, or the resulting `tau_ext` training target
(the current/load signal) will reflect real contact forces instead of the free-space baseline
NEXT is meant to learn.

Usage (run from repo root):
    python -m examples.omx.force_sensing.collect_free_motion_teleop \\
        --port /dev/ttyACM0 --leader_port /dev/ttyACM1 \\
        --output data/omx_free_motion/teleop_run1.npz
Teleoperate naturally (empty gripper, no contact) for as long as you like, then Ctrl+C to stop
and save. Pass `--duration_sec` to stop automatically after a fixed time instead.
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


def teleop_and_log(
    leader: OmxLeader, robot: OmxFollower, log: FreeMotionLogger, hz: float, duration_sec: float | None
) -> None:
    """Relay `leader.get_action()` to the follower while logging every step via `log`, until
    Ctrl+C or `duration_sec` elapses (whichever first). `goal_q` is logged as the leader's
    commanded arm-joint positions (what was actually sent as `Goal_Position`), consistent with
    `collect_free_motion.py`'s other collection phases.
    """
    dt = 1.0 / hz
    end_time = time.perf_counter() + duration_sec if duration_sec is not None else None
    last_health_check = 0.0
    last_print = 0.0
    start = time.perf_counter()
    try:
        while end_time is None or time.perf_counter() < end_time:
            loop_start = time.perf_counter()
            action = leader.get_action()
            robot.send_action(action)
            goal_pose = {j: action[f"{j}.pos"] for j in ARM_JOINTS}
            log.log_step(robot, goal_pose)
            if loop_start - last_health_check > 0.5:
                _check_motor_health(robot)
                last_health_check = loop_start
            if loop_start - last_print > 1.0:
                elapsed = loop_start - start
                print(f"\rTeleoperating and logging -- {elapsed:6.1f}s elapsed (Ctrl+C to stop)", end="", flush=True)
                last_print = loop_start
            precise_sleep(max(0.0, dt - (time.perf_counter() - loop_start)))
    except KeyboardInterrupt:
        pass
    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", default="/dev/ttyACM0", help="Follower port")
    parser.add_argument("--robot_id", default="omx_follower")
    parser.add_argument("--leader_port", required=True, help="Leader port (required -- this script is teleop-driven)")
    parser.add_argument("--leader_id", default="omx_leader")
    parser.add_argument("--output", required=True, help="Output .npz path for the logged episode")
    parser.add_argument("--hz", type=float, default=100.0)
    parser.add_argument("--duration_sec", type=float, default=None, help="Stop automatically after this many seconds (default: run until Ctrl+C)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    robot = OmxFollower(OmxFollowerConfig(port=args.port, id=args.robot_id))
    robot.connect(calibrate=True)

    leader = OmxLeader(OmxLeaderConfig(port=args.leader_port, id=args.leader_id))
    leader.connect(calibrate=True)

    log = FreeMotionLogger()
    try:
        print("Teleoperate the follower through natural task-like motions (empty gripper, no contact). Press Ctrl+C to stop and save.")
        teleop_and_log(leader, robot, log, args.hz, args.duration_sec)
    except MotorStallError:
        logger.exception("Aborting: motor overload protection tripped.")
    finally:
        leader.disconnect()
        robot.disconnect()
        if len(log.t) > 0:
            log.save(Path(args.output))
        else:
            logger.warning("No samples were logged; nothing saved.")


if __name__ == "__main__":
    main()
