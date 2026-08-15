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
       the teleop script -- the follower holds its last commanded position on its own).
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
"""

import argparse
import logging
import time
from pathlib import Path

from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig
from lerobot.utils.robot_utils import precise_sleep

from .collect_free_motion import FreeMotionLogger, MotorStallError, _check_motor_health
from .common import ARM_JOINTS

logger = logging.getLogger(__name__)


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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    robot = OmxFollower(OmxFollowerConfig(port=args.port, id=args.robot_id))
    robot.connect(calibrate=True)

    log = FreeMotionLogger()
    try:
        obs = robot.get_observation()
        hold_pose = {j: obs[f"{j}.pos"] for j in ARM_JOINTS}
        print(f"Holding current pose for {args.duration_sec}s (Ctrl+C to stop early).")
        hold_and_log(robot, log, hold_pose, args.duration_sec, args.hz)
    except MotorStallError:
        logger.exception("Aborting: motor overload protection tripped.")
    except KeyboardInterrupt:
        print()
    finally:
        robot.disconnect()
        if len(log.t) > 0:
            log.save(Path(args.output))
        else:
            logger.warning("No samples were logged; nothing saved.")


if __name__ == "__main__":
    main()
