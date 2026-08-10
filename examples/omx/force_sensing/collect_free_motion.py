#!/usr/bin/env python3
"""Collect free-motion (no-contact) data for training a FACTR2-style NEXT external-torque
estimator (arXiv:2606.12406) on the OMX follower arm.

Moves the arm through per-joint sweeps (each of the 5 arm joints individually, across its
safe range, at a slow and a fast speed) followed by randomized multi-joint motion, for a
configurable duration. No contact should occur during collection — keep the workspace clear.
At every control step, logs (with real elapsed timestamps): present position, present
velocity, the commanded goal position (for the tracking-error feature), and present
current/load, for each arm joint.

IMPORTANT: `JOINT_SWEEP_RANGE` below is a conservative default reusing ranges already
exercised by `examples/omx/reset_environment.py`'s HOME_POSE/SWEEP_WAYPOINTS. Verify it is
safe for your specific workspace/cabling before running unattended.

Usage (run from repo root):
    python -m examples.omx.force_sensing.collect_free_motion \\
        --port /dev/ttyACM0 --robot_id omx_follower \\
        --output data/omx_free_motion/run1.npz --duration_min 12
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig
from lerobot.utils.robot_utils import precise_sleep

from ..reset_environment import HOME_POSE
from .common import ARM_JOINTS

logger = logging.getLogger(__name__)

# Conservative per-joint sweep ranges (normalized units, -100..100), consistent with poses
# already exercised in examples/omx/reset_environment.py (HOME_POSE, SWEEP_WAYPOINTS).
JOINT_SWEEP_RANGE = {
    "shoulder_pan": (-60.0, 60.0),
    "shoulder_lift": (-50.0, 50.0),
    "elbow_flex": (-60.0, 50.0),
    "wrist_flex": (-20.0, 20.0),
    "wrist_roll": (-30.0, 30.0),
}

SLOW_SPEED = 15.0  # units/s
FAST_SPEED = 45.0  # units/s
LOOP_HZ = 100.0


class FreeMotionLogger:
    def __init__(self):
        self.t: list[float] = []
        self.q: list[list[float]] = []
        self.qdot: list[list[float]] = []
        self.goal_q: list[list[float]] = []
        self.current: list[list[float]] = []
        self._t0: float | None = None

    def log_step(self, robot: OmxFollower, goal_pose: dict[str, float]) -> None:
        if self._t0 is None:
            self._t0 = time.perf_counter()
        pos = robot.bus.sync_read("Present_Position")
        vel = robot.bus.sync_read("Present_Velocity")
        cur = robot.bus.sync_read("Present_Current")
        self.t.append(time.perf_counter() - self._t0)
        self.q.append([pos[j] for j in ARM_JOINTS])
        self.qdot.append([vel[j] for j in ARM_JOINTS])
        self.goal_q.append([goal_pose[j] for j in ARM_JOINTS])
        self.current.append([cur[j] for j in ARM_JOINTS])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            t=np.array(self.t, dtype=np.float64),
            q=np.array(self.q, dtype=np.float64),
            qdot=np.array(self.qdot, dtype=np.float64),
            goal_q=np.array(self.goal_q, dtype=np.float64),
            current=np.array(self.current, dtype=np.float64),
            joint_names=np.array(ARM_JOINTS),
        )
        logger.info(f"Saved {len(self.t)} samples ({self.t[-1]:.1f}s) to {path}")


def _move_and_log(
    robot: OmxFollower,
    log: FreeMotionLogger,
    current_pose: dict[str, float],
    target_pose: dict[str, float],
    speed: float,
    hz: float = LOOP_HZ,
) -> dict[str, float]:
    """Interpolate current_pose -> target_pose at `speed` units/s, logging every step.

    Returns target_pose (the new current pose for the caller).
    """
    cur = np.array([current_pose[j] for j in ARM_JOINTS])
    goal = np.array([target_pose[j] for j in ARM_JOINTS])
    max_dist = float(np.max(np.abs(goal - cur)))
    if max_dist < 0.5:
        return target_pose

    n_steps = max(1, int(max_dist / speed * hz))
    dt = 1.0 / hz
    for step in range(1, n_steps + 1):
        t = step / n_steps
        interp = cur + t * (goal - cur)
        action = {f"{j}.pos": float(v) for j, v in zip(ARM_JOINTS, interp, strict=True)}
        robot.send_action(action)
        log.log_step(robot, dict(zip(ARM_JOINTS, interp, strict=True)))
        precise_sleep(dt)
    return target_pose


def collect(robot: OmxFollower, log: FreeMotionLogger, duration_min: float) -> None:
    rng = np.random.default_rng()
    pose = {j: HOME_POSE[f"{j}.pos"] for j in ARM_JOINTS}
    end_time = time.perf_counter() + duration_min * 60

    logger.info("Phase A: per-joint sweeps (slow + fast)...")
    for joint in ARM_JOINTS:
        lo, hi = JOINT_SWEEP_RANGE[joint]
        home_val = HOME_POSE[f"{joint}.pos"]
        for speed in (SLOW_SPEED, FAST_SPEED):
            for target_val in (lo, hi, home_val):
                target = {**pose, joint: target_val}
                pose = _move_and_log(robot, log, pose, target, speed)
        if time.perf_counter() > end_time:
            return

    logger.info("Phase B: randomized combined multi-joint motion...")
    while time.perf_counter() < end_time:
        target = dict(pose)
        for joint in ARM_JOINTS:
            lo, hi = JOINT_SWEEP_RANGE[joint]
            target[joint] = float(rng.uniform(lo, hi))
        speed = float(rng.uniform(SLOW_SPEED, FAST_SPEED))
        pose = _move_and_log(robot, log, pose, target, speed)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--robot_id", default="omx_follower")
    parser.add_argument("--output", required=True, help="Output .npz path for the logged episode")
    parser.add_argument(
        "--duration_min", type=float, default=12.0, help="Target collection duration (minutes)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    robot = OmxFollower(OmxFollowerConfig(port=args.port, id=args.robot_id))
    robot.connect(calibrate=True)

    log = FreeMotionLogger()
    try:
        obs = robot.get_observation()
        start_pose = {j: obs[f"{j}.pos"] for j in ARM_JOINTS}
        home_pose = {j: HOME_POSE[f"{j}.pos"] for j in ARM_JOINTS}
        _move_and_log(robot, log, start_pose, home_pose, SLOW_SPEED)
        collect(robot, log, args.duration_min)
    finally:
        robot.disconnect()
        if len(log.t) > 0:
            log.save(Path(args.output))
        else:
            logger.warning("No samples were logged; nothing saved.")


if __name__ == "__main__":
    main()
