#!/usr/bin/env python3
"""Collect free-motion (no-contact) data for training a FACTR2-style NEXT external-torque
estimator (arXiv:2606.12406) on the OMX follower arm.

Moves the arm through per-joint sweeps (each of the 5 arm joints individually, across its
safe range, at a slow and a fast speed) followed by randomized multi-joint motion, for a
configurable duration. No contact should occur during collection — keep the workspace clear.
At every control step, logs (with real elapsed timestamps): present position, present
velocity, the commanded goal position (for the tracking-error feature), and present
current/load, for each arm joint.

SAFETY: `shoulder_lift` and `elbow_flex` are NOT independent axes on this arm — a low
`elbow_flex` combined with a low `shoulder_lift` drives the wrist/gripper down through the
base plate. This script reuses the validated piecewise-linear `elbow_flex` envelope (as a
function of `shoulder_lift`) and the wrist_flex/shoulder_pan/wrist_roll ranges from
`examples/omx/record_grab.py`'s `_random_stuck_pose()`, which is the only place in this repo
those joints' *combined* safe range has actually been exercised. Every pose this script
commands is built through `safe_pose()` below, which re-derives `elbow_flex`/`wrist_flex`
from the current `shoulder_lift` rather than sweeping them independently.

Even so, verify this envelope against your own physical setup (mounting height, cables,
nearby obstacles) before running unattended — start with a short `--duration_min` and stay
within reach of the power switch. The script aborts immediately (saving what was logged so
far) if it detects a motor has hit its overload/hardware-error protection.

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

from ..reset_environment import HOME_POSE, horizontal_wrist_flex
from .common import ARM_JOINTS

logger = logging.getLogger(__name__)

# Validated joint-space envelope, reused as-is from examples/omx/record_grab.py's
# `_random_stuck_pose()`: shoulder_pan and wrist_roll are safe as independent axes, but
# elbow_flex must stay within `safe_elbow_flex_range(shoulder_lift)` and wrist_flex is
# always derived from (shoulder_lift, elbow_flex) rather than swept independently.
SHOULDER_PAN_RANGE = (-5.0, 35.0)
SHOULDER_LIFT_RANGE = (-50.0, 30.0)
WRIST_ROLL_RANGE = (-15.0, 15.0)
WRIST_FLEX_JITTER_RANGE = (-15.0, 15.0)

SLOW_SPEED = 15.0  # units/s
FAST_SPEED = 45.0  # units/s
LOOP_HZ = 100.0


def safe_elbow_flex_range(shoulder_lift: float) -> tuple[float, float]:
    """Piecewise-linear elbow_flex bounds vs. shoulder_lift that keep the arm in a reachable,
    table-safe envelope (mirrors `examples/omx/record_grab.py`'s `_random_stuck_pose`):
        sl=-50 -> ef in [  0,  50]   (arm raised, can be bent forward)
        sl=  0 -> ef in [-25,  25]   (mid reach)
        sl= 30 -> ef in [-20,   0]   (arm extended, little room to flex)
    Only valid for shoulder_lift within SHOULDER_LIFT_RANGE (clamped otherwise).
    """
    sl = max(SHOULDER_LIFT_RANGE[0], min(SHOULDER_LIFT_RANGE[1], shoulder_lift))
    if sl <= 0.0:
        alpha = (sl + 50.0) / 50.0  # 0 at sl=-50, 1 at sl=0
        return alpha * -25.0, 50.0 + alpha * -25.0
    alpha = sl / 30.0  # 0 at sl=0, 1 at sl=30
    return -25.0 + alpha * 5.0, 25.0 + alpha * -25.0


def safe_pose(
    shoulder_pan: float,
    shoulder_lift: float,
    elbow_flex: float,
    wrist_flex_jitter: float,
    wrist_roll: float,
) -> dict[str, float]:
    """Build a pose that respects the shoulder_lift/elbow_flex envelope, clamping
    `elbow_flex` into `safe_elbow_flex_range(shoulder_lift)` and deriving `wrist_flex` from
    the resulting (shoulder_lift, elbow_flex) plus a small jitter, instead of taking
    `elbow_flex`/`wrist_flex` at face value.
    """
    sl = max(SHOULDER_LIFT_RANGE[0], min(SHOULDER_LIFT_RANGE[1], shoulder_lift))
    lo, hi = safe_elbow_flex_range(sl)
    ef = max(lo, min(hi, elbow_flex))
    wf = horizontal_wrist_flex(sl, ef) + wrist_flex_jitter
    return {
        "shoulder_pan": shoulder_pan,
        "shoulder_lift": sl,
        "elbow_flex": ef,
        "wrist_flex": wf,
        "wrist_roll": wrist_roll,
    }


class MotorStallError(RuntimeError):
    pass


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


def _check_motor_health(robot: OmxFollower) -> None:
    """Abort if any arm motor has tripped its overload/hardware-error protection or had its
    torque disabled — continuing to command a stalled joint risks damaging the motor/gearbox.
    """
    errors = robot.bus.sync_read("Hardware_Error_Status")
    torque = robot.bus.sync_read("Torque_Enable")
    stalled = [j for j in ARM_JOINTS if errors[j] != 0 or torque[j] == 0]
    if stalled:
        raise MotorStallError(
            f"Motor protection tripped on {stalled} (Hardware_Error_Status/Torque_Enable). "
            "Stopping immediately — check the arm for a collision (e.g. the wrist/gripper "
            "against the base plate) before power-cycling and retrying."
        )


def _move_and_log(
    robot: OmxFollower,
    log: FreeMotionLogger,
    current_pose: dict[str, float],
    target_pose: dict[str, float],
    speed: float,
    hz: float = LOOP_HZ,
) -> dict[str, float]:
    """Interpolate current_pose -> target_pose at `speed` units/s, logging every step.

    Returns target_pose (the new current pose for the caller). Raises `MotorStallError` if a
    motor trips its overload protection partway through.
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
    _check_motor_health(robot)
    return target_pose


def collect(robot: OmxFollower, log: FreeMotionLogger, duration_min: float) -> None:
    rng = np.random.default_rng()
    home_pan = HOME_POSE["shoulder_pan.pos"]
    home_sl = HOME_POSE["shoulder_lift.pos"]
    home_ef = HOME_POSE["elbow_flex.pos"]
    home_roll = HOME_POSE["wrist_roll.pos"]

    ref = {"pan": home_pan, "sl": home_sl, "ef": home_ef, "roll": home_roll}
    pose = safe_pose(ref["pan"], ref["sl"], ref["ef"], 0.0, ref["roll"])
    end_time = time.perf_counter() + duration_min * 60

    sweep_axes = [
        ("shoulder_pan", "pan", SHOULDER_PAN_RANGE, home_pan),
        ("shoulder_lift", "sl", SHOULDER_LIFT_RANGE, home_sl),
        ("elbow_flex", "ef", None, home_ef),  # range resolved per-step from current sl
        ("wrist_roll", "roll", WRIST_ROLL_RANGE, home_roll),
    ]

    logger.info("Phase A: per-joint sweeps (slow + fast)...")
    for _joint_name, key, joint_range, home_val in sweep_axes:
        lo, hi = joint_range if joint_range is not None else safe_elbow_flex_range(ref["sl"])
        for speed in (SLOW_SPEED, FAST_SPEED):
            for target_val in (lo, hi, home_val):
                target = dict(ref)
                target[key] = target_val
                target_pose = safe_pose(target["pan"], target["sl"], target["ef"], 0.0, target["roll"])
                pose = _move_and_log(robot, log, pose, target_pose, speed)
        if time.perf_counter() > end_time:
            return

    logger.info("Phase A: wrist_flex jitter sweep (slow + fast)...")
    for speed in (SLOW_SPEED, FAST_SPEED):
        for jitter in (*WRIST_FLEX_JITTER_RANGE, 0.0):
            target_pose = safe_pose(ref["pan"], ref["sl"], ref["ef"], jitter, ref["roll"])
            pose = _move_and_log(robot, log, pose, target_pose, speed)
    if time.perf_counter() > end_time:
        return

    logger.info("Phase B: randomized combined multi-joint motion...")
    while time.perf_counter() < end_time:
        pan = float(rng.uniform(*SHOULDER_PAN_RANGE))
        sl = float(rng.uniform(*SHOULDER_LIFT_RANGE))
        lo, hi = safe_elbow_flex_range(sl)
        ef = float(rng.uniform(lo, hi))
        jitter = float(rng.uniform(*WRIST_FLEX_JITTER_RANGE))
        roll = float(rng.uniform(*WRIST_ROLL_RANGE))
        target_pose = safe_pose(pan, sl, ef, jitter, roll)
        speed = float(rng.uniform(SLOW_SPEED, FAST_SPEED))
        pose = _move_and_log(robot, log, pose, target_pose, speed)


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
    except MotorStallError:
        logger.exception("Aborting: motor overload protection tripped.")
    finally:
        robot.disconnect()
        if len(log.t) > 0:
            log.save(Path(args.output))
        else:
            logger.warning("No samples were logged; nothing saved.")


if __name__ == "__main__":
    main()
