#!/usr/bin/env python3
"""Collect free-motion (no-contact) data for training a FACTR2-style NEXT external-torque
estimator (arXiv:2606.12406) on the OMX follower arm.

Moves the arm through per-joint sweeps (each of the 5 arm joints individually, across its
safe range, at a slow and a fast speed), a dedicated shoulder_lift x elbow_flex grid sweep
(Phase A2, see below), then randomized multi-joint motion, for a configurable duration. No
contact should occur during collection — keep the workspace clear. At every control step,
logs (with real elapsed timestamps): present position, present velocity, the commanded goal
position (for the tracking-error feature), and present current/load, for each arm joint.

Phase A2 (shoulder_lift x elbow_flex grid): a real-world investigation (see
src/lerobot/force_estimation/README.md) found that tau_ext's noise floor has a systematic,
pose-dependent bias concentrated in shoulder_lift/elbow_flex -- the two gravity-loaded,
kinematically-coupled joints -- that persists even near-zero velocity. The other phases only
ever sweep these two joints *individually* (holding the other at home); their *combined*
configuration space was left to Phase B's unstructured random sampling. Phase A2 sweeps a
dedicated (shoulder_lift, elbow_flex) grid to densify exactly that region. Each grid point is
also held still (dwelled) for `--dwell_sec` after arriving -- long enough to exceed
train_next.py's default `--history-length 50` @ `--resample-hz 100.0` (0.5s), so a dwell
period contributes some windows whose *entire* input history is genuinely static, not just a
low-velocity instant mid-sweep (which is all the rest of this script ever produces). If you
change `--history-length`/`--resample-hz` at training time, reconsider `--dwell_sec` too.

To avoid baking in one fixed backlash state, the grid is visited twice by default
(`--grid_passes`): once in a random shuffled order, once in the *exact reverse* of that
order (a deliberate opposite-direction pass, not just another random shuffle), with any
further passes freshly reshuffled. This is a statistical/aggregate argument across the whole
collection run, not a per-point controlled experiment -- it doesn't guarantee every single
grid point is approached from both directions, just that the run as a whole isn't one
repeated monotonic pattern.

SAFETY: `shoulder_lift` and `elbow_flex` are NOT independent axes on this arm — a low
`elbow_flex` combined with a low `shoulder_lift` drives the wrist/gripper down through the
base plate. This script reuses the validated piecewise-linear `elbow_flex` envelope (as a
function of `shoulder_lift`) and the wrist_flex/shoulder_pan/wrist_roll ranges from
`examples/omx/record_grab.py`'s `_random_stuck_pose()`, which is the only place in this repo
those joints' *combined* safe range has actually been exercised. Every pose this script
commands is built through `safe_pose()` below, which re-derives `elbow_flex`/`wrist_flex`
from the current `shoulder_lift` rather than sweeping them independently. Note that while
Phase A2's grid stays within this same envelope *formula*, it deliberately visits combined
(shoulder_lift, elbow_flex) corner extremes together -- a combination Phase A never exercises
(it only ever varies one of the two at a time) and Phase B only reaches by low-probability
chance -- so treat a first run of this script the same as any other envelope change below.

Even so, verify this envelope against your own physical setup (mounting height, cables,
nearby obstacles) before running unattended — start with a short `--duration_min` and stay
within reach of the power switch. The script aborts immediately (saving what was logged so
far) if it detects a motor has hit its overload/hardware-error protection.

Usage (run from repo root):
    python -m examples.omx.force_sensing.collect_free_motion \\
        --port /dev/ttyACM0 --robot_id omx_follower \\
        --output data/omx_free_motion/run1.npz --duration_min 15
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

# elbow_flex must stay within `safe_elbow_flex_range(shoulder_lift)` (validated envelope from
# examples/omx/record_grab.py's `_random_stuck_pose()`) and wrist_flex is always derived from
# (shoulder_lift, elbow_flex) rather than swept independently -- see safe_pose() below.
#
# shoulder_pan and wrist_roll are rotations about the arm's own axes: unlike shoulder_lift/
# elbow_flex, panning/rolling doesn't by itself drive the wrist toward the base plate, so
# their safe range is primarily about clearance in *your* physical setup (cables, mounts,
# nearby objects) rather than the arm's own kinematics. SHOULDER_PAN_RANGE was originally
# (-5.0, 35.0) (reusing _random_stuck_pose()'s task-specific bias toward one side); widened
# here to a symmetric +-35 deg now that +35 deg has been run and confirmed clear. Re-verify
# clearance on the newly-added side (-35 to -5 deg) before running unattended, same as any
# other range change here.
SHOULDER_PAN_RANGE = (-35.0, 35.0)
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


def _move_and_dwell(
    robot: OmxFollower,
    log: FreeMotionLogger,
    current_pose: dict[str, float],
    target_pose: dict[str, float],
    speed: float,
    dwell_sec: float,
    hz: float = LOOP_HZ,
) -> dict[str, float]:
    """`_move_and_log()` to `target_pose`, then hold and keep logging there for `dwell_sec`
    more seconds. A separate wrapper (rather than a parameter on `_move_and_log()` itself) so
    dwelling isn't silently skipped by `_move_and_log()`'s `max_dist < 0.5` early return, and
    so `_move_and_log()`'s own contract ("interpolate and log") stays unchanged.
    """
    pose = _move_and_log(robot, log, current_pose, target_pose, speed, hz)
    if dwell_sec <= 0:
        return pose
    dt = 1.0 / hz
    end = time.perf_counter() + dwell_sec
    last_health_check = time.perf_counter()
    while time.perf_counter() < end:
        loop_start = time.perf_counter()
        robot.send_action({f"{j}.pos": v for j, v in pose.items()})
        log.log_step(robot, pose)
        if loop_start - last_health_check > 0.5:
            _check_motor_health(robot)
            last_health_check = loop_start
        precise_sleep(max(0.0, dt - (time.perf_counter() - loop_start)))
    _check_motor_health(robot)
    return pose


def _build_sl_ef_grid(n_sl: int, n_ef: int) -> list[tuple[float, float]]:
    """`(shoulder_lift, elbow_flex)` grid spanning `SHOULDER_LIFT_RANGE` at `n_sl` points and,
    for each, `safe_elbow_flex_range(sl)` at `n_ef` points -- reuses the existing validated
    envelope directly, so this doesn't add a new safe region, only denser sampling within the
    existing one.
    """
    grid = []
    for sl in np.linspace(*SHOULDER_LIFT_RANGE, n_sl):
        lo, hi = safe_elbow_flex_range(float(sl))
        for ef in np.linspace(lo, hi, n_ef):
            grid.append((float(sl), float(ef)))
    return grid


def _sl_ef_grid_sweep(
    robot: OmxFollower,
    log: FreeMotionLogger,
    pose: dict[str, float],
    ref: dict[str, float],
    rng: np.random.Generator,
    n_sl: int,
    n_ef: int,
    n_passes: int,
    dwell_sec: float,
    end_time: float,
) -> dict[str, float]:
    """Visit `_build_sl_ef_grid(n_sl, n_ef)` for `n_passes` passes: pass 0 in a random shuffled
    order, pass 1 (if any) in the *exact reverse* of pass 0's order (a deliberate
    opposite-direction pass against backlash, rather than hoping a second random shuffle
    happens to differ), further passes freshly reshuffled. shoulder_pan/wrist_roll stay at
    `ref` (home); wrist_flex is still derived via `safe_pose()` as everywhere else in this
    script.
    """
    grid = _build_sl_ef_grid(n_sl, n_ef)
    order = list(rng.permutation(len(grid)))
    for i in range(n_passes):
        if i == 1:
            order = list(reversed(order))
        elif i > 1:
            order = list(rng.permutation(len(grid)))
        for idx in order:
            sl, ef = grid[idx]
            speed = float(rng.uniform(SLOW_SPEED, FAST_SPEED))
            target_pose = safe_pose(ref["pan"], sl, ef, 0.0, ref["roll"])
            pose = _move_and_dwell(robot, log, pose, target_pose, speed, dwell_sec)
            if time.perf_counter() > end_time:
                return pose
    return pose


def collect(
    robot: OmxFollower,
    log: FreeMotionLogger,
    duration_min: float,
    dwell_sec: float = 0.0,
    grid_sl_points: int = 6,
    grid_ef_points: int = 5,
    grid_passes: int = 2,
) -> None:
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
                pose = _move_and_dwell(robot, log, pose, target_pose, speed, dwell_sec)
        if time.perf_counter() > end_time:
            return

    logger.info("Phase A: wrist_flex jitter sweep (slow + fast)...")
    for speed in (SLOW_SPEED, FAST_SPEED):
        for jitter in (*WRIST_FLEX_JITTER_RANGE, 0.0):
            target_pose = safe_pose(ref["pan"], ref["sl"], ref["ef"], jitter, ref["roll"])
            pose = _move_and_dwell(robot, log, pose, target_pose, speed, dwell_sec)
    if time.perf_counter() > end_time:
        return

    logger.info(
        f"Phase A2: shoulder_lift x elbow_flex grid sweep "
        f"({grid_sl_points}x{grid_ef_points} points, {grid_passes} passes)..."
    )
    pose = _sl_ef_grid_sweep(
        robot, log, pose, ref, rng, grid_sl_points, grid_ef_points, grid_passes, dwell_sec, end_time
    )
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
        pose = _move_and_dwell(robot, log, pose, target_pose, speed, dwell_sec)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--robot_id", default="omx_follower")
    parser.add_argument("--output", required=True, help="Output .npz path for the logged episode")
    parser.add_argument(
        "--duration_min", type=float, default=15.0, help="Target collection duration (minutes)"
    )
    parser.add_argument(
        "--dwell_sec",
        type=float,
        default=1.0,
        help=(
            "Seconds to hold still (still logging) after each move, across all phases. Should "
            "exceed train_next.py's --history-length / --resample-hz (default 50/100.0 = 0.5s) "
            "with margin, or dwelling contributes no fully-static training windows."
        ),
    )
    parser.add_argument(
        "--grid_sl_points", type=int, default=6, help="Phase A2: shoulder_lift grid points"
    )
    parser.add_argument("--grid_ef_points", type=int, default=5, help="Phase A2: elbow_flex grid points")
    parser.add_argument(
        "--grid_passes",
        type=int,
        default=2,
        help="Phase A2: traversal passes (pass 2 is the exact reverse of pass 1, for backlash)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if 0 < args.dwell_sec * LOOP_HZ < 50:
        logger.warning(
            f"--dwell_sec {args.dwell_sec} is shorter than train_next.py's default "
            "--history-length 50 @ --resample-hz 100.0 (0.5s) -- dwell periods may not "
            "contribute any fully-static training windows. Consider >= 1.0s."
        )

    robot = OmxFollower(OmxFollowerConfig(port=args.port, id=args.robot_id))
    robot.connect(calibrate=True)

    log = FreeMotionLogger()
    try:
        obs = robot.get_observation()
        start_pose = {j: obs[f"{j}.pos"] for j in ARM_JOINTS}
        home_pose = {j: HOME_POSE[f"{j}.pos"] for j in ARM_JOINTS}
        _move_and_log(robot, log, start_pose, home_pose, SLOW_SPEED)
        collect(
            robot,
            log,
            args.duration_min,
            dwell_sec=args.dwell_sec,
            grid_sl_points=args.grid_sl_points,
            grid_ef_points=args.grid_ef_points,
            grid_passes=args.grid_passes,
        )
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
