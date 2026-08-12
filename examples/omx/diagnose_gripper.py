#!/usr/bin/env python3
"""Read-only diagnostic for the OMX gripper's Drive_Mode / Homing_Offset / calibration state.

Investigates a reported discrepancy: the bilateral force-feedback scripts and stock
`lerobot-teleop` share byte-identical gripper control code (`Operating_Mode=CURRENT_POSITION`,
plain `Goal_Position` passthrough -- confirmed by reading `lerobot_teleoperate.py`,
`omx_leader.py`, `omx_follower.py`, `bilateral_teleop_demo.py`, `record_bilateral.py`), yet were
reported to behave differently (gripper always pulling closed and not returning when opened,
vs. correct open/close behavior under stock `lerobot-teleop`). Since the application code is
identical, the likely culprit is hardware/calibration *state*, not logic:

  - The leader's gripper is calibrated with `Drive_Mode=INVERTED`, `homing_offset=100`
    (`omx_leader.py` `calibrate()`/`configure()`), and `configure()` re-asserts both on every
    connect regardless of `is_calibrated`.
  - The follower's gripper is calibrated with `Drive_Mode=NON_INVERTED`, `homing_offset=0`
    (`omx_follower.py` `calibrate()`), and `configure()` never re-asserts `Drive_Mode` -- it
    relies entirely on whatever `calibrate()` last wrote, with no self-healing on later connects.
  - Dynamixel's `Drive_Mode` is a firmware-level rotation-direction flip that LeRobot's own
    normalize/unnormalize math does *not* compensate for (`DynamixelMotorsBus.apply_drive_mode`
    is `False`), so the same normalized 0-100 gripper value can correspond to different physical
    openness on the leader vs. the follower. The bilateral scripts relay
    `follower_action["gripper.pos"] = leader_pos["gripper"]` with zero compensation for this.

This script does not change anything beyond the normal `connect()`/`configure()` side effects
every OMX script already has -- it only prints live register state and calibration file
contents, so the actual hardware state can be compared between a "broken" run (right after the
bilateral scripts) and a "working" run (right after stock `lerobot-teleop`) without guessing.

UPDATE: a side-by-side comparison of the static diagnostics + manual open/close mapping between
a "broken" bilateral run and a "working" lerobot-teleop run came back *identical* on both counts
(same Drive_Mode/Homing_Offset/calibration, and both leader and follower agree "higher normalized
value = more open" -- no sign inversion). That rules out both hardware-state drift and a simple
direction/sign bug. Re-reading `omx_leader.py`'s `configure()` also confirmed
`enter_current_control_mode()`/`restore_position_mode()` (`leader_safety.py`) never touch the
gripper's `Operating_Mode`/`Current_Limit`/`Goal_Current` (only `ARM_JOINTS`), and that
`sync_read`/`sync_write` both default to `normalize=True`, so the leader-to-follower relay isn't
mixing raw ticks with normalized values either. With every static/code-level hypothesis ruled
out, `--live_relay` (below) reproduces *only* the gripper relay line from
`bilateral_teleop_demo.py`/`record_bilateral.py`
(`follower_action["gripper.pos"] = leader_pos["gripper"]`) in a tight loop with live printing, to
see the relay actually tracking (or not) in real time -- the one kind of evidence a static
snapshot can't capture.

Usage (run from repo root):
    python -m examples.omx.diagnose_gripper \\
        --follower_port /dev/ttyACM0 --follower_id omx_follower \\
        --leader_port /dev/ttyACM1 --leader_id omx_leader

    # Isolated live relay test (needs both leader and follower):
    python -m examples.omx.diagnose_gripper \\
        --follower_port /dev/ttyACM0 --leader_port /dev/ttyACM1 --live_relay

Run once right after reproducing the "gripper always pulls closed" symptom, and once right after
running stock `lerobot-teleop` and confirming it behaves correctly -- ideally without
power-cycling the arms in between -- then diff the two outputs. Pass `--skip_follower` or
`--skip_leader` to check just one side (not compatible with `--live_relay`, which needs both).
"""

import argparse
import logging
import time

from lerobot.motors.dynamixel import DriveMode, OperatingMode
from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig
from lerobot.teleoperators.omx_leader import OmxLeader, OmxLeaderConfig

logger = logging.getLogger(__name__)

GRIPPER_REGISTERS = [
    "Drive_Mode",
    "Homing_Offset",
    "Operating_Mode",
    "Min_Position_Limit",
    "Max_Position_Limit",
]


def format_register(name: str, value: int) -> str:
    if name == "Drive_Mode":
        try:
            return f"{value} ({DriveMode(value).name})"
        except ValueError:
            return str(value)
    if name == "Operating_Mode":
        try:
            return f"{value} ({OperatingMode(value).name})"
        except ValueError:
            return str(value)
    return str(value)


def print_static_diagnostics(label: str, robot) -> None:
    print(f"\n=== {label}: id={robot.id} ===")
    print(f"calibration_fpath: {robot.calibration_fpath}")
    cal = robot.calibration.get("gripper")
    if cal is None:
        print("calibration['gripper']: <missing>")
    else:
        print(
            f"calibration['gripper']: drive_mode={cal.drive_mode} "
            f"homing_offset={cal.homing_offset} range_min={cal.range_min} range_max={cal.range_max}"
        )
    for reg in GRIPPER_REGISTERS:
        value = robot.bus.read(reg, "gripper", normalize=False)
        print(f"live {reg}: {format_register(reg, value)}")
    print(f"is_calibrated: {robot.is_calibrated}")


def stream_gripper_position(label: str, robot, hz: float) -> None:
    print(
        f"\n--- {label}: manually move the gripper by hand (torque disabled on the gripper "
        "motor only -- the arm joints stay servoed). Ctrl+C to stop and move on. ---"
    )
    robot.bus.disable_torque(["gripper"])
    dt = 1.0 / hz
    try:
        while True:
            raw = robot.bus.read("Present_Position", "gripper", normalize=False)
            norm = robot.bus.read("Present_Position", "gripper", normalize=True)
            print(f"\r{label} gripper: raw={raw:5d}  normalized={norm:6.1f}", end="", flush=True)
            time.sleep(dt)
    except KeyboardInterrupt:
        print()


def live_gripper_relay(follower, leader, hz: float) -> None:
    """Reproduce *only* bilateral_teleop_demo.py's/record_bilateral.py's gripper relay line
    (no arm/force logic at all) and print leader target vs. follower actual position + current
    live, each tick. Squeeze/release the leader gripper by hand and watch whether the follower
    tracks it -- this isolates the relay from everything else those scripts also do.
    """
    print(
        "\n--- Live gripper relay: leader Present_Position -> follower Goal_Position, same as "
        "the bilateral scripts, nothing else. Squeeze/release the leader gripper by hand. "
        "Ctrl+C to stop. ---\n"
    )
    dt = 1.0 / hz
    try:
        while True:
            leader_pos = leader.bus.sync_read("Present_Position")
            follower.send_action({"gripper.pos": leader_pos["gripper"]})
            follower_now = follower.bus.read("Present_Position", "gripper", normalize=True)
            follower_current = follower.bus.read("Present_Current", "gripper", normalize=False)
            print(
                f"\rleader_target={leader_pos['gripper']:6.1f}  "
                f"follower_actual={follower_now:6.1f}  follower_current={follower_current:5d}",
                end="",
                flush=True,
            )
            time.sleep(dt)
    except KeyboardInterrupt:
        print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--follower_port", default="/dev/ttyACM0")
    parser.add_argument("--follower_id", default="omx_follower")
    parser.add_argument("--leader_port", default="/dev/ttyACM1")
    parser.add_argument("--leader_id", default="omx_leader")
    parser.add_argument("--skip_follower", action="store_true")
    parser.add_argument("--skip_leader", action="store_true")
    parser.add_argument("--hz", type=float, default=10.0, help="Manual-mapping / live-relay print rate")
    parser.add_argument(
        "--live_relay",
        action="store_true",
        help="Run the isolated live gripper-relay test instead of the manual mapping (needs both leader and follower)",
    )
    args = parser.parse_args()

    if args.live_relay and (args.skip_follower or args.skip_leader):
        raise SystemExit("--live_relay needs both leader and follower connected")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    follower = None
    leader = None
    try:
        if not args.skip_follower:
            follower = OmxFollower(OmxFollowerConfig(port=args.follower_port, id=args.follower_id))
            follower.connect(calibrate=True)
            print_static_diagnostics("Follower", follower)

        if not args.skip_leader:
            leader = OmxLeader(OmxLeaderConfig(port=args.leader_port, id=args.leader_id))
            leader.connect(calibrate=True)
            print_static_diagnostics("Leader", leader)

        if args.live_relay:
            live_gripper_relay(follower, leader, args.hz)
        else:
            print(
                "\nStatic diagnostics done. Next: manually move each connected gripper by hand to "
                "map raw/normalized values to physical open/closed. Ctrl+C after each to move on.\n"
            )
            if follower is not None:
                stream_gripper_position("Follower", follower, args.hz)
            if leader is not None:
                stream_gripper_position("Leader", leader, args.hz)
    finally:
        if follower is not None:
            follower.disconnect()
        if leader is not None:
            leader.disconnect()


if __name__ == "__main__":
    main()
