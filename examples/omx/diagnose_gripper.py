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

UPDATE 2: `--live_relay` alone (arm joints left as `configure()` leaves them -- torque disabled,
free) reproduced the *expected* trigger behavior (opens with light force, stays open, closes and
holds). But the full bilateral scripts additionally put the leader's 5 `ARM_JOINTS` into Current
Control Mode (`enter_current_control_mode`/`restore_position_mode`, `leader_safety.py`) and write
`Goal_Current` to them every tick -- and under the *full* script the reported behavior flips
(closing bias at rest, opening needs force, releasing while open falls back closed). Since
`leader_safety.py`'s own code only ever touches `ARM_JOINTS` (never `gripper`), this can only be
distinguished by testing live: `--with_arm_current_control` additionally switches the leader's
arm joints into Current Control Mode (writing 0mA every tick, i.e. the mode switch itself with no
gravity-comp/damping/feedback math at all) while running the same gripper relay, to isolate
whether merely being in Current Control Mode -- independent of any actual computed force -- is
what changes the gripper's felt behavior.

UPDATE 3: `--with_arm_current_control` reproduces the closing-bias symptom at `--arm_current_ma 0`
just as strongly as at higher values -- it depends *only* on whether the arm joints are in
Current Control Mode at all, not on how much current they actually draw. This rules out the
voltage-sag/dose-response theory (a real supply-voltage sag would scale with current, not appear
identically at 0mA). `Present_Input_Voltage` monitoring is kept since it's still useful to rule
in/out voltage as a factor case by case.

UPDATE 4: two more data points nail down the mechanism. (a) `--gripper_current_limit_ma 0` with
*no* `--with_arm_current_control` at all -- i.e. the gripper simply has zero torque the whole
time -- makes it fall closed on its own. So the gripper mechanism has a genuine mechanical bias
toward closed when unpowered (spring/gravity/detent -- not a bug, just how the hardware is
built). (b) Under `--with_arm_current_control`, *raising* `--gripper_current_limit_ma` makes it
*harder* to pry open, not easier. If the gripper's target were still correctly
`gripper_open_pos` (60) and it just lacked torque to fight friction, more current should make it
easier to reach/hold open -- instead more current means it fights *harder to stay closed*. That
only makes sense if the gripper's actual `Goal_Position` itself has shifted toward the closed end
once the arm enters Current Control Mode, not just "not enough torque."

Putting (a) + (b) together with `enter_current_control_mode()`'s own code
(`src/lerobot/teleoperators/omx_leader/leader_safety.py`):
```
def enter_current_control_mode(leader, current_limit_ma):
    with leader.bus.torque_disabled():        # scopes to ALL motors, gripper included
        for joint in ARM_JOINTS:               # ...but only ARM_JOINTS actually need writing
            leader.bus.write("Operating_Mode", joint, OperatingMode.CURRENT.value)
            leader.bus.write("Current_Limit", joint, current_limit_ma)
```
`torque_disabled()` with no `motors` argument affects every motor, so the gripper's torque is
disabled too, even though nothing inside the `with` block ever needs to touch it. During that
window the gripper (per (a)) falls toward its closed mechanical rest position, and
`DynamixelMotorsBus.enable_torque()`/`disable_torque()` (`dynamixel.py:191-201`) were confirmed
to only ever write `Torque_Enable` -- never `Goal_Position` -- so if the effective target really
has shifted, it's Dynamixel firmware behavior on the Torque_Enable OFF->ON transition in
`CURRENT_POSITION` mode, outside LeRobot's control -- but avoidable entirely by never disabling
the gripper's torque in the first place. `--scope_arm_torque_disable` tests exactly that fix
in isolation (bypassing the real `enter_current_control_mode`, using a copy scoped to
`ARM_JOINTS` only) before touching the shared `leader_safety.py`.

RESOLVED: `--scope_arm_torque_disable` confirmed on hardware -- both leader and follower gripper
behaved correctly. The fix (`torque_disabled(ARM_JOINTS)` instead of unscoped
`torque_disabled()`) has been applied to `leader_safety.enter_current_control_mode()`, so
`--with_arm_current_control` should now behave correctly even *without*
`--scope_arm_torque_disable` (the two paths are equivalent again). `enter_current_control_mode_scoped()`
and `--scope_arm_torque_disable` are kept as a standing regression test / historical record of
the bug hunt, not because they're still needed to get correct behavior.

Usage (run from repo root):
    python -m examples.omx.diagnose_gripper \\
        --follower_port /dev/ttyACM0 --follower_id omx_follower \\
        --leader_port /dev/ttyACM1 --leader_id omx_leader

    # Isolated live relay test (needs both leader and follower):
    python -m examples.omx.diagnose_gripper \\
        --follower_port /dev/ttyACM0 --leader_port /dev/ttyACM1 --live_relay

    # Same, but also puts the leader's arm joints into Current Control Mode (0mA, no gravity
    # comp/damping/feedback) to isolate whether the mode switch itself affects the gripper:
    python -m examples.omx.diagnose_gripper \\
        --follower_port /dev/ttyACM0 --leader_port /dev/ttyACM1 --live_relay --with_arm_current_control

    # Ramp the arm's current draw and watch Present_Input_Voltage / gripper behavior together:
    python -m examples.omx.diagnose_gripper \\
        --follower_port /dev/ttyACM0 --leader_port /dev/ttyACM1 \\
        --live_relay --with_arm_current_control --arm_current_ma 300

    # Test whether more gripper current headroom overpowers the effect:
    python -m examples.omx.diagnose_gripper \\
        --follower_port /dev/ttyACM0 --leader_port /dev/ttyACM1 \\
        --live_relay --with_arm_current_control --gripper_current_limit_ma 400

    # A/B test the candidate fix (gripper torque never disabled during the arm's mode switch):
    python -m examples.omx.diagnose_gripper \\
        --follower_port /dev/ttyACM0 --leader_port /dev/ttyACM1 \\
        --live_relay --with_arm_current_control --scope_arm_torque_disable

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
from lerobot.teleoperators.omx_leader.gravity_compensation import ARM_JOINTS
from lerobot.teleoperators.omx_leader.leader_safety import (
    enter_current_control_mode,
    restore_position_mode,
)

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


def enter_current_control_mode_scoped(leader, current_limit_ma: int) -> None:
    """Candidate fix for `leader_safety.enter_current_control_mode()`: identical, except
    `torque_disabled()` is scoped to `ARM_JOINTS` only, so the gripper's torque is never
    disabled (it doesn't need to be -- only ARM_JOINTS' Operating_Mode/Current_Limit get
    written). Lets `--scope_arm_torque_disable` A/B test the fix before it's applied to the
    shared `leader_safety.py`.
    """
    with leader.bus.torque_disabled(ARM_JOINTS):
        for joint in ARM_JOINTS:
            leader.bus.write("Operating_Mode", joint, OperatingMode.CURRENT.value)
            leader.bus.write("Current_Limit", joint, current_limit_ma)


def live_gripper_relay(
    follower,
    leader,
    hz: float,
    with_arm_current_control: bool,
    arm_current_ma: int,
    gripper_current_limit_ma: int,
    scope_arm_torque_disable: bool,
) -> None:
    """Reproduce *only* bilateral_teleop_demo.py's/record_bilateral.py's gripper relay line
    (no gravity-comp/damping/feedback math) and print leader target vs. follower actual
    position + current live, each tick. Squeeze/release the leader gripper by hand and watch
    whether the follower tracks it -- this isolates the relay from everything else those scripts
    also do.

    If `with_arm_current_control` is set, the leader's `ARM_JOINTS` are additionally switched to
    Current Control Mode and written `arm_current_ma` every tick (the mode switch itself plus a
    controllable, uniform current draw -- no gravity-comp/damping/feedback math). Confirmed by
    hand: the gripper's closing-bias symptom depends only on whether the arm is in Current
    Control Mode at all, not on `arm_current_ma`'s value (identical at 0mA) -- so this isn't a
    supply-voltage-sag effect, `Present_Input_Voltage` (leader gripper + `shoulder_pan`) is
    printed mainly to keep ruling that in/out per setup. `gripper_current_limit_ma` overrides the
    gripper's own `Current_Limit`/`Goal_Current` (`OmxLeader.configure()`'s default is 100) so you
    can test whether simply giving the trigger more current headroom overpowers the effect.

    `scope_arm_torque_disable` swaps in `enter_current_control_mode_scoped()` (above) instead of
    the real `leader_safety.enter_current_control_mode()` -- the candidate fix under test.
    """
    print(
        "\n--- Live gripper relay: leader Present_Position -> follower Goal_Position, same as "
        "the bilateral scripts, nothing else"
        + (
            f" (+ leader ARM_JOINTS in Current Control Mode, {arm_current_ma}mA each"
            + (", torque_disabled scoped to ARM_JOINTS -- fix under test)" if scope_arm_torque_disable else ")")
            if with_arm_current_control
            else ""
        )
        + (
            f" (+ gripper Current_Limit/Goal_Current={gripper_current_limit_ma}mA)"
            if gripper_current_limit_ma != 100
            else ""
        )
        + ". Squeeze/release the leader gripper by hand. Ctrl+C to stop. ---\n"
    )
    if gripper_current_limit_ma != 100:
        # Current_Limit is an EEPROM register; Dynamixel rejects EEPROM writes while torque is on.
        with leader.bus.torque_disabled(["gripper"]):
            leader.bus.write("Current_Limit", "gripper", gripper_current_limit_ma)
        leader.bus.write("Goal_Current", "gripper", gripper_current_limit_ma)
    if with_arm_current_control:
        if scope_arm_torque_disable:
            enter_current_control_mode_scoped(leader, current_limit_ma=max(500, arm_current_ma))
        else:
            enter_current_control_mode(leader, current_limit_ma=max(500, arm_current_ma))
        arm_current = dict.fromkeys(ARM_JOINTS, arm_current_ma)
    dt = 1.0 / hz
    try:
        while True:
            leader_pos = leader.bus.sync_read("Present_Position")
            if with_arm_current_control:
                leader.bus.sync_write("Goal_Current", arm_current)
            follower.send_action({"gripper.pos": leader_pos["gripper"]})
            follower_now = follower.bus.read("Present_Position", "gripper", normalize=True)
            follower_current = follower.bus.read("Present_Current", "gripper", normalize=False)
            gripper_voltage = leader.bus.read("Present_Input_Voltage", "gripper", normalize=False)
            arm_voltage = leader.bus.read("Present_Input_Voltage", "shoulder_pan", normalize=False)
            print(
                f"\rleader_target={leader_pos['gripper']:6.1f}  "
                f"follower_actual={follower_now:6.1f}  follower_current={follower_current:5d}  "
                f"leader_gripper_voltage={gripper_voltage / 10:4.1f}V  "
                f"leader_shoulder_pan_voltage={arm_voltage / 10:4.1f}V",
                end="",
                flush=True,
            )
            time.sleep(dt)
    except KeyboardInterrupt:
        print()
    finally:
        if with_arm_current_control:
            restore_position_mode(leader)


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
    parser.add_argument(
        "--with_arm_current_control",
        action="store_true",
        help="With --live_relay, also switch the leader's ARM_JOINTS to Current Control Mode "
        "to isolate whether the mode switch / current draw itself affects the gripper",
    )
    parser.add_argument(
        "--arm_current_ma",
        type=int,
        default=0,
        help="Current written to each ARM_JOINT every tick with --with_arm_current_control. Try "
        "ramping this up (e.g. 0, 100, 300) to see if the gripper's closing bias and "
        "Present_Input_Voltage scale with it (shared-bus voltage sag hypothesis)",
    )
    parser.add_argument(
        "--gripper_current_limit_ma",
        type=int,
        default=100,
        help="Override the leader gripper's Current_Limit/Goal_Current (default 100, matching "
        "OmxLeader.configure()) to test whether more current headroom overpowers the "
        "closing-bias effect while --with_arm_current_control is active",
    )
    parser.add_argument(
        "--scope_arm_torque_disable",
        action="store_true",
        help="With --with_arm_current_control, use the candidate fix (torque_disabled scoped to "
        "ARM_JOINTS only, gripper torque never touched) instead of the real "
        "leader_safety.enter_current_control_mode()",
    )
    args = parser.parse_args()

    if args.live_relay and (args.skip_follower or args.skip_leader):
        raise SystemExit("--live_relay needs both leader and follower connected")
    if args.with_arm_current_control and not args.live_relay:
        raise SystemExit("--with_arm_current_control requires --live_relay")
    if args.scope_arm_torque_disable and not args.with_arm_current_control:
        raise SystemExit("--scope_arm_torque_disable requires --with_arm_current_control")

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
            live_gripper_relay(
                follower,
                leader,
                args.hz,
                args.with_arm_current_control,
                args.arm_current_ma,
                args.gripper_current_limit_ma,
                args.scope_arm_torque_disable,
            )
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
