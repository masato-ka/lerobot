#!/usr/bin/env python3
"""Active gravity compensation for the omx_leader arm.

Switches the arm's 5 joints (gripper excluded) to Dynamixel Current Control Mode and
continuously commands `Goal_Current` from the RNEA-computed gravity torque
(`OmxGravityModel`), scaled by a nominal XL330 torque constant and a tunable
`--modifier` gain -- so the arm should stay wherever it's placed instead of falling under
its own weight.

SAFETY -- read before running:
  1. Run `preview_gravity_model.py` first and confirm the printed torques/angles look
     physically sensible for your arm. This script trusts that mapping blindly.
  2. Support the arm by hand before starting. `--modifier` defaults to a conservative 0.09
     (per-joint requirements vary a lot -- e.g. shoulder_lift typically needs noticeably more
     than the rest, wrist_roll needs close to none); increase gradually while feeling whether
     the arm gets lighter, rather than jumping to a large value.
  3. `--current_limit_ma` is a hard per-joint ceiling independent of `--modifier`, in case the
     gain/sign mapping is wrong. Keep it conservative until you've validated behavior.
  4. Ctrl+C (or any exception) always disables torque and restores the arm to
     EXTENDED_POSITION mode before exiting.

Per-joint tuning: `--modifier` sets the default gain for every joint; `--modifier_<joint>`
(e.g. `--modifier_shoulder_lift 0.15`) overrides it for one joint. `wrist_roll` defaults to
`0.0` (its gravity torque is close to zero at every pose, and any nonzero current there tends
to just get in the way of manual operation) -- pass `--modifier_wrist_roll` explicitly if you
want it compensated too.

Usage (run from repo root):
    python -m examples.omx.gravity_compensation.gravity_comp_demo \\
        --port /dev/ttyACM1 --robot_id omx_leader \\
        --urdf_path /path/to/omx_l.urdf \\
        --modifier 0.09 --modifier_shoulder_lift 0.15
"""

import argparse
import logging
import time

from lerobot.motors.dynamixel import OperatingMode
from lerobot.teleoperators.omx_leader import OmxLeader, OmxLeaderConfig
from lerobot.teleoperators.omx_leader.gravity_compensation import ARM_JOINTS, OmxGravityModel

logger = logging.getLogger(__name__)

# Nominal XL330 torque constant (Nm per A), derived from ROBOTIS's published stall
# torque/stall current figures (~0.35-0.38 Nm/A across the 3.7-6.0V range). XL330's
# Present_Current/Goal_Current is input-supply current rather than true phase current, so
# treat this as a starting point for `--modifier` tuning, not a precise calibration.
KT_NM_PER_A = 0.36

# wrist_roll's gravity torque is close to zero at essentially every pose (see
# OmxGravityModel docstring/README), and applying any noticeable current there mostly just
# resists manual operation -- so it defaults to uncompensated unless explicitly overridden.
DEFAULT_JOINT_MODIFIER_OVERRIDES: dict[str, float] = {"wrist_roll": 0.0}


def resolve_modifiers(args: argparse.Namespace) -> dict[str, float]:
    """Per-joint modifier = its `--modifier_<joint>` CLI override if given, else
    `DEFAULT_JOINT_MODIFIER_OVERRIDES` if set for that joint, else the global `--modifier`."""
    modifiers = {}
    for joint in ARM_JOINTS:
        cli_override = getattr(args, f"modifier_{joint}")
        if cli_override is not None:
            modifiers[joint] = cli_override
        elif joint in DEFAULT_JOINT_MODIFIER_OVERRIDES:
            modifiers[joint] = DEFAULT_JOINT_MODIFIER_OVERRIDES[joint]
        else:
            modifiers[joint] = args.modifier
    return modifiers


def enter_current_control_mode(leader: OmxLeader, current_limit_ma: int) -> None:
    # torque_disabled() re-enables torque for every motor on exit, which is what we want here:
    # writing Operating_Mode requires torque off, and we want it back on afterward to drive
    # Goal_Current.
    with leader.bus.torque_disabled():
        for joint in ARM_JOINTS:
            leader.bus.write("Operating_Mode", joint, OperatingMode.CURRENT.value)
            leader.bus.write("Current_Limit", joint, current_limit_ma)


def restore_position_mode(leader: OmxLeader) -> None:
    # Leave torque disabled on exit (don't use torque_disabled(), which would re-enable it) --
    # this is cleanup, the arm should be safe to walk away from afterward.
    leader.bus.disable_torque()
    for joint in ARM_JOINTS:
        leader.bus.write("Operating_Mode", joint, OperatingMode.EXTENDED_POSITION.value)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", default="/dev/ttyACM1")
    parser.add_argument("--robot_id", default="omx_leader")
    parser.add_argument("--urdf_path", required=True, help="Path to a local copy of omx_l.urdf")
    parser.add_argument("--modifier", type=float, default=0.09, help="Default gravity-comp gain")
    for joint in ARM_JOINTS:
        parser.add_argument(
            f"--modifier_{joint}",
            type=float,
            default=None,
            help=f"Gravity-comp gain override for {joint} (defaults to --modifier)",
        )
    parser.add_argument("--current_limit_ma", type=int, default=500, help="Hard per-joint current cap")
    parser.add_argument("--hz", type=float, default=50.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    modifiers = resolve_modifiers(args)

    leader = OmxLeader(OmxLeaderConfig(port=args.port, id=args.robot_id))
    leader.connect(calibrate=True)
    gravity_model = OmxGravityModel(args.urdf_path)

    dt = 1.0 / args.hz
    try:
        enter_current_control_mode(leader, args.current_limit_ma)
        modifier_row = "  ".join(f"{j}={modifiers[j]}" for j in ARM_JOINTS)
        print(
            f"Current Control Mode enabled (current_limit={args.current_limit_ma}mA).\n"
            f"modifiers: {modifier_row}\nCtrl+C to stop.\n"
        )
        while True:
            pos = leader.bus.sync_read("Present_Position")
            q_lerobot = {j: pos[j] for j in ARM_JOINTS}
            tau_g = gravity_model.compute_gravity_torque(q_lerobot)

            goal_current_ma = {}
            for joint in ARM_JOINTS:
                current_a = (tau_g[joint] / KT_NM_PER_A) * modifiers[joint]
                current_ma = current_a * 1000.0
                current_ma = max(-args.current_limit_ma, min(args.current_limit_ma, current_ma))
                goal_current_ma[joint] = int(current_ma)
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
