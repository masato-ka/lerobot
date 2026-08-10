#!/usr/bin/env python3
"""Read-only preview of the omx_leader gravity-compensation model.

Disables leader motor torque (the arm goes fully limp/backdrivable) and continuously prints
the computed gravity-compensation torque for the arm's current pose. Move the arm by hand
through a few representative poses (straight up, horizontal, folded) and check that the
printed values change in a physically sensible way -- this is the required first step before
ever running `gravity_comp_demo.py`, which actually writes current commands to the motors.

What to look for:
  - At a pose where the arm is vertical/balanced, torques should be small.
  - At a pose where a joint is holding significant cantilevered weight (e.g. arm extended
    horizontally), that joint's torque magnitude should be clearly larger.
  - The `q (rad)` column should visibly track the direction you move each joint: increasing
    q as you move a joint one way, decreasing as you move it the other way. If a joint's `q`
    moves in the direction you don't expect, its `joint_sign` in the URDF_JOINT_NAMES mapping
    needs to be flipped (see src/lerobot/teleoperators/omx_leader/gravity_compensation.py).

Usage (run from repo root):
    python -m examples.omx.gravity_compensation.preview_gravity_model \\
        --port /dev/ttyACM1 --robot_id omx_leader \\
        --urdf_path /path/to/omx_l.urdf
"""

import argparse
import logging
import time

from lerobot.teleoperators.omx_leader import OmxLeader, OmxLeaderConfig
from lerobot.teleoperators.omx_leader.gravity_compensation import ARM_JOINTS, OmxGravityModel

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", default="/dev/ttyACM1")
    parser.add_argument("--robot_id", default="omx_leader")
    parser.add_argument("--urdf_path", required=True, help="Path to a local copy of omx_l.urdf")
    parser.add_argument("--hz", type=float, default=20.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    leader = OmxLeader(OmxLeaderConfig(port=args.port, id=args.robot_id))
    leader.connect(calibrate=True)
    gravity_model = OmxGravityModel(args.urdf_path)

    dt = 1.0 / args.hz
    try:
        leader.bus.disable_torque()
        print("Torque disabled -- move the arm by hand. Ctrl+C to stop.\n")
        while True:
            pos = leader.bus.sync_read("Present_Position")
            q_lerobot = {j: pos[j] for j in ARM_JOINTS}
            tau_g = gravity_model.compute_gravity_torque(q_lerobot)

            row_q = "  ".join(f"{j}={q_lerobot[j]:+7.1f}" for j in ARM_JOINTS)
            row_tau = "  ".join(f"{j}={tau_g[j]:+7.4f}" for j in ARM_JOINTS)
            print(f"\rq(norm) {row_q}   tau_g(Nm) {row_tau}", end="", flush=True)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        leader.disconnect()


if __name__ == "__main__":
    main()
