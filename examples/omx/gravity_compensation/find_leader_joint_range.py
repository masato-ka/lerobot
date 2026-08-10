#!/usr/bin/env python3
"""Measure a safe joint range for the omx_leader arm, for use as `JOINT_LIMIT_RANGE` in
`gravity_comp_demo.py`.

Disables leader motor torque (fully backdrivable) and tracks the running min/max normalized
position per arm joint while you move it by hand. Move each joint slowly through the range you
consider safe to operate in day-to-day (not necessarily the full mechanical range out to the
hard stops -- leave some margin) one at a time, then Ctrl+C to print a summary.

Usage (run from repo root):
    python -m examples.omx.gravity_compensation.find_leader_joint_range \\
        --port /dev/ttyACM1 --robot_id omx_leader
"""

import argparse
import logging
import time

from lerobot.teleoperators.omx_leader import OmxLeader, OmxLeaderConfig
from lerobot.teleoperators.omx_leader.gravity_compensation import ARM_JOINTS

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", default="/dev/ttyACM1")
    parser.add_argument("--robot_id", default="omx_leader")
    parser.add_argument("--hz", type=float, default=20.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    leader = OmxLeader(OmxLeaderConfig(port=args.port, id=args.robot_id))
    leader.connect(calibrate=True)

    dt = 1.0 / args.hz
    q_min = dict.fromkeys(ARM_JOINTS, float("inf"))
    q_max = dict.fromkeys(ARM_JOINTS, float("-inf"))
    try:
        leader.bus.disable_torque()
        print("Torque disabled -- slowly move each joint through its safe range. Ctrl+C to stop.\n")
        while True:
            pos = leader.bus.sync_read("Present_Position")
            for j in ARM_JOINTS:
                q_min[j] = min(q_min[j], pos[j])
                q_max[j] = max(q_max[j], pos[j])

            row = "  ".join(f"{j}=[{q_min[j]:+6.1f}, {q_max[j]:+6.1f}]" for j in ARM_JOINTS)
            print(f"\r{row}", end="", flush=True)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nJOINT_LIMIT_RANGE = {")
        for j in ARM_JOINTS:
            print(f'    "{j}": ({q_min[j]:.1f}, {q_max[j]:.1f}),')
        print("}")
        leader.disconnect()


if __name__ == "__main__":
    main()
