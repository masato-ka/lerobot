#!/usr/bin/env python3
"""Live demo: reproduce FACTR2's NEXT external-torque estimation (arXiv:2606.12406) on the
OMX follower arm.

Loads a trained checkpoint (see `train_next.py`) and streams estimated external torque per
joint while the arm holds a fixed pose. Push on each joint by hand to see the estimate react.
This is a read-only sensing validation demo: no force feedback is rendered anywhere (bilateral
teleoperation is a future phase — see src/lerobot/force_estimation/README.md).

Usage (run from repo root):
    python -m examples.omx.force_sensing.demo_force_sensing \\
        --port /dev/ttyACM0 --checkpoint checkpoints/omx_next.pt
"""

import argparse
import logging
import time

from lerobot.force_estimation import OnlineExternalTorqueEstimator
from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig
from lerobot.utils.robot_utils import precise_sleep

from .common import ARM_JOINTS

logger = logging.getLogger(__name__)


def read_state(robot: OmxFollower) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    pos = robot.bus.sync_read("Present_Position")
    vel = robot.bus.sync_read("Present_Velocity")
    cur = robot.bus.sync_read("Present_Current")
    return (
        {j: pos[j] for j in ARM_JOINTS},
        {j: vel[j] for j in ARM_JOINTS},
        {j: cur[j] for j in ARM_JOINTS},
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--robot_id", default="omx_follower")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint saved by train_next.py")
    parser.add_argument("--hz", type=float, default=100.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    robot = OmxFollower(OmxFollowerConfig(port=args.port, id=args.robot_id))
    robot.connect(calibrate=True)
    estimator = OnlineExternalTorqueEstimator(args.checkpoint)

    dt = 1.0 / args.hz
    try:
        # Hold the current pose steady: goal_q stays fixed so delta_q_d reflects deviation
        # caused by external force rather than by commanded motion.
        obs = robot.get_observation()
        hold_pose = {j: obs[f"{j}.pos"] for j in ARM_JOINTS}
        robot.send_action({f"{j}.pos": v for j, v in hold_pose.items()})

        print("Holding pose. Push on the arm to see estimated external torque react (Ctrl+C to stop).")
        while True:
            loop_start = time.perf_counter()
            robot.send_action({f"{j}.pos": v for j, v in hold_pose.items()})
            q, qdot, current = read_state(robot)
            tau_ext = estimator.update(q=q, qdot=qdot, goal_q=hold_pose, current=current)
            if tau_ext is not None:
                readout = "  ".join(f"{j}: {tau_ext[j]:+7.1f}" for j in ARM_JOINTS)
                print(f"\rtau_ext  {readout}", end="", flush=True)
            precise_sleep(max(0.0, dt - (time.perf_counter() - loop_start)))
    except KeyboardInterrupt:
        pass
    finally:
        print()
        robot.disconnect()


if __name__ == "__main__":
    main()
