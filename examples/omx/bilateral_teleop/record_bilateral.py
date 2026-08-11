#!/usr/bin/env python3
"""Record bilateral-teleop demonstrations with the estimated external force folded into
`observation.state`.

Runs the same combined loop as `bilateral_teleop_demo.py` (position teleop + gravity comp +
joint-limit barrier + damping + force feedback), while also recording a LeRobotDataset (v3
format). The follower's native `observation.state` (joint positions) is extended with 5 extra
dims (`force.<joint>`, one per arm joint) holding `tau_ext` at that step, via
`combine_feature_dicts()` -- no changes to `OmxFollower` itself. This choice (vs. a separate
`observation.force` key) means the recorded dataset works with `lerobot-train`'s existing
policies (e.g. ACT) with no policy code changes, since only the `observation.state` key is
picked up as model input automatically; a standalone `observation.force` key would be silently
ignored (see src/lerobot/configs/policies.py `robot_state_feature`).

Episode control is intentionally simple for this experimental stage: each episode runs for a
fixed `--episode_duration_s`, and Ctrl+C stops recording early (saving whatever was captured in
the in-progress episode first). No keyboard-driven start/stop/re-record like `lerobot-record`.

No cameras are attached unless `--cameras` is given -- without it, `observation.state` (position
+ force) is recorded but there is no visual observation, which most policies (ACT included) need
to be useful. `--cameras` takes the same YAML-ish dict-of-dataclass syntax as
`examples/omx/record_grab.py`'s `--robot.cameras=...`, decoded via `draccus.decode(dict[str,
CameraConfig], ...)` so any registered camera backend (not just OpenCV) works, e.g.:
    --cameras="{ wrist: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: MJPG} }"

Usage (run from repo root):
    python -m examples.omx.bilateral_teleop.record_bilateral \\
        --follower_port /dev/ttyACM0 --leader_port /dev/ttyACM1 \\
        --urdf_path /path/to/omx_l.urdf --checkpoint checkpoints/omx_next.pt \\
        --modifier 0.09 --modifier_shoulder_lift 0.15 \\
        --damping_gain 0.15 --joint_limit_kp 3 --joint_limit_kd 0 --feedback_gain -0.3 \\
        --repo_id <hf_username>/omx_bilateral_force --root data/omx_bilateral_force \\
        --num_episodes 10 --episode_duration_s 30 --single_task "Pick up the cube" \\
        --cameras="{ wrist: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: MJPG}, top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: MJPG} }" \\
        --push_to_hub --hub_private --hub_tags omx bilateral force
"""

import argparse
import logging
import time

import draccus
import yaml

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401  registers the "opencv" choice
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.force_estimation import OnlineExternalTorqueEstimator
from lerobot.processor import make_default_processors
from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig
from lerobot.teleoperators.omx_leader import OmxLeader, OmxLeaderConfig
from lerobot.teleoperators.omx_leader.gravity_compensation import ARM_JOINTS, OmxGravityModel
from lerobot.teleoperators.omx_leader.leader_safety import (
    JOINT_LIMIT_RANGE,
    KT_NM_PER_A,
    compute_damping_torque,
    compute_joint_limit_torque,
    enter_current_control_mode,
    resolve_modifiers,
    resolve_per_joint,
    restore_position_mode,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts

from .bilateral_teleop_demo import add_leader_control_args

logger = logging.getLogger(__name__)


def parse_cameras(raw: str | None) -> dict[str, CameraConfig]:
    """Parse a `--cameras` value (record_grab.py-style YAML-ish dict-of-dataclass string) into
    `{name: CameraConfig}`. Returns `{}` if `raw` is `None`."""
    if raw is None:
        return {}
    return draccus.decode(dict[str, CameraConfig], yaml.safe_load(raw))


def build_dataset_features(follower: OmxFollower, use_videos: bool) -> dict:
    teleop_action_processor, _, robot_obs_processor = make_default_processors()
    force_features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(ARM_JOINTS),),
            "names": [f"force.{j}" for j in ARM_JOINTS],
        }
    }
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=follower.action_features),
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_obs_processor,
            initial_features=create_initial_features(observation=follower.observation_features),
            use_videos=use_videos,
        ),
        force_features,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--follower_port", default="/dev/ttyACM0")
    parser.add_argument("--follower_id", default="omx_follower")
    parser.add_argument(
        "--cameras",
        default=None,
        help=(
            "USB camera(s) to attach to the follower, e.g. "
            "'{ wrist: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: MJPG} }'. "
            "Omit to record without any visual observation."
        ),
    )
    parser.add_argument("--leader_port", default="/dev/ttyACM1")
    parser.add_argument("--leader_id", default="omx_leader")
    parser.add_argument("--urdf_path", required=True, help="Path to a local copy of omx_l.urdf")
    parser.add_argument("--checkpoint", required=True, help="Path to a NEXT checkpoint from train_next.py")
    add_leader_control_args(parser)
    parser.add_argument("--current_limit_ma", type=int, default=500, help="Hard per-joint current cap")
    parser.add_argument(
        "--feedback_limit_ma", type=int, default=200, help="Hard per-joint cap on the feedback term alone"
    )
    parser.add_argument("--hz", type=float, default=50.0, help="Control loop rate")

    parser.add_argument("--repo_id", required=True, help="e.g. <hf_username>/<dataset_name>")
    parser.add_argument("--root", default=None, help="Local dataset directory (defaults to HF cache)")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--episode_duration_s", type=float, default=30.0)
    parser.add_argument("--single_task", required=True, help="Short description of the demonstrated task")
    parser.add_argument(
        "--fps", type=int, default=30, help="Recorded dataset fps (independent of --hz, the control rate)"
    )
    parser.add_argument("--no_video", action="store_true", help="Store camera frames as images, not video")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Upload the finished dataset to the Hugging Face Hub once recording ends",
    )
    parser.add_argument(
        "--hub_private", action="store_true", help="Create the Hub repo as private (only with --push_to_hub)"
    )
    parser.add_argument(
        "--hub_tags", nargs="+", default=None, help="Tags for the Hub dataset card (only with --push_to_hub)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    modifiers = resolve_modifiers(args)
    damping_gains = resolve_per_joint(args, "damping_gain", args.damping_gain)
    joint_limit_kp = resolve_per_joint(args, "joint_limit_kp", args.joint_limit_kp)
    joint_limit_kd = resolve_per_joint(args, "joint_limit_kd", args.joint_limit_kd)
    feedback_gains = resolve_per_joint(args, "feedback_gain", args.feedback_gain)

    cameras = parse_cameras(args.cameras)
    if not cameras:
        logger.warning("No --cameras given; recording without any visual observation.")
    follower = OmxFollower(OmxFollowerConfig(port=args.follower_port, id=args.follower_id, cameras=cameras))
    leader = OmxLeader(OmxLeaderConfig(port=args.leader_port, id=args.leader_id))
    follower.connect(calibrate=True)
    leader.connect(calibrate=True)

    gravity_model = OmxGravityModel(args.urdf_path)
    estimator = OnlineExternalTorqueEstimator(args.checkpoint)

    use_videos = not args.no_video
    dataset_features = build_dataset_features(follower, use_videos)
    num_cameras = len(follower.cameras) if hasattr(follower, "cameras") else 0
    dataset = LeRobotDataset.create(
        args.repo_id,
        args.fps,
        root=args.root,
        robot_type=follower.name,
        features=dataset_features,
        use_videos=use_videos,
        image_writer_processes=0,
        image_writer_threads=4 * num_cameras if num_cameras > 0 else 0,
    )
    logger.info(f"observation.state: {dataset.features['observation.state']}")

    dt = 1.0 / args.hz
    zero_tau_ext = dict.fromkeys(ARM_JOINTS, 0.0)

    try:
        enter_current_control_mode(leader, args.current_limit_ma)
        print(
            f"Recording {args.num_episodes} episode(s) of ~{args.episode_duration_s}s each. "
            "Ctrl+C to stop early (the in-progress episode is still saved).\n"
        )
        with VideoEncodingManager(dataset):
            for episode_idx in range(args.num_episodes):
                print(f"=== Episode {episode_idx + 1}/{args.num_episodes} ===")
                episode_end = time.perf_counter() + args.episode_duration_s
                frames_this_episode = 0
                try:
                    while time.perf_counter() < episode_end:
                        loop_start = time.perf_counter()

                        # Leader position -> follower (position teleop), gripper included.
                        leader_pos = leader.bus.sync_read("Present_Position")
                        leader_vel = leader.bus.sync_read("Present_Velocity")
                        q_leader = {j: leader_pos[j] for j in ARM_JOINTS}
                        qdot_leader = {j: leader_vel[j] for j in ARM_JOINTS}
                        follower_action = {f"{j}.pos": q_leader[j] for j in ARM_JOINTS}
                        follower_action["gripper.pos"] = leader_pos["gripper"]
                        sent_action = follower.send_action(follower_action)

                        # Follower observation (position + cameras) -> tau_ext. Present_Position
                        # comes from `obs` itself, avoiding a second read of the same register.
                        obs = follower.get_observation()
                        follower_vel = follower.bus.sync_read("Present_Velocity")
                        follower_cur = follower.bus.sync_read("Present_Current")
                        tau_ext = estimator.update(
                            q={j: obs[f"{j}.pos"] for j in ARM_JOINTS},
                            qdot={j: follower_vel[j] for j in ARM_JOINTS},
                            goal_q=q_leader,
                            current={j: follower_cur[j] for j in ARM_JOINTS},
                        )
                        if tau_ext is None:  # history buffer still filling
                            tau_ext = zero_tau_ext

                        # Leader gravity + joint-limit + damping + feedback (same as
                        # bilateral_teleop_demo.py).
                        tau_g = gravity_model.compute_gravity_torque(q_leader)
                        tau_limit_ma = compute_joint_limit_torque(
                            q_leader, qdot_leader, JOINT_LIMIT_RANGE, joint_limit_kp, joint_limit_kd
                        )
                        tau_damping_ma = compute_damping_torque(qdot_leader, damping_gains)

                        goal_current_ma = {}
                        for joint in ARM_JOINTS:
                            gravity_ma = (tau_g[joint] / KT_NM_PER_A) * modifiers[joint] * 1000.0
                            feedback_ma = feedback_gains[joint] * tau_ext[joint]
                            feedback_ma = max(
                                -args.feedback_limit_ma, min(args.feedback_limit_ma, feedback_ma)
                            )
                            total_ma = gravity_ma + tau_limit_ma[joint] + tau_damping_ma[joint] + feedback_ma
                            total_ma = max(-args.current_limit_ma, min(args.current_limit_ma, total_ma))
                            goal_current_ma[joint] = int(total_ma)
                        leader.bus.sync_write("Goal_Current", goal_current_ma)

                        # Merge force into the observation dict and record the frame.
                        for j in ARM_JOINTS:
                            obs[f"force.{j}"] = tau_ext[j]
                        obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
                        action_frame = build_dataset_frame(dataset.features, sent_action, prefix=ACTION)
                        dataset.add_frame({**obs_frame, **action_frame, "task": args.single_task})
                        frames_this_episode += 1

                        elapsed = time.perf_counter() - loop_start
                        time.sleep(max(0.0, dt - elapsed))
                except KeyboardInterrupt:
                    if frames_this_episode > 0:
                        dataset.save_episode()
                        print(f"\nStopped early; saved partial episode {episode_idx + 1}.")
                    else:
                        print(f"\nStopped early before episode {episode_idx + 1} captured any frames.")
                    raise

                dataset.save_episode()
                print(f"Episode {episode_idx + 1} saved.")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            restore_position_mode(leader)
        finally:
            leader.disconnect()
            follower.disconnect()
        dataset.finalize()

    if args.push_to_hub and dataset.num_episodes > 0:
        print(f"Pushing {dataset.num_episodes} episode(s) to the Hub: {args.repo_id}")
        dataset.push_to_hub(tags=args.hub_tags, private=args.hub_private or None)
    elif args.push_to_hub:
        print("Skipping push_to_hub: no episodes were recorded.")

    return dataset


if __name__ == "__main__":
    main()
