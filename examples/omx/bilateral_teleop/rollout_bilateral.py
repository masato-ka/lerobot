#!/usr/bin/env python3
"""DEPRECATED: `lerobot-rollout`'s extension-point gap this script worked around (its CLI having
no way to carry `observation.state` dims beyond `.pos`/`.vel`) has been closed --
`build_rollout_context()` (`src/lerobot/rollout/context.py`) now also keeps `force.<joint>`
scalar features, and `OmxFollower` computes them internally when `force_estimation` is
configured. Prefer:

    lerobot-rollout \\
        --robot.type=omx_follower --robot.port=/dev/ttyACM0 \\
        --robot.force_estimation.checkpoint_path=checkpoints/omx_next.pt \\
        --policy.path=<path/to/trained/policy> --strategy.type=base

This also removes the `goal_q` train/inference split documented below -- `OmxFollower` tracks
`_last_sent_goal_q` uniformly from `send_action()`, so record time and rollout time now share one
definition instead of "leader's live position" vs. "policy's own previous action". See
`src/lerobot/force_estimation/README.md` and `examples/omx/TECHNICAL_REPORT_ja.md` §2.2/Step 7
for the full command reference. This script is kept for reference and is not expected to receive
further changes.

Run a trained policy autonomously on the omx_follower arm, with the estimated external force
(NEXT `tau_ext`) folded into `observation.state` exactly as `record_bilateral.py` records it.

Standalone, follower-only rollout (no leader, no dataset recording of the rollout itself).
`lerobot-rollout`'s CLI has no extension point for `observation.state` dims beyond
`robot.observation_features`, so the custom `force.<joint>` dims recorded by
`record_bilateral.py` can't be reproduced through it as-is -- this script instead follows the
same raw-inference pattern shown in `examples/so100_to_so100_EE/evaluate.py`
("For production policy deployment, use `lerobot-rollout` CLI instead" -- that applies once a
policy trained on a *standard* observation.state is all you need; ours isn't standard).

Per control step:
  1. Read the follower's observation (position + cameras).
  2. Feed q/qdot/current + the *previous* step's sent action (as `goal_q`) through
     `OnlineExternalTorqueEstimator` to get `tau_ext`, and merge it into the observation as
     `force.<joint>`. The feature schema (which names belong in `observation.state`, in what
     order) comes from `record_bilateral.build_dataset_features()` -- imported, not
     reimplemented, so it's structurally guaranteed to match what the policy was trained on.
  3. Run the policy: `build_dataset_frame` -> `predict_action` (preprocess, `select_action`,
     postprocess) -> `make_robot_action`.
  4. Send the resulting action to the follower, and remember it as next step's `goal_q`.

Since there's no leader, `goal_q` at inference time is redefined as "the position we actually
last commanded" (the policy's own previous action) rather than "the leader's current position"
used at recording time -- the natural analogue once the policy replaces the human leader as
goal-generator. This is a documented approximation, not an exact train/inference match (same
class of issue as NEXT's own train/inference sampling-rate mismatch, see
src/lerobot/force_estimation/README.md's tuning section).

SAFETY: the follower stays in its normal EXTENDED_POSITION mode throughout (a firmware position
servo) -- unlike the leader's Current Control Mode, a bad policy output can't make it "run away"
with unbounded current, but it can still command large/fast position jumps. Set
`OmxFollowerConfig.max_relative_target` conservatively and start with a small `--num_steps`,
hand near the arm, before running longer.

`--device` defaults to cuda if available, else cpu -- it deliberately does *not* auto-select
`mps` even on Apple Silicon, since at least ACTPolicy was found (by hand, while grounding this
script) to hit a real device-placement bug in its VAE-disabled eval path on the MPS backend.
Pass `--device mps` explicitly if you want to try it anyway; `--device cpu` is the known-good
fallback on a Mac.

Usage (run from repo root):
    python -m examples.omx.bilateral_teleop.rollout_bilateral \\
        --follower_port /dev/ttyACM0 --checkpoint checkpoints/omx_next.pt \\
        --policy_path <path-or-hf-repo-id-of-trained-policy> \\
        --task "Pick up the cube" --num_steps 200 \\
        --cameras="{ wrist: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: MJPG} }"
"""

import argparse
import logging
import time

import torch

from lerobot.common.control_utils import predict_action
from lerobot.configs import PreTrainedConfig
from lerobot.force_estimation import OnlineExternalTorqueEstimator
from lerobot.policies import get_policy_class, make_pre_post_processors, make_robot_action
from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig
from lerobot.teleoperators.omx_leader.gravity_compensation import ARM_JOINTS
from lerobot.utils.constants import OBS_STR
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.feature_utils import build_dataset_frame

from .record_bilateral import build_dataset_features, parse_cameras

logger = logging.getLogger(__name__)


def resolve_device(requested: str | None) -> torch.device:
    """`--device` override if given, else cuda if available, else cpu.

    Deliberately does *not* auto-select `mps` even when available: at least ACTPolicy hits a
    real device-placement bug on the MPS backend at inference (`encoder_latent_input_proj`
    weight/input device mismatch in `modeling_act.py`'s VAE-disabled eval path, reproduced by
    hand while grounding this script -- fails whether the checkpoint was trained on cpu or on
    mps, in both load directions). `--device mps` is still honored if passed explicitly.
    """
    if requested is not None:
        return get_safe_torch_device(requested, log=True)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
            "Same --cameras syntax as record_bilateral.py. Should match what the policy was "
            "trained with, e.g. "
            "'{ wrist: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: MJPG} }'."
        ),
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a NEXT checkpoint from train_next.py")
    parser.add_argument(
        "--policy_path", required=True, help="Trained policy checkpoint directory or HF Hub repo_id"
    )
    parser.add_argument("--task", default="", help="Task string, same as --single_task used to record")
    parser.add_argument(
        "--force_smoothing_alpha",
        type=float,
        default=None,
        help=(
            "Optional EMA smoothing on tau_ext (see src/lerobot/force_estimation/online.py). "
            "Disabled (raw tau_ext) by default."
        ),
    )
    parser.add_argument(
        "--device", default=None, help="Override the policy's saved device (e.g. cpu, mps, cuda)"
    )
    parser.add_argument("--hz", type=float, default=50.0, help="Control loop rate")
    parser.add_argument(
        "--num_steps", type=int, default=200, help="Stop after this many steps (Ctrl+C also stops early)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cameras = parse_cameras(args.cameras)
    if not cameras:
        logger.warning("No --cameras given; running with no visual observation.")
    follower = OmxFollower(OmxFollowerConfig(port=args.follower_port, id=args.follower_id, cameras=cameras))
    follower.connect(calibrate=True)

    estimator = OnlineExternalTorqueEstimator(args.checkpoint, smoothing_alpha=args.force_smoothing_alpha)
    dataset_features = build_dataset_features(follower, use_videos=len(cameras) > 0)

    device = resolve_device(args.device)
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.device = str(device)
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(args.policy_path, config=policy_cfg)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=args.policy_path
    )
    policy.reset()

    initial_obs = follower.get_observation()
    last_goal_q = {j: initial_obs[f"{j}.pos"] for j in ARM_JOINTS}

    dt = 1.0 / args.hz
    zero_tau_ext = dict.fromkeys(ARM_JOINTS, 0.0)
    try:
        print(
            f"Running policy from {args.policy_path} on device={device} for up to "
            f"{args.num_steps} steps. Ctrl+C to stop early.\n"
        )
        for step in range(args.num_steps):
            loop_start = time.perf_counter()

            obs = follower.get_observation()
            follower_vel = follower.bus.sync_read("Present_Velocity")
            follower_cur = follower.bus.sync_read("Present_Current")
            tau_ext = estimator.update(
                q={j: obs[f"{j}.pos"] for j in ARM_JOINTS},
                qdot={j: follower_vel[j] for j in ARM_JOINTS},
                goal_q=last_goal_q,
                current={j: follower_cur[j] for j in ARM_JOINTS},
            )
            if tau_ext is None:  # history buffer still filling
                tau_ext = zero_tau_ext
            for j in ARM_JOINTS:
                obs[f"force.{j}"] = tau_ext[j]

            frame = build_dataset_frame(dataset_features, obs, prefix=OBS_STR)
            action_tensor = predict_action(
                observation=frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=False,
                task=args.task,
                robot_type=follower.name,
            )
            action_dict = make_robot_action(action_tensor, dataset_features)
            sent = follower.send_action(action_dict)
            last_goal_q = {j: sent[f"{j}.pos"] for j in ARM_JOINTS}

            if step % 10 == 0:
                tau_ext_row = "  ".join(f"{j}={tau_ext[j]:+7.1f}" for j in ARM_JOINTS)
                print(f"[step {step}/{args.num_steps}] tau_ext  {tau_ext_row}")

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, dt - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        print()
        follower.disconnect()


if __name__ == "__main__":
    main()
