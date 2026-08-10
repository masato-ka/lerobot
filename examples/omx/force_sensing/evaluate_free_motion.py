#!/usr/bin/env python3
"""Sanity-check a trained NEXT checkpoint by replaying free-motion logs through it.

Since the replayed data is itself contact-free (see `collect_free_motion.py`), a
well-fit model should report `tau_ext` close to zero (noise only) throughout. This is
the "run inference on the training data" validation described in
`src/lerobot/force_estimation/README.md`. Passing held-out logs (not used for training)
here is a stronger check than passing the training data back in.

Runs `OnlineExternalTorqueEstimator` exactly as `demo_force_sensing.py` does (same ring
buffer, one step at a time), so the numbers reflect real online-inference behavior rather
than a batched offline computation.

Usage (run from repo root):
    python -m examples.omx.force_sensing.evaluate_free_motion \\
        --checkpoint checkpoints/omx_next.pt \\
        --data data/omx_free_motion/run1.npz data/omx_free_motion/run2.npz
"""

import argparse
import logging

import numpy as np

from lerobot.force_estimation import (
    FreeMotionEpisode,
    OnlineExternalTorqueEstimator,
    load_episode,
    resample_uniform,
)

logger = logging.getLogger(__name__)


def evaluate_episode(estimator: OnlineExternalTorqueEstimator, episode: FreeMotionEpisode) -> np.ndarray:
    """Replay one resampled episode through `estimator`, step by step.

    Returns:
        `np.ndarray`: Shape `(T - history_length + 1, num_joints)` of `tau_ext` estimates, in
        `estimator.joint_names` order.
    """
    estimator.reset()
    joint_names = episode.joint_names
    results = []
    for i in range(len(episode.t)):
        q = {j: float(episode.q[i, k]) for k, j in enumerate(joint_names)}
        qdot = {j: float(episode.qdot[i, k]) for k, j in enumerate(joint_names)}
        goal_q = {j: float(episode.goal_q[i, k]) for k, j in enumerate(joint_names)}
        current = {j: float(episode.current[i, k]) for k, j in enumerate(joint_names)}
        tau_ext = estimator.update(q=q, qdot=qdot, goal_q=goal_q, current=current)
        if tau_ext is not None:
            results.append([tau_ext[j] for j in estimator.joint_names])
    return np.array(results)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint saved by train_next.py")
    parser.add_argument("--data", nargs="+", required=True, help="Path(s) to .npz free-motion logs to replay")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    estimator = OnlineExternalTorqueEstimator(args.checkpoint)

    all_tau_ext = []
    for path in args.data:
        episode = resample_uniform(load_episode(path), estimator.resample_hz)
        tau_ext = evaluate_episode(estimator, episode)
        logger.info(f"{path}: {len(tau_ext)} evaluated steps")
        all_tau_ext.append(tau_ext)

    tau_ext = np.concatenate(all_tau_ext, axis=0)
    print(f"\n{len(tau_ext)} total steps evaluated across {len(args.data)} episode(s).")
    print("If this data is contact-free, tau_ext should be close to 0 (noise only) below.\n")
    print(f"{'joint':<15}{'mean':>10}{'std':>10}{'max|.|':>10}")
    for i, joint in enumerate(estimator.joint_names):
        col = tau_ext[:, i]
        print(f"{joint:<15}{col.mean():>10.2f}{col.std():>10.2f}{np.abs(col).max():>10.2f}")


if __name__ == "__main__":
    main()
