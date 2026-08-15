#!/usr/bin/env python3
"""Check whether a trained NEXT checkpoint's free-space noise floor depends on the robot's pose.

Theoretically, a perfectly fit `f_theta(x)` already conditions on `q`, so `tau_ext` should stay
close to 0 in any contact-free pose, not just near the poses `collect_free_motion.py` sampled
most densely. In practice, gaps in training-data coverage, per-joint sensing quality (Present
Load on the XL430 joints vs. true current on the XL330 joints -- see
`src/lerobot/force_estimation/README.md`), or under/overfitting can leave a systematic,
pose-dependent residual bias. This script buckets each joint's own `q` into quantile bins and
reports `tau_ext` noise-floor stats per bin, so a pose-dependent bias shows up as a spread across
bins rather than as a single aggregate number (which is all `evaluate_free_motion.py` reports).

Reuses `evaluate_free_motion.py`'s `evaluate_episode()` replay loop (same ring-buffer semantics
as `demo_force_sensing.py`) and `lerobot.force_estimation`'s log-loading helpers -- no new data
collection is needed, this replays the same `.npz` free-motion logs used for training/eval.

Usage (run from repo root):
    python -m examples.omx.force_sensing.evaluate_pose_dependence \\
        --checkpoint checkpoints/omx_next.pt \\
        --data data/omx_free_motion/run1.npz data/omx_free_motion/run2.npz \\
        --bins 5
"""

import argparse
import logging

import numpy as np

from lerobot.force_estimation import OnlineExternalTorqueEstimator, load_episode, resample_uniform

from .evaluate_free_motion import evaluate_episode

logger = logging.getLogger(__name__)

# See src/lerobot/force_estimation/README.md: shoulder_pan/shoulder_lift/elbow_flex are XL430
# (Present_Current register is actually Present Load -- a coarse PWM-duty-based estimate, not
# true current); wrist_flex/wrist_roll are XL330 (true input-supply current). Noted per-joint
# below so a pose-dependent bias concentrated on the XL430 joints is easy to spot.
MOTOR_FAMILY = {
    "shoulder_pan": "XL430",
    "shoulder_lift": "XL430",
    "elbow_flex": "XL430",
    "wrist_flex": "XL330",
    "wrist_roll": "XL330",
}


def pose_bin_stats(q_col: np.ndarray, tau_col: np.ndarray, n_bins: int) -> list[tuple[float, float, int, float, float, float]]:
    """Bucket `tau_col` by `q_col`'s quantile bins.

    Returns one `(lo, hi, n, mean, std, max|.|)` tuple per bin, in `q_col` order. Bin edges
    collapse (via `np.unique`) when `q_col` barely varies (e.g. a joint that mostly sat at home
    while others were swept), so fewer than `n_bins` rows may come back -- that's expected, not
    an error.
    """
    edges = np.unique(np.quantile(q_col, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        return [(float(edges[0]), float(edges[0]), len(q_col), float(tau_col.mean()), float(tau_col.std()), float(np.abs(tau_col).max()))]
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (q_col >= lo) & (q_col <= hi) if i == len(edges) - 2 else (q_col >= lo) & (q_col < hi)
        bin_tau = tau_col[mask]
        if len(bin_tau) == 0:
            continue
        rows.append(
            (float(lo), float(hi), len(bin_tau), float(bin_tau.mean()), float(bin_tau.std()), float(np.abs(bin_tau).max()))
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint saved by train_next.py")
    parser.add_argument("--data", nargs="+", required=True, help="Path(s) to .npz free-motion logs to replay")
    parser.add_argument("--bins", type=int, default=5, help="Number of quantile bins per joint (default: 5)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    estimator = OnlineExternalTorqueEstimator(args.checkpoint)

    all_tau_ext = []
    all_q = []
    for path in args.data:
        episode = resample_uniform(load_episode(path), estimator.resample_hz)
        tau_ext, q = evaluate_episode(estimator, episode)
        logger.info(f"{path}: {len(tau_ext)} evaluated steps")
        all_tau_ext.append(tau_ext)
        all_q.append(q)

    tau_ext = np.concatenate(all_tau_ext, axis=0)
    q = np.concatenate(all_q, axis=0)
    print(f"\n{len(tau_ext)} total steps evaluated across {len(args.data)} episode(s).")
    print(
        "Per-joint tau_ext noise floor, binned by that joint's own pose (quantile bins). If this "
        "data is contact-free, a well-fit model should show roughly flat mean/std across bins --\n"
        "a bin-to-bin spread indicates a pose-dependent residual, not genuine contact.\n"
    )

    sensitivity: list[tuple[str, float]] = []
    for i, joint in enumerate(estimator.joint_names):
        family = MOTOR_FAMILY.get(joint, "?")
        print(f"{joint} ({family})")
        print(f"  {'q range':<24}{'n':>8}{'mean':>10}{'std':>10}{'max|.|':>10}")
        rows = pose_bin_stats(q[:, i], tau_ext[:, i], args.bins)
        for lo, hi, n, mean, std, max_abs in rows:
            q_range = f"[{lo:.1f}, {hi:.1f}]"
            print(f"  {q_range:<24}{n:>8}{mean:>10.2f}{std:>10.2f}{max_abs:>10.2f}")
        means = [r[3] for r in rows]
        spread = max(means) - min(means) if len(means) > 1 else 0.0
        sensitivity.append((joint, spread))
        print(f"  pose-sensitivity (max bin mean - min bin mean): {spread:.2f}\n")

    print("Joints ranked by pose-sensitivity (most pose-dependent first):")
    for joint, spread in sorted(sensitivity, key=lambda x: -x[1]):
        print(f"  {joint:<15}({MOTOR_FAMILY.get(joint, '?')}){spread:>10.2f}")


if __name__ == "__main__":
    main()
