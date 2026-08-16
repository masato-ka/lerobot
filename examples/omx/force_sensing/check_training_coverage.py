#!/usr/bin/env python3
"""Check how densely `collect_free_motion.py`'s training logs cover q-space, per joint --
especially `shoulder_lift`/`elbow_flex`, the coupled joint pair NEXT's pose-dependent bias
concentrates in (see `evaluate_pose_dependence.py`). A checkpoint retrained on this data can
only reduce bias in regions the data actually visited; sparse or empty regions are where a
pose-dependent residual is most likely to persist -- or even worsen after retraining, if the
model ends up overfitting the dense regions at the expense of the sparse ones (this is the
suspected explanation for `wrist_flex`'s detection signal improving at the "initial" test pose
but vanishing at "long" after retraining -- see the checkpoint-comparison results).

Usage (run from repo root):
    python -m examples.omx.force_sensing.check_training_coverage \\
        --data data/omx_free_motion/run1.npz data/omx_free_motion/run2.npz \\
        --bins 20

Pass `--reference` (one or more `collect_static_hold.py` logs) to check specific test poses
against the training distribution directly -- reports, per joint, how many training samples
fall within `--tolerance` of that pose's mean q, plus the joint density for the
`shoulder_lift`/`elbow_flex` pair specifically (both within tolerance simultaneously), since a
joint being individually well-covered doesn't mean the *combination* was:
    python -m examples.omx.force_sensing.check_training_coverage \\
        --data data/omx_free_motion/run1.npz data/omx_free_motion/run2.npz \\
        --reference data/omx_static_hold/N1_long_unloaded.npz data/omx_static_hold/N1_initial_unloaded.npz
"""

import argparse
import logging

import numpy as np

from lerobot.force_estimation import load_episode

logger = logging.getLogger(__name__)


def histogram_report(q: np.ndarray, joint_names: list[str], bins: int) -> None:
    """Print a per-joint ASCII histogram of `q` over its own observed range, flagging bins that
    are empty or hold under 2% of that joint's samples -- q-space the training data barely (or
    never) visited.
    """
    for i, joint in enumerate(joint_names):
        col = q[:, i]
        lo, hi = float(col.min()), float(col.max())
        counts, edges = np.histogram(col, bins=bins, range=(lo, hi))
        print(f"{joint}  (range [{lo:.2f}, {hi:.2f}], n={len(col)})")
        max_count = int(counts.max()) if counts.max() > 0 else 1
        sparse_threshold = 0.02 * len(col)
        for c, e0, e1 in zip(counts, edges[:-1], edges[1:], strict=True):
            bar = "#" * int(40 * c / max_count)
            flag = "  <-- EMPTY" if c == 0 else ("  <-- sparse" if c < sparse_threshold else "")
            print(f"  [{e0:8.2f}, {e1:8.2f})  n={c:6d}  {bar}{flag}")
        print()


def reference_density(q: np.ndarray, joint_names: list[str], ref_q: dict[str, float], tolerance: float) -> None:
    """Report, per joint, how many training samples fall within `tolerance` of `ref_q`'s value --
    plus the `shoulder_lift`/`elbow_flex` joint density specifically, since each joint being
    individually well-covered doesn't mean the training data ever visited that *combination*.
    """
    print(f"Training-sample density within +/-{tolerance:g} of this reference pose, per joint:")
    for i, joint in enumerate(joint_names):
        if joint not in ref_q:
            continue
        col = q[:, i]
        near = np.abs(col - ref_q[joint]) < tolerance
        print(f"  {joint:<15}ref={ref_q[joint]:8.2f}  {near.sum():6d}/{len(col)} within +/-{tolerance:g}")

    if "shoulder_lift" in ref_q and "elbow_flex" in ref_q:
        sl_idx = joint_names.index("shoulder_lift")
        ef_idx = joint_names.index("elbow_flex")
        near_sl = np.abs(q[:, sl_idx] - ref_q["shoulder_lift"]) < tolerance
        near_ef = np.abs(q[:, ef_idx] - ref_q["elbow_flex"]) < tolerance
        both = near_sl & near_ef
        print(
            f"  {'shoulder_lift & elbow_flex jointly':<35}{both.sum():6d}/{len(q)} within +/-{tolerance:g} of BOTH "
            f"(sl={ref_q['shoulder_lift']:.2f}, ef={ref_q['elbow_flex']:.2f})"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", required=True, help="Training free-motion .npz logs (collect_free_motion.py output)")
    parser.add_argument("--bins", type=int, default=20, help="Histogram bins per joint (default: 20)")
    parser.add_argument("--reference", nargs="+", default=None, help="Optional static-hold .npz log(s) (collect_static_hold.py output) to check coverage against")
    parser.add_argument("--tolerance", type=float, default=5.0, help="Window (native q units) around the reference pose counted as 'covered' (default: 5.0)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    episodes = [load_episode(p) for p in args.data]
    joint_names = episodes[0].joint_names
    for path, ep in zip(args.data, episodes, strict=True):
        if ep.joint_names != joint_names:
            raise ValueError(f"joint_names mismatch: {args.data[0]} has {joint_names}, {path} has {ep.joint_names}")
    q = np.concatenate([ep.q for ep in episodes], axis=0)
    print(f"{len(q)} total training samples across {len(args.data)} file(s).\n")

    histogram_report(q, joint_names, args.bins)

    if args.reference is not None:
        for ref_path in args.reference:
            ref_ep = load_episode(ref_path)
            ref_q = {j: float(ref_ep.q[:, k].mean()) for k, j in enumerate(ref_ep.joint_names)}
            print(f"=== Reference: {ref_path} ===")
            reference_density(q, joint_names, ref_q, args.tolerance)


if __name__ == "__main__":
    main()
