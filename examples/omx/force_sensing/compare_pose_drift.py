#!/usr/bin/env python3
"""Check whether the robot's actual pose (`q`) differs between the "unloaded" and "loaded" halves
of a `collect_static_hold.py` A/B pair -- a model-independent complement to
`evaluate_free_motion.py`.

Motivation: an "unloaded"/"loaded" pair nominally recorded "at the same pose" is actually two
separate teleoperation sessions (position from scratch each time, or reposition after Ctrl+C),
so the arm's actual joint angles can differ by a non-trivial amount between the two recordings.
Since `tau_ext`'s noise floor is known to be pose-dependent (see `evaluate_pose_dependence.py`),
any such pose mismatch shows up as a `tau_ext` difference between the two logs even with zero
added mass -- indistinguishable, from `evaluate_free_motion.py`'s output alone, from a genuine
load-detection signal. This script reads `q` straight out of the `.npz` logs (no checkpoint
involved) to check directly whether that mismatch is actually happening, and how large it is
relative to the within-hold noise of `q` itself.

Usage (run from repo root):
    python -m examples.omx.force_sensing.compare_pose_drift \\
        --unloaded data/omx_static_hold/N1_zero_unloaded.npz data/omx_static_hold/N1_long_unloaded.npz \\
        --loaded   data/omx_static_hold/N1_zero_loaded.npz   data/omx_static_hold/N1_long_loaded.npz

`--unloaded`/`--loaded` are paired by position (first with first, second with second, ...), so
the two lists must be the same length and in matching order. Pass `--labels` to override the
default label per pair (the unloaded file's stem with `_unloaded` stripped) -- useful when
filenames don't line up neatly (e.g. a typo like `_inital_` vs `_initial_`).
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from lerobot.force_estimation import load_episode

logger = logging.getLogger(__name__)


def pair_stats(unloaded_path: str, loaded_path: str) -> list[tuple[str, float, float, float, float, float, float]]:
    """Per-joint `q` drift between one unloaded/loaded pair.

    Returns one `(joint, mean_u, std_u, mean_l, std_l, delta, ratio)` tuple per joint, where
    `delta = mean_l - mean_u` and `ratio = |delta| / sqrt(std_u**2 + std_l**2)` -- the pose shift
    expressed in units of the combined within-hold noise of `q` (a z-score-like measure: ratio >>
    1 means the two recordings were not at the same pose).
    """
    ep_u = load_episode(unloaded_path)
    ep_l = load_episode(loaded_path)
    if ep_u.joint_names != ep_l.joint_names:
        raise ValueError(f"joint_names mismatch: {unloaded_path} has {ep_u.joint_names}, {loaded_path} has {ep_l.joint_names}")

    rows = []
    for i, joint in enumerate(ep_u.joint_names):
        mean_u, std_u = float(ep_u.q[:, i].mean()), float(ep_u.q[:, i].std())
        mean_l, std_l = float(ep_l.q[:, i].mean()), float(ep_l.q[:, i].std())
        delta = mean_l - mean_u
        combined_std = float(np.sqrt(std_u**2 + std_l**2))
        ratio = abs(delta) / combined_std if combined_std > 1e-9 else float("inf")
        rows.append((joint, mean_u, std_u, mean_l, std_l, delta, ratio))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unloaded", nargs="+", required=True, help="Unloaded .npz paths")
    parser.add_argument("--loaded", nargs="+", required=True, help="Loaded .npz paths, paired by position with --unloaded")
    parser.add_argument("--labels", nargs="+", default=None, help="Optional label per pair (default: unloaded file's stem, minus '_unloaded')")
    parser.add_argument("--flag_ratio", type=float, default=3.0, help="Ratio above which a joint is flagged as likely pose-mismatched (default: 3.0)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(args.unloaded) != len(args.loaded):
        raise ValueError(f"--unloaded ({len(args.unloaded)} paths) and --loaded ({len(args.loaded)} paths) must be the same length -- pass them in matching pairs.")

    labels = args.labels if args.labels is not None else [Path(p).stem.replace("_unloaded", "") for p in args.unloaded]
    if len(labels) != len(args.unloaded):
        raise ValueError(f"--labels ({len(labels)}) must match --unloaded/--loaded ({len(args.unloaded)}) in length.")

    print(
        "Per-pair, per-joint pose (q) drift between the unloaded and loaded recordings -- "
        "model-independent (reads q straight from the logs, no checkpoint involved).\n"
        "ratio = |delta| / sqrt(std_unloaded^2 + std_loaded^2): the pose shift relative to the "
        f"within-hold noise of q itself. ratio > {args.flag_ratio:.1f} is flagged: the two "
        "recordings likely were NOT at the same pose, which alone can produce a tau_ext "
        "difference indistinguishable from a load-detection signal.\n"
    )

    all_ratios: dict[str, list[float]] = {}
    for label, u_path, l_path in zip(labels, args.unloaded, args.loaded, strict=True):
        rows = pair_stats(u_path, l_path)
        print(f"{label}  ({u_path} vs {l_path})")
        print(f"  {'joint':<15}{'q_unloaded':>12}{'std_u':>8}{'q_loaded':>12}{'std_l':>8}{'delta':>10}{'ratio':>8}")
        for joint, mean_u, std_u, mean_l, std_l, delta, ratio in rows:
            flag = "  <-- likely pose mismatch" if ratio > args.flag_ratio else ""
            print(f"  {joint:<15}{mean_u:>12.4f}{std_u:>8.4f}{mean_l:>12.4f}{std_l:>8.4f}{delta:>10.4f}{ratio:>8.1f}{flag}")
            all_ratios.setdefault(joint, []).append(ratio)
        print()

    print(f"Average drift ratio per joint across all {len(labels)} pair(s) (higher = more consistently mismatched pose):")
    for joint, ratios in sorted(all_ratios.items(), key=lambda kv: -np.mean([r for r in kv[1] if np.isfinite(r)] or [0.0])):
        finite = [r for r in ratios if np.isfinite(r)]
        avg = float(np.mean(finite)) if finite else float("inf")
        print(f"  {joint:<15}{avg:>8.1f}")


if __name__ == "__main__":
    main()
