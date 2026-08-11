#!/usr/bin/env python3
"""Quantitatively summarize the `force.*` slice of `observation.state` in a LeRobotDataset
recorded by `examples/omx/bilateral_teleop/record_bilateral.py`.

`evaluate_free_motion.py` answers "what does the noise floor look like on contact-free logs?"
This script answers the complementary question on real recorded task data: "how far does
`tau_ext` actually move during a session that includes real contact?" Compare the overall
mean/std here against `evaluate_free_motion.py`'s noise-floor numbers -- if they're close, the
estimator isn't separating contact from noise well; if the per-episode max|.| and percentiles
show a much wider spread than the noise floor, contact is being picked up.

Reads only the `observation.state`/`episode_index` columns directly from the dataset's
underlying table (`select_columns`), so no video decoding happens even if the dataset has
cameras attached.

Usage (run from repo root):
    python -m examples.omx.force_sensing.evaluate_dataset_force \\
        --repo_id <hf_username>/omx_bilateral_force --root data/omx_bilateral_force

    # Also print a thinned per-frame timeline for one episode:
    python -m examples.omx.force_sensing.evaluate_dataset_force \\
        --repo_id <hf_username>/omx_bilateral_force --root data/omx_bilateral_force \\
        --episode_index 0 --episode_stride 5
"""

import argparse
import logging

import numpy as np

from lerobot.datasets import LeRobotDataset

logger = logging.getLogger(__name__)


def force_column_indices(state_names: list[str]) -> tuple[list[int], list[str]]:
    """Indices into `observation.state` whose name is `force.<joint>`, and the bare joint names."""
    indices = [i for i, name in enumerate(state_names) if name.startswith("force.")]
    joints = [state_names[i].removeprefix("force.") for i in indices]
    return indices, joints


def side_stats(col: np.ndarray) -> tuple[int, float, float, float]:
    """`(n, mean, std, max)` of `col`'s magnitude, restricted to values with `col`'s sign
    (i.e. call with `col` and with `-col` to get the +side/-side breakdown separately). `n=0`
    stats are `0.0` rather than `nan` so the table stays printable when one side is empty.
    """
    side = col[col > 0]
    if len(side) == 0:
        return 0, 0.0, 0.0, 0.0
    return len(side), float(side.mean()), float(side.std()), float(side.max())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo_id", required=True, help="e.g. <hf_username>/<dataset_name>")
    parser.add_argument("--root", default=None, help="Local dataset directory (defaults to HF cache)")
    parser.add_argument(
        "--episode_index", type=int, default=None, help="Also print a thinned per-frame timeline for this episode"
    )
    parser.add_argument("--episode_stride", type=int, default=5, help="Timeline thinning stride")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dataset = LeRobotDataset(args.repo_id, root=args.root, download_videos=False)
    state_names = dataset.features["observation.state"]["names"]
    force_idx, joints = force_column_indices(state_names)
    if not force_idx:
        raise ValueError(
            f"No force.* columns found in observation.state names: {state_names}. "
            "Was this dataset recorded with record_bilateral.py?"
        )

    cols = dataset.hf_dataset.select_columns(["observation.state", "episode_index"])[:]
    state = np.asarray(cols["observation.state"], dtype=np.float32)
    episode_index = np.asarray(cols["episode_index"], dtype=np.int64)
    force = state[:, force_idx]

    print(f"\n{dataset.repo_id}: {len(force)} frames across {dataset.num_episodes} episode(s).")
    print("Compare these against evaluate_free_motion.py's noise-floor mean/std/max|.| per joint.\n")

    print("Overall (all episodes combined):")
    print(f"{'joint':<15}{'mean':>10}{'std':>10}{'min':>10}{'p5':>10}{'p50':>10}{'p95':>10}{'max':>10}")
    for j, joint in enumerate(joints):
        col = force[:, j]
        print(
            f"{joint:<15}{col.mean():>10.2f}{col.std():>10.2f}{col.min():>10.2f}"
            f"{np.percentile(col, 5):>10.2f}{np.percentile(col, 50):>10.2f}"
            f"{np.percentile(col, 95):>10.2f}{col.max():>10.2f}"
        )

    print(
        "\n+side / -side breakdown (compare against evaluate_free_motion.py's own +/- "
        "breakdown on contact-free data -- if the noise floor is already lopsided the same "
        "way, this is a free-space/hardware artifact, not a real per-direction contact "
        "difference):"
    )
    print(f"{'joint':<15}{'n+':>6}{'mean+':>9}{'std+':>9}{'max+':>9}   {'n-':>6}{'mean-':>9}{'std-':>9}{'max-':>9}")
    for j, joint in enumerate(joints):
        col = force[:, j]
        n_pos, mean_pos, std_pos, max_pos = side_stats(col)
        n_neg, mean_neg, std_neg, max_neg = side_stats(-col)
        print(
            f"{joint:<15}{n_pos:>6}{mean_pos:>9.2f}{std_pos:>9.2f}{max_pos:>9.2f}   "
            f"{n_neg:>6}{mean_neg:>9.2f}{std_neg:>9.2f}{max_neg:>9.2f}"
        )

    episode_ids = sorted(set(episode_index.tolist()))
    print("\nPer-episode max|force| (spot an episode/joint with little dynamic range):")
    print(f"{'episode':<10}" + "".join(f"{j:>15}" for j in joints))
    for ep in episode_ids:
        row = force[episode_index == ep]
        print(f"{ep:<10}" + "".join(f"{np.abs(row[:, j]).max():>15.2f}" for j in range(len(joints))))

    if args.episode_index is not None:
        if args.episode_index not in episode_ids:
            raise ValueError(f"episode_index {args.episode_index} not in dataset (have {episode_ids})")
        row = force[episode_index == args.episode_index]
        print(f"\nEpisode {args.episode_index} timeline (every {args.episode_stride}th frame):")
        print(f"{'frame':<10}" + "".join(f"{j:>15}" for j in joints))
        for i in range(0, len(row), args.episode_stride):
            print(f"{i:<10}" + "".join(f"{row[i, k]:>15.2f}" for k in range(len(joints))))


if __name__ == "__main__":
    main()
