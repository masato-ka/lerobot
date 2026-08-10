#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Free-motion episode loading and the sliding-window dataset used to train NEXT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class FreeMotionEpisode:
    """One continuous free-motion (no-contact) recording.

    As saved by `examples/omx/force_sensing/collect_free_motion.py`. All arrays are
    `(T, num_joints)` except `t`, which is `(T,)`.

    **Attributes**:
        - **t** (`np.ndarray`) -- Elapsed time in seconds per sample, from the recording clock.
        - **q** (`np.ndarray`) -- Present position, in the robot's normalized position units (as
          returned by [`~robots.Robot.get_observation`]).
        - **qdot** (`np.ndarray`) -- Present velocity, in raw `Present_Velocity` register units.
        - **goal_q** (`np.ndarray`) -- Commanded goal position, same units as `q`.
        - **current** (`np.ndarray`) -- Present current/load, in raw `Present_Current` register
          units (meaning differs per motor model, see `README.md`).
        - **joint_names** (`list[str]`) -- Joint name for each column, in array order.
    """

    t: np.ndarray
    q: np.ndarray
    qdot: np.ndarray
    goal_q: np.ndarray
    current: np.ndarray
    joint_names: list[str]


def load_episode(path: str | Path) -> FreeMotionEpisode:
    """Load a free-motion episode saved by `collect_free_motion.py`.

    Args:
        path (`str | Path`):
            Path to the `.npz` log file.

    Returns:
        `FreeMotionEpisode`: The loaded episode.
    """
    data = np.load(path, allow_pickle=False)
    return FreeMotionEpisode(
        t=data["t"],
        q=data["q"],
        qdot=data["qdot"],
        goal_q=data["goal_q"],
        current=data["current"],
        joint_names=[str(name) for name in data["joint_names"]],
    )


def resample_uniform(episode: FreeMotionEpisode, target_hz: float = 100.0) -> FreeMotionEpisode:
    """Linearly resample an episode onto a uniform time grid.

    Necessary because `DynamixelMotorsBus.sync_read` round trips during collection are not
    guaranteed to hit a fixed rate (see `README.md`).

    Args:
        episode (`FreeMotionEpisode`):
            Episode recorded at irregular control-loop timestamps.
        target_hz (`float`, *optional*, defaults to `100.0`):
            Target uniform sampling rate.

    Returns:
        `FreeMotionEpisode`: A new episode resampled onto a uniform `target_hz` time grid.
    """
    if len(episode.t) < 2:
        raise ValueError("Episode is too short to resample (need at least 2 samples).")

    duration = episode.t[-1] - episode.t[0]
    num_samples = max(2, int(duration * target_hz) + 1)
    t_uniform = np.linspace(episode.t[0], episode.t[-1], num_samples)

    def interp(arr: np.ndarray) -> np.ndarray:
        return np.stack([np.interp(t_uniform, episode.t, arr[:, j]) for j in range(arr.shape[1])], axis=1)

    return FreeMotionEpisode(
        t=t_uniform,
        q=interp(episode.q),
        qdot=interp(episode.qdot),
        goal_q=interp(episode.goal_q),
        current=interp(episode.current),
        joint_names=episode.joint_names,
    )


def build_features(episode: FreeMotionEpisode) -> np.ndarray:
    """Build the per-timestep NEXT input feature vector for an episode.

    Concatenates `[q, qdot, delta_q_d]` across joints, matching FACTR2's NEXT input
    (arXiv:2606.12406, eq. 3), where `delta_q_d` is the commanded-minus-actual tracking error
    (`goal_q - q`).

    Args:
        episode (`FreeMotionEpisode`):
            Episode to build features from (typically already resampled).

    Returns:
        `np.ndarray`: Shape `(T, 3 * num_joints)` feature array.
    """
    delta_q_d = episode.goal_q - episode.q
    return np.concatenate([episode.q, episode.qdot, delta_q_d], axis=1).astype(np.float32)


class NextWindowDataset(Dataset):
    """Sliding-window `(history, target)` pairs for training [`~force_estimation.NextTorqueEstimator`].

    Each sample is a `history_length`-step window of [`build_features`] ending at timestep `i`,
    paired with the measured torque-proxy target (`current[i]`) at that same timestep — the
    supervised free-space-torque regression target from FACTR2's NEXT (eq. 4). Windows never
    cross episode boundaries.

    Args:
        episodes (`list[FreeMotionEpisode]`):
            Episodes to draw windows from (typically already resampled to a uniform rate).
        history_length (`int`, *optional*, defaults to `50`):
            Number of timesteps per window.

    **Attributes**:
        - **features** (`list[np.ndarray]`) -- Per-episode feature arrays from [`build_features`].
        - **targets** (`list[np.ndarray]`) -- Per-episode measured torque-proxy targets.
        - **index** (`list[tuple[int, int]]`) -- `(episode_idx, window_end_idx)` for every sample.
    """

    def __init__(self, episodes: list[FreeMotionEpisode], history_length: int = 50):
        """Build the window index over all episodes; see the class docstring for the parameters."""
        self.history_length = history_length
        self.features = [build_features(ep) for ep in episodes]
        self.targets = [ep.current.astype(np.float32) for ep in episodes]

        self.index: list[tuple[int, int]] = []
        for ep_idx, feat in enumerate(self.features):
            for end in range(history_length - 1, len(feat)):
                self.index.append((ep_idx, end))

    def __len__(self) -> int:
        """Total number of windows across all episodes."""
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the `(history, target)` pair for window `idx` (see the class docstring)."""
        ep_idx, end = self.index[idx]
        start = end - self.history_length + 1
        x = self.features[ep_idx][start : end + 1]
        y = self.targets[ep_idx][end]
        return torch.from_numpy(x), torch.from_numpy(y)
