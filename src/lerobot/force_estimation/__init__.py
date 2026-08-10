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

"""Neural External Torque Estimation (NEXT), from FACTR2 (arXiv:2606.12406).

Estimates external joint torque on commodity robot arms without dedicated
force/torque sensors, by learning a model of "expected" free-space motor
torque from joint kinematic history and subtracting it from the measured
motor torque at inference time.

See README.md in this directory for the full usage guide.
"""

from .dataset import FreeMotionEpisode, NextWindowDataset, load_episode, resample_uniform
from .next_model import NextTorqueEstimator
from .online import OnlineExternalTorqueEstimator
from .train import NextTrainConfig, train_next

__all__ = [
    "FreeMotionEpisode",
    "NextWindowDataset",
    "NextTorqueEstimator",
    "NextTrainConfig",
    "OnlineExternalTorqueEstimator",
    "load_episode",
    "resample_uniform",
    "train_next",
]
