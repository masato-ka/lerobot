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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class OmxFollowerForceEstimationConfig:
    """Online NEXT external-torque estimation (`lerobot.force_estimation.OnlineExternalTorqueEstimator`),
    exposed as extra `force.<joint>` observation keys. Disabled by default (`checkpoint_path == ""`) --
    `OmxFollower` stays a plain position+camera observation source, with no extra `Present_Velocity`/
    `Present_Current` bus reads (and their latency cost) unless this is set.
    """

    # Path to a checkpoint produced by `examples/omx/force_sensing/train_next.py`. Empty string (the
    # default) disables force estimation entirely.
    checkpoint_path: str = ""

    # Optional EMA smoothing on the returned tau_ext -- see `OnlineExternalTorqueEstimator`'s docstring.
    smoothing_alpha: float | None = None

    # Torch device for the NEXT estimator's forward pass. `None` (the default) auto-selects cuda if
    # available, else cpu -- same as `OnlineExternalTorqueEstimator`'s own default, and independent of
    # whatever device the main policy runs on. Force this to `"cpu"` to rule out GPU contention/kernel-
    # launch overhead as a real-time control-loop bottleneck: this is a sub-1M-parameter model run once
    # per tick, so CPU inference is often *faster* in wall-clock terms than a GPU call once launch/sync
    # overhead and contention with a concurrently-running policy are accounted for.
    device: str | None = None


@RobotConfig.register_subclass("omx_follower")
@dataclass
class OmxFollowerConfig(RobotConfig):
    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False

    # Online external-torque (force) estimation. Disabled by default -- see
    # `OmxFollowerForceEstimationConfig.checkpoint_path`.
    force_estimation: OmxFollowerForceEstimationConfig = field(default_factory=OmxFollowerForceEstimationConfig)
