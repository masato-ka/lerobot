#!/usr/bin/env python3
"""Shared constants for the OMX FACTR2/NEXT force-sensing scripts.

Excludes the gripper: it already has its own current-limited grasp-force control
(see `OmxFollower.configure()`), and the NEXT external-torque estimator targets the
5 arm joints only (see src/lerobot/force_estimation/README.md).
"""

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
