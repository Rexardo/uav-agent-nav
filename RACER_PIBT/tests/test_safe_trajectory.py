from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from safe_trajectory import (
    build_pairwise_lsc,
    minimum_jerk_bernstein,
    optimize_safe_trajectory,
    trajectories_are_separated,
)


class SafeTrajectoryTests(unittest.TestCase):
    def test_minimum_jerk_segment_obeys_dynamics(self) -> None:
        trajectory = optimize_safe_trajectory((0, 0), (1, 0), 2.0, 1.0, 2.0, 0.2, [])
        self.assertTrue(trajectory.feasible)
        self.assertLessEqual(trajectory.max_velocity, 1.0)
        self.assertLessEqual(trajectory.max_acceleration, 2.0)

    def test_following_motion_has_lsc_separation(self) -> None:
        starts = [(0, 0), (1, 0)]
        ends = [(1, 0), (2, 0)]
        controls = [
            minimum_jerk_bernstein(np.asarray(start, float), np.asarray(end, float), 2.0)
            for start, end in zip(starts, ends)
        ]
        trajectories = [
            optimize_safe_trajectory(start, end, 2.0, 1.0, 2.0, 0.2, build_pairwise_lsc(controls, i, 0.2))
            for i, (start, end) in enumerate(zip(starts, ends))
        ]
        self.assertTrue(all(trajectory.feasible for trajectory in trajectories))
        self.assertTrue(trajectories_are_separated(trajectories, 0.2))


if __name__ == "__main__":
    unittest.main()
