from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from pibt_core import PIBTStepPlanner


class PIBTCoreTests(unittest.TestCase):
    def test_priority_inheritance_pushes_a_chain(self) -> None:
        grid = np.ones((1, 4), dtype=bool)
        planner = PIBTStepPlanner(grid, [(3, 0), (3, 0), (3, 0)], seed=1)
        next_config, stats = planner.step([(0, 0), (1, 0), (2, 0)], [3.0, 2.0, 1.0])
        self.assertEqual(next_config, [(1, 0), (2, 0), (3, 0)])
        self.assertEqual(len(set(next_config)), 3)
        self.assertGreaterEqual(stats.priority_inheritances, 2)

    def test_prevents_vertex_and_edge_swap_collisions(self) -> None:
        grid = np.ones((2, 3), dtype=bool)
        starts = [(0, 0), (1, 0)]
        planner = PIBTStepPlanner(grid, [(1, 0), (0, 0)], seed=2)
        next_config, _ = planner.step(starts, [2.0, 1.0])
        self.assertEqual(len(set(next_config)), 2)
        self.assertFalse(next_config[0] == starts[1] and next_config[1] == starts[0])

    def test_backtracks_when_a_push_reaches_a_dead_end(self) -> None:
        grid = np.ones((1, 2), dtype=bool)
        starts = [(0, 0), (1, 0)]
        planner = PIBTStepPlanner(grid, [(1, 0), (1, 0)], seed=3)
        next_config, stats = planner.step(starts, [2.0, 1.0])
        self.assertEqual(next_config, starts)
        self.assertGreaterEqual(stats.backtracks, 1)


if __name__ == "__main__":
    unittest.main()
