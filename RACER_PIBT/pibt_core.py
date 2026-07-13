"""Official-pypibt-compatible PIBT core adapted to RACER's (x, y) grid."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np

Coord = tuple[int, int]


@dataclass
class PIBTStepStats:
    requests: int = 0
    responses: int = 0
    priority_inheritances: int = 0
    backtracks: int = 0
    waits: int = 0


class DistanceTable:
    """Shortest grid distances to one goal, evaluated once with BFS."""

    def __init__(self, free_grid: np.ndarray, goal: Coord):
        self.grid = free_grid
        self.goal = goal
        self.distance = np.full(free_grid.shape, free_grid.size, dtype=np.int32)
        if self.is_free(goal):
            queue: deque[Coord] = deque([goal])
            self.distance[goal[1], goal[0]] = 0
            while queue:
                current = queue.popleft()
                next_distance = int(self.distance[current[1], current[0]]) + 1
                for neighbor in neighbors(self.grid, current):
                    x, y = neighbor
                    if next_distance < self.distance[y, x]:
                        self.distance[y, x] = next_distance
                        queue.append(neighbor)

    def is_free(self, coord: Coord) -> bool:
        x, y = coord
        return 0 <= y < self.grid.shape[0] and 0 <= x < self.grid.shape[1] and bool(self.grid[y, x])

    def get(self, coord: Coord) -> int:
        if not self.is_free(coord):
            return int(self.grid.size)
        return int(self.distance[coord[1], coord[0]])


def neighbors(free_grid: np.ndarray, coord: Coord) -> list[Coord]:
    x, y = coord
    result: list[Coord] = []
    for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
        if 0 <= ny < free_grid.shape[0] and 0 <= nx < free_grid.shape[1] and free_grid[ny, nx]:
            result.append((nx, ny))
    return result


class PIBTStepPlanner:
    """One receding-horizon PIBT step with request/response instrumentation.

    The recursion follows Kei Okumura's official pypibt implementation. RACER
    uses (x, y), so array indexing is converted at the boundary.
    """

    def __init__(self, free_grid: np.ndarray, goals: list[Coord], seed: int = 0):
        self.grid = np.asarray(free_grid, dtype=bool)
        self.goals = list(goals)
        self.distances = [DistanceTable(self.grid, goal) for goal in goals]
        self.rng = random.Random(seed)
        self.stats = PIBTStepStats()
        self.occupied_now: dict[Coord, int] = {}
        self.occupied_next: dict[Coord, int] = {}
        self.next_config: list[Coord | None] = []
        self.current_config: list[Coord] = []

    def step(self, current: list[Coord], priorities: list[float]) -> tuple[list[Coord], PIBTStepStats]:
        if len(current) != len(self.goals) or len(current) != len(priorities):
            raise ValueError("current, goals, and priorities must have equal lengths")
        if len(set(current)) != len(current):
            raise ValueError("PIBT requires distinct current vertices")
        self.current_config = list(current)
        self.next_config = [None] * len(current)
        self.occupied_now = {coord: index for index, coord in enumerate(current)}
        self.occupied_next = {}

        order = sorted(range(len(current)), key=lambda i: (priorities[i], -i), reverse=True)
        for index in order:
            if self.next_config[index] is None:
                self._assign(index)

        result = [coord if coord is not None else current[i] for i, coord in enumerate(self.next_config)]
        self.stats.waits = sum(start == end for start, end in zip(current, result))
        return result, self.stats

    def _assign(self, index: int) -> bool:
        current = self.current_config[index]
        candidates = [current] + neighbors(self.grid, current)
        self.rng.shuffle(candidates)
        candidates.sort(key=self.distances[index].get)

        for candidate in candidates:
            if candidate in self.occupied_next:
                continue
            occupant = self.occupied_now.get(candidate)
            if occupant is not None and self.next_config[occupant] == current:
                continue

            self.next_config[index] = candidate
            self.occupied_next[candidate] = index
            if occupant is not None and occupant != index and self.next_config[occupant] is None:
                self.stats.requests += 1
                self.stats.priority_inheritances += 1
                accepted = self._assign(occupant)
                self.stats.responses += 1
                if not accepted:
                    self.stats.backtracks += 1
                    if self.occupied_next.get(candidate) == index:
                        del self.occupied_next[candidate]
                    self.next_config[index] = None
                    continue
            return True

        self.next_config[index] = current
        self.occupied_next[current] = index
        return False


def project_goal_to_reachable(free_grid: np.ndarray, start: Coord, goal: Coord) -> tuple[Coord, bool]:
    """Project a RACER target onto the currently known reachable free graph."""
    if not free_grid[start[1], start[0]]:
        return start, True
    queue: deque[Coord] = deque([start])
    visited = {start}
    best = start
    best_key = (abs(start[0] - goal[0]) + abs(start[1] - goal[1]), 0)
    depth = {start: 0}
    while queue:
        current = queue.popleft()
        key = (abs(current[0] - goal[0]) + abs(current[1] - goal[1]), depth[current])
        if key < best_key:
            best, best_key = current, key
        if current == goal:
            return current, False
        for neighbor in neighbors(free_grid, current):
            if neighbor not in visited:
                visited.add(neighbor)
                depth[neighbor] = depth[current] + 1
                queue.append(neighbor)
    return best, best != goal
