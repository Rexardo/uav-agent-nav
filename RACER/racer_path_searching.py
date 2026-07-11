"""path_searching-style grid front-end and trajectory safety sampling."""

from __future__ import annotations

import heapq
import math

import numpy as np

from racer_map import bresenham, eight_neighbors, four_neighbors
from racer_types import OBSTACLE, RACERConfig, UAV

def astar(
    start: tuple[int, int],
    goal: tuple[int, int],
    grid: np.ndarray,
    config: RACERConfig | None = None,
) -> list[tuple[int, int]]:
    if start == goal:
        return [start]
    height, width = grid.shape
    if not (0 <= goal[0] < width and 0 <= goal[1] < height) or grid[goal[1], goal[0]] == OBSTACLE:
        return []
    neighbor_fn = eight_neighbors if (config and config.allow_diagonal_motion) else four_neighbors
    turn_penalty = config.turn_penalty if config else 0.0
    start_state = (start[0], start[1], 0, 0)
    open_heap = [(0.0, start_state)]
    came_from: dict[tuple[int, int, int, int], tuple[int, int, int, int]] = {}
    g_score = {start_state: 0.0}
    best_goal_state = None
    closed = set()
    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        cx, cy, pdx, pdy = current
        if (cx, cy) == goal:
            best_goal_state = current
            break
        closed.add(current)
        for nx, ny in neighbor_fn(cx, cy):
            if not (0 <= nx < width and 0 <= ny < height) or grid[ny, nx] == OBSTACLE:
                continue
            dx = nx - cx
            dy = ny - cy
            if dx and dy:
                if grid[cy, nx] == OBSTACLE or grid[ny, cx] == OBSTACLE:
                    continue
            next_state = (nx, ny, dx, dy)
            step = math.sqrt(2.0) if dx and dy else 1.0
            if (pdx, pdy) != (0, 0) and (dx, dy) != (pdx, pdy):
                step += turn_penalty
            tentative = g_score[current] + step
            if tentative < g_score.get(next_state, float("inf")):
                came_from[next_state] = current
                g_score[next_state] = tentative
                priority = tentative + math.hypot(goal[0] - nx, goal[1] - ny)
                heapq.heappush(open_heap, (priority, next_state))
    return reconstruct_path(came_from, best_goal_state) if best_goal_state is not None else []


def kinodynamic_astar(
    start: tuple[int, int],
    start_velocity: tuple[int, int],
    goal: tuple[int, int],
    grid: np.ndarray,
    config: RACERConfig,
) -> list[tuple[int, int]]:
    if start == goal:
        return [start]
    height, width = grid.shape
    if not (0 <= goal[0] < width and 0 <= goal[1] < height) or grid[goal[1], goal[0]] == OBSTACLE:
        return []

    max_speed = max(1, int(config.kinodynamic_max_speed))
    max_accel = max(1, int(config.kinodynamic_max_accel))
    start_state = (
        start[0],
        start[1],
        clamp_int(start_velocity[0], -max_speed, max_speed),
        clamp_int(start_velocity[1], -max_speed, max_speed),
    )
    open_heap = [(0.0, start_state)]
    came_from: dict[tuple[int, int, int, int], tuple[int, int, int, int]] = {}
    g_score = {start_state: 0.0}
    closed = set()
    best_goal_state = None
    expanded = 0

    accelerations = [
        (ax, ay)
        for ax in range(-max_accel, max_accel + 1)
        for ay in range(-max_accel, max_accel + 1)
        if ax != 0 or ay != 0
    ] + [(0, 0)]

    while open_heap and expanded < config.kinodynamic_max_nodes:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        expanded += 1
        x, y, vx, vy = current
        if (x, y) == goal:
            best_goal_state = current
            break
        closed.add(current)

        for ax, ay in accelerations:
            nvx = clamp_int(vx + ax, -max_speed, max_speed)
            nvy = clamp_int(vy + ay, -max_speed, max_speed)
            if nvx == 0 and nvy == 0:
                continue
            nx = x + nvx
            ny = y + nvy
            if not (0 <= nx < width and 0 <= ny < height) or grid[ny, nx] == OBSTACLE:
                continue
            if not segment_is_free((x, y), (nx, ny), grid, sample_step=config.shortcut_safety_step):
                continue
            next_state = (nx, ny, nvx, nvy)
            effort = 0.08 * (abs(ax) + abs(ay))
            speed_cost = math.hypot(nvx, nvy)
            turn_cost = 0.0 if (vx, vy) == (0, 0) else config.turn_penalty * heading_change((vx, vy), (nvx, nvy))
            tentative = g_score[current] + speed_cost + effort + turn_cost
            if tentative < g_score.get(next_state, float("inf")):
                came_from[next_state] = current
                g_score[next_state] = tentative
                heuristic = math.hypot(goal[0] - nx, goal[1] - ny)
                heapq.heappush(open_heap, (tentative + heuristic, next_state))

    return reconstruct_path(came_from, best_goal_state) if best_goal_state is not None else []


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def heading_change(v1: tuple[int, int], v2: tuple[int, int]) -> float:
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 <= 1e-9 or n2 <= 1e-9:
        return 0.0
    dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot) / math.pi

def reconstruct_path(
    came_from: dict[tuple[int, int, int, int], tuple[int, int, int, int]],
    current: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    path = [(current[0], current[1])]
    while current in came_from:
        current = came_from[current]
        path.append((current[0], current[1]))
    path.reverse()
    return path

def shortcut_path(path: list[tuple[int, int]], grid: np.ndarray, config: RACERConfig | None = None) -> list[tuple[int, int]]:
    if len(path) <= 2:
        return path
    result = [path[0]]
    anchor = 0
    probe = 2
    while probe < len(path):
        if segment_is_free(path[anchor], path[probe], grid, sample_step=(config.shortcut_safety_step if config else 0.25)):
            probe += 1
            continue
        result.append(path[probe - 1])
        anchor = probe - 1
        probe = anchor + 2
    result.append(path[-1])
    return expand_path(result)

def segment_is_free(
    start: tuple[int, int],
    end: tuple[int, int],
    grid: np.ndarray,
    include_start: bool = True,
    sample_step: float = 0.25,
) -> bool:
    height, width = grid.shape
    cells = sampled_segment_cells(start, end, sample_step)
    if not include_start:
        cells = cells[1:]
    for x, y in cells:
        if not (0 <= x < width and 0 <= y < height) or grid[y, x] == OBSTACLE:
            return False
    return True

def sampled_segment_cells(start: tuple[int, int], end: tuple[int, int], step: float = 0.25) -> list[tuple[int, int]]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max(1.0, math.hypot(dx, dy))
    sample_num = max(1, int(math.ceil(length / max(1e-6, step))))
    cells = []
    for i in range(sample_num + 1):
        alpha = i / sample_num
        cell = (int(round(start[0] + alpha * dx)), int(round(start[1] + alpha * dy)))
        if not cells or cells[-1] != cell:
            cells.append(cell)
    return cells

def path_is_free(path: list[tuple[int, int]], grid: np.ndarray, include_start: bool = True) -> bool:
    if not path:
        return True
    height, width = grid.shape
    cells = path if include_start else path[1:]
    for x, y in cells:
        if not (0 <= x < width and 0 <= y < height) or grid[y, x] == OBSTACLE:
            return False
    for idx, (a, b) in enumerate(zip(path, path[1:])):
        if not segment_is_free(a, b, grid, include_start=(include_start or idx > 0)):
            return False
    return True

def expand_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not path:
        return []
    expanded = [path[0]]
    for start, end in zip(path, path[1:]):
        segment = bresenham(start, end)
        expanded.extend(segment[1:])
    return expanded

def future_position(uav: UAV, step_idx: int) -> tuple[int, int]:
    if step_idx < len(uav.path):
        return uav.path[step_idx]
    return uav.path[-1] if uav.path else uav.pos
