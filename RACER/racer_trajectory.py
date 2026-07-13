"""bspline / bspline_opt-style trajectory smoothing and validation."""

from __future__ import annotations

import math

import numpy as np

from racer_path_searching import path_is_free, sampled_segment_cells, segment_is_free
from racer_types import RACERConfig


def smooth_trajectory(
    path: list[tuple[int, int]],
    grid: np.ndarray,
    config: RACERConfig,
    soft_obstacle_grid: np.ndarray | None = None,
    swarm_paths: list[list[tuple[int, int]]] | None = None,
) -> list[tuple[int, int]]:
    if not config.use_bspline_smoothing or len(path) <= 2:
        return densify_grid_path(path)

    controls = [(float(x), float(y)) for x, y in remove_duplicate_points(path)]
    if len(controls) <= 2:
        return path

    curve = optimize_control_points(
        controls,
        soft_obstacle_grid if soft_obstacle_grid is not None else grid,
        config,
        swarm_paths,
    )
    for _ in range(max(1, config.bspline_smooth_iterations)):
        curve = chaikin_refine(curve)

    sampled = sample_polyline(curve, max(1, config.bspline_samples_per_segment))
    raster_path = rasterize_float_path(sampled)
    if len(raster_path) <= 1:
        return path
    if path_is_free(raster_path, grid):
        return densify_grid_path(raster_path)

    # Try a conservative line-of-sight shortcut before giving up. This mirrors
    # the RACER stack's habit of refining a guide path while keeping the final
    # trajectory collision-checked.
    safe_path = [path[0]]
    anchor = path[0]
    previous = path[0]
    for point in path[1:]:
        if segment_is_free(anchor, point, grid, sample_step=config.shortcut_safety_step):
            previous = point
            continue
        if safe_path[-1] != previous:
            safe_path.append(previous)
        anchor = previous
        previous = point
    if safe_path[-1] != path[-1]:
        safe_path.append(path[-1])
    safe_path = densify_grid_path(safe_path)
    return safe_path if path_is_free(safe_path, grid) else densify_grid_path(path)


def optimize_control_points(
    controls: list[tuple[float, float]],
    grid: np.ndarray,
    config: RACERConfig,
    swarm_paths: list[list[tuple[int, int]]] | None = None,
) -> list[tuple[float, float]]:
    """Nudge the guide control polygon with RACER-style smoothness and distance costs."""
    if len(controls) <= 2:
        return controls

    points = np.array(controls, dtype=float)
    obstacle_yx = np.argwhere(grid != 0)
    height, width = grid.shape
    smooth_weight = max(0.0, config.bspline_smooth_weight)
    obstacle_weight = max(0.0, config.bspline_obstacle_weight)
    clearance = max(0.0, config.bspline_obstacle_clearance_cells())
    swarm_weight = max(0.0, config.bspline_swarm_weight)
    swarm_clearance = max(0.0, config.meters_to_cells(config.swarm_safe_distance))
    use_swarm_cost = bool(swarm_paths) and swarm_weight > 0.0 and swarm_clearance > 0.0

    if obstacle_yx.size == 0 and not use_swarm_cost:
        return [(float(x), float(y)) for x, y in points]
    if smooth_weight <= 0.0 and obstacle_weight <= 0.0 and not use_swarm_cost:
        return [(float(x), float(y)) for x, y in points]

    obstacle_xy = obstacle_yx[:, ::-1].astype(float)
    for _ in range(max(1, config.bspline_smooth_iterations)):
        previous = points.copy()
        for idx in range(1, len(points) - 1):
            smooth_delta = 0.5 * (previous[idx - 1] + previous[idx + 1]) - previous[idx]
            delta = smooth_weight * smooth_delta

            if obstacle_xy.size > 0 and obstacle_weight > 0.0 and clearance > 0.0:
                repulsion = nearest_obstacle_repulsion(previous[idx], previous[idx - 1], previous[idx + 1], obstacle_xy, clearance)
                delta += obstacle_weight * repulsion

            if use_swarm_cost:
                repulsion = swarm_trajectory_repulsion(previous[idx], idx, swarm_paths or [], swarm_clearance)
                delta += 0.1 * swarm_weight * repulsion

            points[idx] = previous[idx] + delta
            points[idx, 0] = float(np.clip(points[idx, 0], 0.0, width - 1.0))
            points[idx, 1] = float(np.clip(points[idx, 1], 0.0, height - 1.0))

    return [(float(x), float(y)) for x, y in points]


def swarm_trajectory_repulsion(
    point: np.ndarray,
    step_idx: int,
    swarm_paths: list[list[tuple[int, int]]],
    clearance: float,
) -> np.ndarray:
    """Gradient direction of RACER's soft separation cost at one time step."""
    repulsion = np.zeros(2, dtype=float)
    for path in swarm_paths:
        if not path:
            continue
        other = np.array(path[min(step_idx, len(path) - 1)], dtype=float)
        diff = point - other
        distance = float(np.linalg.norm(diff))
        if distance <= 1e-9 or distance >= clearance:
            continue
        repulsion += (clearance - distance) * diff / distance
    return repulsion


def nearest_obstacle_repulsion(
    point: np.ndarray,
    prev_point: np.ndarray,
    next_point: np.ndarray,
    obstacle_xy: np.ndarray,
    clearance: float,
) -> np.ndarray:
    diff = point - obstacle_xy
    dist_sq = np.einsum("ij,ij->i", diff, diff)
    nearest_idx = int(np.argmin(dist_sq))
    distance = math.sqrt(max(1e-12, float(dist_sq[nearest_idx])))
    if distance >= clearance:
        return np.zeros(2, dtype=float)

    if distance <= 1e-6:
        tangent = next_point - prev_point
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        norm = float(np.linalg.norm(normal))
        direction = normal / norm if norm > 1e-9 else np.array([1.0, 0.0], dtype=float)
    else:
        direction = diff[nearest_idx] / distance
    return (clearance - distance) * direction


def remove_duplicate_points(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result = []
    for point in path:
        if not result or result[-1] != point:
            result.append(point)
    return result


def densify_grid_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not path:
        return []
    dense = [path[0]]
    for start, end in zip(path, path[1:]):
        for cell in sampled_segment_cells(start, end, step=0.5)[1:]:
            if dense[-1] != cell:
                dense.append(cell)
    return dense


def chaikin_refine(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    refined = [points[0]]
    for p0, p1 in zip(points, points[1:]):
        q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
        r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
        refined.extend([q, r])
    refined.append(points[-1])
    return refined


def sample_polyline(points: list[tuple[float, float]], samples_per_segment: int) -> list[tuple[float, float]]:
    samples = [points[0]]
    for p0, p1 in zip(points, points[1:]):
        for i in range(1, samples_per_segment + 1):
            alpha = i / samples_per_segment
            samples.append((p0[0] + alpha * (p1[0] - p0[0]), p0[1] + alpha * (p1[1] - p0[1])))
    return samples


def rasterize_float_path(points: list[tuple[float, float]]) -> list[tuple[int, int]]:
    if not points:
        return []
    raster = [(int(round(points[0][0])), int(round(points[0][1])))]
    for p0, p1 in zip(points, points[1:]):
        start = (int(round(p0[0])), int(round(p0[1])))
        end = (int(round(p1[0])), int(round(p1[1])))
        for cell in sampled_segment_cells(start, end, step=0.25):
            if not raster or raster[-1] != cell:
                raster.append(cell)
    return raster


def max_turn_angle_deg(path: list[tuple[int, int]]) -> float:
    max_angle = 0.0
    for a, b, c in zip(path, path[1:], path[2:]):
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 <= 1e-9 or n2 <= 1e-9:
            continue
        dot = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        max_angle = max(max_angle, math.degrees(math.acos(dot)))
    return max_angle
