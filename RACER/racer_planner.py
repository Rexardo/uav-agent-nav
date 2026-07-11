"""exploration_manager-style frontier/FIS and CP-guided viewpoint planning."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from racer_hgrid import HGrid, HGridBlock, generate_coverage_path_for_blocks
from racer_map import (
    KnownMap,
    eight_neighbors,
    four_neighbors,
    planning_grid_from_known_map,
    planning_grid_for_uav,
)
from racer_path_searching import astar, kinodynamic_astar, shortcut_path
from racer_trajectory import smooth_trajectory
from racer_types import FREE, OBSTACLE, UNKNOWN, RACERConfig, UAV

@dataclass
class FrontierInfo:
    cluster_id: int
    cells: list[tuple[int, int]]
    centroid: tuple[float, float]
    viewpoints: list[tuple[int, int, int]]


class FrontierInformationStructure:
    """Incremental-ish frontier container for this grid simulator."""

    def __init__(self) -> None:
        self.frontiers: list[FrontierInfo] = []

    def update(self, known_map: KnownMap, uav_pos: tuple[int, int], config: RACERConfig) -> list[FrontierInfo]:
        frontier_cells = detect_explorable_frontier_cells(known_map)
        clusters = cluster_cells(frontier_cells)
        infos: list[FrontierInfo] = []
        for cluster_id, cluster in enumerate(clusters):
            if len(cluster) < config.fis_min_cluster_size:
                continue
            for sub_cluster in split_large_cluster(cluster, config.fis_max_cluster_size):
                centroid = cluster_centroid(sub_cluster)
                viewpoints = sample_viewpoints_for_cluster(known_map, sub_cluster, uav_pos, config)
                if viewpoints:
                    infos.append(FrontierInfo(cluster_id, sub_cluster, centroid, viewpoints))
        self.frontiers = infos
        return infos


def detect_explorable_frontier_cells(known_map: KnownMap) -> set[tuple[int, int]]:
    frontiers: set[tuple[int, int]] = set()
    grid = known_map.grid
    for y in range(known_map.world.height):
        for x in range(known_map.world.width):
            if grid[y, x] != FREE:
                continue
            if known_map.world.raw_obstacle_map[y, x] == OBSTACLE:
                continue
            if frontier_cell_has_explorable_unknown(known_map, x, y):
                frontiers.add((x, y))
    return frontiers


def is_explorable_unknown_cell(known_map: KnownMap, x: int, y: int) -> bool:
    return (
        known_map.in_bounds(x, y)
        and known_map.grid[y, x] == UNKNOWN
        and known_map.world.raw_obstacle_map[y, x] != OBSTACLE
    )


def frontier_cell_has_explorable_unknown(known_map: KnownMap, x: int, y: int) -> bool:
    return any(is_explorable_unknown_cell(known_map, nx, ny) for nx, ny in four_neighbors(x, y))


def cluster_cells(cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    remaining = set(cells)
    clusters = []
    while remaining:
        seed = remaining.pop()
        cluster = [seed]
        queue = deque([seed])
        while queue:
            x, y = queue.popleft()
            for nx, ny in eight_neighbors(x, y):
                if (nx, ny) in remaining:
                    remaining.remove((nx, ny))
                    queue.append((nx, ny))
                    cluster.append((nx, ny))
        clusters.append(cluster)
    return clusters


def split_large_cluster(cluster: list[tuple[int, int]], max_size: int) -> list[list[tuple[int, int]]]:
    if max_size <= 0 or len(cluster) <= max_size:
        return [cluster]
    ordered = sorted(cluster, key=lambda cell: (cell[1], cell[0]))
    return [ordered[i : i + max_size] for i in range(0, len(ordered), max_size)]


def cluster_centroid(cluster: list[tuple[int, int]]) -> tuple[float, float]:
    return (sum(x for x, _ in cluster) / len(cluster), sum(y for _, y in cluster) / len(cluster))


def sample_viewpoints_for_cluster(known_map: KnownMap, cluster: list[tuple[int, int]], uav_pos: tuple[int, int], config: RACERConfig) -> list[tuple[int, int, int]]:
    centroid = cluster_centroid(cluster)
    candidates: list[tuple[float, tuple[int, int, int]]] = []
    radius = max(1, int(config.viewpoint_sample_radius))
    min_x = max(0, min(x for x, _ in cluster) - radius)
    max_x = min(known_map.world.width - 1, max(x for x, _ in cluster) + radius)
    min_y = max(0, min(y for _, y in cluster) - radius)
    max_y = min(known_map.world.height - 1, max(y for _, y in cluster) + radius)
    planning_grid = planning_grid_from_known_map(known_map, config)
    candidate_clearance = config.frontier_min_candidate_clearance_cells()
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if known_map.grid[y, x] != FREE:
                continue
            dist_to_frontier = min(math.hypot(x - fx, y - fy) for fx, fy in cluster)
            if dist_to_frontier > config.coverage_radius:
                continue
            if planning_grid[y, x] == OBSTACLE:
                continue
            if not has_known_obstacle_clearance(known_map, (x, y), candidate_clearance):
                continue
            gain = unknown_gain_from_viewpoint(known_map, (x, y), config.coverage_radius)
            if gain < config.viewpoint_min_gain:
                continue
            dist_to_uav = math.hypot(x - uav_pos[0], y - uav_pos[1])
            dist_to_cluster = math.hypot(x - centroid[0], y - centroid[1])
            candidates.append((-2.5 * gain + 0.12 * dist_to_cluster + 0.05 * dist_to_uav, (x, y, gain)))
    candidates.sort(key=lambda item: item[0])
    return [candidate for _, candidate in candidates[: max(1, config.viewpoint_samples_per_cluster)]]


def has_known_obstacle_clearance(known_map: KnownMap, point: tuple[int, int], clearance: float) -> bool:
    if clearance <= 0.0:
        return True
    px, py = point
    radius = int(math.ceil(clearance))
    for y in range(max(0, py - radius), min(known_map.world.height, py + radius + 1)):
        for x in range(max(0, px - radius), min(known_map.world.width, px + radius + 1)):
            if known_map.grid[y, x] == OBSTACLE and math.hypot(x - px, y - py) <= clearance:
                return False
    return True


def unknown_gain_from_viewpoint(known_map: KnownMap, point: tuple[int, int], radius: float) -> int:
    x0, y0 = point
    count = 0
    r = int(math.ceil(radius))
    for y in range(max(0, y0 - r), min(known_map.world.height, y0 + r + 1)):
        for x in range(max(0, x0 - r), min(known_map.world.width, x0 + r + 1)):
            if not is_explorable_unknown_cell(known_map, x, y):
                continue
            if cell_visible_from(known_map, (x0, y0), (x, y), radius):
                count += 1
    return count


def cell_visible_from(
    known_map: KnownMap,
    origin: tuple[int, int],
    cell: tuple[int, int],
    radius: float,
) -> bool:
    ox, oy = origin
    tx, ty = cell
    if math.hypot(tx - ox, ty - oy) > radius:
        return False
    dx = tx - ox
    dy = ty - oy
    steps = max(int(math.ceil(max(abs(dx), abs(dy)) * 2)), 1)
    for idx in range(1, steps):
        x = int(round(ox + dx * idx / steps))
        y = int(round(oy + dy * idx / steps))
        if not known_map.in_bounds(x, y):
            return False
        if known_map.world.raw_obstacle_map[y, x] == OBSTACLE:
            return False
    return True


def update_cp_cursor(uav: UAV, cp_path: list[tuple[int, int]], reach_radius: float) -> int:
    if not cp_path:
        uav.cp_cursor = 0
        return 0
    cursor = max(0, min(int(getattr(uav, "cp_cursor", 0)), len(cp_path) - 1))
    while cursor < len(cp_path) - 1 and math.hypot(uav.pos[0] - cp_path[cursor][0], uav.pos[1] - cp_path[cursor][1]) <= reach_radius:
        cursor += 1
    nearest = min(range(cursor, len(cp_path)), key=lambda i: math.hypot(uav.pos[0] - cp_path[i][0], uav.pos[1] - cp_path[i][1]))
    uav.cp_cursor = nearest
    return nearest


def cp_score_for_cluster(cluster: list[tuple[int, int]], cp_path: list[tuple[int, int]], cursor: int, config: RACERConfig) -> tuple[float, int]:
    if not cp_path:
        return 0.0, 0
    centroid = cluster_centroid(cluster)
    best_idx = min(range(cursor, len(cp_path)), key=lambda i: math.hypot(centroid[0] - cp_path[i][0], centroid[1] - cp_path[i][1]))
    dist = math.hypot(centroid[0] - cp_path[best_idx][0], centroid[1] - cp_path[best_idx][1])
    return config.cp_guidance_weight * dist + config.cp_index_weight * (best_idx - cursor), best_idx


def target_is_blacklisted(uav: UAV, target: tuple[int, int], config: RACERConfig) -> bool:
    radius = config.target_blacklist_radius
    return any(math.hypot(target[0] - old[0], target[1] - old[1]) <= radius for old in uav.target_blacklist)


def decay_target_blacklist(uav: UAV) -> None:
    expired = []
    for target, ttl in uav.target_blacklist.items():
        ttl -= 1
        if ttl <= 0:
            expired.append(target)
        else:
            uav.target_blacklist[target] = ttl
    for target in expired:
        del uav.target_blacklist[target]


def blacklist_target(uav: UAV, target: tuple[int, int] | None, config: RACERConfig) -> None:
    if target is not None:
        uav.target_blacklist[target] = config.target_blacklist_steps


def target_has_visible_gain(uav: UAV, known_map: KnownMap, target: tuple[int, int] | None, config: RACERConfig) -> bool:
    if target is None:
        return False
    if target_is_blacklisted(uav, target, config):
        return False
    x, y = target
    if not known_map.in_bounds(x, y) or known_map.grid[y, x] != FREE:
        return False
    return unknown_gain_from_viewpoint(known_map, target, config.coverage_radius) >= config.viewpoint_min_gain


def select_cp_guided_viewpoint(uav: UAV, known_map: KnownMap, frontier_infos: list[FrontierInfo], cp_path: list[tuple[int, int]], config: RACERConfig) -> tuple[tuple[int, int] | None, list[tuple[int, int]]]:
    planning_grid = planning_grid_for_uav(known_map, config, uav)
    soft_grid = planning_grid_from_known_map(known_map, config, block_unknown=False)
    cursor = update_cp_cursor(uav, cp_path, config.cp_reach_radius)
    ranked = []
    for info in frontier_infos:
        cp_score, cp_idx = cp_score_for_cluster(info.cells, cp_path, cursor, config)
        dist = math.hypot(uav.pos[0] - info.centroid[0], uav.pos[1] - info.centroid[1])
        ranked.append((cp_score + 0.1 * dist, cp_idx, info))
    ranked.sort(key=lambda item: (item[0], item[1], item[2].cluster_id))

    best_score = float("inf")
    best_target = None
    best_path: list[tuple[int, int]] = []
    best_cp_idx = cursor
    priority_window = max(1, config.local_cluster_window)
    search_batches = [ranked[:priority_window], ranked[priority_window:]]
    for batch_idx, batch in enumerate(search_batches):
        if not batch:
            continue
        best_score, best_target, best_path, best_cp_idx = select_best_viewpoint_from_ranked_batch(
            uav,
            batch,
            cp_path,
            cursor,
            planning_grid,
            config,
            extra_sequence_penalty=batch_idx * 2.0,
            current_best=(best_score, best_target, best_path, best_cp_idx),
        )
        if best_target is not None:
            break
    if best_target is None:
        uav.last_planner_mode = "none"
        return None, []
    uav.cp_cursor = best_cp_idx
    shortcut = shortcut_path(best_path, planning_grid, config)
    smoothed = smooth_trajectory(shortcut, planning_grid, config, soft_grid)
    uav.last_planner_mode = "kinodynamic_bspline" if config.use_kinodynamic_astar else "astar_bspline"
    return best_target, smoothed


def select_best_viewpoint_from_ranked_batch(
    uav: UAV,
    ranked_batch: list[tuple[float, int, FrontierInfo]],
    cp_path: list[tuple[int, int]],
    cursor: int,
    planning_grid: np.ndarray,
    config: RACERConfig,
    extra_sequence_penalty: float,
    current_best: tuple[float, tuple[int, int] | None, list[tuple[int, int]], int],
) -> tuple[float, tuple[int, int] | None, list[tuple[int, int]], int]:
    best_score, best_target, best_path, best_cp_idx = current_best
    for seq_idx, (_, _, info) in enumerate(ranked_batch):
        cp_score, cp_idx = cp_score_for_cluster(info.cells, cp_path, cursor, config)
        for vx, vy, gain in info.viewpoints:
            target = (vx, vy)
            if target_is_blacklisted(uav, target, config) or math.hypot(vx - uav.pos[0], vy - uav.pos[1]) < 0.75:
                continue
            rough_path = []
            if config.use_kinodynamic_astar:
                rough_path = kinodynamic_astar(uav.pos, uav.velocity, target, planning_grid, config)
            if not rough_path:
                rough_path = astar(uav.pos, target, planning_grid, config)
            if len(rough_path) <= 1:
                continue
            score = len(rough_path) - 0.85 * gain - 0.20 * len(info.cells) + cp_score + 0.35 * seq_idx + extra_sequence_penalty
            if uav.target is not None and math.hypot(target[0] - uav.target[0], target[1] - uav.target[1]) <= config.target_commit_radius:
                score -= config.current_target_bonus
            if score < best_score:
                best_score = score
                best_target = target
                best_path = rough_path
                best_cp_idx = cp_idx
    return best_score, best_target, best_path, best_cp_idx


def plan_uav_with_hgrid(uav: UAV, known_map: KnownMap, hgrid: HGrid, config: RACERConfig, step_idx: int) -> tuple[bool, str]:
    decay_target_blacklist(uav)
    if uav.path and uav.target is not None:
        target_dist = math.hypot(uav.pos[0] - uav.target[0], uav.pos[1] - uav.target[1])
        if target_dist > config.target_commit_radius and target_has_visible_gain(uav, known_map, uav.target, config):
            return True, "keep"
    if (
        uav.path
        and uav.target is not None
        and not target_is_blacklisted(uav, uav.target, config)
        and step_idx % max(1, config.local_replan_interval) != 0
    ):
        return True, "keep"

    if uav.fis is None:
        uav.fis = FrontierInformationStructure()
    frontier_infos = uav.fis.update(known_map, uav.pos, config)
    owned_blocks = hgrid.blocks_for_uav(uav.id)
    owned_infos = [info for info in frontier_infos if hgrid.cluster_inside_owned_blocks(info.cells, owned_blocks)]
    cp_path = generate_coverage_path_for_blocks(owned_blocks, config.cp_step, uav.pos)
    uav.cp_path = cp_path

    target, path = select_cp_guided_viewpoint(uav, known_map, owned_infos, cp_path, config)
    if path:
        uav.assigned_cluster_count = len(owned_infos)
        uav.set_plan(target, path)
        return True, "cp_owned"
    target, path = select_cp_guided_viewpoint(uav, known_map, frontier_infos, cp_path, config)
    uav.assigned_cluster_count = len(owned_infos)
    uav.set_plan(target, path)
    return bool(path), "cp_fallback" if path else "none"


