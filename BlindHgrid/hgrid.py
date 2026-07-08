"""
RACER-style hgrid exploration prototype.

Adds on top of the previous hgrid version:
    1. The initially unknown 50 x 50 region is split into hgrid blocks.
    2. Each block has an owner UAV.
    3. Each UAV prioritizes frontier clusters inside its own hgrid blocks.
    4. UAVs share maps inside communication range.
    5. UAV pairs inside communication range reassign their held hgrid blocks.
    6. Pairwise assignment balances remaining unknown work and travel distance.
    7. Each UAV builds a simple coverage path over its hgrid blocks.
    8. Frontier viewpoints are selected with CP guidance.
    9. Local collision-free paths are found in the known map.
    10. Local paths are converted to cubic B-spline trajectories when safe.
    11. UAVs exchange short future trajectories.
    12. If inter-UAV collision is predicted, the affected UAV replans to the
        same target while treating neighbor trajectories as dynamic obstacles.
    13. After exploration, all UAVs return to their own starts.

This version uses a 2-D omnidirectional range sensor with ray-casting occlusion
by default. Finite FOV can still be enabled from the command line for experiments.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import math
from dataclasses import dataclass

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
    from scipy.optimize import minimize
except Exception:  # pragma: no cover - keeps the demo runnable without SciPy.
    distance_transform_edt = None
    minimize = None

from map_generator import generate_test_map

from multi_blind import (
    MultiExplorerConfig,
    MultiUAV,
    communication_components,
    global_known_ratio,
    make_column_starts,
    merge_known_maps_for_component,
    union_known_grid,
)
from single_blind import (
    FREE,
    OBSTACLE,
    UNKNOWN,
    FrontierDetector,
    GridWorld,
    KnownMap,
    ViewpointPlanner,
    astar,
)


# Central configuration for this simulation version.
CFG = {
    # Map and run control
    "map_random_seed": 42,
    "map_width": 53,
    "map_height": 50,
    "known_strip_width": 3,
    "max_steps": 1500,
    "stop_known_ratio": 0.985,
    "stall_stop_min_known_ratio": 0.95,
    "global_stall_steps": 250,
    "render_pause": 0.001,
    "render_interval": 2,
    "render_history_tail": 160,
    "render_max_frontier_points": 450,
    "obstacle_count": 50,
    "obstacle_inflation_radius": 0.8,
    "planning_inflation_radius": 0.8,

    # UAV sensing and motion
    "num_uavs": 2,
    "uav_max_speed": 1.0,
    "uav_max_acceleration": 1.0,
    "exploration_radius": 3.0,
    "safe_radius": 1.0,
    "communication_range": 7.0,
    "future_path_horizon": 5,
    "conflict_margin": 0.25,
    "reciprocal_replan_rounds": 2,
    "dynamic_obstacle_inflation": 1.2,

    # HGrid and pairwise task interaction
    "hgrid_block_size": 10,
    "hgrid_level_sizes": (20, 10, 5),
    "hgrid_split_known_ratio": 0.35,
    "hgrid_min_unknown_cells": 2,
    "hgrid_update_interval": 4,
    "pairwise_reassign_interval": 8,
    "hgrid_balance_weight": 0.08,
    "pairwise_request_cooldown": 3,
    "pairwise_success_cooldown": 12,
    "cvrp_max_exact_blocks": 14,
    "cvrp_route_weight": 1.0,
    "cvrp_work_weight": 0.08,

    # CP-guided exploration
    "cp_step": 3,
    "cp_guidance_weight": 0.18,
    "cp_index_weight": 0.04,
    "cp_reach_radius": 2.0,
    "local_cluster_window": 3,
    "local_replan_interval": 5,
    "viewpoint_samples_per_cluster": 4,
    "viewpoint_sample_radius": 4,
    "viewpoint_min_gain": 1,
    "fis_min_cluster_size": 1,
    "fis_max_cluster_size": 28,
    "target_blacklist_steps": 35,
    "target_blacklist_radius": 2.5,
    "target_commit_radius": 1.2,
    "no_progress_steps": 18,
    "no_info_steps": 12,
    "no_progress_distance": 0.2,
    "show_cp_path": True,

    # Continuous local trajectory smoothing
    "use_bspline_smoothing": True,
    "bspline_samples_per_segment": 5,
    "bspline_min_step": 0.35,
    "bspline_max_smoothness_cost": 1.25,
    "bspline_max_turn_angle_deg": 65.0,
    "use_kinodynamic_astar": False,
    "kinodynamic_dt": 1.0,
    "kinodynamic_position_resolution": 0.5,
    "kinodynamic_velocity_resolution": 0.5,
    "kinodynamic_goal_tolerance": 0.75,
    "kinodynamic_max_nodes": 3000,
    "use_kinodynamic_for_viewpoint_scoring": False,
    "safety_sample_step": 0.20,
    "bspline_opt_iterations": 12,
    "bspline_max_control_points": 24,
    "bspline_smoothness_weight": 5.0,
    "bspline_distance_weight": 10.0,
    "bspline_feasibility_weight": 1.0,
    "bspline_guide_weight": 0.15,
    "bspline_swarm_weight": 8.0,
    "bspline_swarm_vertical_scale": 1.0,

    # 2-D sensor model
    "use_camera_fov": False,
    "camera_use_occlusion": True,
    "camera_fov_deg": 360.0,
    "camera_yaw_rate_deg": 90.0,
}


@dataclass
class HGridExplorerConfig(MultiExplorerConfig):
    width: int = CFG["map_width"]
    height: int = CFG["map_height"]
    known_strip_width: int = CFG["known_strip_width"]
    max_steps: int = CFG["max_steps"]
    obstacle_count: int = CFG["obstacle_count"]
    random_seed: int = CFG["map_random_seed"]
    coverage_radius: float = CFG["exploration_radius"]
    safe_radius: float = CFG["safe_radius"]
    inflation_radius: float = CFG["obstacle_inflation_radius"]
    planning_inflation_radius: float = CFG["planning_inflation_radius"]
    stop_known_ratio: float = CFG["stop_known_ratio"]
    stall_stop_min_known_ratio: float = CFG["stall_stop_min_known_ratio"]
    global_stall_steps: int = CFG["global_stall_steps"]
    render_pause: float = CFG["render_pause"]
    render_interval: int = CFG["render_interval"]
    render_history_tail: int = CFG["render_history_tail"]
    render_max_frontier_points: int = CFG["render_max_frontier_points"]
    num_uavs: int = CFG["num_uavs"]
    comm_range: float = CFG["communication_range"]
    max_speed: float = CFG["uav_max_speed"]
    max_acceleration: float = CFG["uav_max_acceleration"]
    future_path_horizon: int = CFG["future_path_horizon"]
    conflict_margin: float = CFG["conflict_margin"]
    reciprocal_replan_rounds: int = CFG["reciprocal_replan_rounds"]
    dynamic_obstacle_inflation: float = CFG["dynamic_obstacle_inflation"]
    hgrid_block_size: int = CFG["hgrid_block_size"]
    hgrid_level_sizes: tuple[int, ...] = CFG["hgrid_level_sizes"]
    hgrid_split_known_ratio: float = CFG["hgrid_split_known_ratio"]
    hgrid_min_unknown_cells: int = CFG["hgrid_min_unknown_cells"]
    hgrid_update_interval: int = CFG["hgrid_update_interval"]
    pairwise_reassign_interval: int = CFG["pairwise_reassign_interval"]
    hgrid_balance_weight: float = CFG["hgrid_balance_weight"]
    pairwise_request_cooldown: int = CFG["pairwise_request_cooldown"]
    pairwise_success_cooldown: int = CFG["pairwise_success_cooldown"]
    cvrp_max_exact_blocks: int = CFG["cvrp_max_exact_blocks"]
    cvrp_route_weight: float = CFG["cvrp_route_weight"]
    cvrp_work_weight: float = CFG["cvrp_work_weight"]
    cp_step: int = CFG["cp_step"]
    cp_guidance_weight: float = CFG["cp_guidance_weight"]
    cp_index_weight: float = CFG["cp_index_weight"]
    cp_reach_radius: float = CFG["cp_reach_radius"]
    local_cluster_window: int = CFG["local_cluster_window"]
    local_replan_interval: int = CFG["local_replan_interval"]
    viewpoint_samples_per_cluster: int = CFG["viewpoint_samples_per_cluster"]
    viewpoint_sample_radius: int = CFG["viewpoint_sample_radius"]
    viewpoint_min_gain: int = CFG["viewpoint_min_gain"]
    fis_min_cluster_size: int = CFG["fis_min_cluster_size"]
    fis_max_cluster_size: int = CFG["fis_max_cluster_size"]
    target_blacklist_steps: int = CFG["target_blacklist_steps"]
    target_blacklist_radius: float = CFG["target_blacklist_radius"]
    target_commit_radius: float = CFG["target_commit_radius"]
    no_progress_steps: int = CFG["no_progress_steps"]
    no_info_steps: int = CFG["no_info_steps"]
    no_progress_distance: float = CFG["no_progress_distance"]
    show_cp_path: bool = CFG["show_cp_path"]
    use_bspline_smoothing: bool = CFG["use_bspline_smoothing"]
    bspline_samples_per_segment: int = CFG["bspline_samples_per_segment"]
    bspline_min_step: float = CFG["bspline_min_step"]
    bspline_max_smoothness_cost: float = CFG["bspline_max_smoothness_cost"]
    bspline_max_turn_angle_deg: float = CFG["bspline_max_turn_angle_deg"]
    use_kinodynamic_astar: bool = CFG["use_kinodynamic_astar"]
    kinodynamic_dt: float = CFG["kinodynamic_dt"]
    kinodynamic_position_resolution: float = CFG["kinodynamic_position_resolution"]
    kinodynamic_velocity_resolution: float = CFG["kinodynamic_velocity_resolution"]
    kinodynamic_goal_tolerance: float = CFG["kinodynamic_goal_tolerance"]
    kinodynamic_max_nodes: int = CFG["kinodynamic_max_nodes"]
    use_kinodynamic_for_viewpoint_scoring: bool = CFG["use_kinodynamic_for_viewpoint_scoring"]
    safety_sample_step: float = CFG["safety_sample_step"]
    bspline_opt_iterations: int = CFG["bspline_opt_iterations"]
    bspline_max_control_points: int = CFG["bspline_max_control_points"]
    bspline_smoothness_weight: float = CFG["bspline_smoothness_weight"]
    bspline_distance_weight: float = CFG["bspline_distance_weight"]
    bspline_feasibility_weight: float = CFG["bspline_feasibility_weight"]
    bspline_guide_weight: float = CFG["bspline_guide_weight"]
    bspline_swarm_weight: float = CFG["bspline_swarm_weight"]
    bspline_swarm_vertical_scale: float = CFG["bspline_swarm_vertical_scale"]
    use_camera_fov: bool = CFG["use_camera_fov"]
    camera_use_occlusion: bool = CFG["camera_use_occlusion"]
    camera_fov_deg: float = CFG["camera_fov_deg"]
    camera_yaw_rate_deg: float = CFG["camera_yaw_rate_deg"]


class GridWorld:
    """
    Map generation matching multi_path_plan.py.

    map_generator.generate_test_map() creates mixed circle/rectangle obstacles,
    then this class rasterizes them with an inflation radius. The only extra
    handling here is filtering/clearing the left known launch strip used by
    blind exploration.
    """

    def __init__(self, config: HGridExplorerConfig):
        self.config = config
        self.width = config.width
        self.height = config.height
        self.circles, self.rectangles, self.base_density = self._generate_obstacle_shapes()
        self.raw_obstacle_map = self._rasterize_shapes(inflation_radius=0.0)
        self.true_map = self._rasterize_shapes(inflation_radius=config.inflation_radius)
        self._clear_launch_area(self.raw_obstacle_map)
        self._clear_launch_area(self.true_map)

    def _generate_obstacle_shapes(self) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float, float]], float]:
        circles, rectangles, density = generate_test_map(
            width=self.width,
            height=self.height,
            num_obstacles=self.config.obstacle_count,
            seed=self.config.random_seed,
            safe_zone_size=6,
        )

        circles, rectangles = self._filter_launch_strip_obstacles(circles, rectangles)
        return circles, rectangles, density

    def _filter_launch_strip_obstacles(
        self,
        circles: list[tuple[float, float, float]],
        rectangles: list[tuple[float, float, float, float]],
    ) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float, float]]]:
        strip = float(self.config.known_strip_width)

        filtered_circles = [
            (cx, cy, r)
            for cx, cy, r in circles
            if (cx - r) >= strip
        ]
        filtered_rectangles = [
            (rx, ry, rw, rh)
            for rx, ry, rw, rh in rectangles
            if rx >= strip
        ]

        return filtered_circles, filtered_rectangles

    def _rasterize_shapes(self, inflation_radius: float) -> np.ndarray:
        grid = np.zeros((self.height, self.width), dtype=np.int8)

        for y in range(self.height):
            for x in range(self.width):
                occupied = False

                for cx, cy, r in self.circles:
                    if math.hypot(x - cx, y - cy) <= r + inflation_radius:
                        occupied = True
                        break

                if not occupied:
                    for rx, ry, rw, rh in self.rectangles:
                        if (rx - inflation_radius) <= x <= (rx + rw + inflation_radius) and (
                            ry - inflation_radius
                        ) <= y <= (ry + rh + inflation_radius):
                            occupied = True
                            break

                if occupied:
                    grid[y, x] = OBSTACLE

        return grid

    def _clear_launch_area(self, grid: np.ndarray) -> None:
        strip = self.config.known_strip_width
        grid[:, :strip] = FREE


@dataclass
class HGridBlock:
    block_id: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    level: int = 0
    parent_id: int | None = None
    owner_id: int | None = None
    unknown_cells: int = 0

    def contains(self, x: int, y: int) -> bool:
        return self.x_min <= x < self.x_max and self.y_min <= y < self.y_max

    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max - 1) / 2.0, (self.y_min + self.y_max - 1) / 2.0)

    def width(self) -> int:
        return self.x_max - self.x_min

    def height(self) -> int:
        return self.y_max - self.y_min

    def area(self) -> int:
        return self.width() * self.height()


class HGrid:
    """Online 2-D hgrid active-cell representation of unexplored space."""

    def __init__(self, config: HGridExplorerConfig):
        self.config = config
        self.level_sizes = tuple(sorted(set(config.hgrid_level_sizes), reverse=True))
        if not self.level_sizes:
            self.level_sizes = (config.hgrid_block_size,)
        self.blocks: list[HGridBlock] = []
        self.next_block_id = 0
        self.blocks = self._make_initial_blocks()

    def _new_block(
        self,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        level: int,
        parent_id: int | None = None,
        owner_id: int | None = None,
    ) -> HGridBlock:
        block = HGridBlock(
            block_id=self.next_block_id,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            level=level,
            parent_id=parent_id,
            owner_id=owner_id,
        )
        self.next_block_id += 1
        return block

    def _make_initial_blocks(self) -> list[HGridBlock]:
        blocks = []
        x0 = self.config.known_strip_width
        size = self.level_sizes[0]

        for y_min in range(0, self.config.height, size):
            for x_min in range(x0, self.config.width, size):
                blocks.append(
                    self._new_block(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=min(self.config.width, x_min + size),
                        y_max=min(self.config.height, y_min + size),
                        level=0,
                    )
                )

        return blocks

    def can_split(self, block: HGridBlock) -> bool:
        return block.level + 1 < len(self.level_sizes)

    def child_size_for(self, block: HGridBlock) -> int:
        return self.level_sizes[min(block.level + 1, len(self.level_sizes) - 1)]

    def split_block(self, block: HGridBlock, known_grid: np.ndarray) -> list[HGridBlock]:
        child_size = self.child_size_for(block)
        children = []

        for y_min in range(block.y_min, block.y_max, child_size):
            for x_min in range(block.x_min, block.x_max, child_size):
                child = self._new_block(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=min(block.x_max, x_min + child_size),
                    y_max=min(block.y_max, y_min + child_size),
                    level=block.level + 1,
                    parent_id=block.block_id,
                    owner_id=block.owner_id,
                )
                child.unknown_cells = hgrid_block_unknown_work(child, known_grid, self.raw_obstacle_map)
                if child.unknown_cells > 0:
                    children.append(child)

        return children

    def update_active_cells(self, known_grid: np.ndarray, raw_obstacle_map: np.ndarray | None = None) -> tuple[int, int]:
        """
        Update the active hgrid list as the map changes.

        Coarse cells that are partly explored are subdivided. Finest cells with
        almost no unknown space left are removed from the active task set.
        """
        updated_blocks: list[HGridBlock] = []
        split_count = 0
        removed_count = 0
        self.raw_obstacle_map = raw_obstacle_map

        for block in self.blocks:
            block.unknown_cells = hgrid_block_unknown_work(block, known_grid, raw_obstacle_map)
            if block.unknown_cells <= 0:
                removed_count += 1
                continue

            known_ratio = 1.0 - (block.unknown_cells / max(1, block.area()))
            if self.can_split(block) and known_ratio >= self.config.hgrid_split_known_ratio:
                children = self.split_block(block, known_grid)
                if children:
                    updated_blocks.extend(children)
                    split_count += 1
                else:
                    removed_count += 1
                continue

            if (
                not self.can_split(block)
                and block.unknown_cells <= self.config.hgrid_min_unknown_cells
            ):
                removed_count += 1
                continue

            updated_blocks.append(block)

        self.blocks = updated_blocks
        return split_count, removed_count

    def prune_blocks_without_frontiers(self, frontier_cells: set[tuple[int, int]]) -> int:
        if not frontier_cells:
            removed = len(self.blocks)
            self.blocks = []
            return removed

        kept = []
        removed = 0
        for block in self.blocks:
            has_frontier = any(block.contains(x, y) for x, y in frontier_cells)
            if has_frontier:
                kept.append(block)
            else:
                removed += 1

        self.blocks = kept
        return removed

    def assign_initial_owners(self, uavs: list[MultiUAV]) -> None:
        """Assign each block to the nearest UAV start, with a light load tie-break."""
        load = {uav.id: 0 for uav in uavs}

        for block in self.blocks:
            cx, cy = block.center()
            owner = min(
                uavs,
                key=lambda uav: (
                    math.hypot(uav.start[0] - cx, uav.start[1] - cy),
                    load[uav.id],
                ),
            )
            block.owner_id = owner.id
            load[owner.id] += block.width() * block.height()

    def blocks_for_uav(self, uav_id: int) -> list[HGridBlock]:
        return [block for block in self.blocks if block.owner_id == uav_id]

    def block_for_cell(self, x: int, y: int) -> HGridBlock | None:
        for block in self.blocks:
            if block.contains(x, y):
                return block
        return None

    def cluster_inside_owned_blocks(
        self,
        cluster: list[tuple[int, int]],
        owned_blocks: list[HGridBlock],
    ) -> bool:
        if not owned_blocks:
            return False

        for x, y in cluster:
            for block in owned_blocks:
                if block.contains(x, y):
                    return True

        return False


@dataclass
class FrontierInfo:
    cluster_id: int
    cells: list[tuple[int, int]]
    centroid: tuple[float, float]
    viewpoints: list[tuple[int, int, float, int]]


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def angle_between(start: tuple[float, float], end: tuple[float, float]) -> float:
    return math.atan2(end[1] - start[1], end[0] - start[0])


def point_in_camera_fov(
    origin: tuple[float, float],
    point: tuple[float, float],
    yaw: float,
    config: HGridExplorerConfig,
) -> bool:
    if not config.use_camera_fov:
        return True
    bearing = angle_between(origin, point)
    half_fov = math.radians(config.camera_fov_deg) / 2.0
    return abs(normalize_angle(bearing - yaw)) <= half_fov + 1e-9


def cell_visible_from(
    known_map: KnownMap,
    origin: tuple[float, float],
    cell: tuple[int, int],
    yaw: float,
    config: HGridExplorerConfig,
) -> bool:
    target = (float(cell[0]), float(cell[1]))
    if math.hypot(target[0] - origin[0], target[1] - origin[1]) > config.coverage_radius:
        return False
    if not point_in_camera_fov(origin, target, yaw, config):
        return False
    if not config.camera_use_occlusion:
        return True

    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    steps = max(int(math.ceil(max(abs(dx), abs(dy)) * 2)), 1)
    for idx in range(1, steps):
        x = int(round(origin[0] + dx * idx / steps))
        y = int(round(origin[1] + dy * idx / steps))
        if not (0 <= x < known_map.world.width and 0 <= y < known_map.world.height):
            return False
        if known_map.world.raw_obstacle_map[y, x] == OBSTACLE:
            return False

    return True


def update_known_map_with_camera(
    known_map: KnownMap,
    uav: MultiUAV,
    config: HGridExplorerConfig,
) -> int:
    cx, cy = (float(uav.pos[0]), float(uav.pos[1]))
    yaw = float(getattr(uav, "yaw", 0.0))
    r_int = int(math.ceil(config.coverage_radius))
    newly_known = 0

    for y in range(max(0, int(round(cy)) - r_int), min(known_map.world.height, int(round(cy)) + r_int + 1)):
        for x in range(max(0, int(round(cx)) - r_int), min(known_map.world.width, int(round(cx)) + r_int + 1)):
            if not cell_visible_from(known_map, (cx, cy), (x, y), yaw, config):
                continue
            if known_map.grid[y, x] == UNKNOWN:
                newly_known += 1
            known_map.grid[y, x] = known_map.world.raw_obstacle_map[y, x]

    return newly_known


def safe_planning_grid_from_known_map(
    known_map: KnownMap,
    config: HGridExplorerConfig,
) -> list[list[int]]:
    """
    Build a planning grid from the perceived map.

    Perception stores raw observed obstacles, while planning needs a safety
    margin around known obstacles. Unknown cells are kept occupied for safety.
    """
    planning = np.where(known_map.grid == FREE, FREE, OBSTACLE)
    obstacle_y, obstacle_x = np.where(known_map.grid == OBSTACLE)
    planning_inflation = max(float(config.planning_inflation_radius), float(config.safe_radius))
    r_int = int(math.ceil(planning_inflation))

    for ox, oy in zip(obstacle_x, obstacle_y):
        for y in range(max(0, oy - r_int), min(known_map.world.height, oy + r_int + 1)):
            for x in range(max(0, ox - r_int), min(known_map.world.width, ox + r_int + 1)):
                if math.hypot(x - ox, y - oy) <= planning_inflation:
                    planning[y, x] = OBSTACLE

    return planning.tolist()


@dataclass
class PlanningContext:
    """
    Planning data shared by kinodynamic search and trajectory optimization.

    grid blocks unknown space and inflated known obstacles. obstacle_clearance is
    a continuous EDT-like distance to known physical obstacles only, so frontier
    cells beside unknown space can still be selected while real obstacles cannot
    enter the UAV safety radius.
    """

    grid: list[list[int]]
    obstacle_clearance: np.ndarray
    clearance_grad_x: np.ndarray
    clearance_grad_y: np.ndarray
    min_clearance: float

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0]) if self.grid else 0


def _nearest_obstacle_distance_fallback(obstacle_mask: np.ndarray) -> np.ndarray:
    obstacle_cells = np.argwhere(obstacle_mask)
    height, width = obstacle_mask.shape
    if obstacle_cells.size == 0:
        return np.full((height, width), float("inf"), dtype=float)

    yy, xx = np.indices((height, width))
    dist_sq = np.full((height, width), float("inf"), dtype=float)
    for oy, ox in obstacle_cells:
        dist_sq = np.minimum(dist_sq, (xx - ox) ** 2 + (yy - oy) ** 2)
    return np.sqrt(dist_sq)


def _obstacle_clearance_field(obstacle_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.any(obstacle_mask):
        clearance = np.full(obstacle_mask.shape, float("inf"), dtype=float)
        grad_y = np.zeros_like(clearance)
        grad_x = np.zeros_like(clearance)
        return clearance, grad_x, grad_y

    if distance_transform_edt is not None:
        clearance = distance_transform_edt(~obstacle_mask).astype(float)
    else:
        clearance = _nearest_obstacle_distance_fallback(obstacle_mask)

    grad_y, grad_x = np.gradient(clearance)
    return clearance, grad_x, grad_y


def planning_context_from_known_map(
    known_map: KnownMap,
    config: HGridExplorerConfig,
    planning_grid: list[list[int]] | None = None,
) -> PlanningContext:
    if planning_grid is None:
        planning_grid = safe_planning_grid_from_known_map(known_map, config)

    obstacle_mask = known_map.grid == OBSTACLE
    clearance, grad_x, grad_y = _obstacle_clearance_field(obstacle_mask)
    return PlanningContext(
        grid=planning_grid,
        obstacle_clearance=clearance,
        clearance_grad_x=grad_x,
        clearance_grad_y=grad_y,
        min_clearance=float(config.safe_radius),
    )


def _bilinear_sample(array: np.ndarray, point: tuple[float, float], default: float = 0.0) -> float:
    if array.size == 0:
        return default

    x, y = point
    height, width = array.shape
    if not (0.0 <= x <= width - 1 and 0.0 <= y <= height - 1):
        return default

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(width - 1, x0 + 1)
    y1 = min(height - 1, y0 + 1)
    tx = x - x0
    ty = y - y0

    v00 = array[y0, x0]
    v10 = array[y0, x1]
    v01 = array[y1, x0]
    v11 = array[y1, x1]
    values = (v00, v10, v01, v11)
    if any(math.isinf(float(value)) for value in values):
        if all(math.isinf(float(value)) for value in values):
            return float("inf")
        finite_values = [float(value) for value in values if math.isfinite(float(value))]
        return max(finite_values) if finite_values else default
    return float((1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11)


def clearance_at(planning: PlanningContext, point: tuple[float, float]) -> float:
    return _bilinear_sample(planning.obstacle_clearance, point, default=-float("inf"))


def clearance_gradient_at(planning: PlanningContext, point: tuple[float, float]) -> tuple[float, float]:
    gx = _bilinear_sample(planning.clearance_grad_x, point, default=0.0)
    gy = _bilinear_sample(planning.clearance_grad_y, point, default=0.0)
    return gx, gy


def split_large_frontier_cluster(
    cluster: list[tuple[int, int]],
    max_size: int,
) -> list[list[tuple[int, int]]]:
    if len(cluster) <= max_size:
        return [cluster]

    cx, cy = cluster_centroid(cluster)
    ordered = sorted(cluster, key=lambda cell: math.atan2(cell[1] - cy, cell[0] - cx))
    return [ordered[idx : idx + max_size] for idx in range(0, len(ordered), max_size)]


def unknown_gain_from_viewpoint(
    known_map: KnownMap,
    viewpoint: tuple[int, int],
    yaw: float,
    config: HGridExplorerConfig,
) -> int:
    grid = known_map.grid
    x0, y0 = viewpoint
    r_int = int(math.ceil(config.coverage_radius))
    count = 0

    for y in range(max(0, y0 - r_int), min(grid.shape[0], y0 + r_int + 1)):
        for x in range(max(0, x0 - r_int), min(grid.shape[1], x0 + r_int + 1)):
            if not is_explorable_unknown_cell(known_map, x, y):
                continue
            if cell_visible_from(known_map, (x0, y0), (x, y), yaw, config):
                count += 1

    return count


def yaw_toward_unknown(
    known_map: KnownMap,
    viewpoint: tuple[int, int],
    fallback_point: tuple[float, float],
    config: HGridExplorerConfig,
) -> float:
    grid = known_map.grid
    x0, y0 = viewpoint
    r_int = int(math.ceil(config.coverage_radius))
    ux = 0.0
    uy = 0.0
    count = 0

    for y in range(max(0, y0 - r_int), min(grid.shape[0], y0 + r_int + 1)):
        for x in range(max(0, x0 - r_int), min(grid.shape[1], x0 + r_int + 1)):
            if not is_explorable_unknown_cell(known_map, x, y):
                continue
            if math.hypot(x - x0, y - y0) > config.coverage_radius:
                continue
            ux += x - x0
            uy += y - y0
            count += 1

    if count > 0 and math.hypot(ux, uy) > 1e-6:
        return math.atan2(uy, ux)
    return angle_between((x0, y0), fallback_point)


def candidate_viewpoint_cells_for_cluster(
    known_map: KnownMap,
    cluster: list[tuple[int, int]],
    config: HGridExplorerConfig,
) -> list[tuple[int, int]]:
    grid = known_map.grid
    sample_radius = max(1, int(config.viewpoint_sample_radius))
    min_x = max(0, min(x for x, _ in cluster) - sample_radius)
    max_x = min(grid.shape[1] - 1, max(x for x, _ in cluster) + sample_radius)
    min_y = max(0, min(y for _, y in cluster) - sample_radius)
    max_y = min(grid.shape[0] - 1, max(y for _, y in cluster) + sample_radius)
    candidates = []

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if grid[y, x] != FREE:
                continue
            dist_to_frontier = min(math.hypot(x - fx, y - fy) for fx, fy in cluster)
            if dist_to_frontier > config.coverage_radius:
                continue
            candidates.append((x, y))

    return candidates


def sample_viewpoints_for_cluster(
    known_map: KnownMap,
    current_pos: tuple[float, float],
    cluster: list[tuple[int, int]],
    config: HGridExplorerConfig,
) -> list[tuple[int, int, float, int]]:
    grid = known_map.grid
    cx, cy = cluster_centroid(cluster)
    candidates = []

    for x, y in candidate_viewpoint_cells_for_cluster(known_map, cluster, config):
        yaw = yaw_toward_unknown(known_map, (x, y), (cx, cy), config)
        gain = unknown_gain_from_viewpoint(known_map, (x, y), yaw, config)
        if gain < config.viewpoint_min_gain:
            continue
        dist_to_center = math.hypot(x - cx, y - cy)
        score = -2.5 * gain + 0.12 * dist_to_center
        candidates.append((score, x, y, yaw, gain))

    candidates.sort()
    return [
        (x, y, yaw, gain)
        for _, x, y, yaw, gain in candidates[: max(1, config.viewpoint_samples_per_cluster)]
    ]


class FrontierInformationStructure:
    """2-D incremental-FIS-style cache for frontier clusters and viewpoints."""

    def __init__(self) -> None:
        self.signature: tuple[int, ...] | None = None
        self.frontiers: list[FrontierInfo] = []

    def update(
        self,
        known_map: KnownMap,
        current_pos: tuple[float, float],
        config: HGridExplorerConfig,
    ) -> list[FrontierInfo]:
        known_count = int(np.sum(known_map.grid != UNKNOWN))
        free_count = int(np.sum(known_map.grid == FREE))
        signature = (known_count, free_count)
        if self.signature == signature:
            return self.frontiers

        raw_clusters = FrontierDetector(known_map).cluster_frontiers()
        clusters: list[list[tuple[int, int]]] = []
        for cluster in raw_clusters:
            cluster = [
                (x, y)
                for x, y in cluster
                if known_map.grid[y, x] == FREE
                and known_map.world.raw_obstacle_map[y, x] != OBSTACLE
                and frontier_cell_has_explorable_unknown(known_map, x, y)
            ]
            if len(cluster) < config.fis_min_cluster_size:
                continue
            clusters.extend(split_large_frontier_cluster(cluster, config.fis_max_cluster_size))

        infos = []
        for cluster_id, cluster in enumerate(clusters):
            viewpoints = sample_viewpoints_for_cluster(known_map, current_pos, cluster, config)
            if not viewpoints:
                continue
            infos.append(
                FrontierInfo(
                    cluster_id=cluster_id,
                    cells=cluster,
                    centroid=cluster_centroid(cluster),
                    viewpoints=viewpoints,
                )
            )

        self.signature = signature
        self.frontiers = infos
        return infos


def hgrid_block_unknown_work(
    block: HGridBlock,
    known_grid: np.ndarray,
    raw_obstacle_map: np.ndarray | None = None,
) -> int:
    region = known_grid[block.y_min : block.y_max, block.x_min : block.x_max]
    unknown = region == UNKNOWN
    if raw_obstacle_map is not None:
        obstacle_region = raw_obstacle_map[block.y_min : block.y_max, block.x_min : block.x_max]
        unknown = unknown & (obstacle_region != OBSTACLE)
    return int(np.sum(unknown))


def is_explorable_unknown_cell(known_map: KnownMap, x: int, y: int) -> bool:
    return (
        0 <= x < known_map.world.width
        and 0 <= y < known_map.world.height
        and known_map.grid[y, x] == UNKNOWN
        and known_map.world.raw_obstacle_map[y, x] != OBSTACLE
    )


def frontier_cell_has_explorable_unknown(known_map: KnownMap, x: int, y: int) -> bool:
    for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
        if is_explorable_unknown_cell(known_map, nx, ny):
            return True
    return False


def detect_explorable_frontier_cells(known_map: KnownMap) -> set[tuple[int, int]]:
    frontiers = set()
    for y in range(known_map.world.height):
        for x in range(known_map.world.width):
            if known_map.grid[y, x] != FREE:
                continue
            if known_map.world.raw_obstacle_map[y, x] == OBSTACLE:
                continue
            if frontier_cell_has_explorable_unknown(known_map, x, y):
                frontiers.add((x, y))
    return frontiers


def explorable_known_ratio(
    known_maps: list[KnownMap],
    raw_obstacle_map: np.ndarray,
    known_strip_width: int,
) -> float:
    merged = union_known_grid(known_maps)
    region = merged[:, known_strip_width:]
    explorable_region = raw_obstacle_map[:, known_strip_width:] != OBSTACLE
    if not np.any(explorable_region):
        return 1.0
    return float(np.mean(region[explorable_region] != UNKNOWN))


def communicating_pairs(uavs: list[MultiUAV], comm_range: float) -> list[tuple[int, int]]:
    pairs = []
    for i in range(len(uavs)):
        for j in range(i + 1, len(uavs)):
            if uavs[i].distance_to(uavs[j]) <= comm_range:
                pairs.append((i, j))
    return pairs


def block_route_length(
    start_pos: tuple[float, float] | tuple[int, int],
    blocks: list[HGridBlock],
    step: int,
) -> float:
    if not blocks:
        return 0.0

    remaining = list(blocks)
    current = (float(start_pos[0]), float(start_pos[1]))
    route_length = 0.0

    while remaining:
        next_idx = min(
            range(len(remaining)),
            key=lambda idx: math.hypot(
                current[0] - remaining[idx].center()[0],
                current[1] - remaining[idx].center()[1],
            ),
        )
        block = remaining.pop(next_idx)
        center = block.center()
        route_length += math.hypot(current[0] - center[0], current[1] - center[1])
        route_length += max(block.width(), block.height()) / max(1, step)
        current = center

    return route_length


def cvrp_like_pair_partition(
    uav_a: MultiUAV,
    uav_b: MultiUAV,
    candidate_blocks: list[tuple[HGridBlock, int]],
    config: HGridExplorerConfig,
) -> dict[int, int]:
    blocks = [block for block, _ in candidate_blocks]
    work_by_block = {block.block_id: work for block, work in candidate_blocks}
    total_work = sum(work_by_block.values())

    def assignment_score(bits: tuple[int, ...]) -> float:
        blocks_a = [block for bit, block in zip(bits, blocks) if bit == 0]
        blocks_b = [block for bit, block in zip(bits, blocks) if bit == 1]
        work_a = sum(work_by_block[block.block_id] for block in blocks_a)
        work_b = total_work - work_a
        route_a = block_route_length(uav_a.pos, blocks_a, config.cp_step)
        route_b = block_route_length(uav_b.pos, blocks_b, config.cp_step)
        return (
            config.cvrp_route_weight * (route_a + route_b)
            + config.cvrp_work_weight * abs(work_a - work_b)
        )

    if len(blocks) <= config.cvrp_max_exact_blocks:
        best_bits = min(itertools.product((0, 1), repeat=len(blocks)), key=assignment_score)
    else:
        sorted_items = sorted(
            candidate_blocks,
            key=lambda item: (
                -item[1],
                item[0].block_id,
            ),
        )
        work_load = {uav_a.id: 0.0, uav_b.id: 0.0}
        assigned: dict[int, int] = {}
        for block, work in sorted_items:
            owners = (uav_a, uav_b)
            owner = min(
                owners,
                key=lambda uav: (
                    block_route_length(uav.pos, [block], config.cp_step)
                    + config.hgrid_balance_weight * (work_load[uav.id] + work)
                ),
            )
            assigned[block.block_id] = owner.id
            work_load[owner.id] += work
        return assigned

    return {
        block.block_id: (uav_a.id if bit == 0 else uav_b.id)
        for bit, block in zip(best_bits, blocks)
    }


def pairwise_reassign_hgrid_blocks(
    uav_a: MultiUAV,
    uav_b: MultiUAV,
    hgrid: HGrid,
    known_grid: np.ndarray,
    config: HGridExplorerConfig,
) -> int:
    """
    Pairwise CVRP-like active-cell reassignment.

    The pair pools both UAVs' active hgrid cells and searches for a two-route
    partition that trades off coverage-route length and unknown workload.
    """
    pair_ids = {uav_a.id, uav_b.id}
    raw_obstacle_map = getattr(hgrid, "raw_obstacle_map", None)
    candidate_blocks = [
        (block, hgrid_block_unknown_work(block, known_grid, raw_obstacle_map))
        for block in hgrid.blocks
        if block.owner_id in pair_ids
    ]
    candidate_blocks = [(block, work) for block, work in candidate_blocks if work > 0]

    if len(candidate_blocks) <= 1:
        return 0

    new_owner_by_block_id = cvrp_like_pair_partition(uav_a, uav_b, candidate_blocks, config)

    changed = 0
    for block, _ in candidate_blocks:
        new_owner = new_owner_by_block_id[block.block_id]
        if block.owner_id != new_owner:
            changed += 1
            block.owner_id = new_owner

    return changed


def pairwise_request_response_hgrid_blocks(
    uavs: list[MultiUAV],
    known_maps: list[KnownMap],
    hgrid: HGrid,
    step_idx: int,
    config: HGridExplorerConfig,
) -> tuple[int, int]:
    total_changed = 0
    success_count = 0
    busy: set[int] = set()
    merged_grid = union_known_grid(known_maps)

    for i, uav in enumerate(uavs):
        if i in busy:
            continue
        last_attempt = int(getattr(uav, "hgrid_last_attempt", -10**9))
        if step_idx - last_attempt < config.pairwise_request_cooldown:
            continue

        candidates = []
        for j, other in enumerate(uavs):
            if i == j or j in busy:
                continue
            if uav.distance_to(other) > config.comm_range:
                continue
            other_attempt = int(getattr(other, "hgrid_last_attempt", -10**9))
            if step_idx - other_attempt < config.pairwise_request_cooldown:
                continue
            success_times = getattr(uav, "hgrid_last_success", {})
            last_success = int(success_times.get(other.id, -10**9))
            if step_idx - last_success < config.pairwise_success_cooldown:
                continue
            candidates.append((last_success, j, other))

        if not candidates:
            continue

        _, j, other = min(candidates, key=lambda item: (item[0], item[2].id))
        setattr(uav, "hgrid_last_attempt", step_idx)
        setattr(other, "hgrid_last_attempt", step_idx)

        before = total_changed
        total_changed += pairwise_reassign_hgrid_blocks(
            uav_a=uav,
            uav_b=other,
            hgrid=hgrid,
            known_grid=merged_grid,
            config=config,
        )
        busy.add(i)
        busy.add(j)
        success_count += 1

        for src, dst in ((uav, other), (other, uav)):
            success_times = dict(getattr(src, "hgrid_last_success", {}))
            success_times[dst.id] = step_idx
            setattr(src, "hgrid_last_success", success_times)

        if total_changed != before:
            setattr(uav, "cp_cursor", 0)
            setattr(other, "cp_cursor", 0)

    return total_changed, success_count


def generate_coverage_path_for_blocks(
    blocks: list[HGridBlock],
    step: int,
    start_pos: tuple[float, float] | tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    """Generate a CP over assigned hgrid cells using nearest-neighbor routing."""
    if not blocks:
        return []

    step = max(1, step)
    waypoints = []
    remaining = list(blocks)
    ordered_blocks: list[HGridBlock] = []

    if start_pos is None:
        ordered_blocks = sorted(blocks, key=lambda block: (block.y_min, block.x_min, block.level))
    else:
        current = (float(start_pos[0]), float(start_pos[1]))
        while remaining:
            idx = min(
                range(len(remaining)),
                key=lambda i: math.hypot(current[0] - remaining[i].center()[0], current[1] - remaining[i].center()[1]),
            )
            block = remaining.pop(idx)
            ordered_blocks.append(block)
            current = block.center()

    for row_idx, block in enumerate(ordered_blocks):
        y_values = list(range(block.y_min, block.y_max, step))
        if not y_values or y_values[-1] != block.y_max - 1:
            y_values.append(block.y_max - 1)

        for local_row, y in enumerate(y_values):
            x_values = list(range(block.x_min, block.x_max, step))
            if not x_values or x_values[-1] != block.x_max - 1:
                x_values.append(block.x_max - 1)

            reverse = (row_idx + local_row) % 2 == 1
            if reverse:
                x_values.reverse()

            for x in x_values:
                waypoints.append((x, y))

    return waypoints


def cluster_centroid(cluster: list[tuple[int, int]]) -> tuple[float, float]:
    return (
        sum(cell[0] for cell in cluster) / len(cluster),
        sum(cell[1] for cell in cluster) / len(cluster),
    )


def update_cp_cursor(uav: MultiUAV, cp_path: list[tuple[int, int]], reach_radius: float) -> int:
    if not cp_path:
        setattr(uav, "cp_cursor", 0)
        return 0

    cursor = int(getattr(uav, "cp_cursor", 0))
    cursor = max(0, min(cursor, len(cp_path) - 1))

    # Move forward once the UAV is close to the current CP point.
    while cursor < len(cp_path) - 1:
        px, py = cp_path[cursor]
        if math.hypot(uav.pos[0] - px, uav.pos[1] - py) > reach_radius:
            break
        cursor += 1

    # If task ownership changed, recover by snapping to the nearest future CP
    # point instead of forcing the UAV to chase an old cursor.
    nearest_idx = min(
        range(cursor, len(cp_path)),
        key=lambda idx: math.hypot(uav.pos[0] - cp_path[idx][0], uav.pos[1] - cp_path[idx][1]),
    )
    cursor = max(cursor, nearest_idx)
    setattr(uav, "cp_cursor", cursor)
    return cursor


def cp_score_for_cluster(
    cluster: list[tuple[int, int]],
    cp_path: list[tuple[int, int]],
    cursor: int,
    config: HGridExplorerConfig,
) -> tuple[float, int]:
    if not cp_path:
        return 0.0, 0

    cx, cy = cluster_centroid(cluster)
    best_idx = min(
        range(cursor, len(cp_path)),
        key=lambda idx: math.hypot(cx - cp_path[idx][0], cy - cp_path[idx][1]),
    )
    best_dist = math.hypot(cx - cp_path[best_idx][0], cy - cp_path[best_idx][1])
    cp_score = config.cp_guidance_weight * best_dist
    cp_score += config.cp_index_weight * max(0, best_idx - cursor)
    return cp_score, best_idx


def discretize_pos(pos: tuple[float, float] | tuple[int, int]) -> tuple[int, int]:
    return (int(round(pos[0])), int(round(pos[1])))


def get_uav_velocity(uav: MultiUAV) -> tuple[int, int]:
    velocity = getattr(uav, "velocity", (0, 0))
    return (float(velocity[0]), float(velocity[1]))


PlanningInput = PlanningContext | list[list[int]]


def planning_grid_cells(planning: PlanningInput) -> list[list[int]]:
    return planning.grid if isinstance(planning, PlanningContext) else planning


def in_planning_bounds(point: tuple[int, int], planning_grid: PlanningInput) -> bool:
    grid = planning_grid_cells(planning_grid)
    x, y = point
    return 0 <= y < len(grid) and 0 <= x < len(grid[0])


def point_is_safe(
    point: tuple[float, float],
    planning: PlanningInput,
) -> bool:
    grid = planning_grid_cells(planning)
    grid_point = discretize_pos(point)
    if not in_planning_bounds(grid_point, grid):
        return False
    if grid[grid_point[1]][grid_point[0]] == OBSTACLE:
        return False
    if isinstance(planning, PlanningContext):
        return clearance_at(planning, point) >= planning.min_clearance - 1e-6
    return True


def segment_is_free(
    start: tuple[float, float] | tuple[int, int],
    end: tuple[float, float] | tuple[int, int],
    planning_grid: PlanningInput,
    sample_step: float = 0.25,
) -> bool:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    step = max(1e-6, float(sample_step))
    steps = max(int(math.ceil(math.hypot(dx, dy) / step)), 1)

    for i in range(steps + 1):
        t = i / steps
        px = start[0] + dx * t
        py = start[1] + dy * t
        if not point_is_safe((px, py), planning_grid):
            return False

    return True


def shortcut_smooth_path(
    path: list[tuple[int, int]],
    planning_grid: list[list[int]],
) -> list[tuple[int, int]]:
    """Remove unnecessary intermediate grid points when line of sight is safe."""
    if len(path) <= 2:
        return path

    smoothed = [path[0]]
    anchor_idx = 0

    while anchor_idx < len(path) - 1:
        next_idx = len(path) - 1
        while next_idx > anchor_idx + 1:
            if segment_is_free(path[anchor_idx], path[next_idx], planning_grid):
                break
            next_idx -= 1

        smoothed.append(path[next_idx])
        anchor_idx = next_idx

    return smoothed


def expand_segment_to_grid_steps(
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    steps = max(abs(dx), abs(dy), 1)
    expanded = []

    for i in range(1, steps + 1):
        t = i / steps
        expanded.append((int(round(start[0] + dx * t)), int(round(start[1] + dy * t))))

    return expanded


def resample_smoothed_path(smoothed_path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(smoothed_path) <= 1:
        return smoothed_path

    resampled = [smoothed_path[0]]
    for start, end in zip(smoothed_path[:-1], smoothed_path[1:]):
        for point in expand_segment_to_grid_steps(start, end):
            if point != resampled[-1]:
                resampled.append(point)

    return resampled


def line_of_sight_smooth_float_path(
    path: list[tuple[float, float]] | list[tuple[int, int]],
    planning: PlanningInput,
    config: HGridExplorerConfig,
) -> list[tuple[float, float]]:
    if len(path) <= 2:
        return [(float(x), float(y)) for x, y in path]

    points = [(float(x), float(y)) for x, y in path]
    smoothed = [points[0]]
    anchor_idx = 0

    while anchor_idx < len(points) - 1:
        next_idx = len(points) - 1
        while next_idx > anchor_idx + 1:
            if segment_is_free(
                points[anchor_idx],
                points[next_idx],
                planning,
                config.safety_sample_step,
            ):
                break
            next_idx -= 1
        smoothed.append(points[next_idx])
        anchor_idx = next_idx

    return deduplicate_nearby_points(smoothed)


def clamp_velocity(
    velocity: tuple[float, float],
    max_speed: float,
) -> tuple[float, float]:
    speed = math.hypot(velocity[0], velocity[1])
    if speed <= max_speed + 1e-9:
        return velocity
    scale = max_speed / max(speed, 1e-9)
    return (velocity[0] * scale, velocity[1] * scale)


def quantize_value(value: float, resolution: float) -> int:
    return int(round(value / max(1e-6, resolution)))


def kinodynamic_acceleration_set(max_acceleration: float) -> list[tuple[float, float]]:
    a = max(1e-6, float(max_acceleration))
    diag = a / math.sqrt(2.0)
    return [
        (0.0, 0.0),
        (a, 0.0),
        (-a, 0.0),
        (0.0, a),
        (0.0, -a),
        (diag, diag),
        (diag, -diag),
        (-diag, diag),
        (-diag, -diag),
    ]


def kinodynamic_segment(
    start_pos: tuple[float, float],
    start_velocity: tuple[float, float],
    acceleration: tuple[float, float],
    dt: float,
    max_speed: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    raw_velocity = (
        start_velocity[0] + acceleration[0] * dt,
        start_velocity[1] + acceleration[1] * dt,
    )
    next_velocity = clamp_velocity(raw_velocity, max_speed)
    avg_velocity = (
        0.5 * (start_velocity[0] + next_velocity[0]),
        0.5 * (start_velocity[1] + next_velocity[1]),
    )
    next_pos = (
        start_pos[0] + avg_velocity[0] * dt,
        start_pos[1] + avg_velocity[1] * dt,
    )
    return next_pos, next_velocity


def segment_clearance_penalty(
    start: tuple[float, float],
    end: tuple[float, float],
    planning: PlanningInput,
    config: HGridExplorerConfig,
) -> float:
    if not isinstance(planning, PlanningContext):
        return 0.0

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    steps = max(int(math.ceil(math.hypot(dx, dy) / max(1e-6, config.safety_sample_step))), 1)
    soft_clearance = planning.min_clearance + 1.0
    penalty = 0.0
    for idx in range(steps + 1):
        t = idx / steps
        point = (start[0] + dx * t, start[1] + dy * t)
        clearance = clearance_at(planning, point)
        if clearance < soft_clearance:
            penalty += (soft_clearance - clearance) ** 2
    return penalty / (steps + 1)


def reconstruct_kinodynamic_path(
    came_from: dict[tuple[int, int, int, int], tuple[int, int, int, int] | None],
    positions: dict[tuple[int, int, int, int], tuple[float, float]],
    state: tuple[int, int, int, int],
    target: tuple[int, int],
    planning: PlanningInput,
    config: HGridExplorerConfig,
) -> list[tuple[float, float]]:
    reversed_path = [positions[state]]
    while came_from[state] is not None:
        state = came_from[state]
        reversed_path.append(positions[state])

    path = list(reversed(reversed_path))
    target_point = (float(target[0]), float(target[1]))
    if math.hypot(path[-1][0] - target_point[0], path[-1][1] - target_point[1]) > 1e-6:
        if segment_is_free(path[-1], target_point, planning, config.safety_sample_step):
            path.append(target_point)
    return deduplicate_nearby_points(path)


def kinodynamic_astar_2d(
    start_pos: tuple[float, float],
    start_velocity: tuple[float, float],
    target: tuple[int, int],
    planning: PlanningInput,
    config: HGridExplorerConfig,
) -> list[tuple[float, float]]:
    if not config.use_kinodynamic_astar:
        return []
    if not point_is_safe(start_pos, planning) or not point_is_safe((float(target[0]), float(target[1])), planning):
        return []

    max_speed = max(1e-6, float(config.max_speed))
    max_acc = max(1e-6, float(config.max_acceleration))
    dt = max(1e-6, float(config.kinodynamic_dt))
    pos_res = max(1e-6, float(config.kinodynamic_position_resolution))
    vel_res = max(1e-6, float(config.kinodynamic_velocity_resolution))
    goal = (float(target[0]), float(target[1]))

    start_velocity = clamp_velocity(start_velocity, max_speed)
    start_state = (
        quantize_value(start_pos[0], pos_res),
        quantize_value(start_pos[1], pos_res),
        quantize_value(start_velocity[0], vel_res),
        quantize_value(start_velocity[1], vel_res),
    )

    def heuristic(pos: tuple[float, float]) -> float:
        return math.hypot(pos[0] - goal[0], pos[1] - goal[1]) / max_speed

    open_heap: list[tuple[float, int, tuple[int, int, int, int]]] = []
    heapq.heappush(open_heap, (heuristic(start_pos), 0, start_state))
    came_from: dict[tuple[int, int, int, int], tuple[int, int, int, int] | None] = {start_state: None}
    positions: dict[tuple[int, int, int, int], tuple[float, float]] = {start_state: start_pos}
    velocities: dict[tuple[int, int, int, int], tuple[float, float]] = {start_state: start_velocity}
    g_score: dict[tuple[int, int, int, int], float] = {start_state: 0.0}
    visited: set[tuple[int, int, int, int]] = set()
    counter = 0

    for _ in range(max(1, config.kinodynamic_max_nodes)):
        if not open_heap:
            break

        _, _, state = heapq.heappop(open_heap)
        if state in visited:
            continue
        visited.add(state)

        pos = positions[state]
        vel = velocities[state]
        dist_to_goal = math.hypot(pos[0] - goal[0], pos[1] - goal[1])
        if dist_to_goal <= config.kinodynamic_goal_tolerance and segment_is_free(
            pos,
            goal,
            planning,
            config.safety_sample_step,
        ):
            return reconstruct_kinodynamic_path(came_from, positions, state, target, planning, config)

        for accel in kinodynamic_acceleration_set(max_acc):
            next_pos, next_vel = kinodynamic_segment(pos, vel, accel, dt, max_speed)
            if not segment_is_free(pos, next_pos, planning, config.safety_sample_step):
                continue

            next_state = (
                quantize_value(next_pos[0], pos_res),
                quantize_value(next_pos[1], pos_res),
                quantize_value(next_vel[0], vel_res),
                quantize_value(next_vel[1], vel_res),
            )
            if next_state in visited:
                continue

            accel_cost = 0.04 * (math.hypot(accel[0], accel[1]) / max_acc) ** 2
            clearance_cost = 0.35 * segment_clearance_penalty(pos, next_pos, planning, config)
            tentative_g = g_score[state] + dt + accel_cost + clearance_cost
            if tentative_g >= g_score.get(next_state, float("inf")):
                continue

            came_from[next_state] = state
            positions[next_state] = next_pos
            velocities[next_state] = next_vel
            g_score[next_state] = tentative_g
            counter += 1
            heapq.heappush(open_heap, (tentative_g + heuristic(next_pos), counter, next_state))

    return []


def path_respects_dynamics(
    path: list[tuple[float, float]] | list[tuple[int, int]],
    start_velocity: tuple[float, float] | tuple[int, int],
    max_speed: float,
    max_acceleration: float,
) -> bool:
    if len(path) <= 1:
        return True

    prev_velocity = start_velocity
    for current, nxt in zip(path[:-1], path[1:]):
        velocity = (nxt[0] - current[0], nxt[1] - current[1])
        acceleration = (velocity[0] - prev_velocity[0], velocity[1] - prev_velocity[1])

        if math.hypot(velocity[0], velocity[1]) > max_speed + 1e-6:
            return False
        if math.hypot(acceleration[0], acceleration[1]) > max_acceleration + 1e-6:
            return False

        prev_velocity = velocity

    return True


def deduplicate_nearby_points(
    path: list[tuple[float, float]],
    min_distance: float = 1e-4,
) -> list[tuple[float, float]]:
    if not path:
        return []

    deduped = [path[0]]
    for point in path[1:]:
        if math.hypot(point[0] - deduped[-1][0], point[1] - deduped[-1][1]) >= min_distance:
            deduped.append(point)
    return deduped


def cubic_bspline_curve(
    control_points: list[tuple[float, float]],
    samples_per_segment: int,
) -> list[tuple[float, float]]:
    if len(control_points) < 4:
        return []

    samples_per_segment = max(2, samples_per_segment)
    controls = [control_points[0]] * 3 + control_points + [control_points[-1]] * 3
    curve = []

    for i in range(len(controls) - 3):
        p0 = controls[i]
        p1 = controls[i + 1]
        p2 = controls[i + 2]
        p3 = controls[i + 3]

        for sample_idx in range(samples_per_segment):
            t = sample_idx / samples_per_segment
            t2 = t * t
            t3 = t2 * t

            b0 = (1 - 3 * t + 3 * t2 - t3) / 6.0
            b1 = (4 - 6 * t2 + 3 * t3) / 6.0
            b2 = (1 + 3 * t + 3 * t2 - 3 * t3) / 6.0
            b3 = t3 / 6.0

            x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
            y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
            curve.append((x, y))

    curve.append(control_points[-1])
    return deduplicate_nearby_points(curve)


def trajectory_is_collision_free(
    path: list[tuple[float, float]],
    planning_grid: PlanningInput,
    sample_step: float = 0.25,
) -> bool:
    if not path:
        return False

    for point in path:
        if not point_is_safe(point, planning_grid):
            return False

    for start, end in zip(path[:-1], path[1:]):
        if not segment_is_free(start, end, planning_grid, sample_step):
            return False

    return True


def trim_initial_close_points(
    path: list[tuple[float, float]],
    current_pos: tuple[float, float],
    min_step: float,
) -> list[tuple[float, float]]:
    if len(path) <= 2:
        return path

    trimmed = [current_pos]
    keep_idx = 1
    while keep_idx < len(path) - 1:
        if math.hypot(path[keep_idx][0] - current_pos[0], path[keep_idx][1] - current_pos[1]) >= min_step:
            break
        keep_idx += 1

    trimmed.extend(path[keep_idx:])
    return deduplicate_nearby_points(trimmed)


def trajectory_smoothness_cost(path: list[tuple[float, float]]) -> float:
    if len(path) < 3:
        return 0.0

    cost = 0.0
    count = 0
    for p0, p1, p2 in zip(path[:-2], path[1:-1], path[2:]):
        ax = p2[0] - 2.0 * p1[0] + p0[0]
        ay = p2[1] - 2.0 * p1[1] + p0[1]
        cost += ax * ax + ay * ay
        count += 1
    return cost / max(1, count)


def max_turn_angle_deg(path: list[tuple[float, float]], min_segment: float = 1e-3) -> float:
    max_angle = 0.0
    if len(path) < 3:
        return max_angle

    for p0, p1, p2 in zip(path[:-2], path[1:-1], path[2:]):
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 < min_segment or n2 < min_segment:
            continue
        cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        max_angle = max(max_angle, math.degrees(math.acos(cos_angle)))
    return max_angle


def reduce_control_points(
    path: list[tuple[float, float]],
    max_points: int,
) -> list[tuple[float, float]]:
    if len(path) <= max_points:
        return path

    indices = np.linspace(0, len(path) - 1, max(4, max_points))
    reduced = [path[int(round(idx))] for idx in indices]
    deduped = deduplicate_nearby_points(reduced, min_distance=1e-3)
    if deduped[0] != path[0]:
        deduped.insert(0, path[0])
    if deduped[-1] != path[-1]:
        deduped.append(path[-1])
    return deduped


def _path_feasibility_violation(
    path: list[tuple[float, float]],
    start_velocity: tuple[float, float],
    max_speed: float,
    max_acceleration: float,
) -> float:
    if len(path) <= 1:
        return 0.0

    prev_velocity = start_velocity
    cost = 0.0
    for current, nxt in zip(path[:-1], path[1:]):
        velocity = (nxt[0] - current[0], nxt[1] - current[1])
        speed = math.hypot(velocity[0], velocity[1])
        if speed > max_speed:
            cost += (speed - max_speed) ** 2

        acceleration = (velocity[0] - prev_velocity[0], velocity[1] - prev_velocity[1])
        accel_norm = math.hypot(acceleration[0], acceleration[1])
        if accel_norm > max_acceleration:
            cost += (accel_norm - max_acceleration) ** 2
        prev_velocity = velocity

    return cost


def _swarm_distance_cost(
    path: list[tuple[float, float]],
    uav: MultiUAV | None,
    neighbors: list[MultiUAV] | None,
    config: HGridExplorerConfig,
) -> float:
    if not path or uav is None or not neighbors:
        return 0.0

    cost = 0.0
    for sample_idx, point in enumerate(path):
        step_idx = min(config.future_path_horizon - 1, sample_idx)
        for neighbor in neighbors:
            if neighbor.id == uav.id:
                continue
            if uav.distance_to(neighbor) > config.comm_range:
                continue
            other = future_position(neighbor, step_idx)
            safe_dist = float(uav.safe_radius + neighbor.safe_radius + config.conflict_margin)
            dist = math.hypot(point[0] - other[0], point[1] - other[1])
            if dist < safe_dist:
                cost += (safe_dist - dist) ** 2
    return cost / max(1, len(path))


def optimize_bspline_control_points(
    control_points: list[tuple[float, float]],
    start_velocity: tuple[float, float],
    planning: PlanningInput,
    config: HGridExplorerConfig,
    uav: MultiUAV | None = None,
    neighbors: list[MultiUAV] | None = None,
) -> list[tuple[float, float]]:
    if minimize is None or len(control_points) <= 3 or config.bspline_opt_iterations <= 0:
        return control_points

    guide = np.asarray(control_points, dtype=float)
    point_count = len(control_points)
    if point_count <= 2:
        return control_points

    bounds = []
    if isinstance(planning, PlanningContext):
        width, height = planning.width, planning.height
    else:
        width = len(planning[0])
        height = len(planning)

    for _ in range(point_count - 2):
        bounds.append((0.0, float(width - 1)))
        bounds.append((0.0, float(height - 1)))

    x0 = guide[1:-1].reshape(-1)
    if x0.size == 0:
        return control_points

    def unpack(values: np.ndarray) -> np.ndarray:
        points = guide.copy()
        points[1:-1] = values.reshape((-1, 2))
        return points

    def objective(values: np.ndarray) -> float:
        points = unpack(values)
        curve = cubic_bspline_curve(
            [(float(x), float(y)) for x, y in points],
            config.bspline_samples_per_segment,
        )
        if not curve:
            return 1e9

        smoothness = 0.0
        for p0, p1, p2 in zip(points[:-2], points[1:-1], points[2:]):
            second_diff = p2 - 2.0 * p1 + p0
            smoothness += float(np.dot(second_diff, second_diff))
        smoothness /= max(1, point_count - 2)

        guide_cost = float(np.mean(np.sum((points - guide) ** 2, axis=1)))

        distance_cost = 0.0
        blocked_cost = 0.0
        for point in curve:
            if not point_is_safe(point, planning):
                blocked_cost += 1.0
            if isinstance(planning, PlanningContext):
                dist = clearance_at(planning, point)
                if math.isfinite(dist) and dist < planning.min_clearance:
                    distance_cost += (planning.min_clearance - dist) ** 2
        distance_cost /= max(1, len(curve))
        blocked_cost /= max(1, len(curve))

        feasibility = _path_feasibility_violation(
            curve,
            start_velocity,
            config.max_speed,
            config.max_acceleration,
        )
        swarm = _swarm_distance_cost(curve, uav, neighbors, config)

        return (
            config.bspline_smoothness_weight * smoothness
            + config.bspline_distance_weight * distance_cost
            + 1000.0 * blocked_cost
            + config.bspline_feasibility_weight * feasibility
            + config.bspline_guide_weight * guide_cost
            + config.bspline_swarm_weight * swarm
        )

    try:
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(config.bspline_opt_iterations), "ftol": 1e-4},
        )
    except Exception:
        return control_points

    if not result.success and not np.isfinite(result.fun):
        return control_points

    optimized = unpack(result.x)
    return [(float(x), float(y)) for x, y in optimized]


def try_make_bspline_trajectory(
    path: list[tuple[float, float]] | list[tuple[int, int]],
    current_pos: tuple[float, float],
    target: tuple[int, int],
    start_velocity: tuple[float, float],
    planning_grid: PlanningInput,
    config: HGridExplorerConfig,
    uav: MultiUAV | None = None,
    neighbors: list[MultiUAV] | None = None,
) -> list[tuple[float, float]]:
    if not config.use_bspline_smoothing or len(path) < 4:
        return []

    control_points = [(float(x), float(y)) for x, y in path]
    control_points[0] = current_pos
    control_points[-1] = (float(target[0]), float(target[1]))
    control_points = reduce_control_points(control_points, config.bspline_max_control_points)
    control_points = optimize_bspline_control_points(
        control_points,
        start_velocity,
        planning_grid,
        config,
        uav=uav,
        neighbors=neighbors,
    )

    curve = cubic_bspline_curve(control_points, config.bspline_samples_per_segment)
    if not curve:
        return []

    curve[0] = current_pos
    if math.hypot(curve[-1][0] - target[0], curve[-1][1] - target[1]) > 1e-6:
        curve.append((float(target[0]), float(target[1])))
    else:
        curve[-1] = (float(target[0]), float(target[1]))

    curve = deduplicate_nearby_points(curve)
    curve = trim_initial_close_points(curve, current_pos, config.bspline_min_step)
    if not trajectory_is_collision_free(curve, planning_grid, config.safety_sample_step):
        return []
    if not path_respects_dynamics(curve, start_velocity, config.max_speed, config.max_acceleration):
        return []
    if trajectory_smoothness_cost(curve) > config.bspline_max_smoothness_cost:
        return []
    if max_turn_angle_deg(curve) > config.bspline_max_turn_angle_deg:
        return []

    return curve


def plan_local_trajectory(
    uav: MultiUAV,
    target: tuple[int, int],
    planning_grid: PlanningInput,
    config: HGridExplorerConfig,
    neighbors: list[MultiUAV] | None = None,
) -> tuple[list[tuple[float, float]] | list[tuple[int, int]], str]:
    current_pos = (float(uav.pos[0]), float(uav.pos[1]))
    start = discretize_pos(uav.pos)
    target = discretize_pos(target)
    start_velocity = get_uav_velocity(uav)

    if start == target:
        return [], "none"

    path = kinodynamic_astar_2d(
        start_pos=current_pos,
        start_velocity=start_velocity,
        target=target,
        planning=planning_grid,
        config=config,
    )
    planner_prefix = "kinodynamic"

    if not path:
        path = [(float(x), float(y)) for x, y in astar(start, target, planning_grid_cells(planning_grid))]
        planner_prefix = "astar_fallback"

    if not path:
        return [], "none"

    if path and math.hypot(path[0][0] - current_pos[0], path[0][1] - current_pos[1]) > 1e-6:
        path.insert(0, current_pos)
    fallback_path_base = line_of_sight_smooth_float_path(path, planning_grid, config)
    bspline_source_path = path if len(path) >= 4 else fallback_path_base

    bspline_path = try_make_bspline_trajectory(
        path=bspline_source_path,
        current_pos=current_pos,
        target=target,
        start_velocity=start_velocity,
        planning_grid=planning_grid,
        config=config,
        uav=uav,
        neighbors=neighbors,
    )
    if bspline_path:
        limited_bspline = enforce_step_distance_limit(bspline_path, current_pos, config.max_speed)
        if trajectory_is_collision_free(limited_bspline, planning_grid, config.safety_sample_step):
            return limited_bspline, f"{planner_prefix}_bspline"

    fallback_path = [(float(x), float(y)) for x, y in fallback_path_base]
    fallback_path[0] = current_pos
    fallback_path[-1] = (float(target[0]), float(target[1]))

    limited_fallback = enforce_step_distance_limit(fallback_path, current_pos, config.max_speed)
    if not trajectory_is_collision_free(limited_fallback, planning_grid, config.safety_sample_step):
        return [], "none"
    return limited_fallback, f"{planner_prefix}_polyline"


def rotate_toward(current: float, target: float, max_delta: float) -> float:
    delta = normalize_angle(target - current)
    delta = max(-max_delta, min(max_delta, delta))
    return normalize_angle(current + delta)


def step_uav_with_velocity(
    uav: MultiUAV,
    config: HGridExplorerConfig,
    planning_grid: PlanningInput | None = None,
) -> None:
    old_pos = (float(uav.pos[0]), float(uav.pos[1]))
    if uav.path:
        next_point = (float(uav.path[0][0]), float(uav.path[0][1]))
        distance = math.hypot(next_point[0] - old_pos[0], next_point[1] - old_pos[1])
        max_step = max(1e-6, float(config.max_speed))

        if distance > max_step:
            ratio = max_step / distance
            uav.pos = (
                old_pos[0] + ratio * (next_point[0] - old_pos[0]),
                old_pos[1] + ratio * (next_point[1] - old_pos[1]),
            )
            if planning_grid is not None and not segment_is_free(
                old_pos,
                uav.pos,
                planning_grid,
                config.safety_sample_step,
            ):
                uav.pos = old_pos
                uav.path = []
                setattr(uav, "target", None)
                setattr(uav, "velocity", (0, 0))
                return
            uav.history.append(uav.pos)
        else:
            candidate_pos = uav.path.pop(0)
            if planning_grid is not None and not segment_is_free(
                old_pos,
                candidate_pos,
                planning_grid,
                config.safety_sample_step,
            ):
                uav.pos = old_pos
                uav.path = []
                setattr(uav, "target", None)
                setattr(uav, "velocity", (0, 0))
                return
            uav.pos = candidate_pos
            uav.history.append(uav.pos)

    new_pos = (float(uav.pos[0]), float(uav.pos[1]))
    velocity = (new_pos[0] - old_pos[0], new_pos[1] - old_pos[1])
    setattr(uav, "velocity", velocity)

    yaw = float(getattr(uav, "yaw", 0.0))
    yaw_step = math.radians(config.camera_yaw_rate_deg)
    if math.hypot(velocity[0], velocity[1]) > 1e-6:
        yaw_target = math.atan2(velocity[1], velocity[0])
        yaw = rotate_toward(yaw, yaw_target, yaw_step)
    elif getattr(uav, "target_yaw", None) is not None:
        yaw = rotate_toward(yaw, float(getattr(uav, "target_yaw")), yaw_step)
    setattr(uav, "yaw", yaw)


def enforce_step_distance_limit(
    path: list[tuple[float, float]] | list[tuple[int, int]],
    start_pos: tuple[float, float],
    max_step: float,
) -> list[tuple[float, float]]:
    """Densify a path so a UAV never visually or physically jumps too far."""
    if not path:
        return []

    max_step = max(1e-6, float(max_step))
    limited: list[tuple[float, float]] = []
    previous = (float(start_pos[0]), float(start_pos[1]))

    for raw_point in path:
        point = (float(raw_point[0]), float(raw_point[1]))
        distance = math.hypot(point[0] - previous[0], point[1] - previous[1])
        if distance < 1e-9:
            continue

        segments = max(1, int(math.ceil(distance / max_step)))
        for idx in range(1, segments + 1):
            alpha = idx / segments
            interp = (
                previous[0] + alpha * (point[0] - previous[0]),
                previous[1] + alpha * (point[1] - previous[1]),
            )
            limited.append(interp)

        previous = point

    return limited


def future_position(uav: MultiUAV, step_idx: int) -> tuple[float, float]:
    if step_idx < len(uav.path):
        point = uav.path[step_idx]
        return (float(point[0]), float(point[1]))
    if uav.path:
        point = uav.path[-1]
        return (float(point[0]), float(point[1]))
    return (float(uav.pos[0]), float(uav.pos[1]))


def detect_future_conflict(
    uav_a: MultiUAV,
    uav_b: MultiUAV,
    config: HGridExplorerConfig,
) -> int | None:
    if (
        uav_a.pos[0] < config.known_strip_width + 1
        and uav_b.pos[0] < config.known_strip_width + 1
    ):
        return None

    safe_distance = uav_a.safe_radius + uav_b.safe_radius + config.conflict_margin

    for step_idx in range(config.future_path_horizon):
        ax, ay = future_position(uav_a, step_idx)
        bx, by = future_position(uav_b, step_idx)
        if math.hypot(ax - bx, ay - by) < safe_distance:
            return step_idx

        if step_idx == 0:
            continue

        prev_a = future_position(uav_a, step_idx - 1)
        prev_b = future_position(uav_b, step_idx - 1)
        if math.hypot(ax - prev_b[0], ay - prev_b[1]) < safe_distance:
            if math.hypot(bx - prev_a[0], by - prev_a[1]) < safe_distance:
                return step_idx

    return None


def copy_planning_grid(planning_grid: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in planning_grid]


def mark_dynamic_obstacle(
    planning_grid: list[list[int]],
    point: tuple[float, float],
    inflation_radius: float,
) -> None:
    cx, cy = discretize_pos(point)
    r_int = int(math.ceil(inflation_radius))

    for y in range(max(0, cy - r_int), min(len(planning_grid), cy + r_int + 1)):
        for x in range(max(0, cx - r_int), min(len(planning_grid[0]), cx + r_int + 1)):
            if math.hypot(x - point[0], y - point[1]) <= inflation_radius:
                planning_grid[y][x] = OBSTACLE


def planning_grid_with_neighbor_trajectories(
    base_grid: list[list[int]],
    uav: MultiUAV,
    neighbors: list[MultiUAV],
    target: tuple[int, int],
    config: HGridExplorerConfig,
) -> list[list[int]]:
    """
    RACER-style reciprocal collision term in grid form.

    The paper adds an interdrone collision penalty Jc,q to B-spline trajectory
    optimization using received neighbor trajectories. In this grid simulator,
    we approximate that penalty by marking received short-horizon trajectories
    as dynamic inflated obstacles, then replanning to the same target.
    """
    grid = copy_planning_grid(base_grid)
    start = discretize_pos(uav.pos)

    for neighbor in neighbors:
        if neighbor.id == uav.id:
            continue
        if uav.distance_to(neighbor) > config.comm_range:
            continue

        for step_idx in range(config.future_path_horizon):
            mark_dynamic_obstacle(
                planning_grid=grid,
                point=future_position(neighbor, step_idx),
                inflation_radius=max(config.dynamic_obstacle_inflation, uav.safe_radius + neighbor.safe_radius),
            )

    # Do not let dynamic obstacle inflation block the current state or intended
    # endpoint; otherwise reciprocal replanning can fail trivially.
    if in_planning_bounds(start, grid):
        grid[start[1]][start[0]] = FREE
    if in_planning_bounds(target, grid):
        grid[target[1]][target[0]] = FREE

    return grid


def received_trajectory_conflicts(
    uav: MultiUAV,
    neighbors: list[MultiUAV],
    config: HGridExplorerConfig,
) -> bool:
    for neighbor in neighbors:
        if neighbor.id == uav.id:
            continue
        if uav.distance_to(neighbor) > config.comm_range:
            continue
        if detect_future_conflict(uav, neighbor, config) is not None:
            return True
    return False


def resolve_reciprocal_trajectory_conflicts(
    uavs: list[MultiUAV],
    known_maps: list[KnownMap],
    config: HGridExplorerConfig,
) -> tuple[int, int, int]:
    """
    Decentralized reciprocal trajectory replanning.

    Each UAV receives neighboring trajectories. If its current trajectory is
    predicted to collide, it replans to the same target while treating received
    trajectories as dynamic obstacles. This mirrors RACER's "receive trajectory,
    detect collision, immediately generate a new trajectory to the same target"
    behavior, adapted to this 2-D Python simulator.
    """
    collision_count = 0
    replan_count = 0
    unresolved_count = 0

    for _ in range(max(1, config.reciprocal_replan_rounds)):
        changed_this_round = False

        for idx, uav in enumerate(uavs):
            if not uav.path or uav.target is None:
                continue

            neighbors = [
                other
                for other in uavs
                if other.id != uav.id and uav.distance_to(other) <= config.comm_range
            ]
            if not received_trajectory_conflicts(uav, neighbors, config):
                continue

            collision_count += 1
            target = discretize_pos(uav.target)
            base_context = planning_context_from_known_map(known_maps[idx], config)
            dynamic_grid = planning_grid_with_neighbor_trajectories(
                base_grid=base_context.grid,
                uav=uav,
                neighbors=neighbors,
                target=target,
                config=config,
            )
            dynamic_context = planning_context_from_known_map(
                known_maps[idx],
                config,
                planning_grid=dynamic_grid,
            )
            new_path, planner_mode = plan_local_trajectory(
                uav=uav,
                target=target,
                planning_grid=dynamic_context,
                config=config,
                neighbors=neighbors,
            )

            if new_path:
                uav.set_plan(target, new_path)
                setattr(uav, "last_planner_mode", f"{planner_mode}_reciprocal")
                replan_count += 1
                changed_this_round = True
            else:
                unresolved_count += 1

        if not changed_this_round:
            break

    return collision_count, replan_count, unresolved_count


def uav_is_at_start(uav: MultiUAV) -> bool:
    return discretize_pos(uav.pos) == uav.start


def decay_target_blacklist(uav: MultiUAV) -> None:
    blacklist = dict(getattr(uav, "target_blacklist", {}))
    next_blacklist = {
        target: remaining - 1
        for target, remaining in blacklist.items()
        if remaining > 1
    }
    setattr(uav, "target_blacklist", next_blacklist)


def target_is_blacklisted(uav: MultiUAV, target: tuple[int, int]) -> bool:
    blacklist = getattr(uav, "target_blacklist", {})
    tx, ty = target
    for blocked_target, remaining in blacklist.items():
        if int(remaining) <= 0:
            continue
        bx, by = blocked_target
        radius = float(getattr(uav, "target_blacklist_radius", 0.0))
        if math.hypot(tx - bx, ty - by) <= radius:
            return True
    return False


def blacklist_target(uav: MultiUAV, target: tuple[int, int] | None, config: HGridExplorerConfig) -> None:
    if target is None:
        return
    blacklist = dict(getattr(uav, "target_blacklist", {}))
    blacklist[tuple(discretize_pos(target))] = config.target_blacklist_steps
    setattr(uav, "target_blacklist", blacklist)


def target_has_visible_gain(
    uav: MultiUAV,
    known_map: KnownMap,
    target: tuple[int, int] | None,
    config: HGridExplorerConfig,
) -> bool:
    if target is None:
        return False
    target = discretize_pos(target)
    if target_is_blacklisted(uav, target):
        return False
    if not (0 <= target[0] < known_map.world.width and 0 <= target[1] < known_map.world.height):
        return False
    if known_map.grid[target[1], target[0]] != FREE:
        return False

    yaw = yaw_toward_unknown(known_map, target, uav.pos, config)
    return unknown_gain_from_viewpoint(known_map, target, yaw, config) >= config.viewpoint_min_gain


def update_progress_monitor(
    uav: MultiUAV,
    known_map: KnownMap,
    newly_known: int,
    config: HGridExplorerConfig,
) -> None:
    last_pos = getattr(uav, "last_progress_pos", uav.pos)
    moved = math.hypot(uav.pos[0] - last_pos[0], uav.pos[1] - last_pos[1])
    target_dist = None
    if uav.target is not None:
        target_dist = math.hypot(uav.pos[0] - uav.target[0], uav.pos[1] - uav.target[1])
    prev_target_dist = getattr(uav, "last_target_distance", None)

    if newly_known > 0:
        setattr(uav, "last_progress_pos", uav.pos)
        setattr(uav, "stagnant_steps", 0)
        setattr(uav, "no_info_steps", 0)
        setattr(uav, "last_target_distance", target_dist)
        return

    approaching_target = (
        moved >= config.no_progress_distance
        and target_dist is not None
        and (prev_target_dist is None or target_dist < prev_target_dist - 0.05)
    )

    no_info_steps = int(getattr(uav, "no_info_steps", 0)) + 1
    setattr(uav, "no_info_steps", no_info_steps)

    if approaching_target:
        setattr(uav, "last_progress_pos", uav.pos)
        setattr(uav, "stagnant_steps", 0)
        setattr(uav, "last_target_distance", target_dist)
        moving_no_info_limit = max(config.no_info_steps * 3, config.no_info_steps + 24)
        if no_info_steps < moving_no_info_limit:
            return

    if no_info_steps >= config.no_info_steps:
        blacklist_target(uav, uav.target, config)
        uav.set_plan(None, [])
        setattr(uav, "no_info_steps", 0)
        setattr(uav, "stagnant_steps", 0)
        setattr(uav, "last_progress_pos", uav.pos)
        return

    if moved >= config.no_progress_distance:
        setattr(uav, "last_progress_pos", uav.pos)
        setattr(uav, "stagnant_steps", 0)
        return

    stagnant_steps = int(getattr(uav, "stagnant_steps", 0)) + 1
    setattr(uav, "stagnant_steps", stagnant_steps)
    if stagnant_steps >= config.no_progress_steps:
        blacklist_target(uav, uav.target, config)
        uav.set_plan(None, [])
        setattr(uav, "stagnant_steps", 0)
        setattr(uav, "last_progress_pos", uav.pos)


def select_cp_guided_viewpoint(
    uav: MultiUAV,
    known_map: KnownMap,
    frontier_infos: list[FrontierInfo],
    cp_path: list[tuple[int, int]],
    config: HGridExplorerConfig,
) -> tuple[tuple[int, int] | None, list[tuple[int, int]], int]:
    planning_grid = planning_context_from_known_map(known_map, config)
    cursor = update_cp_cursor(uav, cp_path, config.cp_reach_radius)

    best_target = None
    best_path = []
    best_cluster_size = 0
    best_score = float("inf")
    best_cp_idx = cursor
    best_yaw = float(getattr(uav, "yaw", 0.0))

    ranked_infos = []
    for info in frontier_infos:
        cp_score, cp_idx = cp_score_for_cluster(info.cells, cp_path, cursor, config)
        dist = math.hypot(uav.pos[0] - info.centroid[0], uav.pos[1] - info.centroid[1])
        ranked_infos.append((cp_score + 0.1 * dist, cp_idx, info))

    ranked_infos.sort(key=lambda item: (item[0], item[1], item[2].cluster_id))
    local_infos = [info for _, _, info in ranked_infos[: max(1, config.local_cluster_window)]]
    setattr(uav, "local_frontier_sequence", [info.cluster_id for info in local_infos])

    for sequence_idx, info in enumerate(local_infos):
        cp_score, cp_idx = cp_score_for_cluster(info.cells, cp_path, cursor, config)
        for vx, vy, yaw, gain in info.viewpoints:
            viewpoint = (vx, vy)
            if target_is_blacklisted(uav, viewpoint):
                continue
            if math.hypot(vx - uav.pos[0], vy - uav.pos[1]) < 0.75:
                continue
            if config.use_kinodynamic_for_viewpoint_scoring:
                rough_path = kinodynamic_astar_2d(
                    start_pos=(float(uav.pos[0]), float(uav.pos[1])),
                    start_velocity=get_uav_velocity(uav),
                    target=viewpoint,
                    planning=planning_grid,
                    config=config,
                )
            else:
                rough_path = []
            if not rough_path:
                rough_path = astar(discretize_pos(uav.pos), viewpoint, planning_grid.grid)
            if not rough_path or len(rough_path) <= 1:
                continue

            score = (
                len(rough_path)
                - 0.85 * gain
                - 0.20 * len(info.cells)
                + cp_score
                + 0.35 * sequence_idx
            )
            if score < best_score:
                best_score = score
                best_target = viewpoint
                best_path = rough_path
                best_cluster_size = len(info.cells)
                best_cp_idx = cp_idx
                best_yaw = yaw

    if best_target is not None:
        local_path, planner_mode = plan_local_trajectory(
            uav,
            best_target,
            planning_grid,
            config,
        )
        if local_path:
            best_path = local_path
            setattr(uav, "last_planner_mode", planner_mode)
            setattr(uav, "target_yaw", best_yaw)
        else:
            setattr(uav, "last_planner_mode", "none")
            return None, [], 0

        setattr(uav, "cp_cursor", best_cp_idx)

    return best_target, best_path, best_cluster_size


def plan_uav_with_hgrid(
    uav: MultiUAV,
    known_map: KnownMap,
    hgrid: HGrid,
    config: HGridExplorerConfig,
    step_idx: int,
) -> tuple[bool, str]:
    """
    Plan one UAV.

    Priority:
        1. Frontier clusters inside this UAV's hgrid blocks.
        2. If none are reachable, any reachable frontier as a helper fallback.
    """
    decay_target_blacklist(uav)
    if uav.path and uav.target is not None:
        target_dist = math.hypot(uav.pos[0] - uav.target[0], uav.pos[1] - uav.target[1])
        if (
            target_dist > config.target_commit_radius
            and target_has_visible_gain(uav, known_map, uav.target, config)
        ):
            return True, "keep"

    if (
        uav.path
        and uav.target is not None
        and not target_is_blacklisted(uav, discretize_pos(uav.target))
        and config.local_replan_interval > 1
        and step_idx % config.local_replan_interval != 0
    ):
        return True, "keep"

    fis = getattr(uav, "fis", None)
    if fis is None:
        fis = FrontierInformationStructure()
        setattr(uav, "fis", fis)

    frontier_infos = fis.update(known_map, uav.pos, config)
    owned_blocks = hgrid.blocks_for_uav(uav.id)
    owned_infos = [
        info
        for info in frontier_infos
        if hgrid.cluster_inside_owned_blocks(info.cells, owned_blocks)
    ]

    cp_path = generate_coverage_path_for_blocks(owned_blocks, config.cp_step, uav.pos)
    setattr(uav, "cp_path", cp_path)

    target, path, _ = select_cp_guided_viewpoint(uav, known_map, owned_infos, cp_path, config)
    if path:
        uav.assigned_cluster_count = len(owned_infos)
        uav.set_plan(target, path)
        return True, "cp_owned"

    target, path, _ = select_cp_guided_viewpoint(uav, known_map, frontier_infos, cp_path, config)
    uav.assigned_cluster_count = len(owned_infos)
    uav.set_plan(target, path)
    return bool(path), "cp_fallback" if path else "none"


def render_hgrid_state(
    ax,
    world: GridWorld,
    known_maps: list[KnownMap],
    uavs: list[MultiUAV],
    hgrid: HGrid,
    step_idx: int,
    known_ratio: float,
    phase: str,
) -> None:
    ax.clear()
    cmap = plt.get_cmap("tab20")
    merged_grid = union_known_grid(known_maps)

    display = np.zeros((world.height, world.width, 3), dtype=float)
    display[:, :] = [0.90, 0.90, 0.90]
    display[merged_grid == FREE] = [1.00, 1.00, 1.00]
    display[merged_grid == OBSTACLE] = [1.00, 1.00, 1.00]
    ax.imshow(display, origin="lower", extent=[-0.5, world.width - 0.5, -0.5, world.height - 0.5])

    obstacle_patches = [
        patches.Rectangle((rx, ry), rw, rh)
        for rx, ry, rw, rh in world.rectangles
    ]
    obstacle_patches.extend(
        patches.Circle((cx, cy), r)
        for cx, cy, r in world.circles
    )
    if obstacle_patches:
        ax.add_collection(
            PatchCollection(
                obstacle_patches,
                facecolor="gray",
                edgecolor="black",
                linewidth=1,
                alpha=0.5,
            )
        )

    ax.axvspan(0, world.config.known_strip_width, color=cmap(0), alpha=0.08)

    block_patches = []
    block_edges = []
    block_linewidths = []
    block_alphas = []
    for block in hgrid.blocks:
        if block.owner_id is None:
            edge_color = "black"
        else:
            edge_color = cmap((block.owner_id - 1) % 20)

        block_patches.append(patches.Rectangle((block.x_min, block.y_min), block.width(), block.height()))
        block_edges.append(edge_color)
        block_linewidths.append(max(0.6, 1.5 - 0.25 * block.level))
        block_alphas.append(max(0.30, 0.60 - 0.08 * block.level))

    if block_patches:
        block_collection = PatchCollection(
            block_patches,
            facecolor="none",
            edgecolor=block_edges,
            linewidth=block_linewidths,
            alpha=min(block_alphas) if block_alphas else 0.4,
        )
        ax.add_collection(block_collection)

    global_view = KnownMap(world)
    global_view.grid[:, :] = merged_grid
    frontier_points = sorted(detect_explorable_frontier_cells(global_view))
    if frontier_points:
        max_frontiers = int(max(0, world.config.render_max_frontier_points))
        if max_frontiers and len(frontier_points) > max_frontiers:
            sample_idx = np.linspace(0, len(frontier_points) - 1, max_frontiers, dtype=int)
            frontier_points = [frontier_points[idx] for idx in sample_idx]
        fx = [p[0] for p in frontier_points]
        fy = [p[1] for p in frontier_points]
        ax.plot(fx, fy, ".", color="green", markersize=3, alpha=0.45)

    for uav in uavs:
        color = cmap((uav.id - 1) % 20)

        if world.config.show_cp_path:
            cp_path = getattr(uav, "cp_path", [])
            if cp_path:
                cp_x = [point[0] for point in cp_path]
                cp_y = [point[1] for point in cp_path]
                ax.plot(cp_x, cp_y, ":", color=color, linewidth=0.8, alpha=0.25)

        if uav.path:
            px = [uav.pos[0]] + [p[0] for p in uav.path]
            py = [uav.pos[1]] + [p[1] for p in uav.path]
            ax.plot(px, py, "--", color=color, alpha=0.5)

        if uav.history:
            history_tail = int(max(0, world.config.render_history_tail))
            history = uav.history[-history_tail:] if history_tail else uav.history
            hx = [p[0] for p in history]
            hy = [p[1] for p in history]
            ax.plot(hx, hy, color=color, linewidth=1.5, alpha=0.8)

        if world.config.use_camera_fov:
            yaw_deg = math.degrees(float(getattr(uav, "yaw", 0.0)))
            half_fov = world.config.camera_fov_deg / 2.0
            ax.add_patch(
                patches.Wedge(
                    (uav.pos[0], uav.pos[1]),
                    uav.coverage_radius,
                    yaw_deg - half_fov,
                    yaw_deg + half_fov,
                    color=color,
                    alpha=0.10,
                )
            )
        else:
            ax.add_patch(
                plt.Circle(
                    (uav.pos[0], uav.pos[1]),
                    uav.coverage_radius,
                    color=color,
                    alpha=0.08,
                )
            )
        ax.add_patch(
            plt.Circle(
                (uav.pos[0], uav.pos[1]),
                uav.safe_radius,
                color=color,
                alpha=0.15,
            )
        )
        ax.plot(uav.pos[0], uav.pos[1], "o", color=color, markersize=5)
        ax.text(uav.pos[0] + 1.0, uav.pos[1] + 1.0, f"UAV{uav.id}", fontsize=9)

    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title(
        f"Multi-UAV HGrid Exploration - {phase} | "
        f"step={step_idx} known={known_ratio * 100:.1f}% "
        f"uavs={len(uavs)} active_hgrid={len(hgrid.blocks)} levels={hgrid.level_sizes}"
    )


def run_simulation(
    config: HGridExplorerConfig,
    show: bool = True,
) -> dict[str, float | int | bool | None]:
    world = GridWorld(config)
    starts = make_column_starts(config)
    uavs = [
        MultiUAV(
            uav_id=i + 1,
            start=start,
            safe_radius=config.safe_radius,
            coverage_radius=config.coverage_radius,
            max_speed_cells=1,
        )
        for i, start in enumerate(starts)
    ]
    for uav in uavs:
        setattr(uav, "velocity", (0, 0))
        setattr(uav, "yaw", 0.0)
        setattr(uav, "target_yaw", 0.0)
        setattr(uav, "hgrid_last_attempt", -10**9)
        setattr(uav, "hgrid_last_success", {})
        setattr(uav, "target_blacklist", {})
        setattr(uav, "target_blacklist_radius", config.target_blacklist_radius)
        setattr(uav, "last_progress_pos", uav.pos)
        setattr(uav, "last_target_distance", None)
        setattr(uav, "stagnant_steps", 0)
        setattr(uav, "no_info_steps", 0)

    hgrid = HGrid(config)
    hgrid.assign_initial_owners(uavs)

    known_maps = [KnownMap(world) for _ in uavs]
    for known_map in known_maps:
        known_map.grid[:, : config.known_strip_width] = world.raw_obstacle_map[:, : config.known_strip_width]
    for uav, known_map in zip(uavs, known_maps):
        update_known_map_with_camera(known_map, uav, config)
    hgrid.update_active_cells(union_known_grid(known_maps), world.raw_obstacle_map)

    fig = ax = None
    if show:
        plt.ion()
        fig, ax = plt.subplots(figsize=(9, 8))

    phase = "explore"
    final_step = 0
    final_known_ratio = explorable_known_ratio(known_maps, world.raw_obstacle_map, config.known_strip_width)
    best_known_ratio = final_known_ratio
    last_global_improvement_step = 0
    exploration_finished_step = None
    returned_home = False
    owned_plan_count = 0
    fallback_plan_count = 0
    hgrid_reassign_count = 0
    astar_polyline_count = 0
    bspline_plan_count = 0
    reciprocal_collision_count = 0
    reciprocal_replan_count = 0
    unresolved_collision_count = 0
    hgrid_split_count = 0
    hgrid_removed_count = 0
    pairwise_success_count = 0

    for step_idx in range(config.max_steps):
        final_step = step_idx
        should_render = (
            show
            and ax is not None
            and max(1, int(config.render_interval)) > 0
            and step_idx % max(1, int(config.render_interval)) == 0
        )

        components = communication_components(uavs, config.comm_range)
        for component in components:
            merge_known_maps_for_component(component, known_maps)

        known_ratio = explorable_known_ratio(known_maps, world.raw_obstacle_map, config.known_strip_width)
        final_known_ratio = known_ratio
        if known_ratio > best_known_ratio + 1e-4:
            best_known_ratio = known_ratio
            last_global_improvement_step = step_idx

        if phase == "explore" and known_ratio >= config.stop_known_ratio:
            phase = "return_home"
            exploration_finished_step = step_idx
        elif (
            phase == "explore"
            and known_ratio >= config.stall_stop_min_known_ratio
            and config.global_stall_steps > 0
            and step_idx - last_global_improvement_step >= config.global_stall_steps
        ):
            phase = "return_home"
            exploration_finished_step = step_idx

        if phase == "explore":
            if (
                config.hgrid_update_interval > 0
                and step_idx % config.hgrid_update_interval == 0
            ):
                split_count, removed_count = hgrid.update_active_cells(union_known_grid(known_maps), world.raw_obstacle_map)
                hgrid_split_count += split_count
                hgrid_removed_count += removed_count

            if (
                config.pairwise_reassign_interval > 0
                and step_idx % config.pairwise_reassign_interval == 0
            ):
                changed, successes = pairwise_request_response_hgrid_blocks(
                    uavs=uavs,
                    known_maps=known_maps,
                    hgrid=hgrid,
                    step_idx=step_idx,
                    config=config,
                )
                hgrid_reassign_count += changed
                pairwise_success_count += successes

            planned_count = 0
            for uav, known_map in zip(uavs, known_maps):
                planned, plan_mode = plan_uav_with_hgrid(uav, known_map, hgrid, config, step_idx)
                if planned:
                    planned_count += 1
                    if plan_mode != "keep":
                        last_planner_mode = getattr(uav, "last_planner_mode", "none")
                        if "bspline" in last_planner_mode:
                            bspline_plan_count += 1
                        elif "polyline" in last_planner_mode:
                            astar_polyline_count += 1

                        if plan_mode == "cp_owned":
                            owned_plan_count += 1
                        elif plan_mode == "cp_fallback":
                            fallback_plan_count += 1

            if planned_count == 0:
                global_view = KnownMap(world)
                global_view.grid[:, :] = union_known_grid(known_maps)
                frontier_cells = detect_explorable_frontier_cells(global_view)
                removed_count = hgrid.prune_blocks_without_frontiers(frontier_cells)
                hgrid_removed_count += removed_count
                if removed_count > 0 and hgrid.blocks:
                    continue
                phase = "return_home"
                exploration_finished_step = step_idx

        if phase == "return_home":
            for uav, known_map in zip(uavs, known_maps):
                if uav_is_at_start(uav):
                    uav.set_plan(uav.start, [])
                    setattr(uav, "velocity", (0, 0))
                    continue
                if not uav.path or uav.target != uav.start:
                    return_path, return_planner_mode = plan_local_trajectory(
                        uav=uav,
                        target=uav.start,
                        planning_grid=planning_context_from_known_map(known_map, config),
                        config=config,
                    )
                    if "bspline" in return_planner_mode:
                        bspline_plan_count += 1
                    elif "polyline" in return_planner_mode:
                        astar_polyline_count += 1
                    uav.set_plan(uav.start, return_path)

            if all(uav_is_at_start(uav) for uav in uavs):
                returned_home = True
                if show and ax is not None:
                    render_hgrid_state(ax, world, known_maps, uavs, hgrid, step_idx, known_ratio, phase)
                    plt.pause(config.render_pause)
                break

        step_collisions, step_replans, step_unresolved = resolve_reciprocal_trajectory_conflicts(
            uavs,
            known_maps,
            config,
        )
        reciprocal_collision_count += step_collisions
        reciprocal_replan_count += step_replans
        unresolved_collision_count += step_unresolved

        movable_uavs = [uav for uav in uavs if uav.path]
        if not movable_uavs:
            if show and ax is not None:
                render_hgrid_state(ax, world, known_maps, uavs, hgrid, step_idx, known_ratio, phase)
                plt.pause(config.render_pause)
            break

        known_map_by_id = {uav.id: known_map for uav, known_map in zip(uavs, known_maps)}
        for uav in movable_uavs:
            step_uav_with_velocity(
                uav,
                config,
                planning_context_from_known_map(known_map_by_id[uav.id], config),
            )

        for uav, known_map in zip(uavs, known_maps):
            newly_known = update_known_map_with_camera(known_map, uav, config)
            update_progress_monitor(uav, known_map, newly_known, config)

        if should_render:
            render_hgrid_state(ax, world, known_maps, uavs, hgrid, step_idx, known_ratio, phase)
            plt.pause(config.render_pause)

    if show:
        plt.ioff()
        plt.show()

    return {
        "steps": final_step,
        "known_ratio": final_known_ratio,
        "num_uavs": len(uavs),
        "hgrid_blocks": len(hgrid.blocks),
        "hgrid_split_count": hgrid_split_count,
        "hgrid_removed_count": hgrid_removed_count,
        "path_length_total": sum(len(uav.history) for uav in uavs),
        "exploration_finished_step": exploration_finished_step,
        "returned_home": returned_home,
        "owned_plan_count": owned_plan_count,
        "fallback_plan_count": fallback_plan_count,
        "hgrid_reassign_count": hgrid_reassign_count,
        "pairwise_success_count": pairwise_success_count,
        "astar_polyline_count": astar_polyline_count,
        "bspline_plan_count": bspline_plan_count,
        "reciprocal_collision_count": reciprocal_collision_count,
        "reciprocal_replan_count": reciprocal_replan_count,
        "unresolved_collision_count": unresolved_collision_count,
        "final_positions": [tuple(round(float(v), 2) for v in uav.pos) for uav in uavs],
        "home_flags": [uav_is_at_start(uav) for uav in uavs],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-UAV CP-guided hgrid exploration demo.")
    parser.add_argument("--no-show", action="store_true", help="run without opening matplotlib window")
    parser.add_argument("--seed", type=int, default=HGridExplorerConfig.random_seed)
    parser.add_argument("--obstacle-count", type=int, default=HGridExplorerConfig.obstacle_count)
    parser.add_argument("--num-uavs", type=int, default=HGridExplorerConfig.num_uavs)
    parser.add_argument("--comm-range", type=float, default=HGridExplorerConfig.comm_range)
    parser.add_argument("--max-speed", type=float, default=HGridExplorerConfig.max_speed)
    parser.add_argument("--max-acceleration", type=float, default=HGridExplorerConfig.max_acceleration)
    parser.add_argument("--future-path-horizon", type=int, default=HGridExplorerConfig.future_path_horizon)
    parser.add_argument("--conflict-margin", type=float, default=HGridExplorerConfig.conflict_margin)
    parser.add_argument(
        "--reciprocal-replan-rounds",
        type=int,
        default=HGridExplorerConfig.reciprocal_replan_rounds,
    )
    parser.add_argument(
        "--dynamic-obstacle-inflation",
        type=float,
        default=HGridExplorerConfig.dynamic_obstacle_inflation,
    )
    parser.add_argument("--hgrid-block-size", type=int, default=HGridExplorerConfig.hgrid_block_size)
    parser.add_argument(
        "--hgrid-level-sizes",
        type=str,
        default=",".join(str(v) for v in HGridExplorerConfig.hgrid_level_sizes),
        help="comma-separated active hgrid cell sizes from coarse to fine",
    )
    parser.add_argument("--hgrid-split-known-ratio", type=float, default=HGridExplorerConfig.hgrid_split_known_ratio)
    parser.add_argument("--hgrid-min-unknown-cells", type=int, default=HGridExplorerConfig.hgrid_min_unknown_cells)
    parser.add_argument("--hgrid-update-interval", type=int, default=HGridExplorerConfig.hgrid_update_interval)
    parser.add_argument(
        "--pairwise-reassign-interval",
        type=int,
        default=HGridExplorerConfig.pairwise_reassign_interval,
    )
    parser.add_argument("--hgrid-balance-weight", type=float, default=HGridExplorerConfig.hgrid_balance_weight)
    parser.add_argument("--pairwise-request-cooldown", type=int, default=HGridExplorerConfig.pairwise_request_cooldown)
    parser.add_argument("--pairwise-success-cooldown", type=int, default=HGridExplorerConfig.pairwise_success_cooldown)
    parser.add_argument("--cvrp-max-exact-blocks", type=int, default=HGridExplorerConfig.cvrp_max_exact_blocks)
    parser.add_argument("--cvrp-route-weight", type=float, default=HGridExplorerConfig.cvrp_route_weight)
    parser.add_argument("--cvrp-work-weight", type=float, default=HGridExplorerConfig.cvrp_work_weight)
    parser.add_argument("--cp-step", type=int, default=HGridExplorerConfig.cp_step)
    parser.add_argument("--cp-guidance-weight", type=float, default=HGridExplorerConfig.cp_guidance_weight)
    parser.add_argument("--cp-index-weight", type=float, default=HGridExplorerConfig.cp_index_weight)
    parser.add_argument("--cp-reach-radius", type=float, default=HGridExplorerConfig.cp_reach_radius)
    parser.add_argument("--local-cluster-window", type=int, default=HGridExplorerConfig.local_cluster_window)
    parser.add_argument("--local-replan-interval", type=int, default=HGridExplorerConfig.local_replan_interval)
    parser.add_argument("--viewpoint-samples-per-cluster", type=int, default=HGridExplorerConfig.viewpoint_samples_per_cluster)
    parser.add_argument("--viewpoint-sample-radius", type=int, default=HGridExplorerConfig.viewpoint_sample_radius)
    parser.add_argument("--viewpoint-min-gain", type=int, default=HGridExplorerConfig.viewpoint_min_gain)
    parser.add_argument("--fis-min-cluster-size", type=int, default=HGridExplorerConfig.fis_min_cluster_size)
    parser.add_argument("--fis-max-cluster-size", type=int, default=HGridExplorerConfig.fis_max_cluster_size)
    parser.add_argument("--target-blacklist-steps", type=int, default=HGridExplorerConfig.target_blacklist_steps)
    parser.add_argument("--target-blacklist-radius", type=float, default=HGridExplorerConfig.target_blacklist_radius)
    parser.add_argument("--target-commit-radius", type=float, default=HGridExplorerConfig.target_commit_radius)
    parser.add_argument("--no-progress-steps", type=int, default=HGridExplorerConfig.no_progress_steps)
    parser.add_argument("--no-info-steps", type=int, default=HGridExplorerConfig.no_info_steps)
    parser.add_argument("--no-progress-distance", type=float, default=HGridExplorerConfig.no_progress_distance)
    parser.add_argument("--hide-cp-path", action="store_false", dest="show_cp_path")
    parser.add_argument("--disable-bspline", action="store_false", dest="use_bspline_smoothing")
    parser.add_argument(
        "--enable-kinodynamic",
        action="store_true",
        dest="use_kinodynamic_astar",
        default=HGridExplorerConfig.use_kinodynamic_astar,
    )
    parser.add_argument(
        "--bspline-samples-per-segment",
        type=int,
        default=HGridExplorerConfig.bspline_samples_per_segment,
    )
    parser.add_argument("--bspline-min-step", type=float, default=HGridExplorerConfig.bspline_min_step)
    parser.add_argument("--bspline-max-smoothness-cost", type=float, default=HGridExplorerConfig.bspline_max_smoothness_cost)
    parser.add_argument("--bspline-max-turn-angle-deg", type=float, default=HGridExplorerConfig.bspline_max_turn_angle_deg)
    parser.add_argument("--bspline-opt-iterations", type=int, default=HGridExplorerConfig.bspline_opt_iterations)
    parser.add_argument("--bspline-max-control-points", type=int, default=HGridExplorerConfig.bspline_max_control_points)
    parser.add_argument("--kinodynamic-max-nodes", type=int, default=HGridExplorerConfig.kinodynamic_max_nodes)
    parser.add_argument("--kinodynamic-goal-tolerance", type=float, default=HGridExplorerConfig.kinodynamic_goal_tolerance)
    parser.add_argument("--kinodynamic-viewpoint-scoring", action="store_true", dest="use_kinodynamic_for_viewpoint_scoring")
    parser.add_argument("--safety-sample-step", type=float, default=HGridExplorerConfig.safety_sample_step)
    parser.add_argument(
        "--coverage-radius",
        "--sensor-radius",
        type=float,
        default=HGridExplorerConfig.coverage_radius,
    )
    parser.add_argument("--safe-radius", type=float, default=HGridExplorerConfig.safe_radius)
    parser.add_argument("--inflation-radius", type=float, default=HGridExplorerConfig.inflation_radius)
    parser.add_argument("--planning-inflation-radius", type=float, default=HGridExplorerConfig.planning_inflation_radius)
    parser.add_argument("--max-steps", type=int, default=HGridExplorerConfig.max_steps)
    parser.add_argument("--stop-known-ratio", type=float, default=HGridExplorerConfig.stop_known_ratio)
    parser.add_argument("--stall-stop-min-known-ratio", type=float, default=HGridExplorerConfig.stall_stop_min_known_ratio)
    parser.add_argument("--global-stall-steps", type=int, default=HGridExplorerConfig.global_stall_steps)
    parser.add_argument("--render-pause", type=float, default=HGridExplorerConfig.render_pause)
    parser.add_argument("--render-interval", type=int, default=HGridExplorerConfig.render_interval)
    parser.add_argument("--render-history-tail", type=int, default=HGridExplorerConfig.render_history_tail)
    parser.add_argument("--render-max-frontier-points", type=int, default=HGridExplorerConfig.render_max_frontier_points)
    parser.add_argument("--enable-camera-fov", action="store_true", dest="use_camera_fov", default=HGridExplorerConfig.use_camera_fov)
    parser.add_argument(
        "--disable-camera-occlusion",
        action="store_false",
        dest="camera_use_occlusion",
        default=HGridExplorerConfig.camera_use_occlusion,
    )
    parser.add_argument("--camera-fov-deg", type=float, default=HGridExplorerConfig.camera_fov_deg)
    parser.add_argument("--camera-yaw-rate-deg", type=float, default=HGridExplorerConfig.camera_yaw_rate_deg)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hgrid_level_sizes = tuple(
        int(part.strip())
        for part in args.hgrid_level_sizes.split(",")
        if part.strip()
    )
    cfg = HGridExplorerConfig(
        random_seed=args.seed,
        obstacle_count=args.obstacle_count,
        num_uavs=args.num_uavs,
        comm_range=args.comm_range,
        max_speed=args.max_speed,
        max_acceleration=args.max_acceleration,
        future_path_horizon=args.future_path_horizon,
        conflict_margin=args.conflict_margin,
        reciprocal_replan_rounds=args.reciprocal_replan_rounds,
        dynamic_obstacle_inflation=args.dynamic_obstacle_inflation,
        hgrid_block_size=args.hgrid_block_size,
        hgrid_level_sizes=hgrid_level_sizes,
        hgrid_split_known_ratio=args.hgrid_split_known_ratio,
        hgrid_min_unknown_cells=args.hgrid_min_unknown_cells,
        hgrid_update_interval=args.hgrid_update_interval,
        pairwise_reassign_interval=args.pairwise_reassign_interval,
        hgrid_balance_weight=args.hgrid_balance_weight,
        pairwise_request_cooldown=args.pairwise_request_cooldown,
        pairwise_success_cooldown=args.pairwise_success_cooldown,
        cvrp_max_exact_blocks=args.cvrp_max_exact_blocks,
        cvrp_route_weight=args.cvrp_route_weight,
        cvrp_work_weight=args.cvrp_work_weight,
        cp_step=args.cp_step,
        cp_guidance_weight=args.cp_guidance_weight,
        cp_index_weight=args.cp_index_weight,
        cp_reach_radius=args.cp_reach_radius,
        local_cluster_window=args.local_cluster_window,
        local_replan_interval=args.local_replan_interval,
        viewpoint_samples_per_cluster=args.viewpoint_samples_per_cluster,
        viewpoint_sample_radius=args.viewpoint_sample_radius,
        viewpoint_min_gain=args.viewpoint_min_gain,
        fis_min_cluster_size=args.fis_min_cluster_size,
        fis_max_cluster_size=args.fis_max_cluster_size,
        target_blacklist_steps=args.target_blacklist_steps,
        target_blacklist_radius=args.target_blacklist_radius,
        target_commit_radius=args.target_commit_radius,
        no_progress_steps=args.no_progress_steps,
        no_info_steps=args.no_info_steps,
        no_progress_distance=args.no_progress_distance,
        show_cp_path=args.show_cp_path,
        use_bspline_smoothing=args.use_bspline_smoothing,
        bspline_samples_per_segment=args.bspline_samples_per_segment,
        bspline_min_step=args.bspline_min_step,
        bspline_max_smoothness_cost=args.bspline_max_smoothness_cost,
        bspline_max_turn_angle_deg=args.bspline_max_turn_angle_deg,
        use_kinodynamic_astar=args.use_kinodynamic_astar,
        bspline_opt_iterations=args.bspline_opt_iterations,
        bspline_max_control_points=args.bspline_max_control_points,
        kinodynamic_max_nodes=args.kinodynamic_max_nodes,
        kinodynamic_goal_tolerance=args.kinodynamic_goal_tolerance,
        use_kinodynamic_for_viewpoint_scoring=args.use_kinodynamic_for_viewpoint_scoring,
        safety_sample_step=args.safety_sample_step,
        coverage_radius=args.coverage_radius,
        safe_radius=args.safe_radius,
        inflation_radius=args.inflation_radius,
        planning_inflation_radius=args.planning_inflation_radius,
        max_steps=args.max_steps,
        stop_known_ratio=args.stop_known_ratio,
        stall_stop_min_known_ratio=args.stall_stop_min_known_ratio,
        global_stall_steps=args.global_stall_steps,
        render_pause=args.render_pause,
        render_interval=args.render_interval,
        render_history_tail=args.render_history_tail,
        render_max_frontier_points=args.render_max_frontier_points,
        use_camera_fov=args.use_camera_fov,
        camera_use_occlusion=args.camera_use_occlusion,
        camera_fov_deg=args.camera_fov_deg,
        camera_yaw_rate_deg=args.camera_yaw_rate_deg,
    )
    result = run_simulation(cfg, show=not args.no_show)
    print(
        "Simulation finished: "
        f"steps={result['steps']}, "
        f"known_ratio={result['known_ratio']:.2%}, "
        f"num_uavs={result['num_uavs']}, "
        f"hgrid_blocks={result['hgrid_blocks']}, "
        f"hgrid_split_count={result['hgrid_split_count']}, "
        f"hgrid_removed_count={result['hgrid_removed_count']}, "
        f"path_length_total={result['path_length_total']}, "
        f"exploration_finished_step={result['exploration_finished_step']}, "
        f"returned_home={result['returned_home']}, "
        f"owned_plan_count={result['owned_plan_count']}, "
        f"fallback_plan_count={result['fallback_plan_count']}, "
        f"hgrid_reassign_count={result['hgrid_reassign_count']}, "
        f"pairwise_success_count={result['pairwise_success_count']}, "
        f"astar_polyline_count={result['astar_polyline_count']}, "
        f"bspline_plan_count={result['bspline_plan_count']}, "
        f"reciprocal_collision_count={result['reciprocal_collision_count']}, "
        f"reciprocal_replan_count={result['reciprocal_replan_count']}, "
        f"unresolved_collision_count={result['unresolved_collision_count']}, "
        f"home_flags={result['home_flags']}, "
        f"final_positions={result['final_positions']}"
    )
