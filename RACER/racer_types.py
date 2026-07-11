"""Shared RACER data types and constants."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

FREE = 0
OBSTACLE = 1
UNKNOWN = -1

@dataclass
class RACERConfig:
    width: int = 53
    height: int = 50
    map_resolution: float = 1.0
    known_strip_width: int = 3
    obstacle_count: int = 50
    random_seed: int = 42
    obstacle_inflation_radius: float = 0.199
    planning_inflation_radius: float = 0.199
    manager_clearance_threshold: float = 0.2
    frontier_min_candidate_clearance: float = 0.21
    search_optimistic: bool = False
    execution_check_horizon: int = 6

    num_uavs: int = 4
    max_steps: int = 1500
    max_speed: int = 1
    coverage_radius: float = 3.0
    safe_radius: float = 1.0
    comm_range: float = 7.0
    stop_known_ratio: float = 0.985
    stall_stop_min_known_ratio: float = 0.95
    global_stall_steps: int = 250

    hgrid_level_sizes: tuple[int, ...] = (20, 10, 5)
    hgrid_split_known_ratio: float = 0.35
    hgrid_min_unknown_cells: int = 2
    hgrid_update_interval: int = 4
    pairwise_reassign_interval: int = 8
    pairwise_request_cooldown: int = 3
    pairwise_success_cooldown: int = 12
    cvrp_max_exact_blocks: int = 14
    cvrp_route_weight: float = 1.0
    cvrp_work_weight: float = 0.08
    hgrid_balance_weight: float = 0.08

    cp_step: int = 3
    cp_reach_radius: float = 2.0
    cp_guidance_weight: float = 0.18
    cp_index_weight: float = 0.04
    local_cluster_window: int = 3
    viewpoint_samples_per_cluster: int = 4
    viewpoint_sample_radius: int = 4
    viewpoint_min_gain: int = 1
    fis_min_cluster_size: int = 1
    fis_max_cluster_size: int = 28
    local_replan_interval: int = 5
    target_commit_radius: float = 1.2
    current_target_bonus: float = 2.0
    target_blacklist_steps: int = 35
    target_blacklist_radius: float = 2.5
    no_progress_steps: int = 18
    no_info_steps: int = 12
    no_progress_distance: float = 0.2

    future_path_horizon: int = 5
    conflict_margin: float = 0.25
    reciprocal_replan_rounds: int = 2
    dynamic_obstacle_inflation: float = 1.2

    render_interval: int = 2
    render_pause: float = 0.001
    render_history_tail: int = 0
    turn_penalty: float = 0.35
    allow_diagonal_motion: bool = False
    shortcut_safety_step: float = 0.10
    use_kinodynamic_astar: bool = True
    kinodynamic_max_speed: int = 2
    kinodynamic_max_accel: int = 1
    kinodynamic_max_nodes: int = 6000
    use_bspline_smoothing: bool = True
    bspline_samples_per_segment: int = 5
    bspline_smooth_iterations: int = 4
    bspline_obstacle_clearance: float = 0.7
    bspline_smooth_weight: float = 0.35
    bspline_obstacle_weight: float = 0.16

    def meters_to_cells(self, value: float) -> float:
        return value / max(1e-9, self.map_resolution)

    def obstacle_inflation_cells(self) -> float:
        return self.meters_to_cells(self.obstacle_inflation_radius)

    def planning_inflation_cells(self) -> float:
        return self.meters_to_cells(self.planning_inflation_radius)

    def manager_clearance_threshold_cells(self) -> float:
        return self.meters_to_cells(self.manager_clearance_threshold)

    def frontier_min_candidate_clearance_cells(self) -> float:
        return self.meters_to_cells(self.frontier_min_candidate_clearance)

    def bspline_obstacle_clearance_cells(self) -> float:
        return self.meters_to_cells(self.bspline_obstacle_clearance)


@dataclass
class UAV:
    uav_id: int
    start: tuple[int, int]
    safe_radius: float
    coverage_radius: float
    max_speed_cells: int = 1
    pos: tuple[int, int] = field(init=False)
    path: list[tuple[int, int]] = field(default_factory=list)
    target: tuple[int, int] | None = None
    history: list[tuple[int, int]] = field(init=False)

    def __post_init__(self) -> None:
        self.pos = self.start
        self.history = [self.start]
        self.velocity = (0, 0)
        self.cp_cursor = 0
        self.cp_path: list[tuple[int, int]] = []
        self.fis = None
        self.target_blacklist: dict[tuple[int, int], int] = {}
        self.local_blocked_cells: set[tuple[int, int]] = set()
        self.hgrid_last_attempt = -10**9
        self.hgrid_last_success: dict[int, int] = {}
        self.last_progress_pos = self.pos
        self.last_target_distance: float | None = None
        self.stagnant_steps = 0
        self.no_info_steps = 0
        self.last_planner_mode = "none"
        self.assigned_cluster_count = 0

    @property
    def id(self) -> int:
        return self.uav_id

    def distance_to(self, other: "UAV") -> float:
        return math.hypot(self.pos[0] - other.pos[0], self.pos[1] - other.pos[1])

    def set_plan(self, target: tuple[int, int] | None, path: list[tuple[int, int]]) -> None:
        self.target = target
        self.path = list(path)
        if self.path and self.path[0] == self.pos:
            self.path = self.path[1:]

    def step(self) -> None:
        old = self.pos
        for _ in range(max(1, self.max_speed_cells)):
            if not self.path:
                break
            self.pos = self.path.pop(0)
            self.history.append(self.pos)
        self.velocity = (self.pos[0] - old[0], self.pos[1] - old[1])
