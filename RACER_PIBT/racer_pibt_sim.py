"""RACER target assignment combined with PIBT-based motion planning."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

try:
    from . import racer_bridge  # noqa: F401
except ImportError:  # Support running RACER_PIBT.py directly.
    import racer_bridge  # noqa: F401
from racer_hgrid import pairwise_request_response_hgrid_blocks
from racer_map import FREE, communication_components, union_known_grid
from racer_planner import plan_uav_with_hgrid
from racer_sim import RACERSimulator
from racer_types import RACERConfig

try:
    from .pibt_motion import PIBTMotionConfig, PairwisePIBTMotionPlanner
except ImportError:
    from pibt_motion import PIBTMotionConfig, PairwisePIBTMotionPlanner


class RACERPIBTSimulator(RACERSimulator):
    """Keep RACER allocation intact and replace only its motion layer."""

    def __init__(self, config: RACERConfig, motion_config: PIBTMotionConfig | None = None):
        super().__init__(config)
        self.motion_config = motion_config or PIBTMotionConfig(seed=config.random_seed)
        collision_monitor_radius = config.meters_to_cells(config.swarm_collision_check_distance) / 2.0
        if self.motion_config.agent_radius < collision_monitor_radius:
            self.motion_config = replace(self.motion_config, agent_radius=collision_monitor_radius)
        self.motion_config.validate(config.map_resolution, config.comm_range)
        self.motion_planner = PairwisePIBTMotionPlanner(self.motion_config)
        self.continuous_trajectories: dict[int, Any] = {}
        self.pibt_messages = []
        self.pibt_coordinators: dict[int, int] = {}
        self.stats.update(
            {
                "pibt_epochs": 0,
                "pibt_requests": 0,
                "pibt_responses": 0,
                "pibt_priority_inheritances": 0,
                "pibt_backtracks": 0,
                "pibt_waits": 0,
                "pibt_target_projections": 0,
                "pibt_trajectory_rejections": 0,
                "pibt_messages": 0,
                "pibt_message_hops": 0,
                "pibt_sfc_count": 0,
                "pibt_lsc_pair_count": 0,
            }
        )
        self.max_velocity_observed = 0.0
        self.max_acceleration_observed = 0.0

    def exploration_tick(self, step_idx: int) -> None:
        if step_idx % max(1, self.config.hgrid_update_interval) == 0:
            split_count, removed_count = self.hgrid.update_active_cells(
                union_known_grid(self.known_maps), self.world.raw_obstacle_map
            )
            self.stats["hgrid_split_count"] += split_count
            self.stats["hgrid_removed_count"] += removed_count
        if step_idx % max(1, self.config.pairwise_reassign_interval) == 0:
            changed, successes = pairwise_request_response_hgrid_blocks(
                self.uavs, self.known_maps, self.hgrid, step_idx, self.config
            )
            self.stats["hgrid_reassign_count"] += changed
            self.stats["pairwise_success_count"] += successes

        planned_count = 0
        for uav, known_map in zip(self.uavs, self.known_maps):
            keep_committed_target = (
                uav.target is not None
                and uav.pos != uav.target
                and step_idx % max(1, self.config.local_replan_interval) != 0
            )
            if keep_committed_target:
                planned, mode = True, "keep"
            else:
                planned, mode = plan_uav_with_hgrid(uav, known_map, self.hgrid, self.config, step_idx)
            if planned:
                planned_count += 1
                if mode == "cp_owned":
                    self.stats["owned_plan_count"] += 1
                elif mode == "cp_fallback":
                    self.stats["fallback_plan_count"] += 1
            # The RACER path is used only to select/score the target. PIBT owns execution.
            uav.path = []

        if planned_count == 0:
            self.hgrid.update_active_cells(union_known_grid(self.known_maps), self.world.raw_obstacle_map)
            if not self.hgrid.blocks:
                self.phase = "return_home"
                self.exploration_finished_step = step_idx
        self._plan_pibt_motion(step_idx)

    def return_home_tick(self) -> None:
        for uav in self.uavs:
            uav.target = uav.start
            uav.path = []
        self._plan_pibt_motion(self.stats["pibt_epochs"])

    def _plan_pibt_motion(self, epoch: int) -> None:
        components = communication_components(self.uavs, self.config.comm_range)
        self.stats["pibt_epochs"] += 1
        for component_index, component in enumerate(components):
            known_grid = self.known_maps[component[0]].grid
            free_grid = np.asarray(known_grid == FREE, dtype=bool)
            for global_index in component:
                uav = self.uavs[global_index]
                for x, y in uav.local_blocked_cells:
                    if 0 <= y < free_grid.shape[0] and 0 <= x < free_grid.shape[1]:
                        free_grid[y, x] = False
            for global_index in component:
                uav = self.uavs[global_index]
                free_grid[uav.pos[1], uav.pos[0]] = True

            uav_ids = [self.uavs[index].id for index in component]
            positions = [self.uavs[index].pos for index in component]
            targets = [self.uavs[index].target or self.uavs[index].pos for index in component]
            plan = self.motion_planner.plan_component(
                epoch, uav_ids, free_grid, positions, targets, self.config.comm_range
            )
            self.pibt_coordinators[component_index] = plan.coordinator_id
            self.pibt_messages.extend(plan.messages)
            self.stats["pibt_requests"] += plan.pibt_stats.requests
            self.stats["pibt_responses"] += plan.pibt_stats.responses
            self.stats["pibt_priority_inheritances"] += plan.pibt_stats.priority_inheritances
            self.stats["pibt_backtracks"] += plan.pibt_stats.backtracks
            self.stats["pibt_waits"] += plan.pibt_stats.waits
            self.stats["pibt_target_projections"] += plan.projected_count
            self.stats["pibt_trajectory_rejections"] += plan.trajectory_rejections
            self.stats["pibt_messages"] += len(plan.messages)
            self.stats["pibt_message_hops"] += sum(max(0, len(message.route) - 1) for message in plan.messages)
            self.stats["pibt_sfc_count"] += len(plan.trajectories)
            self.stats["pibt_lsc_pair_count"] += len(component) * (len(component) - 1) // 2

            for global_index, next_position, trajectory in zip(component, plan.next_positions, plan.trajectories):
                uav = self.uavs[global_index]
                uav.set_plan(uav.target, [] if next_position == uav.pos else [next_position])
                self.continuous_trajectories[uav.id] = trajectory
                self.max_velocity_observed = max(self.max_velocity_observed, trajectory.max_velocity)
                self.max_acceleration_observed = max(self.max_acceleration_observed, trajectory.max_acceleration)

    def resolve_swarm_conflicts(self) -> None:
        """PIBT vertex/edge reservations plus LSC replace RACER's independent replan."""

    def enforce_static_execution_safety(self, movable) -> None:
        """Reject a stale/incorrect known-map step without invoking RACER motion planning."""
        for uav in movable:
            if not uav.path:
                continue
            next_position = uav.path[0]
            if self.world.segment_collides_raw_obstacle(
                uav.pos,
                next_position,
                margin=self.motion_config.agent_radius,
                step=self.config.shortcut_safety_step,
            ):
                uav.local_blocked_cells.add(next_position)
                uav.set_plan(uav.target, [])
                self.stats["static_collision_stop_count"] += 1

    def result(self, final_step: int, final_ratio: float) -> dict[str, Any]:
        result = super().result(final_step, final_ratio)
        result.update(
            {
                "planner": "RACER target allocation + PIBT/SFC/LSC/Bernstein motion",
                "pibt_agent_radius": self.motion_config.agent_radius,
                "pibt_max_velocity_observed": self.max_velocity_observed,
                "pibt_max_acceleration_observed": self.max_acceleration_observed,
                "pibt_component_coordinators": dict(self.pibt_coordinators),
            }
        )
        return result


def run_simulation(
    config: RACERConfig,
    motion_config: PIBTMotionConfig | None = None,
    show: bool = True,
) -> dict[str, Any]:
    return RACERPIBTSimulator(config, motion_config).run(show=show)
