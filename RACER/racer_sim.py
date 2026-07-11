"""fast_exploration_fsm / planner_manager-style simulation loop."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
except Exception:  # pragma: no cover - visualization is optional.
    patches = None
    plt = None
    PatchCollection = None

from racer_hgrid import HGrid, pairwise_request_response_hgrid_blocks
from racer_map import (
    GridWorld,
    KnownMap,
    communication_components,
    explorable_known_ratio,
    make_column_starts,
    merge_known_maps_for_component,
    planning_grid_from_known_map,
    planning_grid_for_uav,
    union_known_grid,
    update_known_map_with_sensor,
)
from racer_planner import (
    blacklist_target,
    detect_explorable_frontier_cells,
    plan_uav_with_hgrid,
)
from racer_path_searching import (
    astar,
    kinodynamic_astar,
    path_is_free,
)
from racer_trajectory import densify_grid_path, smooth_trajectory
from racer_types import FREE, OBSTACLE, RACERConfig, UAV

class RACERSimulator:
    def __init__(self, config: RACERConfig):
        self.config = config
        self.world = GridWorld(config)
        self.uavs = [
            UAV(i + 1, start, config.safe_radius, config.coverage_radius, max(1, int(config.max_speed)))
            for i, start in enumerate(make_column_starts(config))
        ]
        self.known_maps = [KnownMap(self.world) for _ in self.uavs]
        for known_map in self.known_maps:
            known_map.grid[:, : config.known_strip_width] = self.world.raw_obstacle_map[:, : config.known_strip_width]
        for uav, known_map in zip(self.uavs, self.known_maps):
            update_known_map_with_sensor(known_map, uav, config)

        self.hgrid = HGrid(config)
        self.hgrid.assign_initial_owners(self.uavs)
        self.hgrid.update_active_cells(union_known_grid(self.known_maps), self.world.raw_obstacle_map)
        self.phase = "explore"
        self.stats = {
            "hgrid_split_count": 0,
            "hgrid_removed_count": 0,
            "hgrid_reassign_count": 0,
            "pairwise_success_count": 0,
            "owned_plan_count": 0,
            "fallback_plan_count": 0,
            "astar_polyline_count": 0,
            "bspline_plan_count": 0,
            "swarm_collision_count": 0,
            "swarm_replan_count": 0,
            "swarm_replan_failure_count": 0,
            "static_collision_replan_count": 0,
            "static_collision_stop_count": 0,
        }
        self.exploration_finished_step: int | None = None
        self.returned_home = False

    def run(self, show: bool = True) -> dict[str, Any]:
        if show and plt is None:
            print("matplotlib is not available; running headless.")
            show = False
        fig = ax = None
        if show:
            plt.ion()
            fig, ax = plt.subplots(figsize=(9, 8))

        best_ratio = self.known_ratio()
        last_improvement = 0
        final_step = 0
        final_ratio = best_ratio

        for step_idx in range(self.config.max_steps):
            final_step = step_idx
            self.merge_maps()
            final_ratio = self.known_ratio()
            if final_ratio > best_ratio + 1e-4:
                best_ratio = final_ratio
                last_improvement = step_idx
            self.update_phase(final_ratio, step_idx, last_improvement)

            if self.phase == "explore":
                self.exploration_tick(step_idx)
            else:
                self.return_home_tick()
                if all(uav.pos == uav.start for uav in self.uavs):
                    self.returned_home = True
                    self.render(show, ax, step_idx, final_ratio, force=True)
                    break

            self.resolve_swarm_conflicts()
            movable = [uav for uav in self.uavs if uav.path]
            if not movable:
                self.render(show, ax, step_idx, final_ratio, force=True)
                if self.phase == "explore":
                    continue
                continue
            self.enforce_static_execution_safety(movable)
            movable = [uav for uav in self.uavs if uav.path]
            if not movable:
                self.render(show, ax, step_idx, final_ratio, force=True)
                if self.phase == "explore":
                    continue
                continue
            self.move_uavs_safely(movable)
            for uav, known_map in zip(self.uavs, self.known_maps):
                newly_known = update_known_map_with_sensor(known_map, uav, self.config)
                self.update_progress_monitor(uav, newly_known)
            self.render(show, ax, step_idx, final_ratio)

        if show:
            plt.ioff()
            plt.show()
        return self.result(final_step, final_ratio)

    def merge_maps(self) -> None:
        for component in communication_components(self.uavs, self.config.comm_range):
            merge_known_maps_for_component(component, self.known_maps)

    def known_ratio(self) -> float:
        return explorable_known_ratio(self.known_maps, self.world.raw_obstacle_map, self.config.known_strip_width)

    def update_phase(self, known_ratio: float, step_idx: int, last_improvement: int) -> None:
        if self.phase != "explore":
            return
        stalled = (
            known_ratio >= self.config.stall_stop_min_known_ratio
            and self.config.global_stall_steps > 0
            and step_idx - last_improvement >= self.config.global_stall_steps
        )
        if known_ratio >= self.config.stop_known_ratio or stalled:
            self.phase = "return_home"
            self.exploration_finished_step = step_idx

    def exploration_tick(self, step_idx: int) -> None:
        if step_idx % max(1, self.config.hgrid_update_interval) == 0:
            split_count, removed_count = self.hgrid.update_active_cells(union_known_grid(self.known_maps), self.world.raw_obstacle_map)
            self.stats["hgrid_split_count"] += split_count
            self.stats["hgrid_removed_count"] += removed_count
        if step_idx % max(1, self.config.pairwise_reassign_interval) == 0:
            changed, successes = pairwise_request_response_hgrid_blocks(self.uavs, self.known_maps, self.hgrid, step_idx, self.config)
            self.stats["hgrid_reassign_count"] += changed
            self.stats["pairwise_success_count"] += successes

        planned_count = 0
        for uav, known_map in zip(self.uavs, self.known_maps):
            planned, mode = plan_uav_with_hgrid(uav, known_map, self.hgrid, self.config, step_idx)
            if not planned:
                continue
            planned_count += 1
            if mode != "keep":
                if mode == "cp_owned":
                    self.stats["owned_plan_count"] += 1
                elif mode == "cp_fallback":
                    self.stats["fallback_plan_count"] += 1
                if "polyline" in uav.last_planner_mode:
                    self.stats["astar_polyline_count"] += 1
                if "bspline" in uav.last_planner_mode:
                    self.stats["bspline_plan_count"] += 1
        if planned_count == 0:
            self.hgrid.update_active_cells(union_known_grid(self.known_maps), self.world.raw_obstacle_map)
            if not self.hgrid.blocks:
                self.phase = "return_home"
                self.exploration_finished_step = step_idx

    def return_home_tick(self) -> None:
        snapshots = self.swarm_trajectory_snapshots()
        for uav, known_map in zip(self.uavs, self.known_maps):
            if uav.pos == uav.start:
                uav.set_plan(uav.start, [])
                continue
            if uav.path and uav.target == uav.start:
                continue
            other_paths = [path for index, path in enumerate(snapshots) if index != uav.id - 1]
            path = self.plan_trajectory_to_target(uav, known_map, uav.start, other_paths)
            uav.set_plan(uav.start, path)
            if path:
                self.stats["bspline_plan_count"] += 1

    def plan_trajectory_to_target(
        self,
        uav: UAV,
        known_map: KnownMap,
        target: tuple[int, int],
        swarm_paths: list[list[tuple[int, int]]],
    ) -> list[tuple[int, int]]:
        """Plan independently while treating received swarm trajectories as a soft cost."""
        grid = planning_grid_for_uav(known_map, self.config, uav)
        guide = []
        if self.config.use_kinodynamic_astar:
            guide = kinodynamic_astar(uav.pos, uav.velocity, target, grid, self.config)
        if not guide:
            guide = astar(uav.pos, target, grid, self.config)
        if len(guide) <= 1:
            return []

        guide = densify_grid_path(guide)
        soft_grid = planning_grid_from_known_map(known_map, self.config, block_unknown=False)
        path = smooth_trajectory(guide, grid, self.config, soft_grid, swarm_paths)
        if not path:
            return []
        if self.first_geometric_unsafe_cell([uav.pos] + path) is not None:
            return []
        return path if path_is_free([uav.pos] + path, grid, include_start=False) else []

    def update_progress_monitor(self, uav: UAV, newly_known: int) -> None:
        moved = math.hypot(uav.pos[0] - uav.last_progress_pos[0], uav.pos[1] - uav.last_progress_pos[1])
        target_dist = math.hypot(uav.pos[0] - uav.target[0], uav.pos[1] - uav.target[1]) if uav.target is not None else None
        previous_target_dist = uav.last_target_distance
        if newly_known > 0:
            uav.last_progress_pos = uav.pos
            uav.stagnant_steps = 0
            uav.no_info_steps = 0
            uav.last_target_distance = target_dist
            return

        approaching_target = (
            moved >= self.config.no_progress_distance
            and target_dist is not None
            and (previous_target_dist is None or target_dist < previous_target_dist - 0.05)
        )

        uav.no_info_steps += 1
        if approaching_target:
            uav.last_progress_pos = uav.pos
            uav.stagnant_steps = 0
            uav.last_target_distance = target_dist
            moving_no_info_limit = max(self.config.no_info_steps * 3, self.config.no_info_steps + 24)
            if uav.no_info_steps < moving_no_info_limit:
                return

        if uav.no_info_steps >= self.config.no_info_steps:
            blacklist_target(uav, uav.target, self.config)
            uav.set_plan(None, [])
            uav.no_info_steps = 0
            uav.stagnant_steps = 0
            uav.last_progress_pos = uav.pos
            uav.last_target_distance = None
            return

        if moved >= self.config.no_progress_distance:
            uav.last_progress_pos = uav.pos
            uav.stagnant_steps = 0
            return

        uav.stagnant_steps += 1
        if uav.stagnant_steps >= self.config.no_progress_steps:
            blacklist_target(uav, uav.target, self.config)
            uav.set_plan(None, [])
            uav.stagnant_steps = 0
            uav.last_progress_pos = uav.pos
            uav.last_target_distance = None

    def enforce_static_execution_safety(self, movable: list[UAV]) -> None:
        for uav in movable:
            known_map = self.known_maps[uav.id - 1]
            safety_grid = planning_grid_for_uav(known_map, self.config, uav)
            future = [uav.pos] + uav.path[: max(1, self.config.execution_check_horizon)]
            unsafe_cell = self.first_geometric_unsafe_cell(future)
            if unsafe_cell is None and path_is_free(future, safety_grid, include_start=False):
                continue
            if unsafe_cell is not None:
                uav.local_blocked_cells.add(unsafe_cell)
                safety_grid = planning_grid_for_uav(known_map, self.config, uav)
            self.stats["static_collision_replan_count"] += 1
            if uav.target is None:
                self.stop_unsafe_uav(uav)
                continue
            snapshots = self.swarm_trajectory_snapshots()
            other_paths = [path for index, path in enumerate(snapshots) if index != uav.id - 1]
            replanned = self.plan_trajectory_to_target(uav, known_map, uav.target, other_paths)
            candidate = [uav.pos] + replanned[: max(1, self.config.execution_check_horizon)]
            if replanned and self.first_geometric_unsafe_cell(candidate) is None and path_is_free(candidate, safety_grid, include_start=False):
                uav.set_plan(uav.target, replanned)
                continue
            self.stop_unsafe_uav(uav)

    def first_geometric_unsafe_cell(self, path: list[tuple[int, int]]) -> tuple[int, int] | None:
        if len(path) <= 1:
            return None
        margin = max(self.config.obstacle_inflation_cells(), self.config.manager_clearance_threshold_cells())
        for start, end in zip(path, path[1:]):
            if self.world.segment_collides_raw_obstacle(start, end, margin=margin, step=self.config.shortcut_safety_step):
                return end
        return None

    def move_uavs_safely(self, movable: list[UAV]) -> None:
        for uav in movable:
            if not uav.path:
                continue
            next_pos = uav.path[0]
            unsafe_by_geometry = self.world.segment_collides_raw_obstacle(
                uav.pos,
                next_pos,
                margin=max(self.config.obstacle_inflation_cells(), self.config.manager_clearance_threshold_cells()),
                step=self.config.shortcut_safety_step,
            )
            if (
                path_is_free([uav.pos, next_pos], self.world.raw_obstacle_map, include_start=False)
                and not unsafe_by_geometry
            ):
                uav.step()
                continue

            known_map = self.known_maps[uav.id - 1]
            if unsafe_by_geometry:
                uav.local_blocked_cells.add(next_pos)
            for x, y in [uav.pos, next_pos]:
                if 0 <= x < self.world.width and 0 <= y < self.world.height:
                    known_map.grid[y, x] = self.world.raw_obstacle_map[y, x]
            self.stop_unsafe_uav(uav)

    def stop_unsafe_uav(self, uav: UAV) -> None:
        # Official RACER hovers after a failed replan and retries the same target.
        uav.set_plan(uav.target, [])
        self.stats["static_collision_stop_count"] += 1

    def swarm_trajectory_snapshots(self) -> list[list[tuple[int, int]]]:
        """Represent the newest trajectory broadcast by every UAV."""
        return [[uav.pos] + list(uav.path) for uav in self.uavs]

    def detect_swarm_collision_pairs(
        self,
        snapshots: list[list[tuple[int, int]]],
    ) -> list[tuple[int, int]]:
        """Check time-aligned trajectory samples, matching official RACER's callback."""
        threshold = self.config.meters_to_cells(self.config.swarm_collision_check_distance)
        collisions = []
        for i in range(len(snapshots)):
            for j in range(i + 1, len(snapshots)):
                overlap = min(len(snapshots[i]), len(snapshots[j]))
                for step in range(overlap):
                    first = snapshots[i][step]
                    second = snapshots[j][step]
                    if math.hypot(first[0] - second[0], first[1] - second[1]) < threshold:
                        collisions.append((i, j))
                        break
        return collisions

    def resolve_swarm_conflicts(self) -> None:
        """Independently replan every colliding UAV with no winner/loser arbitration."""
        snapshots = self.swarm_trajectory_snapshots()
        collisions = self.detect_swarm_collision_pairs(snapshots)
        if not collisions:
            return

        self.stats["swarm_collision_count"] += len(collisions)
        colliding_indices = sorted({index for pair in collisions for index in pair})
        for index in colliding_indices:
            uav = self.uavs[index]
            if uav.target is None:
                uav.set_plan(uav.target, [])
                self.stats["swarm_replan_failure_count"] += 1
                continue
            other_paths = [path for other_index, path in enumerate(snapshots) if other_index != index]
            path = self.plan_trajectory_to_target(uav, self.known_maps[index], uav.target, other_paths)
            if path:
                uav.set_plan(uav.target, path)
                self.stats["swarm_replan_count"] += 1
            else:
                uav.set_plan(uav.target, [])
                self.stats["swarm_replan_failure_count"] += 1

    def render(self, show: bool, ax, step_idx: int, known_ratio: float, force: bool = False) -> None:
        if not show or ax is None or plt is None:
            return
        if not force and step_idx % max(1, self.config.render_interval) != 0:
            return
        render_state(ax, self.world, self.known_maps, self.uavs, self.hgrid, step_idx, known_ratio, self.phase, self.config)
        plt.pause(self.config.render_pause)

    def result(self, final_step: int, final_ratio: float) -> dict[str, Any]:
        result = {
            "steps": final_step,
            "known_ratio": final_ratio,
            "num_uavs": len(self.uavs),
            "hgrid_blocks": len(self.hgrid.blocks),
            "path_length_total": sum(len(uav.history) for uav in self.uavs),
            "exploration_finished_step": self.exploration_finished_step,
            "returned_home": self.returned_home,
            "final_positions": [uav.pos for uav in self.uavs],
            "home_flags": [uav.pos == uav.start for uav in self.uavs],
        }
        result.update(self.stats)
        return result


def render_state(ax, world: GridWorld, known_maps: list[KnownMap], uavs: list[UAV], hgrid: HGrid, step_idx: int, known_ratio: float, phase: str, config: RACERConfig) -> None:
    ax.clear()
    cmap = plt.get_cmap("tab20")
    merged_grid = union_known_grid(known_maps)
    display = np.zeros((world.height, world.width, 3), dtype=float)
    display[:, :] = [0.90, 0.90, 0.90]
    display[merged_grid == FREE] = [1.0, 1.0, 1.0]
    display[merged_grid == OBSTACLE] = [0.35, 0.35, 0.35]
    ax.imshow(display, origin="lower", extent=[-0.5, world.width - 0.5, -0.5, world.height - 0.5])

    obstacle_patches = [patches.Rectangle((rx, ry), rw, rh) for rx, ry, rw, rh in world.rectangles]
    obstacle_patches.extend(patches.Circle((cx, cy), radius) for cx, cy, radius in world.circles)
    if obstacle_patches:
        ax.add_collection(PatchCollection(obstacle_patches, facecolor="gray", edgecolor="black", linewidth=1, alpha=0.45))

    for block in hgrid.blocks:
        color = "black" if block.owner_id is None else cmap((block.owner_id - 1) % 20)
        rect = patches.Rectangle((block.x_min, block.y_min), block.width(), block.height(), fill=False, edgecolor=color, linewidth=max(0.7, 1.5 - 0.25 * block.level), alpha=0.65)
        ax.add_patch(rect)

    for uav in uavs:
        color = cmap((uav.id - 1) % 20)
        history = uav.history if config.render_history_tail <= 0 else uav.history[-config.render_history_tail :]
        hx = [p[0] for p in history]
        hy = [p[1] for p in history]
        ax.plot(hx, hy, color=color, linewidth=1.5)
        if uav.cp_path:
            ax.plot([p[0] for p in uav.cp_path], [p[1] for p in uav.cp_path], color=color, linewidth=0.7, alpha=0.35)
        if uav.path:
            ax.plot([uav.pos[0]] + [p[0] for p in uav.path], [uav.pos[1]] + [p[1] for p in uav.path], color=color, linestyle="--", linewidth=1)
        ax.scatter([uav.pos[0]], [uav.pos[1]], s=80, color=[color], edgecolors="black", zorder=5)
        ax.text(uav.pos[0] + 0.25, uav.pos[1] + 0.25, f"U{uav.id}", color="black")

    ax.set_title(f"RACER modular | step={step_idx} | known={known_ratio:.1%} | {phase}")
    ax.set_xlim(-0.5, world.width - 0.5)
    ax.set_ylim(-0.5, world.height - 0.5)
    ax.set_aspect("equal")


def run_simulation(config: RACERConfig, show: bool = True) -> dict[str, Any]:
    return RACERSimulator(config).run(show=show)
