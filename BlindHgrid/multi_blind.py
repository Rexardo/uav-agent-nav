"""
Version 2: multi-UAV unknown-environment frontier exploration.

Adds on top of single_blind.py:
    1. Multiple UAVs start from the known 50 x 3 left strip.
    2. UAVs merge known_map inside communication range.
    3. Frontier clusters are directly assigned to UAVs.
    4. Each UAV plans with A* only inside its current known safe map.
    5. After the exploration target is reached, all UAVs return to their starts.

This version intentionally does not include hgrid tasks, CP-guided exploration,
or multi-UAV conflict handling. Those are later versions.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from single_blind import (
    FREE,
    OBSTACLE,
    UNKNOWN,
    ExplorerConfig,
    FrontierDetector,
    GridWorld,
    KnownMap,
    ViewpointPlanner,
    astar,
)


@dataclass
class MultiExplorerConfig(ExplorerConfig):
    num_uavs: int = 4
    comm_range: float = 7.0
    frontier_balance_weight: float = 0.20


class MultiUAV:
    def __init__(
        self,
        uav_id: int,
        start: tuple[int, int],
        safe_radius: float,
        coverage_radius: float,
        max_speed_cells: int = 1,
    ):
        self.id = uav_id
        self.start = start
        self.pos = start
        self.safe_radius = safe_radius
        self.coverage_radius = coverage_radius
        self.max_speed_cells = max_speed_cells
        self.path: list[tuple[int, int]] = []
        self.history = [start]
        self.target: tuple[int, int] | None = None
        self.assigned_cluster_count = 0

    def distance_to(self, other: "MultiUAV") -> float:
        return math.hypot(self.pos[0] - other.pos[0], self.pos[1] - other.pos[1])

    def set_plan(self, target: tuple[int, int] | None, path: list[tuple[int, int]]) -> None:
        self.target = target
        self.path = list(path)
        if self.path and self.path[0] == self.pos:
            self.path = self.path[1:]

    def step(self) -> None:
        for _ in range(self.max_speed_cells):
            if not self.path:
                break
            self.pos = self.path.pop(0)
            self.history.append(self.pos)

    def at_start(self) -> bool:
        return self.pos == self.start


def make_column_starts(config: MultiExplorerConfig) -> list[tuple[int, int]]:
    """Place UAVs in one vertical column inside the known left strip."""
    x = max(0, min(config.known_strip_width - 2, 1))
    margin = max(3, int(math.ceil(config.coverage_radius + config.safe_radius)))

    if config.num_uavs == 1:
        return [(x, config.height // 2)]

    ys = np.linspace(margin, config.height - margin - 1, config.num_uavs)
    return [(x, int(round(y))) for y in ys]


def communication_components(uavs: list[MultiUAV], comm_range: float) -> list[list[int]]:
    """Return UAV indices connected by communication-range links."""
    remaining = set(range(len(uavs)))
    components = []

    while remaining:
        seed = remaining.pop()
        component = [seed]
        stack = [seed]

        while stack:
            i = stack.pop()
            for j in list(remaining):
                if uavs[i].distance_to(uavs[j]) <= comm_range:
                    remaining.remove(j)
                    stack.append(j)
                    component.append(j)

        components.append(component)

    return components


def merge_known_maps_for_component(component: list[int], known_maps: list[KnownMap]) -> None:
    """Merge all non-unknown cells among UAVs in one communication component."""
    merged = np.full_like(known_maps[component[0]].grid, UNKNOWN)

    for idx in component:
        grid = known_maps[idx].grid
        known_mask = grid != UNKNOWN
        merged[known_mask] = grid[known_mask]

    for idx in component:
        known_maps[idx].grid[:, :] = merged


def union_known_grid(known_maps: list[KnownMap]) -> np.ndarray:
    merged = np.full_like(known_maps[0].grid, UNKNOWN)
    for known_map in known_maps:
        known_mask = known_map.grid != UNKNOWN
        merged[known_mask] = known_map.grid[known_mask]
    return merged


def global_known_ratio(known_maps: list[KnownMap], known_strip_width: int) -> float:
    merged = union_known_grid(known_maps)
    region = merged[:, known_strip_width:]
    return float(np.mean(region != UNKNOWN))


def assign_clusters_to_uavs(
    clusters: list[list[tuple[int, int]]],
    uavs: list[MultiUAV],
    component: list[int],
    balance_weight: float,
) -> dict[int, list[list[tuple[int, int]]]]:
    """
    Direct frontier-cluster assignment.

    Larger clusters are assigned first. The score combines distance to cluster
    center and a small load term, so one UAV does not greedily take everything.
    """
    assignments = {uavs[idx].id: [] for idx in component}
    loads = {uavs[idx].id: 0 for idx in component}

    for cluster in sorted(clusters, key=len, reverse=True):
        cx = sum(cell[0] for cell in cluster) / len(cluster)
        cy = sum(cell[1] for cell in cluster) / len(cluster)

        best_idx = min(
            component,
            key=lambda idx: (
                math.hypot(uavs[idx].pos[0] - cx, uavs[idx].pos[1] - cy)
                + balance_weight * loads[uavs[idx].id]
            ),
        )
        uav_id = uavs[best_idx].id
        assignments[uav_id].append(cluster)
        loads[uav_id] += len(cluster)

    return assignments


def render_multi_state(
    ax,
    world: GridWorld,
    known_maps: list[KnownMap],
    uavs: list[MultiUAV],
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
    display[merged_grid == OBSTACLE] = [0.50, 0.50, 0.50]
    ax.imshow(display, origin="lower", extent=[0, world.width, 0, world.height])

    hidden_obstacles = (world.raw_obstacle_map == OBSTACLE) & (merged_grid == UNKNOWN)
    hy, hx = np.where(hidden_obstacles)
    if len(hx) > 0:
        ax.scatter(hx + 0.5, hy + 0.5, s=6, c="gray", alpha=0.18, marker="s")

    ax.axvspan(0, world.config.known_strip_width, color=cmap(0), alpha=0.08)

    global_view = KnownMap(world)
    global_view.grid[:, :] = merged_grid
    clusters = FrontierDetector(global_view).cluster_frontiers()
    if clusters:
        frontier_points = [cell for cluster in clusters for cell in cluster]
        fx = [p[0] + 0.5 for p in frontier_points]
        fy = [p[1] + 0.5 for p in frontier_points]
        ax.plot(fx, fy, ".", color="green", markersize=3, alpha=0.45)

    for uav in uavs:
        color = cmap((uav.id - 1) % 20)

        if uav.path:
            px = [uav.pos[0] + 0.5] + [p[0] + 0.5 for p in uav.path]
            py = [uav.pos[1] + 0.5] + [p[1] + 0.5 for p in uav.path]
            ax.plot(px, py, "--", color=color, alpha=0.5)

        if uav.history:
            hx = [p[0] + 0.5 for p in uav.history]
            hy = [p[1] + 0.5 for p in uav.history]
            ax.plot(hx, hy, color=color, linewidth=1.5, alpha=0.8)

        ax.add_patch(
            plt.Circle(
                (uav.pos[0] + 0.5, uav.pos[1] + 0.5),
                uav.coverage_radius,
                color=color,
                alpha=0.08,
            )
        )
        ax.add_patch(
            plt.Circle(
                (uav.pos[0] + 0.5, uav.pos[1] + 0.5),
                uav.safe_radius,
                color=color,
                alpha=0.15,
            )
        )
        ax.plot(uav.pos[0] + 0.5, uav.pos[1] + 0.5, "o", color=color, markersize=5)
        ax.text(uav.pos[0] + 1.0, uav.pos[1] + 1.0, f"UAV{uav.id}", fontsize=9)

    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title(
        f"Multi-UAV Frontier Exploration - {phase} | "
        f"step={step_idx} known={known_ratio * 100:.1f}% "
        f"uavs={len(uavs)} comm={world.config.comm_range}"
    )


def plan_component_exploration(
    component: list[int],
    uavs: list[MultiUAV],
    known_maps: list[KnownMap],
    config: MultiExplorerConfig,
) -> int:
    """Assign component frontiers and plan one target for each UAV."""
    component_map = known_maps[component[0]]
    clusters = FrontierDetector(component_map).cluster_frontiers()
    assignments = assign_clusters_to_uavs(
        clusters=clusters,
        uavs=uavs,
        component=component,
        balance_weight=config.frontier_balance_weight,
    )

    planned_count = 0
    for idx in component:
        uav = uavs[idx]
        assigned_clusters = assignments[uav.id]
        uav.assigned_cluster_count = len(assigned_clusters)

        planner = ViewpointPlanner(known_maps[idx], config.coverage_radius)
        target, path, _ = planner.select_next_viewpoint(uav.pos, assigned_clusters)

        if not path and assigned_clusters != clusters:
            target, path, _ = planner.select_next_viewpoint(uav.pos, clusters)

        uav.set_plan(target, path)
        if path:
            planned_count += 1

    return planned_count


def run_simulation(
    config: MultiExplorerConfig,
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
        )
        for i, start in enumerate(starts)
    ]
    known_maps = [KnownMap(world) for _ in uavs]

    for uav, known_map in zip(uavs, known_maps):
        known_map.update_with_sensor(uav.pos)

    fig = ax = None
    if show:
        plt.ion()
        fig, ax = plt.subplots(figsize=(9, 8))

    phase = "explore"
    final_step = 0
    final_known_ratio = global_known_ratio(known_maps, config.known_strip_width)
    exploration_finished_step = None
    returned_home = False

    for step_idx in range(config.max_steps):
        final_step = step_idx

        components = communication_components(uavs, config.comm_range)
        for component in components:
            merge_known_maps_for_component(component, known_maps)

        known_ratio = global_known_ratio(known_maps, config.known_strip_width)
        final_known_ratio = known_ratio

        if phase == "explore" and known_ratio >= config.stop_known_ratio:
            phase = "return_home"
            exploration_finished_step = step_idx

        if phase == "explore":
            planned_count = 0
            for component in components:
                planned_count += plan_component_exploration(component, uavs, known_maps, config)

            if planned_count == 0:
                phase = "return_home"
                exploration_finished_step = step_idx

        if phase == "return_home":
            for uav, known_map in zip(uavs, known_maps):
                if uav.at_start():
                    uav.set_plan(uav.start, [])
                    continue
                if not uav.path or uav.target != uav.start:
                    return_path = astar(uav.pos, uav.start, known_map.as_planning_grid())
                    uav.set_plan(uav.start, return_path)

            if all(uav.at_start() for uav in uavs):
                returned_home = True
                if show and ax is not None:
                    render_multi_state(ax, world, known_maps, uavs, step_idx, known_ratio, phase)
                    plt.pause(config.render_pause)
                break

        movable_uavs = [uav for uav in uavs if uav.path]
        if not movable_uavs:
            if show and ax is not None:
                render_multi_state(ax, world, known_maps, uavs, step_idx, known_ratio, phase)
                plt.pause(config.render_pause)
            break

        for uav in movable_uavs:
            uav.step()

        for uav, known_map in zip(uavs, known_maps):
            known_map.update_with_sensor(uav.pos)

        if show and ax is not None:
            render_multi_state(ax, world, known_maps, uavs, step_idx, known_ratio, phase)
            plt.pause(config.render_pause)

    if show:
        plt.ioff()
        plt.show()

    return {
        "steps": final_step,
        "known_ratio": final_known_ratio,
        "path_length_total": sum(len(uav.history) for uav in uavs),
        "exploration_finished_step": exploration_finished_step,
        "returned_home": returned_home,
        "num_uavs": len(uavs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-UAV frontier exploration demo.")
    parser.add_argument("--no-show", action="store_true", help="run without opening matplotlib window")
    parser.add_argument("--seed", type=int, default=MultiExplorerConfig.random_seed)
    parser.add_argument("--num-uavs", type=int, default=MultiExplorerConfig.num_uavs)
    parser.add_argument("--comm-range", type=float, default=MultiExplorerConfig.comm_range)
    parser.add_argument(
        "--coverage-radius",
        "--sensor-radius",
        type=float,
        default=MultiExplorerConfig.coverage_radius,
    )
    parser.add_argument("--safe-radius", type=float, default=MultiExplorerConfig.safe_radius)
    parser.add_argument("--inflation-radius", type=float, default=MultiExplorerConfig.inflation_radius)
    parser.add_argument("--max-steps", type=int, default=MultiExplorerConfig.max_steps)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = MultiExplorerConfig(
        random_seed=args.seed,
        num_uavs=args.num_uavs,
        comm_range=args.comm_range,
        coverage_radius=args.coverage_radius,
        safe_radius=args.safe_radius,
        inflation_radius=args.inflation_radius,
        max_steps=args.max_steps,
    )
    result = run_simulation(cfg, show=not args.no_show)
    print(
        "Simulation finished: "
        f"steps={result['steps']}, "
        f"known_ratio={result['known_ratio']:.2%}, "
        f"num_uavs={result['num_uavs']}, "
        f"path_length_total={result['path_length_total']}, "
        f"exploration_finished_step={result['exploration_finished_step']}, "
        f"returned_home={result['returned_home']}"
    )
