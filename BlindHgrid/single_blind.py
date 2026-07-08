"""
Version 1: single-UAV unknown-environment frontier exploration.

Simulation pipeline:
    true_map -> limited sensor -> known_map -> frontier clusters
    -> viewpoint -> A* on known free space -> rolling exploration

Map convention:
    width = 53, height = 50.
    x = 0..2 is the initially known left strip.
    x = 3..52 is initially unknown and explored online.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import deque
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from a_star import astar


UNKNOWN = -1
FREE = 0
OBSTACLE = 1


@dataclass
class ExplorerConfig:
    width: int = 53
    height: int = 50
    known_strip_width: int = 3
    coverage_radius: float = 3.0
    safe_radius: float = 0.8
    inflation_radius: float = 0.8
    max_steps: int = 1000
    obstacle_count: int = 34
    random_seed: int = 42
    replan_every_step: bool = True
    goal_reach_radius: float = 0.5
    stop_known_ratio: float = 0.985
    render_pause: float = 0.02


class GridWorld:
    """Ground-truth map owned by the simulator, not by the UAV."""

    def __init__(self, config: ExplorerConfig):
        self.config = config
        self.width = config.width
        self.height = config.height
        self.raw_obstacle_map = self._generate_raw_obstacle_map()
        self.true_map = self._inflate_obstacles(self.raw_obstacle_map, config.inflation_radius)

    def _generate_raw_obstacle_map(self) -> np.ndarray:
        rng = random.Random(self.config.random_seed)
        grid = np.zeros((self.height, self.width), dtype=np.int8)

        # Keep the initial known strip obstacle-free.
        blocked_until_x = self.config.known_strip_width

        # Deterministic larger obstacles in the unknown area. They make the
        # frontier behavior visible without closing the whole map.
        rectangles = [
            (11, 6, 5, 12),
            (18, 31, 4, 11),
            (28, 8, 6, 8),
            (36, 25, 5, 15),
            (45, 5, 3, 10),
        ]
        for rx, ry, rw, rh in rectangles:
            grid[ry : ry + rh, rx : rx + rw] = OBSTACLE

        # Add small random blocks only to the initially unknown 50 x 50 area.
        attempts = 0
        while int(np.sum(grid == OBSTACLE)) < self.config.obstacle_count * 8 and attempts < 500:
            attempts += 1
            w = rng.randint(2, 4)
            h = rng.randint(2, 5)
            x = rng.randint(blocked_until_x + 2, self.width - w - 1)
            y = rng.randint(1, self.height - h - 1)

            # Leave several horizontal gaps so A* usually has choices.
            if 21 <= y <= 28 and rng.random() < 0.55:
                continue

            grid[y : y + h, x : x + w] = OBSTACLE

        grid[:, : self.config.known_strip_width] = FREE
        return grid

    def _inflate_obstacles(self, raw_grid: np.ndarray, inflation_radius: float) -> np.ndarray:
        if inflation_radius <= 0:
            return raw_grid.copy()

        inflated = raw_grid.copy()
        obstacle_y, obstacle_x = np.where(raw_grid == OBSTACLE)
        r_int = int(math.ceil(inflation_radius))

        for ox, oy in zip(obstacle_x, obstacle_y):
            for y in range(max(0, oy - r_int), min(self.height, oy + r_int + 1)):
                for x in range(max(0, ox - r_int), min(self.width, ox + r_int + 1)):
                    if math.hypot(x - ox, y - oy) <= inflation_radius:
                        inflated[y, x] = OBSTACLE

        inflated[:, : self.config.known_strip_width] = FREE
        return inflated


class KnownMap:
    """The UAV's map. Unknown space is never used as free space for planning."""

    def __init__(self, world: GridWorld):
        self.world = world
        self.config = world.config
        self.grid = np.full((world.height, world.width), UNKNOWN, dtype=np.int8)
        self.grid[:, : self.config.known_strip_width] = world.true_map[
            :, : self.config.known_strip_width
        ]

    def update_with_sensor(self, position: tuple[int, int]) -> int:
        """Reveal ground-truth cells inside a circular range around the UAV."""
        cx, cy = position
        r = float(self.config.coverage_radius)
        r_int = int(math.ceil(r))
        newly_known = 0

        for y in range(max(0, cy - r_int), min(self.world.height, cy + r_int + 1)):
            for x in range(max(0, cx - r_int), min(self.world.width, cx + r_int + 1)):
                if math.hypot(x - cx, y - cy) > r:
                    continue
                if self.grid[y, x] == UNKNOWN:
                    newly_known += 1
                self.grid[y, x] = self.world.true_map[y, x]

        return newly_known

    def known_ratio_in_unknown_region(self) -> float:
        region = self.grid[:, self.config.known_strip_width :]
        return float(np.mean(region != UNKNOWN))

    def as_planning_grid(self) -> list[list[int]]:
        """A* input: unknown cells are treated as occupied for safety."""
        planning = np.where(self.grid == FREE, FREE, OBSTACLE)
        return planning.tolist()


class FrontierDetector:
    """Extract and cluster frontier cells from the current known map."""

    def __init__(self, known_map: KnownMap):
        self.known_map = known_map
        self.width = known_map.world.width
        self.height = known_map.world.height

    def detect_frontier_cells(self) -> set[tuple[int, int]]:
        frontiers = set()
        grid = self.known_map.grid

        for y in range(self.height):
            for x in range(self.width):
                if grid[y, x] != FREE:
                    continue
                for nx, ny in four_neighbors(x, y):
                    if self.in_bounds(nx, ny) and grid[ny, nx] == UNKNOWN:
                        frontiers.add((x, y))
                        break

        return frontiers

    def cluster_frontiers(self) -> list[list[tuple[int, int]]]:
        frontiers = self.detect_frontier_cells()
        clusters = []

        while frontiers:
            seed = frontiers.pop()
            cluster = [seed]
            queue = deque([seed])

            while queue:
                x, y = queue.popleft()
                for nx, ny in eight_neighbors(x, y):
                    if (nx, ny) in frontiers:
                        frontiers.remove((nx, ny))
                        queue.append((nx, ny))
                        cluster.append((nx, ny))

            clusters.append(cluster)

        clusters.sort(key=len, reverse=True)
        return clusters

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height


class ViewpointPlanner:
    """Choose reachable known-free viewpoints for frontier clusters."""

    def __init__(self, known_map: KnownMap, coverage_radius: float):
        self.known_map = known_map
        self.coverage_radius = coverage_radius

    def select_next_viewpoint(
        self,
        current: tuple[int, int],
        clusters: list[list[tuple[int, int]]],
    ) -> tuple[tuple[int, int] | None, list[tuple[int, int]], int]:
        planning_grid = self.known_map.as_planning_grid()
        best_viewpoint = None
        best_path = []
        best_cluster_size = 0
        best_score = float("inf")

        for cluster in clusters:
            viewpoint = self._cluster_viewpoint(current, cluster)
            if viewpoint is None:
                continue

            path = astar(current, viewpoint, planning_grid)
            if not path:
                continue

            # Prefer high-information clusters, but keep travel distance sane.
            score = len(path) - 0.45 * len(cluster)
            if score < best_score:
                best_score = score
                best_viewpoint = viewpoint
                best_path = path
                best_cluster_size = len(cluster)

        return best_viewpoint, best_path, best_cluster_size

    def _cluster_viewpoint(
        self,
        current: tuple[int, int],
        cluster: list[tuple[int, int]],
    ) -> tuple[int, int] | None:
        grid = self.known_map.grid
        cx = sum(p[0] for p in cluster) / len(cluster)
        cy = sum(p[1] for p in cluster) / len(cluster)

        candidates = []
        for x, y in cluster:
            if grid[y, x] != FREE:
                continue
            # Frontier cells are already known free and adjacent to unknown.
            info_gain = self._unknown_cells_seen_from(x, y)
            if info_gain == 0:
                continue
            dist_to_current = math.hypot(x - current[0], y - current[1])
            dist_to_center = math.hypot(x - cx, y - cy)
            score = -2.0 * info_gain + 0.35 * dist_to_current + 0.1 * dist_to_center
            candidates.append((score, x, y))

        if not candidates:
            return None

        candidates.sort()
        _, x, y = candidates[0]
        return (x, y)

    def _unknown_cells_seen_from(self, x0: int, y0: int) -> int:
        grid = self.known_map.grid
        r = float(self.coverage_radius)
        r_int = int(math.ceil(r))
        count = 0

        for y in range(max(0, y0 - r_int), min(grid.shape[0], y0 + r_int + 1)):
            for x in range(max(0, x0 - r_int), min(grid.shape[1], x0 + r_int + 1)):
                if math.hypot(x - x0, y - y0) <= r and grid[y, x] == UNKNOWN:
                    count += 1

        return count


class SingleUAV:
    def __init__(
        self,
        start: tuple[int, int],
        safe_radius: float,
        coverage_radius: float,
        max_speed_cells: int = 1,
    ):
        self.pos = start
        self.safe_radius = safe_radius
        self.coverage_radius = coverage_radius
        self.max_speed_cells = max_speed_cells
        self.path: list[tuple[int, int]] = []
        self.history = [start]
        self.target: tuple[int, int] | None = None

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


def four_neighbors(x: int, y: int) -> list[tuple[int, int]]:
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def eight_neighbors(x: int, y: int) -> list[tuple[int, int]]:
    return [
        (x + 1, y),
        (x - 1, y),
        (x, y + 1),
        (x, y - 1),
        (x + 1, y + 1),
        (x + 1, y - 1),
        (x - 1, y + 1),
        (x - 1, y - 1),
    ]


def render_state(
    ax,
    world: GridWorld,
    known_map: KnownMap,
    uav: SingleUAV,
    clusters: list[list[tuple[int, int]]],
    step_idx: int,
    known_ratio: float,
    cluster_size: int,
    phase: str,
) -> None:
    ax.clear()
    cmap = plt.get_cmap("tab20")
    color = cmap(0)

    display = np.zeros((world.height, world.width, 3), dtype=float)
    display[:, :] = [0.90, 0.90, 0.90]          # unknown
    display[known_map.grid == FREE] = [1.00, 1.00, 1.00]
    display[known_map.grid == OBSTACLE] = [0.50, 0.50, 0.50]

    ax.imshow(display, origin="lower", extent=[0, world.width, 0, world.height])

    hidden_obstacles = (world.raw_obstacle_map == OBSTACLE) & (known_map.grid == UNKNOWN)
    hy, hx = np.where(hidden_obstacles)
    if len(hx) > 0:
        ax.scatter(hx + 0.5, hy + 0.5, s=6, c="gray", alpha=0.18, marker="s")

    inflated_only = (
        (world.true_map == OBSTACLE)
        & (world.raw_obstacle_map != OBSTACLE)
        & (known_map.grid == OBSTACLE)
    )
    iy, ix = np.where(inflated_only)
    if len(ix) > 0:
        ax.scatter(ix + 0.5, iy + 0.5, s=5, c="gray", alpha=0.22, marker="s")

    ax.axvspan(0, world.config.known_strip_width, color=color, alpha=0.08)

    if clusters:
        frontier_points = [cell for cluster in clusters for cell in cluster]
        fx = [p[0] + 0.5 for p in frontier_points]
        fy = [p[1] + 0.5 for p in frontier_points]
        ax.plot(fx, fy, ".", color="green", markersize=3, alpha=0.45)

    if uav.path:
        px = [uav.pos[0] + 0.5] + [p[0] + 0.5 for p in uav.path]
        py = [uav.pos[1] + 0.5] + [p[1] + 0.5 for p in uav.path]
        ax.plot(px, py, "--", color=color, alpha=0.5)

    if uav.history:
        hx = [p[0] + 0.5 for p in uav.history]
        hy = [p[1] + 0.5 for p in uav.history]
        ax.plot(hx, hy, color=color, linewidth=1.5, alpha=0.8)

    # Draw camera/sensor coverage range.
    ax.add_patch(
        plt.Circle(
            (uav.pos[0] + 0.5, uav.pos[1] + 0.5),
            uav.coverage_radius,
            color=color,
            alpha=0.08,
        )
    )

    # Draw UAV safety radius.
    ax.add_patch(
        plt.Circle(
            (uav.pos[0] + 0.5, uav.pos[1] + 0.5),
            uav.safe_radius,
            color=color,
            alpha=0.15,
        )
    )

    ax.plot(uav.pos[0] + 0.5, uav.pos[1] + 0.5, "o", color=color, markersize=5)
    ax.text(uav.pos[0] + 1.0, uav.pos[1] + 1.0, "UAV1", fontsize=9)

    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title(
        f"Single UAV Frontier Exploration - {phase} | "
        f"step={step_idx} known={known_ratio * 100:.1f}% "
        f"clusters={len(clusters)} target_cluster={cluster_size} | "
        f"coverage={uav.coverage_radius}, safe={uav.safe_radius}, inflation={world.config.inflation_radius}"
    )


def run_simulation(config: ExplorerConfig, show: bool = True) -> dict[str, float | int | bool | None]:
    world = GridWorld(config)
    known_map = KnownMap(world)
    detector = FrontierDetector(known_map)
    viewpoint_planner = ViewpointPlanner(known_map, config.coverage_radius)

    start = (1, config.height // 2)
    uav = SingleUAV(
        start=start,
        safe_radius=config.safe_radius,
        coverage_radius=config.coverage_radius,
    )
    known_map.update_with_sensor(uav.pos)

    fig = ax = None
    if show:
        plt.ion()
        fig, ax = plt.subplots(figsize=(9, 8))

    final_step = 0
    final_clusters = 0
    final_known_ratio = known_map.known_ratio_in_unknown_region()
    phase = "explore"
    exploration_finished_step = None
    returned_home = False

    for step_idx in range(config.max_steps):
        final_step = step_idx
        clusters = detector.cluster_frontiers()
        final_clusters = len(clusters)
        known_ratio = known_map.known_ratio_in_unknown_region()
        final_known_ratio = known_ratio

        if phase == "explore" and (not clusters or known_ratio >= config.stop_known_ratio):
            phase = "return_home"
            exploration_finished_step = step_idx
            return_path = astar(uav.pos, start, known_map.as_planning_grid())
            uav.set_plan(start, return_path)

        if phase == "explore" and (config.replan_every_step or not uav.path):
            target, path, cluster_size = viewpoint_planner.select_next_viewpoint(uav.pos, clusters)
            uav.set_plan(target, path)
        else:
            cluster_size = 0

        if phase == "explore" and not uav.path:
            # No reachable frontier remains in the known free map; return home.
            phase = "return_home"
            exploration_finished_step = step_idx
            return_path = astar(uav.pos, start, known_map.as_planning_grid())
            uav.set_plan(start, return_path)
            cluster_size = 0

        if phase == "return_home":
            uav.target = start
            if uav.pos == start:
                returned_home = True
                if show and ax is not None:
                    render_state(ax, world, known_map, uav, clusters, step_idx, known_ratio, cluster_size, phase)
                    plt.pause(config.render_pause)
                break
            if not uav.path:
                return_path = astar(uav.pos, start, known_map.as_planning_grid())
                uav.set_plan(start, return_path)

        if not uav.path:
            if show and ax is not None:
                render_state(ax, world, known_map, uav, clusters, step_idx, known_ratio, cluster_size, phase)
                plt.pause(config.render_pause)
            break

        uav.step()
        known_map.update_with_sensor(uav.pos)

        if show and ax is not None:
            render_state(ax, world, known_map, uav, clusters, step_idx, known_ratio, cluster_size, phase)
            plt.pause(config.render_pause)

    if show:
        plt.ioff()
        plt.show()

    return {
        "steps": final_step,
        "known_ratio": final_known_ratio,
        "remaining_frontier_clusters": final_clusters,
        "path_length": len(uav.history),
        "exploration_finished_step": exploration_finished_step,
        "returned_home": returned_home,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-UAV frontier exploration demo.")
    parser.add_argument("--no-show", action="store_true", help="run without opening matplotlib window")
    parser.add_argument("--seed", type=int, default=ExplorerConfig.random_seed)
    parser.add_argument(
        "--coverage-radius",
        "--sensor-radius",
        type=float,
        default=ExplorerConfig.coverage_radius,
    )
    parser.add_argument("--safe-radius", type=float, default=ExplorerConfig.safe_radius)
    parser.add_argument("--inflation-radius", type=float, default=ExplorerConfig.inflation_radius)
    parser.add_argument("--max-steps", type=int, default=ExplorerConfig.max_steps)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = ExplorerConfig(
        random_seed=args.seed,
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
        f"remaining_frontier_clusters={result['remaining_frontier_clusters']}, "
        f"path_length={result['path_length']}, "
        f"exploration_finished_step={result['exploration_finished_step']}, "
        f"returned_home={result['returned_home']}"
    )
