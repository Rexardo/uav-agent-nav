"""plan_env-style map, sensing, communication, and occupancy-grid helpers."""

from __future__ import annotations

import math
import random

import numpy as np

from racer_types import FREE, OBSTACLE, UNKNOWN, RACERConfig, UAV

class GridWorld:
    """Random obstacle map using the same circle/rectangle style as hgrid.py."""

    def __init__(self, config: RACERConfig):
        self.config = config
        self.width = config.width
        self.height = config.height
        self.circles, self.rectangles = self._generate_obstacles()
        self.raw_obstacle_map = self._rasterize(inflation_radius=0.0)
        self.true_map = self._rasterize(config.obstacle_inflation_cells())
        self._clear_launch_area(self.raw_obstacle_map)
        self._clear_launch_area(self.true_map)

    def _generate_obstacles(self) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float, float]]]:
        rng = random.Random(self.config.random_seed)
        circles: list[tuple[float, float, float]] = []
        rectangles: list[tuple[float, float, float, float]] = []
        attempts = 0
        safe_zone = max(6, self.config.known_strip_width + 2)

        while len(circles) + len(rectangles) < self.config.obstacle_count and attempts < self.config.obstacle_count * 25:
            attempts += 1
            if rng.random() < 0.5:
                radius = rng.uniform(1.0, 2.8)
                cx = rng.uniform(safe_zone + radius, self.width - radius - 1)
                cy = rng.uniform(radius, self.height - radius - 1)
                circles.append((cx, cy, radius))
            else:
                rw = rng.uniform(2.0, 5.5)
                rh = rng.uniform(2.0, 5.5)
                rx = rng.uniform(safe_zone, self.width - rw - 1)
                ry = rng.uniform(0, self.height - rh - 1)
                rectangles.append((rx, ry, rw, rh))

        return circles, rectangles

    def _rasterize(self, inflation_radius: float) -> np.ndarray:
        grid = np.zeros((self.height, self.width), dtype=np.int8)
        for y in range(self.height):
            for x in range(self.width):
                occupied = False
                for cx, cy, radius in self.circles:
                    if math.hypot(x - cx, y - cy) <= radius + inflation_radius:
                        occupied = True
                        break
                if not occupied:
                    for rx, ry, rw, rh in self.rectangles:
                        if rx - inflation_radius <= x <= rx + rw + inflation_radius and ry - inflation_radius <= y <= ry + rh + inflation_radius:
                            occupied = True
                            break
                if occupied:
                    grid[y, x] = OBSTACLE
        return grid

    def _clear_launch_area(self, grid: np.ndarray) -> None:
        grid[:, : self.config.known_strip_width] = FREE

    def point_collides_raw_obstacle(self, point: tuple[float, float], margin: float = 0.0) -> bool:
        x, y = point
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        for cx, cy, radius in self.circles:
            if math.hypot(x - cx, y - cy) <= radius + margin:
                return True
        for rx, ry, rw, rh in self.rectangles:
            if rx - margin <= x <= rx + rw + margin and ry - margin <= y <= ry + rh + margin:
                return True
        return False

    def segment_collides_raw_obstacle(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        margin: float = 0.0,
        step: float = 0.10,
    ) -> bool:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = max(1e-6, math.hypot(dx, dy))
        sample_num = max(1, int(math.ceil(length / max(1e-6, step))))
        for i in range(sample_num + 1):
            alpha = i / sample_num
            point = (start[0] + alpha * dx, start[1] + alpha * dy)
            if self.point_collides_raw_obstacle(point, margin):
                return True
        return False


class KnownMap:
    def __init__(self, world: GridWorld):
        self.world = world
        self.grid = np.full((world.height, world.width), UNKNOWN, dtype=np.int8)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.world.width and 0 <= y < self.world.height


def make_column_starts(config: RACERConfig) -> list[tuple[int, int]]:
    x = max(0, min(config.known_strip_width - 2, 1))
    margin = max(3, int(math.ceil(config.coverage_radius + config.safe_radius)))
    if config.num_uavs == 1:
        return [(x, config.height // 2)]
    ys = np.linspace(margin, config.height - margin - 1, config.num_uavs)
    return [(x, int(round(y))) for y in ys]


def communication_components(uavs: list[UAV], comm_range: float) -> list[list[int]]:
    remaining = set(range(len(uavs)))
    components: list[list[int]] = []
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


def explorable_known_ratio(known_maps: list[KnownMap], raw_obstacle_map: np.ndarray, known_strip_width: int) -> float:
    merged = union_known_grid(known_maps)
    region = merged[:, known_strip_width:]
    explorable = raw_obstacle_map[:, known_strip_width:] == FREE
    return float(np.mean(region[explorable] != UNKNOWN)) if np.any(explorable) else 1.0


def update_known_map_with_sensor(known_map: KnownMap, uav: UAV, config: RACERConfig) -> int:
    newly_known = 0
    ux, uy = uav.pos
    radius = int(math.ceil(config.coverage_radius))
    for y in range(max(0, uy - radius), min(known_map.world.height, uy + radius + 1)):
        for x in range(max(0, ux - radius), min(known_map.world.width, ux + radius + 1)):
            if math.hypot(x - ux, y - uy) > config.coverage_radius:
                continue
            if not line_of_sight(known_map.world.raw_obstacle_map, (ux, uy), (x, y)):
                continue
            if known_map.grid[y, x] == UNKNOWN:
                newly_known += 1
            known_map.grid[y, x] = known_map.world.raw_obstacle_map[y, x]
    return newly_known


def line_of_sight(grid: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> bool:
    for x, y in bresenham(start, end):
        if (x, y) == start or (x, y) == end:
            continue
        if grid[y, x] == OBSTACLE:
            return False
    return True


def bresenham(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    x0, y0 = start
    x1, y1 = end
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return points
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def planning_grid_from_known_map(
    known_map: KnownMap,
    config: RACERConfig,
    dynamic_obstacles: set[tuple[int, int]] | None = None,
    block_unknown: bool | None = None,
) -> np.ndarray:
    grid = np.zeros_like(known_map.grid, dtype=np.int8)

    # Match RACER's separation of concerns: inflate known occupied cells, then
    # let the search layer decide whether unknown cells are traversable. Unknown
    # space should not be inflated like a physical obstacle, otherwise frontiers
    # are artificially sealed off and a UAV can get stuck at the exploration
    # boundary.
    grid[known_map.grid == OBSTACLE] = OBSTACLE
    obstacle_clearance = max(config.planning_inflation_cells(), config.manager_clearance_threshold_cells())
    radius = int(math.ceil(obstacle_clearance))
    obstacle_cells = np.argwhere(grid == OBSTACLE)
    for y, x in obstacle_cells:
        for yy in range(max(0, y - radius), min(grid.shape[0], y + radius + 1)):
            for xx in range(max(0, x - radius), min(grid.shape[1], x + radius + 1)):
                if math.hypot(xx - x, yy - y) <= obstacle_clearance:
                    grid[yy, xx] = OBSTACLE

    if block_unknown is None:
        block_unknown = not config.search_optimistic
    if block_unknown:
        grid[known_map.grid == UNKNOWN] = OBSTACLE

    if dynamic_obstacles:
        dyn_radius = int(math.ceil(config.dynamic_obstacle_inflation))
        for x, y in dynamic_obstacles:
            for yy in range(max(0, y - dyn_radius), min(grid.shape[0], y + dyn_radius + 1)):
                for xx in range(max(0, x - dyn_radius), min(grid.shape[1], x + dyn_radius + 1)):
                    if math.hypot(xx - x, yy - y) <= config.dynamic_obstacle_inflation:
                        grid[yy, xx] = OBSTACLE
    return grid


def planning_grid_for_uav(
    known_map: KnownMap,
    config: RACERConfig,
    uav: UAV,
    dynamic_obstacles: set[tuple[int, int]] | None = None,
    block_unknown: bool | None = None,
) -> np.ndarray:
    grid = planning_grid_from_known_map(known_map, config, dynamic_obstacles, block_unknown)
    for x, y in getattr(uav, "local_blocked_cells", set()):
        if 0 <= x < known_map.world.width and 0 <= y < known_map.world.height:
            grid[y, x] = OBSTACLE
    return grid


def four_neighbors(x: int, y: int) -> list[tuple[int, int]]:
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def eight_neighbors(x: int, y: int) -> list[tuple[int, int]]:
    return [(x + dx, y + dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]
