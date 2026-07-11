"""plan_env-style map, sensing, communication, and occupancy-grid helpers."""

from __future__ import annotations

import math
import random

import numpy as np

from racer_types import FREE, OBSTACLE, UNKNOWN, RACERConfig, UAV


def dense_maze_layout(config: RACERConfig) -> tuple[int, int, int, int, int]:
    """Return origin, coarse dimensions, and pitch for the fixed dense maze."""
    if config.width < 40 or config.height < 40:
        raise ValueError("Map 2 requires width and height of at least 40 cells.")
    pitch = 5  # Three free cells followed by a two-cell wall.
    columns = (config.width - 2) // pitch
    rows = (config.height - 2) // pitch
    maze_width = columns * 3 + (columns - 1) * 2
    maze_height = rows * 3 + (rows - 1) * 2
    origin_x = (config.width - maze_width) // 2
    origin_y = (config.height - maze_height) // 2
    return origin_x, origin_y, columns, rows, pitch


class GridWorld:
    """Selectable obstacle map using circle/rectangle geometry."""

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
        # 地图 1：原来的随机障碍物地图。
        if self.config.map_id == 1:
            return self._generate_random_obstacles()
        # 地图 2：固定 Dense Maze。三格宽长通道、密集直角转弯、盲巷和
        # 少量环路用于测试多机在狭窄迷宫中的轨迹冲突与死锁。
        if self.config.map_id == 2:
            return self._generate_dense_maze_obstacles()
        raise ValueError(f"Unsupported map_id={self.config.map_id}; expected 1 or 2.")

    def _generate_random_obstacles(self) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float, float]]]:
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

    def _generate_dense_maze_obstacles(self) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float, float]]]:
        """Build a reproducible connected maze with four side entrances."""
        origin_x, origin_y, columns, rows, pitch = dense_maze_layout(self.config)
        grid = np.ones((self.height, self.width), dtype=np.int8)

        def chamber_origin(cell: tuple[int, int]) -> tuple[int, int]:
            column, row = cell
            return origin_x + column * pitch, origin_y + row * pitch

        def carve_chamber(cell: tuple[int, int]) -> None:
            x, y = chamber_origin(cell)
            grid[y : y + 3, x : x + 3] = FREE

        def carve_connection(first: tuple[int, int], second: tuple[int, int]) -> None:
            x1, y1 = chamber_origin(first)
            x2, y2 = chamber_origin(second)
            if x1 == x2:
                y = min(y1, y2) + 3
                grid[y : y + 2, x1 : x1 + 3] = FREE
            else:
                x = min(x1, x2) + 3
                grid[y1 : y1 + 3, x : x + 2] = FREE

        for row in range(rows):
            for column in range(columns):
                carve_chamber((column, row))

        rng = random.Random(20240711)
        visited = {(0, 0)}
        stack = [(0, 0)]
        tree_edges: set[frozenset[tuple[int, int]]] = set()
        motions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while stack:
            current = stack[-1]
            neighbors = []
            for dx, dy in motions:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < columns and 0 <= neighbor[1] < rows and neighbor not in visited:
                    neighbors.append(neighbor)
            if not neighbors:
                stack.pop()
                continue
            next_cell = rng.choice(neighbors)
            carve_connection(current, next_cell)
            tree_edges.add(frozenset((current, next_cell)))
            visited.add(next_cell)
            stack.append(next_cell)

        # Add a few deterministic loops without turning the maze into an open room.
        extra_edges = []
        for row in range(rows):
            for column in range(columns):
                current = (column, row)
                for neighbor in ((column + 1, row), (column, row + 1)):
                    if neighbor[0] >= columns or neighbor[1] >= rows:
                        continue
                    edge = frozenset((current, neighbor))
                    if edge not in tree_edges:
                        extra_edges.append((current, neighbor))
        rng.shuffle(extra_edges)
        for first, second in extra_edges[:6]:
            carve_connection(first, second)

        entrance_cells = [(0, 1), (columns - 1, 2), (0, rows - 2), (columns - 1, rows - 3)]
        for column, row in entrance_cells:
            x, y = chamber_origin((column, row))
            if column == 0:
                grid[y : y + 3, : x + 3] = FREE
            else:
                grid[y : y + 3, x:] = FREE

        rectangles: list[tuple[float, float, float, float]] = []
        active_runs: dict[tuple[int, int], tuple[int, int]] = {}
        for y in range(self.height):
            row_runs: set[tuple[int, int]] = set()
            x = 0
            while x < self.width:
                if grid[y, x] == FREE:
                    x += 1
                    continue
                start_x = x
                while x + 1 < self.width and grid[y, x + 1] == OBSTACLE:
                    x += 1
                row_runs.add((start_x, x - start_x + 1))
                x += 1

            for run, (start_y, height) in list(active_runs.items()):
                if run in row_runs:
                    active_runs[run] = (start_y, height + 1)
                    row_runs.remove(run)
                    continue
                start_x, width = run
                rectangles.append(
                    (float(start_x) - 0.5, float(start_y) - 0.5, float(width), float(height))
                )
                del active_runs[run]

            for run in row_runs:
                active_runs[run] = (y, 1)

        for (start_x, width), (start_y, height) in active_runs.items():
            rectangles.append(
                (float(start_x) - 0.5, float(start_y) - 0.5, float(width), float(height))
            )
        return [], rectangles

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
        if self.config.map_id == 2:
            return
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
    if config.map_id == 2:
        origin_x, origin_y, columns, rows, pitch = dense_maze_layout(config)
        west_y1 = origin_y + pitch + 1
        east_y1 = origin_y + 2 * pitch + 1
        west_y2 = origin_y + (rows - 2) * pitch + 1
        east_y2 = origin_y + (rows - 3) * pitch + 1
        starts = [
            (1, west_y1),
            (config.width - 2, east_y1),
            (1, west_y2),
            (config.width - 2, east_y2),
        ]
        if config.num_uavs > 4:
            raise ValueError("Map 2 is a four-UAV dense-maze scenario and supports at most 4 UAVs.")
        return starts[: config.num_uavs]

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

    return grid


def planning_grid_for_uav(
    known_map: KnownMap,
    config: RACERConfig,
    uav: UAV,
    block_unknown: bool | None = None,
) -> np.ndarray:
    grid = planning_grid_from_known_map(known_map, config, block_unknown)
    for x, y in getattr(uav, "local_blocked_cells", set()):
        if 0 <= x < known_map.world.width and 0 <= y < known_map.world.height:
            grid[y, x] = OBSTACLE
    return grid


def four_neighbors(x: int, y: int) -> list[tuple[int, int]]:
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def eight_neighbors(x: int, y: int) -> list[tuple[int, int]]:
    return [(x + dx, y + dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]
