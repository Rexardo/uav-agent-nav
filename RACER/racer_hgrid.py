"""active_perception/hgrid-style online decomposition and pairwise allocation."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from racer_map import KnownMap, union_known_grid
from racer_types import FREE, UNKNOWN, RACERConfig, UAV

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
    """Online hgrid over unexplored space."""

    def __init__(self, config: RACERConfig):
        self.config = config
        self.level_sizes = tuple(sorted(set(config.hgrid_level_sizes), reverse=True)) or (config.hgrid_level_sizes[0],)
        self.next_block_id = 0
        self.blocks = self._make_initial_blocks()
        self.raw_obstacle_map: np.ndarray | None = None

    def _make_initial_blocks(self) -> list[HGridBlock]:
        blocks = []
        size = self.level_sizes[0]
        for y_min in range(0, self.config.height, size):
            for x_min in range(self.config.known_strip_width, self.config.width, size):
                blocks.append(self._new_block(x_min, y_min, min(self.config.width, x_min + size), min(self.config.height, y_min + size), 0))
        return blocks

    def _new_block(self, x_min: int, y_min: int, x_max: int, y_max: int, level: int, parent_id: int | None = None, owner_id: int | None = None) -> HGridBlock:
        block = HGridBlock(self.next_block_id, x_min, y_min, x_max, y_max, level, parent_id, owner_id)
        self.next_block_id += 1
        return block

    def can_split(self, block: HGridBlock) -> bool:
        return block.level + 1 < len(self.level_sizes)

    def split_block(self, block: HGridBlock, known_grid: np.ndarray) -> list[HGridBlock]:
        child_size = self.level_sizes[block.level + 1]
        children = []
        for y_min in range(block.y_min, block.y_max, child_size):
            for x_min in range(block.x_min, block.x_max, child_size):
                child = self._new_block(
                    x_min,
                    y_min,
                    min(block.x_max, x_min + child_size),
                    min(block.y_max, y_min + child_size),
                    block.level + 1,
                    block.block_id,
                    block.owner_id,
                )
                child.unknown_cells = hgrid_block_unknown_work(child, known_grid, self.raw_obstacle_map)
                if child.unknown_cells > 0:
                    children.append(child)
        return children

    def update_active_cells(self, known_grid: np.ndarray, raw_obstacle_map: np.ndarray | None) -> tuple[int, int]:
        self.raw_obstacle_map = raw_obstacle_map
        updated: list[HGridBlock] = []
        split_count = 0
        removed_count = 0
        for block in self.blocks:
            block.unknown_cells = hgrid_block_unknown_work(block, known_grid, raw_obstacle_map)
            if block.unknown_cells <= 0:
                removed_count += 1
                continue
            known_ratio = 1.0 - block.unknown_cells / max(1, block.area())
            if self.can_split(block) and known_ratio >= self.config.hgrid_split_known_ratio:
                children = self.split_block(block, known_grid)
                if children:
                    updated.extend(children)
                    split_count += 1
                else:
                    removed_count += 1
                continue
            if not self.can_split(block) and block.unknown_cells <= self.config.hgrid_min_unknown_cells:
                removed_count += 1
                continue
            updated.append(block)
        self.blocks = updated
        return split_count, removed_count

    def assign_initial_owners(self, uavs: list[UAV]) -> None:
        load = {uav.id: 0 for uav in uavs}
        for block in self.blocks:
            cx, cy = block.center()
            owner = min(uavs, key=lambda uav: (math.hypot(uav.start[0] - cx, uav.start[1] - cy), load[uav.id]))
            block.owner_id = owner.id
            load[owner.id] += block.area()

    def blocks_for_uav(self, uav_id: int) -> list[HGridBlock]:
        return [block for block in self.blocks if block.owner_id == uav_id]

    def cluster_inside_owned_blocks(self, cluster: list[tuple[int, int]], owned_blocks: list[HGridBlock]) -> bool:
        return any(block.contains(x, y) for x, y in cluster for block in owned_blocks)

    def prune_blocks_without_frontiers(self, frontier_cells: set[tuple[int, int]]) -> int:
        if not frontier_cells:
            removed = len(self.blocks)
            self.blocks = []
            return removed
        kept = [block for block in self.blocks if any(block.contains(x, y) for x, y in frontier_cells)]
        removed = len(self.blocks) - len(kept)
        self.blocks = kept
        return removed


def hgrid_block_unknown_work(block: HGridBlock, known_grid: np.ndarray, raw_obstacle_map: np.ndarray | None) -> int:
    region = known_grid[block.y_min : block.y_max, block.x_min : block.x_max]
    unknown = region == UNKNOWN
    if raw_obstacle_map is not None:
        raw_region = raw_obstacle_map[block.y_min : block.y_max, block.x_min : block.x_max]
        unknown &= raw_region == FREE
    return int(np.sum(unknown))


def generate_coverage_path_for_blocks(blocks: list[HGridBlock], step: int, start_pos: tuple[int, int] | None = None) -> list[tuple[int, int]]:
    if not blocks:
        return []
    step = max(1, step)
    remaining = list(blocks)
    ordered: list[HGridBlock] = []
    current = (float(start_pos[0]), float(start_pos[1])) if start_pos else blocks[0].center()
    while remaining:
        idx = min(range(len(remaining)), key=lambda i: math.hypot(current[0] - remaining[i].center()[0], current[1] - remaining[i].center()[1]))
        block = remaining.pop(idx)
        ordered.append(block)
        current = block.center()
    waypoints = []
    for block_index, block in enumerate(ordered):
        ys = list(range(block.y_min, block.y_max, step))
        if not ys or ys[-1] != block.y_max - 1:
            ys.append(block.y_max - 1)
        for row, y in enumerate(ys):
            xs = list(range(block.x_min, block.x_max, step))
            if not xs or xs[-1] != block.x_max - 1:
                xs.append(block.x_max - 1)
            if (block_index + row) % 2:
                xs.reverse()
            waypoints.extend((x, y) for x in xs)
    return waypoints


def block_route_length(start: tuple[int, int], blocks: list[HGridBlock], step: int) -> float:
    route = generate_coverage_path_for_blocks(blocks, step, start)
    if not route:
        return 0.0
    total = math.hypot(start[0] - route[0][0], start[1] - route[0][1])
    for a, b in zip(route, route[1:]):
        total += math.hypot(a[0] - b[0], a[1] - b[1])
    return total


def cvrp_like_pair_partition(uav_a: UAV, uav_b: UAV, candidate_blocks: list[tuple[HGridBlock, int]], config: RACERConfig) -> dict[int, int]:
    blocks = [block for block, _ in candidate_blocks]
    work_by_block = {block.block_id: work for block, work in candidate_blocks}
    total_work = sum(work_by_block.values())

    def score(bits: tuple[int, ...]) -> float:
        blocks_a = [block for bit, block in zip(bits, blocks) if bit == 0]
        blocks_b = [block for bit, block in zip(bits, blocks) if bit == 1]
        work_a = sum(work_by_block[block.block_id] for block in blocks_a)
        route_a = block_route_length(uav_a.pos, blocks_a, config.cp_step)
        route_b = block_route_length(uav_b.pos, blocks_b, config.cp_step)
        return config.cvrp_route_weight * (route_a + route_b) + config.cvrp_work_weight * abs(work_a - (total_work - work_a))

    if len(blocks) <= config.cvrp_max_exact_blocks:
        best_bits = min(itertools.product((0, 1), repeat=len(blocks)), key=score)
        return {block.block_id: (uav_a.id if bit == 0 else uav_b.id) for bit, block in zip(best_bits, blocks)}

    assigned: dict[int, int] = {}
    loads = {uav_a.id: 0.0, uav_b.id: 0.0}
    for block, work in sorted(candidate_blocks, key=lambda item: (-item[1], item[0].block_id)):
        owner = min((uav_a, uav_b), key=lambda uav: block_route_length(uav.pos, [block], config.cp_step) + config.hgrid_balance_weight * (loads[uav.id] + work))
        assigned[block.block_id] = owner.id
        loads[owner.id] += work
    return assigned


def pairwise_reassign_hgrid_blocks(uav_a: UAV, uav_b: UAV, hgrid: HGrid, known_grid: np.ndarray, config: RACERConfig) -> int:
    pair_ids = {uav_a.id, uav_b.id}
    candidates = [
        (block, hgrid_block_unknown_work(block, known_grid, hgrid.raw_obstacle_map))
        for block in hgrid.blocks
        if block.owner_id in pair_ids
    ]
    candidates = [(block, work) for block, work in candidates if work > 0]
    if len(candidates) <= 1:
        return 0
    new_owners = cvrp_like_pair_partition(uav_a, uav_b, candidates, config)
    changed = 0
    for block, _ in candidates:
        if block.owner_id != new_owners[block.block_id]:
            block.owner_id = new_owners[block.block_id]
            changed += 1
    if changed:
        uav_a.cp_cursor = 0
        uav_b.cp_cursor = 0
    return changed


def pairwise_request_response_hgrid_blocks(uavs: list[UAV], known_maps: list[KnownMap], hgrid: HGrid, step_idx: int, config: RACERConfig) -> tuple[int, int]:
    total_changed = 0
    success_count = 0
    busy: set[int] = set()
    merged_grid = union_known_grid(known_maps)
    for i, uav in enumerate(uavs):
        if i in busy or step_idx - uav.hgrid_last_attempt < config.pairwise_request_cooldown:
            continue
        candidates = []
        for j, other in enumerate(uavs):
            if i == j or j in busy or uav.distance_to(other) > config.comm_range:
                continue
            if step_idx - other.hgrid_last_attempt < config.pairwise_request_cooldown:
                continue
            last_success = uav.hgrid_last_success.get(other.id, -10**9)
            if step_idx - last_success < config.pairwise_success_cooldown:
                continue
            candidates.append((last_success, j, other))
        if not candidates:
            continue
        _, j, other = min(candidates, key=lambda item: (item[0], item[2].id))
        uav.hgrid_last_attempt = step_idx
        other.hgrid_last_attempt = step_idx
        total_changed += pairwise_reassign_hgrid_blocks(uav, other, hgrid, merged_grid, config)
        uav.hgrid_last_success[other.id] = step_idx
        other.hgrid_last_success[uav.id] = step_idx
        busy.update({i, j})
        success_count += 1
    return total_changed, success_count
