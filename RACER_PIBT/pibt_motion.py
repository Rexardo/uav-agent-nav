"""RACER-aware local communication and PIBT-based trajectory planning layer."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

try:
    from .pibt_core import PIBTStepPlanner, PIBTStepStats, project_goal_to_reachable
    from .safe_trajectory import (
        BernsteinTrajectory,
        build_pairwise_lsc,
        minimum_jerk_bernstein,
        optimize_safe_trajectory,
        trajectories_are_separated,
    )
except ImportError:  # Support running RACER_PIBT.py directly.
    from pibt_core import PIBTStepPlanner, PIBTStepStats, project_goal_to_reachable
    from safe_trajectory import (
        BernsteinTrajectory,
        build_pairwise_lsc,
        minimum_jerk_bernstein,
        optimize_safe_trajectory,
        trajectories_are_separated,
    )

Coord = tuple[int, int]


@dataclass
class PIBTMotionConfig:
    agent_radius: float = 0.25
    max_velocity: float = 1.0
    max_acceleration: float = 2.0
    segment_duration: float = 2.0
    seed: int = 42

    def validate(self, cell_size: float, communication_range: float) -> None:
        if cell_size <= 2.0 * math.sqrt(2.0) * self.agent_radius:
            raise ValueError("PIBT cycle condition requires cell_size > 2*sqrt(2)*agent_radius")
        if communication_range <= 2.0 * cell_size:
            raise ValueError("PIBT adaptation requires communication_range > 2*cell_size")
        if self.segment_duration <= 0.0 or self.max_velocity <= 0.0 or self.max_acceleration <= 0.0:
            raise ValueError("trajectory duration and dynamic limits must be positive")


@dataclass(frozen=True)
class PIBTEpochMessage:
    epoch: int
    sender_id: int
    coordinator_id: int
    message_type: str
    position: Coord
    target: Coord
    route: tuple[int, ...]


@dataclass
class ComponentPlan:
    next_positions: list[Coord]
    projected_goals: list[Coord]
    trajectories: list[BernsteinTrajectory]
    coordinator_id: int
    messages: list[PIBTEpochMessage]
    pibt_stats: PIBTStepStats
    projected_count: int = 0
    trajectory_rejections: int = 0


class PairwisePIBTMotionPlanner:
    """Plan one local communication component at each RACER replanning epoch."""

    def __init__(self, config: PIBTMotionConfig):
        self.config = config
        self.waiting_age: dict[int, float] = {}
        self.previous_trajectories: dict[int, BernsteinTrajectory] = {}

    def plan_component(
        self,
        epoch: int,
        uav_ids: list[int],
        free_grid: np.ndarray,
        positions: list[Coord],
        targets: list[Coord],
        communication_range: float,
    ) -> ComponentPlan:
        projected_goals: list[Coord] = []
        projected_count = 0
        for start, target in zip(positions, targets):
            projected, changed = project_goal_to_reachable(free_grid, start, target)
            projected_goals.append(projected)
            projected_count += int(changed)

        priorities = []
        for uav_id, position, goal in zip(uav_ids, positions, projected_goals):
            age = self.waiting_age.get(uav_id, 0.0)
            base = (abs(position[0] - goal[0]) + abs(position[1] - goal[1])) / max(1, free_grid.size)
            priorities.append(age + base)
        coordinator_local_index = max(range(len(uav_ids)), key=lambda i: (priorities[i], -uav_ids[i]))
        coordinator_id = uav_ids[coordinator_local_index]
        routes = _routes_to_coordinator(uav_ids, positions, coordinator_id, communication_range)
        messages = [
            PIBTEpochMessage(
                epoch,
                uav_id,
                coordinator_id,
                "state_and_previous_trajectory",
                position,
                target,
                routes[uav_id],
            )
            for uav_id, position, target in zip(uav_ids, positions, projected_goals)
        ]

        pibt = PIBTStepPlanner(free_grid, projected_goals, self.config.seed + epoch * 997 + coordinator_id)
        next_positions, pibt_stats = pibt.step(positions, priorities)
        nominal_controls = [
            minimum_jerk_bernstein(np.asarray(start, dtype=float), np.asarray(end, dtype=float), self.config.segment_duration)
            for start, end in zip(positions, next_positions)
        ]
        trajectories: list[BernsteinTrajectory] = []
        trajectory_rejections = 0
        for index, (start, waypoint) in enumerate(zip(positions, next_positions)):
            constraints = build_pairwise_lsc(nominal_controls, index, self.config.agent_radius)
            trajectory = optimize_safe_trajectory(
                start,
                waypoint,
                self.config.segment_duration,
                self.config.max_velocity,
                self.config.max_acceleration,
                self.config.agent_radius,
                constraints,
            )
            if not trajectory.feasible:
                trajectory_rejections += 1
                trajectory = optimize_safe_trajectory(
                    start,
                    start,
                    self.config.segment_duration,
                    self.config.max_velocity,
                    self.config.max_acceleration,
                    self.config.agent_radius,
                    [],
                )
                next_positions[index] = start
            trajectories.append(trajectory)

        if not trajectories_are_separated(trajectories, self.config.agent_radius):
            trajectory_rejections += len(trajectories)
            next_positions = list(positions)
            trajectories = [
                optimize_safe_trajectory(
                    start,
                    start,
                    self.config.segment_duration,
                    self.config.max_velocity,
                    self.config.max_acceleration,
                    self.config.agent_radius,
                    [],
                )
                for start in positions
            ]

        for uav_id, position, next_position, goal, trajectory in zip(
            uav_ids, positions, next_positions, projected_goals, trajectories
        ):
            if next_position == goal:
                self.waiting_age[uav_id] = self.waiting_age.get(uav_id, 0.0) % 1.0
            elif next_position == position:
                self.waiting_age[uav_id] = self.waiting_age.get(uav_id, 0.0) + 1.0
            else:
                self.waiting_age[uav_id] = self.waiting_age.get(uav_id, 0.0) + 1.0
            self.previous_trajectories[uav_id] = trajectory
            messages.append(
                PIBTEpochMessage(
                    epoch,
                    coordinator_id,
                    uav_id,
                    "waypoint_subgoal_trajectory",
                    next_position,
                    goal,
                    tuple(reversed(routes[uav_id])),
                )
            )

        return ComponentPlan(
            next_positions,
            projected_goals,
            trajectories,
            coordinator_id,
            messages,
            pibt_stats,
            projected_count,
            trajectory_rejections,
        )


def _routes_to_coordinator(
    uav_ids: list[int],
    positions: list[Coord],
    coordinator_id: int,
    communication_range: float,
) -> dict[int, tuple[int, ...]]:
    """Return shortest pairwise request/response routes inside one component."""
    by_id = dict(zip(uav_ids, positions))
    adjacency: dict[int, list[int]] = {uav_id: [] for uav_id in uav_ids}
    for first_index, first_id in enumerate(uav_ids):
        first = by_id[first_id]
        for second_id in uav_ids[first_index + 1 :]:
            second = by_id[second_id]
            if math.hypot(first[0] - second[0], first[1] - second[1]) <= communication_range:
                adjacency[first_id].append(second_id)
                adjacency[second_id].append(first_id)

    routes: dict[int, tuple[int, ...]] = {coordinator_id: (coordinator_id,)}
    frontier = [coordinator_id]
    while frontier:
        current = frontier.pop(0)
        for neighbor in adjacency[current]:
            if neighbor in routes:
                continue
            routes[neighbor] = (neighbor,) + routes[current]
            frontier.append(neighbor)
    if len(routes) != len(uav_ids):
        raise ValueError("communication component is not connected by pairwise links")
    return routes
