"""Subgoal, SFC/LSC, and dynamically feasible Bernstein trajectory helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

Coord = tuple[int, int]


@dataclass(frozen=True)
class SafeFlightCorridor:
    lower: np.ndarray
    upper: np.ndarray

    def contains(self, point: np.ndarray, tolerance: float = 1e-9) -> bool:
        return bool(np.all(point >= self.lower - tolerance) and np.all(point <= self.upper + tolerance))


@dataclass(frozen=True)
class LinearSafeConstraint:
    normal: np.ndarray
    offset: float
    control_index: int

    def contains(self, point: np.ndarray, tolerance: float = 1e-9) -> bool:
        return float(np.dot(self.normal, point)) <= self.offset + tolerance


@dataclass
class BernsteinTrajectory:
    control_points: np.ndarray
    duration: float
    waypoint: Coord
    subgoal: tuple[float, float]
    max_velocity: float
    max_acceleration: float
    feasible: bool
    reason: str = "ok"

    def sample(self, count: int = 41) -> np.ndarray:
        degree = len(self.control_points) - 1
        values = []
        for s in np.linspace(0.0, 1.0, max(2, count)):
            point = np.zeros(2, dtype=float)
            for k, control in enumerate(self.control_points):
                weight = math.comb(degree, k) * (s**k) * ((1.0 - s) ** (degree - k))
                point += weight * control
            values.append(point)
        return np.asarray(values)


def build_sfc(start: Coord, waypoint: Coord, radius: float) -> SafeFlightCorridor:
    """Build the convex corridor inside the union of two adjacent free cells."""
    p0 = np.asarray(start, dtype=float)
    p1 = np.asarray(waypoint, dtype=float)
    padding = 0.5 - radius
    if padding <= 0.0:
        raise ValueError("agent radius must be smaller than half a grid cell")
    if start[1] == waypoint[1]:
        lower = np.array([min(p0[0], p1[0]) - padding, p0[1] - padding])
        upper = np.array([max(p0[0], p1[0]) + padding, p0[1] + padding])
    elif start[0] == waypoint[0]:
        lower = np.array([p0[0] - padding, min(p0[1], p1[1]) - padding])
        upper = np.array([p0[0] + padding, max(p0[1], p1[1]) + padding])
    else:
        lower = np.minimum(p0, p1) - padding
        upper = np.maximum(p0, p1) + padding
    return SafeFlightCorridor(lower, upper)


def minimum_jerk_bernstein(start: np.ndarray, end: np.ndarray, duration: float) -> np.ndarray:
    """Degree-five Bernstein form of the rest-to-rest minimum-jerk segment."""
    return np.vstack((start, start, start, end, end, end)).astype(float)


def build_pairwise_lsc(
    all_control_points: list[np.ndarray],
    agent_index: int,
    radius: float,
) -> list[LinearSafeConstraint]:
    """Construct control-point-wise separating half-spaces for one agent."""
    own = all_control_points[agent_index]
    constraints: list[LinearSafeConstraint] = []
    for other_index, other in enumerate(all_control_points):
        if other_index == agent_index:
            continue
        for k, (own_point, other_point) in enumerate(zip(own, other)):
            delta = other_point - own_point
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-12:
                normal = np.array([1.0, 0.0])
                offset = -math.inf
            else:
                normal = delta / distance
                midpoint = 0.5 * (own_point + other_point)
                offset = float(np.dot(normal, midpoint) - radius)
            constraints.append(LinearSafeConstraint(normal, offset, k))
    return constraints


def solve_subgoal_lp(
    start: Coord,
    waypoint: Coord,
    sfc: SafeFlightCorridor,
    endpoint_constraints: list[LinearSafeConstraint],
) -> tuple[np.ndarray, float]:
    """Solve the paper's line-segment subgoal LP as a one-dimensional interval."""
    origin = np.asarray(start, dtype=float)
    direction = np.asarray(waypoint, dtype=float) - origin
    low, high = 0.0, 1.0
    for axis in range(2):
        if abs(direction[axis]) <= 1e-12:
            continue
        first = (sfc.lower[axis] - origin[axis]) / direction[axis]
        second = (sfc.upper[axis] - origin[axis]) / direction[axis]
        low = max(low, min(first, second))
        high = min(high, max(first, second))
    for constraint in endpoint_constraints:
        coefficient = float(np.dot(constraint.normal, direction))
        residual = constraint.offset - float(np.dot(constraint.normal, origin))
        if coefficient > 1e-12:
            high = min(high, residual / coefficient)
        elif coefficient < -1e-12:
            low = max(low, residual / coefficient)
        elif residual < 0.0:
            high = -1.0
    alpha = float(np.clip(high, 0.0, 1.0)) if high >= low - 1e-9 else 0.0
    return origin + alpha * direction, alpha


def trajectory_dynamics(samples: np.ndarray, duration: float) -> tuple[float, float]:
    dt = duration / max(1, len(samples) - 1)
    velocity = np.diff(samples, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    max_velocity = float(np.max(np.abs(velocity))) if velocity.size else 0.0
    max_acceleration = float(np.max(np.abs(acceleration))) if acceleration.size else 0.0
    return max_velocity, max_acceleration


def optimize_safe_trajectory(
    start: Coord,
    waypoint: Coord,
    duration: float,
    max_velocity: float,
    max_acceleration: float,
    radius: float,
    lsc: list[LinearSafeConstraint],
) -> BernsteinTrajectory:
    """Create and validate the one-segment constrained minimum-jerk solution."""
    sfc = build_sfc(start, waypoint, radius)
    endpoint_lsc = [constraint for constraint in lsc if constraint.control_index >= 3]
    subgoal, alpha = solve_subgoal_lp(start, waypoint, sfc, endpoint_lsc)
    if alpha < 1.0 - 1e-9:
        subgoal = np.asarray(start, dtype=float)
        waypoint = start
        sfc = build_sfc(start, start, radius)
    control_points = minimum_jerk_bernstein(np.asarray(start, dtype=float), subgoal, duration)
    if not all(sfc.contains(point) for point in control_points):
        return BernsteinTrajectory(control_points, duration, waypoint, tuple(subgoal), 0.0, 0.0, False, "sfc")
    if any(not constraint.contains(control_points[constraint.control_index]) for constraint in lsc):
        return BernsteinTrajectory(control_points, duration, waypoint, tuple(subgoal), 0.0, 0.0, False, "lsc")
    samples = BernsteinTrajectory(control_points, duration, waypoint, tuple(subgoal), 0.0, 0.0, True).sample(101)
    observed_velocity, observed_acceleration = trajectory_dynamics(samples, duration)
    feasible = observed_velocity <= max_velocity + 1e-6 and observed_acceleration <= max_acceleration + 1e-6
    reason = "ok" if feasible else "dynamics"
    return BernsteinTrajectory(
        control_points,
        duration,
        waypoint,
        (float(subgoal[0]), float(subgoal[1])),
        observed_velocity,
        observed_acceleration,
        feasible,
        reason,
    )


def trajectories_are_separated(trajectories: list[BernsteinTrajectory], radius: float) -> bool:
    samples = [trajectory.sample(81) for trajectory in trajectories]
    for first in range(len(samples)):
        for second in range(first + 1, len(samples)):
            distances = np.linalg.norm(samples[first] - samples[second], axis=1)
            if float(np.min(distances)) < 2.0 * radius - 1e-8:
                return False
    return True
