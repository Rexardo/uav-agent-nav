"""Command-line entry point for the RACER + PIBT hybrid simulator."""

from __future__ import annotations

import argparse
from typing import Any

import racer_bridge  # noqa: F401
from racer_types import RACERConfig

from pibt_motion import PIBTMotionConfig
from racer_pibt_sim import run_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RACER allocation with PIBT/SFC/LSC/Bernstein motion planning."
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--map-id",
        type=int,
        choices=(1, 2),
        default=RACERConfig.map_id,
        help="1: original random map; 2: fixed one-cell-wide dense maze.",
    )
    parser.add_argument("--seed", type=int, default=RACERConfig.random_seed)
    parser.add_argument("--num-uavs", type=int, default=RACERConfig.num_uavs)
    parser.add_argument("--obstacle-count", type=int, default=RACERConfig.obstacle_count)
    parser.add_argument("--max-steps", type=int, default=RACERConfig.max_steps)
    parser.add_argument("--comm-range", type=float, default=RACERConfig.comm_range)
    parser.add_argument("--coverage-radius", type=float, default=RACERConfig.coverage_radius)
    parser.add_argument("--safe-radius", type=float, default=RACERConfig.safe_radius)
    parser.add_argument("--stop-known-ratio", type=float, default=RACERConfig.stop_known_ratio)
    parser.add_argument("--render-interval", type=int, default=RACERConfig.render_interval)
    parser.add_argument("--render-pause", type=float, default=RACERConfig.render_pause)
    parser.add_argument("--render-frames", type=int, default=RACERConfig.render_interpolation_frames)
    parser.add_argument("--agent-radius", type=float, default=PIBTMotionConfig.agent_radius)
    parser.add_argument("--max-velocity", type=float, default=PIBTMotionConfig.max_velocity)
    parser.add_argument("--max-acceleration", type=float, default=PIBTMotionConfig.max_acceleration)
    parser.add_argument("--segment-duration", type=float, default=PIBTMotionConfig.segment_duration)
    return parser.parse_args()


def configs_from_args(args: argparse.Namespace) -> tuple[RACERConfig, PIBTMotionConfig]:
    racer = RACERConfig(
        map_id=args.map_id,
        random_seed=args.seed,
        num_uavs=args.num_uavs,
        obstacle_count=args.obstacle_count,
        max_steps=args.max_steps,
        comm_range=args.comm_range,
        coverage_radius=args.coverage_radius,
        safe_radius=args.safe_radius,
        stop_known_ratio=args.stop_known_ratio,
        render_interval=args.render_interval,
        render_pause=args.render_pause,
        render_interpolation_frames=max(1, args.render_frames),
    )
    motion = PIBTMotionConfig(
        agent_radius=args.agent_radius,
        max_velocity=args.max_velocity,
        max_acceleration=args.max_acceleration,
        segment_duration=args.segment_duration,
        seed=args.seed,
    )
    return racer, motion


def format_result(result: dict[str, Any]) -> str:
    fields = (
        "steps",
        "known_ratio",
        "num_uavs",
        "path_length_total",
        "termination_reason",
        "collision_events",
        "returned_home",
        "pibt_epochs",
        "pibt_requests",
        "pibt_responses",
        "pibt_priority_inheritances",
        "pibt_backtracks",
        "pibt_waits",
        "pibt_target_projections",
        "pibt_trajectory_rejections",
        "pibt_messages",
        "pibt_message_hops",
        "pibt_max_velocity_observed",
        "pibt_max_acceleration_observed",
        "final_positions",
    )
    values = []
    for field in fields:
        value = result[field]
        if field == "known_ratio":
            value = f"{value:.2%}"
        elif isinstance(value, float):
            value = f"{value:.4f}"
        values.append(f"{field}={value}")
    return "RACER_PIBT finished: " + ", ".join(values)


def main() -> None:
    args = parse_args()
    racer_config, motion_config = configs_from_args(args)
    result = run_simulation(racer_config, motion_config, show=not args.no_show)
    print(format_result(result))


if __name__ == "__main__":
    main()
