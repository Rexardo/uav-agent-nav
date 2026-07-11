"""Command-line entry point for the modular RACER-style exploration demo."""

from __future__ import annotations

import argparse
from typing import Any

from racer_sim import run_simulation
from racer_types import RACERConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modular RACER multi-UAV exploration demo.")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--map-id",
        type=int,
        choices=(1, 2),
        default=RACERConfig.map_id,
        help="1: original random map; 2: fixed four-UAV dense maze.",
    )
    parser.add_argument("--seed", type=int, default=RACERConfig.random_seed)
    parser.add_argument("--num-uavs", type=int, default=RACERConfig.num_uavs)
    parser.add_argument("--obstacle-count", type=int, default=RACERConfig.obstacle_count)
    parser.add_argument("--max-steps", type=int, default=RACERConfig.max_steps)
    parser.add_argument("--comm-range", type=float, default=RACERConfig.comm_range)
    parser.add_argument("--coverage-radius", type=float, default=RACERConfig.coverage_radius)
    parser.add_argument("--safe-radius", type=float, default=RACERConfig.safe_radius)
    parser.add_argument("--stop-known-ratio", type=float, default=RACERConfig.stop_known_ratio)
    parser.add_argument("--hgrid-level-sizes", type=str, default=",".join(str(v) for v in RACERConfig.hgrid_level_sizes))
    parser.add_argument("--render-interval", type=int, default=RACERConfig.render_interval)
    parser.add_argument("--render-pause", type=float, default=RACERConfig.render_pause)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> RACERConfig:
    return RACERConfig(
        map_id=args.map_id,
        random_seed=args.seed,
        num_uavs=args.num_uavs,
        obstacle_count=args.obstacle_count,
        max_steps=args.max_steps,
        comm_range=args.comm_range,
        coverage_radius=args.coverage_radius,
        safe_radius=args.safe_radius,
        stop_known_ratio=args.stop_known_ratio,
        hgrid_level_sizes=tuple(int(part.strip()) for part in args.hgrid_level_sizes.split(",") if part.strip()),
        render_interval=args.render_interval,
        render_pause=args.render_pause,
    )


def format_result(result: dict[str, Any]) -> str:
    return (
        "Simulation finished: "
        f"steps={result['steps']}, "
        f"known_ratio={result['known_ratio']:.2%}, "
        f"num_uavs={result['num_uavs']}, "
        f"hgrid_blocks={result['hgrid_blocks']}, "
        f"hgrid_split_count={result['hgrid_split_count']}, "
        f"hgrid_removed_count={result['hgrid_removed_count']}, "
        f"path_length_total={result['path_length_total']}, "
        f"exploration_finished_step={result['exploration_finished_step']}, "
        f"returned_home={result['returned_home']}, "
        f"owned_plan_count={result['owned_plan_count']}, "
        f"fallback_plan_count={result['fallback_plan_count']}, "
        f"hgrid_reassign_count={result['hgrid_reassign_count']}, "
        f"pairwise_success_count={result['pairwise_success_count']}, "
        f"astar_polyline_count={result['astar_polyline_count']}, "
        f"bspline_plan_count={result['bspline_plan_count']}, "
        f"swarm_collision_count={result['swarm_collision_count']}, "
        f"swarm_replan_count={result['swarm_replan_count']}, "
        f"swarm_replan_failure_count={result['swarm_replan_failure_count']}, "
        f"static_collision_replan_count={result['static_collision_replan_count']}, "
        f"static_collision_stop_count={result['static_collision_stop_count']}, "
        f"home_flags={result['home_flags']}, "
        f"final_positions={result['final_positions']}"
    )


def main() -> None:
    args = parse_args()
    result = run_simulation(config_from_args(args), show=not args.no_show)
    print(format_result(result))


if __name__ == "__main__":
    main()
