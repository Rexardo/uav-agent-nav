"""Batch benchmark for RACER target allocation with PIBT motion planning."""

from __future__ import annotations

import argparse
import importlib.util
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

import racer_bridge
from pibt_motion import PIBTMotionConfig
from racer_pibt_sim import RACERPIBTSimulator
from racer_types import RACERConfig


def _load_metric_helpers():
    helper_path = racer_bridge.RACER_DIR / "basic_benchmark.py"
    spec = importlib.util.spec_from_file_location("_racer_metric_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load benchmark helpers from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


METRICS = _load_metric_helpers()


def run_one(
    num_uavs: int,
    map_seed: int,
    map_id: int,
    max_steps: int,
    obstacle_count: int,
    width: int,
    height: int,
    coverage_radius: float,
) -> dict[str, Any]:
    config = RACERConfig(
        num_uavs=num_uavs,
        random_seed=map_seed,
        map_id=map_id,
        max_steps=max_steps,
        obstacle_count=obstacle_count,
        width=width,
        height=height,
        coverage_radius=coverage_radius,
    )
    simulator = RACERPIBTSimulator(config, PIBTMotionConfig(seed=map_seed))
    raw_result = simulator.run(show=False)
    path_lengths = [METRICS.trajectory_path_length(uav.history, config.map_resolution) for uav in simulator.uavs]
    angles = [angle for uav in simulator.uavs for angle in METRICS.trajectory_turning_angles(uav.history)]
    total_path = float(sum(path_lengths))
    return {
        "coverage": float(raw_result["known_ratio"]) * 100.0,
        "total_steps": int(raw_result["steps"]) + 1,
        "total_path": total_path,
        "avg_path": total_path / len(simulator.uavs),
        "turn_sum": float(sum(angles)),
        "turn_count": len(angles),
        "min_obstacle_distance": METRICS.trajectory_min_obstacle_distance(simulator),
        "returned_home": bool(raw_result["returned_home"]),
        "collisions": len(raw_result["collision_events"]),
    }


def summarize(results: dict[int, list[dict[str, Any]]]) -> dict[int, dict[str, float]]:
    summary: dict[int, dict[str, float]] = {}
    print("\nRACER_PIBT benchmark summary")
    print("-" * 112)
    print(
        f"{'UAV':>4} | {'coverage %':>10} | {'steps':>9} | {'total path':>11} | "
        f"{'avg path':>10} | {'avg turn deg':>12} | {'min obstacle':>12} | {'collisions':>10}"
    )
    print("-" * 112)
    for uav_count, runs in results.items():
        turn_count = sum(run["turn_count"] for run in runs)
        finite_clearances = [run["min_obstacle_distance"] for run in runs if math.isfinite(run["min_obstacle_distance"])]
        row = {
            "coverage": float(np.mean([run["coverage"] for run in runs])),
            "total_steps": float(np.mean([run["total_steps"] for run in runs])),
            "total_path": float(np.mean([run["total_path"] for run in runs])),
            "avg_path": float(np.mean([run["avg_path"] for run in runs])),
            "avg_turning_angle": sum(run["turn_sum"] for run in runs) / turn_count if turn_count else 0.0,
            "min_obstacle_distance": min(finite_clearances) if finite_clearances else math.inf,
            "collisions": float(sum(run["collisions"] for run in runs)),
        }
        summary[uav_count] = row
        clearance = "N/A" if math.isinf(row["min_obstacle_distance"]) else f"{row['min_obstacle_distance']:.3f}"
        print(
            f"{uav_count:>4} | {row['coverage']:>10.2f} | {row['total_steps']:>9.1f} | "
            f"{row['total_path']:>11.2f} | {row['avg_path']:>10.2f} | "
            f"{row['avg_turning_angle']:>12.2f} | {clearance:>12} | {row['collisions']:>10.0f}"
        )
    print("-" * 112)
    return summary


def plot_primary_metrics(
    summary: dict[int, dict[str, float]], output: Path, num_maps: int, show: bool
) -> None:
    if plt is None:
        print("matplotlib is unavailable; terminal metrics are complete and no PNG was generated.")
        return
    counts = list(summary)
    fields = ("coverage", "total_steps", "total_path", "avg_path")
    titles = ("Task Coverage Ratio (%)", "Total Steps", "Total Path Length", "Average Path per UAV")
    colors = ("#1769aa", "#d97706", "#15803d", "#7e22ce")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"RACER_PIBT Benchmark ({num_maps} maps)")
    for axis, field, title, color in zip(axes.flat, fields, titles, colors):
        axis.plot(counts, [summary[count][field] for count in counts], "-o", color=color)
        axis.set_title(title)
        axis.set_xlabel("Number of UAVs")
        axis.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    print(f"Primary benchmark chart saved to: {output.resolve()}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RACER_PIBT on seeded maps.")
    parser.add_argument("--map-id", type=int, choices=(1, 2), default=1)
    parser.add_argument("--num-maps", type=int, default=5)
    parser.add_argument(
        "--uav-counts",
        type=str,
        default="4",
        help="Comma-separated UAV counts; use 4 to test only four UAVs.",
    )
    parser.add_argument("--benchmark-seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=RACERConfig.max_steps)
    parser.add_argument("--obstacle-count", type=int, default=RACERConfig.obstacle_count)
    parser.add_argument("--width", type=int, default=RACERConfig.width)
    parser.add_argument("--height", type=int, default=RACERConfig.height)
    parser.add_argument("--coverage-radius", type=float, default=RACERConfig.coverage_radius)
    parser.add_argument("--output", type=Path, default=Path("racer_pibt_benchmark_results.png"))
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = sorted({max(1, int(value.strip())) for value in args.uav_counts.split(",") if value.strip()})
    if args.map_id == 2 and any(count > 4 for count in counts):
        raise ValueError("Dense Maze map 2 supports at most four UAVs")
    rng = random.Random(args.benchmark_seed)
    seeds = [rng.randint(0, 999_999) for _ in range(max(1, args.num_maps))]
    results = {count: [] for count in counts}
    for map_index, seed in enumerate(seeds, start=1):
        print(f"Map {map_index}/{len(seeds)}, seed={seed}")
        for count in counts:
            result = run_one(
                count,
                seed,
                args.map_id,
                max(1, args.max_steps),
                max(0, args.obstacle_count),
                max(8, args.width),
                max(8, args.height),
                args.coverage_radius,
            )
            results[count].append(result)
            if args.verbose:
                print(f"  UAV={count}: {result}")
    summary = summarize(results)
    plot_primary_metrics(summary, args.output, len(seeds), not args.no_show)


if __name__ == "__main__":
    main()
