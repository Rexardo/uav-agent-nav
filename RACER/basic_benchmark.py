"""Batch benchmark for the modular RACER simulator."""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional for headless runs.
    plt = None

# RACER modules use same-directory imports. Adding this directory keeps both
# ``python RACER/basic_benchmark.py`` and ``python -m RACER.basic_benchmark`` usable.
RACER_DIR = Path(__file__).resolve().parent
if str(RACER_DIR) not in sys.path:
    sys.path.insert(0, str(RACER_DIR))

from racer_sim import RACERSimulator
from racer_types import RACERConfig


Point = tuple[float, float]


def _point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.hypot(px - ax, py - ay)
    ratio = ((px - ax) * dx + (py - ay) * dy) / length_sq
    ratio = max(0.0, min(1.0, ratio))
    nearest = (ax + ratio * dx, ay + ratio * dy)
    return math.hypot(px - nearest[0], py - nearest[1])


def _cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(point: Point, start: Point, end: Point) -> bool:
    eps = 1e-9
    return (
        abs(_cross(start, end, point)) <= eps
        and min(start[0], end[0]) - eps <= point[0] <= max(start[0], end[0]) + eps
        and min(start[1], end[1]) - eps <= point[1] <= max(start[1], end[1]) + eps
    )


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    ab_c = _cross(a, b, c)
    ab_d = _cross(a, b, d)
    cd_a = _cross(c, d, a)
    cd_b = _cross(c, d, b)
    eps = 1e-9
    if ((ab_c > eps and ab_d < -eps) or (ab_c < -eps and ab_d > eps)) and (
        (cd_a > eps and cd_b < -eps) or (cd_a < -eps and cd_b > eps)
    ):
        return True
    return (
        (abs(ab_c) <= eps and _on_segment(c, a, b))
        or (abs(ab_d) <= eps and _on_segment(d, a, b))
        or (abs(cd_a) <= eps and _on_segment(a, c, d))
        or (abs(cd_b) <= eps and _on_segment(b, c, d))
    )


def _segment_to_segment_distance(a: Point, b: Point, c: Point, d: Point) -> float:
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(
        _point_to_segment_distance(a, c, d),
        _point_to_segment_distance(b, c, d),
        _point_to_segment_distance(c, a, b),
        _point_to_segment_distance(d, a, b),
    )


def _segment_to_rectangle_distance(
    start: Point,
    end: Point,
    rectangle: tuple[float, float, float, float],
) -> float:
    rx, ry, width, height = rectangle
    corners = [
        (rx, ry),
        (rx + width, ry),
        (rx + width, ry + height),
        (rx, ry + height),
    ]
    if any(rx <= p[0] <= rx + width and ry <= p[1] <= ry + height for p in (start, end)):
        return 0.0
    edges = list(zip(corners, corners[1:] + corners[:1]))
    return min(_segment_to_segment_distance(start, end, edge_start, edge_end) for edge_start, edge_end in edges)


def trajectory_path_length(history: list[tuple[int, int]], resolution: float = 1.0) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(history, history[1:])) * resolution


def trajectory_turning_angles(history: list[tuple[int, int]]) -> list[float]:
    """Return absolute heading changes in degrees, including zero-degree straight motion."""
    directions: list[Point] = []
    for start, end in zip(history, history[1:]):
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        if math.hypot(dx, dy) > 1e-12:
            directions.append((dx, dy))

    angles: list[float] = []
    for first, second in zip(directions, directions[1:]):
        first_norm = math.hypot(first[0], first[1])
        second_norm = math.hypot(second[0], second[1])
        cosine = (first[0] * second[0] + first[1] * second[1]) / (first_norm * second_norm)
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    return angles


def trajectory_min_obstacle_distance(simulator: RACERSimulator) -> float:
    """Return center-line clearance to the nearest true obstacle boundary."""
    world = simulator.world
    minimum = math.inf

    for uav in simulator.uavs:
        history: list[Point] = [(float(x), float(y)) for x, y in uav.history]
        segments = list(zip(history, history[1:]))
        if not segments and history:
            segments = [(history[0], history[0])]

        for start, end in segments:
            for cx, cy, radius in world.circles:
                clearance = max(0.0, _point_to_segment_distance((cx, cy), start, end) - radius)
                minimum = min(minimum, clearance)
            for rectangle in world.rectangles:
                minimum = min(minimum, _segment_to_rectangle_distance(start, end, rectangle))

    if math.isinf(minimum):
        return minimum
    return minimum * simulator.config.map_resolution


def run_simulation(
    num_uavs: int = RACERConfig.num_uavs,
    map_seed: int = RACERConfig.random_seed,
    max_steps: int = RACERConfig.max_steps,
    obstacle_count: int = RACERConfig.obstacle_count,
    width: int = RACERConfig.width,
    height: int = RACERConfig.height,
    coverage_radius: float = RACERConfig.coverage_radius,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run one headless RACER simulation and calculate benchmark metrics."""
    config = RACERConfig(
        num_uavs=max(1, int(num_uavs)),
        random_seed=int(map_seed),
        max_steps=max(1, int(max_steps)),
        obstacle_count=max(0, int(obstacle_count)),
        width=max(8, int(width)),
        height=max(8, int(height)),
        coverage_radius=float(coverage_radius),
    )
    simulator = RACERSimulator(config)
    racer_result = simulator.run(show=False)

    path_lengths = [trajectory_path_length(uav.history, config.map_resolution) for uav in simulator.uavs]
    turning_angles = [angle for uav in simulator.uavs for angle in trajectory_turning_angles(uav.history)]
    total_path = float(sum(path_lengths))
    total_steps = int(racer_result["steps"]) + 1
    min_obstacle_distance = trajectory_min_obstacle_distance(simulator)

    result = {
        "coverage": float(racer_result["known_ratio"]) * 100.0,
        "total_steps": total_steps,
        "total_path": total_path,
        "avg_path": total_path / len(simulator.uavs),
        "avg_turning_angle": float(np.mean(turning_angles)) if turning_angles else 0.0,
        "turning_angle_sum": float(sum(turning_angles)),
        "turning_angle_count": len(turning_angles),
        "min_obstacle_distance": min_obstacle_distance,
        "returned_home": bool(racer_result["returned_home"]),
    }

    if verbose:
        clearance_text = "N/A" if math.isinf(min_obstacle_distance) else f"{min_obstacle_distance:.3f}"
        print(
            f"    UAV={num_uavs}: 覆盖率={result['coverage']:.2f}%, "
            f"总步数={total_steps}, 总路径={total_path:.2f}, 平均路径={result['avg_path']:.2f}, "
            f"平均转向角={result['avg_turning_angle']:.2f} deg, "
            f"最小障碍距离={clearance_text}, 返回起点={result['returned_home']}"
        )
    return result


def _print_terminal_summary(metrics: dict[int, dict[str, list[float]]]) -> dict[int, dict[str, float]]:
    summary: dict[int, dict[str, float]] = {}
    print("\nRACER Benchmark 汇总（路径与距离单位由 map_resolution 决定）")
    print("-" * 116)
    print(
        f"{'UAV':>4} | {'任务覆盖率(%)':>13} | {'总执行步数':>12} | {'总路径长度':>12} | "
        f"{'单机平均路径':>13} | {'平均转向角(deg)':>17} | {'最小障碍距离':>14}"
    )
    print("-" * 116)

    for uav_num, values in metrics.items():
        turn_count = int(sum(values["turning_angle_count"]))
        average_turn = sum(values["turning_angle_sum"]) / turn_count if turn_count else 0.0
        finite_clearances = [value for value in values["min_obstacle_distance"] if math.isfinite(value)]
        minimum_clearance = min(finite_clearances) if finite_clearances else math.inf
        row = {
            "coverage": float(np.mean(values["coverage"])),
            "total_steps": float(np.mean(values["total_steps"])),
            "total_path": float(np.mean(values["total_path"])),
            "avg_path": float(np.mean(values["avg_path"])),
            "avg_turning_angle": float(average_turn),
            "min_obstacle_distance": float(minimum_clearance),
        }
        summary[uav_num] = row
        clearance_text = "N/A" if math.isinf(minimum_clearance) else f"{minimum_clearance:.3f}"
        print(
            f"{uav_num:>4} | {row['coverage']:>13.2f} | {row['total_steps']:>12.1f} | "
            f"{row['total_path']:>12.2f} | {row['avg_path']:>13.2f} | "
            f"{row['avg_turning_angle']:>17.2f} | {clearance_text:>14}"
        )
    print("-" * 116)
    print("注：前四项为各地图均值；平均转向角汇总全部有效转向；最小障碍距离取所有地图中的最小值。")
    return summary


def _plot_original_metrics(
    metrics: dict[int, dict[str, list[float]]],
    num_maps: int,
    output_path: str | Path,
    show: bool,
) -> None:
    if plt is None:
        print("matplotlib 不可用，已完成终端统计，但未生成 benchmark_results.png。")
        return

    x_axis = list(metrics)
    mean_coverage = [np.mean(metrics[u]["coverage"]) for u in x_axis]
    mean_steps = [np.mean(metrics[u]["total_steps"]) for u in x_axis]
    mean_total_path = [np.mean(metrics[u]["total_path"]) for u in x_axis]
    mean_avg_path = [np.mean(metrics[u]["avg_path"]) for u in x_axis]

    fig, axs = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(f"RACER Benchmark Results (Averaged over {num_maps} Random Maps)", fontsize=16)

    axs[0, 0].plot(x_axis, mean_coverage, "-o", color="blue")
    axs[0, 0].set_title("Task Coverage Ratio (%)")
    axs[0, 0].set_xlabel("Number of UAVs")
    axs[0, 0].set_ylabel("Coverage (%)")

    axs[0, 1].plot(x_axis, mean_steps, "-o", color="orange")
    axs[0, 1].set_title("Total Steps (Time)")
    axs[0, 1].set_xlabel("Number of UAVs")
    axs[0, 1].set_ylabel("Steps")

    axs[1, 0].plot(x_axis, mean_total_path, "-o", color="green")
    axs[1, 0].set_title("Total Path Length")
    axs[1, 0].set_xlabel("Number of UAVs")
    axs[1, 0].set_ylabel("Distance")

    axs[1, 1].plot(x_axis, mean_avg_path, "-o", color="purple")
    axs[1, 1].set_title("Average Path Length per UAV")
    axs[1, 1].set_xlabel("Number of UAVs")
    axs[1, 1].set_ylabel("Distance")

    for ax in axs.flat:
        ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=300)
    print(f"图表已保存到: {Path(output_path).resolve()}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_benchmark_and_plot(
    num_maps: int = 5,
    max_uavs: int = 8,
    benchmark_seed: int = 42,
    max_steps: int = RACERConfig.max_steps,
    obstacle_count: int = RACERConfig.obstacle_count,
    width: int = RACERConfig.width,
    height: int = RACERConfig.height,
    coverage_radius: float = RACERConfig.coverage_radius,
    output_path: str | Path = "benchmark_results.png",
    show: bool = True,
    verbose: bool = False,
) -> dict[int, dict[str, float]]:
    """Benchmark RACER on identical seeded maps for every UAV count."""
    num_maps = max(1, int(num_maps))
    max_uavs = max(1, int(max_uavs))
    rng = random.Random(benchmark_seed)
    map_seeds = [rng.randint(0, 999_999) for _ in range(num_maps)]

    print("\n==============================================")
    print("开始 RACER 批量基准测试 (Benchmark)")
    print(f"共 {num_maps} 张随机地图 | 每张地图运行 1 到 {max_uavs} 架无人机")
    print(f"地图种子: {map_seeds}")
    print("==============================================\n")

    metric_names = (
        "coverage",
        "total_steps",
        "total_path",
        "avg_path",
        "turning_angle_sum",
        "turning_angle_count",
        "min_obstacle_distance",
    )
    metrics: dict[int, dict[str, list[float]]] = {
        uav_num: {name: [] for name in metric_names}
        for uav_num in range(1, max_uavs + 1)
    }

    for map_index, current_seed in enumerate(map_seeds, start=1):
        print(f">>> 正在测试 Map {map_index}/{num_maps} (Seed: {current_seed})")
        for num_uavs in range(1, max_uavs + 1):
            result = run_simulation(
                num_uavs=num_uavs,
                map_seed=current_seed,
                max_steps=max_steps,
                obstacle_count=obstacle_count,
                width=width,
                height=height,
                coverage_radius=coverage_radius,
                verbose=verbose,
            )
            for name in metric_names:
                metrics[num_uavs][name].append(result[name])

    summary = _print_terminal_summary(metrics)
    _plot_original_metrics(metrics, num_maps, output_path, show)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RACER multi-UAV benchmark.")
    parser.add_argument("--num-maps", type=int, default=5, help="Number of seeded random maps.")
    parser.add_argument("--max-uavs", type=int, default=8, help="Test UAV counts from 1 through this value.")
    parser.add_argument("--benchmark-seed", type=int, default=42, help="Seed used to generate the map-seed list.")
    parser.add_argument("--max-steps", type=int, default=RACERConfig.max_steps)
    parser.add_argument("--obstacle-count", type=int, default=RACERConfig.obstacle_count)
    parser.add_argument("--width", type=int, default=RACERConfig.width)
    parser.add_argument("--height", type=int, default=RACERConfig.height)
    parser.add_argument("--coverage-radius", type=float, default=RACERConfig.coverage_radius)
    parser.add_argument("--output", type=str, default="benchmark_results.png")
    parser.add_argument("--no-show", action="store_true", help="Save the plot without opening a window.")
    parser.add_argument("--verbose", action="store_true", help="Print every individual simulation result.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark_and_plot(
        num_maps=args.num_maps,
        max_uavs=args.max_uavs,
        benchmark_seed=args.benchmark_seed,
        max_steps=args.max_steps,
        obstacle_count=args.obstacle_count,
        width=args.width,
        height=args.height,
        coverage_radius=args.coverage_radius,
        output_path=args.output,
        show=not args.no_show,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
