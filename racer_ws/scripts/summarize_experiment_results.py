#!/usr/bin/env python3
"""Aggregate RACER experiment summary.csv files into avg +/- sample std statistics."""

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


METRICS: Sequence[Tuple[str, str]] = (
    ("exploration_time_s", "exploration_time_s"),
    ("total_path_length_m", "total_path_length_m"),
    ("coverage_pct", "coverage_pct"),
    ("mean_turn_angle_deg", "mean_turn_angle_deg"),
    ("mean_planning_time_s", "mean_planning_time_s"),
    ("min_obstacle_distance_m", "min_obstacle_distance_m"),
    ("total_collision_count", "total_collision_count"),
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find RACER summary.csv files and calculate the mean and sample "
            "standard deviation of selected experiment metrics."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path.home() / "racer_ws" / "results",
        help="Root directory containing scene/communication-range summary.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/d/26暑研无人机巡检/results/experiment_summary_statistics.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--display-precision",
        type=int,
        default=3,
        help="Decimal places used in the human-readable avg +/- std columns.",
    )
    return parser.parse_args()


def require_finite(row: Dict[str, str], column: str, source: Path, row_number: int) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source}: row {row_number} has invalid {column!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"{source}: row {row_number} has non-finite {column!r}")
    return value


def optional_finite(row: Dict[str, str], column: str, source: Path, row_number: int) -> float:
    raw_value = row.get(column, "")
    if raw_value is None or not raw_value.strip():
        return math.nan
    return require_finite(row, column, source, row_number)


def read_summary(path: Path) -> List[Dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"{path}: missing CSV header")

        required = {
            "mission_time_s",
            "total_path_length_m",
            "coverage_pct",
            "mean_turn_angle_deg",
            "min_obstacle_distance_m",
            "total_collision_count",
        }
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")

        records: List[Dict[str, float]] = []
        for row_number, row in enumerate(reader, start=2):
            records.append(
                {
                    "exploration_time_s": require_finite(
                        row, "mission_time_s", path, row_number
                    ),
                    "total_path_length_m": require_finite(
                        row, "total_path_length_m", path, row_number
                    ),
                    "coverage_pct": require_finite(row, "coverage_pct", path, row_number),
                    "mean_turn_angle_deg": require_finite(
                        row, "mean_turn_angle_deg", path, row_number
                    ),
                    "mean_planning_time_s": optional_finite(
                        row, "mean_planning_time_s", path, row_number
                    ),
                    "min_obstacle_distance_m": require_finite(
                        row, "min_obstacle_distance_m", path, row_number
                    ),
                    "total_collision_count": require_finite(
                        row, "total_collision_count", path, row_number
                    ),
                }
            )
    if not records:
        raise ValueError(f"{path}: contains no experiment rows")
    return records


def mean_and_sample_std(values: Iterable[float]) -> Tuple[float, float]:
    data = list(values)
    average = statistics.fmean(data)
    sample_std = statistics.stdev(data) if len(data) >= 2 else 0.0
    return average, sample_std


def scene_and_range(input_root: Path, summary_path: Path) -> Tuple[str, str]:
    try:
        relative_parent = summary_path.parent.relative_to(input_root)
        parts = relative_parent.parts
    except ValueError:
        parts = summary_path.parent.parts
    scene = parts[-2] if len(parts) >= 2 else summary_path.parent.name
    communication_range = parts[-1] if parts else "unknown"
    return scene, communication_range


def build_output_rows(
    input_root: Path, summary_paths: Sequence[Path], display_precision: int
) -> List[Dict[str, object]]:
    output_rows: List[Dict[str, object]] = []
    for summary_path in summary_paths:
        records = read_summary(summary_path)
        scene, communication_range = scene_and_range(input_root, summary_path)
        output: Dict[str, object] = {
            "scene": scene,
            "communication_range": communication_range,
            "run_count": len(records),
            "exploration_time_source": "mission_time_s",
        }
        for source_name, output_name in METRICS:
            values = [record[source_name] for record in records if math.isfinite(record[source_name])]
            if source_name == "mean_planning_time_s":
                output["planning_time_run_count"] = len(values)
            if not values:
                output[f"{output_name}_avg"] = ""
                output[f"{output_name}_std"] = ""
                output[f"{output_name}_avg_pm_std"] = ""
                continue
            average, sample_std = mean_and_sample_std(values)
            output[f"{output_name}_avg"] = f"{average:.6f}"
            output[f"{output_name}_std"] = f"{sample_std:.6f}"
            output[f"{output_name}_avg_pm_std"] = (
                f"{average:.{display_precision}f} +/- {sample_std:.{display_precision}f}"
            )
        output["source_summary_csv"] = str(summary_path.resolve())
        output_rows.append(output)
    return output_rows


def output_fieldnames() -> List[str]:
    fields = [
        "scene",
        "communication_range",
        "run_count",
        "planning_time_run_count",
        "exploration_time_source",
    ]
    for _, output_name in METRICS:
        fields.extend(
            [
                f"{output_name}_avg",
                f"{output_name}_std",
                f"{output_name}_avg_pm_std",
            ]
        )
    fields.append("source_summary_csv")
    return fields


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_path = args.output.expanduser()
    summary_paths = sorted(input_root.rglob("summary.csv"))
    if not summary_paths:
        raise SystemExit(f"No summary.csv files found under {input_root}")

    output_rows = build_output_rows(input_root, summary_paths, args.display_precision)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fieldnames())
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Processed {len(summary_paths)} summary file(s), {sum(row['run_count'] for row in output_rows)} run(s)")
    print(output_path.resolve())


if __name__ == "__main__":
    main()
