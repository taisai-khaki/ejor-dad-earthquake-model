from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RAW_COLUMNS = [
    "instance_id",
    "scenario_class_id",
    "seed",
    "size",
    "available_drones",
    "active_drones_used",
    "drone_served_customers",
    "drone_share",
    "objective",
    "priority_tardiness_cost",
    "high_priority_tardiness",
    "p90_high_priority_tardiness",
    "high_priority_on_time_rate",
    "mean_high_priority_delivery_time",
    "cpu",
    "feasible",
    "error",
    "reused_original_fleet_run",
]


def _read_shards(root: Path, shard_prefix: str) -> pd.DataFrame:
    paths = sorted(root.glob(f"{shard_prefix}_p*/fleet_size_sweep_nils_r_partial.csv"))
    if len(paths) != 8:
        raise RuntimeError(f"Expected 8 fleet shards, found {len(paths)}")
    frames = [pd.read_csv(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    for column in RAW_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    frame = frame[RAW_COLUMNS].copy()
    frame["instance_id"] = frame["instance_id"].astype(str)
    frame["available_drones"] = pd.to_numeric(frame["available_drones"], errors="coerce").astype("Int64")
    frame["size"] = pd.to_numeric(frame["size"], errors="coerce").astype("Int64")
    frame["feasible"] = frame["feasible"].astype(str).str.lower().isin({"true", "1", "yes"})
    frame = frame.drop_duplicates(["instance_id", "available_drones"], keep="last")
    return frame.sort_values(["size", "instance_id", "available_drones"]).reset_index(drop=True)


def _mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["size_label"] = work["size"].astype(str)
    groups: list[tuple[str, pd.DataFrame]] = [("All", work)]
    for size in sorted(work["size"].dropna().unique()):
        groups.append((str(int(size)), work[work["size"].eq(size)]))

    rows: list[dict[str, object]] = []
    for size_label, size_frame in groups:
        by_drones: dict[int, dict[str, object]] = {}
        for available in range(4):
            subset = size_frame[size_frame["available_drones"].eq(available)]
            feasible = subset[subset["feasible"]]
            by_drones[available] = {
                "size": size_label,
                "available_drones": available,
                "runs": int(len(feasible)),
                "mean_active_drones_used": _mean(feasible, "active_drones_used"),
                "mean_objective": _mean(feasible, "objective"),
                "mean_priority_tardiness_cost": _mean(feasible, "priority_tardiness_cost"),
                "mean_high_priority_tardiness": _mean(feasible, "high_priority_tardiness"),
                "mean_p90_high_priority_tardiness": _mean(feasible, "p90_high_priority_tardiness"),
                "mean_high_priority_on_time_rate": _mean(feasible, "high_priority_on_time_rate"),
                "mean_drone_share": _mean(feasible, "drone_share"),
                "mean_cpu": _mean(feasible, "cpu"),
            }
        for available in range(4):
            current = by_drones[available]
            previous = by_drones.get(available - 1)
            if previous is None:
                current["marginal_objective_improvement_vs_previous_drone"] = np.nan
                current["marginal_priority_tardiness_reduction_vs_previous_drone"] = np.nan
                current["marginal_high_priority_on_time_gain_vs_previous_drone"] = np.nan
            else:
                current["marginal_objective_improvement_vs_previous_drone"] = (
                    previous["mean_objective"] - current["mean_objective"]
                    if pd.notna(previous["mean_objective"]) and pd.notna(current["mean_objective"])
                    else np.nan
                )
                current["marginal_priority_tardiness_reduction_vs_previous_drone"] = (
                    previous["mean_priority_tardiness_cost"] - current["mean_priority_tardiness_cost"]
                    if pd.notna(previous["mean_priority_tardiness_cost"])
                    and pd.notna(current["mean_priority_tardiness_cost"])
                    else np.nan
                )
                current["marginal_high_priority_on_time_gain_vs_previous_drone"] = (
                    current["mean_high_priority_on_time_rate"] - previous["mean_high_priority_on_time_rate"]
                    if pd.notna(previous["mean_high_priority_on_time_rate"])
                    and pd.notna(current["mean_high_priority_on_time_rate"])
                    else np.nan
                )
            rows.append(current)
    columns = [
        "size",
        "available_drones",
        "runs",
        "mean_active_drones_used",
        "mean_objective",
        "mean_priority_tardiness_cost",
        "mean_high_priority_tardiness",
        "mean_p90_high_priority_tardiness",
        "mean_high_priority_on_time_rate",
        "mean_drone_share",
        "mean_cpu",
        "marginal_objective_improvement_vs_previous_drone",
        "marginal_priority_tardiness_reduction_vs_previous_drone",
        "marginal_high_priority_on_time_gain_vs_previous_drone",
    ]
    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--shard-prefix", default="fleet_size_sweep_full_20260827")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_dir / "outputs"
    frame = _read_shards(root, args.shard_prefix)
    if len(frame) != 2880 or frame["instance_id"].nunique() != 720:
        raise RuntimeError(
            f"Expected 2880 rows and 720 instances, found {len(frame)} and {frame['instance_id'].nunique()}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "fleet_size_sweep_nils_r.csv", index=False, float_format="%.10f")
    table = aggregate(frame)
    table.to_csv(args.output_dir / "table_fleet_size_marginal_value.csv", index=False, float_format="%.10f")
    print(f"Merged rows={len(frame)} instances={frame['instance_id'].nunique()} errors={frame['error'].fillna('').ne('').sum()}")
    print(table.to_string(index=False))
    print(f"Output={args.output_dir}")


if __name__ == "__main__":
    main()
