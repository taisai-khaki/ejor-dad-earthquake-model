from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.aggregate_nils_r_comparisons import aggregate as aggregate_comparisons
from src.experiments.run_priority_service_speed_by_method import _aggregate as aggregate_priority_speed


METHOD_LABELS = {
    "truck_only": "M1 Truck-only",
    "nils": "M5 NILS",
    "feasibility_aware_alns": "M8 Feasibility-aware ALNS",
    "nils_r": "M9 NILS-R",
}
METHOD_NAMES_BY_LABEL = {label: name for name, label in METHOD_LABELS.items()}


def _read_many(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _size_values(frame: pd.DataFrame) -> list[str]:
    return ["All"] + [str(int(value)) for value in sorted(frame["n"].dropna().unique())]


def _status_for_size(status: pd.DataFrame, size: str) -> pd.DataFrame:
    if size == "All":
        return status.copy()
    return status[status["n"].eq(int(size))].copy()


def _customer_run_summary(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(columns=["instance_id", "method_name"])
    rows = detail.copy()
    rows["method_name"] = rows["method"].astype(str).map(METHOD_NAMES_BY_LABEL).fillna(rows["method"].astype(str))
    rows["priority_class"] = rows["priority_class"].astype(str).str.lower()
    for column in [
        "tardiness",
        "weighted_tardiness",
        "sync_wait_time",
        "battery_slack_after_sortie",
    ]:
        if column in rows:
            rows[column] = _number(rows[column])
    rows["is_drone_served"] = _bool_series(rows["is_drone_served"])
    rows["is_high_priority"] = _bool_series(rows["is_high_priority"])
    rows["drone_id_present"] = rows["drone_id"].notna() & rows["drone_id"].astype(str).ne("")
    rows["high_tardiness"] = rows["tardiness"].where(rows["is_high_priority"], 0.0)
    rows["medium_tardiness"] = rows["tardiness"].where(rows["priority_class"].eq("medium"), 0.0)
    rows["low_tardiness"] = rows["tardiness"].where(rows["priority_class"].eq("low"), 0.0)
    grouped = rows.groupby(["instance_id", "method_name"], as_index=False).agg(
        detail_customers=("customer_id", "nunique"),
        detail_drone_customers=("is_drone_served", "sum"),
        detail_active_drones=("drone_id_present", "sum"),
        high_tardiness_detail=("high_tardiness", "sum"),
        medium_tardiness_detail=("medium_tardiness", "sum"),
        low_tardiness_detail=("low_tardiness", "sum"),
        detail_sync_wait=("sync_wait_time", "sum"),
        detail_battery_slack=("battery_slack_after_sortie", "mean"),
    )
    active = rows[rows["drone_id_present"]].groupby(["instance_id", "method_name"])["drone_id"].nunique()
    grouped = grouped.drop(columns=["detail_active_drones"]).merge(
        active.rename("active_drones_used"),
        on=["instance_id", "method_name"],
        how="left",
    )
    grouped["active_drones_used"] = grouped["active_drones_used"].fillna(0)
    return grouped


def _build_decision_tables(
    status: pd.DataFrame,
    detail: pd.DataFrame,
    source: pd.DataFrame,
    output_dir: Path,
) -> None:
    work = status.copy()
    work["feasible"] = _bool_series(work["feasible"])
    for column in [
        "objective",
        "priority_tardiness_cost",
        "high_priority_tardiness",
        "max_high_priority_tardiness",
        "p90_high_priority_tardiness",
        "drone_served_customers",
        "drone_share",
        "cpu_time_seconds",
        "truck_travel_cost",
        "drone_travel_cost",
        "waiting_sync_cost",
        "truck_route_distance_km",
        "drone_flight_distance_km",
        "total_truck_waiting_time",
        "total_drone_waiting_time",
        "average_synchronization_delay",
        "maximum_synchronization_delay",
        "average_battery_usage_ratio",
        "maximum_battery_usage_ratio",
        "makespan_minutes",
    ]:
        if column in work:
            work[column] = _number(work[column])
    work = work.merge(_customer_run_summary(detail), on=["instance_id", "method_name"], how="left")
    source_meta = source[["instance_id", "number_of_drones"]].drop_duplicates("instance_id")
    work = work.merge(source_meta, on="instance_id", how="left")
    work["active_drones_used"] = work["active_drones_used"].fillna(0)
    work["detail_drone_customers"] = work["detail_drone_customers"].fillna(work["drone_served_customers"])
    work["detail_drone_customers"] = work["detail_drone_customers"].fillna(0)
    work["drone_share"] = work["drone_share"].fillna(0)
    truck_reference = work[work["method_name"].eq("truck_only")][
        ["instance_id", "priority_tardiness_cost", "high_priority_tardiness"]
    ].rename(
        columns={
            "priority_tardiness_cost": "truck_reference_ptc",
            "high_priority_tardiness": "truck_reference_hpt",
        }
    )
    work = work.merge(truck_reference, on="instance_id", how="left")
    work["service_value"] = work["truck_reference_ptc"] - work["priority_tardiness_cost"]
    work["service_value_per_customer"] = work["service_value"].where(
        work["detail_drone_customers"].gt(0)
    ) / work["detail_drone_customers"].where(work["detail_drone_customers"].gt(0))
    work["service_value_per_active_drone"] = work["service_value"].where(
        work["active_drones_used"].gt(0)
    ) / work["active_drones_used"].where(work["active_drones_used"].gt(0))
    work["service_value_per_drone_km"] = work["service_value"].where(
        work["drone_flight_distance_km"].gt(0)
    ) / work["drone_flight_distance_km"].where(work["drone_flight_distance_km"].gt(0))
    work["high_priority_protection_ratio"] = (
        work["truck_reference_hpt"] - work["high_priority_tardiness"]
    ).where(work["truck_reference_hpt"].gt(0)) / work["truck_reference_hpt"].where(
        work["truck_reference_hpt"].gt(0)
    )
    work["method"] = work["method_name"].map(METHOD_LABELS).fillna(work["method_name"])
    work["size"] = work["n"]

    quality_columns = [
        "method",
        "size",
        "feasible_runs",
        "mean_objective",
        "mean_priority_tardiness_cost",
        "mean_high_priority_tardiness",
        "mean_medium_priority_tardiness",
        "mean_low_priority_tardiness",
        "mean_drone_share",
        "mean_drone_distance",
        "mean_truck_distance",
        "mean_total_sync_wait",
        "mean_cpu",
        "service_value_per_drone_customer",
        "service_value_per_drone_km",
        "high_priority_protection_ratio",
    ]
    quality_rows: list[dict[str, object]] = []
    use_columns = [
        "method",
        "size",
        "feasible_runs",
        "available_drones",
        "mean_active_drones_used",
        "mean_active_drones_used_ratio",
        "mean_drone_served_customers",
        "mean_drone_share",
        "mean_high_priority_served_by_drone_share",
        "mean_medium_priority_served_by_drone_share",
        "mean_low_priority_served_by_drone_share",
        "mean_priority_tardiness_cost",
        "mean_high_priority_tardiness",
        "mean_p90_high_priority_tardiness",
        "mean_drone_distance",
        "mean_sync_wait",
        "mean_battery_slack_after_sortie",
        "service_value_per_drone_customer",
        "service_value_per_active_drone",
        "service_value_per_drone_km",
    ]
    use_rows: list[dict[str, object]] = []
    for method_name, method_frame in work.groupby("method_name", sort=True):
        for size in _size_values(work):
            frame = _status_for_size(method_frame, size)
            feasible = frame[frame["feasible"]].copy()
            detail_feasible = feasible
            if feasible.empty:
                continue
            detail_lookup = detail.copy()
            detail_lookup["method_name"] = detail_lookup["method"].astype(str).map(METHOD_NAMES_BY_LABEL).fillna(detail_lookup["method"].astype(str))
            detail_lookup = detail_lookup.merge(
                feasible[["instance_id", "method_name"]],
                on=["instance_id", "method_name"],
                how="inner",
            )
            detail_lookup["priority_class"] = detail_lookup["priority_class"].astype(str).str.lower()
            detail_lookup["is_drone_served"] = _bool_series(detail_lookup["is_drone_served"])
            priority_shares = {}
            for priority in ["high", "medium", "low"]:
                subset = detail_lookup[detail_lookup["priority_class"].eq(priority)]
                priority_shares[priority] = (
                    subset.groupby("instance_id")["is_drone_served"].mean().mean()
                    if not subset.empty
                    else np.nan
                )
            quality_rows.append(
                {
                    "method": METHOD_LABELS.get(method_name, method_name),
                    "size": size,
                    "feasible_runs": int(len(feasible)),
                    "mean_objective": feasible["objective"].mean(),
                    "mean_priority_tardiness_cost": feasible["priority_tardiness_cost"].mean(),
                    "mean_high_priority_tardiness": feasible["high_priority_tardiness"].mean(),
                    "mean_medium_priority_tardiness": feasible["medium_tardiness_detail"].mean(),
                    "mean_low_priority_tardiness": feasible["low_tardiness_detail"].mean(),
                    "mean_drone_share": feasible["drone_share"].mean(),
                    "mean_drone_distance": feasible["drone_flight_distance_km"].mean(),
                    "mean_truck_distance": feasible["truck_route_distance_km"].mean(),
                    "mean_total_sync_wait": (feasible["total_truck_waiting_time"] + feasible["total_drone_waiting_time"]).mean(),
                    "mean_cpu": feasible["cpu_time_seconds"].mean(),
                    "service_value_per_drone_customer": feasible["service_value_per_customer"].mean(),
                    "service_value_per_drone_km": feasible["service_value_per_drone_km"].mean(),
                    "high_priority_protection_ratio": feasible["high_priority_protection_ratio"].mean(),
                }
            )
            use_rows.append(
                {
                    "method": METHOD_LABELS.get(method_name, method_name),
                    "size": size,
                    "feasible_runs": int(len(feasible)),
                    "available_drones": feasible["number_of_drones"].mean(),
                    "mean_active_drones_used": feasible["active_drones_used"].mean(),
                    "mean_active_drones_used_ratio": (
                        feasible["active_drones_used"] / feasible["number_of_drones"].replace(0, np.nan)
                    ).mean(),
                    "mean_drone_served_customers": feasible["detail_drone_customers"].mean(),
                    "mean_drone_share": feasible["drone_share"].mean(),
                    "mean_high_priority_served_by_drone_share": priority_shares["high"],
                    "mean_medium_priority_served_by_drone_share": priority_shares["medium"],
                    "mean_low_priority_served_by_drone_share": priority_shares["low"],
                    "mean_priority_tardiness_cost": feasible["priority_tardiness_cost"].mean(),
                    "mean_high_priority_tardiness": feasible["high_priority_tardiness"].mean(),
                    "mean_p90_high_priority_tardiness": feasible["p90_high_priority_tardiness"].mean(),
                    "mean_drone_distance": feasible["drone_flight_distance_km"].mean(),
                    "mean_sync_wait": (feasible["total_truck_waiting_time"] + feasible["total_drone_waiting_time"]).mean(),
                    "mean_battery_slack_after_sortie": feasible["detail_battery_slack"].mean(),
                    "service_value_per_drone_customer": feasible["service_value_per_customer"].mean(),
                    "service_value_per_active_drone": feasible["service_value_per_active_drone"].mean(),
                    "service_value_per_drone_km": feasible["service_value_per_drone_km"].mean(),
                }
            )
    pd.DataFrame(quality_rows, columns=quality_columns).to_csv(
        output_dir / "table_decision_quality_existing_methods.csv", index=False
    )
    pd.DataFrame(use_rows, columns=use_columns).to_csv(
        output_dir / "table_drone_use_decision_quality.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--shard-prefix",
        default="priority_service_speed_m9_full_20260827",
        help="Prefix used by the raw shard directories under outputs/.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    root = args.project_dir / "outputs"
    shard_dirs = sorted(root.glob(f"{args.shard_prefix}_*_p*"))
    m9_dirs = [path for path in shard_dirs if "_m9_p" in path.name]
    base_dirs = [path for path in shard_dirs if "_base_p" in path.name]
    if len(m9_dirs) != 8 or len(base_dirs) != 8:
        raise RuntimeError(f"Expected 8 M9 and 8 baseline shards, found {len(m9_dirs)} and {len(base_dirs)}")

    status = _read_many([path / "run_status_partial.csv" for path in base_dirs + m9_dirs])
    status = status.drop_duplicates(["instance_id", "method_name"], keep="last")
    status.to_csv(output_dir / "run_status_full.csv", index=False)
    for filename, destination in [
        ("customer_service_detail.csv", "customer_service_detail.csv"),
        ("customer_delivery_observations_partial.csv", "customer_delivery_observations.csv"),
        ("drone_decision_log.csv", "drone_decision_log.csv"),
        ("nils_r_reconstruction_log.csv", "nils_r_reconstruction_log.csv"),
    ]:
        frame = _read_many([path / filename for path in base_dirs + m9_dirs])
        if filename == "customer_service_detail.csv" and not frame.empty:
            frame = frame.drop_duplicates(["instance_id", "method", "customer_id"], keep="last")
        if filename == "customer_delivery_observations_partial.csv" and not frame.empty:
            frame = frame.drop_duplicates(["instance_id", "method_name", "customer_id"], keep="last")
        if filename == "nils_r_reconstruction_log.csv" and not frame.empty:
            frame = frame.drop_duplicates(["instance_id", "reconstruction_attempt"], keep="last")
        frame.to_csv(output_dir / destination, index=False)

    observations = pd.read_csv(output_dir / "customer_delivery_observations.csv")
    priority_table = aggregate_priority_speed(status, observations)
    priority_table.to_csv(output_dir / "table_priority_service_speed_by_method.csv", index=False)
    aggregate_comparisons(output_dir / "run_status_full.csv", output_dir / "comparisons")
    source = pd.read_csv(args.source_output_dir / "tables" / "heuristic_study_runs.csv")
    detail = pd.read_csv(output_dir / "customer_service_detail.csv")
    _build_decision_tables(status, detail, source, output_dir)
    print(f"Merged rows={len(status)} unique_instances={status['instance_id'].nunique()}")
    print(status.groupby("method_name").size().to_string())
    print(f"Output={output_dir}")


if __name__ == "__main__":
    main()


