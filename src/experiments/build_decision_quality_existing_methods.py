"""Build decision-quality summary table for existing M1--M8 methods."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METHOD_ORDER = {
    "truck_only": ("M1 Truck-only", 1),
    "simple_drone": ("M2 Simple drone", 2),
    "paired_baseline": ("M3 Paired baseline", 3),
    "no_unpairing": ("M4 Flexible same-truck", 4),
    "nils": ("M5 NILS", 5),
    "nils_no_local_search": ("M6 NILS-no-local-search", 6),
    "nils_no_perturbation": ("M7 NILS-no-perturbation", 7),
    "feasibility_aware_alns": ("M8 Feasibility-aware ALNS", 8),
}

MAIN_METHODS = [
    "truck_only",
    "simple_drone",
    "paired_baseline",
    "no_unpairing",
    "nils",
    "nils_no_local_search",
    "nils_no_perturbation",
]

M8_METHOD = "feasibility_aware_alns"

REQUIRED_COLUMNS = {
    "instance_id",
    "scenario_id",
    "method_name",
    "n",
    "objective_for_reporting",
    "priority_tardiness_cost",
    "high_priority_tardiness_minutes",
    "medium_priority_tardiness_minutes",
    "low_priority_tardiness_minutes",
    "drone_service_share",
    "drone_flight_distance_km",
    "truck_route_distance_km",
    "total_truck_waiting_time",
    "total_drone_waiting_time",
    "customers_served_by_drone",
    "cpu_time_seconds",
}


def _finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray([float(value) for value in values if pd.notna(value) and np.isfinite(float(value))], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def _feasible_mask(df: pd.DataFrame) -> pd.Series:
    if "is_feasible_run" in df.columns:
        return df["is_feasible_run"].fillna(False).astype(bool)
    if "final_solution_feasibility_flag" in df.columns:
        return df["final_solution_feasibility_flag"].fillna(False).astype(bool)
    if "reporting_included" in df.columns:
        return df["reporting_included"].fillna(False).astype(bool)
    return pd.Series(True, index=df.index)


def _require_columns(df: pd.DataFrame, path: Path) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _load_method_rows(main_path: Path, alns_path: Path) -> pd.DataFrame:
    main = pd.read_csv(main_path)
    alns = pd.read_csv(alns_path)
    _require_columns(main, main_path)
    _require_columns(alns, alns_path)

    main_rows = main[main["method_name"].isin(MAIN_METHODS)].copy()
    alns_rows = alns[alns["method_name"] == M8_METHOD].copy()
    rows = pd.concat([main_rows, alns_rows], ignore_index=True)
    rows["feasible_flag"] = _feasible_mask(rows)
    rows["total_sync_wait"] = (
        pd.to_numeric(rows["total_truck_waiting_time"], errors="coerce").fillna(0.0)
        + pd.to_numeric(rows["total_drone_waiting_time"], errors="coerce").fillna(0.0)
    )
    return rows


def _load_truck_reference(main_path: Path) -> pd.DataFrame:
    main = pd.read_csv(main_path)
    _require_columns(main, main_path)
    ref = main[main["method_name"] == "truck_only"][
        [
            "instance_id",
            "scenario_id",
            "priority_tardiness_cost",
            "high_priority_tardiness_minutes",
        ]
    ].copy()
    ref = ref.rename(
        columns={
            "priority_tardiness_cost": "truck_ref_priority_tardiness_cost",
            "high_priority_tardiness_minutes": "truck_ref_high_priority_tardiness",
        }
    )
    return ref.drop_duplicates(["instance_id", "scenario_id"])


def _summarise(group: pd.DataFrame, method_name: str, size: str) -> dict[str, float | int | str]:
    feasible = group[group["feasible_flag"]].copy()
    ptc_ref_mean = _finite_mean(feasible["truck_ref_priority_tardiness_cost"])
    hpt_ref_mean = _finite_mean(feasible["truck_ref_high_priority_tardiness"])
    ptc_mean = _finite_mean(feasible["priority_tardiness_cost"])
    hpt_mean = _finite_mean(feasible["high_priority_tardiness_minutes"])
    drone_customers_mean = _finite_mean(feasible["customers_served_by_drone"])
    drone_distance_mean = _finite_mean(feasible["drone_flight_distance_km"])
    service_numerator = ptc_ref_mean - ptc_mean
    hpt_numerator = hpt_ref_mean - hpt_mean
    label, order = METHOD_ORDER[method_name]
    return {
        "method": label,
        "_method_order": order,
        "size": size,
        "_size_order": -1 if size == "All" else int(size),
        "feasible_runs": int(len(feasible)),
        "mean_objective": _finite_mean(feasible["objective_for_reporting"]),
        "mean_priority_tardiness_cost": ptc_mean,
        "mean_high_priority_tardiness": hpt_mean,
        "mean_medium_priority_tardiness": _finite_mean(feasible["medium_priority_tardiness_minutes"]),
        "mean_low_priority_tardiness": _finite_mean(feasible["low_priority_tardiness_minutes"]),
        "mean_drone_share": _finite_mean(feasible["drone_service_share"]),
        "mean_drone_distance": drone_distance_mean,
        "mean_truck_distance": _finite_mean(feasible["truck_route_distance_km"]),
        "mean_total_sync_wait": _finite_mean(feasible["total_sync_wait"]),
        "mean_cpu": _finite_mean(feasible["cpu_time_seconds"]),
        "service_value_per_drone_customer": service_numerator / max(1.0, drone_customers_mean),
        "service_value_per_drone_km": service_numerator / max(1e-6, drone_distance_mean),
        "high_priority_protection_ratio": hpt_numerator / max(1e-6, hpt_ref_mean),
    }


def build_table(main_path: Path, alns_path: Path) -> pd.DataFrame:
    rows = _load_method_rows(main_path, alns_path)
    ref = _load_truck_reference(main_path)
    merged = rows.merge(ref, on=["instance_id", "scenario_id"], how="left", validate="many_to_one")
    if merged["truck_ref_priority_tardiness_cost"].isna().any():
        missing = int(merged["truck_ref_priority_tardiness_cost"].isna().sum())
        raise ValueError(f"Missing truck-only references for {missing} method rows")

    summaries = []
    for method_name in METHOD_ORDER:
        method_rows = merged[merged["method_name"] == method_name]
        if method_rows.empty:
            raise ValueError(f"No rows found for {method_name}")
        summaries.append(_summarise(method_rows, method_name, "All"))
        for size, size_rows in method_rows.groupby("n", dropna=False):
            summaries.append(_summarise(size_rows, method_name, str(int(size))))

    table = pd.DataFrame(summaries).sort_values(["_method_order", "_size_order"]).reset_index(drop=True)
    return table.drop(columns=["_method_order", "_size_order"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-runs",
        type=Path,
        default=Path("outputs/doptimal_full_beta010_tables5_14_rerun2_20260601_144538/tables/heuristic_study_runs.csv"),
    )
    parser.add_argument(
        "--alns-runs",
        type=Path,
        default=Path("outputs/alns_comparison_beta010_full720/tables/alns_full_matched_runs.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/decision_quality_existing_methods/table_decision_quality_existing_methods.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = build_table(args.main_runs, args.alns_runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, float_format="%.6f")
    print(f"Wrote {len(table)} rows to {args.output}")


if __name__ == "__main__":
    main()
