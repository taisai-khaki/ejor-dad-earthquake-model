from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ejor_dad.checkpoint import atomic_write_dataframe


EXPECTED_FEASIBLE = {"pilot": 83, "full": 996}
TOTAL_GRID = {"pilot": 243, "full": 3125}


def main() -> None:
    args = parse_args()
    output_name = "access_experiment_pilot" if args.mode == "pilot" else "access_experiment"
    output_dir = Path("data_work/noto") / output_name
    table_dir = output_dir / "tables"
    summary = pd.read_csv(table_dir / "table_noto_access_dda_summary.csv")
    sectors = pd.read_csv(table_dir / "table_noto_access_sector_comparison.csv")
    near_optimal = pd.read_csv(table_dir / "table_noto_access_near_optimal.csv")
    probability_shifts = pd.read_csv(table_dir / "table_noto_access_probability_shifts.csv")
    zones = pd.read_csv(Path("data_work/noto/prepared/noto_zones.csv"), dtype={"municipality_code": str})
    centers = pd.read_csv(Path("data_work/noto/prepared/noto_centers.csv"), dtype={"municipality_code": str})
    corridors = pd.read_csv(Path("data_work/noto/prepared/noto_corridors.csv"))

    write_table(build_main_results(sectors, args.paper_rho), table_dir, "table_noto_access_main_results")
    write_table(build_link_decisions(summary, corridors), table_dir, "table_noto_access_link_decisions")
    write_table(build_zone_decisions(summary, zones), table_dir, "table_noto_access_zone_decisions")
    write_table(build_capacity_decisions(summary, centers), table_dir, "table_noto_access_capacity_decisions")
    write_table(
        build_claim_diagnostics(summary, near_optimal),
        table_dir,
        "table_noto_access_claim_diagnostics",
    )
    write_table(
        build_ambiguity_support_diagnostic(summary, probability_shifts, corridors),
        table_dir,
        "table_noto_access_ambiguity_support_diagnostic",
    )
    write_table(
        build_numerical_audit(output_dir / "checkpoints", args.mode),
        table_dir,
        "table_noto_access_numerical_audit",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-ready Noto result and numerical-audit tables.")
    parser.add_argument("--mode", choices=sorted(EXPECTED_FEASIBLE), default="full")
    parser.add_argument("--paper-rho", type=float, default=0.10)
    return parser.parse_args()


def parse_vector(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def build_main_results(sectors: pd.DataFrame, paper_rho: float) -> pd.DataFrame:
    selected = sectors[np.isclose(sectors["rho"], paper_rho)].copy()
    if selected.empty:
        raise ValueError(f"No sector-comparison results found for rho={paper_rho:.2f}.")
    no_investment = float(selected.loc[selected["comparison"] == "no investment", "objective"].iloc[0])
    best_objective = float(selected["objective"].min())
    grid_label = "three-level" if selected["mode"].iloc[0] == "pilot" else "five-level"
    selected["reduction_percent"] = 100.0 * selected["reduction_from_no_investment"] / no_investment
    selected["gap_to_best_percent"] = 100.0 * selected["gap_to_best"] / best_objective
    selected["result_status"] = np.where(
        selected["comparison"] == "all-sector discretized",
        f"global optimum over every budget-feasible policy on the declared {grid_label} grid",
        "exact fixed-road sector benchmark",
    )
    return selected[
        [
            "rho",
            "mode",
            "comparison",
            "objective",
            "reduction_from_no_investment",
            "reduction_percent",
            "gap_to_best",
            "gap_to_best_percent",
            "result_status",
        ]
    ]


def build_link_decisions(summary: pd.DataFrame, corridors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_row in summary.itertuples(index=False):
        retrofit = parse_vector(summary_row.best_y_json)
        if len(retrofit) != len(corridors):
            raise ValueError("The best-y vector does not match the prepared Noto corridor order.")
        for corridor_index, corridor in enumerate(corridors.itertuples(index=False)):
            selected_y = float(retrofit[corridor_index])
            rows.append(
                {
                    "rho": float(summary_row.rho),
                    "mode": summary_row.mode,
                    "link_id": corridor.link_id,
                    "corridor": corridor.label,
                    "normal_minutes": float(corridor.normal_minutes),
                    "disrupted_minutes": float(corridor.disrupted_minutes),
                    "failure_penalty_minutes": float(corridor.failure_penalty_minutes),
                    "baseline_failure_probability": float(corridor.baseline_failure_probability),
                    "selected_retrofit_y": selected_y,
                    "post_retrofit_failure_probability": float(corridor.baseline_failure_probability)
                    * (1.0 - selected_y),
                    "retrofit_cost": float(corridor.retrofit_cost),
                    "retrofit_budget_used": float(corridor.retrofit_cost) * selected_y,
                }
            )
    return pd.DataFrame(rows)


def build_zone_decisions(summary: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_row in summary.itertuples(index=False):
        renovation = parse_vector(summary_row.best_z_json)
        if len(renovation) != len(zones):
            raise ValueError("The best-z vector does not match the prepared Noto zone order.")
        for zone_index, zone in enumerate(zones.itertuples(index=False)):
            selected_z = float(renovation[zone_index])
            rows.append(
                {
                    "rho": float(summary_row.rho),
                    "mode": summary_row.mode,
                    "zone_id": zone.zone_id,
                    "municipality": zone.municipality_name_en,
                    "population": float(zone.population),
                    "households": float(zone.households),
                    "collapse_fraction": float(zone.collapse_fraction),
                    "baseline_at_risk_population": float(zone.at_risk_population),
                    "selected_renovation_z": selected_z,
                    "residual_at_risk_population": float(zone.at_risk_population) * (1.0 - selected_z),
                    "renovation_cost": float(zone.renovation_cost),
                    "renovation_budget_used": float(zone.renovation_cost) * selected_z,
                }
            )
    return pd.DataFrame(rows)


def build_capacity_decisions(summary: pd.DataFrame, centers: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_row in summary.itertuples(index=False):
        expansion = parse_vector(summary_row.best_w_json)
        if len(expansion) != len(centers):
            raise ValueError("The best-w vector does not match the prepared Noto center order.")
        for center_index, center in enumerate(centers.itertuples(index=False)):
            selected_w = float(expansion[center_index])
            rows.append(
                {
                    "rho": float(summary_row.rho),
                    "mode": summary_row.mode,
                    "center_id": center.center_id,
                    "municipality": center.municipality_name_en,
                    "reported_beds": float(center.reported_beds),
                    "operational_share": float(center.operational_share),
                    "existing_capacity": float(center.existing_capacity),
                    "selected_capacity_expansion_w": selected_w,
                    "post_investment_capacity": float(center.existing_capacity) + selected_w,
                    "capacity_budget_used": float(center.capacity_unit_cost) * selected_w,
                }
            )
    return pd.DataFrame(rows)


def build_claim_diagnostics(summary: pd.DataFrame, near_optimal: pd.DataFrame) -> pd.DataFrame:
    near_wide = near_optimal.pivot(index="rho", columns="threshold_percent", values="near_optimal_policy_count")
    rho0_row = summary.loc[np.isclose(summary["rho"], 0.0)].iloc[0]
    rho0_z = parse_vector(rho0_row["best_z_json"])
    rho0_w = parse_vector(rho0_row["best_w_json"])
    rows: list[dict[str, Any]] = []
    for summary_row in summary.itertuples(index=False):
        y_changed = float(summary_row.y_diff_norm_from_rho0) > 1e-9
        z_diff_norm = float(np.linalg.norm(parse_vector(summary_row.best_z_json) - rho0_z))
        w_diff_norm = float(np.linalg.norm(parse_vector(summary_row.best_w_json) - rho0_w))
        relative_delta = float(summary_row.delta_rho_value) / max(float(summary_row.best_discretized_objective), 1.0)
        if not y_changed:
            interpretation = "DDA changes evaluation but not the best discretized retrofit policy"
        elif relative_delta >= 0.001:
            interpretation = "DDA changes the discretized policy and produces measurable value"
        else:
            interpretation = "DDA changes the optimizer in a near-flat policy landscape"
        near_counts = near_wide.loc[summary_row.rho]
        rows.append(
            {
                "rho": float(summary_row.rho),
                "mode": summary_row.mode,
                "best_discretized_objective": float(summary_row.best_discretized_objective),
                "best_y_json": summary_row.best_y_json,
                "y_diff_norm_from_rho0": float(summary_row.y_diff_norm_from_rho0),
                "z_diff_norm_from_rho0": z_diff_norm,
                "w_diff_norm_from_rho0": w_diff_norm,
                "all_decisions_same_as_rho0": not y_changed and z_diff_norm <= 1e-8 and w_diff_norm <= 1e-6,
                "delta_rho_value": float(summary_row.delta_rho_value),
                "delta_rho_percent_of_best": 100.0 * relative_delta,
                "road_value_over_no_retrofit": float(summary_row.road_value_over_no_retrofit),
                "road_value_percent": 100.0
                * float(summary_row.road_value_over_no_retrofit)
                / float(summary_row.no_retrofit_zw_objective),
                "near_optimal_count_0.01pct": int(near_counts.loc[0.01]),
                "near_optimal_count_0.05pct": int(near_counts.loc[0.05]),
                "near_optimal_count_0.10pct": int(near_counts.loc[0.10]),
                "near_optimal_count_0.50pct": int(near_counts.loc[0.50]),
                "allowed_interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def build_ambiguity_support_diagnostic(
    summary: pd.DataFrame,
    probability_shifts: pd.DataFrame,
    corridors: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_row in summary.itertuples(index=False):
        rho_shifts = probability_shifts[np.isclose(probability_shifts["rho"], summary_row.rho)].copy()
        positive = rho_shifts[rho_shifts["probability_shift"] > 1e-10]
        support_expansion = positive[positive["nominal_probability"] <= 1e-12]
        total_positive_shift = float(positive["probability_shift"].sum())
        zero_nominal_shift = float(support_expansion["probability_shift"].sum())
        selected_y = parse_vector(summary_row.best_y_json)
        fully_retrofitted = {
            corridor.link_id
            for corridor_index, corridor in enumerate(corridors.itertuples(index=False))
            if selected_y[corridor_index] >= 1.0 - 1e-9
        }
        failed_in_added_states: set[str] = set()
        for failed_links_json in support_expansion["failed_links_json"]:
            failed_in_added_states.update(json.loads(failed_links_json))
        hardened_failures = sorted(fully_retrofitted.intersection(failed_in_added_states))
        rows.append(
            {
                "rho": float(summary_row.rho),
                "mode": summary_row.mode,
                "total_positive_probability_shift": total_positive_shift,
                "mass_added_to_zero_nominal_states": zero_nominal_shift,
                "share_added_outside_nominal_support": zero_nominal_shift / total_positive_shift
                if total_positive_shift > 0.0
                else 0.0,
                "support_expansion_occurs": zero_nominal_shift > 1e-10,
                "fully_retrofitted_links_json": json.dumps(sorted(fully_retrofitted)),
                "fully_retrofitted_links_failed_in_added_states_json": json.dumps(hardened_failures),
                "hardened_link_failure_occurs": bool(hardened_failures),
                "diagnostic_interpretation": (
                    "TV ambiguity adds mass to zero-nominal states, including failures of fully retrofitted links"
                    if hardened_failures
                    else "No support-expanding hardened-link failure at this radius"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_numerical_audit(checkpoint_dir: Path, mode: str) -> pd.DataFrame:
    grouped: dict[float, list[dict[str, Any]]] = {}
    pattern = re.compile(r"rho([0-9]+\.[0-9]+)__grid_")
    for checkpoint_path in checkpoint_dir.glob("*__grid_*.json"):
        match = pattern.search(checkpoint_path.name)
        if not match:
            continue
        rho = float(match.group(1))
        with checkpoint_path.open(encoding="utf-8") as checkpoint_file:
            grouped.setdefault(rho, []).append(json.load(checkpoint_file))

    rows: list[dict[str, Any]] = []
    for rho, payloads in sorted(grouped.items()):
        absolute_gaps = [abs(float(payload["objective"]) - float(payload["lower_bound"])) for payload in payloads]
        relative_gaps = [
            gap / max(abs(float(payload["objective"])), 1.0) for gap, payload in zip(absolute_gaps, payloads)
        ]
        nominal_errors = [abs(sum(payload["nominal_distribution"]) - 1.0) for payload in payloads]
        worst_case_errors = [abs(sum(payload["worst_case_distribution"]) - 1.0) for payload in payloads]
        iterations = [int(payload.get("iterations", 0)) for payload in payloads]
        rows.append(
            {
                "rho": rho,
                "mode": mode,
                "total_grid_candidates": TOTAL_GRID[mode],
                "expected_feasible_candidates": EXPECTED_FEASIBLE[mode],
                "audited_feasible_candidates": len(payloads),
                "grid_complete": len(payloads) == EXPECTED_FEASIBLE[mode],
                "max_absolute_master_gap": max(absolute_gaps),
                "max_relative_master_gap": max(relative_gaps),
                "mean_iterations": float(np.mean(iterations)),
                "max_iterations": max(iterations),
                "max_nominal_probability_sum_error": max(nominal_errors),
                "max_worst_case_probability_sum_error": max(worst_case_errors),
                "minimum_z": min(min(payload["z"]) for payload in payloads),
                "minimum_w": min(min(payload["w"]) for payload in payloads),
                "all_master_gaps_below_1e-5": max(absolute_gaps) <= 1e-5,
            }
        )
    return pd.DataFrame(rows)


def write_table(dataframe: pd.DataFrame, table_dir: Path, stem: str) -> None:
    atomic_write_dataframe(dataframe, table_dir / f"{stem}.csv")
    atomic_write_dataframe(dataframe, table_dir / f"{stem}.tex", kind="latex", escape=True)


if __name__ == "__main__":
    main()
