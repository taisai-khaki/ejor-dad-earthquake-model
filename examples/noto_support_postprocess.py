from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ejor_dad.checkpoint import atomic_write_dataframe
from ejor_dad.fixed_y import evaluate_fixed_y

import noto_access_experiment as noto


RHOS = np.asarray([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])
COLORS = {
    "unrestricted": "#DC2626",
    "support": "#1D4ED8",
    "road": "#059669",
}


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    unrestricted_dir = root / "access_experiment" / "tables"
    support_full_dir = root / "support_preserving_full" / "tables"
    support_pilot_dir = root / "support_preserving_pilot" / "tables"
    m2_dir = root / "support_preserving_m2" / "tables"
    m3_dir = root / "support_preserving_m3" / "tables"
    output_dir = root / "support_preserving_analysis"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    unrestricted = read_table(unrestricted_dir / "table_noto_access_dda_summary.csv")
    unrestricted_support = read_table(
        unrestricted_dir / "table_noto_access_ambiguity_support_diagnostic.csv"
    )
    support = read_table(support_full_dir / "table_noto_support_dda_summary.csv")
    pilot = read_table(support_pilot_dir / "table_noto_support_dda_summary.csv")
    m2 = read_table(m2_dir / "table_noto_support_m2_certification.csv")
    m3 = read_table(m3_dir / "table_noto_support_m3_certification.csv")
    require_complete_rho_sweep(unrestricted, "unrestricted full-grid")
    require_complete_rho_sweep(support, "support-preserving full-grid")
    require_complete_rho_sweep(m2, "support-preserving m=2")
    require_complete_rho_sweep(m3, "support-preserving m=3")

    comparison = build_model_comparison(unrestricted, unrestricted_support, support)
    cap_sensitivity = build_cap_sensitivity(pilot)
    sector_comparison = build_sector_comparison(support)
    numerical_audit = build_numerical_audit(root / "support_preserving_full" / "checkpoints")
    claims = build_claim_diagnostics(support, m2, m3)
    write_table(comparison, table_dir, "table_noto_support_model_comparison")
    write_table(cap_sensitivity, table_dir, "table_noto_support_cap_sensitivity")
    write_table(sector_comparison, table_dir, "table_noto_support_sector_comparison")
    write_table(numerical_audit, table_dir, "table_noto_support_numerical_audit")
    write_table(claims, table_dir, "table_noto_support_claim_diagnostics")
    write_table(m2, table_dir, "table_noto_support_m2_certification")
    write_table(m3, table_dir, "table_noto_support_m3_certification")

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    make_penalty_figure(comparison, figure_dir)
    make_support_figure(comparison, figure_dir)
    make_cap_sensitivity_figure(cap_sensitivity, figure_dir)
    make_road_value_figure(comparison, figure_dir)
    make_sector_figure(sector_comparison, figure_dir, paper_rho=0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create final tables and figures for the support-preserving Noto repair."
    )
    parser.add_argument("--root", default="data_work/noto")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required completed result table is missing: {path}")
    return pd.read_csv(path)


def require_complete_rho_sweep(dataframe: pd.DataFrame, label: str) -> None:
    observed = np.sort(dataframe["rho"].unique())
    if observed.shape != RHOS.shape or not np.allclose(observed, RHOS):
        raise RuntimeError(f"{label} is incomplete: expected rho={RHOS.tolist()}, found {observed.tolist()}")


def build_model_comparison(
    unrestricted: pd.DataFrame,
    unrestricted_support: pd.DataFrame,
    support: pd.DataFrame,
) -> pd.DataFrame:
    unrestricted = unrestricted.sort_values("rho").reset_index(drop=True)
    unrestricted_support = unrestricted_support.sort_values("rho").reset_index(drop=True)
    support = support.sort_values("rho").reset_index(drop=True)
    unrestricted_baseline = float(unrestricted.loc[np.isclose(unrestricted["rho"], 0.0), "best_discretized_objective"].iloc[0])
    support_baseline = float(support.loc[np.isclose(support["rho"], 0.0), "best_objective"].iloc[0])
    return pd.DataFrame(
        {
            "rho": support["rho"],
            "density_cap": support["density_cap"],
            "unrestricted_objective": unrestricted["best_discretized_objective"],
            "support_preserving_objective": support["best_objective"],
            "unrestricted_penalty_from_rho0": unrestricted["best_discretized_objective"]
            - unrestricted_baseline,
            "support_preserving_penalty_from_rho0": support["best_objective"] - support_baseline,
            "penalty_reduction_from_support_repair": (
                unrestricted["best_discretized_objective"] - unrestricted_baseline
            )
            - (support["best_objective"] - support_baseline),
            "unrestricted_best_y_json": unrestricted["best_y_json"],
            "support_preserving_best_y_json": support["best_y_json"],
            "unrestricted_delta_rho_value": unrestricted["delta_rho_value"],
            "support_preserving_delta_rho_value": support["delta_rho_value"],
            "unrestricted_road_value": unrestricted["road_value_over_no_retrofit"],
            "support_preserving_road_value": support["road_value"],
            "unrestricted_mass_added_outside_support": unrestricted_support[
                "mass_added_to_zero_nominal_states"
            ],
            "support_preserving_mass_added_outside_support": support[
                "mass_added_to_zero_nominal_states"
            ],
            "support_preserving_max_density_ratio": support["max_realized_density_ratio"],
            "support_preserving_hardened_link_failure": support["hardened_link_failure_occurs"],
        }
    )


def build_cap_sensitivity(pilot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for density_cap, group in pilot.groupby("density_cap", sort=True):
        group = group.sort_values("rho")
        baseline = float(group.loc[np.isclose(group["rho"], 0.0), "best_objective"].iloc[0])
        for _, row in group.iterrows():
            rows.append(
                {
                    "density_cap": density_cap,
                    "rho": row["rho"],
                    "objective": row["best_objective"],
                    "objective_penalty_from_rho0": row["best_objective"] - baseline,
                    "best_y_json": row["best_y_json"],
                    "delta_rho_value": row["delta_rho_value"],
                    "road_value": row["road_value"],
                    "total_positive_probability_shift": row["total_positive_probability_shift"],
                    "mass_added_to_zero_nominal_states": row["mass_added_to_zero_nominal_states"],
                    "max_realized_density_ratio": row["max_realized_density_ratio"],
                    "density_cap_respected": row["density_cap_respected"],
                }
            )
    return pd.DataFrame(rows)


def build_sector_comparison(support: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for summary in support.sort_values("rho").itertuples(index=False):
        uncapped_instance, _ = noto.build_noto_instance(float(summary.rho))
        instance = replace(
            uncapped_instance,
            ambiguity_density_cap=float(summary.density_cap),
        )
        zero_y = np.zeros(len(instance.links), dtype=float)
        exposure_only = evaluate_fixed_y(
            replace(instance, budget_capacity=0.0),
            zero_y,
            epsilon=1e-5,
            max_iterations=160,
        )
        capacity_only = evaluate_fixed_y(
            replace(instance, budget_renovation=0.0),
            zero_y,
            epsilon=1e-5,
            max_iterations=160,
        )
        no_investment = noto.evaluate_no_investment(instance)
        comparisons = [
            ("no investment", no_investment),
            ("exposure only", float(exposure_only.objective)),
            ("capacity only", float(capacity_only.objective)),
            ("exposure + capacity; no road retrofit", float(summary.no_retrofit_objective)),
            ("all-sector discretized", float(summary.best_objective)),
        ]
        for label, objective in comparisons:
            rows.append(
                {
                    "rho": float(summary.rho),
                    "density_cap": float(summary.density_cap),
                    "comparison": label,
                    "objective": objective,
                    "reduction_from_no_investment": no_investment - objective,
                    "reduction_percent": 100.0 * (no_investment - objective) / no_investment,
                    "gap_to_all_sector": objective - float(summary.best_objective),
                    "best_y_json": json.dumps(json.loads(summary.best_y_json)),
                }
            )
    return pd.DataFrame(rows)


def build_numerical_audit(checkpoint_dir: Path) -> pd.DataFrame:
    records = []
    for path in checkpoint_dir.glob("*__grid_*.json"):
        match = re.search(r"_rho([0-9]+\.[0-9]+)__grid_", path.name)
        if match is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        nominal = np.asarray(payload["nominal_distribution"], dtype=float)
        worst_case = np.asarray(payload["worst_case_distribution"], dtype=float)
        records.append(
            {
                "rho": float(match.group(1)),
                "objective_gap": max(0.0, float(payload["objective"]) - float(payload["lower_bound"])),
                "iterations": int(payload["iterations"]),
                "nominal_sum_error": abs(float(nominal.sum()) - 1.0),
                "worst_case_sum_error": abs(float(worst_case.sum()) - 1.0),
                "minimum_probability": min(float(nominal.min()), float(worst_case.min())),
                "eval_runtime_sec": float(payload.get("eval_runtime_sec", np.nan)),
            }
        )
    data = pd.DataFrame(records)
    if len(data) != 5976:
        raise RuntimeError(f"Expected 5,976 full-grid checkpoints, found {len(data):,}.")
    rows = []
    for scope, group in [(f"rho={rho:.2f}", group) for rho, group in data.groupby("rho", sort=True)] + [
        ("all radii", data)
    ]:
        rows.append(
            {
                "scope": scope,
                "exact_evaluations": len(group),
                "max_objective_lower_bound_gap": float(group["objective_gap"].max()),
                "max_algorithm1_iterations": int(group["iterations"].max()),
                "max_nominal_probability_sum_error": float(group["nominal_sum_error"].max()),
                "max_worst_case_probability_sum_error": float(group["worst_case_sum_error"].max()),
                "minimum_reported_probability": float(group["minimum_probability"].min()),
                "mean_policy_runtime_sec": float(group["eval_runtime_sec"].mean()),
                "max_policy_runtime_sec": float(group["eval_runtime_sec"].max()),
            }
        )
    return pd.DataFrame(rows)


def build_claim_diagnostics(
    support: pd.DataFrame,
    m2: pd.DataFrame,
    m3: pd.DataFrame,
) -> pd.DataFrame:
    max_m2_gap = float(m2["absolute_gap"].max())
    max_m3_gap = float(m3["absolute_gap"].max())
    m2_certified = bool(m2["certified_at_tolerance"].all())
    m3_certified = bool(m3["certified_at_tolerance"].all())
    max_m5_delta = float(np.abs(support["delta_rho_value"]).max())
    max_y_change = float(support["y_diff_norm_from_rho0"].max())
    max_z_change = float(support["z_diff_norm_from_rho0"].max())
    max_w_change = float(support["w_diff_norm_from_rho0"].max())
    max_leakage = float(support["mass_added_to_zero_nominal_states"].max())
    density_excess = float((support["max_realized_density_ratio"] - support["density_cap"]).max())
    total_evaluations = int(support["feasible_evaluated_candidates"].sum())
    return pd.DataFrame(
        [
            {
                "diagnostic": "support preservation",
                "value": max_leakage,
                "criterion": "mass added to zero-nominal states = 0",
                "passed": max_leakage <= 1e-10,
                "implication": "the repaired ambiguity set cannot undo a zero-probability hardened-link state",
            },
            {
                "diagnostic": "density-cap feasibility",
                "value": density_excess,
                "criterion": "maximum realized density ratio does not exceed kappa",
                "passed": density_excess <= 1e-9,
                "implication": "all reported adversarial distributions satisfy the declared cap",
            },
            {
                "diagnostic": "m=2 continuous global certificate",
                "value": max_m2_gap,
                "criterion": "all six global gaps <= 0.1 absolute (<= 0.002% relative); bases disclosed",
                "passed": m2_certified,
                "implication": "the small mechanism-active continuous instance is globally bounded at the declared tolerance",
            },
            {
                "diagnostic": "m=3 continuous gap diagnostic",
                "value": max_m3_gap,
                "criterion": "report the maximum eight-state gap without relabeling it as certified",
                "passed": m3_certified,
                "implication": "m=3 is retained as requested, with its actual certificate status disclosed",
            },
            {
                "diagnostic": "m=5 discrete-grid completeness",
                "value": total_evaluations,
                "criterion": "996 feasible policies at each of six radii",
                "passed": bool((support["feasible_evaluated_candidates"] == 996).all()),
                "implication": "the empirical road policy is globally best on the declared five-level grid",
            },
            {
                "diagnostic": "DDA road-policy change",
                "value": max_y_change,
                "criterion": "positive change from the rho=0 policy",
                "passed": max_y_change > 1e-8,
                "implication": "no policy-change claim is allowed when this diagnostic is false",
            },
            {
                "diagnostic": "DDA switching value",
                "value": max_m5_delta,
                "criterion": "positive delta_rho value",
                "passed": max_m5_delta > 1e-6,
                "implication": "DDA changes evaluation but not policy value when this diagnostic is false",
            },
            {
                "diagnostic": "z and w stability",
                "value": max(max_z_change, max_w_change),
                "criterion": "report numerical norms from the rho=0 solution",
                "passed": max(max_z_change, max_w_change) <= 1e-6,
                "implication": "exposure and capacity decisions are stable across the repaired DDA sweep",
            },
            {
                "diagnostic": "road mechanism activity",
                "value": float(support["road_value"].min()),
                "criterion": "road value remains positive at every radius",
                "passed": bool((support["road_value"] > 0.0).all()),
                "implication": "road retrofit has material practical value even without DDA policy switching",
            },
        ]
    )


def write_table(dataframe: pd.DataFrame, table_dir: Path, stem: str) -> None:
    atomic_write_dataframe(dataframe, table_dir / f"{stem}.csv")
    atomic_write_dataframe(dataframe, table_dir / f"{stem}.tex", kind="latex", escape=True)


def make_penalty_figure(comparison: pd.DataFrame, figure_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(
        comparison["rho"],
        comparison["unrestricted_penalty_from_rho0"],
        marker="o",
        linewidth=2.2,
        color=COLORS["unrestricted"],
        label="Unrestricted TV",
    )
    axis.plot(
        comparison["rho"],
        comparison["support_preserving_penalty_from_rho0"],
        marker="o",
        linewidth=2.2,
        color=COLORS["support"],
        label="Support-preserving TV ($\\kappa=2$)",
    )
    axis.set_xlabel("Ambiguity radius $\\rho$")
    axis.set_ylabel("Objective increase from $\\rho=0$")
    axis.set_title("Ambiguity penalty before and after support repair")
    axis.legend(frameon=False)
    save_figure(figure, figure_dir, "fig_noto_support_01_penalty_comparison")


def make_support_figure(comparison: pd.DataFrame, figure_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    width = 0.018
    axis.bar(
        comparison["rho"] - width / 2,
        comparison["unrestricted_mass_added_outside_support"],
        width=width,
        color=COLORS["unrestricted"],
        label="Unrestricted TV",
    )
    axis.bar(
        comparison["rho"] + width / 2,
        comparison["support_preserving_mass_added_outside_support"],
        width=width,
        color=COLORS["support"],
        label="Support-preserving TV",
    )
    axis.set_xlabel("Ambiguity radius $\\rho$")
    axis.set_ylabel("Mass added outside nominal support")
    axis.set_title("Support leakage is eliminated by the density cap")
    axis.legend(frameon=False)
    save_figure(figure, figure_dir, "fig_noto_support_02_support_leakage")


def make_cap_sensitivity_figure(cap_sensitivity: pd.DataFrame, figure_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    for density_cap, group in cap_sensitivity.groupby("density_cap", sort=True):
        group = group.sort_values("rho")
        axis.plot(
            group["rho"],
            group["objective_penalty_from_rho0"],
            marker="o",
            linewidth=1.9,
            label=f"$\\kappa={density_cap:g}$",
        )
    axis.set_xlabel("Ambiguity radius $\\rho$")
    axis.set_ylabel("Objective increase from $\\rho=0$")
    axis.set_title("Pilot sensitivity to the support-preserving density cap")
    axis.legend(frameon=False, ncol=2)
    save_figure(figure, figure_dir, "fig_noto_support_03_cap_sensitivity")


def make_road_value_figure(comparison: pd.DataFrame, figure_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(
        comparison["rho"],
        comparison["unrestricted_road_value"],
        marker="o",
        linewidth=2.0,
        color=COLORS["unrestricted"],
        label="Unrestricted TV",
    )
    axis.plot(
        comparison["rho"],
        comparison["support_preserving_road_value"],
        marker="o",
        linewidth=2.2,
        color=COLORS["road"],
        label="Support-preserving TV ($\\kappa=2$)",
    )
    axis.set_xlabel("Ambiguity radius $\\rho$")
    axis.set_ylabel("Gain over optimized no-road policy")
    axis.set_title("Road-retrofit value remains material after ambiguity repair")
    axis.legend(frameon=False)
    save_figure(figure, figure_dir, "fig_noto_support_04_road_value")


def make_sector_figure(sector_comparison: pd.DataFrame, figure_dir: Path, paper_rho: float) -> None:
    data = sector_comparison[np.isclose(sector_comparison["rho"], paper_rho)].copy()
    order = [
        "no investment",
        "exposure only",
        "capacity only",
        "exposure + capacity; no road retrofit",
        "all-sector discretized",
    ]
    data["comparison"] = pd.Categorical(data["comparison"], categories=order, ordered=True)
    data = data.sort_values("comparison")
    labels = [
        "No investment",
        "Exposure only",
        "Capacity only",
        "Exposure + capacity",
        "All sectors",
    ]
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    bars = axis.barh(labels, data["reduction_from_no_investment"], color="#1D4ED8")
    axis.invert_yaxis()
    axis.set_xlabel("Reduction in worst-case modeled loss")
    axis.set_title(f"Sector contribution after ambiguity repair ($\\rho={paper_rho:.2f}$)")
    axis.bar_label(bars, fmt="%.1f", padding=4, fontsize=8)
    save_figure(figure, figure_dir, "fig_noto_support_05_sector_contribution")


def save_figure(figure: plt.Figure, figure_dir: Path, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(figure_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(figure_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
