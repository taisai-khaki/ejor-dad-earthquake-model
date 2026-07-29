from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import noto_access_experiment as noto
from ejor_dad import evaluate_fixed_plan
from ejor_dad.checkpoint import atomic_write_dataframe


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    design = load_json(output_dir / "run_design.json")
    summary_path = output_dir / "tables" / "table_noto_practical_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing practical-run summary: {summary_path}")
    summary = pd.read_csv(summary_path).sort_values("rho")
    if summary.empty or not np.isclose(summary["rho"].astype(float), 0.0).any():
        raise RuntimeError("The practical-run summary must include rho=0.")

    reference = summary.loc[np.isclose(summary["rho"].astype(float), 0.0)].iloc[0]
    reference_y = parse_vector(reference.best_y_json)
    reference_z = parse_vector(reference.best_z_json)
    reference_w = parse_vector(reference.best_w_json)
    rows: list[dict[str, Any]] = []
    for selected in summary.itertuples(index=False):
        rho = float(selected.rho)
        instance = build_instance(rho, design)
        selected_y = parse_vector(selected.best_y_json)
        selected_z = parse_vector(selected.best_z_json)
        selected_w = parse_vector(selected.best_w_json)
        reference_result = evaluate_fixed_plan(instance, reference_z, reference_w, reference_y)
        selected_result = evaluate_fixed_plan(instance, selected_z, selected_w, selected_y)
        y_distance = float(np.linalg.norm(selected_y - reference_y, ord=1))
        z_distance = float(np.linalg.norm(selected_z - reference_z, ord=1))
        w_distance = float(np.linalg.norm(selected_w - reference_w, ord=1))
        adjustment_value = float(reference_result.objective - selected_result.objective)
        changed = max(y_distance, z_distance, w_distance) > args.atol
        if not changed:
            interpretation = "same complete policy"
        elif abs(adjustment_value) <= args.atol:
            interpretation = "alternate optimal policy (zero-value tie)"
        elif adjustment_value > args.atol:
            interpretation = "value-improving reoptimization"
        else:
            interpretation = "unexpected: selected policy is worse than rho=0 policy"
        rows.append(
            {
                "rho": rho,
                "rho0_complete_policy_objective": float(reference_result.objective),
                "selected_complete_policy_objective": float(selected_result.objective),
                "complete_policy_adjustment_value": adjustment_value,
                "rho0_complete_policy_nominal_objective": float(reference_result.nominal_objective),
                "selected_complete_policy_nominal_objective": float(selected_result.nominal_objective),
                "y_l1_distance_from_rho0": y_distance,
                "z_l1_distance_from_rho0": z_distance,
                "w_l1_distance_from_rho0": w_distance,
                "complete_policy_changed": changed,
                "interpretation": interpretation,
                "rho0_y_json": json.dumps(reference_y.tolist()),
                "rho0_z_json": json.dumps(reference_z.tolist()),
                "rho0_w_json": json.dumps(reference_w.tolist()),
                "selected_y_json": json.dumps(selected_y.tolist()),
                "selected_z_json": json.dumps(selected_z.tolist()),
                "selected_w_json": json.dumps(selected_w.tolist()),
            }
        )
    table = pd.DataFrame(rows)
    table_dir = output_dir / "tables"
    atomic_write_dataframe(table, table_dir / "table_noto_complete_policy_adjustment.csv")
    atomic_write_dataframe(
        table,
        table_dir / "table_noto_complete_policy_adjustment.tex",
        kind="latex",
        escape=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the entire rho=0 decision at each completed practical Noto radius "
            "and distinguish value-improving reoptimization from alternate optimal bases."
        )
    )
    parser.add_argument("--output-dir", default="data_work/noto/practical_resilience")
    parser.add_argument("--atol", type=float, default=1e-7)
    return parser.parse_args()


def build_instance(rho: float, design: dict[str, Any]):
    instance, _ = noto.build_noto_instance(
        rho,
        residual_failure_ratio=float(design.get("residual_failure_ratio", 0.0)),
        failure_delay_reduction=float(design.get("failure_delay_reduction", 0.0)),
        time_sensitive_fraction=float(design.get("time_sensitive_fraction", 1.0)),
        immediate_loss_fraction=float(design.get("immediate_loss_fraction", 0.0)),
        capacity_throughput_per_bed=design.get("capacity_throughput_per_bed"),
        response_threshold_minutes=design.get("response_threshold_minutes"),
    )
    return replace(
        instance,
        ambiguity_density_cap=design.get("density_cap"),
        budget_retrofit=float(design.get("retrofit_budget_scale", 1.0)) * instance.budget_retrofit,
    )


def parse_vector(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing run design: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
