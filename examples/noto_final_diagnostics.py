from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import noto_practical_resilience_experiment as practical
from ejor_dad.channels import decompose_optimized_road_channels, decompose_road_retrofit_channels
from ejor_dad.fixed_y import evaluate_fixed_y
from ejor_dad.identification import fixed_y_capacity_ranges
from ejor_dad.tv import minimum_saturation_radius


def model_args(design: dict, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        mode=design["mode"], density_cap=float(design["density_cap"]),
        residual_failure_ratio=float(design["residual_failure_ratio"]),
        failure_delay_reduction=float(design["failure_delay_reduction"]),
        retrofit_budget_scale=float(design["retrofit_budget_scale"]),
        time_sensitive_fraction=float(design["time_sensitive_fraction"]),
        immediate_loss_fraction=float(design["immediate_loss_fraction"]),
        capacity_throughput_per_bed=design.get("capacity_throughput_per_bed"),
        response_threshold_minutes=design.get("response_threshold_minutes"),
        graded_response=bool(design.get("graded_response", False)),
        output_dir=str(output_dir), workers=1, force=False,
    )


def select_from_top10(top: pd.DataFrame, rho: float) -> tuple[np.ndarray, float, int, float]:
    block = top[np.isclose(top["rho"], rho)].copy()
    minimum = float(block["objective"].min())
    tolerance = max(1e-8, 1e-10 * max(1.0, abs(minimum)))
    tied = block[block["objective"] <= minimum + tolerance].copy()
    tied["policy_key"] = tied["selected_y_json"].map(lambda value: tuple(json.loads(value)))
    selected = tied.sort_values("policy_key", kind="stable").iloc[0]
    return np.asarray(json.loads(selected["selected_y_json"]), dtype=float), minimum, len(tied), tolerance


def run(output_dir: Path) -> None:
    design = json.loads((output_dir / "run_design.json").read_text(encoding="utf-8"))
    args = model_args(design, output_dir)
    top = pd.read_csv(output_dir / "tables" / "table_noto_practical_top10.csv")
    selection_rows, fixed_rows, optimized_rows, saturation_rows, capacity_rows = [], [], [], [], []
    for rho in sorted(float(value) for value in design["rho_values"]):
        y, minimum, tie_count, tolerance = select_from_top10(top, rho)
        instance, _ = practical.build_instance(rho, args)
        selected = evaluate_fixed_y(instance, y, epsilon=1e-8, max_iterations=240)
        fixed = decompose_road_retrofit_channels(instance, selected.z, selected.w, y)
        optimized = decompose_optimized_road_channels(instance, y, epsilon=1e-7, max_iterations=240)
        saturation = minimum_saturation_radius(
            selected.nominal_distribution, selected.state_losses,
            density_cap=float(instance.ambiguity_density_cap),
        )
        common = {"rho": rho, "selected_y_json": json.dumps(y.tolist())}
        selection_rows.append(common | {
            "numerical_minimum": minimum, "selected_replay_objective": selected.objective,
            "selected_minus_numerical_minimum": selected.objective - minimum,
            "tie_tolerance": tolerance, "tied_policy_count": tie_count,
            "tie_break_rule": "lexicographically smallest y among objective ties",
        })
        fixed_rows.append(common | {
            "no_retrofit_objective": fixed.no_retrofit.objective,
            "conditional_only_objective": fixed.conditional_consequence.objective,
            "full_dda_objective": fixed.decision_dependent_probability.objective,
            "conditional_improvement": fixed.conditional_consequence_improvement,
            "probability_improvement": fixed.probability_channel_improvement,
            "total_improvement": fixed.total_road_improvement,
            "scope": "selected z,w held fixed",
        })
        optimized_rows.append(common | {
            "A_no_retrofit_reoptimized_zw": optimized.no_retrofit.objective,
            "B_conditional_reoptimized_zw": optimized.conditional_consequence.objective,
            "C_full_dda_reoptimized_zw": optimized.decision_dependent_probability.objective,
            "conditional_improvement": optimized.conditional_consequence_improvement,
            "probability_improvement": optimized.probability_channel_improvement,
            "total_improvement": optimized.total_road_improvement,
            "decomposition_error": optimized.total_road_improvement - optimized.conditional_consequence_improvement - optimized.probability_channel_improvement,
            "scope": "z,w reoptimized in A, B, and C",
        })
        for capacity_range in fixed_y_capacity_ranges(
            instance, y, objective_tolerance=tolerance, separation_tolerance=1e-7
        ):
            capacity_rows.append(common | {
                "center_id": capacity_range.center_id,
                "capacity_minimum": capacity_range.minimum,
                "capacity_maximum": capacity_range.maximum,
                "range_width": capacity_range.maximum - capacity_range.minimum,
                "objective_equivalence_tolerance": tolerance,
                "minimum_iterations": capacity_range.minimum_iterations,
                "maximum_iterations": capacity_range.maximum_iterations,
                "scope": "selected y fixed; z and w reoptimized",
            })
        saturation_rows.append(common | {
            "selected_objective": selected.objective,
            "cap_only_value": saturation.cap_only_value,
            "minimum_saturation_radius": saturation.minimum_radius,
            "universal_redundancy_radius": 1.0 - 1.0 / float(instance.ambiguity_density_cap),
            "saturation_distribution_json": json.dumps(saturation.distribution.tolist()),
        })
    tables = output_dir / "tables"
    for name, rows in [
        ("table_noto_deterministic_selection", selection_rows),
        ("table_noto_fixed_zw_road_channels_final", fixed_rows),
        ("table_noto_optimized_road_channels", optimized_rows),
        ("table_noto_saturation_radius", saturation_rows),
        ("table_noto_capacity_ranges", capacity_rows),
    ]:
        pd.DataFrame(rows).to_csv(tables / f"{name}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    run(Path(parser.parse_args().output_dir))

