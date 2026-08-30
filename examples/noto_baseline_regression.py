from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from critical_revision_common import atomic_json, base_output_root, build_m4_instance, candidate_grid, finish_run_metadata, json_string, save_table, write_run_metadata, write_status
from ejor_dad.fixed_y import evaluate_fixed_y

EXPECTED = {0.0: (1070.808480778403, [0.25, 1.0, 1.0, 0.0, 0.0]), 0.125: (1174.574248900284, [0.0, 1.0, 1.0, 0.0, 0.25]), 0.25: (1222.787875223676, [0.0, 1.0, 1.0, 0.0, 0.25])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-output-dir", default=str(base_output_root()))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    base = Path(args.base_output_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.joinpath("tables").mkdir(parents=True, exist_ok=True)
    if not (output / "run_manifest.json").exists():
        write_run_metadata(output, experiment="noto_baseline_regression", parameters=vars(args), expected_work={"rhos": list(EXPECTED), "states": 128, "design_basis_states": 18, "budget_feasible_tier_policies": 996})
    rows = []
    candidate_count = None
    passed = True
    for rho, (expected_objective, expected_y) in EXPECTED.items():
        instance = build_m4_instance(base, rho)
        candidate_count = len(candidate_grid(instance))
        result = evaluate_fixed_y(instance, np.asarray(expected_y, dtype=float), epsilon=1e-6, max_iterations=500)
        probability_error = abs(float(result.nominal_distribution.sum()) - 1.0)
        objective_error = abs(float(result.objective) - expected_objective)
        y_error = float(np.max(np.abs(result.y - np.asarray(expected_y, dtype=float))))
        row = {"rho": rho, "objective": result.objective, "expected_objective": expected_objective, "objective_error": objective_error, "selected_y_json": json_string(result.y), "expected_y_json": json_string(expected_y), "y_max_error": y_error, "probability_sum_error": probability_error, "state_count": len(instance.states), "design_basis_state_count": len(instance.critical_service_state_ids or ()), "budget_feasible_tier_count": candidate_count, "oracle_gap": result.objective - result.lower_bound, "pass": objective_error <= 1e-7 and y_error <= 1e-8 and probability_error <= 1e-8 and len(instance.states) == 128 and len(instance.critical_service_state_ids or ()) == 18 and candidate_count == 996}
        rows.append(row)
        passed = passed and bool(row["pass"])
        write_status(output / "status.json", status="running", block="baseline_regression", rho=rho, completed=len(rows), total=len(EXPECTED), pass_so_far=passed)
    save_table(rows, output / "tables" / "table_noto_baseline_regression.csv", ["rho"])
    summary = {"status": "passed" if passed else "failed", "rows": len(rows), "candidate_count": candidate_count, "required_state_count": 128, "required_design_basis_state_count": 18}
    atomic_json(output / "baseline_gate.json", summary)
    write_status(output / "status.json", **summary, block="baseline_regression")
    finish_run_metadata(output, status=summary["status"], runtime_seconds=time.perf_counter() - started, extra=summary)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
