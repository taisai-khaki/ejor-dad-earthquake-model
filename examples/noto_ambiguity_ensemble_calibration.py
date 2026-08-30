from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critical_revision_common import BASE_RHOS, atomic_json, base_output_root, build_m4_instance, candidate_grid, finish_run_metadata, json_string, model_hash, save_table, write_run_metadata, write_status
from ejor_dad.checkpoint import CheckpointStore
from ejor_dad.fixed_y import evaluate_fixed_y
from ejor_dad.certification import budget_intersecting_grid_cells
from ejor_dad.monotone_bb import OracleEvaluation, run_monotone_box_bb

REGIMES = ("normal", "north", "central", "widespread")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="inputs/regime_weight_ensemble.csv")
    parser.add_argument("--base-output-dir", default=str(base_output_root()))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_weights(path: Path) -> tuple[pd.DataFrame, list[dict]]:
    columns = ["source_id", "source_type", "source_citation_or_file", "assessment_date", *REGIMES, "mapping_notes", "empirical_status"]
    if not path.exists():
        return pd.DataFrame(columns=columns), [{"status": "blocked_missing_regime_weight_ensemble", "reason": f"Missing input: {path}"}]
    frame = pd.read_csv(path, keep_default_na=False)
    errors = []
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        return frame, [{"status": "blocked_missing_regime_weight_ensemble", "reason": f"Missing columns: {missing}"}]
    for row_number, row in frame.iterrows():
        if all(str(row.get(column, "")).strip() == "" for column in REGIMES) and str(row.get("source_id", "")).strip() == "":
            continue
        try:
            weights = np.asarray([float(row[name]) for name in REGIMES], dtype=float)
            error = abs(float(weights.sum()) - 1.0)
            if np.any(weights < -1e-12) or error > 1e-10:
                errors.append({"row": int(row_number), "source_id": row.source_id, "reason": "weights must be nonnegative and sum to one", "sum_error": error})
        except (TypeError, ValueError) as exc:
            errors.append({"row": int(row_number), "source_id": row.get("source_id", ""), "reason": str(exc)})
    if frame[frame.empirical_status.astype(str).str.lower().eq("externally_supported")].empty:
        errors.append({"status": "blocked_missing_regime_weight_ensemble", "reason": "No externally_supported rows were supplied; illustrative rows cannot anchor calibration."})
    return frame, errors


def containment(frame: pd.DataFrame, baseline: dict[str, float]) -> list[dict]:
    rows = []
    for _, row in frame.iterrows():
        try:
            weights = {name: float(row[name]) for name in REGIMES}
        except (TypeError, ValueError):
            continue
        diffs = {name: abs(weights[name] - baseline[name]) for name in REGIMES}
        ratios = {name: weights[name] / baseline[name] for name in REGIMES}
        rows.append({"source_id": row.source_id, "source_type": row.source_type, "source_citation_or_file": row.source_citation_or_file, "assessment_date": row.assessment_date, "empirical_status": row.empirical_status, **{f"weight_{name}": weights[name] for name in REGIMES}, "rho_required": 0.5 * sum(diffs.values()), "kappa_required": max(ratios.values()), "rho_binding_regime": max(diffs, key=diffs.get), "kappa_binding_regime": max(ratios, key=ratios.get), "sum_error": sum(weights.values()) - 1.0})
    return rows


def nearest_grid(value: float, grid: tuple[float, ...]) -> float:
    return min(grid, key=lambda candidate: abs(candidate - value))


def run_anchored(args: argparse.Namespace, base: Path, output: Path, rows: list[dict], ensemble_rho: float, ensemble_kappa: float) -> list[dict]:
    anchors = [(f"source_{row['source_id']}", float(row["rho_required"]), float(row["kappa_required"]), "source_specific") for row in rows]
    anchors.append(("joint_ensemble", ensemble_rho, ensemble_kappa, "joint_ensemble"))
    results = []
    for anchor_id, rho, kappa, anchor_type in anchors:
        instance = build_m4_instance(base, rho, density_cap=kappa)
        cache = CheckpointStore(output / "checkpoints" / anchor_id)
        best = None
        candidates = candidate_grid(instance)
        for completed, (index, y) in enumerate(candidates, start=1):
            key = f"grid{index:04d}"
            if args.resume and cache.exists(key):
                payload = cache.load(key)
            else:
                try:
                    result = evaluate_fixed_y(instance, y, epsilon=1e-6, max_iterations=500)
                    payload = {"status": "feasible", "objective": result.objective, "lower_bound": result.lower_bound, "oracle_gap": result.objective - result.lower_bound, "y": result.y.tolist(), "z": result.z.tolist(), "w": result.w.tolist(), "candidate_index": index}
                except Exception as error:
                    payload = {"status": "failed", "candidate_index": index, "y": y.tolist(), "error": str(error)}
                cache.save(key, payload)
            if payload.get("status") == "feasible" and (best is None or payload["objective"] < best["objective"]):
                best = payload
            if completed % 50 == 0 or completed == len(candidates):
                write_status(output / "status.json", status="running", block="ambiguity_anchored", anchor_id=anchor_id, completed=completed, total=len(candidates), rho_required=rho, kappa_required=kappa)
        if best is None:
            results.append({"anchor_id": anchor_id, "anchor_type": anchor_type, "rho_required": rho, "kappa_required": kappa, "status": "failed_no_feasible_policy"})
        else:
            results.append({"anchor_id": anchor_id, "anchor_type": anchor_type, "rho_required": rho, "kappa_required": kappa, "nearest_paper_rho": nearest_grid(rho, BASE_RHOS), "nearest_paper_kappa": nearest_grid(kappa, (1.5, 2.0, 3.0)), "status": "completed", "objective": best["objective"], "lower_bound": best["lower_bound"], "oracle_gap": best["oracle_gap"], "selected_y_json": json_string(best["y"]), "selected_z_json": json_string(best["z"]), "selected_w_json": json_string(best["w"]), "candidate_count": len(candidates), "model_hash": model_hash(base, density_cap=kappa)})
    return results


def run_ensemble_continuous(args: argparse.Namespace, base: Path, output: Path, rho: float, kappa: float, seed_y: np.ndarray) -> dict:
    instance = build_m4_instance(base, rho, density_cap=kappa)
    cells = budget_intersecting_grid_cells(instance.retrofit_costs, instance.budget_retrofit, (0.0, 0.25, 0.50, 0.75, 1.0))
    cache = CheckpointStore(output / "continuous_checkpoint")
    memory = {}
    def oracle(y: np.ndarray) -> OracleEvaluation:
        key = "corner_" + "_".join(f"{value:.12f}" for value in np.round(y, 12))
        vector = tuple(np.round(y, 12).tolist())
        if vector in memory:
            return memory[vector]
        if args.resume and cache.exists(key):
            payload = cache.load(key)
            evaluation = OracleEvaluation("feasible", payload["objective"], payload["lower_bound"], np.asarray(payload["y"]), np.asarray(payload["z"]), np.asarray(payload["w"]), int(payload.get("iterations", 0)), float(payload.get("oracle_gap", 0.0)))
        else:
            try:
                result = evaluate_fixed_y(instance, y, epsilon=1e-6, max_iterations=500, enforce_retrofit_budget=False)
                evaluation = OracleEvaluation("feasible", result.objective, result.lower_bound, result.y, result.z, result.w, result.iterations, result.objective - result.lower_bound)
            except RuntimeError as error:
                result = evaluate_fixed_y(instance, y, epsilon=1e-6, max_iterations=1000, enforce_retrofit_budget=False)
                evaluation = OracleEvaluation("feasible", result.objective, result.lower_bound, result.y, result.z, result.w, result.iterations, result.objective - result.lower_bound)
            cache.save(key, {"status": evaluation.status, "objective": evaluation.objective, "lower_bound": evaluation.lower_bound, "oracle_gap": evaluation.oracle_gap, "y": evaluation.y.tolist(), "z": evaluation.z.tolist(), "w": evaluation.w.tolist(), "iterations": evaluation.iterations})
        memory[vector] = evaluation
        write_status(output / "status.json", status="running", block="ambiguity_continuous", rho=rho, kappa=kappa, oracle_calls=len(memory))
        return evaluation
    seed = evaluate_fixed_y(instance, seed_y, epsilon=1e-6, max_iterations=500, enforce_retrofit_budget=True)
    result = run_monotone_box_bb(initial_boxes=[(cell.lower, cell.upper) for cell in cells], costs=instance.retrofit_costs, budget=instance.budget_retrofit, oracle=oracle, incumbent=OracleEvaluation("feasible", seed.objective, seed.lower_bound, seed.y, seed.z, seed.w, seed.iterations, seed.objective - seed.lower_bound), rel_gap_target=0.001)
    row = {"rho": rho, "density_cap": kappa, "incumbent_objective": result.incumbent.objective, "global_lower_bound": result.global_lower_bound, "absolute_gap": result.absolute_gap, "relative_gap_percent": result.relative_gap_percent, "converged": result.converged, "termination_reason": result.termination_reason, "best_y_json": json_string(result.incumbent.y), "unique_oracle_calls": result.unique_oracle_calls, "model_hash": model_hash(base, density_cap=kappa)}
    save_table([row], output / "tables" / "table_noto_ambiguity_ensemble_continuous.csv", ["rho"])
    return row


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    base = Path(args.base_output_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.joinpath("tables").mkdir(parents=True, exist_ok=True)
    if not (output / "run_manifest.json").exists():
        write_run_metadata(output, experiment="empirical_ambiguity_ensemble_calibration", parameters=vars(args), expected_work={"full_tier_enumeration": True})
    frame, errors = read_weights(Path(args.weights).resolve())
    baseline = {"normal": 0.70, "north": 0.15, "central": 0.10, "widespread": 0.05}
    rows = containment(frame, baseline)
    frame.to_csv(output / "tables" / "table_noto_regime_weight_ensemble.csv", index=False)
    save_table(rows, output / "tables" / "table_noto_ambiguity_containment.csv", ["source_id"])
    supported = [row for row in rows if str(row.get("empirical_status", "")).lower() == "externally_supported" and abs(float(row.get("sum_error", 1.0))) <= 1e-10]
    if errors or not supported:
        payload = {"status": "blocked_missing_regime_weight_ensemble", "errors": errors, "externally_supported_valid_rows": len(supported), "input": str(args.weights)}
        atomic_json(output / "blocked.json", payload)
        write_status(output / "status.json", **payload, block="ambiguity_calibration")
        finish_run_metadata(output, status="blocked_missing_regime_weight_ensemble", runtime_seconds=time.perf_counter() - started, extra=payload)
        return
    ensemble_rho = max(float(row["rho_required"]) for row in supported)
    ensemble_kappa = max(float(row["kappa_required"]) for row in supported)
    rho_binding = max(supported, key=lambda row: float(row["rho_required"]))
    kappa_binding = max(supported, key=lambda row: float(row["kappa_required"]))
    anchored = run_anchored(args, base, output, supported, ensemble_rho, ensemble_kappa)
    save_table(anchored, output / "tables" / "table_noto_anchored_ambiguity_results.csv", ["anchor_id"])
    joint = next(row for row in anchored if row["anchor_id"] == "joint_ensemble")
    continuous = None
    if joint.get("status") == "completed":
        continuous = run_ensemble_continuous(args, base, output, ensemble_rho, ensemble_kappa, np.asarray(json.loads(joint["selected_y_json"]), dtype=float))
        atomic_json(output / "ensemble_continuous.json", continuous)
    summary = {"status": "completed", "rho_ensemble": ensemble_rho, "kappa_ensemble": ensemble_kappa, "rho_binding_source": rho_binding["source_id"], "kappa_binding_source": kappa_binding["source_id"]}
    atomic_json(output / "calibration_summary.json", summary)
    write_status(output / "status.json", **summary, block="ambiguity_calibration")
    finish_run_metadata(output, status="completed", runtime_seconds=time.perf_counter() - started, extra=summary)


if __name__ == "__main__":
    main()
