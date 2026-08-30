from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import noto_mechanism_full_grid as mechanism
from critical_revision_common import (
    DEVELOPMENT_RHOS,
    GRID_LEVELS,
    atomic_json,
    base_output_root,
    build_mechanism_instances,
    candidate_grid,
    finish_run_metadata,
    json_string,
    load_source_records,
    model_hash,
    save_table,
    source_commit,
    write_progress_log,
    write_run_metadata,
    write_status,
)
from ejor_dad.checkpoint import CheckpointStore
from ejor_dad.fixed_y import evaluate_fixed_plan, evaluate_fixed_y
from ejor_dad.recourse import solve_capability


MODELS = ("M0", "M1", "M2", "M3", "M4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-output-dir", default=str(base_output_root()))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rhos", default=','.join(str(value) for value in DEVELOPMENT_RHOS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def source_plan_from_base(base_output_dir: Path, model: str, rho: float) -> dict | None:
    if model == "M4":
        path = base_output_dir / "correlated_facility_separated_capability_marginal_v2" / "tables" / "table_noto_correlated_facility.csv"
        if not path.exists():
            return None
        table = pd.read_csv(path)
        table = table.iloc[np.argmin(np.abs(table["rho"].to_numpy(dtype=float) - rho))]
        if abs(float(table.rho) - rho) > 1e-8:
            return None
        top_path = base_output_dir / "correlated_facility_separated_capability_marginal_v2" / "tables" / "table_noto_dense_full_grid_top10.csv"
        second_objective = float(table.objective)
        if top_path.exists():
            top = pd.read_csv(top_path)
            top = top[(np.isclose(top.rho, rho)) & (top.rank == 2)]
            if not top.empty:
                second_objective = float(top.iloc[0].objective)
        return {"objective": float(table.objective), "second_objective": second_objective, "y": json.loads(table.selected_y_json), "z": json.loads(table.selected_z_json), "w": json.loads(table.selected_w_json), "maximum_oracle_gap": 0.0}
    path = base_output_dir / "mechanism_separated_capability_marginal_v2" / "tables" / "table_noto_mechanism_ablation_full_grid.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    rows = table[(table.model == model) & np.isclose(table.rho, rho)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {"objective": float(row.objective), "second_objective": float(row.second_objective) if "second_objective" in row.index else float(row.objective), "y": json.loads(row.selected_y_json), "z": json.loads(row.selected_z_json), "w": json.loads(row.selected_w_json), "maximum_oracle_gap": 0.0}


def optimize_source_plan(instance, model: str, rho: float, output_dir: Path, base_output_dir: Path, resume: bool) -> dict:
    root = output_dir / "source_plans"
    cache = CheckpointStore(root / model)
    key = f"rho{rho:.3f}"
    if resume and cache.exists(key):
        return cache.load(key)
    stored = source_plan_from_base(base_output_dir, model, rho)
    if stored is not None:
        cache.save(key, stored)
        return stored
    best = None
    candidates = candidate_grid(instance)
    for index, y in candidates:
        checkpoint_key = f"rho{rho:.3f}_grid{index:04d}"
        if resume and cache.exists(checkpoint_key):
            payload = cache.load(checkpoint_key)
        else:
            result = evaluate_fixed_y(instance, y, epsilon=1e-6, max_iterations=500)
            payload = {"status": "feasible", "objective": result.objective, "lower_bound": result.lower_bound, "oracle_gap": result.objective - result.lower_bound, "y": result.y.tolist(), "z": result.z.tolist(), "w": result.w.tolist()}
            cache.save(checkpoint_key, payload)
        if payload.get("status") == "feasible" and (best is None or float(payload["objective"]) < float(best["objective"])):
            best = payload
    if best is None:
        raise RuntimeError(f"No feasible source plan for {model} at rho={rho}.")
    cache.save(key, best)
    return best


def capability_check(instance, plan: dict) -> tuple[bool, int]:
    z = np.asarray(plan["z"], dtype=float)
    w = np.asarray(plan["w"], dtype=float)
    y = np.asarray(plan["y"], dtype=float)
    count = 0
    for state in instance.states:
        if not np.any(instance.service_fractions_for_state(state) > 1e-12):
            continue
        if not solve_capability(instance, state, z, w, y=y).feasible:
            count += 1
    return count == 0, count


def regret_status(regret: float, maximum_oracle_gap: float, policy_margin: float, admissible: bool) -> str:
    if not admissible:
        return "inadmissible_transfer"
    if regret <= 10.0 * max(maximum_oracle_gap, 1e-12):
        return "numerically_zero"
    if regret <= policy_margin + 1e-8:
        return "below_policy_margin"
    return "above_policy_margin"


def heatmap(rows: pd.DataFrame, rho: float, value: str, path: Path) -> None:
    matrix = rows[rows.rho == rho].pivot(index="source_model", columns="evaluation_model", values=value).reindex(index=MODELS, columns=MODELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis")
    ax.set_xticks(range(len(MODELS)), MODELS)
    ax.set_yticks(range(len(MODELS)), MODELS)
    ax.set_xlabel("Evaluation model")
    ax.set_ylabel("Source model")
    ax.set_title(f"{value}, rho={rho:g}")
    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            value_text = matrix.iloc[i, j]
            if pd.notna(value_text):
                ax.text(j, i, f"{value_text:.3g}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    base_output_dir = Path(args.base_output_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    rhos = tuple(float(value) for value in args.rhos.split(',') if value.strip())
    if not (output_dir / "run_manifest.json").exists():
        write_run_metadata(output_dir, experiment="value_of_road_facility_dependence", parameters=vars(args), expected_work={"models": MODELS, "rhos": rhos, "cross_evaluations": len(MODELS) ** 2 * len(rhos)})
    rows: list[dict] = []
    if args.resume and (output_dir / "tables" / "table_noto_mechanism_cross_evaluation.csv").exists():
        rows = pd.read_csv(output_dir / "tables" / "table_noto_mechanism_cross_evaluation.csv").to_dict("records")
    total = len(MODELS) ** 2 * len(rhos)
    for rho in rhos:
        instances = build_mechanism_instances(base_output_dir, rho)
        plans = {model: optimize_source_plan(instances[model], model, rho, output_dir, base_output_dir, args.resume) for model in MODELS}
        own_capability = {model: capability_check(instances[model], plans[model]) for model in MODELS}
        for source_model in MODELS:
            source = plans[source_model]
            for evaluation_model in MODELS:
                if any(float(row["rho"]) == rho and row["source_model"] == source_model and row["evaluation_model"] == evaluation_model for row in rows):
                    continue
                evaluation = instances[evaluation_model]
                benchmark = plans[evaluation_model]
                benchmark_value = float(benchmark["objective"])
                road = evaluate_fixed_y(evaluation, source["y"], epsilon=1e-6, max_iterations=500)
                full = evaluate_fixed_plan(evaluation, source["z"], source["w"], source["y"])
                capability_feasible, infeasible_count = capability_check(evaluation, source)
                raw_road_regret = float(road.objective - benchmark_value)
                raw_full_regret = float(full.objective - benchmark_value)
                if raw_road_regret < -1e-7 or raw_full_regret < -1e-7:
                    raise RuntimeError(f"Negative material regret for {source_model}->{evaluation_model} at rho={rho}.")
                road_regret = max(0.0, raw_road_regret)
                full_regret = max(0.0, raw_full_regret)
                margin = max(float(benchmark.get("second_objective", benchmark_value) - benchmark_value), 0.0)
                maximum_gap = max(float(road.objective - road.lower_bound), float(source.get("maximum_oracle_gap", 0.0)))
                row = {
                    "rho": rho,
                    "source_model": source_model,
                    "evaluation_model": evaluation_model,
                    "source_y_json": json_string(source["y"]),
                    "source_z_json": json_string(source["z"]),
                    "source_w_json": json_string(source["w"]),
                    "evaluation_model_benchmark": benchmark_value,
                    "road_reoptimized_objective": float(road.objective),
                    "road_regret": road_regret,
                    "road_regret_percent": 100.0 * road_regret / max(1.0, abs(benchmark_value)),
                    "complete_plan_objective": float(full.objective),
                    "complete_plan_regret": full_regret,
                    "complete_plan_regret_percent": 100.0 * full_regret / max(1.0, abs(benchmark_value)),
                    "capability_feasible": capability_feasible,
                    "infeasible_design_basis_state_count": infeasible_count,
                    "maximum_oracle_gap": maximum_gap,
                    "regret_divided_by_m4_policy_margin": road_regret / max(margin, 1e-12),
                    "regret_divided_by_oracle_tolerance": road_regret / max(1e-5, maximum_gap),
                    "regret_status": regret_status(road_regret, maximum_gap, margin, True),
                    "complete_regret_status": regret_status(full_regret, maximum_gap, margin, capability_feasible),
                    "source_plan_own_capability_feasible": own_capability[source_model][0],
                }
                rows.append(row)
                save_table(rows, output_dir / "tables" / "table_noto_mechanism_cross_evaluation.csv", ["rho", "source_model", "evaluation_model"])
                write_status(output_dir / "status.json", status="running", block="mechanism_cross_evaluation", completed=len(rows), total=total, rho=rho, source_model=source_model, evaluation_model=evaluation_model)
    frame = pd.DataFrame(rows)
    contrasts = frame[(frame.source_model.isin(["M2", "M3"])) & (frame.evaluation_model == "M4")].copy()
    contrast_rows = []
    for _, row in contrasts.iterrows():
        contrast_rows.extend([
            {"rho": row.rho, "contrast": f"{row.source_model}->M4", "transfer_type": "road_reoptimized", "regret": row.road_regret, "regret_percent": row.road_regret_percent, "regret_over_policy_margin": row.regret_divided_by_m4_policy_margin, "regret_over_oracle_gap": row.regret_divided_by_oracle_tolerance, "status": row.regret_status},
            {"rho": row.rho, "contrast": f"{row.source_model}->M4", "transfer_type": "complete_plan", "regret": row.complete_plan_regret, "regret_percent": row.complete_plan_regret_percent, "regret_over_policy_margin": row.regret_divided_by_m4_policy_margin, "regret_over_oracle_gap": row.complete_plan_regret / max(1e-5, row.maximum_oracle_gap), "status": row.complete_regret_status},
        ])
    save_table(contrast_rows, output_dir / "tables" / "table_noto_value_shared_dependence.csv", ["rho", "contrast", "transfer_type"])
    for rho in rhos:
        heatmap(frame, rho, "road_regret", output_dir / "figures" / f"road_policy_regret_rho_{rho:.3f}.png")
        heatmap(frame, rho, "complete_plan_regret", output_dir / "figures" / f"complete_plan_regret_rho_{rho:.3f}.png")
    atomic_json(output_dir / "validation.json", {"status": "passed", "rows": len(frame), "models": MODELS, "rhos": rhos})
    write_status(output_dir / "status.json", status="completed", block="mechanism_cross_evaluation", completed=len(frame), total=total, rows=len(frame))
    finish_run_metadata(output_dir, status="completed", runtime_seconds=time.perf_counter() - started, extra={"rows": len(frame), "rhos": rhos})


if __name__ == "__main__":
    main()


