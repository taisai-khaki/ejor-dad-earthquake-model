from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import psutil
except ImportError:
    psutil = None
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import noto_correlated_validation_postprocess as validation
from critical_revision_common import (
    BASE_RHOS,
    DEVELOPMENT_RHOS,
    GRID_LEVELS,
    atomic_json,
    base_output_root,
    build_m4_instance,
    file_digest,
    json_string,
    load_design,
    model_hash,
    save_table,
    source_commit,
    write_progress_log,
    write_run_metadata,
    write_status,
)
from ejor_dad.certification import budget_intersecting_grid_cells, validate_upper_corner_certificate_instance
from ejor_dad.checkpoint import CheckpointStore, atomic_write_dataframe
from ejor_dad.fixed_y import evaluate_fixed_y
from ejor_dad.monotone_bb import OracleEvaluation, OracleIncomplete, classify_cover, optimizer_cover, run_monotone_box_bb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-output-dir", default=str(base_output_root()))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rhos", default=','.join(str(value) for value in DEVELOPMENT_RHOS))
    parser.add_argument("--initial-partition", choices=("grid_seeded", "root"), default="grid_seeded")
    parser.add_argument("--relative-gap", type=float, default=0.001)
    parser.add_argument("--absolute-gap", type=float, default=0.0)
    parser.add_argument("--oracle-epsilon", type=float, default=1e-6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stretch", action="store_true")
    return parser.parse_args()


def rho_values(text: str) -> tuple[float, ...]:
    return tuple(float(value.strip()) for value in text.split(',') if value.strip())


def payload_to_eval(payload: dict, y: np.ndarray) -> OracleEvaluation:
    return OracleEvaluation(
        status=str(payload.get("status", "feasible")),
        objective=None if payload.get("objective") is None else float(payload["objective"]),
        lower_bound=None if payload.get("lower_bound") is None else float(payload["lower_bound"]),
        y=np.asarray(payload.get("y", y), dtype=float),
        z=None if payload.get("z") is None else np.asarray(payload["z"], dtype=float),
        w=None if payload.get("w") is None else np.asarray(payload["w"], dtype=float),
        iterations=int(payload.get("iterations", 0)),
        oracle_gap=float(payload.get("oracle_gap", 0.0)),
        payload=payload,
        error=payload.get("error"),
    )


def eval_payload(result: OracleEvaluation) -> dict:
    return {
        "status": result.status,
        "objective": result.objective,
        "lower_bound": result.lower_bound,
        "oracle_gap": result.oracle_gap,
        "y": None if result.y is None else result.y.tolist(),
        "z": None if result.z is None else result.z.tolist(),
        "w": None if result.w is None else result.w.tolist(),
        "iterations": result.iterations,
        "error": result.error,
    }


def corner_key(rho: float, model_digest: str, y: np.ndarray) -> str:
    vector = '_'.join(f"{value:.12f}".replace('-', 'm').replace('.', 'p') for value in np.asarray(y, dtype=float))
    return f"rho{rho:.6f}_model{model_digest[:16]}_y{vector}"


def solve_with_retry(instance, y: np.ndarray, epsilon: float, cache: CheckpointStore, key: str) -> OracleEvaluation:
    try:
        result = evaluate_fixed_y(instance, y, epsilon=epsilon, max_iterations=500, enforce_retrofit_budget=False)
        evaluation = OracleEvaluation(
            "feasible", result.objective, result.lower_bound, result.y, result.z, result.w,
            result.iterations, result.objective - result.lower_bound,
        )
        cache.save(key, eval_payload(evaluation))
        return evaluation
    except RuntimeError as error:
        message = str(error).lower()
        if "infeasible" not in message and "master lp failed" not in message and "did not converge" not in message:
            raise
        try:
            result = evaluate_fixed_y(instance, y, epsilon=epsilon, max_iterations=1000, enforce_retrofit_budget=False)
            evaluation = OracleEvaluation(
                "feasible", result.objective, result.lower_bound, result.y, result.z, result.w,
                result.iterations, result.objective - result.lower_bound,
            )
            cache.save(key, eval_payload(evaluation))
            return evaluation
        except RuntimeError as retry_error:
            retry_message = str(retry_error)
            if "infeasible" in retry_message.lower() or "master lp failed" in retry_message.lower():
                evaluation = OracleEvaluation("capability_infeasible", y=y, error=retry_message)
                cache.save(key, eval_payload(evaluation))
                return evaluation
            failure = OracleEvaluation("failed", y=y, error=retry_message)
            cache.save(key, eval_payload(failure))
            raise OracleIncomplete(retry_message) from retry_error


def seed_incumbent(instance, base_output_dir: Path, rho: float, epsilon: float) -> OracleEvaluation:
    table = pd.read_csv(base_output_dir / "correlated_facility_separated_capability_marginal_v2" / "tables" / "table_noto_correlated_facility.csv")
    row = table.iloc[np.argmin(np.abs(table["rho"].to_numpy(dtype=float) - rho))]
    y = np.asarray(json.loads(row["selected_y_json"]), dtype=float)
    result = evaluate_fixed_y(instance, y, epsilon=epsilon, max_iterations=500, enforce_retrofit_budget=True)
    return OracleEvaluation("feasible", result.objective, result.lower_bound, result.y, result.z, result.w, result.iterations, result.objective - result.lower_bound)


def run_radius(args: argparse.Namespace, base_output_dir: Path, output_dir: Path, rho: float) -> tuple[dict, dict, list[dict], list[dict]]:
    started = time.perf_counter()
    instance = build_m4_instance(base_output_dir, rho)
    validate_upper_corner_certificate_instance(instance)
    model_digest = model_hash(base_output_dir)
    radius_dir = output_dir / "radii" / f"rho_{rho:.3f}".replace('.', 'p')
    radius_dir.mkdir(parents=True, exist_ok=True)
    cache = CheckpointStore(radius_dir / "corner_checkpoints")
    state_path = radius_dir / "state.json"
    log_path = output_dir / "logs" / f"continuous_bb_rho_{rho:.3f}.log"
    seed = seed_incumbent(instance, base_output_dir, rho, args.oracle_epsilon)
    if args.initial_partition == "root":
        initial = [(np.zeros(len(instance.links)), np.ones(len(instance.links)))]
    else:
        cells = budget_intersecting_grid_cells(instance.retrofit_costs, instance.budget_retrofit, GRID_LEVELS)
        initial = [(cell.lower, cell.upper) for cell in cells]
    cache_values: dict[tuple[float, ...], OracleEvaluation] = {}
    for path in cache.root.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("y") is not None:
                cache_values[tuple(np.round(payload["y"], 12).tolist())] = payload_to_eval(payload, np.asarray(payload["y"], dtype=float))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    event_counter = 0

    initial_lower_bounds = []

    def oracle(y: np.ndarray) -> OracleEvaluation:
        key_tuple = tuple(np.round(y, 12).tolist())
        if key_tuple in cache_values:
            return cache_values[key_tuple]
        key = corner_key(rho, model_digest, y)
        if not args.force and cache.exists(key):
            payload = cache.load(key)
            evaluation = payload_to_eval(payload, y)
            cache_values[key_tuple] = evaluation
            return evaluation
        evaluation = solve_with_retry(instance, y, args.oracle_epsilon, cache, key)
        cache_values[key_tuple] = evaluation
        return evaluation

    def progress(record: dict) -> None:
        nonlocal event_counter
        event_counter += 1
        record_with_rho = {"rho": rho, "elapsed_seconds": time.perf_counter() - started, **record}
        convergence_path = output_dir / "tables" / "table_noto_continuous_bb_convergence.csv"
        prior = pd.read_csv(convergence_path).to_dict("records") if convergence_path.exists() else []
        prior.append(record_with_rho)
        atomic_write_dataframe(pd.DataFrame(prior), convergence_path)
        write_status(
            output_dir / "status.json",
            status="running",
            block="continuous_bb",
            rho=rho,
            event_index=event_counter,
            oracle_calls=record.get("oracle_calls", 0),
            nodes_processed=record.get("nodes_processed", 0),
            active_nodes=record.get("active_nodes", 0),
            incumbent_UB=record.get("incumbent_UB"),
            global_LB=record.get("global_LB"),
            absolute_gap=record.get("absolute_gap"),
            relative_gap_percent=record.get("relative_gap_percent"),
            log_path=str(log_path),
        )
        if event_counter % 10 == 0:
            write_progress_log(log_path, f"rho={rho:.3f} event={event_counter} calls={record.get('oracle_calls')} nodes={record.get('nodes_processed')} gap={record.get('relative_gap_percent'):.6f}%")

    pending = []
    for _, upper in initial:
        key_tuple = tuple(np.round(np.asarray(upper, dtype=float), 12).tolist())
        if key_tuple in cache_values:
            evaluation = cache_values[key_tuple]
            if evaluation.status not in {"infeasible", "capability_infeasible", "pruned"} and evaluation.lower_bound is not None:
                initial_lower_bounds.append(float(evaluation.lower_bound))
        else:
            pending.append(np.asarray(upper, dtype=float))
    def solve_initial_corner(upper: np.ndarray):
        key = corner_key(rho, model_digest, upper)
        return upper, solve_with_retry(instance, upper, args.oracle_epsilon, cache, key)
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [executor.submit(solve_initial_corner, upper) for upper in pending]
        for completed_initial, future in enumerate(as_completed(futures), start=1):
            upper, initial_evaluation = future.result()
            cache_values[tuple(np.round(upper, 12).tolist())] = initial_evaluation
            if initial_evaluation.status not in {"infeasible", "capability_infeasible", "pruned"} and initial_evaluation.lower_bound is not None:
                initial_lower_bounds.append(float(initial_evaluation.lower_bound))
            write_status(output_dir / "status.json", status="running", block="continuous_bb_seed", rho=rho, completed=completed_initial + len(initial) - len(pending), total=len(initial), cached=len(initial) - len(pending), pending=len(pending) - completed_initial)
    initial_uniform_lower = min(initial_lower_bounds, default=float("nan"))

    result = run_monotone_box_bb(
        initial_boxes=initial,
        costs=instance.retrofit_costs,
        budget=instance.budget_retrofit,
        oracle=oracle,
        incumbent=seed,
        budget_tolerance=1e-10,
        box_width_tolerance=1e-8,
        abs_gap_target=args.absolute_gap,
        rel_gap_target=(0.00025 if args.stretch else args.relative_gap),
        branching_rule="width",
        cache=cache_values,
        progress_callback=progress,
    )
    target = max(args.absolute_gap, (0.00025 if args.stretch else args.relative_gap) * max(1.0, abs(result.incumbent.objective)))
    cover = optimizer_cover(result, result.incumbent.objective, target)
    cover_summary = classify_cover(cover)
    initial_lower = initial_uniform_lower
    row = {
        "rho": rho,
        "density_cap": instance.ambiguity_density_cap,
        "incumbent_objective": result.incumbent.objective,
        "global_lower_bound": result.global_lower_bound,
        "absolute_gap": result.absolute_gap,
        "relative_gap_percent": result.relative_gap_percent,
        "converged": result.converged,
        "termination_reason": result.termination_reason,
        "best_y_json": json_string(result.incumbent.y),
        "best_z_json": json_string(result.incumbent.z),
        "best_w_json": json_string(result.incumbent.w),
        "road_budget_used": float(instance.retrofit_costs @ result.incumbent.y),
        "road_budget_slack": float(instance.budget_retrofit - instance.retrofit_costs @ result.incumbent.y),
        "initial_uniform_lower_bound": initial_lower,
        "initial_uniform_gap_percent": 100.0 * (result.incumbent.objective - initial_lower) / max(1.0, abs(result.incumbent.objective)),
        "lower_bound_improvement": result.global_lower_bound - initial_lower,
        "nodes_created": result.nodes_created,
        "nodes_processed": result.nodes_processed,
        "nodes_budget_pruned": result.nodes_budget_pruned,
        "nodes_capability_pruned": result.nodes_capability_pruned,
        "nodes_bound_pruned": result.nodes_bound_pruned,
        "nodes_epsilon_pruned": result.nodes_epsilon_pruned,
        "nodes_closed_feasible_upper": result.nodes_closed_feasible_upper,
        "nodes_active_at_end": result.nodes_active_at_end,
        "maximum_depth": result.maximum_depth,
        "unique_fixed_y_oracle_calls": result.unique_oracle_calls,
        "oracle_cache_hits": result.oracle_cache_hits,
        "total_fixed_y_iterations": result.total_fixed_y_iterations,
        "maximum_fixed_y_oracle_gap": result.maximum_oracle_gap,
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory_mb": None if psutil is None else psutil.Process().memory_info().rss / (1024.0 * 1024.0),
        "starting_commit": source_commit(),
        "model_hash": model_digest,
    }
    cover_row = {"rho": rho, "objective_gap_percent": result.relative_gap_percent, **{f"y{i+1}_min": value for i, value in enumerate(cover_summary["y_min"])}, **{f"y{i+1}_max": value for i, value in enumerate(cover_summary["y_max"])}, **{key: value for key, value in cover_summary.items() if key not in {"y_min", "y_max"}}}
    milestone_rows = []
    for threshold in (1.0, 0.5, 0.25, 0.10, 0.05, 0.025):
        matching = [event for event in result.convergence_events if float(event.get("relative_gap_percent", math.inf)) <= threshold]
        milestone_rows.append({"rho": rho, "target_gap_percent": threshold, "first_event_index": matching[0]["event_index"] if matching else None, "first_oracle_calls": matching[0]["oracle_calls"] if matching else None, "first_runtime_seconds": matching[0].get("elapsed_seconds") if matching else None})
    incumbent_rows = [{"rho": rho, **history} for history in result.incumbent_history]
    atomic_json(state_path, {"status": "completed", "rho": rho, "result": row, "cover": cover_row})
    return row, cover_row, milestone_rows, incumbent_rows


def make_figures(output_dir: Path) -> None:
    convergence_path = output_dir / "tables" / "table_noto_continuous_bb_convergence.csv"
    if not convergence_path.exists():
        return
    data = pd.read_csv(convergence_path)
    if data.empty:
        return
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for x, name, xlabel in (("oracle_calls", "objective_gap_vs_oracle_calls", "Unique fixed-y oracle calls"), ("elapsed_seconds", "objective_gap_vs_runtime", "Runtime (seconds)")):
        fig, ax = plt.subplots(figsize=(7, 4))
        for rho, group in data.groupby("rho"):
            ax.plot(group[x], group["relative_gap_percent"], label=f"rho={rho:g}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Objective gap (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{name}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    base_output_dir = Path(args.base_output_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    rhos = rho_values(args.rhos)
    if not args.resume and not args.force and (output_dir / "run_manifest.json").exists():
        raise RuntimeError("Output already exists; use --resume or choose a new versioned output directory.")
    if not (output_dir / "run_manifest.json").exists() or args.force:
        write_run_metadata(output_dir, experiment="adaptive_continuous_monotone_branch_and_bound", parameters=vars(args), expected_work={"rhos": rhos, "initial_partition": args.initial_partition})
    rows: list[dict] = []
    cover_rows: list[dict] = []
    milestones: list[dict] = []
    incumbents: list[dict] = []
    existing_milestones = output_dir / "tables" / "table_noto_continuous_bb_milestones.csv"
    existing_incumbents = output_dir / "tables" / "table_noto_continuous_incumbents.csv"
    if args.resume and not args.force:
        if existing_milestones.exists():
            milestones = pd.read_csv(existing_milestones).to_dict(orient="records")
        if existing_incumbents.exists():
            incumbents = pd.read_csv(existing_incumbents).to_dict(orient="records")
    for rho in rhos:
        existing = output_dir / "radii" / f"rho_{rho:.3f}".replace('.', 'p') / "state.json"
        if args.resume and existing.exists() and not args.force:
            payload = json.loads(existing.read_text(encoding="utf-8"))
            if payload.get("status") == "completed":
                rows.append(payload["result"])
                cover_rows.append(payload["cover"])
                continue
        row, cover_row, radius_milestones, radius_incumbents = run_radius(args, base_output_dir, output_dir, rho)
        rows.append(row)
        cover_rows.append(cover_row)
        milestones.extend(radius_milestones)
        incumbents.extend(radius_incumbents)
        save_table(rows, output_dir / "tables" / "table_noto_continuous_monotone_bb.csv", ["rho"])
        save_table(cover_rows, output_dir / "tables" / "table_noto_continuous_policy_cover.csv", ["rho"])
        save_table(milestones, output_dir / "tables" / "table_noto_continuous_bb_milestones.csv", ["rho", "target_gap_percent"])
        save_table(incumbents, output_dir / "tables" / "table_noto_continuous_incumbents.csv", ["rho"])
    save_table(rows, output_dir / "tables" / "table_noto_continuous_monotone_bb.csv", ["rho"])
    save_table(cover_rows, output_dir / "tables" / "table_noto_continuous_policy_cover.csv", ["rho"])
    save_table(milestones, output_dir / "tables" / "table_noto_continuous_bb_milestones.csv", ["rho", "target_gap_percent"])
    save_table(incumbents, output_dir / "tables" / "table_noto_continuous_incumbents.csv", ["rho"])
    make_figures(output_dir)
    all_radii_reached = all(row["converged"] for row in rows)
    write_status(output_dir / "status.json", status="completed", block="continuous_bb", rows=len(rows), development_gate_passed=all_radii_reached, rhos=rhos)
    from critical_revision_common import finish_run_metadata
    finish_run_metadata(output_dir, status="completed", runtime_seconds=time.perf_counter() - started, extra={"rhos": rhos, "development_gate_passed": all_radii_reached})


if __name__ == "__main__":
    main()


