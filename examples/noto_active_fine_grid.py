from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ejor_dad.checkpoint import CheckpointStore, atomic_write_text

import noto_access_experiment as noto
import noto_tight_budget_analysis as tight


DEFAULT_RHOS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
DEFAULT_ACTIVE_BUDGET_MULTIPLIER = (0.4694201989666984 + 0.4794786682180999) / 2.0
EXPERIMENT_VERSION = "noto-active-fine-grid-v1"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    for path in [output_dir / "tables", output_dir / "logs", output_dir / "checkpoints", output_dir / "configs"]:
        path.mkdir(parents=True, exist_ok=True)
    external_log = os.environ.get("EJOR_LOG_PATH")
    log_path = (
        Path(external_log)
        if external_log
        else output_dir / "logs" / f"noto_active_fine_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    cache = CheckpointStore(output_dir / "checkpoints")
    rho_values = tight.parse_float_list(args.rho_values, 0.0, 1.0)
    if 0.0 not in rho_values:
        raise ValueError("rho=0 must be included to define the reference policy.")
    base_instance, _ = noto.build_noto_instance(0.0)
    critical_indices = identify_critical_indices(base_instance)
    budget = args.budget_multiplier * base_instance.budget_retrofit
    candidates = enumerate_candidates(base_instance, critical_indices, budget, args.grid_step)
    write_design(
        output_dir,
        base_instance,
        critical_indices,
        rho_values,
        budget,
        args,
        candidates,
    )
    write_status(
        output_dir,
        "running",
        "startup",
        "Critical-link fine-grid validation started.",
        log_path,
        extra={"completed_checkpoints": count_checkpoints(output_dir), "expected_evaluations": len(candidates) * len(rho_values)},
    )
    started = time.time()
    try:
        run_sweep(
            output_dir,
            cache,
            log_path,
            base_instance,
            critical_indices,
            rho_values,
            budget,
            args,
            candidates,
        )
        atomic_write_text(
            output_dir / "runtime_summary.json",
            json.dumps({"runtime_sec": time.time() - started}, indent=2),
        )
        write_status(
            output_dir,
            "completed",
            "complete",
            "Critical-link fine-grid validation completed.",
            log_path,
            extra={"completed_checkpoints": count_checkpoints(output_dir), "expected_evaluations": len(candidates) * len(rho_values)},
        )
    except Exception as exc:
        write_status(output_dir, "failed", "error", str(exc), log_path)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a restart-safe fine grid on the two Noto critical links."
    )
    parser.add_argument(
        "--output-dir",
        default="data_work/noto/support_preserving_active_fine_grid",
    )
    parser.add_argument("--rho-values", default=",".join(str(value) for value in DEFAULT_RHOS))
    parser.add_argument("--density-cap", type=float, default=2.0)
    parser.add_argument("--budget-multiplier", type=float, default=DEFAULT_ACTIVE_BUDGET_MULTIPLIER)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.density_cap < 1.0:
        parser.error("--density-cap must be at least 1.")
    if not 0.0 < args.budget_multiplier <= 1.0:
        parser.error("--budget-multiplier must lie in (0, 1].")
    if not 0.0 < args.grid_step <= 0.25:
        parser.error("--grid-step must lie in (0, 0.25].")
    reciprocal = 1.0 / args.grid_step
    if not np.isclose(reciprocal, round(reciprocal), atol=1e-9):
        parser.error("--grid-step must divide one exactly.")
    if args.workers < 1:
        parser.error("--workers must be at least 1.")
    return args


def identify_critical_indices(instance: Any) -> list[int]:
    summary = pd.read_csv(
        "data_work/noto/support_preserving_full/tables/table_noto_support_dda_summary.csv"
    )
    row = summary[
        np.isclose(summary["density_cap"], 2.0) & np.isclose(summary["rho"], 0.0)
    ].iloc[0]
    y = np.asarray(json.loads(row["best_y_json"]), dtype=float)
    indices = [index for index, value in enumerate(y) if value >= 1.0 - 1e-9]
    if len(indices) != 2:
        raise ValueError("Expected exactly two fully hardened critical links in the baseline grid optimum.")
    return indices


def enumerate_candidates(
    instance: Any,
    critical_indices: list[int],
    budget: float,
    step: float,
) -> list[dict[str, Any]]:
    levels = np.round(np.linspace(0.0, 1.0, int(round(1.0 / step)) + 1), 12)
    candidates: list[dict[str, Any]] = []
    for candidate_index, values in enumerate(product(levels, repeat=2), start=1):
        y = np.zeros(len(instance.links), dtype=float)
        y[critical_indices] = values
        budget_used = float(np.dot(instance.retrofit_costs, y))
        if budget_used > budget + 1e-9:
            continue
        candidates.append(
            {
                "candidate_index": candidate_index,
                "y": y,
                "budget_used": budget_used,
            }
        )
    return candidates


def run_sweep(
    output_dir: Path,
    cache: CheckpointStore,
    log_path: Path,
    base_instance: Any,
    critical_indices: list[int],
    rho_values: list[float],
    budget: float,
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
) -> None:
    summary_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    near_rows: list[dict[str, Any]] = []
    reference_y: tuple[float, ...] | None = None
    for rho in rho_values:
        instance, _ = noto.build_noto_instance(rho)
        instance = replace(
            instance,
            budget_retrofit=budget,
            ambiguity_density_cap=args.density_cap,
        )
        experiment_id = (
            f"noto_active_fine_b{format_number(args.budget_multiplier)}_"
            f"g{format_number(args.grid_step)}_rho{rho:.2f}"
        )
        results = evaluate_candidates(
            output_dir,
            cache,
            log_path,
            instance,
            experiment_id,
            rho,
            args,
            candidates,
            len(rho_values),
        )
        results.sort(key=lambda row: row.objective)
        best = results[0]
        if rho == 0.0:
            reference_y = tuple(float(value) for value in best.y)
        if reference_y is None:
            raise RuntimeError("rho=0 must be evaluated first.")
        reference = next(
            row for row in results if np.allclose(row.y, reference_y, atol=1e-12, rtol=0.0)
        )
        summary_rows.append(
            {
                "rho": rho,
                "density_cap": args.density_cap,
                "budget_multiplier": args.budget_multiplier,
                "B_Y": budget,
                "grid_step": args.grid_step,
                "restricted_link_indices_json": json.dumps(critical_indices),
                "restricted_link_ids_json": json.dumps(
                    [instance.links[index].id for index in critical_indices]
                ),
                "feasible_grid_candidates": len(results),
                "best_objective": float(best.objective),
                "best_y_json": json.dumps(best.y.tolist()),
                "best_z_json": json.dumps(best.z.tolist()),
                "best_w_json": json.dumps(best.w.tolist()),
                "budget_used": float(best.budget_used),
                "objective_using_rho0_policy": float(reference.objective),
                "delta_rho_value": float(reference.objective - best.objective),
                "policy_changed_from_rho0": not np.allclose(
                    best.y,
                    reference_y,
                    atol=1e-12,
                    rtol=0.0,
                ),
                "second_best_gap": float(results[1].objective - best.objective),
                "algorithm1_gap": float(best.objective - best.lower_bound),
                "mass_added_to_zero_nominal_states": support_leakage(best),
                "max_realized_density_ratio": max_density_ratio(best),
            }
        )
        for result in results:
            candidate_rows.append(
                {
                    "rho": rho,
                    "candidate_index": result.candidate_index,
                    "objective": float(result.objective),
                    "gap_to_best": float(result.objective - best.objective),
                    "selected_y_json": json.dumps(result.y.tolist()),
                    "critical_link_1_y": float(result.y[critical_indices[0]]),
                    "critical_link_2_y": float(result.y[critical_indices[1]]),
                    "budget_used": float(result.budget_used),
                    "algorithm1_gap": float(result.objective - result.lower_bound),
                }
            )
        for rank, result in enumerate(results[:10], start=1):
            top_rows.append(
                {
                    "rho": rho,
                    "rank": rank,
                    "objective": float(result.objective),
                    "gap_to_best": float(result.objective - best.objective),
                    "gap_percent": 100.0
                    * float(result.objective - best.objective)
                    / max(1.0, abs(float(best.objective))),
                    "selected_y_json": json.dumps(result.y.tolist()),
                    "budget_used": float(result.budget_used),
                }
            )
        for threshold in [0.01, 0.05, 0.10, 0.50]:
            count = sum(
                100.0 * float(result.objective - best.objective)
                / max(1.0, abs(float(best.objective)))
                <= threshold + 1e-12
                for result in results
            )
            near_rows.append(
                {
                    "rho": rho,
                    "threshold_percent": threshold,
                    "near_optimal_policy_count": count,
                    "feasible_grid_candidates": len(results),
                }
            )
        write_tables(output_dir, summary_rows, candidate_rows, top_rows, near_rows)


def evaluate_candidates(
    output_dir: Path,
    cache: CheckpointStore,
    log_path: Path,
    instance: Any,
    experiment_id: str,
    rho: float,
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    num_rhos: int,
) -> list[Any]:
    results: list[Any] = []
    pending: list[tuple[dict[str, Any], str]] = []
    for candidate in candidates:
        key = noto.versioned_key(
            f"{EXPERIMENT_VERSION}__{experiment_id}__grid_{candidate['candidate_index']:04d}_"
            f"{noto.hash_array(candidate['y'])}"
        )
        if not args.force and cache.exists(key):
            payload = cache.load(key)
            results.append(
                noto.result_from_candidate_payload(
                    payload,
                    candidate["candidate_index"],
                    candidate["budget_used"],
                    loaded_from_cache=True,
                )
            )
        else:
            pending.append((candidate, key))

    if args.workers == 1:
        for candidate, key in pending:
            payload = noto.evaluate_candidate_payload(
                instance,
                candidate["y"],
                candidate["candidate_index"],
                candidate["budget_used"],
            )
            cache.save(key, payload)
            results.append(
                noto.result_from_candidate_payload(
                    payload,
                    candidate["candidate_index"],
                    candidate["budget_used"],
                    loaded_from_cache=False,
                )
            )
            report_progress(output_dir, log_path, rho, results, candidates, num_rhos, args.workers)
        return results

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                noto.evaluate_candidate_payload,
                instance,
                candidate["y"],
                candidate["candidate_index"],
                candidate["budget_used"],
            ): (candidate, key)
            for candidate, key in pending
        }
        for future in as_completed(futures):
            candidate, key = futures[future]
            payload = future.result()
            cache.save(key, payload)
            results.append(
                noto.result_from_candidate_payload(
                    payload,
                    candidate["candidate_index"],
                    candidate["budget_used"],
                    loaded_from_cache=False,
                )
            )
            report_progress(output_dir, log_path, rho, results, candidates, num_rhos, args.workers)
    return results


def report_progress(
    output_dir: Path,
    log_path: Path,
    rho: float,
    results: list[Any],
    candidates: list[dict[str, Any]],
    num_rhos: int,
    workers: int,
) -> None:
    if len(results) % 10 != 0 and len(results) != len(candidates):
        return
    completed = count_checkpoints(output_dir)
    expected = len(candidates) * num_rhos
    message = f"rho={rho:.2f}: {len(results)}/{len(candidates)} policies; total {completed}/{expected}."
    append_log(log_path, message)
    write_status(
        output_dir,
        "running",
        "fine_grid",
        message,
        log_path,
        extra={
            "rho": rho,
            "completed_at_current_rho": len(results),
            "candidates_per_rho": len(candidates),
            "completed_checkpoints": completed,
            "expected_evaluations": expected,
            "workers": workers,
        },
    )


def support_leakage(result: Any) -> float:
    nominal = np.asarray(result.nominal_distribution, dtype=float)
    worst = np.asarray(result.worst_case_distribution, dtype=float)
    return float(np.maximum(worst - nominal, 0.0)[nominal <= 1e-12].sum())


def max_density_ratio(result: Any) -> float:
    nominal = np.asarray(result.nominal_distribution, dtype=float)
    worst = np.asarray(result.worst_case_distribution, dtype=float)
    positive = nominal > 1e-12
    return float(np.max(worst[positive] / nominal[positive]))


def write_tables(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    top_rows: list[dict[str, Any]],
    near_rows: list[dict[str, Any]],
) -> None:
    noto.write_table(pd.DataFrame(summary_rows), output_dir, "table_noto_active_fine_grid_summary")
    noto.write_table(pd.DataFrame(candidate_rows), output_dir, "table_noto_active_fine_grid_candidates")
    noto.write_table(pd.DataFrame(top_rows), output_dir, "table_noto_active_fine_grid_top10")
    noto.write_table(pd.DataFrame(near_rows), output_dir, "table_noto_active_fine_grid_near_optimal")


def write_design(
    output_dir: Path,
    instance: Any,
    critical_indices: list[int],
    rho_values: list[float],
    budget: float,
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
) -> None:
    payload = {
        "experiment_version": EXPERIMENT_VERSION,
        "density_cap": args.density_cap,
        "rho_values": rho_values,
        "budget_multiplier": args.budget_multiplier,
        "baseline_B_Y": instance.budget_retrofit,
        "active_B_Y": budget,
        "grid_step": args.grid_step,
        "workers": args.workers,
        "critical_link_indices_zero_based": critical_indices,
        "critical_link_ids": [instance.links[index].id for index in critical_indices],
        "fixed_other_link_values": 0.0,
        "feasible_candidates_per_rho": len(candidates),
        "expected_exact_evaluations": len(candidates) * len(rho_values),
        "scope": "Full five-link probability/recourse model with only the two critical y variables refined.",
    }
    atomic_write_text(output_dir / "run_design.json", json.dumps(payload, indent=2))


def count_checkpoints(output_dir: Path) -> int:
    checkpoint_dir = output_dir / "checkpoints"
    return sum(1 for _ in checkpoint_dir.glob(f"*{EXPERIMENT_VERSION}*.json")) if checkpoint_dir.exists() else 0


def format_number(value: float) -> str:
    return f"{value:.6f}".replace(".", "p")


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")


def write_status(
    output_dir: Path,
    status: str,
    block: str,
    message: str,
    log_path: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "status": status,
        "block": block,
        "message": message,
        "pid": os.getpid(),
        "updated_at_epoch": time.time(),
        "log_path": str(log_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "checkpoint_dir": str((output_dir / "checkpoints").resolve()),
    }
    if extra:
        payload.update(extra)
    atomic_write_text(output_dir / "run_status.json", json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
