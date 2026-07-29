from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ejor_dad.checkpoint import CheckpointStore, atomic_write_text

import noto_access_experiment as noto


DEFAULT_CAPS = [1.5, 2.0, 5.0, 10.0]
DEFAULT_RHOS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
EXPERIMENT_VERSION = "noto-support-v1"


def main() -> None:
    args = parse_args()
    output_name = "support_preserving_pilot" if args.mode == "pilot" else "support_preserving_full"
    output_dir = Path("data_work/noto") / output_name
    for path in [output_dir / "tables", output_dir / "logs", output_dir / "checkpoints", output_dir / "configs"]:
        path.mkdir(parents=True, exist_ok=True)
    external_log = os.environ.get("EJOR_LOG_PATH")
    log_path = (
        Path(external_log)
        if external_log
        else output_dir / "logs" / f"noto_support_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    cache = CheckpointStore(output_dir / "checkpoints")
    density_caps = parse_float_list(args.density_caps, minimum=1.0)
    rho_values = parse_float_list(args.rho_values, minimum=0.0, maximum=1.0)
    if 0.0 not in rho_values:
        raise ValueError("rho=0 must be included to define the nominal reference policy.")
    write_run_design(output_dir, args.mode, density_caps, rho_values, args.workers)
    write_status(output_dir, "running", "startup", "Support-preserving Noto sweep started.", log_path)
    started = time.time()
    try:
        run_sweep(
            output_dir=output_dir,
            cache=cache,
            force=args.force,
            log_path=log_path,
            mode=args.mode,
            density_caps=density_caps,
            rho_values=rho_values,
            workers=args.workers,
        )
        atomic_write_text(
            output_dir / "runtime_summary.json",
            json.dumps({"runtime_sec": time.time() - started}, indent=2),
        )
        write_status(output_dir, "completed", "complete", "Support-preserving Noto sweep completed.", log_path)
    except Exception as exc:
        write_status(output_dir, "failed", "error", str(exc), log_path)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the support-preserving capped-TV Noto experiment.")
    parser.add_argument("--mode", choices=sorted(noto.GRID_LEVELS), default="pilot")
    parser.add_argument("--density-caps", default=",".join(str(value) for value in DEFAULT_CAPS))
    parser.add_argument("--rho-values", default=",".join(str(value) for value in DEFAULT_RHOS))
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1.")
    return args


def parse_float_list(value: str, minimum: float, maximum: float | None = None) -> list[float]:
    values = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not values:
        raise ValueError("Expected a nonempty comma-separated numeric list.")
    if any(item < minimum or (maximum is not None and item > maximum) for item in values):
        raise ValueError(f"Values must lie in [{minimum}, {maximum if maximum is not None else 'infinity'}].")
    return values


def run_sweep(
    output_dir: Path,
    cache: CheckpointStore,
    force: bool,
    log_path: Path,
    mode: str,
    density_caps: list[float],
    rho_values: list[float],
    workers: int,
) -> None:
    grid = noto.GRID_LEVELS[mode]
    base_instance, _ = noto.build_noto_instance(0.0)
    total_grid = int(len(grid) ** len(base_instance.links))
    summary_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    near_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []

    for density_cap in density_caps:
        reference_y: np.ndarray | None = None
        reference_z: np.ndarray | None = None
        reference_w: np.ndarray | None = None
        for rho in rho_values:
            experiment_started = time.time()
            uncapped_instance, metadata = noto.build_noto_instance(rho)
            instance = replace(uncapped_instance, ambiguity_density_cap=density_cap)
            cap_label = format_cap(density_cap)
            experiment_id = f"noto_support_m5_{mode}_k{cap_label}_rho{rho:.2f}"
            noto.write_config(
                output_dir,
                experiment_id,
                metadata
                | {
                    "experiment_version": EXPERIMENT_VERSION,
                    "ambiguity_model": "TV with statewise density-ratio upper cap",
                    "ambiguity_density_cap": density_cap,
                    "rho": rho,
                    "mode": mode,
                    "grid_levels": grid.tolist(),
                    "total_grid_candidates": total_grid,
                },
            )
            results, infeasible = noto.evaluate_grid(
                instance=instance,
                grid=grid,
                experiment_id=experiment_id,
                cache=cache,
                force=force,
                out=output_dir,
                log_path=log_path,
                mode=mode,
                rho=rho,
                total_grid=total_grid,
                workers=workers,
            )
            if not results:
                raise RuntimeError(f"No feasible policies found for cap={density_cap}, rho={rho}.")
            results.sort(key=lambda item: item.objective)
            best = results[0]
            if rho == 0.0:
                reference_y = np.asarray(best.y, dtype=float)
                reference_z = np.asarray(best.z, dtype=float)
                reference_w = np.asarray(best.w, dtype=float)
            if reference_y is None or reference_z is None or reference_w is None:
                raise RuntimeError("rho=0 must be evaluated first for every density cap.")

            reference_evaluation = noto.cached_fixed_y(
                cache,
                f"{experiment_id}_rho0_policy",
                instance,
                reference_y,
                force,
            )
            no_retrofit = noto.cached_fixed_y(
                cache,
                f"{experiment_id}_no_retrofit",
                instance,
                np.zeros(len(instance.links)),
                force,
            )
            diagnostics = support_diagnostics(instance, best)
            summary_rows.append(
                {
                    "experiment_id": experiment_id,
                    "mode": mode,
                    "density_cap": density_cap,
                    "rho": rho,
                    "num_states": len(instance.states),
                    "total_grid_candidates": total_grid,
                    "feasible_evaluated_candidates": len(results),
                    "infeasible_budget_candidates": infeasible,
                    "best_objective": float(best.objective),
                    "best_y_json": json.dumps(best.y.tolist()),
                    "best_z_json": json.dumps(best.z.tolist()),
                    "best_w_json": json.dumps(best.w.tolist()),
                    "objective_using_rho0_policy": float(reference_evaluation.objective),
                    "delta_rho_value": float(reference_evaluation.objective - best.objective),
                    "y_diff_norm_from_rho0": float(np.linalg.norm(np.asarray(best.y) - reference_y, ord=1)),
                    "z_diff_norm_from_rho0": float(np.linalg.norm(np.asarray(best.z) - reference_z)),
                    "w_diff_norm_from_rho0": float(np.linalg.norm(np.asarray(best.w) - reference_w)),
                    "no_retrofit_objective": float(no_retrofit.objective),
                    "road_value": float(no_retrofit.objective - best.objective),
                    "road_value_percent": 100.0
                    * float(no_retrofit.objective - best.objective)
                    / max(1.0, abs(float(no_retrofit.objective))),
                    "runtime_sec": time.time() - experiment_started,
                }
                | diagnostics
            )

            for rank, result in enumerate(results[:10], start=1):
                top_rows.append(
                    {
                        "mode": mode,
                        "density_cap": density_cap,
                        "rho": rho,
                        "rank": rank,
                        "objective": float(result.objective),
                        "gap_to_best": float(result.objective - best.objective),
                        "gap_percent": 100.0
                        * float(result.objective - best.objective)
                        / max(1.0, abs(float(best.objective))),
                        "selected_y_json": json.dumps(result.y.tolist()),
                    }
                )

            for threshold in [0.01, 0.05, 0.10, 0.50]:
                count = sum(
                    1
                    for result in results
                    if 100.0 * float(result.objective - best.objective) / max(1.0, abs(float(best.objective)))
                    <= threshold + 1e-12
                )
                near_rows.append(
                    {
                        "mode": mode,
                        "density_cap": density_cap,
                        "rho": rho,
                        "threshold_percent": threshold,
                        "near_optimal_policy_count": count,
                        "feasible_evaluated_candidates": len(results),
                    }
                )

            support_rows.extend(state_shift_rows(instance, best, density_cap, rho, mode))
            write_tables(output_dir, summary_rows, top_rows, near_rows, support_rows)

    write_tables(output_dir, summary_rows, top_rows, near_rows, support_rows)


def support_diagnostics(instance: Any, result: Any) -> dict[str, Any]:
    nominal = np.asarray(result.nominal_distribution, dtype=float)
    worst_case = np.asarray(result.worst_case_distribution, dtype=float)
    positive_shift = np.maximum(worst_case - nominal, 0.0)
    zero_nominal = nominal <= 1e-12
    mass_outside_support = float(positive_shift[zero_nominal].sum())
    positive_nominal = nominal > 1e-12
    max_ratio = float(np.max(worst_case[positive_nominal] / nominal[positive_nominal]))
    fully_retrofitted = {
        link.id for link_index, link in enumerate(instance.links) if result.y[link_index] >= 1.0 - 1e-9
    }
    failed_in_added_states: set[str] = set()
    for state_index, state in enumerate(instance.states):
        if positive_shift[state_index] > 1e-10:
            failed_in_added_states.update(state.failed_links)
    hardened_failures = sorted(fully_retrofitted.intersection(failed_in_added_states))
    return {
        "total_positive_probability_shift": float(positive_shift.sum()),
        "mass_added_to_zero_nominal_states": mass_outside_support,
        "max_realized_density_ratio": max_ratio,
        "density_cap_respected": max_ratio <= float(instance.ambiguity_density_cap) + 1e-9,
        "fully_retrofitted_links_failed_in_added_states_json": json.dumps(hardened_failures),
        "hardened_link_failure_occurs": bool(hardened_failures),
    }


def state_shift_rows(instance: Any, result: Any, density_cap: float, rho: float, mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state_index, state in enumerate(instance.states):
        shift = float(result.worst_case_distribution[state_index] - result.nominal_distribution[state_index])
        if abs(shift) <= 1e-10:
            continue
        nominal_probability = float(result.nominal_distribution[state_index])
        worst_case_probability = float(result.worst_case_distribution[state_index])
        rows.append(
            {
                "mode": mode,
                "density_cap": density_cap,
                "rho": rho,
                "state_id": state.id,
                "failed_links_json": json.dumps(list(state.failed_links)),
                "nominal_probability": nominal_probability,
                "worst_case_probability": worst_case_probability,
                "probability_shift": shift,
                "density_ratio": worst_case_probability / nominal_probability
                if nominal_probability > 1e-12
                else np.nan,
                "state_loss": float(result.state_losses[state_index]),
            }
        )
    return rows


def write_tables(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    top_rows: list[dict[str, Any]],
    near_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
) -> None:
    noto.write_table(pd.DataFrame(summary_rows), output_dir, "table_noto_support_dda_summary")
    noto.write_table(pd.DataFrame(top_rows), output_dir, "table_noto_support_top10")
    noto.write_table(pd.DataFrame(near_rows), output_dir, "table_noto_support_near_optimal")
    noto.write_table(pd.DataFrame(support_rows), output_dir, "table_noto_support_probability_shifts")


def write_run_design(
    output_dir: Path,
    mode: str,
    density_caps: list[float],
    rho_values: list[float],
    workers: int,
) -> None:
    grid = noto.GRID_LEVELS[mode]
    instance, _ = noto.build_noto_instance(0.0)
    feasible_per_scenario = sum(
        float(np.dot(instance.retrofit_costs, np.asarray(values, dtype=float)))
        <= instance.budget_retrofit + 1e-9
        for values in product(grid, repeat=len(instance.links))
    )
    payload = {
        "mode": mode,
        "density_caps": density_caps,
        "rho_values": rho_values,
        "workers": workers,
        "grid_levels": grid.tolist(),
        "total_grid_vectors_per_scenario": int(len(grid) ** len(instance.links)),
        "feasible_grid_vectors_per_scenario": int(feasible_per_scenario),
        "expected_exact_evaluations": int(feasible_per_scenario * len(density_caps) * len(rho_values)),
    }
    atomic_write_text(output_dir / "run_design.json", json.dumps(payload, indent=2))


def format_cap(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def write_status(output_dir: Path, status: str, block: str, message: str, log_path: Path) -> None:
    atomic_write_text(
        output_dir / "run_status.json",
        json.dumps(
            {
                "status": status,
                "block": block,
                "message": message,
                "pid": os.getpid(),
                "updated_at_epoch": time.time(),
                "log_path": str(log_path.resolve()),
                "output_dir": str(output_dir.resolve()),
                "checkpoint_dir": str((output_dir / "checkpoints").resolve()),
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
