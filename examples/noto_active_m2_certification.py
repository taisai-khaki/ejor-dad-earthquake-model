from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ejor_dad import evaluate_fixed_y, generate_failure_states
from ejor_dad.checkpoint import CheckpointStore, atomic_write_text
from ejor_dad.sbb import solve_global_sbb

import noto_access_experiment as noto
from noto_support_m3_certification import statewise_nominal_root
from noto_active_fine_grid import DEFAULT_ACTIVE_BUDGET_MULTIPLIER


DEFAULT_RHOS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
CRITICAL_INDICES = [1, 2]
EXPERIMENT_VERSION = "noto-active-critical-m2-v1"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    for path in [output_dir / "tables", output_dir / "logs", output_dir / "checkpoints", output_dir / "configs"]:
        path.mkdir(parents=True, exist_ok=True)
    external_log = os.environ.get("EJOR_LOG_PATH")
    log_path = (
        Path(external_log)
        if external_log
        else output_dir / "logs" / f"noto_active_m2_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    cache = CheckpointStore(output_dir / "checkpoints")
    rho_values = parse_float_list(args.rho_values)
    if 0.0 not in rho_values:
        raise ValueError("rho=0 must be included to define the reference policy.")
    write_status(output_dir, "running", "startup", "Active-budget critical-link SBB started.", log_path)
    started = time.time()
    try:
        run_certification(output_dir, cache, log_path, rho_values, args)
        atomic_write_text(
            output_dir / "runtime_summary.json",
            json.dumps({"runtime_sec": time.time() - started}, indent=2),
        )
        write_status(output_dir, "completed", "complete", "Active-budget critical-link SBB completed.", log_path)
    except Exception as exc:
        write_status(output_dir, "failed", "error", str(exc), log_path)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Certify the two critical Noto links at the mechanism-active budget."
    )
    parser.add_argument(
        "--output-dir",
        default="data_work/noto/support_preserving_active_m2_sbb",
    )
    parser.add_argument("--rho-values", default=",".join(str(value) for value in DEFAULT_RHOS))
    parser.add_argument("--density-cap", type=float, default=2.0)
    parser.add_argument("--budget-multiplier", type=float, default=DEFAULT_ACTIVE_BUDGET_MULTIPLIER)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--time-limit-sec", type=float, default=1800.0)
    parser.add_argument("--certificate-absolute-tolerance", type=float, default=0.1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.density_cap < 1.0:
        parser.error("--density-cap must be at least 1.")
    if not 0.0 < args.budget_multiplier <= 1.0:
        parser.error("--budget-multiplier must lie in (0, 1].")
    if args.max_nodes < 1 or args.time_limit_sec <= 0.0:
        parser.error("Node and time limits must be positive.")
    if args.certificate_absolute_tolerance <= 0.0:
        parser.error("--certificate-absolute-tolerance must be positive.")
    return args


def parse_float_list(value: str) -> list[float]:
    values = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not values or any(item < 0.0 or item > 1.0 for item in values):
        raise ValueError("rho values must lie in [0, 1].")
    return values


def build_active_m2_instance(rho: float, density_cap: float, budget_multiplier: float):
    full_instance, metadata = noto.build_noto_instance(rho)
    links = [full_instance.links[index] for index in CRITICAL_INDICES]
    states = generate_failure_states(links, max_failures=None, include_tail=False)
    travel_times = {
        state.id: np.asarray(full_instance.precomputed_travel_times[state.id], dtype=float)
        for state in states
    }
    budget = budget_multiplier * full_instance.budget_retrofit
    instance = replace(
        full_instance,
        links=links,
        states=states,
        precomputed_travel_times=travel_times,
        ambiguity_density_cap=density_cap,
        budget_retrofit=budget,
    )
    return instance, metadata, budget


def run_certification(
    output_dir: Path,
    cache: CheckpointStore,
    log_path: Path,
    rho_values: list[float],
    args: argparse.Namespace,
) -> None:
    rows: list[dict[str, Any]] = []
    reference_y: np.ndarray | None = None
    reference_z: np.ndarray | None = None
    reference_w: np.ndarray | None = None
    warm_start = np.asarray([1.00, 0.20], dtype=float)
    for rho in rho_values:
        instance, metadata, budget = build_active_m2_instance(
            rho,
            args.density_cap,
            args.budget_multiplier,
        )
        if float(np.dot(instance.retrofit_costs, warm_start)) > budget + 1e-9:
            warm_start *= budget / float(np.dot(instance.retrofit_costs, warm_start))
        experiment_id = (
            f"noto_active_critical_m2_k{format_number(args.density_cap)}_"
            f"b{format_number(args.budget_multiplier)}_rho{rho:.2f}"
        )
        noto.write_config(
            output_dir,
            experiment_id,
            metadata
            | {
                "experiment_version": EXPERIMENT_VERSION,
                "critical_link_indices_zero_based": CRITICAL_INDICES,
                "critical_link_ids": [link.id for link in instance.links],
                "num_states": len(instance.states),
                "ambiguity_density_cap": args.density_cap,
                "budget_multiplier": args.budget_multiplier,
                "B_Y": budget,
                "rho": rho,
                "max_nodes": args.max_nodes,
                "time_limit_sec": args.time_limit_sec,
                "scope": "Reduced four-state instance; all excluded links are treated as intact.",
            },
        )
        key = (
            f"{EXPERIMENT_VERSION}__{experiment_id}__sbb_"
            f"n{args.max_nodes}_t{args.time_limit_sec:g}"
        )

        def compute() -> dict[str, Any]:
            last_reported_nodes = -50

            def trace(record: dict[str, Any]) -> None:
                nonlocal last_reported_nodes
                processed = int(record.get("processed_nodes", 0))
                if record.get("event") not in {"finish", "incumbent"} and processed < last_reported_nodes + 50:
                    return
                last_reported_nodes = processed
                message = (
                    f"rho={rho:.2f}: nodes={processed}, UB={record.get('incumbent_UB')}, "
                    f"LB={record.get('global_LB')}"
                )
                append_log(log_path, message)
                write_status(output_dir, "running", "sbb", message, log_path)

            result = solve_global_sbb(
                instance,
                epsilon=1e-2,
                fixed_y_epsilon=1e-5,
                cut_epsilon=1e-7,
                max_nodes=args.max_nodes,
                initial_y=warm_start,
                time_limit_sec=args.time_limit_sec,
                trace_callback=trace,
                initial_root=statewise_nominal_root(instance, warm_start),
            )
            fixed = result.fixed_y_result
            return {
                "objective": float(result.objective),
                "lower_bound": float(result.lower_bound),
                "gap": float(result.gap),
                "relative_gap_percent": 100.0
                * max(0.0, float(result.objective - result.lower_bound))
                / max(1.0, abs(float(result.objective))),
                "y": result.y.tolist(),
                "z": result.z.tolist(),
                "w": result.w.tolist(),
                "nominal_distribution": fixed.nominal_distribution.tolist(),
                "worst_case_distribution": fixed.worst_case_distribution.tolist(),
                "state_losses": fixed.state_losses.tolist(),
                "nodes_processed": int(result.nodes_processed),
                "nodes_remaining": int(result.nodes_remaining),
                "runtime_sec": float(result.runtime_sec),
                "converged": bool(result.converged),
                "termination_reason": result.termination_reason,
            }

        payload = cache.get_or_compute(key, compute, force=args.force)
        selected_y = np.asarray(payload["y"], dtype=float)
        warm_start = selected_y.copy()
        if rho == 0.0:
            reference_y = selected_y.copy()
            reference_z = np.asarray(payload["z"], dtype=float)
            reference_w = np.asarray(payload["w"], dtype=float)
        if reference_y is None or reference_z is None or reference_w is None:
            raise RuntimeError("rho=0 must be solved first.")
        reference = evaluate_fixed_y(instance, reference_y, epsilon=1e-5, max_iterations=160)
        nominal = np.asarray(payload["nominal_distribution"], dtype=float)
        worst = np.asarray(payload["worst_case_distribution"], dtype=float)
        positive = nominal > 1e-12
        positive_shift = np.maximum(worst - nominal, 0.0)
        rows.append(
            {
                "experiment_id": experiment_id,
                "rho": rho,
                "density_cap": args.density_cap,
                "budget_multiplier": args.budget_multiplier,
                "B_Y": budget,
                "num_candidate_links": 2,
                "num_states": len(instance.states),
                "objective": payload["objective"],
                "lower_bound": payload["lower_bound"],
                "absolute_gap": payload["gap"],
                "relative_gap_percent": payload["relative_gap_percent"],
                "selected_y_json": json.dumps(payload["y"]),
                "selected_z_json": json.dumps(payload["z"]),
                "selected_w_json": json.dumps(payload["w"]),
                "objective_using_rho0_policy": float(reference.objective),
                "delta_rho_value": float(reference.objective - payload["objective"]),
                "y_diff_norm_from_rho0": float(np.linalg.norm(selected_y - reference_y, ord=1)),
                "z_diff_norm_from_rho0": float(np.linalg.norm(np.asarray(payload["z"]) - reference_z)),
                "w_diff_norm_from_rho0": float(np.linalg.norm(np.asarray(payload["w"]) - reference_w)),
                "mass_added_to_zero_nominal_states": float(positive_shift[nominal <= 1e-12].sum()),
                "max_realized_density_ratio": float(np.max(worst[positive] / nominal[positive])),
                "nodes_processed": payload["nodes_processed"],
                "nodes_remaining": payload["nodes_remaining"],
                "runtime_sec": payload["runtime_sec"],
                "converged": payload["converged"],
                "termination_reason": payload["termination_reason"],
                "certificate_absolute_tolerance": args.certificate_absolute_tolerance,
                "certified_at_tolerance": float(payload["gap"])
                <= args.certificate_absolute_tolerance,
                "scope": "Reduced critical-link instance; excluded links fixed intact.",
            }
        )
        noto.write_table(
            pd.DataFrame(rows),
            output_dir,
            "table_noto_active_critical_m2_sbb",
        )


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
) -> None:
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
