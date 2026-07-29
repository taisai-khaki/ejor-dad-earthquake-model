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
from ejor_dad.sbb import root_node, solve_global_sbb

import noto_access_experiment as noto


DEFAULT_RHOS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
EXPERIMENT_VERSION = "noto-support-reduced-v7"


def main() -> None:
    args = parse_args()
    output_dir = Path(f"data_work/noto/support_preserving_m{args.num_links}")
    for path in [output_dir / "tables", output_dir / "logs", output_dir / "checkpoints", output_dir / "configs"]:
        path.mkdir(parents=True, exist_ok=True)
    external_log = os.environ.get("EJOR_LOG_PATH")
    log_path = (
        Path(external_log)
        if external_log
        else output_dir / "logs" / f"noto_support_m{args.num_links}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    cache = CheckpointStore(output_dir / "checkpoints")
    rho_values = parse_float_list(args.rho_values, 0.0, 1.0)
    if 0.0 not in rho_values:
        raise ValueError("rho=0 must be included for the reference policy.")
    if args.monotonicity_anchor is not None and not any(
        np.isclose(args.monotonicity_anchor, rho) for rho in rho_values
    ):
        raise ValueError("The monotonicity anchor must be included in --rho-values.")
    write_status(
        output_dir,
        "running",
        "startup",
        f"Support-preserving Noto m={args.num_links} certification started.",
        log_path,
    )
    started = time.time()
    try:
        run_certification(
            output_dir=output_dir,
            cache=cache,
            force=args.force,
            log_path=log_path,
            density_cap=args.density_cap,
            rho_values=rho_values,
            max_nodes=args.max_nodes,
            time_limit_sec=args.time_limit_sec,
            num_links=args.num_links,
            monotonicity_anchor=args.monotonicity_anchor,
            certificate_absolute_tolerance=args.certificate_absolute_tolerance,
        )
        atomic_write_text(
            output_dir / "runtime_summary.json",
            json.dumps({"runtime_sec": time.time() - started}, indent=2),
        )
        write_status(
            output_dir,
            "completed",
            "complete",
            f"Support-preserving Noto m={args.num_links} certification completed.",
            log_path,
        )
    except Exception as exc:
        write_status(output_dir, "failed", "error", str(exc), log_path)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Certify a reduced support-preserving Noto instance with SBB.")
    parser.add_argument("--num-links", type=int, choices=[2, 3], default=3)
    parser.add_argument("--density-cap", type=float, default=2.0)
    parser.add_argument("--rho-values", default=",".join(str(value) for value in DEFAULT_RHOS))
    parser.add_argument("--max-nodes", type=int, default=10000)
    parser.add_argument("--time-limit-sec", type=float, default=900.0)
    parser.add_argument("--monotonicity-anchor", type=float)
    parser.add_argument("--certificate-absolute-tolerance", type=float, default=0.1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.density_cap < 1.0:
        parser.error("--density-cap must be at least 1.")
    if args.max_nodes < 1 or args.time_limit_sec <= 0.0:
        parser.error("Node and time limits must be positive.")
    if args.monotonicity_anchor is not None and not 0.0 <= args.monotonicity_anchor <= 1.0:
        parser.error("--monotonicity-anchor must lie in [0, 1].")
    if args.certificate_absolute_tolerance <= 0.0:
        parser.error("--certificate-absolute-tolerance must be positive.")
    return args


def parse_float_list(value: str, minimum: float, maximum: float) -> list[float]:
    values = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not values or any(item < minimum or item > maximum for item in values):
        raise ValueError(f"Values must lie in [{minimum}, {maximum}].")
    return values


def build_reduced_instance(rho: float, density_cap: float, num_links: int):
    full_instance, metadata = noto.build_noto_instance(rho)
    links = list(full_instance.links[:num_links])
    states = generate_failure_states(links, max_failures=None, include_tail=False)
    travel_times = {
        state.id: np.asarray(full_instance.precomputed_travel_times[state.id], dtype=float)
        for state in states
    }
    instance = replace(
        full_instance,
        links=links,
        states=states,
        precomputed_travel_times=travel_times,
        ambiguity_density_cap=density_cap,
        budget_retrofit=min(full_instance.budget_retrofit, sum(link.retrofit_cost for link in links)),
    )
    return instance, metadata


def build_m3_instance(rho: float, density_cap: float):
    return build_reduced_instance(rho, density_cap, 3)


def run_certification(
    output_dir: Path,
    cache: CheckpointStore,
    force: bool,
    log_path: Path,
    density_cap: float,
    rho_values: list[float],
    max_nodes: int,
    time_limit_sec: float,
    num_links: int,
    monotonicity_anchor: float | None,
    certificate_absolute_tolerance: float,
) -> None:
    rows: list[dict[str, Any]] = []
    reference_y: np.ndarray | None = None
    reference_z: np.ndarray | None = None
    reference_w: np.ndarray | None = None
    reference_objective: float | None = None
    anchor_payload: dict[str, Any] | None = None
    warm_start = np.concatenate([np.zeros(1), np.ones(num_links - 1)])

    for rho in rho_values:
        instance, metadata = build_reduced_instance(rho, density_cap, num_links)
        experiment_id = f"noto_support_m{num_links}_k{format_cap(density_cap)}_rho{rho:.2f}"
        noto.write_config(
            output_dir,
            experiment_id,
            metadata
            | {
                "experiment_version": EXPERIMENT_VERSION,
                "num_candidate_links": num_links,
                "num_states": len(instance.states),
                "ambiguity_density_cap": density_cap,
                "rho": rho,
                "max_nodes": max_nodes,
                "time_limit_sec": time_limit_sec,
                "certificate_absolute_tolerance": certificate_absolute_tolerance,
            },
        )
        key = (
            f"{EXPERIMENT_VERSION}__{experiment_id}__sbb_"
            f"n{max_nodes}_t{time_limit_sec:g}"
        )

        def compute() -> dict[str, Any]:
            last_reported_nodes = -50

            def trace(record: dict[str, Any]) -> None:
                nonlocal last_reported_nodes
                processed_nodes = int(record.get("processed_nodes", 0))
                if record.get("event") not in {"finish", "incumbent"} and processed_nodes < last_reported_nodes + 50:
                    return
                last_reported_nodes = processed_nodes
                message = (
                    f"m={num_links}, rho={rho:.2f}, cap={density_cap:g}: nodes={processed_nodes}, "
                    f"UB={record.get('incumbent_UB')}, LB={record.get('global_LB')}"
                )
                append_log(log_path, message)
                write_status(output_dir, "running", "sbb", message, log_path)

            result = solve_global_sbb(
                instance,
                epsilon=1e-2,
                fixed_y_epsilon=1e-5,
                cut_epsilon=1e-7,
                max_nodes=max_nodes,
                initial_y=warm_start,
                time_limit_sec=time_limit_sec,
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
                "nominal_objective": float(np.dot(fixed.nominal_distribution, fixed.state_losses)),
                "nominal_distribution": fixed.nominal_distribution.tolist(),
                "worst_case_distribution": fixed.worst_case_distribution.tolist(),
                "state_losses": fixed.state_losses.tolist(),
                "nodes_processed": int(result.nodes_processed),
                "nodes_remaining": int(result.nodes_remaining),
                "pruned_nodes": int(result.pruned_nodes),
                "max_tree_depth": int(result.max_tree_depth),
                "fixed_y_oracle_calls": int(result.fixed_y_oracle_calls),
                "recourse_cuts_generated": int(result.recourse_cuts_generated),
                "runtime_sec": float(result.runtime_sec),
                "converged": bool(result.converged),
                "termination_reason": result.termination_reason,
            }

        if monotonicity_anchor is not None and rho > monotonicity_anchor + 1e-12:
            if anchor_payload is None or float(anchor_payload["gap"]) > certificate_absolute_tolerance:
                raise RuntimeError("The monotonicity anchor must satisfy the declared certificate tolerance.")
            payload = derive_monotonicity_certificate(
                instance,
                anchor_payload,
                certificate_absolute_tolerance,
            )
        else:
            payload = cache.get_or_compute(key, compute, force=force)
        if monotonicity_anchor is not None and np.isclose(rho, monotonicity_anchor):
            anchor_payload = payload
        selected_y = np.asarray(payload["y"], dtype=float)
        warm_start = selected_y
        if rho == 0.0:
            reference_y = selected_y.copy()
            reference_z = np.asarray(payload["z"], dtype=float)
            reference_w = np.asarray(payload["w"], dtype=float)
            reference_objective = float(payload["objective"])
        if reference_y is None or reference_z is None or reference_w is None or reference_objective is None:
            raise RuntimeError("rho=0 must be solved first.")
        reference_evaluation = evaluate_fixed_y(instance, reference_y, epsilon=1e-5, max_iterations=160)
        nominal = np.asarray(payload["nominal_distribution"], dtype=float)
        worst_case = np.asarray(payload["worst_case_distribution"], dtype=float)
        positive_shift = np.maximum(worst_case - nominal, 0.0)
        positive_nominal = nominal > 1e-12
        max_density_ratio = float(np.max(worst_case[positive_nominal] / nominal[positive_nominal]))
        rows.append(
            {
                "experiment_id": experiment_id,
                "density_cap": density_cap,
                "rho": rho,
                "num_candidate_links": num_links,
                "num_states": len(instance.states),
                "objective": payload["objective"],
                "lower_bound": payload["lower_bound"],
                "absolute_gap": payload["gap"],
                "relative_gap_percent": payload["relative_gap_percent"],
                "selected_y_json": json.dumps(payload["y"]),
                "selected_z_json": json.dumps(payload["z"]),
                "selected_w_json": json.dumps(payload["w"]),
                "objective_using_rho0_policy": float(reference_evaluation.objective),
                "delta_rho_value": float(reference_evaluation.objective - payload["objective"]),
                "objective_increase_from_rho0": float(payload["objective"] - reference_objective),
                "y_diff_norm_from_rho0": float(np.linalg.norm(selected_y - reference_y, ord=1)),
                "z_diff_norm_from_rho0": float(np.linalg.norm(np.asarray(payload["z"]) - reference_z)),
                "w_diff_norm_from_rho0": float(np.linalg.norm(np.asarray(payload["w"]) - reference_w)),
                "total_positive_probability_shift": float(positive_shift.sum()),
                "mass_added_to_zero_nominal_states": float(positive_shift[nominal <= 1e-12].sum()),
                "max_realized_density_ratio": max_density_ratio,
                "density_cap_respected": max_density_ratio <= density_cap + 1e-9,
                "nodes_processed": payload["nodes_processed"],
                "runtime_sec": payload["runtime_sec"],
                "converged": payload["converged"],
                "termination_reason": payload["termination_reason"],
                "certificate_basis": payload.get("certificate_basis", "direct_sbb"),
                "certificate_absolute_tolerance": certificate_absolute_tolerance,
                "certified_at_tolerance": float(payload["gap"]) <= certificate_absolute_tolerance,
            }
        )
        noto.write_table(
            pd.DataFrame(rows),
            output_dir,
            f"table_noto_support_m{num_links}_certification",
        )


def statewise_nominal_root(instance: Any, feasible_y: np.ndarray):
    node = root_node(instance)
    nominal_instance = replace(instance, ambiguity_radius=0.0, ambiguity_density_cap=None)
    state_minima = []
    for state_index in range(len(instance.states)):
        state_distribution = np.zeros(len(instance.states), dtype=float)
        state_distribution[state_index] = 1.0
        state_minimum = evaluate_fixed_y(
            nominal_instance,
            feasible_y,
            epsilon=1e-6,
            max_iterations=120,
            nominal_distribution_override=state_distribution,
        ).objective
        state_minima.append(max(0.0, float(state_minimum) - 1e-6))
    if instance.ambiguity_density_cap is not None or instance.ambiguity_radius <= 1e-12:
        for state_index, state_minimum in enumerate(state_minima):
            node.theta_bounds[state_index, 0] = max(
                node.theta_bounds[state_index, 0],
                state_minimum,
            )
    if instance.ambiguity_radius > 1e-12:
        node.omega_bounds = (
            max(node.omega_bounds[0], min(state_minima)),
            node.omega_bounds[1],
        )
    return node


def derive_monotonicity_certificate(
    instance: Any,
    anchor_payload: dict[str, Any],
    certificate_absolute_tolerance: float,
) -> dict[str, Any]:
    started = time.time()
    fixed = evaluate_fixed_y(
        instance,
        np.asarray(anchor_payload["y"], dtype=float),
        epsilon=1e-6,
        max_iterations=160,
    )
    anchor_objective = float(anchor_payload["objective"])
    if abs(float(fixed.objective) - anchor_objective) > 1e-4:
        raise RuntimeError(
            "The anchor policy does not reproduce the anchor value, so monotonicity cannot close this row."
        )
    lower_bound = float(anchor_payload["lower_bound"])
    gap = float(fixed.objective - lower_bound)
    return {
        "objective": float(fixed.objective),
        "lower_bound": lower_bound,
        "gap": max(0.0, gap),
        "relative_gap_percent": 100.0 * max(0.0, gap) / max(1.0, abs(float(fixed.objective))),
        "y": fixed.y.tolist(),
        "z": fixed.z.tolist(),
        "w": fixed.w.tolist(),
        "nominal_objective": float(np.dot(fixed.nominal_distribution, fixed.state_losses)),
        "nominal_distribution": fixed.nominal_distribution.tolist(),
        "worst_case_distribution": fixed.worst_case_distribution.tolist(),
        "state_losses": fixed.state_losses.tolist(),
        "nodes_processed": 0,
        "nodes_remaining": 0,
        "pruned_nodes": 0,
        "max_tree_depth": 0,
        "fixed_y_oracle_calls": 1,
        "recourse_cuts_generated": 0,
        "runtime_sec": time.time() - started,
        "converged": gap <= certificate_absolute_tolerance,
        "termination_reason": "monotonicity_tolerance_certificate",
        "certificate_basis": "monotonicity from tolerance-certified anchor plus matching feasible incumbent",
    }


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")


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
