from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import noto_access_experiment as noto
from ejor_dad.checkpoint import CheckpointStore, atomic_write_text


EXPERIMENT_VERSION = "noto-practical-resilience-v4"
DEFAULT_DENSITY_CAP = 2.0
DEFAULT_RESIDUAL_FAILURE_RATIO = 0.10
DEFAULT_FAILURE_DELAY_REDUCTION = 0.50
DEFAULT_TIME_SENSITIVE_FRACTION = 1.0
DEFAULT_IMMEDIATE_LOSS_FRACTION = 0.0
DEFAULT_GRADED_RESPONSE_KNOTS = ((0.0, 1.0), (30.0, 1.0), (60.0, 0.75), (120.0, 0.25), (180.0, 0.0))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    for directory in [output_dir / "tables", output_dir / "logs", output_dir / "checkpoints", output_dir / "configs"]:
        directory.mkdir(parents=True, exist_ok=True)
    external_log = os.environ.get("EJOR_LOG_PATH")
    log_path = Path(external_log) if external_log else output_dir / "logs" / f"noto_practical_{time.strftime('%Y%m%d_%H%M%S')}.log"
    cache = CheckpointStore(output_dir / "checkpoints")
    rho_values = noto.parse_rho_values(args.rho_values)
    design = design_payload(args, rho_values)
    atomic_write_text(output_dir / "run_design.json", json.dumps(design, indent=2))
    append_log(log_path, "Started practical Noto residual-risk and conditional-delay experiment.")
    noto.write_status(
        output_dir,
        "running",
        "startup",
        "Practical Noto experiment started; exact fixed-y checkpoints are restart-safe.",
        log_path,
        extra={"experiment_version": EXPERIMENT_VERSION, "rho_values": rho_values},
    )

    started = time.time()
    try:
        run_sweep(output_dir, cache, log_path, args, rho_values)
        runtime = time.time() - started
        atomic_write_text(output_dir / "runtime_summary.json", json.dumps({"runtime_sec": runtime}, indent=2))
        append_log(log_path, f"Completed practical Noto experiment in {runtime:.1f} seconds.")
        noto.write_status(
            output_dir,
            "completed",
            "complete",
            "Practical Noto experiment completed.",
            log_path,
            exit_code=0,
            extra={"experiment_version": EXPERIMENT_VERSION, "runtime_sec": runtime},
        )
    except Exception as exc:
        append_log(log_path, f"FAILED: {exc}")
        noto.write_status(
            output_dir,
            "failed",
            "error",
            str(exc),
            log_path,
            exit_code=1,
            extra={"experiment_version": EXPERIMENT_VERSION},
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restart-safe exact Noto enumeration with residual link risk, "
            "retrofit-dependent disruption delays, and support-preserving capped-TV ambiguity."
        )
    )
    parser.add_argument("--mode", choices=sorted(noto.GRID_LEVELS), default="full")
    parser.add_argument("--rho-values", default=",".join(str(value) for value in noto.DEFAULT_RHO_VALUES))
    parser.add_argument("--density-cap", type=float, default=DEFAULT_DENSITY_CAP)
    parser.add_argument("--residual-failure-ratio", type=float, default=DEFAULT_RESIDUAL_FAILURE_RATIO)
    parser.add_argument("--failure-delay-reduction", type=float, default=DEFAULT_FAILURE_DELAY_REDUCTION)
    parser.add_argument("--time-sensitive-fraction", type=float, default=DEFAULT_TIME_SENSITIVE_FRACTION)
    parser.add_argument("--immediate-loss-fraction", type=float, default=DEFAULT_IMMEDIATE_LOSS_FRACTION)
    parser.add_argument(
        "--capacity-throughput-per-bed",
        type=float,
        default=None,
        help="Response-window throughput per operational bed; defaults to the prepared four-cycle value.",
    )
    parser.add_argument(
        "--response-threshold-minutes",
        type=float,
        default=None,
        help="Use a binary timely-access outcome at this response threshold instead of the legacy survival curve.",
    )
    parser.add_argument(
        "--graded-response",
        action="store_true",
        help="Use the declared piecewise-linear graded timely-access credit instead of a binary cutoff.",
    )
    parser.add_argument("--retrofit-budget-scale", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--output-dir",
        default="data_work/noto/practical_resilience",
        help="Directory containing restartable checkpoints and paper-ready tables.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute existing exact fixed-y checkpoints.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1.")
    if args.density_cap < 1.0:
        parser.error("--density-cap must be at least 1.")
    if not 0.0 <= args.residual_failure_ratio <= 1.0:
        parser.error("--residual-failure-ratio must lie in [0, 1].")
    if not 0.0 <= args.failure_delay_reduction <= 1.0:
        parser.error("--failure-delay-reduction must lie in [0, 1].")
    if not 0.0 <= args.time_sensitive_fraction <= 1.0:
        parser.error("--time-sensitive-fraction must lie in [0, 1].")
    if not 0.0 <= args.immediate_loss_fraction <= 1.0:
        parser.error("--immediate-loss-fraction must lie in [0, 1].")
    if args.time_sensitive_fraction + args.immediate_loss_fraction > 1.0 + 1e-12:
        parser.error("--time-sensitive-fraction and --immediate-loss-fraction must sum to at most 1.")
    if args.capacity_throughput_per_bed is not None and args.capacity_throughput_per_bed <= 0.0:
        parser.error("--capacity-throughput-per-bed must be positive when provided.")
    if args.response_threshold_minutes is not None and args.response_threshold_minutes < 0.0:
        parser.error("--response-threshold-minutes must be nonnegative when provided.")
    if args.graded_response and args.response_threshold_minutes is not None:
        parser.error("--graded-response cannot be combined with --response-threshold-minutes.")
    if args.retrofit_budget_scale <= 0.0:
        parser.error("--retrofit-budget-scale must be positive.")
    return args


def design_payload(args: argparse.Namespace, rho_values: Sequence[float]) -> dict[str, Any]:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "mode": args.mode,
        "grid_levels": noto.GRID_LEVELS[args.mode].tolist(),
        "rho_values": list(rho_values),
        "ambiguity_set": "support-preserving capped total variation",
        "density_cap": args.density_cap,
        "residual_failure_ratio": args.residual_failure_ratio,
        "failure_delay_reduction": args.failure_delay_reduction,
        "retrofit_budget_scale": args.retrofit_budget_scale,
        "time_sensitive_fraction": args.time_sensitive_fraction,
        "immediate_loss_fraction": args.immediate_loss_fraction,
        "capacity_throughput_per_bed": args.capacity_throughput_per_bed,
        "response_threshold_minutes": args.response_threshold_minutes,
        "graded_response": args.graded_response,
        "graded_response_knots": (
            [list(knot) for knot in DEFAULT_GRADED_RESPONSE_KNOTS] if args.graded_response else None
        ),
        "outcome_definition": (
            "graded timely-access credit for time-sensitive post-damage demand"
            if args.graded_response
            else (
                "timely access for time-sensitive post-damage demand"
                if args.response_threshold_minutes is not None
                else "legacy travel-time survival outcome"
            )
        ),
        "failure_probability_formula": "phi_ij(y) = floor_ij + (Phi_ij - floor_ij) * (1 - y_ij)",
        "conditional_delay_formula": "penalty_ij(y) = base_penalty_ij * (1 - gamma_ij * y_ij)",
        "solution_method": "exhaustive finite-grid fixed-y evaluation; each candidate uses Algorithm 1",
        "workers": args.workers,
    }


def run_sweep(
    output_dir: Path,
    cache: CheckpointStore,
    log_path: Path,
    args: argparse.Namespace,
    rho_values: Sequence[float],
) -> None:
    grid = noto.GRID_LEVELS[args.mode]
    reference_y: np.ndarray | None = None
    reference_z: np.ndarray | None = None
    reference_w: np.ndarray | None = None
    summary_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    near_rows: list[dict[str, Any]] = []
    heuristic_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    sector_rows: list[dict[str, Any]] = []

    template, _ = build_instance(0.0, args)
    total_grid = int(len(grid) ** len(template.links))

    for rho in rho_values:
        block_started = time.time()
        instance, metadata = build_instance(rho, args)
        experiment_id = experiment_key(args, rho)
        noto.write_config(
            output_dir,
            experiment_id,
            metadata
            | {
                "experiment_version": EXPERIMENT_VERSION,
                "rho": rho,
                "mode": args.mode,
                "grid_levels": grid.tolist(),
                "total_grid_candidates": total_grid,
            },
        )
        append_log(log_path, f"rho={rho:.2f}: evaluating up to {total_grid} finite-grid retrofit policies.")
        results, infeasible = noto.evaluate_grid(
            instance=instance,
            grid=grid,
            experiment_id=experiment_id,
            cache=cache,
            force=args.force,
            out=output_dir,
            log_path=log_path,
            mode=args.mode,
            rho=rho,
            total_grid=total_grid,
            workers=args.workers,
        )
        if not results:
            raise RuntimeError(f"No grid-feasible retrofit policy found for rho={rho:.2f}.")
        results.sort(key=lambda result: result.objective)
        best = results[0]
        if np.isclose(rho, 0.0):
            reference_y = np.asarray(best.y, dtype=float)
            reference_z = np.asarray(best.z, dtype=float)
            reference_w = np.asarray(best.w, dtype=float)
        if reference_y is None or reference_z is None or reference_w is None:
            raise RuntimeError("rho=0 must be evaluated before positive ambiguity radii.")

        reference_eval = noto.cached_fixed_y(
            cache, f"{experiment_id}_rho0_policy", instance, reference_y, args.force
        )
        no_retrofit = noto.cached_fixed_y(
            cache,
            f"{experiment_id}_no_road_retrofit",
            instance,
            np.zeros(len(instance.links), dtype=float),
            args.force,
        )
        reference_complete_plan = noto.evaluate_fixed_plan(
            instance,
            reference_z,
            reference_w,
            reference_y,
        )
        selected_complete_plan = noto.evaluate_fixed_plan(instance, best.z, best.w, best.y)
        nominal_instance, _ = build_instance(0.0, args)
        nominal_selected = noto.cached_fixed_y(
            cache, f"{experiment_id}_nominal_selected", nominal_instance, best.y, args.force
        )
        no_investment = noto.evaluate_no_investment(instance)
        heuristic_rows.extend(
            evaluate_heuristics(cache, experiment_id, instance, grid, best, args.force, rho)
        )
        target_index = int(np.argmax(best.worst_case_distribution - best.nominal_distribution))
        target_state = instance.states[target_index]
        density_ratios = np.divide(
            best.worst_case_distribution,
            best.nominal_distribution,
            out=np.zeros_like(best.worst_case_distribution),
            where=best.nominal_distribution > 1e-12,
        )
        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "rho": rho,
                "mode": args.mode,
                "num_zones": len(instance.zones),
                "num_centers": len(instance.centers),
                "num_links": len(instance.links),
                "num_states": len(instance.states),
                "total_grid_candidates": total_grid,
                "feasible_evaluated_candidates": len(results),
                "infeasible_budget_candidates": infeasible,
                "best_discretized_objective": float(best.objective),
                "best_y_json": json.dumps(best.y.tolist()),
                "best_z_json": json.dumps(best.z.tolist()),
                "best_w_json": json.dumps(best.w.tolist()),
                "best_y_budget_used": float(best.budget_used),
                "effective_failure_probabilities_json": json.dumps(
                    instance.effective_failure_probabilities(best.y).tolist()
                ),
                "rho0_policy_objective": float(reference_eval.objective),
                "delta_rho_value": float(reference_eval.objective - best.objective),
                "y_l1_distance_from_rho0": float(np.linalg.norm(best.y - reference_y, ord=1)),
                "z_l1_distance_from_rho0": float(np.linalg.norm(best.z - reference_z, ord=1)),
                "w_l1_distance_from_rho0": float(np.linalg.norm(best.w - reference_w, ord=1)),
                "rho0_complete_policy_objective": float(reference_complete_plan.objective),
                "complete_policy_adjustment_value": float(
                    reference_complete_plan.objective - selected_complete_plan.objective
                ),
                "rho0_complete_policy_nominal_objective": float(reference_complete_plan.nominal_objective),
                "selected_complete_policy_nominal_objective": float(selected_complete_plan.nominal_objective),
                "nominal_objective_selected_y": float(nominal_selected.objective),
                "no_road_retrofit_objective": float(no_retrofit.objective),
                "road_value_over_no_retrofit": float(no_retrofit.objective - best.objective),
                "no_investment_objective": float(no_investment),
                "adversarial_target_state": target_state.id,
                "adversarial_target_failed_links_json": json.dumps(list(target_state.failed_links)),
                "adversarial_target_mass_shift": float(
                    best.worst_case_distribution[target_index] - best.nominal_distribution[target_index]
                ),
                "maximum_density_ratio": float(density_ratios.max()),
                "density_cap": instance.ambiguity_density_cap,
                "residual_failure_ratio": args.residual_failure_ratio,
                "failure_delay_reduction": args.failure_delay_reduction,
                "retrofit_budget_scale": args.retrofit_budget_scale,
                "time_sensitive_fraction": args.time_sensitive_fraction,
                "immediate_loss_fraction": args.immediate_loss_fraction,
                "capacity_throughput_per_bed": metadata["capacity_throughput_per_bed"],
                "response_threshold_minutes": args.response_threshold_minutes,
                "runtime_sec": time.time() - block_started,
            }
        )

        for rank, result in enumerate(results[:10], start=1):
            top_rows.append(
                {
                    "rho": rho,
                    "rank": rank,
                    "objective": float(result.objective),
                    "gap_to_best": float(result.objective - best.objective),
                    "gap_percent": percent_gap(result.objective, best.objective),
                    "selected_y_json": json.dumps(result.y.tolist()),
                    "y_budget_used": float(result.budget_used),
                    "candidate_index": int(result.candidate_index),
                }
            )
        for threshold in [0.01, 0.05, 0.10, 0.50]:
            count = sum(percent_gap(result.objective, best.objective) <= threshold + 1e-12 for result in results)
            near_rows.append(
                {
                    "rho": rho,
                    "threshold_percent": threshold,
                    "near_optimal_policy_count": count,
                    "share_of_feasible_candidates": count / len(results),
                    "feasible_evaluated_candidates": len(results),
                }
            )
        for state_index, state in enumerate(instance.states):
            shift = float(best.worst_case_distribution[state_index] - best.nominal_distribution[state_index])
            if abs(shift) <= 1e-10:
                continue
            shift_rows.append(
                {
                    "rho": rho,
                    "state_id": state.id,
                    "failed_links_json": json.dumps(list(state.failed_links)),
                    "nominal_probability": float(best.nominal_distribution[state_index]),
                    "worst_case_probability": float(best.worst_case_distribution[state_index]),
                    "probability_shift": shift,
                    "state_loss": float(best.state_losses[state_index]),
                    "density_ratio": float(density_ratios[state_index]),
                }
            )
        for comparison, objective in [
            ("no investment", no_investment),
            ("exposure and capacity; no road retrofit", no_retrofit.objective),
            ("practical all-sector grid optimum", best.objective),
        ]:
            sector_rows.append(
                {
                    "rho": rho,
                    "comparison": comparison,
                    "objective": float(objective),
                    "gap_to_best": float(objective - best.objective),
                    "reduction_from_no_investment": float(no_investment - objective),
                }
            )

        write_tables(output_dir, summary_rows, top_rows, near_rows, heuristic_rows, shift_rows, sector_rows)
        append_log(log_path, f"rho={rho:.2f}: completed {len(results)} feasible policies; best={best.objective:.6f}.")
        noto.write_status(
            output_dir,
            "running",
            "rho_complete",
            f"Completed rho={rho:.2f} with {len(results)} exact grid policies.",
            log_path,
            extra={
                "experiment_version": EXPERIMENT_VERSION,
                "rho": rho,
                "evaluated": len(results),
                "total_grid": total_grid,
                "infeasible": infeasible,
                "completed_rho_values": [row["rho"] for row in summary_rows],
            },
        )

    noto.copy_prepared_tables(output_dir)
    write_tables(output_dir, summary_rows, top_rows, near_rows, heuristic_rows, shift_rows, sector_rows)


def build_instance(rho: float, args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    instance, metadata = noto.build_noto_instance(
        rho,
        residual_failure_ratio=args.residual_failure_ratio,
        failure_delay_reduction=args.failure_delay_reduction,
        time_sensitive_fraction=args.time_sensitive_fraction,
        immediate_loss_fraction=args.immediate_loss_fraction,
        capacity_throughput_per_bed=args.capacity_throughput_per_bed,
        response_threshold_minutes=args.response_threshold_minutes,
        response_curve_knots=(DEFAULT_GRADED_RESPONSE_KNOTS if args.graded_response else None),
    )
    instance = replace(
        instance,
        ambiguity_density_cap=args.density_cap,
        budget_retrofit=args.retrofit_budget_scale * instance.budget_retrofit,
    )
    return instance, metadata | {
        "ambiguity_set": "support-preserving capped total variation",
        "density_cap": args.density_cap,
        "retrofit_budget_scale": args.retrofit_budget_scale,
        "time_sensitive_fraction": args.time_sensitive_fraction,
        "immediate_loss_fraction": args.immediate_loss_fraction,
        "capacity_throughput_per_bed": args.capacity_throughput_per_bed,
        "response_threshold_minutes": args.response_threshold_minutes,
        "graded_response": args.graded_response,
        "graded_response_knots": (
            [list(knot) for knot in DEFAULT_GRADED_RESPONSE_KNOTS] if args.graded_response else None
        ),
    }


def experiment_key(args: argparse.Namespace, rho: float) -> str:
    response_tag = (
        "graded"
        if args.graded_response
        else (
            "legacy"
            if args.response_threshold_minutes is None
            else f"threshold{args.response_threshold_minutes:.0f}"
        )
    )
    return (
        f"{EXPERIMENT_VERSION}_m5_{args.mode}_rho{rho:.2f}"
        f"_k{args.density_cap:.2f}_floor{args.residual_failure_ratio:.2f}"
        f"_delay{args.failure_delay_reduction:.2f}_budget{args.retrofit_budget_scale:.3f}"
        f"_demand{args.time_sensitive_fraction:.2f}_direct{args.immediate_loss_fraction:.2f}"
        f"_cap{'prepared' if args.capacity_throughput_per_bed is None else f'{args.capacity_throughput_per_bed:.2f}'}"
        f"_response{response_tag}"
    ).replace(".", "p")


def evaluate_heuristics(
    cache: CheckpointStore,
    experiment_id: str,
    instance: Any,
    grid: np.ndarray,
    best: Any,
    force: bool,
    rho: float,
) -> list[dict[str, Any]]:
    scores = heuristic_scores(instance)
    rows: list[dict[str, Any]] = []
    for policy_name, score in scores.items():
        order = np.argsort(-score, kind="stable")
        y = greedy_grid_policy(instance, grid, order)
        result = noto.cached_fixed_y(cache, f"{experiment_id}_heuristic_{policy_name}", instance, y, force)
        rows.append(
            {
                "rho": rho,
                "heuristic": policy_name,
                "objective": float(result.objective),
                "gap_to_grid_optimum": float(result.objective - best.objective),
                "gap_percent": percent_gap(result.objective, best.objective),
                "selected_y_json": json.dumps(y.tolist()),
                "retrofit_budget_used": float(np.dot(instance.retrofit_costs, y)),
                "scores_json": json.dumps(score.tolist()),
            }
        )
    return rows


def heuristic_scores(instance: Any) -> dict[str, np.ndarray]:
    costs = np.maximum(np.asarray(instance.retrofit_costs, dtype=float), 1e-12)
    baseline = np.asarray(instance.failure_probabilities, dtype=float)
    delay_impact = np.asarray(
        [
            np.asarray(instance.failure_penalty_matrices[link.id], dtype=float).mean()
            for link in instance.links
        ],
        dtype=float,
    )
    return {
        "least_cost_first": 1.0 / costs,
        "baseline_failure_per_cost": baseline / costs,
        "failure_delay_per_cost": baseline * delay_impact / costs,
    }


def greedy_grid_policy(instance: Any, grid: np.ndarray, order: np.ndarray) -> np.ndarray:
    y = np.zeros(len(instance.links), dtype=float)
    remaining = float(instance.budget_retrofit)
    for index in order:
        cost = float(instance.retrofit_costs[index])
        if cost <= 1e-12:
            y[index] = float(grid[-1])
            continue
        affordable = min(1.0, remaining / cost)
        feasible_levels = grid[grid <= affordable + 1e-12]
        if feasible_levels.size == 0:
            continue
        y[index] = float(feasible_levels[-1])
        remaining -= cost * y[index]
    return y


def percent_gap(value: float, best: float) -> float:
    return 100.0 * float(value - best) / max(1.0, abs(float(best)))


def write_tables(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    top_rows: list[dict[str, Any]],
    near_rows: list[dict[str, Any]],
    heuristic_rows: list[dict[str, Any]],
    shift_rows: list[dict[str, Any]],
    sector_rows: list[dict[str, Any]],
) -> None:
    noto.write_table(pd.DataFrame(summary_rows), output_dir, "table_noto_practical_summary")
    noto.write_table(pd.DataFrame(top_rows), output_dir, "table_noto_practical_top10")
    noto.write_table(pd.DataFrame(near_rows), output_dir, "table_noto_practical_near_optimal")
    noto.write_table(pd.DataFrame(heuristic_rows), output_dir, "table_noto_practical_heuristics")
    noto.write_table(pd.DataFrame(shift_rows), output_dir, "table_noto_practical_probability_shifts")
    noto.write_table(pd.DataFrame(sector_rows), output_dir, "table_noto_practical_sector_comparison")


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


if __name__ == "__main__":
    main()

