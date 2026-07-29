from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ejor_dad import FailureMomentEnvelope, evaluate_fixed_y, nominal_probabilities
from ejor_dad.checkpoint import CheckpointStore, atomic_write_text
from ejor_dad.fixed_y import evaluate_plan_losses
from ejor_dad.moments import (
    build_failure_moment_system,
    failure_indicator_matrix,
    moment_bound_diagnostics,
)
from ejor_dad.tv import worst_case_tv_distribution

import noto_access_experiment as noto
import noto_active_fine_grid as fine
import noto_tight_budget_analysis as tight


DEFAULT_RHOS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
EXPERIMENT_VERSION = "noto-moment-envelope-v1"
BASELINE_SUMMARY = Path(
    "data_work/noto/support_preserving_active_fine_grid/tables/"
    "table_noto_active_fine_grid_summary.csv"
)
CENTER_LABELS = ["Kanazawa", "Nanao", "Wajima", "Suzu", "Anamizu", "Noto"]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    for path in [output_dir / "tables", output_dir / "figures", output_dir / "logs", output_dir / "checkpoints"]:
        path.mkdir(parents=True, exist_ok=True)
    external_log = os.environ.get("EJOR_LOG_PATH")
    log_path = (
        Path(external_log)
        if external_log
        else output_dir / "logs" / f"noto_moment_progress_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    envelope = build_envelope(args)
    rho_values = tight.parse_float_list(args.rho_values, 0.0, 1.0)
    base_instance, _ = noto.build_noto_instance(0.0)
    critical_indices = fine.identify_critical_indices(base_instance)
    budget = args.budget_multiplier * base_instance.budget_retrofit
    candidates = fine.enumerate_candidates(base_instance, critical_indices, budget, args.grid_step)
    initial_completed = count_checkpoints(output_dir)
    profile_id = envelope_id(envelope, args.profile_name)
    write_design(
        output_dir,
        args,
        envelope,
        profile_id,
        rho_values,
        base_instance,
        critical_indices,
        budget,
        candidates,
    )
    write_status(
        output_dir,
        "running",
        "startup",
        "Moment-constrained Noto grid started.",
        log_path,
        completed=count_checkpoints(output_dir),
        expected=len(candidates) * len(rho_values),
    )
    started = time.time()
    try:
        run_sweep(
            output_dir,
            CheckpointStore(output_dir / "checkpoints"),
            log_path,
            args,
            envelope,
            profile_id,
            rho_values,
            critical_indices,
            budget,
            candidates,
        )
        write_numerical_audit(output_dir, envelope, args, budget)
        update_runtime_summary(
            output_dir,
            last_run_runtime_sec=time.time() - started,
            initial_completed=initial_completed,
            final_completed=count_checkpoints(output_dir),
            expected=len(candidates) * len(rho_values),
        )
        write_status(
            output_dir,
            "completed",
            "complete",
            "Moment-constrained Noto grid completed.",
            log_path,
            completed=count_checkpoints(output_dir),
            expected=len(candidates) * len(rho_values),
        )
    except Exception as exc:
        write_status(
            output_dir,
            "failed",
            "error",
            str(exc),
            log_path,
            completed=count_checkpoints(output_dir),
            expected=len(candidates) * len(rho_values),
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a restart-safe Noto grid with failure-moment ambiguity bounds."
    )
    parser.add_argument(
        "--output-dir",
        default="data_work/noto/moment_constrained_active_fine_grid",
    )
    parser.add_argument("--rho-values", default=",".join(str(value) for value in DEFAULT_RHOS))
    parser.add_argument("--density-cap", type=float, default=2.0)
    parser.add_argument("--budget-multiplier", type=float, default=fine.DEFAULT_ACTIVE_BUDGET_MULTIPLIER)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--profile-name", default="moderate")
    parser.add_argument("--marginal-relative", type=float, default=0.25)
    parser.add_argument("--marginal-absolute", type=float, default=0.02)
    parser.add_argument("--joint-relative", type=float, default=0.50)
    parser.add_argument("--joint-absolute", type=float, default=0.01)
    parser.add_argument("--count-mean-absolute", type=float, default=0.15)
    parser.add_argument("--count-second-relative", type=float, default=0.25)
    parser.add_argument("--count-second-absolute", type=float, default=0.05)
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


def build_envelope(args: argparse.Namespace) -> FailureMomentEnvelope:
    return FailureMomentEnvelope(
        marginal_relative_tolerance=args.marginal_relative,
        marginal_absolute_tolerance=args.marginal_absolute,
        joint_relative_tolerance=args.joint_relative,
        joint_absolute_tolerance=args.joint_absolute,
        count_mean_absolute_tolerance=args.count_mean_absolute,
        count_second_moment_relative_tolerance=args.count_second_relative,
        count_second_moment_absolute_tolerance=args.count_second_absolute,
    )


def run_sweep(
    output_dir: Path,
    cache: CheckpointStore,
    log_path: Path,
    args: argparse.Namespace,
    envelope: FailureMomentEnvelope,
    profile_id: str,
    rho_values: list[float],
    critical_indices: list[int],
    budget: float,
    candidates: list[dict[str, Any]],
) -> None:
    baseline_summary = pd.read_csv(BASELINE_SUMMARY)
    baseline_reference_row = baseline_summary[np.isclose(baseline_summary["rho"], 0.0)].iloc[0]
    baseline_reference_y = np.asarray(json.loads(baseline_reference_row["best_y_json"]), dtype=float)
    baseline_reference_z = np.asarray(json.loads(baseline_reference_row["best_z_json"]), dtype=float)
    baseline_reference_w = np.asarray(json.loads(baseline_reference_row["best_w_json"]), dtype=float)
    summary_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    near_rows: list[dict[str, Any]] = []
    bound_rows: list[dict[str, Any]] = []
    marginal_rows: list[dict[str, Any]] = []
    reference_y = baseline_reference_y.copy()
    reference_z = baseline_reference_z.copy()
    reference_w = baseline_reference_w.copy()

    for rho in rho_values:
        instance, _ = noto.build_noto_instance(rho)
        instance = replace(
            instance,
            budget_retrofit=budget,
            ambiguity_density_cap=args.density_cap,
            failure_moment_envelope=envelope,
        )
        experiment_id = f"{profile_id}__rho{rho:.2f}"
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
            reference_y = best.y.copy()
            reference_z = best.z.copy()
            reference_w = best.w.copy()
        reference = next(
            result
            for result in results
            if np.allclose(result.y, reference_y, atol=1e-12, rtol=0.0)
        )
        reference_full_objective = fixed_plan_worst_case_value(
            instance,
            reference_y,
            reference_z,
            reference_w,
        )
        baseline_matches = baseline_summary[
            np.isclose(baseline_summary["rho"], rho)
            & np.isclose(baseline_summary["grid_step"], args.grid_step)
            & np.isclose(baseline_summary["budget_multiplier"], args.budget_multiplier)
        ]
        comparison_is_grid_matched = not baseline_matches.empty
        baseline_row = baseline_matches.iloc[0] if comparison_is_grid_matched else None
        baseline_y = (
            np.asarray(json.loads(baseline_row["best_y_json"]), dtype=float)
            if baseline_row is not None
            else best.y.copy()
        )
        baseline_instance = replace(
            instance,
            failure_moment_envelope=None,
        )
        baseline_result = evaluate_fixed_y(
            baseline_instance,
            baseline_y,
            epsilon=1e-5,
            max_iterations=160,
        )
        baseline_at_moment_policy = (
            baseline_result
            if np.allclose(baseline_y, best.y, atol=1e-12, rtol=0.0)
            else evaluate_fixed_y(
                baseline_instance,
                best.y,
                epsilon=1e-5,
                max_iterations=160,
            )
        )
        baseline_reference_full_objective = fixed_plan_worst_case_value(
            baseline_instance,
            baseline_reference_y,
            baseline_reference_z,
            baseline_reference_w,
        )
        moment_statistics = distribution_statistics(instance, best)
        baseline_statistics = distribution_statistics(baseline_instance, baseline_result)
        system = build_failure_moment_system(instance, best.nominal_distribution)
        diagnostics = moment_bound_diagnostics(system, best.worst_case_distribution, tolerance=1e-7)
        max_violation = max(
            [
                max(0.0, -float(row["lower_slack"]), -float(row["upper_slack"]))
                for row in diagnostics
            ],
            default=0.0,
        )
        active_labels = [str(row["name"]) for row in diagnostics if bool(row["active"])]
        limiting_labels = [
            str(row["name"])
            for row in diagnostics
            if bool(row["active"])
            and not (
                abs(float(row["nominal_value"])) <= 1e-10
                and abs(float(row["value"])) <= 1e-10
            )
        ]
        summary_rows.append(
            {
                "rho": rho,
                "density_cap": args.density_cap,
                "budget_multiplier": args.budget_multiplier,
                "B_Y": budget,
                "grid_step": args.grid_step,
                "feasible_grid_candidates": len(results),
                "best_objective": float(best.objective),
                "baseline_comparison_grid_matched": comparison_is_grid_matched,
                "baseline_capped_tv_objective": (
                    float(baseline_row["best_objective"])
                    if baseline_row is not None
                    else np.nan
                ),
                "capped_tv_objective_at_moment_policy": float(baseline_at_moment_policy.objective),
                "moment_reduction_from_capped_tv": (
                    float(baseline_row["best_objective"] - best.objective)
                    if baseline_row is not None
                    else np.nan
                ),
                "moment_reduction_at_same_policy": float(
                    baseline_at_moment_policy.objective - best.objective
                ),
                "baseline_objective_using_rho0_full_policy": baseline_reference_full_objective,
                "baseline_full_policy_delta_rho_value": float(
                    baseline_reference_full_objective - float(baseline_row["best_objective"])
                    if baseline_row is not None
                    else np.nan
                ),
                "best_y_json": json.dumps(best.y.tolist()),
                "best_z_json": json.dumps(best.z.tolist()),
                "best_w_json": json.dumps(best.w.tolist()),
                "baseline_best_y_json": json.dumps(baseline_y.tolist()),
                "policy_changed_from_capped_tv": (
                    not np.allclose(best.y, baseline_y, atol=1e-12, rtol=0.0)
                    if comparison_is_grid_matched
                    else None
                ),
                "objective_using_rho0_policy": float(reference.objective),
                "delta_rho_value": float(reference.objective - best.objective),
                "objective_using_rho0_y_with_reoptimized_zw": float(reference.objective),
                "road_policy_delta_rho_value": float(reference.objective - best.objective),
                "objective_using_rho0_full_policy": reference_full_objective,
                "full_policy_delta_rho_value": float(reference_full_objective - best.objective),
                "full_policy_delta_rho_percent": 100.0
                * float(reference_full_objective - best.objective)
                / max(1.0, abs(float(best.objective))),
                "policy_changed_from_rho0": not np.allclose(
                    best.y,
                    reference_y,
                    atol=1e-12,
                    rtol=0.0,
                ),
                "y_diff_norm_from_rho0": float(np.linalg.norm(best.y - reference_y, ord=1)),
                "z_diff_norm_from_rho0": float(np.linalg.norm(best.z - reference_z)),
                "w_diff_norm_from_rho0": float(np.linalg.norm(best.w - reference_w)),
                "capacity_policy_changed_from_rho0": not np.allclose(
                    best.w,
                    reference_w,
                    atol=1e-8,
                    rtol=0.0,
                ),
                "second_best_gap": float(results[1].objective - best.objective),
                "algorithm1_gap": float(best.objective - best.lower_bound),
                "tv_mass_used": moment_statistics["tv_mass_used"],
                "nominal_failure_count_mean": moment_statistics["nominal_count_mean"],
                "worst_failure_count_mean": moment_statistics["worst_count_mean"],
                "nominal_failure_count_sd": moment_statistics["nominal_count_sd"],
                "worst_failure_count_sd": moment_statistics["worst_count_sd"],
                "baseline_worst_failure_count_mean": baseline_statistics["worst_count_mean"],
                "baseline_worst_failure_count_sd": baseline_statistics["worst_count_sd"],
                "active_moment_bounds": len(active_labels),
                "active_moment_bound_labels_json": json.dumps(active_labels),
                "active_nonzero_moment_bounds": len(limiting_labels),
                "active_nonzero_moment_bound_labels_json": json.dumps(limiting_labels),
                "max_moment_violation": max_violation,
                "mass_added_to_zero_nominal_states": fine.support_leakage(best),
                "max_realized_density_ratio": fine.max_density_ratio(best),
            }
        )
        for row in diagnostics:
            bound_rows.append({"rho": rho} | row)
        for link_index, link in enumerate(instance.links):
            marginal_rows.append(
                {
                    "rho": rho,
                    "link_index": link_index + 1,
                    "link_id": link.id,
                    "moment_nominal_marginal": moment_statistics["nominal_marginals"][link_index],
                    "moment_worst_marginal": moment_statistics["worst_marginals"][link_index],
                    "baseline_nominal_marginal": baseline_statistics["nominal_marginals"][link_index],
                    "baseline_worst_marginal": baseline_statistics["worst_marginals"][link_index],
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
        write_tables(
            output_dir,
            summary_rows,
            candidate_rows,
            top_rows,
            near_rows,
            bound_rows,
            marginal_rows,
        )

    summary = pd.DataFrame(summary_rows)
    marginals = pd.DataFrame(marginal_rows)
    make_figures(summary, marginals, output_dir / "figures", critical_indices)


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
        "moment_grid",
        message,
        log_path,
        completed=completed,
        expected=expected,
    )


def distribution_statistics(instance: Any, result: Any) -> dict[str, Any]:
    indicators = failure_indicator_matrix(instance)
    failure_count = indicators.sum(axis=1)
    nominal = np.asarray(result.nominal_distribution, dtype=float)
    worst = np.asarray(result.worst_case_distribution, dtype=float)
    nominal_mean = float(nominal @ failure_count)
    worst_mean = float(worst @ failure_count)
    return {
        "nominal_marginals": nominal @ indicators,
        "worst_marginals": worst @ indicators,
        "nominal_count_mean": nominal_mean,
        "worst_count_mean": worst_mean,
        "nominal_count_sd": float(np.sqrt(nominal @ (failure_count - nominal_mean) ** 2)),
        "worst_count_sd": float(np.sqrt(worst @ (failure_count - worst_mean) ** 2)),
        "worst_fixed_center_second_moment": float(worst @ (failure_count - nominal_mean) ** 2),
        "tv_mass_used": 0.5 * float(np.abs(worst - nominal).sum()),
    }


def fixed_plan_worst_case_value(
    instance: Any,
    y: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
) -> float:
    nominal = nominal_probabilities(instance.links, instance.states, y)
    losses, _, _ = evaluate_plan_losses(instance, z, w)
    system = build_failure_moment_system(instance, nominal)
    result = worst_case_tv_distribution(
        nominal,
        losses,
        instance.ambiguity_radius,
        maximize=True,
        density_cap=instance.ambiguity_density_cap,
        inequality_matrix=system.inequality_matrix if system.bounds else None,
        inequality_rhs=system.inequality_rhs if system.bounds else None,
    )
    return float(result.value)


def write_tables(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    top_rows: list[dict[str, Any]],
    near_rows: list[dict[str, Any]],
    bound_rows: list[dict[str, Any]],
    marginal_rows: list[dict[str, Any]],
) -> None:
    noto.write_table(pd.DataFrame(summary_rows), output_dir, "table_noto_moment_envelope_summary")
    noto.write_table(pd.DataFrame(candidate_rows), output_dir, "table_noto_moment_envelope_candidates")
    noto.write_table(pd.DataFrame(top_rows), output_dir, "table_noto_moment_envelope_top10")
    noto.write_table(pd.DataFrame(near_rows), output_dir, "table_noto_moment_envelope_near_optimal")
    noto.write_table(pd.DataFrame(bound_rows), output_dir, "table_noto_moment_envelope_bounds")
    noto.write_table(pd.DataFrame(marginal_rows), output_dir, "table_noto_moment_envelope_marginals")


def write_numerical_audit(
    output_dir: Path,
    envelope: FailureMomentEnvelope,
    args: argparse.Namespace,
    budget: float,
) -> None:
    metrics = {
        "evaluated_policies": 0,
        "max_algorithm1_gap": 0.0,
        "max_nominal_probability_sum_error": 0.0,
        "max_worst_probability_sum_error": 0.0,
        "max_support_leakage": 0.0,
        "max_density_ratio": 0.0,
        "max_tv_radius_excess": 0.0,
        "max_moment_bound_violation": 0.0,
        "max_objective_identity_error": 0.0,
        "max_algorithm1_iterations": 0,
    }
    instances: dict[float, Any] = {}
    for path in (output_dir / "checkpoints").glob(f"*{EXPERIMENT_VERSION}*.json"):
        match = re.search(r"__rho([0-9]+\.[0-9]+)__", path.name)
        if match is None:
            raise RuntimeError(f"Cannot recover rho from checkpoint name: {path.name}")
        rho = float(match.group(1))
        if rho not in instances:
            instance, _ = noto.build_noto_instance(rho)
            instances[rho] = replace(
                instance,
                budget_retrofit=budget,
                ambiguity_density_cap=args.density_cap,
                failure_moment_envelope=envelope,
            )
        instance = instances[rho]
        payload = json.loads(path.read_text(encoding="utf-8"))
        nominal = np.asarray(payload["nominal_distribution"], dtype=float)
        worst = np.asarray(payload["worst_case_distribution"], dtype=float)
        losses = np.asarray(payload["state_losses"], dtype=float)
        system = build_failure_moment_system(instance, nominal)
        positive = nominal > 1e-12
        metrics["evaluated_policies"] += 1
        metrics["max_algorithm1_gap"] = max(
            metrics["max_algorithm1_gap"],
            abs(float(payload["objective"]) - float(payload["lower_bound"])),
        )
        metrics["max_nominal_probability_sum_error"] = max(
            metrics["max_nominal_probability_sum_error"],
            abs(float(nominal.sum()) - 1.0),
        )
        metrics["max_worst_probability_sum_error"] = max(
            metrics["max_worst_probability_sum_error"],
            abs(float(worst.sum()) - 1.0),
        )
        metrics["max_support_leakage"] = max(
            metrics["max_support_leakage"],
            float(np.maximum(worst - nominal, 0.0)[~positive].sum()),
        )
        metrics["max_density_ratio"] = max(
            metrics["max_density_ratio"],
            float(np.max(worst[positive] / nominal[positive])),
        )
        metrics["max_tv_radius_excess"] = max(
            metrics["max_tv_radius_excess"],
            0.5 * float(np.abs(worst - nominal).sum()) - rho,
        )
        if system.inequality_matrix.size:
            metrics["max_moment_bound_violation"] = max(
                metrics["max_moment_bound_violation"],
                float(np.max(system.inequality_matrix @ worst - system.inequality_rhs)),
            )
        metrics["max_objective_identity_error"] = max(
            metrics["max_objective_identity_error"],
            abs(float(payload["objective"]) - float(worst @ losses)),
        )
        metrics["max_algorithm1_iterations"] = max(
            metrics["max_algorithm1_iterations"],
            int(payload["iterations"]),
        )
    if metrics["max_algorithm1_gap"] > 1e-5:
        raise RuntimeError("Moment-constrained fixed-y evaluation exceeded the declared gap tolerance.")
    if metrics["max_support_leakage"] > 1e-10:
        raise RuntimeError("Moment-constrained ambiguity leaked probability outside nominal support.")
    if metrics["max_density_ratio"] > args.density_cap + 1e-8:
        raise RuntimeError("Moment-constrained ambiguity violated the density cap.")
    if metrics["max_tv_radius_excess"] > 1e-8:
        raise RuntimeError("Moment-constrained ambiguity exceeded the TV radius.")
    if metrics["max_moment_bound_violation"] > 1e-8:
        raise RuntimeError("Moment-constrained ambiguity violated a moment bound.")
    noto.write_table(
        pd.DataFrame([metrics]),
        output_dir,
        "table_noto_moment_envelope_numerical_audit",
    )
    atomic_write_text(
        output_dir / "numerical_audit.json",
        json.dumps(metrics, indent=2),
    )


def make_figures(
    summary: pd.DataFrame,
    marginals: pd.DataFrame,
    output_dir: Path,
    critical_indices: list[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    baseline_objective = summary["baseline_capped_tv_objective"].fillna(
        summary["capped_tv_objective_at_moment_policy"]
    )
    axis.plot(summary["rho"], baseline_objective, marker="o", label="Capped TV")
    axis.plot(summary["rho"], summary["best_objective"], marker="s", label="Moment-constrained TV")
    axis.set_xlabel(r"Ambiguity radius $\rho$")
    axis.set_ylabel("Best worst-case modeled loss")
    axis.set_title("Effect of failure-moment bounds")
    axis.legend()
    save_figure(figure, output_dir / "fig_noto_moment_01_objective")

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    axes[0].plot(summary["rho"], summary["nominal_failure_count_mean"], marker="o", label="Nominal")
    axes[0].plot(summary["rho"], summary["baseline_worst_failure_count_mean"], marker="o", label="Capped TV")
    axes[0].plot(summary["rho"], summary["worst_failure_count_mean"], marker="s", label="Moment constrained")
    axes[0].set_xlabel(r"Ambiguity radius $\rho$")
    axes[0].set_ylabel("Expected failed links")
    axes[0].set_title("Failure-count mean")
    axes[0].legend()
    axes[1].plot(summary["rho"], summary["nominal_failure_count_sd"], marker="o", label="Nominal")
    axes[1].plot(summary["rho"], summary["baseline_worst_failure_count_sd"], marker="o", label="Capped TV")
    axes[1].plot(summary["rho"], summary["worst_failure_count_sd"], marker="s", label="Moment constrained")
    axes[1].set_xlabel(r"Ambiguity radius $\rho$")
    axes[1].set_ylabel("Failed-link standard deviation")
    axes[1].set_title("Failure-count dispersion")
    axes[1].legend()
    save_figure(figure, output_dir / "fig_noto_moment_02_failure_moments")

    last_rho = float(summary["rho"].max())
    last = marginals[np.isclose(marginals["rho"], last_rho)].sort_values("link_index")
    positions = np.arange(len(last))
    width = 0.25
    figure, axis = plt.subplots(figsize=(9.2, 5.0))
    axis.bar(positions - width, last["moment_nominal_marginal"], width, label="Nominal")
    axis.bar(positions, last["baseline_worst_marginal"], width, label="Capped TV")
    axis.bar(positions + width, last["moment_worst_marginal"], width, label="Moment constrained")
    axis.set_xticks(positions, [f"L{int(index)}" for index in last["link_index"]])
    axis.set_ylabel("Failure probability")
    axis.set_title(rf"Link-failure marginals at $\rho={last_rho:.2f}$")
    axis.legend()
    save_figure(figure, output_dir / "fig_noto_moment_03_link_marginals")

    y_values = np.vstack(summary["best_y_json"].map(json.loads).map(np.asarray))
    figure, axis = plt.subplots(figsize=(8.4, 5.0))
    for index in critical_indices:
        axis.plot(summary["rho"], y_values[:, index], marker="o", label=f"L{index + 1}")
    axis.set_xlabel(r"Ambiguity radius $\rho$")
    axis.set_ylabel("Retrofit level")
    axis.set_ylim(-0.03, 1.03)
    axis.set_title("Moment-constrained road-policy trajectory")
    axis.legend()
    save_figure(figure, output_dir / "fig_noto_moment_04_policy")

    w_values = np.vstack(summary["best_w_json"].map(json.loads).map(np.asarray))
    active_centers = np.flatnonzero(np.max(w_values, axis=0) > 1e-8)
    figure, axis = plt.subplots(figsize=(8.4, 5.0))
    for index in active_centers:
        axis.plot(summary["rho"], w_values[:, index], marker="o", label=CENTER_LABELS[index])
    axis.set_xlabel(r"Ambiguity radius $\rho$")
    axis.set_ylabel("Capacity expansion units")
    axis.set_title("Moment-constrained capacity-policy trajectory")
    axis.legend()
    save_figure(figure, output_dir / "fig_noto_moment_05_capacity_policy")


def save_figure(figure: Any, path_without_suffix: Path) -> None:
    figure.tight_layout()
    figure.savefig(path_without_suffix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(path_without_suffix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def envelope_id(envelope: FailureMomentEnvelope, profile_name: str) -> str:
    values = [
        envelope.marginal_relative_tolerance,
        envelope.marginal_absolute_tolerance,
        envelope.joint_relative_tolerance,
        envelope.joint_absolute_tolerance,
        envelope.count_mean_absolute_tolerance,
        envelope.count_second_moment_relative_tolerance,
        envelope.count_second_moment_absolute_tolerance,
    ]
    encoded = "_".join("none" if value is None else f"{value:.4f}".replace(".", "p") for value in values)
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", profile_name).strip("_") or "profile"
    return f"{safe_name}__{encoded}"


def write_design(
    output_dir: Path,
    args: argparse.Namespace,
    envelope: FailureMomentEnvelope,
    profile_id: str,
    rho_values: list[float],
    instance: Any,
    critical_indices: list[int],
    budget: float,
    candidates: list[dict[str, Any]],
) -> None:
    payload = {
        "experiment_version": EXPERIMENT_VERSION,
        "profile_id": profile_id,
        "failure_moment_envelope": asdict(envelope),
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
        "scope": "Full five-link state and recourse model with moment-constrained ambiguity and two refined y variables.",
    }
    atomic_write_text(output_dir / "run_design.json", json.dumps(payload, indent=2))


def count_checkpoints(output_dir: Path) -> int:
    checkpoint_dir = output_dir / "checkpoints"
    return sum(1 for _ in checkpoint_dir.glob(f"*{EXPERIMENT_VERSION}*.json")) if checkpoint_dir.exists() else 0


def update_runtime_summary(
    output_dir: Path,
    last_run_runtime_sec: float,
    initial_completed: int,
    final_completed: int,
    expected: int,
) -> None:
    path = output_dir / "runtime_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload["last_run_runtime_sec"] = last_run_runtime_sec
    payload["initial_completed_evaluations"] = initial_completed
    payload["final_completed_evaluations"] = final_completed
    payload["expected_evaluations"] = expected
    if initial_completed == 0 and final_completed == expected:
        payload["full_compute_runtime_sec"] = last_run_runtime_sec
    atomic_write_text(path, json.dumps(payload, indent=2))


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
    completed: int,
    expected: int,
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
                "completed_checkpoints": completed,
                "expected_evaluations": expected,
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
