from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from ejor_dad import (
    AidCenter,
    DADInstance,
    Link,
    PiecewiseLinearResponseParams,
    SurvivalParams,
    ThresholdResponseParams,
    Zone,
    evaluate_fixed_plan,
    evaluate_fixed_y,
    generate_failure_states,
)
from ejor_dad.checkpoint import CheckpointStore, atomic_write_dataframe, atomic_write_text


DEFAULT_RHO_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
GRID_LEVELS = {
    "pilot": np.asarray([0.0, 0.50, 1.0], dtype=float),
    "full": np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=float),
}
CACHE_VERSION = "noto-access-v3"
PREPARED = Path("data_work/noto/prepared")
LOCAL_TRANSFER_MINUTES = 10.0
RENOVATION_BUDGET_FRACTION = 0.12
RETROFIT_BUDGET_FRACTION = 0.42
CAPACITY_BUDGET_FRACTION = 0.30


def main() -> None:
    args = parse_args()
    output_name = "access_experiment_pilot" if args.mode == "pilot" else "access_experiment"
    out = Path("data_work/noto") / output_name
    for path in [out / "tables", out / "logs", out / "checkpoints", out / "configs"]:
        path.mkdir(parents=True, exist_ok=True)
    external_log = os.environ.get("EJOR_LOG_PATH")
    log_path = Path(external_log) if external_log else out / "logs" / f"noto_access_{time.strftime('%Y%m%d_%H%M%S')}.log"
    cache = CheckpointStore(out / "checkpoints")
    rho_values = parse_rho_values(args.rho_values)
    write_status(out, "running", "startup", f"Noto {args.mode} experiment started.", log_path)

    try:
        started = time.time()
        run_discretized_dda(
            out=out,
            cache=cache,
            force=args.force,
            log_path=log_path,
            mode=args.mode,
            rho_values=rho_values,
            workers=args.workers,
        )
        atomic_write_text(out / "runtime_summary.json", json.dumps({"runtime_sec": time.time() - started}, indent=2))
        write_status(out, "completed", "complete", f"Noto {args.mode} experiment completed.", log_path, exit_code=0)
    except Exception as exc:
        write_status(out, "failed", "error", str(exc), log_path, exit_code=1)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the observed Noto access/capacity DR-DAD experiment.")
    parser.add_argument("--mode", choices=sorted(GRID_LEVELS), default="full")
    parser.add_argument("--rho-values", default=",".join(str(value) for value in DEFAULT_RHO_VALUES))
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true", help="Recompute fixed-y checkpoints.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1.")
    return args


def parse_rho_values(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0.0 or item > 1.0 for item in values):
        raise ValueError("rho values must be a nonempty comma-separated list in [0,1].")
    if 0.0 not in values:
        raise ValueError("rho=0 must be included to compute the nominal policy benchmark.")
    return sorted(set(values))


def run_discretized_dda(
    out: Path,
    cache: CheckpointStore,
    force: bool,
    log_path: Path,
    mode: str,
    rho_values: list[float],
    workers: int,
) -> None:
    grid = GRID_LEVELS[mode]
    summary_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    near_rows: list[dict[str, Any]] = []
    sector_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    y0: np.ndarray | None = None
    nominal_instance, _ = build_noto_instance(0.0)
    total_grid = int(len(grid) ** len(nominal_instance.links))

    for rho in rho_values:
        started = time.time()
        instance, metadata = build_noto_instance(rho)
        experiment_id = f"noto_access_m5_{mode}_rho{rho:.2f}"
        write_config(
            out,
            experiment_id,
            metadata | {"rho": rho, "mode": mode, "grid_levels": grid.tolist(), "total_grid_candidates": total_grid},
        )
        results, infeasible = evaluate_grid(
            instance=instance,
            grid=grid,
            experiment_id=experiment_id,
            cache=cache,
            force=force,
            out=out,
            log_path=log_path,
            mode=mode,
            rho=rho,
            total_grid=total_grid,
            workers=workers,
        )

        if not results:
            raise RuntimeError(f"No feasible retrofit policy found for rho={rho:.2f}.")
        results.sort(key=lambda item: item.objective)
        best = results[0]
        if rho == 0.0:
            y0 = np.asarray(best.y, dtype=float)
        if y0 is None:
            raise RuntimeError("rho=0 must be evaluated before positive ambiguity radii.")

        y0_eval = cached_fixed_y(cache, f"{experiment_id}_rho0_policy", instance, y0, force)
        nominal_selected = cached_fixed_y(cache, f"{experiment_id}_nominal_selected", nominal_instance, best.y, force)
        no_retrofit = cached_fixed_y(cache, f"{experiment_id}_no_retrofit", instance, np.zeros(len(instance.links)), force)
        exposure_only_instance = replace(instance, budget_capacity=0.0)
        capacity_only_instance = replace(instance, budget_renovation=0.0)
        exposure_only = cached_fixed_y(
            cache,
            f"{experiment_id}_exposure_only",
            exposure_only_instance,
            np.zeros(len(instance.links)),
            force,
        )
        capacity_only = cached_fixed_y(
            cache,
            f"{experiment_id}_capacity_only",
            capacity_only_instance,
            np.zeros(len(instance.links)),
            force,
        )
        no_investment_objective = evaluate_no_investment(instance)
        target_index = int(np.argmax(best.worst_case_distribution - best.nominal_distribution))
        target_state = instance.states[target_index]

        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "rho": rho,
                "mode": mode,
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
                "objective_using_rho0_policy": float(y0_eval.objective),
                "delta_rho_value": float(y0_eval.objective - best.objective),
                "y_diff_norm_from_rho0": float(np.linalg.norm(np.asarray(best.y) - y0, ord=1)),
                "nominal_objective_selected_y": float(nominal_selected.objective),
                "no_retrofit_zw_objective": float(no_retrofit.objective),
                "road_value_over_no_retrofit": float(no_retrofit.objective - best.objective),
                "no_investment_objective": float(no_investment_objective),
                "adversarial_target_state": target_state.id,
                "adversarial_target_failed_links_json": json.dumps(list(target_state.failed_links)),
                "adversarial_target_mass_shift": float(
                    best.worst_case_distribution[target_index] - best.nominal_distribution[target_index]
                ),
                "runtime_sec": time.time() - started,
                "data_status": "observed P,q,beds,road geometry and travel times; constructed Phi; scenario-calibrated costs and budgets",
            }
        )

        for rank, result in enumerate(results[:10], start=1):
            top_rows.append(
                {
                    "rho": rho,
                    "mode": mode,
                    "rank": rank,
                    "objective": float(result.objective),
                    "gap_to_best": float(result.objective - best.objective),
                    "gap_percent": 100.0 * float(result.objective - best.objective) / max(1.0, abs(best.objective)),
                    "selected_y_json": json.dumps(result.y.tolist()),
                    "y_budget_used": float(result.budget_used),
                }
            )

        for threshold in [0.01, 0.05, 0.10, 0.50]:
            count = sum(
                1
                for result in results
                if 100.0 * float(result.objective - best.objective) / max(1.0, abs(best.objective)) <= threshold + 1e-12
            )
            near_rows.append(
                {
                    "rho": rho,
                    "mode": mode,
                    "threshold_percent": threshold,
                    "near_optimal_policy_count": count,
                    "share_of_feasible_candidates": count / len(results),
                    "feasible_evaluated_candidates": len(results),
                }
            )

        comparisons = [
            ("no investment", no_investment_objective),
            ("exposure only", exposure_only.objective),
            ("capacity only", capacity_only.objective),
            ("exposure + capacity; no road retrofit", no_retrofit.objective),
            ("all-sector discretized", best.objective),
        ]
        for comparison, objective in comparisons:
            sector_rows.append(
                {
                    "rho": rho,
                    "mode": mode,
                    "comparison": comparison,
                    "objective": float(objective),
                    "gap_to_best": float(objective - best.objective),
                    "reduction_from_no_investment": float(no_investment_objective - objective),
                }
            )

        for state_index, state in enumerate(instance.states):
            shift = float(best.worst_case_distribution[state_index] - best.nominal_distribution[state_index])
            if abs(shift) <= 1e-10:
                continue
            target_rows.append(
                {
                    "rho": rho,
                    "mode": mode,
                    "state_id": state.id,
                    "failed_links_json": json.dumps(list(state.failed_links)),
                    "nominal_probability": float(best.nominal_distribution[state_index]),
                    "worst_case_probability": float(best.worst_case_distribution[state_index]),
                    "probability_shift": shift,
                    "state_loss": float(best.state_losses[state_index]),
                }
            )

        write_intermediate_tables(out, summary_rows, top_rows, near_rows, sector_rows, target_rows)

    write_intermediate_tables(out, summary_rows, top_rows, near_rows, sector_rows, target_rows)
    copy_prepared_tables(out)


def evaluate_grid(
    instance: DADInstance,
    grid: np.ndarray,
    experiment_id: str,
    cache: CheckpointStore,
    force: bool,
    out: Path,
    log_path: Path,
    mode: str,
    rho: float,
    total_grid: int,
    workers: int,
) -> tuple[list[SimpleNamespace], int]:
    results: list[SimpleNamespace] = []
    pending: list[tuple[int, np.ndarray, float, str]] = []
    infeasible = 0
    for candidate_index, values in enumerate(product(grid, repeat=len(instance.links)), start=1):
        y = np.asarray(values, dtype=float)
        budget_used = float(np.dot(instance.retrofit_costs, y))
        if budget_used > instance.budget_retrofit + 1e-9:
            infeasible += 1
            continue
        key = versioned_key(f"{experiment_id}__grid_{candidate_index:04d}_{hash_array(y)}")
        if not force and cache.exists(key):
            payload = cache.load(key)
            results.append(result_from_candidate_payload(payload, candidate_index, budget_used, loaded_from_cache=True))
        else:
            pending.append((candidate_index, y, budget_used, key))

    if workers == 1:
        for candidate_index, y, budget_used, key in pending:
            payload = evaluate_candidate_payload(instance, y, candidate_index, budget_used)
            cache.save(key, payload)
            results.append(result_from_candidate_payload(payload, candidate_index, budget_used, loaded_from_cache=False))
            report_grid_progress(out, log_path, mode, rho, results, infeasible, total_grid, workers)
        return results, infeasible

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_lookup = {
            executor.submit(evaluate_candidate_payload, instance, y, candidate_index, budget_used): (
                candidate_index,
                budget_used,
                key,
            )
            for candidate_index, y, budget_used, key in pending
        }
        for future in as_completed(future_lookup):
            candidate_index, budget_used, key = future_lookup[future]
            payload = future.result()
            cache.save(key, payload)
            results.append(result_from_candidate_payload(payload, candidate_index, budget_used, loaded_from_cache=False))
            report_grid_progress(out, log_path, mode, rho, results, infeasible, total_grid, workers)
    return results, infeasible


def evaluate_candidate_payload(
    instance: DADInstance,
    y: np.ndarray,
    candidate_index: int,
    budget_used: float,
) -> dict[str, Any]:
    evaluation_started = time.time()
    result = evaluate_fixed_y(instance, y, epsilon=1e-5, max_iterations=160)
    return fixed_result_payload(result) | {
        "candidate_index": candidate_index,
        "budget_used": budget_used,
        "eval_runtime_sec": time.time() - evaluation_started,
    }


def result_from_candidate_payload(
    payload: dict[str, Any],
    candidate_index: int,
    budget_used: float,
    loaded_from_cache: bool,
) -> SimpleNamespace:
    result = payload_to_result(payload)
    result.candidate_index = int(payload.get("candidate_index", candidate_index))
    result.budget_used = float(payload.get("budget_used", budget_used))
    result.loaded_from_cache = loaded_from_cache
    return result


def report_grid_progress(
    out: Path,
    log_path: Path,
    mode: str,
    rho: float,
    results: list[SimpleNamespace],
    infeasible: int,
    total_grid: int,
    workers: int,
) -> None:
    if len(results) % 25 != 0:
        return
    message = f"rho={rho:.2f}: evaluated {len(results)} feasible Noto policies with {workers} workers."
    print(message, flush=True)
    write_status(
        out,
        "running",
        "dda_sweep",
        message,
        log_path,
        extra={
            "mode": mode,
            "rho": rho,
            "evaluated": len(results),
            "infeasible": infeasible,
            "total_grid": total_grid,
            "workers": workers,
        },
    )

def build_noto_instance(
    rho: float,
    *,
    residual_failure_ratio: float = 0.0,
    failure_delay_reduction: float = 0.0,
    time_sensitive_fraction: float = 1.0,
    immediate_loss_fraction: float = 0.0,
    capacity_throughput_per_bed: float | None = None,
    response_threshold_minutes: float | None = None,
    response_curve_knots: Sequence[Sequence[float]] | None = None,
) -> tuple[DADInstance, dict[str, Any]]:
    if not 0.0 <= residual_failure_ratio <= 1.0:
        raise ValueError("residual_failure_ratio must lie in [0, 1].")
    if not 0.0 <= failure_delay_reduction <= 1.0:
        raise ValueError("failure_delay_reduction must lie in [0, 1].")
    if not 0.0 <= time_sensitive_fraction <= 1.0:
        raise ValueError("time_sensitive_fraction must lie in [0, 1].")
    if not 0.0 <= immediate_loss_fraction <= 1.0:
        raise ValueError("immediate_loss_fraction must lie in [0, 1].")
    if time_sensitive_fraction + immediate_loss_fraction > 1.0 + 1e-12:
        raise ValueError("time_sensitive_fraction and immediate_loss_fraction must sum to at most 1.")
    if capacity_throughput_per_bed is not None and capacity_throughput_per_bed <= 0.0:
        raise ValueError("capacity_throughput_per_bed must be positive when provided.")
    if response_threshold_minutes is not None and response_threshold_minutes < 0.0:
        raise ValueError("response_threshold_minutes must be nonnegative when provided.")
    if response_curve_knots is not None and response_threshold_minutes is not None:
        raise ValueError("Use either a response threshold or a graded response curve, not both.")
    zones_frame = pd.read_csv(PREPARED / "noto_zones.csv", dtype={"municipality_code": str})
    centers_frame = pd.read_csv(PREPARED / "noto_centers.csv", dtype={"municipality_code": str})
    corridors_frame = pd.read_csv(
        PREPARED / "noto_corridors.csv",
        dtype={"tail_code": str, "head_code": str},
    )
    zones = [
        Zone(
            id=str(row.zone_id),
            population=float(row.population),
            collapse_fraction=float(row.collapse_fraction),
            renovation_cost=float(row.renovation_cost),
            node=str(row.municipality_code),
            region=str(row.municipality_name_en),
            time_sensitive_fraction=time_sensitive_fraction,
            immediate_loss_fraction=immediate_loss_fraction,
        )
        for row in zones_frame.itertuples(index=False)
    ]
    centers = [
        AidCenter(
            id=str(row.center_id),
            node=str(row.municipality_code),
            existing_capacity=(
                float(row.existing_capacity)
                if capacity_throughput_per_bed is None
                else float(row.reported_beds)
                * float(row.operational_share)
                * capacity_throughput_per_bed
            ),
            capacity_unit_cost=float(row.capacity_unit_cost),
        )
        for row in centers_frame.itertuples(index=False)
    ]
    links = [
        Link(
            id=str(row.link_id),
            tail=str(row.tail_code),
            head=str(row.head_code),
            baseline_failure_probability=float(row.baseline_failure_probability),
            retrofit_cost=float(row.retrofit_cost),
            travel_time=float(row.normal_minutes),
            residual_failure_probability=(
                residual_failure_ratio * float(row.baseline_failure_probability)
            ),
            failure_delay_reduction=failure_delay_reduction,
        )
        for row in corridors_frame.itertuples(index=False)
    ]
    states = generate_failure_states(links, max_failures=None, include_tail=False)
    travel_times = precompute_travel_times(states, zones_frame, centers_frame, corridors_frame)
    intact_travel_times = travel_times["intact"]
    intact_array = np.asarray(intact_travel_times, dtype=float)
    failure_penalty_matrices = {
        str(row.link_id): (
            np.asarray(travel_times[f"fail__{row.link_id}"], dtype=float) - intact_array
        ).tolist()
        for row in corridors_frame.itertuples(index=False)
    }
    if response_curve_knots is not None:
        response_rule = PiecewiseLinearResponseParams(
            knots=tuple((float(minutes), float(value)) for minutes, value in response_curve_knots)
        )
        response_rule_description = "piecewise-linear graded timely-access credit"
    elif response_threshold_minutes is None:
        response_rule = SurvivalParams(a=0.96, b=-0.012, c=1.0, d=0.0)
        response_rule_description = "exponential travel-time survival curve"
    else:
        response_rule = ThresholdResponseParams(threshold_minutes=response_threshold_minutes)
        response_rule_description = (
            f"timely-access indicator with a {response_threshold_minutes:g}-minute threshold"
        )
    instance = DADInstance(
        zones=zones,
        links=links,
        centers=centers,
        budget_renovation=RENOVATION_BUDGET_FRACTION * sum(zone.renovation_cost for zone in zones),
        budget_retrofit=RETROFIT_BUDGET_FRACTION * sum(link.retrofit_cost for link in links),
        budget_capacity=CAPACITY_BUDGET_FRACTION * sum(center.existing_capacity for center in centers),
        ambiguity_radius=rho,
        states=states,
        survival=response_rule,
        precomputed_travel_times=travel_times,
        intact_travel_times=intact_travel_times,
        failure_penalty_matrices=failure_penalty_matrices,
    )
    metadata = {
        "source_population": "Ishikawa Prefecture 2022 municipality population and households.",
        "source_damage": "Ishikawa damage report No. 162, full-destroyed dwellings as of 2024-10-01.",
        "source_roads": "MLIT Noto restoration GeoJSON and January/June intercity travel-time observations.",
        "source_capacity": "MHLW FY2023 facility-reported beds.",
        "q_definition": "full_destroyed_dwellings / pre-event households",
        "phi_definition": "declared score using observed delay ratio and initial restoration points",
        "residual_failure_ratio": residual_failure_ratio,
        "failure_delay_reduction": failure_delay_reduction,
        "time_sensitive_fraction": time_sensitive_fraction,
        "immediate_loss_fraction": immediate_loss_fraction,
        "capacity_throughput_per_bed": (
            float(centers_frame.throughput_per_bed.iloc[0])
            if capacity_throughput_per_bed is None
            else capacity_throughput_per_bed
        ),
        "response_rule": response_rule_description,
        "response_curve_knots": (
            None
            if response_curve_knots is None
            else [[float(minutes), float(value)] for minutes, value in response_curve_knots]
        ),
        "outcome_definition": (
            "at-risk population minus modeled survivors"
            if time_sensitive_fraction == 1.0 and immediate_loss_fraction == 0.0
            else "immediate loss plus time-sensitive post-damage demand not served by emergency recourse"
        ),
        "performance_adjustment": (
            "failed-link delay penalty is multiplied by 1 - failure_delay_reduction * y_ij; "
            "nominal failure probability is residual + (baseline - residual) * (1 - y_ij)"
        ),
        "budget_fractions": {
            "renovation": RENOVATION_BUDGET_FRACTION,
            "retrofit": RETROFIT_BUDGET_FRACTION,
            "capacity": CAPACITY_BUDGET_FRACTION,
        },
        "local_transfer_minutes": LOCAL_TRANSFER_MINUTES,
        "calibration_locked_before_optimization": True,
        "zones": zones_frame.to_dict(orient="records"),
        "centers": centers_frame.to_dict(orient="records"),
        "corridors": corridors_frame.to_dict(orient="records"),
        "caveat": "Phi, costs, operational shares, and budgets remain constructed or scenario-calibrated and require sensitivity analysis.",
    }
    return instance, metadata
def precompute_travel_times(
    states: list[Any],
    zones_frame: pd.DataFrame,
    centers_frame: pd.DataFrame,
    corridors_frame: pd.DataFrame,
) -> dict[str, list[list[float]]]:
    graph = nx.Graph()
    for row in corridors_frame.itertuples(index=False):
        graph.add_edge(
            str(row.tail_code),
            str(row.head_code),
            link_id=str(row.link_id),
            normal_minutes=float(row.normal_minutes),
            penalty_minutes=float(row.failure_penalty_minutes),
        )
    paths: dict[tuple[str, str], list[str]] = {}
    for center in centers_frame.itertuples(index=False):
        for zone in zones_frame.itertuples(index=False):
            center_node = str(center.municipality_code)
            zone_node = str(zone.municipality_code)
            paths[(center_node, zone_node)] = nx.shortest_path(graph, center_node, zone_node, weight="normal_minutes")

    matrices: dict[str, list[list[float]]] = {}
    for state in states:
        failed = set(state.failed_links)
        matrix = np.zeros((len(centers_frame), len(zones_frame)), dtype=float)
        for center_index, center in enumerate(centers_frame.itertuples(index=False)):
            for zone_index, zone in enumerate(zones_frame.itertuples(index=False)):
                path = paths[(str(center.municipality_code), str(zone.municipality_code))]
                travel_time = LOCAL_TRANSFER_MINUTES
                for tail, head in zip(path, path[1:]):
                    edge = graph[tail][head]
                    travel_time += edge["normal_minutes"]
                    if edge["link_id"] in failed:
                        travel_time += edge["penalty_minutes"]
                matrix[center_index, zone_index] = travel_time
        matrices[state.id] = matrix.tolist()
    return matrices


def cached_fixed_y(
    cache: CheckpointStore,
    key_prefix: str,
    instance: DADInstance,
    y: np.ndarray,
    force: bool,
) -> SimpleNamespace:
    payload = cache.get_or_compute(
        versioned_key(f"{key_prefix}__fixed_y_{hash_array(y)}"),
        lambda: fixed_result_payload(evaluate_fixed_y(instance, y, epsilon=1e-5, max_iterations=160)),
        force=force,
    )
    return payload_to_result(payload)


def evaluate_no_investment(instance: DADInstance) -> float:
    from ejor_dad.fixed_y import evaluate_plan_losses
    from ejor_dad.states import nominal_probabilities
    from ejor_dad.tv import worst_case_tv_distribution

    y = np.zeros(len(instance.links), dtype=float)
    z = np.zeros(len(instance.zones), dtype=float)
    w = np.zeros(len(instance.centers), dtype=float)
    losses, _, _ = evaluate_plan_losses(instance, z, w, y=y)
    nominal = nominal_probabilities(instance.links, instance.states, y)
    return float(
        worst_case_tv_distribution(
            nominal,
            losses,
            instance.ambiguity_radius,
            maximize=True,
            density_cap=instance.ambiguity_density_cap,
        ).value
    )


def fixed_result_payload(result: Any) -> dict[str, Any]:
    return {
        "objective": float(result.objective),
        "lower_bound": float(result.lower_bound),
        "z": result.z.tolist(),
        "w": result.w.tolist(),
        "y": result.y.tolist(),
        "nominal_distribution": result.nominal_distribution.tolist(),
        "worst_case_distribution": result.worst_case_distribution.tolist(),
        "state_losses": result.state_losses.tolist(),
        "state_survivors": result.state_survivors.tolist(),
        "iterations": int(result.iterations),
    }


def payload_to_result(payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        objective=float(payload["objective"]),
        lower_bound=float(payload.get("lower_bound", payload["objective"])),
        z=np.asarray(payload["z"], dtype=float),
        w=np.asarray(payload["w"], dtype=float),
        y=np.asarray(payload["y"], dtype=float),
        nominal_distribution=np.asarray(payload["nominal_distribution"], dtype=float),
        worst_case_distribution=np.asarray(payload["worst_case_distribution"], dtype=float),
        state_losses=np.asarray(payload["state_losses"], dtype=float),
        state_survivors=np.asarray(payload["state_survivors"], dtype=float),
        iterations=int(payload.get("iterations", 0)),
    )


def hash_array(values: np.ndarray) -> str:
    rounded = np.round(np.asarray(values, dtype=float), 8)
    return hashlib.sha1(rounded.tobytes()).hexdigest()[:16]


def versioned_key(key: str) -> str:
    return f"{CACHE_VERSION}__{key}"


def write_config(out: Path, experiment_id: str, payload: dict[str, Any]) -> None:
    atomic_write_text(out / "configs" / f"{experiment_id}.json", json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def write_table(dataframe: pd.DataFrame, out: Path, stem: str) -> None:
    atomic_write_dataframe(dataframe, out / "tables" / f"{stem}.csv")
    atomic_write_dataframe(dataframe, out / "tables" / f"{stem}.tex", kind="latex", escape=True)


def write_intermediate_tables(
    out: Path,
    summary_rows: list[dict[str, Any]],
    top_rows: list[dict[str, Any]],
    near_rows: list[dict[str, Any]],
    sector_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> None:
    write_table(pd.DataFrame(summary_rows), out, "table_noto_access_dda_summary")
    write_table(pd.DataFrame(top_rows), out, "table_noto_access_top10")
    write_table(pd.DataFrame(near_rows), out, "table_noto_access_near_optimal")
    write_table(pd.DataFrame(sector_rows), out, "table_noto_access_sector_comparison")
    write_table(pd.DataFrame(target_rows), out, "table_noto_access_probability_shifts")


def copy_prepared_tables(out: Path) -> None:
    for name in ["noto_zones", "noto_centers", "noto_corridors", "noto_data_coverage", "noto_source_manifest"]:
        write_table(pd.read_csv(PREPARED / f"{name}.csv"), out, f"table_{name}")


def write_status(
    out: Path,
    status: str,
    block: str,
    message: str,
    log_path: Path,
    exit_code: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "status": status,
        "block": block,
        "message": message,
        "pid": os.getpid(),
        "updated_at_epoch": time.time(),
        "log_path": str(log_path.resolve()),
        "output_dir": str(out.resolve()),
        "checkpoint_dir": str((out / "checkpoints").resolve()),
    }
    if exit_code is not None:
        payload["exit_code"] = exit_code
    if extra:
        payload.update(extra)
    atomic_write_text(out / "run_status.json", json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

