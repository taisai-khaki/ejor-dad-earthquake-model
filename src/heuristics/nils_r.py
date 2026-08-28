"""Priority-risk rollback extension of NILS."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, inf
from time import perf_counter
from typing import Any

import numpy as np

from ..data_models import InstanceData, SolutionData
from ..distance_utils import manhattan_distance
from ..feasibility import is_feasible
from .local_search import apply_two_opt_to_solution
from .nils import (
    _empty_solution_clone,
    _sync_route_assignments,
    evaluate_solution,
)


@dataclass
class ReverseReassignment:
    solution: SolutionData | None
    g_z: float = float("-inf")
    g_hpt: float = float("-inf")
    g_tail: float = float("-inf")
    feasible_reverse_found: bool = False
    best_truck_id: int | None = None
    best_insertion_position: int | None = None
    added_truck_route_duration: float = float("inf")


@dataclass
class InsertionResult:
    reconstructed_solution: SolutionData
    feasible: bool
    inserted_customers: tuple[int, ...] = ()


@dataclass
class RollbackAttempt:
    solution: SolutionData | None
    accepted: bool
    candidates_evaluated: int
    selected_customers: tuple[int, ...]
    log: dict[str, Any]


def _evaluate(instance: InstanceData, solution: SolutionData) -> None:
    _sync_route_assignments(solution, instance)
    objective, components = evaluate_solution(instance, solution)
    solution.objective = objective
    solution.components = components


def _delivery_time(instance: InstanceData, solution: SolutionData, customer_id: int) -> float:
    arrivals = [
        float(solution.a_truck[(truck_id, customer_id)])
        for truck_id in instance.truck_ids()
        if solution.u_truck.get((truck_id, customer_id), 0) == 1
        and (truck_id, customer_id) in solution.a_truck
    ]
    arrivals.extend(
        float(solution.a_drone[(drone_id, customer_id)])
        for drone_id in instance.drone_ids()
        if solution.u_drone.get((drone_id, customer_id), 0) == 1
        and (drone_id, customer_id) in solution.a_drone
    )
    return max(arrivals) if arrivals else float("inf")


def compute_priority_metrics(instance: InstanceData, solution: SolutionData) -> dict[str, Any]:
    """Compute priority-service metrics from a fully evaluated solution."""
    _evaluate(instance, solution)
    tardiness_by_customer = {
        int(customer.node_id): float(solution.tardiness.get(customer.node_id, 0.0))
        for customer in instance.customers
    }
    by_priority = {
        priority: [
            tardiness_by_customer[customer.node_id]
            for customer in instance.customers
            if customer.priority == priority
        ]
        for priority in ("high", "medium", "low")
    }
    high_customers = [customer for customer in instance.customers if customer.priority == "high"]
    high_tardiness = [tardiness_by_customer[customer.node_id] for customer in high_customers]
    high_delivery_times = {
        int(customer.node_id): _delivery_time(instance, solution, customer.node_id)
        for customer in high_customers
    }
    high_on_time = [
        high_delivery_times[customer.node_id] <= float(customer.ub) + 1e-9
        for customer in high_customers
    ]
    drone_customers = solution.served_by_drone()
    return {
        "priority_tardiness_cost": float(solution.components.get("tardiness_cost", 0.0)),
        "high_priority_tardiness": float(sum(high_tardiness)),
        "medium_priority_tardiness": float(sum(by_priority["medium"])),
        "low_priority_tardiness": float(sum(by_priority["low"])),
        "max_high_priority_tardiness": float(max(high_tardiness, default=0.0)),
        "p90_high_priority_tardiness": float(np.percentile(high_tardiness, 90.0)) if high_tardiness else 0.0,
        "high_priority_on_time_rate": float(np.mean(high_on_time)) if high_on_time else 1.0,
        "high_priority_delivery_times": high_delivery_times,
        "tardiness_by_customer": tardiness_by_customer,
        "high_priority_served_by_drone_share": float(
            sum(customer.node_id in drone_customers for customer in high_customers) / max(1, len(high_customers))
        ),
    }


def _remove_customer(candidate: SolutionData, instance: InstanceData, customer_id: int) -> None:
    for truck_id, route in list(candidate.truck_routes.items()):
        route = [node for node in route if node != customer_id]
        candidate.truck_routes[truck_id] = route if len(route) >= 2 else [0, 0]
    for drone_id, route in list(candidate.drone_routes.items()):
        route = [node for node in route if node != customer_id]
        candidate.drone_routes[drone_id] = route if len(route) >= 2 else [0, 0]
    candidate.z1 = {key: value for key, value in candidate.z1.items() if key[0] != customer_id}
    candidate.z2 = {key: value for key, value in candidate.z2.items() if key[0] != customer_id}
    candidate.x_truck = {}
    candidate.x_drone = {}
    _sync_route_assignments(candidate, instance)


def _truck_candidate(
    instance: InstanceData,
    solution: SolutionData,
    customer_id: int,
    truck_id: int,
    position: int,
) -> SolutionData:
    candidate = _empty_solution_clone(solution)
    _remove_customer(candidate, instance, customer_id)
    route = list(candidate.truck_routes.get(truck_id, [0, 0]))
    if not route or route[0] != 0:
        route = [0] + route
    if route[-1] != 0:
        route.append(0)
    route.insert(max(1, min(int(position), len(route) - 1)), int(customer_id))
    candidate.truck_routes[truck_id] = route
    _sync_route_assignments(candidate, instance)
    return candidate


def _truck_route_distance(instance: InstanceData, truck_id: int, route: list[int]) -> float:
    coordinates = instance.coordinate_map()
    return float(
        sum(
            manhattan_distance(coordinates[left], coordinates[right])
            for left, right in zip(route, route[1:])
        )
    )


def _candidate_key(
    instance: InstanceData,
    candidate: SolutionData,
    truck_id: int,
) -> tuple[float, float, float, float, float]:
    customer_by_id = instance.customer_map()
    high_priority_tardiness = sum(
        float(candidate.tardiness.get(customer_id, 0.0))
        for customer_id, customer in customer_by_id.items()
        if customer.priority == "high"
    )
    max_high_priority_tardiness = max(
        (
            float(candidate.tardiness.get(customer_id, 0.0))
            for customer_id, customer in customer_by_id.items()
            if customer.priority == "high"
        ),
        default=0.0,
    )
    route = list(candidate.truck_routes.get(truck_id, [0, 0]))
    return (
        float(candidate.objective),
        high_priority_tardiness,
        max_high_priority_tardiness,
        float(candidate.l_truck.get((truck_id, 0), inf)),
        _truck_route_distance(instance, truck_id, route),
    )


def evaluate_reverse_reassignment(
    instance: InstanceData,
    solution: SolutionData,
    customer_j: int,
) -> ReverseReassignment:
    """Evaluate all feasible truck insertions for one drone-served customer."""
    base = _empty_solution_clone(solution)
    _evaluate(instance, base)
    base_metrics = compute_priority_metrics(instance, base)
    best: SolutionData | None = None
    best_key: tuple[float, float, float, float, float] | None = None
    best_truck_id: int | None = None
    best_position: int | None = None

    for truck_id in instance.truck_ids():
        route = list(base.truck_routes.get(truck_id, [0, 0]))
        for position in range(1, len(route)):
            candidate = _truck_candidate(instance, base, customer_j, truck_id, position)
            _evaluate(instance, candidate)
            if not is_feasible(instance, candidate):
                continue
            key = _candidate_key(instance, candidate, truck_id)
            if best_key is None or key < best_key:
                best = candidate
                best_key = key
                best_truck_id = truck_id
                best_position = position

    if best is None:
        return ReverseReassignment(solution=None)
    after_metrics = compute_priority_metrics(instance, best)
    selected_truck_id = int(best_truck_id) if best_truck_id is not None else 0
    base_duration = float(base.l_truck.get((selected_truck_id, 0), 0.0))
    after_duration = float(best.l_truck.get((selected_truck_id, 0), 0.0))
    return ReverseReassignment(
        solution=best,
        g_z=float(base.objective - best.objective),
        g_hpt=float(base_metrics["high_priority_tardiness"] - after_metrics["high_priority_tardiness"]),
        g_tail=float(base_metrics["max_high_priority_tardiness"] - after_metrics["max_high_priority_tardiness"]),
        feasible_reverse_found=True,
        best_truck_id=best_truck_id,
        best_insertion_position=best_position,
        added_truck_route_duration=float(after_duration - base_duration),
    )


def priority_aware_truck_insertion(
    instance: InstanceData,
    solution: SolutionData,
    customer_set_R: set[int],
) -> InsertionResult:
    """Reinsert rollback customers using priority-aware feasible cheapest insertion."""
    current = _empty_solution_clone(solution)
    _evaluate(instance, current)
    before_metrics = compute_priority_metrics(instance, current)
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    customer_by_id = instance.customer_map()
    ordered = sorted(
        (int(customer_id) for customer_id in customer_set_R),
        key=lambda customer_id: (
            priority_rank.get(customer_by_id[customer_id].priority, 3),
            -float(before_metrics["tardiness_by_customer"].get(customer_id, 0.0)),
            customer_id,
        ),
    )
    inserted: list[int] = []
    for customer_id in ordered:
        best: SolutionData | None = None
        best_key: tuple[float, float, float, float, float] | None = None
        for truck_id in instance.truck_ids():
            route = list(current.truck_routes.get(truck_id, [0, 0]))
            for position in range(1, len(route)):
                candidate = _truck_candidate(instance, current, customer_id, truck_id, position)
                _evaluate(instance, candidate)
                if not is_feasible(instance, candidate):
                    continue
                key = _candidate_key(instance, candidate, truck_id)
                if best_key is None or key < best_key:
                    best = candidate
                    best_key = key
        if best is None:
            return InsertionResult(current, False, tuple(inserted))
        current = best
        inserted.append(customer_id)

    optimized = apply_two_opt_to_solution(current, instance)
    _evaluate(instance, optimized)
    if is_feasible(instance, optimized):
        current = optimized
    return InsertionResult(current, bool(is_feasible(instance, current)), tuple(inserted))


def accept_reconstruction(
    s_old: SolutionData,
    s_new: SolutionData,
    s_best: SolutionData,
    *,
    instance: InstanceData,
    epsilon_z: float = 1e-6,
    epsilon_h: float = 1e-6,
    epsilon_tail: float = 1e-6,
) -> bool:
    """Apply the objective and priority guard for an NILS-R reconstruction."""
    if not is_feasible(instance, s_new):
        return False
    new_metrics = compute_priority_metrics(instance, s_new)
    best_metrics = compute_priority_metrics(instance, s_best)
    return bool(
        float(s_new.objective) < float(s_best.objective) - epsilon_z
        and new_metrics["high_priority_tardiness"] <= best_metrics["high_priority_tardiness"] + epsilon_h
        and new_metrics["max_high_priority_tardiness"] <= best_metrics["max_high_priority_tardiness"] + epsilon_tail
    )


def _attempt_priority_risk_rollback(
    instance: InstanceData,
    current: SolutionData,
    best: SolutionData,
    *,
    trigger_reason: str,
    reconstruction_attempt: int,
    size_cap: int,
    epsilon_z: float,
    epsilon_h: float,
    epsilon_tail: float,
) -> RollbackAttempt:
    started = perf_counter()
    source = _empty_solution_clone(best)
    _evaluate(instance, source)
    before = compute_priority_metrics(instance, source)
    drone_customers = sorted(source.served_by_drone())
    rollback_size = min(max(1, int(size_cap)), int(ceil(0.05 * len(drone_customers))))
    reverse: dict[int, ReverseReassignment] = {
        customer_id: evaluate_reverse_reassignment(instance, source, customer_id)
        for customer_id in drone_customers
    }
    positive = [
        (customer_id, result)
        for customer_id, result in reverse.items()
        if result.feasible_reverse_found and result.g_z > epsilon_z
    ]
    tier_one = [
        item for item in positive if item[1].g_hpt >= -epsilon_h and item[1].g_tail >= -epsilon_tail
    ]
    tier_two = [item for item in positive if item[1].g_hpt >= -epsilon_h]
    selected_pool = tier_one or tier_two or positive
    selected_pool.sort(
        key=lambda item: (
            -item[1].g_tail,
            -item[1].g_hpt,
            -item[1].g_z,
            item[1].added_truck_route_duration,
            item[0],
        )
    )
    selected = tuple(customer_id for customer_id, _ in selected_pool[:rollback_size])
    reconstruction = priority_aware_truck_insertion(instance, source, set(selected)) if selected else None
    accepted = False
    reconstructed = None
    rejection_reason = ""
    if not selected:
        rejection_reason = "no_positive_reverse_reassignment"
    elif reconstruction is None or not reconstruction.feasible:
        rejection_reason = "reconstruction_infeasible"
    else:
        reconstructed = reconstruction.reconstructed_solution
        accepted = accept_reconstruction(
            source,
            reconstructed,
            source,
            instance=instance,
            epsilon_z=epsilon_z,
            epsilon_h=epsilon_h,
            epsilon_tail=epsilon_tail,
        )
        if not accepted:
            rejection_reason = "priority_guard_or_objective_guard"
    after = compute_priority_metrics(instance, reconstructed) if reconstructed is not None else before
    selected_damage = any(
        reverse[customer_id].g_hpt < -epsilon_h or reverse[customer_id].g_tail < -epsilon_tail
        for customer_id in selected
    )
    if selected_damage and accepted:
        rejection_reason = "selected_individual_risk_offset_by_aggregate_guard"
    log = {
        "instance_id": str(instance.name),
        "seed": instance.metadata.get("seed", ""),
        "size": int(instance.num_customers),
        "reconstruction_attempt": int(reconstruction_attempt),
        "trigger_reason": trigger_reason,
        "num_drone_customers_before": len(drone_customers),
        "num_candidates_evaluated": len(reverse),
        "num_customers_returned_to_truck": len(selected) if accepted else 0,
        "customer_ids_returned": ",".join(str(customer_id) for customer_id in selected),
        "Z_before": float(source.objective),
        "Z_after": float(reconstructed.objective) if reconstructed is not None else float(source.objective),
        "HPT_before": float(before["high_priority_tardiness"]),
        "HPT_after": float(after["high_priority_tardiness"]),
        "MaxHPT_before": float(before["max_high_priority_tardiness"]),
        "MaxHPT_after": float(after["max_high_priority_tardiness"]),
        "P90HPT_before": float(before["p90_high_priority_tardiness"]),
        "P90HPT_after": float(after["p90_high_priority_tardiness"]),
        "drone_share_before": len(drone_customers) / max(1, instance.num_customers),
        "drone_share_after": len(reconstructed.served_by_drone()) / max(1, instance.num_customers)
        if reconstructed is not None
        else len(drone_customers) / max(1, instance.num_customers),
        "active_drones_before": sum(bool(set(route) - {0}) for route in source.drone_routes.values()),
        "active_drones_after": sum(bool(set(route) - {0}) for route in reconstructed.drone_routes.values())
        if reconstructed is not None
        else sum(bool(set(route) - {0}) for route in source.drone_routes.values()),
        "accepted": bool(accepted),
        "rejection_reason": rejection_reason,
        "cpu_reconstruction": perf_counter() - started,
    }
    return RollbackAttempt(reconstructed if accepted else None, accepted, len(reverse), selected, log)


def run_nils_r(
    instance: InstanceData,
    seed: int,
    max_iter: int = 30,
    max_no_improve: int = 5,
    time_limit: int = 600,
    reconstruction_max_attempts: int = 1,
    reconstruction_stagnation: int | None = None,
    reconstruction_size_cap: int = 3,
    epsilon_z: float = 1e-6,
    epsilon_h: float = 1e-6,
    epsilon_tail: float = 1e-6,
    **nils_options: Any,
) -> SolutionData:
    """Run NILS with bounded priority-risk rollback and truck reconstruction."""
    from .nils import run_nils

    logs: list[dict[str, Any]] = []
    move_logs: list[dict[str, Any]] = []
    solution = run_nils(
        instance,
        seed=seed,
        max_iter=max_iter,
        max_no_improve=max_no_improve,
        time_limit=time_limit,
        risk_rollback=True,
        risk_rollback_reconstruction_max=reconstruction_max_attempts,
        risk_rollback_stagnation=reconstruction_stagnation,
        risk_rollback_size_cap=reconstruction_size_cap,
        risk_rollback_epsilon_z=epsilon_z,
        risk_rollback_epsilon_h=epsilon_h,
        risk_rollback_epsilon_tail=epsilon_tail,
        risk_rollback_logs=logs,
        risk_rollback_move_logs=move_logs,
        **nils_options,
    )
    solution.status = "nils_r_complete"
    solution.__dict__["nils_r_reconstruction_logs"] = logs
    solution.__dict__["nils_r_drone_decision_logs"] = move_logs
    return solution
