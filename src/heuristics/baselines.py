"""Baseline methods for ablation and comparison runs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ..data_models import InstanceData, SolutionData
from ..distance_utils import euclidean_distance
from ..feasibility import is_feasible
from ..parameters import SearchConfig
from .construction import build_initial_solution
from .nils import (
    _repair_infeasible_solution,
    _eligible_customer_ids,
    default_drone_to_truck,
    evaluate_solution,
    move_to_drone,
    run_nils,
)
from .alns import run_alns


@dataclass
class BaselineResult:
    name: str
    solution: SolutionData


def _clone_solution(solution: SolutionData) -> SolutionData:
    return SolutionData(**{k: v for k, v in solution.__dict__.items()})


def truck_only_baseline(instance: InstanceData, config: SearchConfig) -> BaselineResult:
    sol = build_initial_solution(instance, config.seed)
    obj, comp = evaluate_solution(instance, sol)
    sol.objective = obj
    sol.components = comp
    sol.status = "baseline_truck_only"
    return BaselineResult(name="truck_only", solution=sol)


def paired_baseline(instance: InstanceData, config: SearchConfig) -> BaselineResult:
    drone_to_truck = default_drone_to_truck(instance)
    obj = run_nils(
        instance,
        seed=config.seed + 9,
        max_iter=max(1, config.heuristics.max_outer_iter),
        max_no_improve=max(1, config.heuristics.max_no_improve),
        time_limit=max(10, min(120, config.heuristics.time_limit_seconds)),
        drone_to_truck=drone_to_truck,
        use_paired_initialization=True,
        max_initial_pairings=2,
    )
    obj.status = "baseline_paired"
    return BaselineResult(name="paired_baseline", solution=obj)


def no_priority_baseline(instance: InstanceData, config: SearchConfig) -> BaselineResult:
    # run with flattened penalties temporarily
    original = dict(instance.priority_penalties)
    try:
        for key in instance.priority_penalties:
            instance.priority_penalties[key] = 1.0
        sol = run_nils(instance, config.seed + 1, max_iter=max(1, config.heuristics.max_outer_iter // 2))
        sol.status = "baseline_no_priority"
        return BaselineResult(name="no_priority", solution=sol)
    finally:
        instance.priority_penalties.update(original)


def no_unpairing_baseline(instance: InstanceData, config: SearchConfig) -> BaselineResult:
    obj = run_nils(
        instance,
        seed=config.seed + 2,
        max_iter=max(1, config.heuristics.max_outer_iter // 2),
        max_no_improve=max(1, config.heuristics.max_no_improve),
        time_limit=max(10, min(120, config.heuristics.time_limit_seconds)),
        no_unpairing_mode=True,
    )
    obj.status = "baseline_no_unpairing"
    return BaselineResult(name="no_unpairing", solution=obj)


def _best_drone_insertion_position(
    instance: InstanceData,
    route: List[int],
    customer: int,
) -> Tuple[int, float]:
    coords = instance.coordinate_map()
    cxy = coords[customer]
    best_pos = len(route) - 1
    best_delta = float("inf")
    for pos in range(1, len(route)):
        prev_node = route[pos - 1]
        next_node = route[pos]
        prev_xy = coords[prev_node]
        next_xy = coords[next_node]
        delta = (
            euclidean_distance(prev_xy, cxy)
            + euclidean_distance(cxy, next_xy)
            - euclidean_distance(prev_xy, next_xy)
        )
        if delta < best_delta:
            best_delta = delta
            best_pos = pos
    return best_pos, best_delta


def simple_drone_assignment_baseline(instance: InstanceData, config: SearchConfig) -> BaselineResult:
    """Nearest-insertion drone assignment without nested objective re-evaluation."""
    current = truck_only_baseline(instance, config).solution
    if not is_feasible(instance, current):
        current = build_initial_solution(instance, config.seed + 31)
        if not is_feasible(instance, current):
            repaired = _repair_infeasible_solution(instance, current, max_steps=max(1, int(instance.num_customers)))
            if repaired is not None:
                current = repaired
    eligible = _eligible_customer_ids(instance)
    coords = instance.coordinate_map()
    customers = [
        c
        for c in instance.customer_ids
        if c in eligible and any(current.u_truck.get((t, c), 0) == 1 for t in instance.truck_ids())
    ]
    customers.sort(key=lambda cid: (euclidean_distance(coords[0], coords[cid]), cid))

    for customer in customers:
        best_choice = None
        best_delta = float("inf")
        for drone in instance.drone_ids():
            route = current.drone_routes.get(drone, [0, 0])
            if not route:
                route = [0, 0]
            if route[0] != 0:
                route = [0] + route
            if route[-1] != 0:
                route = route + [0]
            pos, delta = _best_drone_insertion_position(instance, route, customer)
            if delta < best_delta:
                best_delta = delta
                best_choice = (drone, pos)

        if best_choice is None:
            continue

        drone, insertion_pos = best_choice
        candidate = move_to_drone(
            instance,
            current,
            customer,
            drone,
            insertion_index=insertion_pos,
            enforce_battery=True,
            battery_slack_ratio=0.0,
        )
        if candidate is None:
            continue
        if not is_feasible(instance, candidate):
            continue
        current = candidate

    current.objective, current.components = evaluate_solution(instance, current)
    if not is_feasible(instance, current):
        repaired = _repair_infeasible_solution(instance, current, max_steps=max(1, int(instance.num_customers)))
        if repaired is not None:
            current = repaired
            current.objective, current.components = evaluate_solution(instance, current)
    current.status = "baseline_simple_drone"
    return BaselineResult(name="simple_drone", solution=current)


def random_feasible_reassignment_baseline(instance: InstanceData, config: SearchConfig) -> BaselineResult:
    """Weak benchmark: random feasible truck->drone reassignments."""
    import random

    rnd = random.Random(config.seed + 47)
    current = build_initial_solution(instance, config.seed + 47)
    current.objective, current.components = evaluate_solution(instance, current)
    eligible = list(_eligible_customer_ids(instance))
    rnd.shuffle(eligible)

    moves = max(1, min(len(eligible), 2 * max(1, instance.num_drones)))
    applied = 0
    for customer in eligible:
        if applied >= moves:
            break
        if not any(current.u_truck.get((t, customer), 0) == 1 for t in instance.truck_ids()):
            continue
        drone_order = instance.drone_ids()
        rnd.shuffle(drone_order)
        for drone in drone_order:
            candidate = move_to_drone(instance, current, customer, drone)
            if candidate is None:
                continue
            candidate.objective, candidate.components = evaluate_solution(instance, candidate)
            current = candidate
            applied += 1
            break

    current.objective, current.components = evaluate_solution(instance, current)
    current.status = "baseline_random_feasible"
    return BaselineResult(name="random_feasible", solution=current)


def nils_no_local_search_baseline(instance: InstanceData, config: SearchConfig) -> BaselineResult:
    sol = run_nils(
        instance,
        seed=config.seed + 73,
        max_iter=max(1, config.heuristics.max_outer_iter),
        max_no_improve=max(1, config.heuristics.max_no_improve),
        time_limit=max(10, min(180, config.heuristics.time_limit_seconds)),
        enable_local_search=False,
    )
    sol.status = "ablation_no_local_search"
    return BaselineResult(name="nils_no_local_search", solution=sol)


def nils_no_perturbation_baseline(instance: InstanceData, config: SearchConfig) -> BaselineResult:
    sol = run_nils(
        instance,
        seed=config.seed + 79,
        max_iter=max(1, config.heuristics.max_outer_iter),
        max_no_improve=max(1, config.heuristics.max_no_improve),
        time_limit=max(10, min(180, config.heuristics.time_limit_seconds)),
        enable_perturbation=False,
    )
    sol.status = "ablation_no_perturbation"
    return BaselineResult(name="nils_no_perturbation", solution=sol)


def nils_no_battery_screening_baseline(instance: InstanceData, config: SearchConfig) -> BaselineResult:
    sol = run_nils(
        instance,
        seed=config.seed + 83,
        max_iter=max(1, config.heuristics.max_outer_iter),
        max_no_improve=max(1, config.heuristics.max_no_improve),
        time_limit=max(10, min(180, config.heuristics.time_limit_seconds)),
        battery_aware_screening=False,
    )
    sol.status = "ablation_no_battery_screening"
    return BaselineResult(name="nils_no_battery_screening", solution=sol)


def nils_soft_drone_target_baseline(instance: InstanceData, config: SearchConfig) -> BaselineResult:
    """Policy variant: soft drone-utilization target via augmented selection over multi-start NILS."""
    exp_cfg = dict(config.experiment or {})
    target_share = float(exp_cfg.get("soft_drone_target_share", 0.20))
    penalty_weight = float(exp_cfg.get("soft_drone_target_penalty", 5000.0))
    raw_offsets = exp_cfg.get("soft_drone_target_seed_offsets", [101, 137])
    if not isinstance(raw_offsets, list) or not raw_offsets:
        raw_offsets = [101, 137]
    seed_offsets = [int(v) for v in raw_offsets]

    best_solution: SolutionData | None = None
    best_augmented = float("inf")
    best_feasible = False
    evaluated = 0

    for off in seed_offsets:
        sol = run_nils(
            instance,
            seed=config.seed + off,
            max_iter=max(1, config.heuristics.max_outer_iter),
            max_no_improve=max(1, config.heuristics.max_no_improve),
            time_limit=max(10, min(180, config.heuristics.time_limit_seconds)),
        )
        evaluated += 1
        drone_share = float(len(sol.served_by_drone()) / max(1, instance.num_customers))
        shortfall = max(0.0, target_share - drone_share)
        augmented = float(sol.objective) + penalty_weight * shortfall
        feasible = bool(is_feasible(instance, sol))

        choose = False
        if best_solution is None:
            choose = True
        elif feasible and not best_feasible:
            choose = True
        elif feasible == best_feasible and augmented < best_augmented:
            choose = True
        if choose:
            best_solution = sol
            best_augmented = augmented
            best_feasible = feasible

    if best_solution is None:
        best_solution = run_nils(
            instance,
            seed=config.seed + 101,
            max_iter=max(1, config.heuristics.max_outer_iter),
            max_no_improve=max(1, config.heuristics.max_no_improve),
            time_limit=max(10, min(180, config.heuristics.time_limit_seconds)),
        )
        best_augmented = float(best_solution.objective)
        best_feasible = bool(is_feasible(instance, best_solution))
        evaluated = max(1, evaluated)

    achieved_share = float(len(best_solution.served_by_drone()) / max(1, instance.num_customers))
    shortfall = max(0.0, target_share - achieved_share)
    comps = dict(best_solution.components or {})
    comps["soft_target_share"] = float(target_share)
    comps["soft_target_penalty_weight"] = float(penalty_weight)
    comps["soft_target_achieved_share"] = float(achieved_share)
    comps["soft_target_shortfall"] = float(shortfall)
    comps["soft_target_augmented_objective"] = float(float(best_solution.objective) + penalty_weight * shortfall)
    comps["soft_target_feasible_selected"] = 1.0 if best_feasible else 0.0
    comps["soft_target_candidates_evaluated"] = float(evaluated)
    best_solution.components = comps
    best_solution.status = "policy_soft_drone_target"
    return BaselineResult(name="nils_soft_drone_target", solution=best_solution)


def feasibility_aware_alns_baseline(instance: InstanceData, config: SearchConfig) -> BaselineResult:
    """Independent ALNS benchmark using shared feasibility and objective logic."""
    exp_cfg = dict(config.experiment or {})
    sol = run_alns(
        instance,
        seed=int(config.seed + 211),
        max_iter=int(exp_cfg.get("alns_max_iter", max(40, config.heuristics.max_outer_iter * 4))),
        time_limit=float(exp_cfg.get("alns_time_limit_seconds", min(60, max(10, config.heuristics.time_limit_seconds)))),
        battery_slack_ratio=float(exp_cfg.get("battery_slack_ratio", exp_cfg.get("alns_battery_slack_ratio", 0.10))),
        max_positions_per_route=int(exp_cfg.get("alns_max_positions_per_route", 6)),
        max_destroy=int(exp_cfg.get("alns_max_destroy", 8)),
    )
    sol.status = "benchmark_feasibility_aware_alns"
    return BaselineResult(name="feasibility_aware_alns", solution=sol)


def run_all_baselines(instance: InstanceData, config: SearchConfig) -> List[BaselineResult]:
    return [
        truck_only_baseline(instance, config),
        simple_drone_assignment_baseline(instance, config),
        random_feasible_reassignment_baseline(instance, config),
        paired_baseline(instance, config),
        no_priority_baseline(instance, config),
        no_unpairing_baseline(instance, config),
        nils_no_local_search_baseline(instance, config),
        nils_no_perturbation_baseline(instance, config),
        nils_no_battery_screening_baseline(instance, config),
        nils_soft_drone_target_baseline(instance, config),
        feasibility_aware_alns_baseline(instance, config),
    ]
