"""Feasibility-aware Adaptive Large Neighborhood Search benchmark.

This module implements an independent ALNS competitor for the computational
study. It intentionally does not call the NILS driver. The algorithm shares the
project's evaluator, feasibility checker, and route-state synchronization so
that comparisons remain mathematically consistent.
"""
from __future__ import annotations

import math
import random
import time
from copy import deepcopy
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from ..data_models import InstanceData, SolutionData
from ..distance_utils import euclidean_distance, manhattan_distance, travel_time_from_distance
from ..feasibility import is_feasible
from .construction import build_initial_solution
from .local_search import apply_two_opt_to_solution
from .nils import _eligible_customer_ids, _sync_route_assignments, evaluate_solution


def _priority_rank(priority: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(str(priority), 3)


def _normal_route(route: Sequence[int]) -> List[int]:
    out = [int(v) for v in route] if route else [0, 0]
    if not out:
        out = [0, 0]
    if out[0] != 0:
        out = [0] + out
    if out[-1] != 0:
        out = out + [0]
    if len(out) == 1:
        out = [0, 0]
    return out


def _route_positions(route: Sequence[int], customer: int, coords: Dict[int, Tuple[float, float]], limit: int) -> List[int]:
    """Return a small set of promising insertion positions before route[pos]."""
    rr = _normal_route(route)
    positions = {1, max(1, len(rr) - 1)}
    if len(rr) <= 3 or limit <= 0:
        return sorted(p for p in positions if 1 <= p <= len(rr) - 1)

    cxy = coords[customer]
    scored = []
    for pos in range(1, len(rr)):
        prev_node = rr[pos - 1]
        next_node = rr[pos]
        delta = (
            euclidean_distance(coords[prev_node], cxy)
            + euclidean_distance(cxy, coords[next_node])
            - euclidean_distance(coords[prev_node], coords[next_node])
        )
        scored.append((delta, pos))
    for _, pos in sorted(scored)[: max(1, limit)]:
        positions.add(int(pos))
    return sorted(p for p in positions if 1 <= p <= len(rr) - 1)


def _remove_customer(solution: SolutionData, customer: int) -> SolutionData:
    out = deepcopy(solution)
    for t, route in list(out.truck_routes.items()):
        rr = [node for node in _normal_route(route) if node != customer]
        out.truck_routes[t] = _normal_route(rr)
    for d, route in list(out.drone_routes.items()):
        rr = [node for node in _normal_route(route) if node != customer]
        out.drone_routes[d] = _normal_route(rr)
    return out


def _set_loaded_flags(instance: InstanceData, solution: SolutionData) -> None:
    solution.y_loaded = {}
    for d in instance.drone_ids():
        for c in instance.customer_ids:
            solution.y_loaded[(d, c)] = 1 if solution.u_drone.get((d, c), 0) == 1 else 0


def _refresh(instance: InstanceData, solution: SolutionData) -> SolutionData:
    for t in instance.truck_ids():
        solution.truck_routes[t] = _normal_route(solution.truck_routes.get(t, [0, 0]))
    for d in instance.drone_ids():
        solution.drone_routes[d] = _normal_route(solution.drone_routes.get(d, [0, 0]))
    solution.z1 = {}
    solution.z2 = {}
    _sync_route_assignments(solution, instance)
    _set_loaded_flags(instance, solution)
    solution.objective, solution.components = evaluate_solution(instance, solution)
    return solution


def _drone_energy_ratio(instance: InstanceData, drone_id: int, route: Sequence[int]) -> float:
    route = _normal_route(route)
    coords = instance.coordinate_map()
    drone = instance.drones[drone_id - 1]
    consumed = 0.0
    for idx in range(len(route) - 1):
        frm = route[idx]
        to = route[idx + 1]
        dist = euclidean_distance(coords[frm], coords[to])
        arc_time = travel_time_from_distance(dist, drone.speed_kmph)
        loaded = to != 0
        burn = drone.energy_per_min_when_loaded if loaded else drone.energy_per_min_when_empty
        if burn <= 0:
            burn = 20.0 if loaded else 15.0
        consumed += burn * arc_time
    return consumed / max(drone.max_battery_wh, 1e-9)


def _battery_slack_ok(instance: InstanceData, drone_id: int, route: Sequence[int], beta: float) -> bool:
    return _drone_energy_ratio(instance, drone_id, route) <= 1.0 - max(0.0, beta) + 1e-9


def _insert_truck(
    instance: InstanceData,
    solution: SolutionData,
    customer: int,
    truck_id: int,
    position: int,
) -> SolutionData | None:
    candidate = _remove_customer(solution, customer)
    route = _normal_route(candidate.truck_routes.get(truck_id, [0, 0]))
    pos = max(1, min(int(position), len(route) - 1))
    route.insert(pos, int(customer))
    candidate.truck_routes[truck_id] = route
    candidate = _refresh(instance, candidate)
    return candidate if is_feasible(instance, candidate) else None


def _insert_drone(
    instance: InstanceData,
    solution: SolutionData,
    customer: int,
    drone_id: int,
    position: int,
    beta: float,
) -> SolutionData | None:
    if customer not in _eligible_customer_ids(instance):
        return None
    candidate = _remove_customer(solution, customer)
    route = _normal_route(candidate.drone_routes.get(drone_id, [0, 0]))
    pos = max(1, min(int(position), len(route) - 1))
    route.insert(pos, int(customer))
    if not _battery_slack_ok(instance, drone_id, route, beta):
        return None
    candidate.drone_routes[drone_id] = route
    candidate = _refresh(instance, candidate)
    return candidate if is_feasible(instance, candidate) else None


def _candidate_insertions(
    instance: InstanceData,
    solution: SolutionData,
    customer: int,
    beta: float,
    max_positions: int,
    deadline: float | None = None,
    max_candidates: int | None = None,
) -> List[SolutionData]:
    coords = instance.coordinate_map()
    candidates: List[SolutionData] = []
    for truck_id in instance.truck_ids():
        if deadline is not None and time.time() >= deadline:
            return candidates
        route = solution.truck_routes.get(truck_id, [0, 0])
        for pos in _route_positions(route, customer, coords, max_positions):
            if deadline is not None and time.time() >= deadline:
                return candidates
            cand = _insert_truck(instance, solution, customer, truck_id, pos)
            if cand is not None:
                candidates.append(cand)
                if max_candidates is not None and len(candidates) >= max_candidates:
                    return candidates
    for drone_id in instance.drone_ids():
        if deadline is not None and time.time() >= deadline:
            return candidates
        route = solution.drone_routes.get(drone_id, [0, 0])
        for pos in _route_positions(route, customer, coords, max_positions):
            if deadline is not None and time.time() >= deadline:
                return candidates
            cand = _insert_drone(instance, solution, customer, drone_id, pos, beta)
            if cand is not None:
                candidates.append(cand)
                if max_candidates is not None and len(candidates) >= max_candidates:
                    return candidates
    return candidates


def _destroy_random(solution: SolutionData, customers: List[int], q: int, rng: random.Random) -> List[int]:
    if not customers:
        return []
    return rng.sample(customers, k=min(q, len(customers)))


def _destroy_drone(solution: SolutionData, customers: List[int], q: int, rng: random.Random) -> List[int]:
    routes = [
        [node for node in route if node != 0]
        for route in solution.drone_routes.values()
        if any(node != 0 for node in route)
    ]
    if not routes:
        return _destroy_random(solution, customers, q, rng)
    route = max(routes, key=len)
    if len(route) <= q:
        return list(route)
    return rng.sample(route, k=q)


def _destroy_related(instance: InstanceData, solution: SolutionData, customers: List[int], q: int, rng: random.Random) -> List[int]:
    if not customers:
        return []
    coords = instance.coordinate_map()
    center = rng.choice(customers)
    ranked = sorted(customers, key=lambda c: euclidean_distance(coords[center], coords[c]))
    return ranked[: min(q, len(ranked))]


def _customer_arc_contribution(instance: InstanceData, solution: SolutionData, customer: int) -> float:
    coords = instance.coordinate_map()
    best = 0.0
    for route in solution.truck_routes.values():
        rr = _normal_route(route)
        if customer in rr:
            idx = rr.index(customer)
            if 0 < idx < len(rr) - 1:
                best = max(
                    best,
                    manhattan_distance(coords[rr[idx - 1]], coords[customer])
                    + manhattan_distance(coords[customer], coords[rr[idx + 1]])
                    - manhattan_distance(coords[rr[idx - 1]], coords[rr[idx + 1]]),
                )
    for route in solution.drone_routes.values():
        rr = _normal_route(route)
        if customer in rr:
            idx = rr.index(customer)
            if 0 < idx < len(rr) - 1:
                best = max(
                    best,
                    euclidean_distance(coords[rr[idx - 1]], coords[customer])
                    + euclidean_distance(coords[customer], coords[rr[idx + 1]])
                    - euclidean_distance(coords[rr[idx - 1]], coords[rr[idx + 1]]),
                )
    return float(best)


def _destroy_worst(instance: InstanceData, solution: SolutionData, customers: List[int], q: int) -> List[int]:
    ranked = sorted(customers, key=lambda c: _customer_arc_contribution(instance, solution, c), reverse=True)
    return ranked[: min(q, len(ranked))]


def _destroy_priority(instance: InstanceData, solution: SolutionData, customers: List[int], q: int) -> List[int]:
    cmap = instance.customer_map()
    ranked = sorted(
        customers,
        key=lambda c: (
            _priority_rank(cmap[c].priority),
            -float(solution.tardiness.get(c, 0.0)),
            -_customer_arc_contribution(instance, solution, c),
        ),
    )
    return ranked[: min(q, len(ranked))]


def _destroy(solution: SolutionData, instance: InstanceData, operator: str, q: int, rng: random.Random) -> Tuple[SolutionData, List[int]]:
    customers = sorted(set(solution.served_by_truck()) | set(solution.served_by_drone()))
    if operator == "drone_route":
        removed = _destroy_drone(solution, customers, q, rng)
    elif operator == "related":
        removed = _destroy_related(instance, solution, customers, q, rng)
    elif operator == "worst":
        removed = _destroy_worst(instance, solution, customers, q)
    elif operator == "priority":
        removed = _destroy_priority(instance, solution, customers, q)
    else:
        removed = _destroy_random(solution, customers, q, rng)
    partial = deepcopy(solution)
    for customer in removed:
        partial = _remove_customer(partial, customer)
    partial = _refresh(instance, partial)
    return partial, removed


def _best_insertion(
    instance: InstanceData,
    partial: SolutionData,
    customer: int,
    beta: float,
    max_positions: int,
    deadline: float | None = None,
    max_candidates: int | None = None,
) -> Tuple[SolutionData | None, float, float]:
    candidates = _candidate_insertions(instance, partial, customer, beta, max_positions, deadline, max_candidates)
    if not candidates:
        return None, math.inf, math.inf
    values = sorted(float(c.objective) for c in candidates)
    best_value = values[0]
    second = values[1] if len(values) > 1 else best_value
    best = min(candidates, key=lambda c: c.objective)
    return best, best_value, second


def _repair(
    instance: InstanceData,
    partial: SolutionData,
    removed: Iterable[int],
    operator: str,
    beta: float,
    max_positions: int,
    deadline: float | None = None,
    max_candidates_per_customer: int | None = None,
) -> SolutionData | None:
    unassigned = list(dict.fromkeys(int(c) for c in removed))
    current = deepcopy(partial)
    cmap = instance.customer_map()

    if operator == "priority":
        unassigned.sort(key=lambda c: (_priority_rank(cmap[c].priority), c))

    while unassigned:
        if deadline is not None and time.time() >= deadline:
            return None
        scored = []
        for customer in unassigned:
            if deadline is not None and time.time() >= deadline:
                break
            cand, best, second = _best_insertion(
                instance,
                current,
                customer,
                beta,
                max_positions,
                deadline,
                max_candidates_per_customer,
            )
            if cand is None:
                continue
            regret = second - best
            if operator == "regret2":
                score = (-regret, best)
            elif operator == "priority":
                score = (_priority_rank(cmap[customer].priority), best)
            else:
                score = (best,)
            scored.append((score, customer, cand))
        if not scored:
            return None
        _, chosen, candidate = min(scored, key=lambda x: x[0])
        current = candidate
        unassigned.remove(chosen)
    return current


def _choose_weighted(items: List[str], weights: Dict[str, float], rng: random.Random) -> str:
    total = sum(max(0.01, weights.get(item, 1.0)) for item in items)
    draw = rng.random() * total
    acc = 0.0
    for item in items:
        acc += max(0.01, weights.get(item, 1.0))
        if draw <= acc:
            return item
    return items[-1]


def _try_initial_drone_insertions(
    instance: InstanceData,
    solution: SolutionData,
    beta: float,
    max_positions: int,
    deadline: float | None = None,
) -> SolutionData:
    """Build a nontrivial hybrid starting point without invoking NILS."""
    current = deepcopy(solution)
    coords = instance.coordinate_map()
    customers = sorted(
        _eligible_customer_ids(instance),
        key=lambda c: (euclidean_distance(coords[0], coords[c]), c),
    )
    for customer in customers:
        if deadline is not None and time.time() >= deadline:
            break
        if customer not in current.served_by_truck():
            continue
        candidates = []
        for drone_id in instance.drone_ids():
            if deadline is not None and time.time() >= deadline:
                break
            route = current.drone_routes.get(drone_id, [0, 0])
            for pos in _route_positions(route, customer, coords, max_positions):
                if deadline is not None and time.time() >= deadline:
                    break
                cand = _insert_drone(instance, current, customer, drone_id, pos, beta)
                if cand is not None:
                    candidates.append(cand)
        if not candidates:
            continue
        best = min(candidates, key=lambda c: c.objective)
        if best.objective <= current.objective + 1e-9:
            current = best
    return current


def _empty_like(instance: InstanceData, solution: SolutionData) -> SolutionData:
    """Return an empty route shell with the same metadata as a solution."""
    out = deepcopy(solution)
    out.truck_routes = {t: [0, 0] for t in instance.truck_ids()}
    out.drone_routes = {d: [0, 0] for d in instance.drone_ids()}
    out.x_truck = {}
    out.x_drone = {}
    out.u_truck = {}
    out.u_drone = {}
    out.z1 = {}
    out.z2 = {}
    out.y_loaded = {}
    out.a_truck = {}
    out.l_truck = {}
    out.a_drone = {}
    out.l_drone = {}
    out.waiting_truck = {}
    out.waiting_drone = {}
    out.tardiness = {}
    return _refresh(instance, out)


def _constructive_destroy_repair_start(
    instance: InstanceData,
    seed_solution: SolutionData,
    beta: float,
    max_positions: int,
    deadline: float | None = None,
) -> SolutionData | None:
    """Build a full ALNS-style repair solution from an empty route shell."""
    empty = _empty_like(instance, seed_solution)
    customers = sorted(
        instance.customer_ids,
        key=lambda c: (_priority_rank(instance.customer_map()[c].priority), c),
    )
    best = None
    for repair_op in ("priority", "regret2", "greedy"):
        if deadline is not None and time.time() >= deadline:
            break
        candidate = _repair(instance, empty, customers, repair_op, beta, max_positions, deadline, max_candidates_per_customer=12)
        if candidate is None:
            continue
        candidate = apply_two_opt_to_solution(candidate, instance)
        candidate = _refresh(instance, candidate)
        if is_feasible(instance, candidate) and (best is None or candidate.objective < best.objective):
            best = candidate
    return best


def run_alns(
    instance: InstanceData,
    *,
    seed: int = 2026,
    max_iter: int = 120,
    time_limit: float = 30.0,
    battery_slack_ratio: float | None = None,
    max_positions_per_route: int = 6,
    min_destroy: int = 2,
    max_destroy: int = 8,
    initial_temperature: float = 0.08,
    cooling: float = 0.985,
) -> SolutionData:
    """Run feasibility-aware ALNS and return the best feasible solution found."""
    start = time.time()
    rng = random.Random(int(seed))
    beta = float(instance.metadata.get("battery_slack_ratio", 0.10) if battery_slack_ratio is None else battery_slack_ratio)
    deadline = start + float(time_limit) if time_limit and float(time_limit) > 0 else None

    current = build_initial_solution(instance, int(seed))
    current = _refresh(instance, current)
    if not is_feasible(instance, current):
        current = build_initial_solution(instance, int(seed) + 17, initializer="greedy")
        current = _refresh(instance, current)
    reconstructed = _constructive_destroy_repair_start(instance, current, beta, max_positions_per_route, deadline)
    if reconstructed is not None and reconstructed.objective < current.objective:
        current = reconstructed
    current = _try_initial_drone_insertions(instance, current, beta, max_positions_per_route, deadline)
    current = apply_two_opt_to_solution(current, instance)
    current = _refresh(instance, current)

    best = deepcopy(current)
    destroy_ops = ["random", "worst", "related", "priority", "drone_route"]
    repair_ops = ["greedy", "regret2", "priority"]
    destroy_weights = {op: 1.0 for op in destroy_ops}
    repair_weights = {op: 1.0 for op in repair_ops}

    accepted = 0
    improving = 0
    rejected_infeasible = 0
    evaluated = 0
    temperature = max(1e-6, abs(current.objective) * initial_temperature)

    for it in range(1, max(1, int(max_iter)) + 1):
        if deadline is not None and time.time() >= deadline:
            break
        q_hi = max(min_destroy, min(max_destroy, max(3, int(math.ceil(0.25 * instance.num_customers)))))
        q = rng.randint(max(1, min_destroy), max(1, q_hi))
        d_op = _choose_weighted(destroy_ops, destroy_weights, rng)
        r_op = _choose_weighted(repair_ops, repair_weights, rng)

        partial, removed = _destroy(current, instance, d_op, q, rng)
        candidate = _repair(
            instance,
            partial,
            removed,
            r_op,
            beta,
            max_positions_per_route,
            deadline,
            max_candidates_per_customer=12,
        )
        evaluated += 1
        if candidate is None or not is_feasible(instance, candidate):
            rejected_infeasible += 1
            temperature *= cooling
            continue

        candidate = apply_two_opt_to_solution(candidate, instance)
        candidate = _refresh(instance, candidate)
        if not is_feasible(instance, candidate):
            rejected_infeasible += 1
            temperature *= cooling
            continue

        delta = candidate.objective - current.objective
        accept = delta <= 0.0
        if not accept and temperature > 1e-9:
            accept = rng.random() < math.exp(-delta / temperature)
        if accept:
            current = candidate
            accepted += 1
            destroy_weights[d_op] += 0.2
            repair_weights[r_op] += 0.2
            if candidate.objective < best.objective - 1e-9:
                best = deepcopy(candidate)
                improving += 1
                destroy_weights[d_op] += 4.0
                repair_weights[r_op] += 4.0
            elif delta < -1e-9:
                destroy_weights[d_op] += 1.0
                repair_weights[r_op] += 1.0
        temperature *= cooling

    best = _refresh(instance, best)
    best.status = "alns_complete" if is_feasible(instance, best) else "alns_infeasible"
    best.run_time_seconds = float(time.time() - start)
    best.components = dict(best.components)
    best.components["iterations"] = float(it if "it" in locals() else 0)
    best.components["candidate_reassignments_evaluated"] = float(evaluated)
    best.components["improving_moves_accepted"] = float(improving)
    best.components["alns_accepted_moves"] = float(accepted)
    best.components["alns_infeasible_rejections"] = float(rejected_infeasible)
    best.components["battery_slack_ratio"] = float(beta)
    best.components["alns_destroy_weight_sum"] = float(sum(destroy_weights.values()))
    best.components["alns_repair_weight_sum"] = float(sum(repair_weights.values()))
    return best
