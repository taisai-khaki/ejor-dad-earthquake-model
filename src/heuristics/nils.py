"""Nested Iterative Local Search (NILS) implementation used in experiments."""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..data_models import InstanceData, SolutionData
from ..distance_utils import euclidean_distance, manhattan_distance
from ..feasibility import is_feasible
from .acceptance import metropolis
from .construction import build_initial_solution
from .local_search import apply_two_opt_to_solution
from .perturbation import relocate_random_customer


def _travel_time(distance_km: float, speed_kmph: float) -> float:
    if speed_kmph <= 0:
        return math.inf
    return distance_km / speed_kmph * 60.0


def evaluate_solution(instance: InstanceData, solution: SolutionData) -> Tuple[float, Dict[str, float]]:
    coords = {0: (0.0, 0.0)}
    coords.update({c.node_id: (c.x, c.y) for c in instance.customers})

    arrival_truck: Dict[Tuple[int, int], float] = {}
    arrival_drone: Dict[Tuple[int, int], float] = {}
    served_time: Dict[int, float] = {j: 0.0 for j in range(1, instance.num_customers + 1)}

    truck_cost = 0.0
    drone_cost = 0.0
    solution.waiting_truck = {}
    solution.waiting_drone = {}

    for t, route in solution.truck_routes.items():
        truck = instance.trucks[t - 1]
        cur_t = 0.0
        for i in range(len(route) - 1):
            frm = route[i]
            to = route[i + 1]
            if i > 0:
                prev = route[i - 1]
                if prev != 0:
                    cur_t += instance.customer_map()[prev].service_time_min
            dist = manhattan_distance(coords[frm], coords[to])
            cur_t += _travel_time(dist, truck.speed_kmph)
            arrival_truck[(t, to)] = cur_t
            if to != 0:
                served_time[to] = max(served_time[to], cur_t)
                solution.a_truck[(t, to)] = cur_t
            solution.l_truck[(t, to)] = cur_t
            truck_cost += truck.cost_per_km * dist

    for d, route in solution.drone_routes.items():
        drone = instance.drones[d - 1]
        cur_t = 0.0
        for i in range(len(route) - 1):
            frm = route[i]
            to = route[i + 1]
            if i > 0:
                prev = route[i - 1]
                if prev != 0:
                    cur_t += instance.customer_map()[prev].service_time_min
            dist = euclidean_distance(coords[frm], coords[to])
            cur_t += _travel_time(dist, drone.speed_kmph)
            arrival_drone[(d, to)] = cur_t
            if to != 0:
                served_time[to] = max(served_time[to], cur_t)
                solution.a_drone[(d, to)] = cur_t
            solution.l_drone[(d, to)] = cur_t
            drone_cost += drone.cost_per_km * dist

    tardiness_cost = 0.0
    for c in range(1, instance.num_customers + 1):
        delivered_at = served_time[c]
        ci = instance.customer_map()[c]
        penalty = max(0.0, delivered_at - ci.ub)
        solution.tardiness[c] = penalty
        tardiness_cost += penalty * instance.priority_penalties[ci.priority]

    sync_delay_total = 0.0
    for (node, drone_id, truck_id), val in solution.z1.items():
        if val <= 0.5:
            continue
        at = solution.a_truck.get((truck_id, node), 0.0)
        ad = solution.a_drone.get((drone_id, node), 0.0)
        delay = abs(at - ad)
        sync_delay_total += delay
        if at < ad:
            solution.waiting_truck[(truck_id, node)] = delay
        elif ad < at:
            solution.waiting_drone[(drone_id, node)] = delay
    for (node, drone_id, truck_id), val in solution.z2.items():
        if val <= 0.5:
            continue
        at = solution.a_truck.get((truck_id, node), 0.0)
        ad = solution.a_drone.get((drone_id, node), 0.0)
        delay = abs(at - ad)
        sync_delay_total += delay
        if at < ad:
            solution.waiting_truck[(truck_id, node)] = max(solution.waiting_truck.get((truck_id, node), 0.0), delay)
        elif ad < at:
            solution.waiting_drone[(drone_id, node)] = max(solution.waiting_drone.get((drone_id, node), 0.0), delay)

    return truck_cost + drone_cost + tardiness_cost + sync_delay_total, {
        "truck_cost": truck_cost,
        "drone_cost": drone_cost,
        "tardiness_cost": tardiness_cost,
        "waiting_cost": sync_delay_total,
        "waiting_sync_cost": sync_delay_total,
    }


def summarize_drone_usage(solution: SolutionData) -> Dict[str, int]:
    """Return compact diagnostics for whether drones were actually used."""
    drone_served_customers = {
        c for (d, c), v in solution.u_drone.items() if c > 0 and v == 1
    }
    if not drone_served_customers:
        for route in solution.drone_routes.values():
            for c in route:
                if c != 0:
                    drone_served_customers.add(c)

    drone_arcs = 0
    nonempty_drone_routes = 0
    for route in solution.drone_routes.values():
        if any(node != 0 for node in route):
            nonempty_drone_routes += 1
            if len(route) >= 2:
                drone_arcs += len(route) - 1

    reload_events = sum(1 for (_, _, _), v in solution.z1.items() if v > 0.5)
    battery_swaps = sum(1 for (_, _, _), v in solution.z2.items() if v > 0.5)

    return {
        "drone_served_customers": len(drone_served_customers),
        "drone_arcs": int(drone_arcs),
        "reload_events": int(reload_events),
        "battery_swaps": int(battery_swaps),
        "nonempty_drone_routes": int(nonempty_drone_routes),
    }


def default_drone_to_truck(instance: InstanceData) -> Dict[int, int]:
    """Deterministic round-robin default fixed-pair assignment map: drone -> truck."""
    trucks = instance.truck_ids()
    return {d: trucks[(d - 1) % len(trucks)] for d in instance.drone_ids()}


def _normalize_drone_to_truck(
    instance: InstanceData,
    drone_to_truck: Dict[int, int] | None,
) -> Dict[int, int]:
    if not drone_to_truck:
        return {}
    valid_trucks = set(instance.truck_ids())
    normalized: Dict[int, int] = {}
    for drone_id in instance.drone_ids():
        truck_id = int(drone_to_truck.get(drone_id, 0))
        if truck_id not in valid_trucks:
            return {}
        normalized[drone_id] = truck_id
    if set(drone_to_truck.keys()) - set(instance.drone_ids()):
        return {}
    return normalized


def _eligible_customer_ids(instance: InstanceData) -> set[int]:
    """Return customers eligible for drone service under the active scenario metadata."""
    raw = instance.metadata.get("drone_eligible_customers")
    if isinstance(raw, list) and raw:
        valid = {int(v) for v in raw if int(v) in set(instance.customer_ids)}
        if valid:
            return valid
    max_drone_cap = max((d.capacity_kg for d in instance.drones), default=0.0)
    return {c.node_id for c in instance.customers if c.demand_kg <= max_drone_cap}


def _service_mask(solution: SolutionData) -> Dict[int, str]:
    mask: Dict[int, str] = {}
    for (t, c), v in solution.u_truck.items():
        if v == 1:
            mask[c] = f"truck_{t}"
    for (d, c), v in solution.u_drone.items():
        if v == 1:
            mask[c] = f"drone_{d}"
    return mask


def _empty_solution_clone(solution: SolutionData) -> SolutionData:
    return SolutionData(
        instance_name=solution.instance_name,
        status=solution.status,
        objective=solution.objective,
        components=dict(solution.components),
        run_time_seconds=solution.run_time_seconds,
        x_truck=dict(solution.x_truck),
        x_drone=dict(solution.x_drone),
        u_truck=dict(solution.u_truck),
        u_drone=dict(solution.u_drone),
        z1=dict(solution.z1),
        z2=dict(solution.z2),
        y_loaded=dict(solution.y_loaded),
        a_truck=dict(solution.a_truck),
        l_truck=dict(solution.l_truck),
        a_drone=dict(solution.a_drone),
        l_drone=dict(solution.l_drone),
        waiting_truck=dict(solution.waiting_truck),
        waiting_drone=dict(solution.waiting_drone),
        load_truck=dict(solution.load_truck),
        battery_drone=dict(solution.battery_drone),
        tardiness=dict(solution.tardiness),
        sync_events=list(solution.sync_events),
        truck_routes={k: v.copy() for k, v in solution.truck_routes.items()},
        drone_routes={k: v.copy() for k, v in solution.drone_routes.items()},
        seed=solution.seed,
        tags=solution.tags,
    )


def _sync_route_assignments(solution: SolutionData, instance: InstanceData) -> None:
    """Keep arc/assignment views consistent with the current explicit routes."""
    solution.x_truck = {}
    solution.x_drone = {}

    # Reset assignments and rebuild from routes.
    for c in instance.customer_ids:
        for t in instance.truck_ids():
            solution.u_truck[(t, c)] = 0
        for d in instance.drone_ids():
            solution.u_drone[(d, c)] = 0

    for t, route in solution.truck_routes.items():
        rr = list(route)
        if not rr:
            rr = [0, 0]
        if rr[0] != 0:
            rr = [0] + rr
        if rr[-1] != 0:
            rr = rr + [0]
        solution.truck_routes[t] = rr
        for idx in range(len(rr) - 1):
            i = int(rr[idx])
            j = int(rr[idx + 1])
            solution.x_truck[(i, j, t)] = 1
            if j != 0:
                solution.u_truck[(t, j)] = 1

    for d, route in solution.drone_routes.items():
        rr = list(route)
        if not rr:
            rr = [0, 0]
        if rr[0] != 0:
            rr = [0] + rr
        if rr[-1] != 0:
            rr = rr + [0]
        solution.drone_routes[d] = rr
        for idx in range(len(rr) - 1):
            i = int(rr[idx])
            j = int(rr[idx + 1])
            solution.x_drone[(i, j, d)] = 1
            if j != 0:
                solution.u_drone[(d, j)] = 1


def _paired_trip_feasible(
    instance: InstanceData,
    drone: int,
    launch_node: int,
    customer: int,
    recovery_node: int,
    coords: Dict[int, Tuple[float, float]],
    battery_slack_ratio: float = 0.0,
) -> bool:
    drone_obj = instance.drones[drone - 1]
    lc = _travel_time(euclidean_distance(coords[launch_node], coords[customer]), drone_obj.speed_kmph)
    cr = _travel_time(euclidean_distance(coords[customer], coords[recovery_node]), drone_obj.speed_kmph)
    if lc == math.inf or cr == math.inf:
        return False
    burn = drone_obj.energy_per_min_when_loaded if drone_obj.energy_per_min_when_loaded > 0 else 20.0
    energy = (lc + cr) * burn
    if energy > drone_obj.max_battery_wh + 1e-9:
        return False
    residual = drone_obj.max_battery_wh - energy
    return residual >= max(0.0, battery_slack_ratio) * drone_obj.max_battery_wh - 1e-9


def _drone_route_energy(instance: InstanceData, drone: int, route: List[int], coords: Dict[int, Tuple[float, float]]) -> float:
    drone_obj = instance.drones[drone - 1]
    consumed = 0.0
    for idx in range(len(route) - 1):
        frm = route[idx]
        to = route[idx + 1]
        arc_time = _travel_time(euclidean_distance(coords[frm], coords[to]), drone_obj.speed_kmph)
        loaded = to != 0
        burn = drone_obj.energy_per_min_when_loaded if loaded else drone_obj.energy_per_min_when_empty
        if burn <= 0:
            burn = 20.0 if loaded else 15.0
        consumed += burn * arc_time
    return consumed


def _route_duration_minutes(
    instance: InstanceData,
    route: List[int],
    *,
    coords: Dict[int, Tuple[float, float]],
    speed_kmph: float,
    metric: str,
) -> float:
    if not route or len(route) < 2:
        return 0.0
    t_minutes = 0.0
    for idx in range(len(route) - 1):
        frm = route[idx]
        to = route[idx + 1]
        if idx > 0:
            prev = route[idx - 1]
            if prev != 0:
                t_minutes += instance.customer_map()[prev].service_time_min
        if metric == "manhattan":
            dist = manhattan_distance(coords[frm], coords[to])
        else:
            dist = euclidean_distance(coords[frm], coords[to])
        t_minutes += _travel_time(dist, speed_kmph)
    return float(t_minutes)


def _quick_local_feasible(
    instance: InstanceData,
    candidate: SolutionData,
    *,
    moved_customer: int,
    drone: int,
    coords: Dict[int, Tuple[float, float]],
    source_truck: int | None = None,
    launch_node: int | None = None,
    recovery_node: int | None = None,
    battery_slack_ratio: float = 0.0,
) -> bool:
    if moved_customer <= 0 or drone <= 0 or drone > instance.num_drones:
        return False

    # Local assignment consistency for the moved customer.
    served_by_truck = sum(int(candidate.u_truck.get((t, moved_customer), 0) == 1) for t in instance.truck_ids())
    served_by_drone = sum(int(candidate.u_drone.get((d, moved_customer), 0) == 1) for d in instance.drone_ids())
    if served_by_truck + served_by_drone != 1:
        return False
    if candidate.u_drone.get((drone, moved_customer), 0) != 1:
        return False

    if source_truck is not None and source_truck > 0:
        truck_route = candidate.truck_routes.get(source_truck, [0, 0])
        if not truck_route or truck_route[0] != 0 or truck_route[-1] != 0:
            return False
        truck_nodes = [n for n in truck_route if n != 0]
        if len(truck_nodes) != len(set(truck_nodes)):
            return False
        if moved_customer in truck_nodes:
            return False
        if launch_node is not None and launch_node not in truck_route:
            return False
        if recovery_node is not None and recovery_node not in truck_route and recovery_node != 0:
            return False
        if launch_node is not None and recovery_node is not None and recovery_node != 0:
            try:
                if truck_route.index(launch_node) >= truck_route.index(recovery_node):
                    return False
            except ValueError:
                return False
        truck_minutes = _route_duration_minutes(
            instance,
            truck_route,
            coords=coords,
            speed_kmph=instance.trucks[source_truck - 1].speed_kmph,
            metric="manhattan",
        )
        if truck_minutes > instance.constants.max_shift_minutes + 1e-9:
            return False

    drone_route = candidate.drone_routes.get(drone, [0, 0])
    if not drone_route or drone_route[0] != 0 or drone_route[-1] != 0:
        return False
    drone_nodes = [n for n in drone_route if n != 0]
    if len(drone_nodes) != len(set(drone_nodes)):
        return False
    if moved_customer not in drone_nodes:
        return False

    drone_obj = instance.drones[drone - 1]
    drone_minutes = _route_duration_minutes(
        instance,
        drone_route,
        coords=coords,
        speed_kmph=drone_obj.speed_kmph,
        metric="euclidean",
    )
    if drone_minutes > instance.constants.max_shift_minutes + 1e-9:
        return False

    energy = _drone_route_energy(instance, drone, drone_route, coords)
    if energy > drone_obj.max_battery_wh + 1e-9:
        return False
    residual_ratio = (drone_obj.max_battery_wh - energy) / max(drone_obj.max_battery_wh, 1e-9)
    if residual_ratio + 1e-9 < max(0.0, battery_slack_ratio):
        return False

    return True


def _select_best_strict_candidate(
    instance: InstanceData,
    ranked_candidates: List[Tuple[float, SolutionData, Tuple[int, int, int]]],
    *,
    strict_check_cap: int,
    stats: Dict[str, int] | None = None,
) -> SolutionData | None:
    if not ranked_candidates:
        return None
    ranked = sorted(ranked_candidates, key=lambda x: x[0])
    max_checks = len(ranked) if strict_check_cap <= 0 else min(len(ranked), int(strict_check_cap))
    for idx in range(max_checks):
        _, candidate, _ = ranked[idx]
        _sync_route_assignments(candidate, instance)
        if stats is not None:
            stats["strict_feasibility_checks"] += 1
        if is_feasible(instance, candidate):
            candidate.components = dict(candidate.components)
            candidate.components["_strict_feasibility_checked"] = 1.0
            return candidate
        if stats is not None:
            stats["infeasible_candidate_rejections"] += 1
    return None


def move_to_drone(
    instance: InstanceData,
    solution: SolutionData,
    customer: int,
    drone: int,
    *,
    launch_node: int | None = None,
    recovery_node: int | None = None,
    source_truck: int | None = None,
    insertion_index: int | None = None,
    enforce_battery: bool = True,
    battery_slack_ratio: float = 0.0,
) -> SolutionData | None:
    """Generate candidate where one customer is moved from truck to a drone route."""
    service = _service_mask(solution)
    src = None
    if service.get(customer, "").startswith("truck_"):
        src = int(service[customer].split("_")[1])
    if source_truck is not None:
        src = source_truck
    if src is None or src <= 0 or src > instance.num_trucks:
        return None
    if customer <= 0 or drone <= 0 or drone > instance.num_drones:
        return None

    ci = instance.customer_map()[customer]
    if ci.demand_kg > instance.drones[drone - 1].capacity_kg:
        return None

    candidate = _empty_solution_clone(solution)

    if src in candidate.truck_routes and customer in candidate.truck_routes[src]:
        candidate.truck_routes[src] = [j for j in candidate.truck_routes[src] if j != customer]
        if len(candidate.truck_routes[src]) == 1:
            candidate.truck_routes[src] = [0, 0]
        candidate.u_truck[(src, customer)] = 0

    for d in instance.drone_ids():
        candidate.u_drone[(d, customer)] = 1 if d == drone else 0

    route = candidate.drone_routes.get(drone, [0, 0])
    if not route:
        route = [0, 0]
    if route[0] != 0:
        route = [0] + route
    if route[-1] != 0:
        route = route + [0]
    if customer not in route:
        if insertion_index is None:
            route.insert(len(route) - 1, customer)
        else:
            insert_at = max(1, min(int(insertion_index), len(route) - 1))
            route.insert(insert_at, customer)
    if enforce_battery:
        coords = instance.coordinate_map()
        total_energy = _drone_route_energy(instance, drone, route, coords)
        max_battery = instance.drones[drone - 1].max_battery_wh
        if total_energy > max_battery + 1e-9:
            return None
        residual_ratio = (max_battery - total_energy) / max(max_battery, 1e-9)
        if residual_ratio + 1e-9 < max(0.0, battery_slack_ratio):
            return None
    candidate.drone_routes[drone] = route
    candidate.x_drone = {}

    if source_truck is not None and launch_node is not None and recovery_node is not None:
        candidate.z1[(launch_node, drone, source_truck)] = 1
        candidate.z2[(recovery_node, drone, source_truck)] = 1

    return candidate


def _extract_best_displacement(
    instance: InstanceData,
    solution: SolutionData,
    *,
    coords: Dict[int, Tuple[float, float]],
    eligible_customers: set[int],
    battery_aware_screening: bool,
    battery_slack_ratio: float,
    strict_check_cap: int,
    tabu_moves: set[Tuple[int, int, int]] | None = None,
    stats: Dict[str, int] | None = None,
) -> SolutionData | None:
    ranked_candidates: List[Tuple[float, SolutionData, Tuple[int, int, int]]] = []

    for customer in range(1, instance.num_customers + 1):
        if customer not in eligible_customers:
            continue
        truck_served = any(solution.u_truck.get((t, customer), 0) == 1 for t in instance.truck_ids())
        if not truck_served:
            continue
        for d in range(1, instance.num_drones + 1):
            source_truck = next((t for t in instance.truck_ids() if solution.u_truck.get((t, customer), 0) == 1), 0)
            move_key = (customer, d, source_truck)
            if tabu_moves is not None and move_key in tabu_moves:
                continue
            if battery_aware_screening and not _paired_trip_feasible(
                instance,
                d,
                0,
                customer,
                0,
                coords,
                battery_slack_ratio=battery_slack_ratio,
            ):
                if stats is not None:
                    stats["battery_infeasible_rejections"] += 1
                    stats["battery_slack_rejections"] += 1
                continue
            candidate = move_to_drone(
                instance,
                solution,
                customer,
                d,
                enforce_battery=battery_aware_screening,
                battery_slack_ratio=battery_slack_ratio,
            )
            if candidate is None:
                if battery_aware_screening and stats is not None:
                    stats["battery_infeasible_rejections"] += 1
                continue
            if not _quick_local_feasible(
                instance,
                candidate,
                moved_customer=customer,
                drone=d,
                coords=coords,
                source_truck=source_truck if source_truck > 0 else None,
                battery_slack_ratio=battery_slack_ratio,
            ):
                if stats is not None:
                    stats["local_screen_rejections"] += 1
                continue
            if stats is not None:
                stats["candidate_reassignments_evaluated"] += 1
            obj, comp = evaluate_solution(instance, candidate)
            candidate.objective = obj
            candidate.components = comp
            ranked_candidates.append((obj, candidate, move_key))
    return _select_best_strict_candidate(
        instance,
        ranked_candidates,
        strict_check_cap=strict_check_cap,
        stats=stats,
    )


def _extract_best_paired_displacement(
    instance: InstanceData,
    solution: SolutionData,
    drone_to_truck: Dict[int, int],
    coords: Dict[int, Tuple[float, float]],
    *,
    eligible_customers: set[int],
    battery_aware_screening: bool,
    battery_slack_ratio: float,
    strict_check_cap: int,
    tabu_moves: set[Tuple[int, int, int]] | None = None,
    stats: Dict[str, int] | None = None,
) -> SolutionData | None:
    ranked_candidates: List[Tuple[float, SolutionData, Tuple[int, int, int]]] = []

    for drone, truck in drone_to_truck.items():
        route = solution.truck_routes.get(truck, [0, 0])
        if len(route) < 3:
            continue

        for customer_idx in range(1, len(route) - 1):
            customer = route[customer_idx]
            if customer == 0:
                continue
            if customer not in eligible_customers:
                continue
            if solution.u_truck.get((truck, customer), 0) != 1:
                continue
            move_key = (customer, drone, truck)
            if tabu_moves is not None and move_key in tabu_moves:
                continue
            found_feasible_for_key = False
            for launch_idx in range(0, customer_idx):
                launch_node = route[launch_idx]
                for recovery_idx in range(customer_idx + 1, len(route)):
                    recovery_node = route[recovery_idx]
                    if recovery_node == launch_node or recovery_node == customer:
                        continue
                    if battery_aware_screening:
                        if not _paired_trip_feasible(
                            instance,
                            drone,
                            launch_node,
                            customer,
                            recovery_node,
                            coords,
                            battery_slack_ratio=battery_slack_ratio,
                        ):
                            if stats is not None:
                                stats["battery_infeasible_rejections"] += 1
                                stats["battery_slack_rejections"] += 1
                            continue
                    candidate = move_to_drone(
                        instance,
                        solution,
                        customer,
                        drone,
                        launch_node=launch_node,
                        recovery_node=recovery_node,
                        source_truck=truck,
                        enforce_battery=battery_aware_screening,
                        battery_slack_ratio=battery_slack_ratio,
                    )
                    if candidate is None:
                        if battery_aware_screening and stats is not None:
                            stats["battery_infeasible_rejections"] += 1
                        continue
                    if not _quick_local_feasible(
                        instance,
                        candidate,
                        moved_customer=customer,
                        drone=drone,
                        coords=coords,
                        source_truck=truck,
                        launch_node=launch_node,
                        recovery_node=recovery_node,
                        battery_slack_ratio=battery_slack_ratio,
                    ):
                        if stats is not None:
                            stats["local_screen_rejections"] += 1
                        continue
                    found_feasible_for_key = True
                    if stats is not None:
                        stats["candidate_reassignments_evaluated"] += 1
                    obj, comp = evaluate_solution(instance, candidate)
                    candidate.objective = obj
                    candidate.components = comp
                    ranked_candidates.append((obj, candidate, move_key))
            if not found_feasible_for_key and tabu_moves is not None:
                tabu_moves.add(move_key)
    return _select_best_strict_candidate(
        instance,
        ranked_candidates,
        strict_check_cap=strict_check_cap,
        stats=stats,
    )


def _extract_best_no_unpairing_displacement(
    instance: InstanceData,
    solution: SolutionData,
    coords: Dict[int, Tuple[float, float]],
    *,
    eligible_customers: set[int],
    battery_aware_screening: bool,
    battery_slack_ratio: float,
    strict_check_cap: int,
    tabu_moves: set[Tuple[int, int, int]] | None = None,
    stats: Dict[str, int] | None = None,
) -> SolutionData | None:
    """Best displacement with cross-truck recovery disabled.

    A drone may launch from the truck serving the customer and recover either:
    1) at a later node on that same truck route, or
    2) at the depot.
    """
    ranked_candidates: List[Tuple[float, SolutionData, Tuple[int, int, int]]] = []

    for truck in instance.truck_ids():
        route = solution.truck_routes.get(truck, [0, 0])
        if len(route) < 3:
            continue

        for customer_idx in range(1, len(route) - 1):
            customer = route[customer_idx]
            if customer == 0:
                continue
            if customer not in eligible_customers:
                continue
            if solution.u_truck.get((truck, customer), 0) != 1:
                continue

            for drone in instance.drone_ids():
                move_key = (customer, drone, truck)
                if tabu_moves is not None and move_key in tabu_moves:
                    continue

                found_feasible_for_key = False
                for launch_idx in range(0, customer_idx):
                    launch_node = route[launch_idx]
                    recovery_candidates = [0] + [route[idx] for idx in range(customer_idx + 1, len(route))]
                    seen_recovery_nodes: set[int] = set()

                    for recovery_node in recovery_candidates:
                        if recovery_node in seen_recovery_nodes:
                            continue
                        seen_recovery_nodes.add(recovery_node)

                        if recovery_node == customer:
                            continue

                        if battery_aware_screening:
                            if not _paired_trip_feasible(
                                instance,
                                drone,
                                launch_node,
                                customer,
                                recovery_node,
                                coords,
                                battery_slack_ratio=battery_slack_ratio,
                            ):
                                if stats is not None:
                                    stats["battery_infeasible_rejections"] += 1
                                    stats["battery_slack_rejections"] += 1
                                continue

                        candidate = move_to_drone(
                            instance,
                            solution,
                            customer,
                            drone,
                            launch_node=launch_node,
                            recovery_node=recovery_node,
                            source_truck=truck,
                            enforce_battery=battery_aware_screening,
                            battery_slack_ratio=battery_slack_ratio,
                        )
                        if candidate is None:
                            if battery_aware_screening and stats is not None:
                                stats["battery_infeasible_rejections"] += 1
                            continue

                        if not _quick_local_feasible(
                            instance,
                            candidate,
                            moved_customer=customer,
                            drone=drone,
                            coords=coords,
                            source_truck=truck,
                            launch_node=launch_node,
                            recovery_node=recovery_node,
                            battery_slack_ratio=battery_slack_ratio,
                        ):
                            if stats is not None:
                                stats["local_screen_rejections"] += 1
                            continue

                        found_feasible_for_key = True
                        if stats is not None:
                            stats["candidate_reassignments_evaluated"] += 1
                        obj, comp = evaluate_solution(instance, candidate)
                        candidate.objective = obj
                        candidate.components = comp
                        ranked_candidates.append((obj, candidate, move_key))

                if not found_feasible_for_key and tabu_moves is not None:
                    tabu_moves.add(move_key)

    return _select_best_strict_candidate(
        instance,
        ranked_candidates,
        strict_check_cap=strict_check_cap,
        stats=stats,
    )


def _apply_initial_paired_sorties(
    instance: InstanceData,
    solution: SolutionData,
    drone_to_truck: Dict[int, int],
    max_pairs: int,
    *,
    eligible_customers: set[int],
    battery_aware_screening: bool,
    battery_slack_ratio: float,
    strict_check_cap: int,
    tabu_moves: set[Tuple[int, int, int]] | None = None,
    stats: Dict[str, int] | None = None,
) -> SolutionData:
    if max_pairs <= 0:
        return solution
    coords = instance.coordinate_map()
    current = solution
    for _ in range(max_pairs):
        candidate = _extract_best_paired_displacement(
            instance,
            current,
            drone_to_truck,
            coords,
            eligible_customers=eligible_customers,
            battery_aware_screening=battery_aware_screening,
            battery_slack_ratio=battery_slack_ratio,
            strict_check_cap=strict_check_cap,
            tabu_moves=tabu_moves,
            stats=stats,
        )
        if candidate is None:
            break
        current = candidate
    return current


def _tabu_covers_all_candidates(
    instance: InstanceData,
    solution: SolutionData,
    eligible_customers: set[int],
    pairing_map: Dict[int, int],
    tabu_moves: set[Tuple[int, int, int]],
) -> bool:
    if not tabu_moves:
        return False

    candidate_keys: set[Tuple[int, int, int]] = set()
    if pairing_map:
        for drone, truck in pairing_map.items():
            route = solution.truck_routes.get(truck, [])
            for node in route:
                if node == 0 or node not in eligible_customers:
                    continue
                if solution.u_truck.get((truck, node), 0) == 1:
                    candidate_keys.add((node, drone, truck))
    else:
        for customer in eligible_customers:
            source_truck = next((t for t in instance.truck_ids() if solution.u_truck.get((t, customer), 0) == 1), 0)
            if source_truck <= 0:
                continue
            for drone in instance.drone_ids():
                candidate_keys.add((customer, drone, source_truck))

    if not candidate_keys:
        return True
    return candidate_keys.issubset(tabu_moves)


def _move_customer_back_to_truck(
    instance: InstanceData,
    solution: SolutionData,
    customer: int,
    *,
    coords: Dict[int, Tuple[float, float]],
) -> SolutionData | None:
    """Try restoring one drone-served customer to a truck route with best local insertion."""
    best: SolutionData | None = None
    best_obj = math.inf

    for truck in instance.truck_ids():
        base_route = list(solution.truck_routes.get(truck, [0, 0]))
        if not base_route:
            base_route = [0, 0]
        if base_route[0] != 0:
            base_route = [0] + base_route
        if base_route[-1] != 0:
            base_route = base_route + [0]

        for pos in range(1, len(base_route)):
            candidate = _empty_solution_clone(solution)

            # Remove customer from all drone routes/assignments.
            for d in instance.drone_ids():
                dr = list(candidate.drone_routes.get(d, [0, 0]))
                dr = [n for n in dr if n != customer]
                if len(dr) < 2:
                    dr = [0, 0]
                if dr[0] != 0:
                    dr = [0] + dr
                if dr[-1] != 0:
                    dr = dr + [0]
                candidate.drone_routes[d] = dr
                candidate.u_drone[(d, customer)] = 0

            # Reset truck assignment for this customer.
            for t in instance.truck_ids():
                candidate.u_truck[(t, customer)] = 1 if t == truck else 0

            new_route = base_route[:pos] + [customer] + base_route[pos:]
            candidate.truck_routes[truck] = new_route

            # Clear stale arc dictionaries to force route-based interpretation.
            candidate.x_truck = {}
            candidate.x_drone = {}

            # Remove stale sync flags that may reference the moved customer node.
            for key in list(candidate.z1.keys()):
                if key[0] == customer:
                    candidate.z1[key] = 0
            for key in list(candidate.z2.keys()):
                if key[0] == customer:
                    candidate.z2[key] = 0

            obj, comp = evaluate_solution(instance, candidate)
            candidate.objective = obj
            candidate.components = comp

            if is_feasible(instance, candidate) and obj < best_obj:
                best_obj = obj
                best = candidate

    return best


def _repair_infeasible_solution(
    instance: InstanceData,
    solution: SolutionData,
    *,
    max_steps: int,
) -> SolutionData | None:
    """Greedy feasibility-recovery by moving drone-served customers back to trucks."""
    _sync_route_assignments(solution, instance)
    if is_feasible(instance, solution):
        return solution

    coords = instance.coordinate_map()
    current = _empty_solution_clone(solution)
    _sync_route_assignments(current, instance)

    for _ in range(max(1, int(max_steps))):
        drone_customers = [
            c
            for c in instance.customer_ids
            if any(current.u_drone.get((d, c), 0) == 1 for d in instance.drone_ids())
        ]
        if not drone_customers:
            break

        best_step: SolutionData | None = None
        best_step_obj = math.inf
        for customer in drone_customers:
            repaired = _move_customer_back_to_truck(instance, current, customer, coords=coords)
            if repaired is None:
                continue
            if repaired.objective < best_step_obj:
                best_step_obj = repaired.objective
                best_step = repaired

        if best_step is None:
            break

        current = best_step
        _sync_route_assignments(current, instance)
        if is_feasible(instance, current):
            return current

    return current if is_feasible(instance, current) else None


@dataclass
class NILSResult:
    solution: SolutionData
    iterations: int
    improved: int


def run_nils(
    instance: InstanceData,
    seed: int,
    max_iter: int = 30,
    max_no_improve: int = 5,
    time_limit: int = 600,
    drone_to_truck: Dict[int, int] | None = None,
    use_paired_initialization: bool = False,
    max_initial_pairings: int = 2,
    enable_local_search: bool = True,
    enable_perturbation: bool = True,
    battery_aware_screening: bool = True,
    battery_slack_ratio: float | None = None,
    no_unpairing_mode: bool = False,
    strict_candidate_check_cap: int = 12,
    full_feasibility_check_interval: int = 6,
    require_feasible_incumbent: bool = True,
    repair_infeasible_candidates: bool = True,
    repair_max_steps: int = 3,
    initialization_mode: str = "greedy",
    initialization_time_limit_seconds: int = 3,
    strict_improving_acceptance: bool = False,
    stagnation_perturb_from_best: bool = False,
    stagnation_perturb_attempts: int = 1,
    metropolis_temperature_scale: float = 1.0,
) -> SolutionData:
    """Run the NILS routine.

    If `drone_to_truck` is provided, the algorithm uses fixed-pair neighborhoods where
    each drone can only launch and recover on its assigned truck.
    If `no_unpairing_mode` is True, cross-truck recovery is disabled and a drone may
    only recover at the depot or on the launch truck route.
    """
    rng = np.random.default_rng(seed)
    rnd = random.Random(seed)
    if battery_slack_ratio is None:
        battery_slack_ratio = float(instance.metadata.get("battery_slack_ratio", 0.15))
    battery_slack_ratio = float(max(0.0, battery_slack_ratio))
    stagnation_perturb_attempts = max(1, int(stagnation_perturb_attempts))
    metropolis_temperature_scale = max(1e-6, float(metropolis_temperature_scale))

    current = build_initial_solution(
        instance,
        seed,
        initializer=initialization_mode,
        ortools_time_limit_seconds=max(1, int(initialization_time_limit_seconds)),
    )
    current.objective, current.components = evaluate_solution(instance, current)
    _sync_route_assignments(current, instance)
    if require_feasible_incumbent and not is_feasible(instance, current):
        repaired = _repair_infeasible_solution(
            instance,
            current,
            max_steps=max(1, int(repair_max_steps)),
        )
        if repaired is not None:
            current = repaired
            current.objective, current.components = evaluate_solution(instance, current)
            _sync_route_assignments(current, instance)
    if require_feasible_incumbent and not is_feasible(instance, current):
        fallback_current: SolutionData | None = None
        fallback_attempts = [
            ("ortools", int(seed)),
            ("greedy", int(seed) + 1),
            ("greedy", int(seed) + 2),
        ]
        for init_mode, init_seed in fallback_attempts:
            trial = build_initial_solution(
                instance,
                init_seed,
                initializer=init_mode,
                ortools_time_limit_seconds=max(1, int(initialization_time_limit_seconds)),
            )
            trial.objective, trial.components = evaluate_solution(instance, trial)
            _sync_route_assignments(trial, instance)
            if is_feasible(instance, trial):
                fallback_current = trial
                break
            repaired_trial = _repair_infeasible_solution(
                instance,
                trial,
                max_steps=max(1, int(repair_max_steps)),
            )
            if repaired_trial is not None:
                repaired_trial.objective, repaired_trial.components = evaluate_solution(instance, repaired_trial)
                _sync_route_assignments(repaired_trial, instance)
                if is_feasible(instance, repaired_trial):
                    fallback_current = repaired_trial
                    break
        if fallback_current is not None:
            current = fallback_current
        else:
            current.status = "nils_infeasible_initialization"
            return current

    pairing_map = _normalize_drone_to_truck(instance, drone_to_truck)
    if pairing_map and len(pairing_map) != instance.num_drones:
        pairing_map = {}
    if pairing_map:
        pairing_map = dict(pairing_map)
    else:
        pairing_map = {}

    eligible_customers = _eligible_customer_ids(instance)
    stats = {
        "iterations": 0,
        "improving_moves_accepted": 0,
        "candidate_reassignments_evaluated": 0,
        "battery_infeasible_rejections": 0,
        "battery_slack_rejections": 0,
        "infeasible_candidate_rejections": 0,
        "tabu_reverts": 0,
        "perturbation_moves": 0,
        "local_search_calls": 0,
        "local_screen_rejections": 0,
        "strict_feasibility_checks": 0,
        "repair_attempts": 0,
        "repair_successes": 0,
        "repair_failures": 0,
    }
    tabu_moves: set[Tuple[int, int, int]] = set()

    if use_paired_initialization and pairing_map:
        current = _apply_initial_paired_sorties(
            instance,
            current,
            pairing_map,
            max_pairs=max_initial_pairings,
            eligible_customers=eligible_customers,
            battery_aware_screening=battery_aware_screening,
            battery_slack_ratio=battery_slack_ratio,
            strict_check_cap=strict_candidate_check_cap,
            tabu_moves=tabu_moves,
            stats=stats,
        )
        current.objective, current.components = evaluate_solution(instance, current)

    start = time.time()
    best = _empty_solution_clone(current)
    best.status = "nils_best"
    no_improve = 0

    coords = instance.coordinate_map()
    for it in range(1, max_iter + 1):
        if time.time() - start > time_limit:
            break
        stats["iterations"] = it

        if pairing_map:
            candidate = _extract_best_paired_displacement(
                instance,
                current,
                pairing_map,
                coords,
                eligible_customers=eligible_customers,
                battery_aware_screening=battery_aware_screening,
                battery_slack_ratio=battery_slack_ratio,
                strict_check_cap=strict_candidate_check_cap,
                tabu_moves=tabu_moves,
                stats=stats,
            )
        elif no_unpairing_mode:
            candidate = _extract_best_no_unpairing_displacement(
                instance,
                current,
                coords,
                eligible_customers=eligible_customers,
                battery_aware_screening=battery_aware_screening,
                battery_slack_ratio=battery_slack_ratio,
                strict_check_cap=strict_candidate_check_cap,
                tabu_moves=tabu_moves,
                stats=stats,
            )
        else:
            candidate = _extract_best_displacement(
                instance,
                current,
                coords=coords,
                eligible_customers=eligible_customers,
                battery_aware_screening=battery_aware_screening,
                battery_slack_ratio=battery_slack_ratio,
                strict_check_cap=strict_candidate_check_cap,
                tabu_moves=tabu_moves,
                stats=stats,
            )

        if candidate is None:
            if enable_perturbation:
                perturb_source = best if stagnation_perturb_from_best else current
                best_perturbed = None
                for _ in range(stagnation_perturb_attempts):
                    trial = relocate_random_customer(perturb_source, instance, rng)
                    stats["perturbation_moves"] += 1
                    if trial is None:
                        continue
                    trial.objective, trial.components = evaluate_solution(instance, trial)
                    _sync_route_assignments(trial, instance)
                    if not is_feasible(instance, trial):
                        continue
                    if best_perturbed is None or trial.objective < best_perturbed.objective:
                        best_perturbed = trial
                candidate = best_perturbed
                if candidate is None:
                    if _tabu_covers_all_candidates(instance, current, eligible_customers, pairing_map, tabu_moves):
                        no_improve += 1
                    continue
            else:
                if not _tabu_covers_all_candidates(instance, current, eligible_customers, pairing_map, tabu_moves):
                    continue
                no_improve += 1
                if no_improve >= max_no_improve:
                    break
                continue
        strict_checked = bool(float(candidate.components.get("_strict_feasibility_checked", 0.0)) >= 0.5)
        if not strict_checked:
            candidate.objective, candidate.components = evaluate_solution(instance, candidate)
        _sync_route_assignments(candidate, instance)

        need_global_check = require_feasible_incumbent or (not strict_checked) or (
            full_feasibility_check_interval > 0 and it % int(full_feasibility_check_interval) == 0
        )
        if need_global_check:
            stats["strict_feasibility_checks"] += 1
            if not is_feasible(instance, candidate):
                repaired = None
                if repair_infeasible_candidates:
                    stats["repair_attempts"] += 1
                    repaired = _repair_infeasible_solution(
                        instance,
                        candidate,
                        max_steps=max(1, int(repair_max_steps)),
                    )
                if repaired is not None:
                    stats["repair_successes"] += 1
                    candidate = repaired
                    candidate.objective, candidate.components = evaluate_solution(instance, candidate)
                    _sync_route_assignments(candidate, instance)
                else:
                    stats["repair_failures"] += 1
                    stats["infeasible_candidate_rejections"] += 1
                    stats["tabu_reverts"] += 1
                    no_improve += 1
                    if no_improve >= max_no_improve:
                        break
                    continue
        if enable_perturbation and it % 4 == 0:
            perturbed = relocate_random_customer(candidate, instance, rng)
            stats["perturbation_moves"] += 1
            perturbed.objective, perturbed.components = evaluate_solution(instance, perturbed)
            if perturbed.objective < candidate.objective or rnd.random() < 0.2:
                candidate = perturbed
            _sync_route_assignments(candidate, instance)

        if require_feasible_incumbent:
            stats["strict_feasibility_checks"] += 1
            if not is_feasible(instance, candidate):
                repaired = None
                if repair_infeasible_candidates:
                    stats["repair_attempts"] += 1
                    repaired = _repair_infeasible_solution(
                        instance,
                        candidate,
                        max_steps=max(1, int(repair_max_steps)),
                    )
                if repaired is not None:
                    stats["repair_successes"] += 1
                    candidate = repaired
                    candidate.objective, candidate.components = evaluate_solution(instance, candidate)
                    _sync_route_assignments(candidate, instance)
                else:
                    stats["repair_failures"] += 1
                    stats["infeasible_candidate_rejections"] += 1
                    stats["tabu_reverts"] += 1
                    no_improve += 1
                    if no_improve >= max_no_improve:
                        break
                    continue

        temp = max(1e-6, float(max_iter - it + 1) * metropolis_temperature_scale)
        accept = candidate.objective <= current.objective + 1e-12
        if not strict_improving_acceptance and not accept:
            accept = metropolis(current.objective, candidate.objective, temp, rnd)
        if accept:
            if enable_local_search:
                candidate = apply_two_opt_to_solution(candidate, instance)
                candidate.objective, candidate.components = evaluate_solution(instance, candidate)
                _sync_route_assignments(candidate, instance)
                stats["local_search_calls"] += 1
            current = candidate
            no_improve = 0
            if candidate.objective < best.objective - 1e-12:
                best = _empty_solution_clone(candidate)
                stats["improving_moves_accepted"] += 1
        else:
            no_improve += 1

        if no_improve >= max_no_improve:
            break

    best.status = "nils_complete"
    best.run_time_seconds = time.time() - start
    _sync_route_assignments(best, instance)
    best.components = dict(best.components)
    best.components["iterations"] = float(stats["iterations"])
    best.components["improving_moves_accepted"] = float(stats["improving_moves_accepted"])
    best.components["candidate_reassignments_evaluated"] = float(stats["candidate_reassignments_evaluated"])
    best.components["battery_infeasible_rejections"] = float(stats["battery_infeasible_rejections"])
    best.components["battery_slack_rejections"] = float(stats["battery_slack_rejections"])
    best.components["infeasible_candidate_rejections"] = float(stats["infeasible_candidate_rejections"])
    best.components["tabu_reverts"] = float(stats["tabu_reverts"])
    best.components["perturbation_moves"] = float(stats["perturbation_moves"])
    best.components["local_search_calls"] = float(stats["local_search_calls"])
    best.components["local_screen_rejections"] = float(stats["local_screen_rejections"])
    best.components["strict_feasibility_checks"] = float(stats["strict_feasibility_checks"])
    best.components["repair_attempts"] = float(stats["repair_attempts"])
    best.components["repair_successes"] = float(stats["repair_successes"])
    best.components["repair_failures"] = float(stats["repair_failures"])
    best.components["battery_screening_enabled"] = 1.0 if battery_aware_screening else 0.0
    best.components["battery_slack_ratio"] = float(max(0.0, battery_slack_ratio))
    best.components["tabu_size_final"] = float(len(tabu_moves))
    best.components["local_search_enabled"] = 1.0 if enable_local_search else 0.0
    best.components["perturbation_enabled"] = 1.0 if enable_perturbation else 0.0
    best.components["no_unpairing_mode"] = 1.0 if no_unpairing_mode else 0.0
    best.components["fixed_pair_mode"] = 1.0 if pairing_map else 0.0
    best.components["require_feasible_incumbent"] = 1.0 if require_feasible_incumbent else 0.0
    best.components["repair_infeasible_candidates"] = 1.0 if repair_infeasible_candidates else 0.0
    best.components["repair_max_steps"] = float(max(1, int(repair_max_steps)))
    best.components["strict_improving_acceptance"] = 1.0 if strict_improving_acceptance else 0.0
    best.components["stagnation_perturb_from_best"] = 1.0 if stagnation_perturb_from_best else 0.0
    best.components["stagnation_perturb_attempts"] = float(stagnation_perturb_attempts)
    best.components["metropolis_temperature_scale"] = float(metropolis_temperature_scale)
    return best
