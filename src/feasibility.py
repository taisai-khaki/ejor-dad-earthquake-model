"""Strict solution validator for UTDRP-DP solution objects."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .data_models import InstanceData, SolutionData
from .distance_utils import euclidean_distance, manhattan_distance, seconds_to_minutes, travel_time_from_distance


@dataclass
class ValidationFinding:
    code: str
    severity: str
    message: str
    vehicle: str
    node: int | None = None


def _route_from_arcs(
    start: int,
    arcs: Dict[Tuple[int, int], float],
    tolerance: float = 1e-8,
) -> List[int]:
    outgoing = {}
    incoming_count = {i: 0 for i in range(0, 1_000)}
    for (i, j), v in arcs.items():
        if v >= 0.5:
            outgoing.setdefault(i, []).append(j)
            incoming_count[j] = incoming_count.get(j, 0) + 1
    current = start
    visited = {start}
    route = [start]
    while True:
        next_nodes = outgoing.get(current, [])
        if not next_nodes:
            break
        if len(next_nodes) != 1:
            raise ValueError(f"Non-functional route: node {current} has {len(next_nodes)} outgoing arcs")
        nxt = next_nodes[0]
        route.append(nxt)
        if nxt == start:
            break
        if nxt in visited:
            raise ValueError(f"Cycle detected at node {nxt}")
        visited.add(nxt)
        current = nxt
        if len(route) > 10_000:
            raise ValueError("Route reconstruction timeout")
    return route


def _infer_arrival(route: Sequence[int], distances: np.ndarray, speed_kmph: float, service_times: Dict[int, float]) -> Dict[int, float]:
    arr: Dict[int, float] = {route[0]: 0.0}
    current = route[0]
    time = 0.0
    for nxt in route[1:]:
        time += travel_time_from_distance(distances[current][nxt], speed_kmph)
        time += service_times.get(current, 0.0)
        arr[nxt] = time
        current = nxt
    return arr


def _node_positions(instance: InstanceData) -> Dict[int, Tuple[float, float]]:
    coords: Dict[int, Tuple[float, float]] = {0: (0.0, 0.0)}
    for c in instance.customers:
        coords[c.node_id] = (c.x, c.y)
    return coords


def validate_solution(instance: InstanceData, solution: SolutionData, *, tol: float = 1e-5) -> List[ValidationFinding]:
    """Validate a complete solution; return findings (ERROR/WARN)."""
    findings: List[ValidationFinding] = []
    customers = instance.customer_ids
    nodes = instance.all_nodes()

    if not solution.u_truck and not solution.u_drone and not solution.x_truck and not solution.x_drone:
        findings.append(ValidationFinding("EMPTY", "ERROR", "Solution has no routing or assignment data", "global"))
        return findings

    served_by_truck = set(solution.served_by_truck())
    served_by_drone = set(solution.served_by_drone())
    for route in solution.truck_routes.values():
        served_by_truck.update(node for node in route if node != 0)
    for route in solution.drone_routes.values():
        served_by_drone.update(node for node in route if node != 0)

    for c in customers:
        in_truck = c in served_by_truck
        in_drone = c in served_by_drone
        if not (in_truck or in_drone):
            findings.append(ValidationFinding("UNSERVED", "ERROR", f"Customer {c} is not served", "global", c))
        if in_truck and in_drone:
            findings.append(ValidationFinding("DUPLICATE", "ERROR", f"Customer {c} assigned to both truck and drone", "global", c))

    # route reconstruction from arc binaries
    truck_routes: Dict[int, List[int]] = {}
    for t in instance.truck_ids():
        arc_map = {(i, j): 1 for (i, j, tt), v in solution.x_truck.items() if tt == t and v >= 0.5}
        if not arc_map and t in solution.truck_routes and len(solution.truck_routes[t]) >= 2:
            route = list(solution.truck_routes[t])
            truck_routes[t] = route
        elif not arc_map:
            truck_routes[t] = [0, 0]
            continue
        else:
            try:
                route = _route_from_arcs(0, arc_map)
                truck_routes[t] = route
            except Exception as ex:
                findings.append(ValidationFinding("ROUTE", "ERROR", str(ex), "truck", t))
                truck_routes[t] = [0, 0]
                continue
        route = truck_routes[t]
        if route[0] != 0:
            findings.append(ValidationFinding("ROUTE", "ERROR", f"Truck {t} route does not start at depot", "truck", t))
        if route[-1] != 0:
            findings.append(ValidationFinding("ROUTE", "ERROR", f"Truck {t} route does not return to depot", "truck", t))

    drone_routes: Dict[int, List[int]] = {}
    for d in instance.drone_ids():
        arc_map = {(i, j): 1 for (i, j, dd), v in solution.x_drone.items() if dd == d and v >= 0.5}
        if not arc_map and d in solution.drone_routes and len(solution.drone_routes[d]) >= 2:
            route = list(solution.drone_routes[d])
            drone_routes[d] = route
        elif not arc_map:
            drone_routes[d] = [0, 0]
            continue
        else:
            try:
                route = _route_from_arcs(0, arc_map)
                drone_routes[d] = route
            except Exception as ex:
                findings.append(ValidationFinding("ROUTE", "ERROR", str(ex), "drone", d))
                drone_routes[d] = [0, 0]
                continue
        route = drone_routes[d]
        if route[0] != 0:
            findings.append(ValidationFinding("ROUTE", "ERROR", f"Drone {d} route does not start at depot", "drone", d))
        if route[-1] != 0:
            findings.append(ValidationFinding("ROUTE", "ERROR", f"Drone {d} route does not return to depot", "drone", d))

    solution.truck_routes = truck_routes
    solution.drone_routes = drone_routes

    coords = _node_positions(instance)
    # Build full distance matrices
    dist_manh = np.zeros((len(nodes), len(nodes)))
    dist_euc = np.zeros((len(nodes), len(nodes)))
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            dist_manh[i, j] = manhattan_distance(coords[i], coords[j])
            dist_euc[i, j] = euclidean_distance(coords[i], coords[j])

    # Truck and drone timing consistency
    for t, route in truck_routes.items():
        if len(route) < 2:
            continue
        speed = instance.trucks[t - 1].speed_kmph
        service_times = {0: 0.0}
        for c in instance.customers:
            service_times[c.node_id] = c.service_time_min
        arrivals = _infer_arrival(route, dist_manh, speed, service_times)
        for node, arr in arrivals.items():
            if node != 0:
                c = instance.customer_map()[node]
                lb, ub = c.window
                if arr > ub + tol:
                    findings.append(ValidationFinding("TW_VIOLATION", "WARN", f"Truck {t} arrives after ub at customer {node}", "truck", node))
                if arr < lb - tol:
                    findings.append(ValidationFinding("TW_VIOLATION", "WARN", f"Truck {t} arrives before lb at customer {node}", "truck", node))
        if arrivals.get(0, 0.0) > instance.constants.max_shift_minutes + tol:
            findings.append(ValidationFinding("SHIFT", "ERROR", f"Truck {t} exceeds shift", "truck", t))

    for d, route in drone_routes.items():
        if len(route) < 2:
            continue
        speed = instance.drones[d - 1].speed_kmph
        service_times = {0: 0.0}
        for c in instance.customers:
            service_times[c.node_id] = c.service_time_min
        arrivals = _infer_arrival(route, dist_euc, speed, service_times)
        for node, arr in arrivals.items():
            if node != 0:
                c = instance.customer_map()[node]
                lb, ub = c.window
                if arr > ub + tol:
                    findings.append(ValidationFinding("TW_VIOLATION", "WARN", f"Drone {d} arrives after ub at customer {node}", "drone", node))
        if arrivals.get(0, 0.0) > instance.constants.max_shift_minutes + tol:
            findings.append(ValidationFinding("SHIFT", "ERROR", f"Drone {d} exceeds shift", "drone", d))

    # Load feasibility
    for t in instance.truck_ids():
        cap = instance.trucks[t - 1].capacity_kg
        load = cap
        visited = set()
        for node in truck_routes[t]:
            if node == 0:
                continue
            if node in visited:
                break
            visited.add(node)
            if node not in served_by_truck:
                continue
            c = instance.customer_map()[node]
            load -= c.demand_kg
            if load < -tol:
                findings.append(ValidationFinding("LOAD", "ERROR", f"Truck {t} overload after {node}", "truck", node))
        if load < -tol:
            findings.append(ValidationFinding("LOAD", "ERROR", f"Truck {t} negative final load", "truck", t))

    # Battery feasibility
    for d in instance.drone_ids():
        battery = instance.drones[d - 1].max_battery_wh
        drone = instance.drones[d - 1]
        speed = drone.speed_kmph
        visited = set()
        for i_idx in range(len(drone_routes[d]) - 1):
            frm = drone_routes[d][i_idx]
            to = drone_routes[d][i_idx + 1]
            if frm in visited:
                break
            visited.add(frm)
            arc_time = travel_time_from_distance(dist_euc[frm, to], speed)
            # fallback if loaded flag is absent
            loaded = solution.y_loaded.get((d, to), 0.0) >= 0.5
            burn = drone.energy_per_min_when_loaded if loaded else drone.energy_per_min_when_empty
            battery -= burn * arc_time
            if battery < -tol:
                findings.append(ValidationFinding("ENERGY", "ERROR", f"Drone {d} battery infeasible on {frm}->{to}", "drone", d))
            # if a swap event is claimed at destination, allow full reset
            swap_here = any(solution.z2.get((to, d, tt), 0) for tt in instance.truck_ids())
            if swap_here:
                battery = max(battery, drone.max_battery_wh)

    # Tardiness
    served_time: Dict[int, float] = {c: 0.0 for c in customers}
    for c in customers:
        t_truck = [solution.a_truck.get((t, c), 0.0) for t in instance.truck_ids()]
        t_drone = [solution.a_drone.get((d, c), 0.0) for d in instance.drone_ids()]
        served_time[c] = max(t_truck + t_drone)
        expected = max(0.0, served_time[c] - instance.customer_map()[c].ub)
        observed = solution.tardiness.get(c, expected)
        if abs(observed - expected) > max(1e-3, 1e-6 * expected + 1e-3):
            findings.append(ValidationFinding("TARDINESS", "WARN", f"Stored tardiness mismatch at customer {c}", "global", c))
            solution.tardiness[c] = expected

    # Sync consistency checks
    for (i, d, t), val in solution.z1.items():
        if val <= 0.5:
            continue
        a_t = solution.a_truck.get((t, i), 0.0)
        a_d = solution.a_drone.get((d, i), 0.0)
        delta = abs(a_t - a_d)
        if delta > seconds_to_minutes(instance.constants.swap_time_s) + tol:
            findings.append(
                ValidationFinding(
                    "SYNC",
                    "WARN",
                    f"Parcel exchange mismatch at node {i} for truck {t}, drone {d}",
                    "sync",
                    i,
                )
            )

    for (i, d, t), val in solution.z2.items():
        if val <= 0.5:
            continue
        a_t = solution.a_truck.get((t, i), 0.0)
        a_d = solution.a_drone.get((d, i), 0.0)
        delta = abs(a_t - a_d)
        if delta > seconds_to_minutes(instance.constants.reload_time_s) + tol:
            findings.append(
                ValidationFinding(
                    "SYNC",
                    "WARN",
                    f"Battery swap mismatch at node {i} for truck {t}, drone {d}",
                    "sync",
                    i,
                )
            )

    return findings


def is_feasible(instance: InstanceData, solution: SolutionData) -> bool:
    findings = validate_solution(instance, solution)
    return all(item.severity != "ERROR" for item in findings)
