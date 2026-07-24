"""Local search operators used by the heuristic stack."""
from __future__ import annotations

from typing import Dict, List, Tuple
from copy import deepcopy

from ..distance_utils import euclidean_distance, manhattan_distance
from ..data_models import SolutionData
from ..data_models import InstanceData


def route_cost(route: List[int], nodes: Dict[int, Tuple[float, float]], metric: str = "manhattan") -> float:
    if not route or len(route) < 2:
        return 0.0
    cost = 0.0
    for i in range(len(route) - 1):
        if metric == "euclidean":
            cost += euclidean_distance(nodes[route[i]], nodes[route[i + 1]])
        else:
            cost += manhattan_distance(nodes[route[i]], nodes[route[i + 1]])
    return cost


def two_opt(route: List[int], nodes: dict[int, tuple[float, float]], metric: str = "manhattan") -> List[int]:
    """Standard 2-opt on a single route starting/ending at depot."""
    if len(route) <= 4:
        return route
    best = route
    best_improved = True
    while best_improved:
        best_improved = False
        best_delta = 0
        candidate = best
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                delta = route_cost(new_route, nodes, metric=metric) - route_cost(best, nodes, metric=metric)
                if delta < best_delta:
                    best = new_route
                    best_delta = delta
                    best_improved = True
        if best_delta < 0 and best_improved:
            candidate = best
        best = candidate
    return best


def evaluate_route_route(route: List[int], nodes: dict[int, tuple[float, float]], metric: str = "manhattan") -> float:
    return route_cost(route, nodes, metric=metric)


def apply_two_opt_to_solution(solution: SolutionData, instance: InstanceData) -> SolutionData:
    node_map = {0: (0.0, 0.0)}
    for c in instance.customers:
        node_map[c.node_id] = (c.x, c.y)
    out = deepcopy(solution)

    for t, route in solution.truck_routes.items():
        if len(route) <= 4:
            continue
        if route[0] != 0 or route[-1] != 0:
            continue
        improved = two_opt(route, node_map, metric="manhattan")
        out.truck_routes[t] = improved

    for d, route in solution.drone_routes.items():
        if len(route) <= 4:
            continue
        if route[0] != 0 or route[-1] != 0:
            continue
        improved = two_opt(route, node_map, metric="euclidean")
        out.drone_routes[d] = improved

    # refresh only route-dependent arrays to stay consistent.
    out.x_truck = {}
    out.x_drone = {}
    out.l_truck = {}
    out.a_truck = {}
    out.l_drone = {}
    out.a_drone = {}
    for t, route in out.truck_routes.items():
        for idx in range(len(route) - 1):
            frm = route[idx]
            to = route[idx + 1]
            out.x_truck[(frm, to, t)] = 1
            out.l_truck[(t, frm)] = out.l_truck.get((t, frm), 0.0)
            out.a_truck[(t, frm)] = out.a_truck.get((t, frm), 0.0)
    for d, route in out.drone_routes.items():
        for idx in range(len(route) - 1):
            frm = route[idx]
            to = route[idx + 1]
            out.x_drone[(frm, to, d)] = 1
            out.l_drone[(d, frm)] = out.l_drone.get((d, frm), 0.0)
            out.a_drone[(d, frm)] = out.a_drone.get((d, frm), 0.0)
    return out
