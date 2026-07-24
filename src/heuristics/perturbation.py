"""Perturbation moves for heuristic diversification."""
from __future__ import annotations

import numpy as np
from copy import deepcopy

from ..data_models import SolutionData
from ..data_models import InstanceData


def relocate_random_customer(solution: SolutionData, instance: InstanceData, rng: np.random.Generator) -> SolutionData:
    """Randomly relocate one truck customer to another truck."""
    if not solution.truck_routes:
        return solution

    # pick a route that has at least one customer
    valid_moves = []
    for t, route in solution.truck_routes.items():
        if len(route) <= 3:
            continue
        for idx in range(1, len(route) - 1):
            valid_moves.append((t, idx))
    if not valid_moves:
        return solution

    src_t, src_idx = valid_moves[rng.integers(0, len(valid_moves))]
    customer = solution.truck_routes[src_t][src_idx]
    if customer == 0:
        return solution

    # pick destination truck (possibly same)
    truck_ids = list(solution.truck_routes.keys())
    dst_t = int(rng.choice(truck_ids))
    if dst_t == src_t and len(solution.truck_routes[src_t]) <= 3:
        return solution

    new = deepcopy(solution)
    src_route = solution.truck_routes[src_t].copy()
    src_route.pop(src_idx)

    dst_route = solution.truck_routes[dst_t].copy()
    insert_pos = rng.integers(1, max(2, len(dst_route)))
    dst_route.insert(int(insert_pos), customer)

    new.truck_routes = solution.truck_routes.copy()
    new.truck_routes[src_t] = src_route
    new.truck_routes[dst_t] = dst_route

    # flip assignment indicators
    for d in range(1, instance.num_drones + 1):
        new.u_drone[(d, customer)] = 0
    for t in solution.truck_routes:
        if t == src_t:
            new.u_truck[(t, customer)] = 1 if t == dst_t else 0
        elif t == dst_t:
            new.u_truck[(t, customer)] = 1

    # reset route-dependent binary fields; downstream evaluator rebuilds if needed
    new.x_truck = {}
    new.a_truck = {}
    new.l_truck = {}
    new.objective = solution.objective
    return new


def remove_and_reinsert(solution: SolutionData, k: int, rng: np.random.Generator, instance: InstanceData) -> SolutionData:
    """Remove up to k customers and greedily reinsert into best truck end position."""
    customers = [c for c in range(1, instance.num_customers + 1) if any((solution.u_truck.get((t, c), 0) == 1) for t in solution.truck_routes)]
    if not customers:
        return solution

    removed = rng.choice(customers, size=min(k, len(customers)), replace=False)
    sol = deepcopy(solution)
    for c in removed:
        # strip service and x entries from existing route
        for t in sol.truck_routes:
            if c in sol.truck_routes[t]:
                sol.truck_routes[t] = [x for x in sol.truck_routes[t] if x != c]
                sol.u_truck[(t, c)] = 0
        for d in range(1, instance.num_drones + 1):
            sol.u_drone[(d, c)] = 0

    # reinsertion greedily to shortest route tail
    truck_ids = list(sol.truck_routes.keys())
    for c in removed:
        t = int(rng.choice(truck_ids))
        route = sol.truck_routes[t]
        if not route or route[-1] != 0:
            route = route + [0]
        insert_pos = len(route) - 1
        route.insert(insert_pos, int(c))
        sol.truck_routes[t] = route
        sol.u_truck[(t, int(c))] = 1

    sol.x_truck = {}
    sol.a_truck = {}
    sol.l_truck = {}
    sol.objective = solution.objective
    return sol
