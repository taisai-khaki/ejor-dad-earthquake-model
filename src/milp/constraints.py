"""Constraint builders for the MILP formulation.

The implementation is intentionally conservative: constraints are written to be
numerically stable and aligned with the data model, while keeping several
manuscript choices explicit in configuration and TODO files.
"""
from __future__ import annotations

from typing import Dict, Tuple

from ..data_models import InstanceData
from .backend import gp


Var = object


def add_return_to_depot(
    model,
    instance: InstanceData,
    x_truck: Dict[Tuple[int, int, int], Var],
    x_drone: Dict[Tuple[int, int, int], Var],
    nodes: list[int],
) -> None:
    """Enforce depot conservation for every used vehicle."""
    customers = _customer_nodes(nodes)
    for t in instance.truck_ids():
        model.addConstr(
            gp.quicksum(x_truck[(0, j, t)] for j in customers)
            == gp.quicksum(x_truck[(j, 0, t)] for j in customers),
            name=f"truck_return_balance_{t}",
        )
        model.addConstr(
            gp.quicksum(x_truck[(0, j, t)] for j in customers) <= 1,
            name=f"truck_single_route_{t}",
        )
    for d in instance.drone_ids():
        model.addConstr(
            gp.quicksum(x_drone[(0, j, d)] for j in customers)
            == gp.quicksum(x_drone[(j, 0, d)] for j in customers),
            name=f"drone_return_balance_{d}",
        )
        model.addConstr(
            gp.quicksum(x_drone[(0, j, d)] for j in customers) <= 1,
            name=f"drone_single_route_{d}",
        )


def add_service_and_flow(
    model,
    instance: InstanceData,
    x_truck: Dict[Tuple[int, int, int], Var],
    x_drone: Dict[Tuple[int, int, int], Var],
    u_truck: Dict[Tuple[int, int], Var],
    u_drone: Dict[Tuple[int, int], Var],
    nodes: list[int],
) -> None:
    """Customer service exclusivity, incoming/outgoing flow, and node conservation."""
    customers = _customer_nodes(nodes)
    for j in customers:
        model.addConstr(
            gp.quicksum(u_truck[(t, j)] for t in instance.truck_ids())
            + gp.quicksum(u_drone[(d, j)] for d in instance.drone_ids())
            == 1,
            name=f"exclusive_service_{j}",
        )
        for t in instance.truck_ids():
            model.addConstr(
                gp.quicksum(x_truck[(i, j, t)] for i in nodes if i != j) == u_truck[(t, j)],
                name=f"truck_in_{t}_{j}",
            )
            model.addConstr(
                gp.quicksum(x_truck[(j, i, t)] for i in nodes if i != j) == u_truck[(t, j)],
                name=f"truck_out_{t}_{j}",
            )
        for d in instance.drone_ids():
            model.addConstr(
                gp.quicksum(x_drone[(i, j, d)] for i in nodes if i != j) == u_drone[(d, j)],
                name=f"drone_in_{d}_{j}",
            )
            model.addConstr(
                gp.quicksum(x_drone[(j, i, d)] for i in nodes if i != j) == u_drone[(d, j)],
                name=f"drone_out_{d}_{j}",
            )

    for t in instance.truck_ids():
        for i in nodes:
            model.addConstr(
                gp.quicksum(x_truck[(i, j, t)] for j in nodes if i != j)
                == gp.quicksum(x_truck[(j, i, t)] for j in nodes if i != j),
                name=f"truck_flow_{t}_{i}",
            )
    for d in instance.drone_ids():
        for i in nodes:
            model.addConstr(
                gp.quicksum(x_drone[(i, j, d)] for j in nodes if i != j)
                == gp.quicksum(x_drone[(j, i, d)] for j in nodes if i != j),
                name=f"drone_flow_{d}_{i}",
            )


def add_demand_and_load(
    model,
    instance: InstanceData,
    x_truck: Dict[Tuple[int, int, int], Var],
    u_truck: Dict[Tuple[int, int], Var],
    y_loaded: Dict[Tuple[int, int], Var],
    w: Dict[Tuple[int, int], Var],
    z1: Dict[Tuple[int, int, int], Var],
    z2: Dict[Tuple[int, int, int], Var],
    nodes: list[int],
    u_drone: Dict[Tuple[int, int], Var],
    big_m: float,
) -> None:
    """Truck load flow and synchronization indicator relations."""
    customers = _customer_nodes(nodes)
    for t in instance.truck_ids():
        truck = instance.trucks[t - 1]
        model.addConstr(w[(t, 0)] == truck.capacity_kg, name=f"truck_load_start_{t}")
        for j in nodes:
            model.addConstr(w[(t, j)] >= 0.0, name=f"truck_load_nonneg_{t}_{j}")
            model.addConstr(w[(t, j)] <= truck.capacity_kg, name=f"truck_load_cap_{t}_{j}")
        for i in nodes:
            for j in customers:
                if i == j:
                    continue
                demand_j = instance.customer_map()[j].demand_kg
                model.addConstr(
                    w[(t, j)]
                    <= w[(t, i)] - demand_j * x_truck[(i, j, t)] + big_m * (1 - x_truck[(i, j, t)]),
                    name=f"truck_load_ub_{t}_{i}_{j}",
                )
                model.addConstr(
                    w[(t, j)]
                    >= w[(t, i)] - demand_j * x_truck[(i, j, t)] - big_m * (1 - x_truck[(i, j, t)]),
                    name=f"truck_load_lb_{t}_{i}_{j}",
                )

    for d in instance.drone_ids():
        for i in customers:
            model.addConstr(y_loaded[(d, i)] <= u_drone[(d, i)], name=f"loaded_ub_{d}_{i}")
            model.addConstr(y_loaded[(d, i)] >= 0.0, name=f"loaded_lb_{d}_{i}")
            for t in instance.truck_ids():
                model.addConstr(z1[(i, d, t)] <= u_truck[(t, i)], name=f"z1_truck_{i}_{d}_{t}")
                model.addConstr(z1[(i, d, t)] <= u_drone[(d, i)], name=f"z1_drone_{i}_{d}_{t}")
                model.addConstr(
                    z1[(i, d, t)] >= u_truck[(t, i)] + u_drone[(d, i)] - 1,
                    name=f"z1_lb_{i}_{d}_{t}",
                )
                model.addConstr(z2[(i, d, t)] <= u_truck[(t, i)], name=f"z2_truck_{i}_{d}_{t}")
                model.addConstr(z2[(i, d, t)] <= u_drone[(d, i)], name=f"z2_drone_{i}_{d}_{t}")
                model.addConstr(
                    z2[(i, d, t)] >= u_truck[(t, i)] + u_drone[(d, i)] - 1,
                    name=f"z2_lb_{i}_{d}_{t}",
                )


def add_energy(
    model,
    instance: InstanceData,
    x_drone: Dict[Tuple[int, int, int], Var],
    r: Dict[Tuple[int, int], Var],
    t_empty: Dict[Tuple[int, int, int], float],
    nodes: list[int],
    big_m: float,
) -> None:
    """Drone energy propagation with conservative loaded/empty split."""
    for d in instance.drone_ids():
        drone = instance.drones[d - 1]
        consumption = max(drone.energy_per_min_when_empty, drone.energy_per_min_when_loaded)
        model.addConstr(r[(d, 0)] == drone.max_battery_wh, name=f"drone_batt_start_{d}")
        for j in nodes:
            model.addConstr(r[(d, j)] >= 0.0, name=f"drone_batt_nonneg_{d}_{j}")
            model.addConstr(r[(d, j)] <= drone.max_battery_wh, name=f"drone_batt_cap_{d}_{j}")
        for i in nodes:
            for j in nodes:
                if i == j or j == 0:
                    continue
                model.addConstr(
                    r[(d, j)] <= r[(d, i)] - consumption * t_empty[(i, j, d)] + big_m * (1 - x_drone[(i, j, d)]),
                    name=f"drone_batt_ub_{d}_{i}_{j}",
                )
                model.addConstr(
                    r[(d, j)] >= r[(d, i)] - consumption * t_empty[(i, j, d)] - big_m * (1 - x_drone[(i, j, d)]),
                    name=f"drone_batt_lb_{d}_{i}_{j}",
                )


def add_timing(
    model,
    instance: InstanceData,
    x_truck: Dict[Tuple[int, int, int], Var],
    x_drone: Dict[Tuple[int, int, int], Var],
    a_truck: Dict[Tuple[int, int], Var],
    a_drone: Dict[Tuple[int, int], Var],
    l_truck: Dict[Tuple[int, int], Var],
    l_drone: Dict[Tuple[int, int], Var],
    tw_truck: Dict[Tuple[int, int], Var],
    tw_drone: Dict[Tuple[int, int], Var],
    u_truck: Dict[Tuple[int, int], Var],
    u_drone: Dict[Tuple[int, int], Var],
    z1: Dict[Tuple[int, int, int], Var],
    z2: Dict[Tuple[int, int, int], Var],
    t_truck: Dict[Tuple[int, int, int], float],
    t_empty: Dict[Tuple[int, int, int], float],
    nodes: list[int],
    big_m: float,
    swap_time_min: float,
    reload_time_min: float,
) -> None:
    customers = _customer_nodes(nodes)
    for t in instance.truck_ids():
        truck = instance.trucks[t - 1]
        model.addConstr(a_truck[(t, 0)] == 0.0, name=f"truck_start_{t}")
        model.addConstr(l_truck[(t, 0)] == 0.0, name=f"truck_start_depart_{t}")
        for i in nodes:
            service_i = instance.customer(i).service_time_min if i != 0 else 0.0
            model.addConstr(
                l_truck[(t, i)]
                >= a_truck[(t, i)]
                + service_i * u_truck.get((t, i), 0.0)
                + reload_time_min * gp.quicksum(z1[(i, d, t)] for d in instance.drone_ids())
                + swap_time_min * gp.quicksum(z2[(i, d, t)] for d in instance.drone_ids())
                + tw_truck[(t, i)],
                name=f"truck_depart_{t}_{i}",
            )
            model.addConstr(tw_truck[(t, i)] >= 0.0, name=f"truck_wait_nonneg_{t}_{i}")
            model.addConstr(tw_truck[(t, i)] <= truck.capacity_kg * 0 + big_m, name=f"truck_wait_ub_{t}_{i}")

    for d in instance.drone_ids():
        model.addConstr(a_drone[(d, 0)] == 0.0, name=f"drone_start_{d}")
        model.addConstr(l_drone[(d, 0)] == 0.0, name=f"drone_start_depart_{d}")
        for i in nodes:
            service_i = instance.customer(i).service_time_min if i != 0 else 0.0
            model.addConstr(
                l_drone[(d, i)] >= a_drone[(d, i)] + service_i * u_drone.get((d, i), 0.0) + tw_drone[(d, i)],
                name=f"drone_depart_{d}_{i}",
            )
            model.addConstr(tw_drone[(d, i)] >= 0.0, name=f"drone_wait_nonneg_{d}_{i}")

    for t in instance.truck_ids():
        for i in nodes:
            for j in nodes:
                if i == j or j == 0:
                    continue
                model.addConstr(
                    l_truck[(t, i)] + t_truck[(i, j, t)] - big_m * (1 - x_truck[(i, j, t)])
                    <= a_truck[(t, j)],
                    name=f"truck_arrival_lb_{t}_{i}_{j}",
                )

    for d in instance.drone_ids():
        for i in nodes:
            for j in nodes:
                if i == j or j == 0:
                    continue
                model.addConstr(
                    l_drone[(d, i)] + t_empty[(i, j, d)] - big_m * (1 - x_drone[(i, j, d)])
                    <= a_drone[(d, j)],
                    name=f"drone_arrival_lb_{d}_{i}_{j}",
                )

    for i in customers:
        for d in instance.drone_ids():
            model.addConstr(
                gp.quicksum(z1[(i, d, t)] + z2[(i, d, t)] for t in instance.truck_ids()) <= 1,
                name=f"sync_single_truck_{i}_{d}",
            )
            for t in instance.truck_ids():
                model.addConstr(
                    a_truck[(t, i)] - a_drone[(d, i)] <= swap_time_min + big_m * (1 - z1[(i, d, t)]),
                    name=f"sync_load_plus_{d}_{t}_{i}",
                )
                model.addConstr(
                    a_drone[(d, i)] - a_truck[(t, i)] <= reload_time_min + big_m * (1 - z2[(i, d, t)]),
                    name=f"sync_load_minus_{d}_{t}_{i}",
                )


def add_tardiness_linearization(
    model,
    instance: InstanceData,
    a_truck: Dict[Tuple[int, int], Var],
    a_drone: Dict[Tuple[int, int], Var],
    a_bar_truck: Dict[Tuple[int, int], Var],
    a_bar_drone: Dict[Tuple[int, int], Var],
    u_truck: Dict[Tuple[int, int], Var],
    u_drone: Dict[Tuple[int, int], Var],
    tardiness: Dict[int, Var],
    nodes: list[int],
) -> None:
    customers = _customer_nodes(nodes)
    max_shift = instance.constants.max_shift_minutes
    for t in instance.truck_ids():
        for i in customers:
            model.addConstr(a_bar_truck[(t, i)] <= a_truck[(t, i)], name=f"mc_atruck_ub_{t}_{i}")
            model.addConstr(a_bar_truck[(t, i)] <= max_shift * u_truck[(t, i)], name=f"mc_atruck_active_{t}_{i}")
            model.addConstr(
                a_bar_truck[(t, i)] >= a_truck[(t, i)] - max_shift * (1 - u_truck[(t, i)]),
                name=f"mc_atruck_lb_{t}_{i}",
            )
    for d in instance.drone_ids():
        for i in customers:
            model.addConstr(a_bar_drone[(d, i)] <= a_drone[(d, i)], name=f"mc_adrone_ub_{d}_{i}")
            model.addConstr(a_bar_drone[(d, i)] <= max_shift * u_drone[(d, i)], name=f"mc_adrone_active_{d}_{i}")
            model.addConstr(
                a_bar_drone[(d, i)] >= a_drone[(d, i)] - max_shift * (1 - u_drone[(d, i)]),
                name=f"mc_adrone_lb_{d}_{i}",
            )
    for i in customers:
        model.addConstr(
            tardiness[i]
            >= gp.quicksum(a_bar_truck[(t, i)] for t in instance.truck_ids())
            + gp.quicksum(a_bar_drone[(d, i)] for d in instance.drone_ids())
            - instance.customer_map()[i].ub,
            name=f"tardiness_ge_{i}",
        )
        model.addConstr(tardiness[i] >= 0.0, name=f"tardiness_nonneg_{i}")


def add_priority_ordering(
    model,
    instance: InstanceData,
    a_bar_truck: Dict[Tuple[int, int], Var],
    a_bar_drone: Dict[Tuple[int, int], Var],
    nodes: list[int],
    big_m: float,
) -> None:
    """Priority structure constraints as precedence by class."""
    customers = _customer_nodes(nodes)
    if not customers:
        return

    precedence_pairs = {("high", "medium"), ("high", "low"), ("medium", "low")}
    for i in customers:
        for j in customers:
            if i == j:
                continue
            ci = instance.customer_map()[i].priority
            cj = instance.customer_map()[j].priority
            if (ci, cj) not in precedence_pairs:
                continue
            model.addConstr(
                gp.quicksum(a_bar_truck[(t, i)] for t in instance.truck_ids())
                + gp.quicksum(a_bar_drone[(d, i)] for d in instance.drone_ids())
                <= gp.quicksum(a_bar_truck[(t, j)] for t in instance.truck_ids())
                + gp.quicksum(a_bar_drone[(d, j)] for d in instance.drone_ids())
                + big_m,
                name=f"priority_order_{ci}_{cj}_{i}_{j}",
            )


def add_time_windows_and_shifts(
    model,
    instance: InstanceData,
    a_truck: Dict[Tuple[int, int], Var],
    a_drone: Dict[Tuple[int, int], Var],
    u_truck: Dict[Tuple[int, int], Var],
    u_drone: Dict[Tuple[int, int], Var],
    nodes: list[int],
    enforce_subtours: bool,
    x_truck: Dict[Tuple[int, int, int], Var],
    x_drone: Dict[Tuple[int, int, int], Var],
    big_m: float = 10_000.0,
) -> None:
    customers = _customer_nodes(nodes)
    for i in customers:
        c = instance.customer_map()[i]
        for t in instance.truck_ids():
            model.addConstr(a_truck[(t, i)] <= c.ub + big_m * (1 - u_truck[(t, i)]), name=f"tw_truck_ub_{t}_{i}")
            model.addConstr(a_truck[(t, i)] >= c.lb - big_m * (1 - u_truck[(t, i)]), name=f"tw_truck_lb_{t}_{i}")
        for d in instance.drone_ids():
            model.addConstr(a_drone[(d, i)] <= c.ub + big_m * (1 - u_drone[(d, i)]), name=f"tw_drone_ub_{d}_{i}")
            model.addConstr(a_drone[(d, i)] >= c.lb - big_m * (1 - u_drone[(d, i)]), name=f"tw_drone_lb_{d}_{i}")

    shift_limit = instance.constants.max_shift_minutes
    for t in instance.truck_ids():
        model.addConstr(gp.quicksum(a_truck[(t, i)] for i in customers) <= shift_limit, name=f"truck_shift_{t}")
    for d in instance.drone_ids():
        model.addConstr(gp.quicksum(a_drone[(d, i)] for i in customers) <= shift_limit, name=f"drone_shift_{d}")

    if enforce_subtours:
        _mtz_subtour(model, instance, x_truck, nodes, "T", big_m)
        _mtz_subtour(model, instance, x_drone, nodes, "D", big_m)


def _mtz_subtour(
    model,
    instance: InstanceData,
    x: Dict[Tuple[int, int, int], Var],
    nodes: list[int],
    suffix: str,
    big_m: float,
) -> None:
    customers = _customer_nodes(nodes)
    if not customers:
        return
    vehicle_ids = instance.truck_ids() if suffix == "T" else instance.drone_ids()
    n = len(customers)
    for v in vehicle_ids:
        u = {i: model.addVar(lb=0.0, ub=n, name=f"mtz_{suffix}_{v}_{i}") for i in customers}
        for i in customers:
            for j in customers:
                if i == j:
                    continue
                model.addConstr(
                    u[i] - u[j] + n * x[(i, j, v)] <= n - 1 + big_m * (1 - x[(i, j, v)]),
                    name=f"mtz_{suffix}_{v}_{i}_{j}",
                )


def _customer_nodes(nodes: list[int]) -> list[int]:
    return [i for i in nodes if i != 0]

