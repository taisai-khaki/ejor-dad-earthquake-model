"""MILP model builder for the UTDRP-DP exact formulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from ..data_models import InstanceData
from ..distance_utils import travel_time_from_distance
from ..parameters import SearchConfig
from . import constraints
from .backend import gp, set_active_backend
from .objective import build_objective


@dataclass
class MilpArtifacts:
    model: object
    variables: Dict[str, Dict]
    metadata: Dict[str, object]
    components: Dict[str, object]


def _build_distance_matrices(instance: InstanceData) -> tuple[
    Dict[Tuple[int, int], float],
    Dict[Tuple[int, int], float],
]:
    coords = instance.coordinate_map()
    nodes = instance.all_nodes()
    manhattan: Dict[Tuple[int, int], float] = {}
    euclid: Dict[Tuple[int, int], float] = {}
    for i in nodes:
        xi, yi = coords[i]
        for j in nodes:
            xj, yj = coords[j]
            if i == j:
                manhattan[(i, j)] = 0.0
                euclid[(i, j)] = 0.0
            else:
                manhattan[(i, j)] = abs(xi - xj) + abs(yi - yj)
                euclid[(i, j)] = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
    return manhattan, euclid


def _build_travel_times(
    instance: InstanceData,
    manhattan: Dict[Tuple[int, int], float],
    euclid: Dict[Tuple[int, int], float],
    nodes: list[int],
) -> tuple[
    Dict[Tuple[int, int, int], float],
    Dict[Tuple[int, int, int], float],
    Dict[Tuple[int, int, int], float],
]:
    t_truck: Dict[Tuple[int, int, int], float] = {}
    t_empty: Dict[Tuple[int, int, int], float] = {}
    t_loaded: Dict[Tuple[int, int, int], float] = {}
    for i in nodes:
        for j in nodes:
            for t in instance.truck_ids():
                t_truck[(i, j, t)] = travel_time_from_distance(manhattan[(i, j)], instance.trucks[t - 1].speed_kmph)
            for d in instance.drone_ids():
                base = travel_time_from_distance(euclid[(i, j)], instance.drones[d - 1].speed_kmph)
                t_empty[(i, j, d)] = base
                t_loaded[(i, j, d)] = base * 1.05
    return t_truck, t_empty, t_loaded


def build_model(instance: InstanceData, config: SearchConfig) -> MilpArtifacts:
    set_active_backend(config.milp.solver_backend)

    model = gp.Model(f"utdrp_dp_{instance.name}")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = config.milp.time_limit_seconds
    model.Params.MIPGap = config.milp.mip_gap
    model.Params.Threads = max(1, int(config.milp.threads))

    nodes = instance.all_nodes()
    manhattan, euclid = _build_distance_matrices(instance)
    t_truck, t_empty, t_loaded = _build_travel_times(instance, manhattan, euclid, nodes)

    x_truck = {
        (i, j, t): model.addVar(vtype=gp.GRB.BINARY, name=f"xT_{t}_{i}_{j}")
        for t in instance.truck_ids()
        for i in nodes
        for j in nodes
    }
    x_drone = {
        (i, j, d): model.addVar(vtype=gp.GRB.BINARY, name=f"xD_{d}_{i}_{j}")
        for d in instance.drone_ids()
        for i in nodes
        for j in nodes
    }

    u_truck = {(t, j): model.addVar(vtype=gp.GRB.BINARY, name=f"uT_{t}_{j}") for t in instance.truck_ids() for j in instance.customer_ids}
    u_drone = {(d, j): model.addVar(vtype=gp.GRB.BINARY, name=f"uD_{d}_{j}") for d in instance.drone_ids() for j in instance.customer_ids}

    z1 = {(i, d, t): model.addVar(vtype=gp.GRB.BINARY, name=f"zLoad_{d}_{t}_{i}") for t in instance.truck_ids() for d in instance.drone_ids() for i in nodes}
    z2 = {(i, d, t): model.addVar(vtype=gp.GRB.BINARY, name=f"zSwap_{d}_{t}_{i}") for t in instance.truck_ids() for d in instance.drone_ids() for i in nodes}
    y_loaded = {(d, i): model.addVar(vtype=gp.GRB.BINARY, name=f"yL_{d}_{i}") for d in instance.drone_ids() for i in instance.customer_ids}

    w = {(t, i): model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name=f"load_{t}_{i}") for t in instance.truck_ids() for i in nodes}
    r = {(d, i): model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name=f"batt_{d}_{i}") for d in instance.drone_ids() for i in nodes}
    a_truck = {(t, i): model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=instance.constants.max_shift_minutes, name=f"aT_{t}_{i}") for t in instance.truck_ids() for i in nodes}
    a_drone = {(d, i): model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=instance.constants.max_shift_minutes, name=f"aD_{d}_{i}") for d in instance.drone_ids() for i in nodes}
    l_truck = {(t, i): model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=instance.constants.max_shift_minutes, name=f"lT_{t}_{i}") for t in instance.truck_ids() for i in nodes}
    l_drone = {(d, i): model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=instance.constants.max_shift_minutes, name=f"lD_{d}_{i}") for d in instance.drone_ids() for i in nodes}
    tw_truck = {(t, i): model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name=f"wT_{t}_{i}") for t in instance.truck_ids() for i in nodes}
    tw_drone = {(d, i): model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name=f"wD_{d}_{i}") for d in instance.drone_ids() for i in nodes}
    a_bar_truck = {(t, i): model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name=f"aBarT_{t}_{i}") for t in instance.truck_ids() for i in instance.customer_ids}
    a_bar_drone = {(d, i): model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name=f"aBarD_{d}_{i}") for d in instance.drone_ids() for i in instance.customer_ids}
    tardiness = {i: model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name=f"T_{i}") for i in instance.customer_ids}

    # Disallow self loops directly
    for t in instance.truck_ids():
        for i in nodes:
            model.addConstr(x_truck[(i, i, t)] == 0, name=f"no_loop_t_{t}_{i}")
    for d in instance.drone_ids():
        for i in nodes:
            model.addConstr(x_drone[(i, i, d)] == 0, name=f"no_loop_d_{d}_{i}")

    constraints.add_return_to_depot(model, instance, x_truck, x_drone, nodes)
    constraints.add_service_and_flow(model, instance, x_truck, x_drone, u_truck, u_drone, nodes)
    constraints.add_demand_and_load(
        model,
        instance,
        x_truck,
        u_truck,
        y_loaded,
        w,
        z1,
        z2,
        nodes,
        u_drone,
        config.constraints.big_m,
    )
    constraints.add_energy(
        model,
        instance,
        x_drone,
        r,
        t_empty,
        nodes,
        config.constraints.big_m,
    )
    constraints.add_timing(
        model,
        instance,
        x_truck,
        x_drone,
        a_truck,
        a_drone,
        l_truck,
        l_drone,
        tw_truck,
        tw_drone,
        u_truck,
        u_drone,
        z1,
        z2,
        t_truck,
        t_empty,
        nodes,
        config.constraints.big_m,
        instance.constants.swap_time_s / 60.0,
        instance.constants.reload_time_s / 60.0,
    )
    constraints.add_tardiness_linearization(
        model,
        instance,
        a_truck,
        a_drone,
        a_bar_truck,
        a_bar_drone,
        u_truck,
        u_drone,
        tardiness,
        nodes,
    )
    constraints.add_time_windows_and_shifts(
        model,
        instance,
        a_truck,
        a_drone,
        u_truck,
        u_drone,
        nodes,
        config.constraints.enforce_subtour_elim,
        x_truck,
        x_drone,
        config.constraints.big_m,
    )
    constraints.add_priority_ordering(
        model,
        instance,
        a_bar_truck,
        a_bar_drone,
        nodes,
        config.constraints.big_m,
    )

    components = build_objective(
        model,
        instance,
        x_truck,
        x_drone,
        y_loaded,
        tw_truck,
        tw_drone,
        tardiness,
        t_truck,
        t_empty,
        t_loaded,
        nodes,
        u_truck,
        u_drone,
        a_bar_truck,
        a_bar_drone,
    )

    return MilpArtifacts(
        model=model,
        variables={
            "x_truck": x_truck,
            "x_drone": x_drone,
            "u_truck": u_truck,
            "u_drone": u_drone,
            "z1": z1,
            "z2": z2,
            "y_loaded": y_loaded,
            "w": w,
            "r": r,
            "a_truck": a_truck,
            "a_drone": a_drone,
            "l_truck": l_truck,
            "l_drone": l_drone,
            "tw_truck": tw_truck,
            "tw_drone": tw_drone,
            "a_bar_truck": a_bar_truck,
            "a_bar_drone": a_bar_drone,
            "tardiness": tardiness,
        },
        metadata={"nodes": nodes, "t_truck": t_truck, "t_empty": t_empty, "t_loaded": t_loaded},
        components=components,
    )

