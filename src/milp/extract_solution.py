"""MILP solution extractor converting gurobi variable values into SolutionData."""
from __future__ import annotations

from typing import Dict, Tuple

from ..data_models import SolutionData, SyncEvent


def _value(var) -> float:
    if var is None:
        return 0.0
    if hasattr(var, "x"):
        return float(var.x)
    if hasattr(var, "varValue"):
        return float(var.varValue)
    if hasattr(var, "value"):
        return float(var.value())
    return float(var)


def extract_solution(instance, artifacts, status: str, runtime_seconds: float = 0.0) -> SolutionData:
    model = artifacts.model
    v = artifacts.variables
    node_count = [i for i in range(0, instance.num_customers + 1)]

    # rounding with tolerance
    def bin01(v):
        return 1 if _value(v) >= 0.5 else 0

    sol = SolutionData(
        instance_name=instance.name,
        status=status,
        objective=float(model.ObjVal) if hasattr(model, "ObjVal") else 0.0,
        components={},
        run_time_seconds=runtime_seconds,
    )

    for k, var in v["x_truck"].items():
        sol.x_truck[k] = bin01(var)
    for k, var in v["x_drone"].items():
        sol.x_drone[k] = bin01(var)
    for k, var in v["u_truck"].items():
        sol.u_truck[k] = bin01(var)
    for k, var in v["u_drone"].items():
        sol.u_drone[k] = bin01(var)
    for k, var in v["z1"].items():
        sol.z1[k] = bin01(var)
    for k, var in v["z2"].items():
        sol.z2[k] = bin01(var)
    for k, var in v["y_loaded"].items():
        sol.y_loaded[k] = bin01(var)
    for k, var in v["r"].items():
        sol.battery_drone[k] = _value(var)
    for k, var in v["a_truck"].items():
        sol.a_truck[k] = _value(var)
    for k, var in v["l_truck"].items():
        sol.l_truck[k] = _value(var)
    for k, var in v["a_drone"].items():
        sol.a_drone[k] = _value(var)
    for k, var in v["l_drone"].items():
        sol.l_drone[k] = _value(var)
    for k, var in v["tw_truck"].items():
        sol.waiting_truck[k] = _value(var)
    for k, var in v["tw_drone"].items():
        sol.waiting_drone[k] = _value(var)
    for k, var in v["w"].items():
        sol.load_truck[k] = _value(var)
    for k, var in v["tardiness"].items():
        sol.tardiness[k] = _value(var)

    trucks = [t.truck_id for t in instance.trucks]
    drones = [d.drone_id for d in instance.drones]

    for i in node_count:
        for d in drones:
            for t in trucks:
                z1_val = sol.z1.get((i, d, t), 0)
                z2_val = sol.z2.get((i, d, t), 0)
                if z1_val > 0:
                    sol.sync_events.append(SyncEvent(i, d, t, "parcel_reload"))
                if z2_val > 0:
                    sol.sync_events.append(SyncEvent(i, d, t, "battery_swap"))

    # Objective decomposition from model expressions
    for name, expr in (artifacts.components or {}).items():
        if hasattr(expr, "getValue"):
            sol.components[name] = float(expr.getValue())
        elif hasattr(expr, "value"):
            sol.components[name] = float(expr.value())
        else:
            sol.components[name] = 0.0

    if hasattr(model, "ObjVal"):
        sol.components["objective_total"] = float(model.ObjVal) if hasattr(model, "ObjVal") else 0.0

    # derive routes for traceability
    route_arc_map_t = {
        t: {(i, j): vv for (i, j, tt), vv in sol.x_truck.items() if tt == t and vv > 0.5}
        for t in trucks
    }
    route_arc_map_d = {
        d: {(i, j): vv for (i, j, dd), vv in sol.x_drone.items() if dd == d and vv > 0.5}
        for d in drones
    }

    def _build_route(start: int, arc_map: Dict[Tuple[int, int], int]) -> list[int]:
        current = start
        route = [start]
        seen = {start}
        while True:
            nxt_candidates = [j for (i, j), vv in arc_map.items() if i == current]
            if not nxt_candidates:
                break
            nxt = nxt_candidates[0]
            route.append(nxt)
            if nxt == start or nxt in seen:
                break
            seen.add(nxt)
            current = nxt
            if len(route) > instance.num_customers + 2:
                break
        if route[-1] != 0:
            route.append(0)
        return route

    for t in trucks:
        sol.truck_routes[t] = _build_route(0, route_arc_map_t[t])
    for d in drones:
        sol.drone_routes[d] = _build_route(0, route_arc_map_d[d])

    return sol
