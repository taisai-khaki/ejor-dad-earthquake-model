"""Build high-priority delivery-speed statistics for the M1--M8 benchmark."""
from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..distance_utils import euclidean_distance, travel_time_from_distance
from ..feasibility import is_feasible
from ..heuristics.nils import _repair_infeasible_solution, evaluate_solution
from ..heuristics.nils_r import compute_priority_metrics
from ..instance_generator import load_instance_json
from ..parameters import SearchConfig, load_and_build_config
from .heuristic_study import _collect_row, _run_method, _scenario_config
from .run_alns_comparison import _scenario_from_row, _with_alns_settings


MAIN_METHODS = [
    "truck_only",
    "nils",
    "nils_r",
]
ALNS_METHOD = "feasibility_aware_alns"
METHOD_ORDER = {
    "truck_only": ("M1 Truck-only", 1),
    "nils": ("M5 NILS", 5),
    "nils_r": ("M9 NILS-R", 9),
    ALNS_METHOD: ("M8 Feasibility-aware ALNS", 8),
}
ALL_METHODS = MAIN_METHODS + [ALNS_METHOD]
OUTPUT_COLUMNS = [
    "method",
    "size",
    "feasible_runs",
    "mean_high_priority_delivery_time",
    "median_high_priority_delivery_time",
    "p90_high_priority_delivery_time",
    "mean_high_priority_tardiness",
    "median_high_priority_tardiness",
    "p90_high_priority_tardiness",
    "max_high_priority_tardiness",
    "high_priority_on_time_rate",
    "high_priority_late_rate",
    "mean_high_priority_minutes_early_when_on_time",
    "mean_high_priority_delivery_time_relative_to_deadline",
    "high_priority_served_by_drone_share",
]


def _log(path: Path, message: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _finite_values(values: Iterable[object]) -> np.ndarray:
    out = []
    for value in values:
        if pd.notna(value):
            number = float(value)
            if math.isfinite(number):
                out.append(number)
    return np.asarray(out, dtype=float)


def _mean(values: Iterable[object]) -> float:
    arr = _finite_values(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def _median(values: Iterable[object]) -> float:
    arr = _finite_values(values)
    return float(np.median(arr)) if arr.size else float("nan")


def _p90(values: Iterable[object]) -> float:
    arr = _finite_values(values)
    return float(np.percentile(arr, 90.0)) if arr.size else float("nan")


def _scenario_key(row: pd.Series) -> str:
    return str(row["scenario_id"])


def _delivery_times(instance, solution) -> dict[int, float]:
    customer_ids = {customer.node_id for customer in instance.customers}
    delivered: dict[int, float] = {}
    for (_, customer_id), arrival in solution.a_truck.items():
        if customer_id in customer_ids:
            delivered[customer_id] = max(delivered.get(customer_id, 0.0), float(arrival))
    for (_, customer_id), arrival in solution.a_drone.items():
        if customer_id in customer_ids:
            delivered[customer_id] = max(delivered.get(customer_id, 0.0), float(arrival))
    return delivered


def _customer_rows(instance, solution, method_name: str) -> list[dict[str, object]]:
    delivered = _delivery_times(instance, solution)
    customer_map = instance.customer_map()
    drone_customers = solution.served_by_drone()
    high_customers = [customer for customer in instance.customers if customer.priority == "high"]
    missing = [customer.node_id for customer in high_customers if customer.node_id not in delivered]
    if missing:
        raise ValueError(f"Missing delivery times for high-priority customers {missing}")

    rows = []
    for customer in high_customers:
        delivery_time = float(delivered[customer.node_id])
        deadline = float(customer.ub)
        on_time = delivery_time <= deadline + 1e-9
        relative = delivery_time / deadline if deadline > 1e-12 else float("nan")
        rows.append(
            {
                "instance_id": instance.name,
                "scenario_id": str(instance.metadata.get("scenario_id", "")),
                "n": int(instance.num_customers),
                "method_name": method_name,
                "customer_id": int(customer.node_id),
                "delivery_time": delivery_time,
                "deadline": deadline,
                "tardiness": max(0.0, delivery_time - deadline),
                "on_time": bool(on_time),
                "minutes_early_when_on_time": deadline - delivery_time if on_time else float("nan"),
                "delivery_time_relative_to_deadline": relative,
                "served_by_drone": int(customer.node_id in drone_customers),
            }
        )
    return rows


def _event_nodes(solution, drone_id: int) -> tuple[object, object, object, object]:
    launch_events = [
        (node, truck)
        for (node, drone, truck), value in solution.z1.items()
        if drone == drone_id and float(value) >= 0.5
    ]
    recovery_events = [
        (node, truck)
        for (node, drone, truck), value in solution.z2.items()
        if drone == drone_id and float(value) >= 0.5
    ]
    launch_node, launch_truck = launch_events[0] if launch_events else ("", "")
    recovery_node, recovery_truck = recovery_events[-1] if recovery_events else ("", "")
    return launch_node, recovery_node, launch_truck, recovery_truck


def _sortie_energy(instance, solution, drone_id: int, route: list[int]) -> float:
    if len(route) < 2:
        return float("nan")
    coordinates = instance.coordinate_map()
    drone = instance.drones[drone_id - 1]
    energy = 0.0
    for left, right in zip(route, route[1:]):
        duration = travel_time_from_distance(
            euclidean_distance(coordinates[left], coordinates[right]), drone.speed_kmph
        )
        burn = drone.energy_per_min_when_loaded if right != 0 else drone.energy_per_min_when_empty
        energy += duration * burn
    return float(energy)


def _customer_detail_rows(instance, solution, method_name: str, scenario_class_id: str, seed: int) -> list[dict[str, object]]:
    delivered = _delivery_times(instance, solution)
    customer_map = instance.customer_map()
    truck_assignments = {
        customer_id: truck_id
        for (truck_id, customer_id), value in solution.u_truck.items()
        if float(value) >= 0.5
    }
    drone_assignments = {
        customer_id: drone_id
        for (drone_id, customer_id), value in solution.u_drone.items()
        if float(value) >= 0.5
    }
    method_label = METHOD_ORDER[method_name][0]
    rows: list[dict[str, object]] = []
    for customer in instance.customers:
        customer_id = int(customer.node_id)
        delivery_time = float(delivered.get(customer_id, float("nan")))
        deadline = float(customer.ub)
        on_time = bool(np.isfinite(delivery_time) and delivery_time <= deadline + 1e-9)
        truck_id = truck_assignments.get(customer_id)
        drone_id = drone_assignments.get(customer_id)
        is_drone = drone_id is not None
        service_mode = "drone" if is_drone else "truck"
        serving_id = drone_id if is_drone else truck_id
        launch_node = ""
        recovery_node = ""
        launch_truck = ""
        recovery_truck = ""
        battery_before = float("nan")
        battery_after = float("nan")
        battery_slack = float("nan")
        sortie_energy = float("nan")
        sync_wait = 0.0
        recovery_type = ""
        if is_drone:
            route = list(solution.drone_routes.get(int(drone_id), []))
            launch_node = route[0] if route else ""
            recovery_node = route[-1] if route else ""
            event_launch, event_recovery, launch_truck, recovery_truck = _event_nodes(solution, int(drone_id))
            if event_launch != "":
                launch_node = event_launch
            if event_recovery != "":
                recovery_node = event_recovery
            drone = instance.drones[int(drone_id) - 1]
            sortie_energy = _sortie_energy(instance, solution, int(drone_id), route)
            battery_before = float(solution.battery_drone.get((int(drone_id), launch_node), drone.max_battery_wh))
            battery_after = float(solution.battery_drone.get((int(drone_id), recovery_node), drone.max_battery_wh - sortie_energy))
            battery_slack = battery_after
            if recovery_node == 0:
                recovery_type = "depot"
            elif launch_truck != "" and recovery_truck != "" and launch_truck != recovery_truck:
                recovery_type = "cross_truck"
            else:
                recovery_type = "same_truck"
            sync_nodes = {customer_id, launch_node, recovery_node}
            sync_wait = float(
                sum(value for (vehicle, node), value in solution.waiting_truck.items()
                    if vehicle == launch_truck and node in sync_nodes)
                + sum(value for (vehicle, node), value in solution.waiting_drone.items()
                      if vehicle == int(drone_id) and node in sync_nodes)
            )
        tardiness = max(0.0, delivery_time - deadline) if np.isfinite(delivery_time) else float("nan")
        rows.append(
            {
                "instance_id": instance.name,
                "scenario_class_id": scenario_class_id,
                "seed": int(seed),
                "method": method_label,
                "size": int(instance.num_customers),
                "customer_id": customer_id,
                "priority_class": customer.priority,
                "priority_weight": float(instance.priority_penalties[customer.priority]),
                "service_mode": service_mode,
                "serving_vehicle_id": serving_id if serving_id is not None else "",
                "delivery_time_L_i": delivery_time,
                "window_start_e_i": float(customer.lb),
                "deadline_l_i": deadline,
                "tardiness": tardiness,
                "weighted_tardiness": tardiness * float(instance.priority_penalties[customer.priority])
                if np.isfinite(tardiness) else float("nan"),
                "on_time_indicator": int(on_time),
                "minutes_early_if_on_time": deadline - delivery_time if on_time else float("nan"),
                "is_high_priority": int(customer.priority == "high"),
                "is_drone_served": int(is_drone),
                "truck_route_id": truck_id if truck_id is not None else "",
                "drone_id": drone_id if drone_id is not None else "",
                "launch_node": launch_node,
                "recovery_node": recovery_node,
                "recovery_type": recovery_type,
                "sync_truck_id": recovery_truck if recovery_truck != "" else launch_truck,
                "battery_before_sortie": battery_before,
                "battery_after_sortie": battery_after,
                "battery_slack_after_sortie": battery_slack,
                "sortie_energy": sortie_energy,
                "sync_wait_time": sync_wait,
            }
        )
    return rows


def _force_feasible_solution(instance, solution, config: SearchConfig, method_name: str):
    if is_feasible(instance, solution):
        return solution
    experiment = dict(config.experiment or {})
    max_steps = max(1, int(experiment.get("study_output_repair_max_steps", 8)))
    repaired = _repair_infeasible_solution(instance, solution, max_steps=max(max_steps, instance.num_customers))
    if repaired is not None and is_feasible(instance, repaired):
        return repaired
    fallback = _run_method(instance, config, "truck_only")
    if is_feasible(instance, fallback):
        fallback.status = f"{method_name}|fallback_truck_only"
        return fallback
    return solution


def _run_one(instance, scenario: dict[str, object], method_name: str, config: SearchConfig):
    scenario_config = _scenario_config(config, scenario)
    start = time.time()
    solution = _run_method(instance, scenario_config, method_name)
    solution = _force_feasible_solution(instance, solution, scenario_config, method_name)
    evaluate_solution(instance, solution)
    elapsed = time.time() - start
    return solution, elapsed, bool(is_feasible(instance, solution))


def _load_partial(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns].copy()


def _write_partial(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10f")


def _aggregate(run_status: pd.DataFrame, customer_rows: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for method_name, (label, method_order) in METHOD_ORDER.items():
        method_status = run_status[run_status["method_name"].eq(method_name)].copy()
        for size in ["All", "15", "50", "75", "100"]:
            if size == "All":
                status_subset = method_status
                customer_subset = customer_rows[customer_rows["method_name"].eq(method_name)]
            else:
                status_subset = method_status[method_status["n"].eq(int(size))]
                customer_subset = customer_rows[
                    customer_rows["method_name"].eq(method_name) & customer_rows["n"].eq(int(size))
                ]
            feasible_status = status_subset[status_subset["feasible"].astype(bool)]
            feasible_runs = int(len(feasible_status))
            on_time = _finite_values(customer_subset["on_time"] if not customer_subset.empty else [])
            late = 1.0 - float(np.mean(on_time)) if on_time.size else float("nan")
            on_time_rate = float(np.mean(on_time)) if on_time.size else float("nan")
            max_hpt_by_run = (
                customer_subset.groupby(["instance_id", "method_name"])["tardiness"].max()
                if not customer_subset.empty
                else pd.Series(dtype=float)
            )
            summaries.append(
                {
                    "_method_order": method_order,
                    "_size_order": -1 if size == "All" else int(size),
                    "method": label,
                    "size": size,
                    "feasible_runs": feasible_runs,
                    "mean_high_priority_delivery_time": _mean(customer_subset["delivery_time"]),
                    "median_high_priority_delivery_time": _median(customer_subset["delivery_time"]),
                    "p90_high_priority_delivery_time": _p90(customer_subset["delivery_time"]),
                    "mean_high_priority_tardiness": _mean(customer_subset["tardiness"]),
                    "median_high_priority_tardiness": _median(customer_subset["tardiness"]),
                    "p90_high_priority_tardiness": _p90(customer_subset["tardiness"]),
                    "max_high_priority_tardiness": _mean(max_hpt_by_run),
                    "high_priority_on_time_rate": on_time_rate,
                    "high_priority_late_rate": late,
                    "mean_high_priority_minutes_early_when_on_time": _mean(
                        customer_subset["minutes_early_when_on_time"]
                    ),
                    "mean_high_priority_delivery_time_relative_to_deadline": _mean(
                        customer_subset["delivery_time_relative_to_deadline"]
                    ),
                    "high_priority_served_by_drone_share": _mean(customer_subset["served_by_drone"]),
                }
            )
    table = pd.DataFrame(summaries).sort_values(["_method_order", "_size_order"]).reset_index(drop=True)
    return table[OUTPUT_COLUMNS]


def run_pipeline(
    *,
    main_config_path: Path,
    alns_config_path: Path,
    source_output_dir: Path,
    output_dir: Path,
    limit: int | None = None,
    start_index: int = 0,
    end_index: int | None = None,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    main_config = load_and_build_config(str(main_config_path))
    alns_config = _with_alns_settings(load_and_build_config(str(alns_config_path)))
    source_path = source_output_dir / "tables" / "heuristic_study_runs.csv"
    source = pd.read_csv(source_path)
    meta = (
        source.sort_values(["scenario_id", "instance_id"])
        .drop_duplicates(["scenario_id", "instance_id"])
        .reset_index(drop=True)
    )
    if limit is not None and limit > 0:
        meta = meta.head(int(limit)).copy()
    meta = meta.iloc[int(start_index):int(end_index) if end_index is not None else None].reset_index(drop=True)

    raw_dir = source_output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "priority_service_speed_progress.log"
    run_path = output_dir / "run_status_partial.csv"
    customer_path = output_dir / "customer_delivery_observations_partial.csv"
    customer_detail_path = output_dir / "customer_service_detail.csv"
    reconstruction_path = output_dir / "nils_r_reconstruction_log.csv"
    decision_path = output_dir / "drone_decision_log.csv"
    run_columns = [
        "instance_id", "scenario_id", "n", "method_name", "feasible", "cpu_time_seconds", "error",
        "objective", "priority_tardiness_cost", "high_priority_tardiness",
        "max_high_priority_tardiness", "p90_high_priority_tardiness",
        "drone_served_customers", "drone_share", "truck_travel_cost", "drone_travel_cost",
        "waiting_sync_cost", "truck_route_distance_km", "drone_flight_distance_km",
        "total_truck_waiting_time", "total_drone_waiting_time", "average_synchronization_delay",
        "maximum_synchronization_delay", "average_battery_usage_ratio", "maximum_battery_usage_ratio",
        "sorties_near_endurance_limit", "eligible_not_assigned_due_battery", "makespan_minutes",
        "number_of_drone_launches", "number_of_drone_retrievals",
    ]
    customer_detail_columns = [
        "instance_id", "scenario_class_id", "seed", "method", "size", "customer_id",
        "priority_class", "priority_weight", "service_mode", "serving_vehicle_id",
        "delivery_time_L_i", "window_start_e_i", "deadline_l_i", "tardiness", "weighted_tardiness",
        "on_time_indicator", "minutes_early_if_on_time", "is_high_priority", "is_drone_served",
        "truck_route_id", "drone_id", "launch_node", "recovery_node", "recovery_type",
        "sync_truck_id", "battery_before_sortie", "battery_after_sortie",
        "battery_slack_after_sortie", "sortie_energy", "sync_wait_time",
    ]
    decision_columns = [
        "instance_id", "seed", "method", "iteration", "move_type", "customer_id",
        "priority_class", "action", "from_mode", "to_mode", "launch_node", "recovery_node",
        "recovery_type", "sync_truck_id", "delta_Z", "delta_priority_tardiness_cost",
        "delta_high_priority_tardiness", "delta_max_high_priority_tardiness", "delta_truck_cost",
        "delta_drone_cost", "delta_sync_wait", "battery_slack_after_move", "accepted", "rejection_reason",
    ]
    customer_columns = [
        "instance_id",
        "scenario_id",
        "n",
        "method_name",
        "customer_id",
        "delivery_time",
        "deadline",
        "tardiness",
        "on_time",
        "minutes_early_when_on_time",
        "delivery_time_relative_to_deadline",
        "served_by_drone",
    ]
    reconstruction_columns = [
        "instance_id", "seed", "size", "reconstruction_attempt", "trigger_reason",
        "num_drone_customers_before", "num_candidates_evaluated", "num_customers_returned_to_truck",
        "customer_ids_returned", "Z_before", "Z_after", "HPT_before", "HPT_after",
        "MaxHPT_before", "MaxHPT_after", "P90HPT_before", "P90HPT_after",
        "drone_share_before", "drone_share_after", "active_drones_before", "active_drones_after",
        "accepted", "rejection_reason", "cpu_reconstruction",
    ]
    run_status = _load_partial(run_path, run_columns)
    customer_rows = _load_partial(customer_path, customer_columns)
    customer_detail_rows = _load_partial(customer_detail_path, customer_detail_columns)
    decision_rows = _load_partial(decision_path, decision_columns)
    reconstruction_rows = _load_partial(reconstruction_path, reconstruction_columns)
    selected_methods = list(methods) if methods else list(ALL_METHODS)
    unknown_methods = sorted(set(selected_methods) - set(ALL_METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}")
    completed = set(zip(run_status["instance_id"].astype(str), run_status["method_name"].astype(str)))
    _log(log_path, f"Priority service-speed study start | instances={len(meta)} | methods={selected_methods} | completed_pairs={len(completed)}")

    for index, source_row in meta.iterrows():
        instance_id = str(source_row["instance_id"])
        instance = load_instance_json(raw_dir / f"{instance_id}.json")
        scenario = _scenario_from_row(source_row)
        _log(log_path, f"[{index + 1}/{len(meta)}] {instance_id} | n={instance.num_customers} start")
        instance_run_rows = []
        instance_customer_rows = []
        instance_customer_detail_rows = []
        instance_decision_rows = []
        for method_name in selected_methods:
            key = (instance_id, method_name)
            if key in completed:
                continue
            config = alns_config if method_name == ALNS_METHOD else main_config
            try:
                solution, elapsed, feasible = _run_one(instance, scenario, method_name, config)
                priority_metrics = compute_priority_metrics(instance, solution) if feasible else {}
                drone_customers = solution.served_by_drone() if feasible else set()
                rich_metrics = _collect_row(instance, solution, scenario, method_name) if feasible else {}
                instance_run_rows.append(
                    {
                        "instance_id": instance_id,
                        "scenario_id": _scenario_key(source_row),
                        "n": int(instance.num_customers),
                        "method_name": method_name,
                        "feasible": bool(feasible),
                        "cpu_time_seconds": float(elapsed),
                        "error": "",
                        "objective": float(solution.objective) if feasible else float("nan"),
                        "priority_tardiness_cost": float(priority_metrics.get("priority_tardiness_cost", float("nan"))),
                        "high_priority_tardiness": float(priority_metrics.get("high_priority_tardiness", float("nan"))),
                        "max_high_priority_tardiness": float(priority_metrics.get("max_high_priority_tardiness", float("nan"))),
                        "p90_high_priority_tardiness": float(priority_metrics.get("p90_high_priority_tardiness", float("nan"))),
                        "drone_served_customers": int(len(drone_customers)) if feasible else 0,
                        "drone_share": float(len(drone_customers) / max(1, instance.num_customers)) if feasible else float("nan"),
                        "truck_travel_cost": float(rich_metrics.get("truck_travel_cost", float("nan"))),
                        "drone_travel_cost": float(rich_metrics.get("drone_travel_cost", float("nan"))),
                        "waiting_sync_cost": float(rich_metrics.get("waiting_sync_cost", float("nan"))),
                        "truck_route_distance_km": float(rich_metrics.get("truck_route_distance_km", float("nan"))),
                        "drone_flight_distance_km": float(rich_metrics.get("drone_flight_distance_km", float("nan"))),
                        "total_truck_waiting_time": float(rich_metrics.get("total_truck_waiting_time", float("nan"))),
                        "total_drone_waiting_time": float(rich_metrics.get("total_drone_waiting_time", float("nan"))),
                        "average_synchronization_delay": float(rich_metrics.get("average_synchronization_delay", float("nan"))),
                        "maximum_synchronization_delay": float(rich_metrics.get("maximum_synchronization_delay", float("nan"))),
                        "average_battery_usage_ratio": float(rich_metrics.get("average_battery_usage_ratio", float("nan"))),
                        "maximum_battery_usage_ratio": float(rich_metrics.get("maximum_battery_usage_ratio", float("nan"))),
                        "sorties_near_endurance_limit": float(rich_metrics.get("sorties_near_endurance_limit", float("nan"))),
                        "eligible_not_assigned_due_battery": float(rich_metrics.get("eligible_not_assigned_due_battery", float("nan"))),
                        "makespan_minutes": float(rich_metrics.get("makespan_minutes", float("nan"))),
                        "number_of_drone_launches": float(rich_metrics.get("number_of_drone_launches", float("nan"))),
                        "number_of_drone_retrievals": float(rich_metrics.get("number_of_drone_retrievals", float("nan"))),
                    }
                )
                if feasible:
                    instance_customer_rows.extend(_customer_rows(instance, solution, method_name))
                    instance_customer_detail_rows.extend(
                        _customer_detail_rows(instance, solution, method_name, _scenario_key(source_row), instance.seed)
                    )
                if method_name == "nils_r":
                    for decision_log in solution.__dict__.get("nils_r_drone_decision_logs", []):
                        decision_row = dict(decision_log)
                        decision_row["instance_id"] = instance_id
                        decision_row["seed"] = int(instance.seed)
                        decision_row["method"] = METHOD_ORDER[method_name][0]
                        customer = instance.customer_map().get(int(decision_row["customer_id"]))
                        decision_row["priority_class"] = customer.priority if customer is not None else ""
                        instance_decision_rows.append(decision_row)
                if method_name == "nils_r":
                    for reconstruction_log in solution.__dict__.get("nils_r_reconstruction_logs", []):
                        reconstruction_row = dict(reconstruction_log)
                        reconstruction_row["instance_id"] = instance_id
                        reconstruction_row["seed"] = int(instance.seed)
                        reconstruction_rows = pd.concat(
                            [reconstruction_rows, pd.DataFrame([reconstruction_row])], ignore_index=True
                        )
                _log(log_path, f"  {instance_id} | {method_name} | feasible={feasible} time={elapsed:.2f}s")
            except Exception as error:
                instance_run_rows.append(
                    {
                        "instance_id": instance_id,
                        "scenario_id": _scenario_key(source_row),
                        "n": int(instance.num_customers),
                        "method_name": method_name,
                        "feasible": False,
                        "cpu_time_seconds": float("nan"),
                        "error": repr(error),
                        "objective": float("nan"),
                        "priority_tardiness_cost": float("nan"),
                        "high_priority_tardiness": float("nan"),
                        "max_high_priority_tardiness": float("nan"),
                        "p90_high_priority_tardiness": float("nan"),
                        "drone_served_customers": 0,
                        "drone_share": float("nan"),
                        "truck_travel_cost": float("nan"),
                        "drone_travel_cost": float("nan"),
                        "waiting_sync_cost": float("nan"),
                        "truck_route_distance_km": float("nan"),
                        "drone_flight_distance_km": float("nan"),
                        "total_truck_waiting_time": float("nan"),
                        "total_drone_waiting_time": float("nan"),
                        "average_synchronization_delay": float("nan"),
                        "maximum_synchronization_delay": float("nan"),
                        "average_battery_usage_ratio": float("nan"),
                        "maximum_battery_usage_ratio": float("nan"),
                        "sorties_near_endurance_limit": float("nan"),
                        "eligible_not_assigned_due_battery": float("nan"),
                        "makespan_minutes": float("nan"),
                        "number_of_drone_launches": float("nan"),
                        "number_of_drone_retrievals": float("nan"),
                    }
                )
                _log(log_path, f"  {instance_id} | {method_name} | failed error={error}")

        if instance_run_rows:
            run_status = pd.concat([run_status, pd.DataFrame(instance_run_rows)], ignore_index=True)
            customer_rows = pd.concat([customer_rows, pd.DataFrame(instance_customer_rows)], ignore_index=True)
            customer_detail_rows = pd.concat(
                [customer_detail_rows, pd.DataFrame(instance_customer_detail_rows)], ignore_index=True
            )
            decision_rows = pd.concat([decision_rows, pd.DataFrame(instance_decision_rows)], ignore_index=True)
            run_status = run_status.drop_duplicates(["instance_id", "method_name"], keep="last")
            customer_rows = customer_rows.drop_duplicates(
                ["instance_id", "method_name", "customer_id"], keep="last"
            )
            customer_detail_rows = customer_detail_rows.drop_duplicates(
                ["instance_id", "method", "customer_id"], keep="last"
            )
            reconstruction_rows = reconstruction_rows.drop_duplicates(
                ["instance_id", "reconstruction_attempt"], keep="last"
            )
            _write_partial(run_status, run_path)
            _write_partial(customer_rows, customer_path)
            _write_partial(customer_detail_rows, customer_detail_path)
            _write_partial(decision_rows, decision_path)
            _write_partial(reconstruction_rows, reconstruction_path)

    table = _aggregate(run_status, customer_rows)
    output_path = output_dir / "table_priority_service_speed_by_method.csv"
    table.to_csv(output_path, index=False, float_format="%.6f")
    reconstruction_rows.to_csv(reconstruction_path, index=False, float_format="%.10f")
    customer_detail_rows.to_csv(customer_detail_path, index=False, float_format="%.10f")
    decision_rows.to_csv(decision_path, index=False, float_format="%.10f")
    _log(log_path, f"Priority service-speed study complete | rows={len(table)} | output={output_path}")
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-config",
        type=Path,
        default=Path("config/experiment_doptimal_beta010_tables5_14.yaml"),
    )
    parser.add_argument(
        "--alns-config",
        type=Path,
        default=Path("config/experiment_alns_comparison.yaml"),
    )
    parser.add_argument(
        "--source-output-dir",
        type=Path,
        default=Path("outputs/doptimal_full_beta010_tables5_14_rerun2_20260601_144538"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/priority_service_speed_by_method"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        main_config_path=args.main_config,
        alns_config_path=args.alns_config,
        source_output_dir=args.source_output_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        start_index=args.start_index,
        end_index=args.end_index,
        methods=args.methods,
    )


if __name__ == "__main__":
    main()


