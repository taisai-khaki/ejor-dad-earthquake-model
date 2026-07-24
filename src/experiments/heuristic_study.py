"""Heuristic-only benchmark study with strong baselines, ablations, tables, and figures."""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data_models import InstanceData, SolutionData
from ..distance_utils import euclidean_distance, manhattan_distance, travel_time_from_distance
from ..feasibility import is_feasible
from ..heuristics import run_nils
from ..heuristics.baselines import (
    feasibility_aware_alns_baseline,
    nils_soft_drone_target_baseline,
    nils_no_battery_screening_baseline,
    nils_no_local_search_baseline,
    nils_no_perturbation_baseline,
    no_priority_baseline,
    no_unpairing_baseline,
    paired_baseline,
    random_feasible_reassignment_baseline,
    simple_drone_assignment_baseline,
    truck_only_baseline,
)
from ..heuristics.construction import build_initial_solution
from ..heuristics.nils import _repair_infeasible_solution, evaluate_solution, summarize_drone_usage
from ..instance_generator import InstanceGenerator
from ..parameters import SearchConfig
from ..reporting.latex_export import dataframe_to_latex
from .statistics import bootstrap_ci, paired_comparison


DEFAULT_STUDY_LEVELS = {
    "n": [25, 50, 75, 100, 150],
    "eligible_share": [0.25, 0.50, 0.75],
    "endurance": ["low", "medium", "high"],
    "speed_ratio": [1.25, 1.50, 2.00],
    "handling_time": ["short", "medium", "long"],
    "spatial_pattern": ["uniform", "clustered"],
    "drones_available": [1, 2, 3],
    "priority_weight_profile": ["base"],
    "battery_slack_ratio": [0.15],
}

DEFAULT_METHODS = [
    "truck_only",
    "simple_drone",
    "random_feasible",
    "paired_baseline",
    "no_unpairing",
    "nils",
    "nils_no_local_search",
    "nils_no_perturbation",
    "nils_no_battery_screening",
    "nils_soft_drone_target",
    "feasibility_aware_alns",
]

PRIORITY_WEIGHT_PROFILES: Dict[str, Dict[str, float]] = {
    "flat": {"high": 1.0, "medium": 1.0, "low": 1.0},
    "moderate": {"high": 20.0, "medium": 5.0, "low": 1.0},
    "base": {"high": 100.0, "medium": 20.0, "low": 1.0},
    "strict": {"high": 200.0, "medium": 40.0, "low": 1.0},
}


def _build_progress_writer(output_dir: str | Path) -> Path:
    out = Path(output_dir) / "logs"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"heuristic_study_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def _log(progress_path: Path, message: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
    print(line)
    with progress_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{line}\n")


def _safe_div(numer: float, denom: float) -> float:
    if denom is None or abs(denom) < 1e-12 or math.isnan(denom):
        return float("nan")
    return float(numer / denom)


def _safe_pct_improvement(baseline: float, method: float) -> float:
    if baseline is None or method is None or not math.isfinite(baseline) or not math.isfinite(method):
        return float("nan")
    if abs(baseline) < 1e-12:
        return float("nan")
    return 100.0 * (baseline - method) / baseline


def _as_levels(config: SearchConfig) -> Dict[str, List]:
    exp_cfg = dict(config.experiment or {})
    grid = dict(exp_cfg.get("study_grid", {}))
    levels: Dict[str, List] = {}
    for key, default in DEFAULT_STUDY_LEVELS.items():
        raw = grid.get(key, default)
        if not isinstance(raw, list) or not raw:
            levels[key] = list(default)
        else:
            levels[key] = list(raw)
    return levels


def _study_methods(config: SearchConfig) -> List[str]:
    raw = config.experiment.get("study_methods", DEFAULT_METHODS)
    if not isinstance(raw, list) or not raw:
        return list(DEFAULT_METHODS)
    return [str(v) for v in raw]


def _scenario_grid(levels: Dict[str, List], max_scenarios: int | None = None) -> List[Dict[str, object]]:
    scenarios = []
    axes = [
        levels["n"],
        levels["eligible_share"],
        levels["endurance"],
        levels["speed_ratio"],
        levels["handling_time"],
        levels["spatial_pattern"],
        levels["drones_available"],
        levels["priority_weight_profile"],
        levels["battery_slack_ratio"],
    ]
    for idx, values in enumerate(itertools.product(*axes), start=1):
        scenario = {
            "scenario_id": f"S{idx:04d}",
            "n": int(values[0]),
            "eligible_share": float(values[1]),
            "endurance": values[2],
            "speed_ratio": float(values[3]),
            "handling_time": values[4],
            "spatial_pattern": str(values[5]),
            "drones_available": int(values[6]),
            "priority_weight_profile": str(values[7]).strip().lower(),
            "battery_slack_ratio": float(values[8]),
            "depot_position": "peripheral",
        }
        scenarios.append(scenario)
        if max_scenarios is not None and len(scenarios) >= max_scenarios:
            break
    return scenarios


def _scale_to_unit(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(float)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if abs(hi - lo) <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (2.0 * (arr - lo) / (hi - lo)) - 1.0


def _design_feature_matrix(levels: Dict[str, List], scenarios: List[Dict[str, object]]) -> np.ndarray:
    """Build mixed-level model matrix for D-optimal selection."""
    if not scenarios:
        return np.zeros((0, 1), dtype=float)

    n_vec = np.array([float(s["n"]) for s in scenarios], dtype=float)
    elig_vec = np.array([float(s["eligible_share"]) for s in scenarios], dtype=float)
    speed_vec = np.array([float(s["speed_ratio"]) for s in scenarios], dtype=float)
    drones_vec = np.array([float(s["drones_available"]) for s in scenarios], dtype=float)
    slack_vec = np.array([float(s.get("battery_slack_ratio", 0.15)) for s in scenarios], dtype=float)

    n_s = _scale_to_unit(n_vec)
    elig_s = _scale_to_unit(elig_vec)
    speed_s = _scale_to_unit(speed_vec)
    drones_s = _scale_to_unit(drones_vec)
    slack_s = _scale_to_unit(slack_vec)

    cols: List[np.ndarray] = [
        np.ones(len(scenarios), dtype=float),
        n_s,
        elig_s,
        speed_s,
        drones_s,
        slack_s,
    ]

    def add_effect_coding(key: str) -> None:
        raw_levels = [str(v) for v in levels.get(key, [])]
        if len(raw_levels) <= 1:
            return
        values = [str(s[key]) for s in scenarios]
        for lvl in raw_levels[1:]:
            cols.append(np.array([1.0 if v == lvl else 0.0 for v in values], dtype=float))

    add_effect_coding("endurance")
    add_effect_coding("handling_time")
    add_effect_coding("spatial_pattern")
    add_effect_coding("priority_weight_profile")

    # Selected interactions to preserve core structural effects.
    cols.extend(
        [
            n_s * drones_s,
            speed_s * elig_s,
            speed_s * n_s,
            elig_s * drones_s,
            slack_s * elig_s,
        ]
    )
    return np.column_stack(cols)


def _d_optimal_select_indices(
    X: np.ndarray,
    target_points: int,
    seed: int,
    restarts: int = 6,
    ridge: float = 1e-8,
) -> List[int]:
    """Greedy multi-start D-optimal subset selection."""
    n_rows, n_cols = X.shape
    if n_rows == 0:
        return []
    k = int(max(1, min(target_points, n_rows)))
    k = max(k, n_cols) if n_rows >= n_cols else n_rows

    eye = np.eye(n_cols, dtype=float)
    all_idx = np.arange(n_rows, dtype=int)
    rng = np.random.default_rng(seed)

    def score(indices: List[int]) -> float:
        if not indices:
            return float("-inf")
        m = X[np.array(indices, dtype=int)].T @ X[np.array(indices, dtype=int)] + ridge * eye
        sign, logdet = np.linalg.slogdet(m)
        if sign <= 0:
            return float("-inf")
        return float(logdet)

    best_idx: List[int] = list(all_idx[:k])
    best_score = score(best_idx)

    for _ in range(max(1, int(restarts))):
        first = int(rng.integers(0, n_rows))
        selected = [first]
        mask = np.ones(n_rows, dtype=bool)
        mask[first] = False
        m = ridge * eye + np.outer(X[first], X[first])

        while len(selected) < k:
            cand_idx = all_idx[mask]
            best_c = None
            best_c_val = float("-inf")
            for c in cand_idx:
                m_new = m + np.outer(X[c], X[c])
                sign, logdet = np.linalg.slogdet(m_new)
                if sign > 0 and logdet > best_c_val:
                    best_c_val = float(logdet)
                    best_c = int(c)
            if best_c is None:
                best_c = int(cand_idx[0]) if len(cand_idx) else int(selected[-1])
            selected.append(best_c)
            mask[best_c] = False
            m = m + np.outer(X[best_c], X[best_c])

        cur = score(selected)
        if cur > best_score:
            best_score = cur
            best_idx = selected

    return sorted(int(i) for i in best_idx)


def _finalize_scenario_ids(scenarios: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for idx, s in enumerate(scenarios, start=1):
        row = dict(s)
        row["scenario_id"] = f"S{idx:04d}"
        out.append(row)
    return out


def _apply_conditional_filters(scenarios: List[Dict[str, object]], exp_cfg: Dict[str, object]) -> List[Dict[str, object]]:
    """Apply optional conditional scenario filters from config."""
    raw = exp_cfg.get("study_conditional_n_by_drones")
    allowed_by_drones: Dict[int, set[int]] = {}
    if isinstance(raw, dict) and raw:
        for key, val in raw.items():
            try:
                d = int(key)
            except (TypeError, ValueError):
                continue
            if not isinstance(val, list) or not val:
                continue
            allowed: set[int] = set()
            for n in val:
                try:
                    allowed.add(int(n))
                except (TypeError, ValueError):
                    continue
            if allowed:
                allowed_by_drones[d] = allowed

    if not allowed_by_drones:
        filtered = list(scenarios)
    else:
        filtered = []
        for scenario in scenarios:
            d = int(scenario["drones_available"])
            n = int(scenario["n"])
            allowed_n = allowed_by_drones.get(d)
            if allowed_n is None or n in allowed_n:
                filtered.append(scenario)

    # Generic exclusion rules.
    # Example:
    # study_exclude_conditions:
    #   - {spatial_pattern: clustered, n_min: 75}
    #   - {spatial_pattern: clustered, number_of_drones_min: 3}
    #   - {spatial_pattern: clustered, handling_time: long}
    raw_exclusions = exp_cfg.get("study_exclude_conditions")
    if not isinstance(raw_exclusions, list) or not raw_exclusions:
        return filtered

    def _matches_rule(scenario: Dict[str, object], rule: Dict[str, object]) -> bool:
        for key, value in rule.items():
            if key.endswith("_min"):
                field = key[:-4]
                if field == "number_of_drones":
                    field = "drones_available"
                if float(scenario.get(field, float("-inf"))) < float(value):
                    return False
                continue
            if key.endswith("_max"):
                field = key[:-4]
                if field == "number_of_drones":
                    field = "drones_available"
                if float(scenario.get(field, float("inf"))) > float(value):
                    return False
                continue
            field = "drones_available" if key == "number_of_drones" else key
            s_val = scenario.get(field)
            if isinstance(value, list):
                if s_val not in value:
                    return False
            else:
                if str(s_val) != str(value):
                    return False
        return True

    out: List[Dict[str, object]] = []
    for scenario in filtered:
        excluded = False
        for rule in raw_exclusions:
            if isinstance(rule, dict) and _matches_rule(scenario, rule):
                excluded = True
                break
        if not excluded:
            out.append(scenario)
    return out


def _build_scenarios(levels: Dict[str, List], config: SearchConfig) -> List[Dict[str, object]]:
    exp_cfg = dict(config.experiment or {})
    design = str(exp_cfg.get("study_design", "full_factorial")).strip().lower()
    max_scenarios = exp_cfg.get("study_max_scenarios")
    max_scenarios = int(max_scenarios) if max_scenarios is not None else None

    full = _scenario_grid(levels, max_scenarios=None)
    full = _apply_conditional_filters(full, exp_cfg)
    if design == "d_optimal":
        target = int(exp_cfg.get("study_d_optimal_points", min(96, len(full))))
        restarts = int(exp_cfg.get("study_d_optimal_restarts", 6))
        X = _design_feature_matrix(levels, full)
        idx = _d_optimal_select_indices(X, target_points=target, seed=int(config.seed), restarts=restarts)
        selected = [full[i] for i in idx]
        return _finalize_scenario_ids(selected)

    if max_scenarios is not None:
        full = full[:max_scenarios]
    return _finalize_scenario_ids(full)


def _endurance_multiplier(level: object) -> float:
    if isinstance(level, (int, float)):
        return float(level)
    label = str(level).strip().lower()
    return {"low": 0.70, "medium": 1.00, "high": 1.30}.get(label, 1.00)


def _handling_seconds(level: object) -> float:
    if isinstance(level, (int, float)):
        return float(level)
    label = str(level).strip().lower()
    return {"short": 30.0, "medium": 60.0, "long": 120.0}.get(label, 60.0)


def _region_from_pattern(pattern: str) -> str:
    tag = str(pattern).strip().lower()
    if tag in {"uniform", "dispersed"}:
        return "dispersed"
    if tag in {"clustered", "dense", "dense_urban"}:
        return "dense_urban"
    return "mixed"


def _priority_penalties_for_profile(profile: object, fallback: Dict[str, float]) -> Dict[str, float]:
    label = str(profile).strip().lower()
    if label in {"no_priority", "nopriority", "uniform"}:
        label = "flat"
    weights = PRIORITY_WEIGHT_PROFILES.get(label)
    if weights is None:
        return {
            "high": float(fallback.get("high", 100.0)),
            "medium": float(fallback.get("medium", 20.0)),
            "low": float(fallback.get("low", 1.0)),
        }
    return {
        "high": float(weights["high"]),
        "medium": float(weights["medium"]),
        "low": float(weights["low"]),
    }


def _scenario_config(base: SearchConfig, scenario: Dict[str, object]) -> SearchConfig:
    n = int(scenario["n"])
    drones = int(scenario["drones_available"])
    speed_ratio = float(scenario["speed_ratio"])
    endurance_mult = _endurance_multiplier(scenario["endurance"])
    handling_s = _handling_seconds(scenario["handling_time"])
    region = _region_from_pattern(str(scenario["spatial_pattern"]))
    priority_profile = str(scenario.get("priority_weight_profile", "base")).strip().lower()
    priority_penalties = _priority_penalties_for_profile(priority_profile, dict(base.priority_penalties))

    truck = replace(base.vehicles.truck, speed_kmph=float(base.vehicles.truck.speed_kmph))
    drone_speed = max(1.0, truck.speed_kmph * speed_ratio)
    drone_batt = max(10.0, float(base.vehicles.drone.max_battery_wh) * endurance_mult)
    drone = replace(base.vehicles.drone, speed_kmph=drone_speed, max_battery_wh=drone_batt)
    vehicles = replace(base.vehicles, truck=truck, drone=drone)

    generation = replace(
        base.generation,
        num_customers=n,
        num_drones=drones,
        region=region,
    )
    runtime = replace(base.vehicle_runtime, swap_time_s=handling_s, reload_time_s=handling_s, region=region)
    return SearchConfig(
        seed=base.seed,
        generation=generation,
        priority_penalties=priority_penalties,
        base_windows=dict(base.base_windows),
        vehicle_costs=base.vehicle_costs,
        milp=replace(base.milp, enabled=False),
        heuristics=base.heuristics,
        constraints=base.constraints,
        vehicles=vehicles,
        vehicle_runtime=runtime,
        experiment=dict(base.experiment),
        sensitivity=dict(base.sensitivity),
    )


def _apply_eligibility_mask(instance: InstanceData, eligible_share: float, seed: int) -> None:
    rng = np.random.default_rng(seed + 991)
    n = max(1, instance.num_customers)
    count = int(round(float(eligible_share) * n))
    count = max(1, min(n, count))
    eligible = sorted(int(v) for v in rng.choice(list(instance.customer_ids), size=count, replace=False))
    instance.metadata["drone_eligible_customers"] = eligible
    instance.metadata["eligible_share_target"] = float(eligible_share)
    instance.metadata["eligible_share_realized"] = float(len(eligible) / n)


def _passes_preprocessing_feasibility(instance: InstanceData) -> bool:
    # Aggregate load feasibility.
    total_demand = float(sum(c.demand_kg for c in instance.customers))
    total_truck_capacity = float(sum(t.capacity_kg for t in instance.trucks))
    if total_demand > total_truck_capacity + 1e-9:
        return False

    # Individual serviceability: each parcel must fit at least one truck.
    max_truck_capacity = float(max((t.capacity_kg for t in instance.trucks), default=0.0))
    if any(float(c.demand_kg) > max_truck_capacity + 1e-9 for c in instance.customers):
        return False

    # Route-duration screen: each customer must be reachable by at least one truck
    # (depot -> customer -> depot plus service) within shift bound.
    coords = instance.coordinate_map()
    shift_limit = float(instance.constants.max_shift_minutes)
    for c in instance.customers:
        feasible_visit = False
        for truck in instance.trucks:
            dist = manhattan_distance(coords[0], coords[c.node_id]) * 2.0
            travel_min = travel_time_from_distance(dist, max(float(truck.speed_kmph), 1e-9))
            total_min = travel_min + float(c.service_time_min)
            if total_min <= shift_limit + 1e-9:
                feasible_visit = True
                break
        if not feasible_visit:
            return False
    return True


def _run_method(instance: InstanceData, config: SearchConfig, method_name: str) -> SolutionData:
    if method_name == "truck_only":
        return truck_only_baseline(instance, config).solution
    if method_name == "simple_drone":
        return simple_drone_assignment_baseline(instance, config).solution
    if method_name == "random_feasible":
        return random_feasible_reassignment_baseline(instance, config).solution
    if method_name == "paired_baseline":
        return paired_baseline(instance, config).solution
    if method_name == "no_priority":
        return no_priority_baseline(instance, config).solution
    if method_name == "no_unpairing":
        return no_unpairing_baseline(instance, config).solution
    if method_name == "nils":
        return run_nils(
            instance,
            seed=config.heuristics.random_seed,
            max_iter=config.heuristics.max_outer_iter,
            max_no_improve=config.heuristics.max_no_improve,
            time_limit=config.heuristics.time_limit_seconds,
        )
    if method_name == "nils_soft_drone_target":
        return nils_soft_drone_target_baseline(instance, config).solution
    if method_name == "nils_no_local_search":
        return nils_no_local_search_baseline(instance, config).solution
    if method_name == "nils_no_perturbation":
        return nils_no_perturbation_baseline(instance, config).solution
    if method_name == "nils_no_battery_screening":
        return nils_no_battery_screening_baseline(instance, config).solution
    if method_name == "feasibility_aware_alns":
        return feasibility_aware_alns_baseline(instance, config).solution
    raise ValueError(f"Unsupported method_name: {method_name}")


def _route_distance(route: List[int], coords: Dict[int, Tuple[float, float]], *, metric: str) -> float:
    if not route or len(route) < 2:
        return 0.0
    dist = 0.0
    for idx in range(len(route) - 1):
        a = coords[route[idx]]
        b = coords[route[idx + 1]]
        if metric == "manhattan":
            dist += manhattan_distance(a, b)
        else:
            dist += euclidean_distance(a, b)
    return float(dist)


def _sync_delay_stats(instance: InstanceData, solution: SolutionData) -> Tuple[float, float, int]:
    delays: List[float] = []
    delayed_events = 0
    sync_window = max(instance.constants.swap_time_s, instance.constants.reload_time_s) / 60.0
    for (node, drone, truck), val in solution.z1.items():
        if val <= 0.5:
            continue
        delta = abs(solution.a_truck.get((truck, node), 0.0) - solution.a_drone.get((drone, node), 0.0))
        delays.append(delta)
        if delta > sync_window + 1e-9:
            delayed_events += 1
    for (node, drone, truck), val in solution.z2.items():
        if val <= 0.5:
            continue
        delta = abs(solution.a_truck.get((truck, node), 0.0) - solution.a_drone.get((drone, node), 0.0))
        delays.append(delta)
        if delta > sync_window + 1e-9:
            delayed_events += 1
    if not delays:
        return 0.0, 0.0, 0
    return float(np.mean(delays)), float(np.max(delays)), int(delayed_events)


def _battery_usage_stats(instance: InstanceData, solution: SolutionData, coords: Dict[int, Tuple[float, float]]) -> Tuple[float, float, int]:
    ratios: List[float] = []
    near = 0
    served_by_drone = solution.served_by_drone()
    for drone_id, route in solution.drone_routes.items():
        if len(route) < 2 or all(node == 0 for node in route):
            continue
        drone = instance.drones[drone_id - 1]
        consumed = 0.0
        for idx in range(len(route) - 1):
            frm = route[idx]
            to = route[idx + 1]
            dist = euclidean_distance(coords[frm], coords[to])
            arc_time = travel_time_from_distance(dist, drone.speed_kmph)
            loaded = to in served_by_drone
            burn = drone.energy_per_min_when_loaded if loaded else drone.energy_per_min_when_empty
            if burn <= 0:
                burn = 20.0 if loaded else 15.0
            consumed += burn * arc_time
        ratio = _safe_div(consumed, drone.max_battery_wh)
        if math.isfinite(ratio):
            ratios.append(ratio)
            if ratio >= 0.90:
                near += 1
    if not ratios:
        return 0.0, 0.0, 0
    return float(np.mean(ratios)), float(np.max(ratios)), int(near)


def _collect_row(
    instance: InstanceData,
    solution: SolutionData,
    scenario: Dict[str, object],
    method_name: str,
) -> Dict[str, object]:
    coords = instance.coordinate_map()
    usage = summarize_drone_usage(solution)
    truck_distance = float(sum(_route_distance(route, coords, metric="manhattan") for route in solution.truck_routes.values()))
    drone_distance = float(sum(_route_distance(route, coords, metric="euclidean") for route in solution.drone_routes.values()))
    truck_served = len(solution.served_by_truck())
    drone_served = len(solution.served_by_drone())
    if drone_served == 0:
        for route in solution.drone_routes.values():
            drone_served += sum(1 for node in route if node != 0)
        drone_served = int(min(instance.num_customers, drone_served))

    makespan = 0.0
    if solution.l_truck:
        makespan = max(makespan, float(max(solution.l_truck.values())))
    if solution.l_drone:
        makespan = max(makespan, float(max(solution.l_drone.values())))
    if solution.a_truck:
        makespan = max(makespan, float(max(solution.a_truck.values())))
    if solution.a_drone:
        makespan = max(makespan, float(max(solution.a_drone.values())))

    launches = int(usage.get("reload_events", 0))
    retrievals = int(usage.get("battery_swaps", 0))
    if launches == 0:
        launches = int(usage.get("nonempty_drone_routes", 0))
    if retrievals == 0:
        retrievals = int(usage.get("nonempty_drone_routes", 0))

    avg_sync_delay, max_sync_delay, delayed_rendezvous_events = _sync_delay_stats(instance, solution)
    avg_batt_ratio, max_batt_ratio, near_endurance = _battery_usage_stats(instance, solution, coords)
    battery_rejected = int(solution.components.get("battery_infeasible_rejections", 0.0))

    truck_cost = float(solution.components.get("truck_cost", 0.0))
    drone_cost = float(solution.components.get("drone_cost", 0.0))
    waiting_cost = float(solution.components.get("waiting_cost", 0.0))
    penalty_cost = float(solution.components.get("tardiness_cost", 0.0))
    feasible_flag = bool(is_feasible(instance, solution))
    customer_map = instance.customer_map()
    def _priority_tardiness_minutes(priority_label: str) -> float:
        return float(
            sum(
                max(0.0, float(t))
                for cid, t in solution.tardiness.items()
                if cid in customer_map and customer_map[cid].priority == priority_label
            )
        )

    high_priority_tardiness = _priority_tardiness_minutes("high")
    medium_priority_tardiness = _priority_tardiness_minutes("medium")
    low_priority_tardiness = _priority_tardiness_minutes("low")
    priority_profile = str(scenario.get("priority_weight_profile", "base")).strip().lower()
    weights = _priority_penalties_for_profile(priority_profile, dict(instance.priority_penalties))
    scenario_class = (
        f"n{scenario['n']}_p{float(scenario['eligible_share']):.2f}_d{scenario['drones_available']}_"
        f"{scenario['spatial_pattern']}_E{scenario['endurance']}_V{float(scenario['speed_ratio']):.2f}_"
        f"H{scenario['handling_time']}_PW{priority_profile}_B{float(scenario.get('battery_slack_ratio', 0.15)):.2f}"
    )

    return {
        "instance_id": instance.name,
        "random_seed": int(instance.seed),
        "method_name": method_name,
        "scenario_id": scenario["scenario_id"],
        "scenario_class": scenario_class,
        "n": int(scenario["n"]),
        "eligible_share": float(instance.metadata.get("eligible_share_realized", scenario["eligible_share"])),
        "eligible_share_target": float(scenario["eligible_share"]),
        "number_of_drones": int(scenario["drones_available"]),
        "number_of_trucks": int(instance.num_trucks),
        "endurance_level": str(scenario["endurance"]),
        "speed_ratio": float(scenario["speed_ratio"]),
        "spatial_pattern": str(scenario["spatial_pattern"]),
        "launch_retrieval_time_class": str(scenario["handling_time"]),
        "priority_weight_profile": priority_profile,
        "battery_slack_ratio": float(scenario.get("battery_slack_ratio", 0.15)),
        "priority_weight_high": float(weights["high"]),
        "priority_weight_medium": float(weights["medium"]),
        "priority_weight_low": float(weights["low"]),
        "depot_position": str(scenario.get("depot_position", "peripheral")),
        "objective_total": float(solution.objective),
        "truck_travel_cost": truck_cost,
        "drone_travel_cost": drone_cost,
        "waiting_sync_cost": waiting_cost,
        "penalty_cost": penalty_cost,
        "priority_tardiness_cost": penalty_cost,
        "high_priority_tardiness_minutes": high_priority_tardiness,
        "medium_priority_tardiness_minutes": medium_priority_tardiness,
        "low_priority_tardiness_minutes": low_priority_tardiness,
        "total_priority_tardiness_minutes": high_priority_tardiness + medium_priority_tardiness + low_priority_tardiness,
        "high_priority_tardiness_weighted_cost": float(weights["high"]) * high_priority_tardiness,
        "medium_priority_tardiness_weighted_cost": float(weights["medium"]) * medium_priority_tardiness,
        "low_priority_tardiness_weighted_cost": float(weights["low"]) * low_priority_tardiness,
        "fixed_deployment_cost": 0.0,
        "makespan_minutes": makespan,
        "truck_route_distance_km": truck_distance,
        "drone_flight_distance_km": drone_distance,
        "customers_served_by_truck": int(truck_served),
        "customers_served_by_drone": int(drone_served),
        "drone_service_share": _safe_div(float(drone_served), float(max(1, instance.num_customers))),
        "number_of_drone_launches": launches,
        "number_of_drone_retrievals": retrievals,
        "average_customers_per_drone_sortie": _safe_div(float(drone_served), float(max(1, launches))),
        "average_customers_per_drone_route": _safe_div(float(drone_served), float(max(1, usage.get("nonempty_drone_routes", 0)))),
        "total_truck_waiting_time": float(sum(solution.waiting_truck.values())) if solution.waiting_truck else 0.0,
        "total_drone_waiting_time": float(sum(solution.waiting_drone.values())) if solution.waiting_drone else 0.0,
        "average_synchronization_delay": avg_sync_delay,
        "maximum_synchronization_delay": max_sync_delay,
        "delayed_rendezvous_events": delayed_rendezvous_events,
        "average_battery_usage_ratio": avg_batt_ratio,
        "maximum_battery_usage_ratio": max_batt_ratio,
        "sorties_near_endurance_limit": int(near_endurance),
        "infeasible_battery_attempts_rejected": battery_rejected,
        "eligible_not_assigned_due_battery": battery_rejected,
        "cpu_time_seconds": float(solution.run_time_seconds),
        "iterations": float(solution.components.get("iterations", 0.0)),
        "improving_moves_accepted": float(solution.components.get("improving_moves_accepted", 0.0)),
        "candidate_reassignments_evaluated": float(solution.components.get("candidate_reassignments_evaluated", 0.0)),
        "final_solution_feasibility_flag": feasible_flag,
        "status": "feasible" if feasible_flag else "infeasible",
        "run_label": str(solution.status),
    }


def _attach_relative_fields(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    base_truck = out[out["method_name"] == "truck_only"][["instance_id", "objective_total", "truck_route_distance_km"]].rename(
        columns={"objective_total": "truck_only_objective", "truck_route_distance_km": "truck_only_distance"}
    )
    base_simple = out[out["method_name"] == "simple_drone"][["instance_id", "objective_total"]].rename(
        columns={"objective_total": "simple_drone_objective"}
    )
    out = out.merge(base_truck, on="instance_id", how="left")
    out = out.merge(base_simple, on="instance_id", how="left")
    out["improvement_vs_truck_only_pct"] = out.apply(
        lambda r: _safe_pct_improvement(float(r["truck_only_objective"]), float(r["objective_total"])),
        axis=1,
    )
    out["improvement_vs_simple_drone_pct"] = out.apply(
        lambda r: _safe_pct_improvement(float(r["simple_drone_objective"]), float(r["objective_total"])),
        axis=1,
    )
    out["truck_distance_reduction_pct"] = out.apply(
        lambda r: _safe_pct_improvement(float(r["truck_only_distance"]), float(r["truck_route_distance_km"])),
        axis=1,
    )
    out.loc[out["method_name"] == "truck_only", "improvement_vs_truck_only_pct"] = float("nan")
    out.loc[out["method_name"].isin(["truck_only", "simple_drone"]), "improvement_vs_simple_drone_pct"] = float("nan")
    return out


def _attach_reporting_fields(df: pd.DataFrame, config: SearchConfig) -> pd.DataFrame:
    """Attach feasibility-aware reporting fields used by aggregate tables/figures."""
    if df.empty:
        return df

    out = df.copy()
    exp_cfg = dict(config.experiment or {})
    policy = str(exp_cfg.get("study_feasibility_policy", "feasible_only")).strip().lower()
    if policy not in {"feasible_only", "penalize", "all"}:
        policy = "feasible_only"
    penalty = float(exp_cfg.get("study_infeasibility_penalty", 1_000_000.0))

    feasible_mask = out["final_solution_feasibility_flag"].fillna(False) & out["status"].eq("feasible")
    out["is_feasible_run"] = feasible_mask.astype(bool)
    out["reporting_policy"] = policy
    out["reporting_penalty_value"] = penalty if policy == "penalize" else 0.0
    out["objective_for_reporting"] = out["objective_total"]
    if policy == "feasible_only":
        out.loc[~feasible_mask, "objective_for_reporting"] = float("nan")
    elif policy == "penalize":
        penalized = ~feasible_mask & out["objective_total"].notna()
        out.loc[penalized, "objective_for_reporting"] = out.loc[penalized, "objective_total"] + penalty

    out["reporting_included"] = out["objective_for_reporting"].apply(
        lambda v: bool(pd.notna(v) and math.isfinite(float(v)))
    )

    base_truck = out[out["method_name"] == "truck_only"][["instance_id", "objective_for_reporting"]].rename(
        columns={"objective_for_reporting": "truck_only_objective_reported"}
    )
    base_simple = out[out["method_name"] == "simple_drone"][["instance_id", "objective_for_reporting"]].rename(
        columns={"objective_for_reporting": "simple_drone_objective_reported"}
    )
    out = out.merge(base_truck, on="instance_id", how="left")
    out = out.merge(base_simple, on="instance_id", how="left")
    out["improvement_vs_truck_only_reported_pct"] = out.apply(
        lambda r: _safe_pct_improvement(float(r["truck_only_objective_reported"]), float(r["objective_for_reporting"])),
        axis=1,
    )
    out["improvement_vs_simple_drone_reported_pct"] = out.apply(
        lambda r: _safe_pct_improvement(
            float(r["simple_drone_objective_reported"]),
            float(r["objective_for_reporting"]),
        ),
        axis=1,
    )
    out.loc[out["method_name"] == "truck_only", "improvement_vs_truck_only_reported_pct"] = float("nan")
    out.loc[out["method_name"].isin(["truck_only", "simple_drone"]), "improvement_vs_simple_drone_reported_pct"] = float(
        "nan"
    )
    return out


def _save_table(df: pd.DataFrame, name: str, out_dir: Path, caption: str, label: str) -> None:
    csv_path = out_dir / f"{name}.csv"
    tex_path = out_dir / f"{name}.tex"
    df.to_csv(csv_path, index=False)
    dataframe_to_latex(df, tex_path, caption=caption, label=label)


def _dominant_reason(row: pd.Series) -> str:
    waiting_total = float(row.get("total_truck_waiting_time", 0.0)) + float(row.get("total_drone_waiting_time", 0.0))
    delayed = float(row.get("delayed_rendezvous_events", 0.0))
    battery = float(row.get("average_battery_usage_ratio", 0.0))
    drone_share = float(row.get("drone_service_share", 0.0))
    improv = float(row.get("avg_improvement", 0.0))
    spatial = str(row.get("spatial_pattern", "")).strip().lower()
    endurance = str(row.get("endurance_level", "")).strip().lower()
    launch_class = str(row.get("launch_retrieval_time_class", "")).strip().lower()
    speed_ratio = float(row.get("speed_ratio", float("nan")))
    eligible = float(row.get("eligible_share_target", 0.0))

    if battery >= 0.90:
        return "Battery near binding"
    if waiting_total >= 10.0 or delayed >= 1.0:
        return "Synchronization burden dominates"
    if eligible >= 0.60 and spatial == "clustered":
        return "High eligibility + clustered geometry"
    if endurance == "low" and math.isfinite(speed_ratio) and speed_ratio >= 1.75:
        return "Speed compensates low endurance"
    if spatial == "uniform" and drone_share <= 0.25:
        return "Uniform layout limits paired-drone value"
    if launch_class == "long" and waiting_total <= 1.0 and improv <= 10.0:
        return "Handling burden suppresses coordination gains"
    if drone_share <= 0.20:
        return "Limited drone-service leverage"
    return "Mixed structural effects"


def _generate_tables(df: pd.DataFrame, levels: Dict[str, List], out_dir: Path) -> None:
    if df.empty:
        return
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    report_df = df[df["reporting_included"]].copy() if "reporting_included" in df.columns else df.copy()
    if report_df.empty:
        report_df = df.copy()
    policy_label = str(df["reporting_policy"].dropna().iloc[0]) if "reporting_policy" in df.columns and not df.empty else "all"

    table1 = pd.DataFrame(
        [
            {"Factor": "Problem size", "Symbol": "n", "Levels": str(levels["n"]), "Operational interpretation": "Number of customers"},
            {"Factor": "Eligible share", "Symbol": "p_e", "Levels": str(levels["eligible_share"]), "Operational interpretation": "Share eligible for drone service"},
            {"Factor": "Endurance", "Symbol": "E", "Levels": str(levels["endurance"]), "Operational interpretation": "Battery-capacity level"},
            {"Factor": "Speed ratio", "Symbol": "v_D / v_T", "Levels": str(levels["speed_ratio"]), "Operational interpretation": "Drone speed relative to truck"},
            {"Factor": "Handling time", "Symbol": "tau", "Levels": str(levels["handling_time"]), "Operational interpretation": "Launch/retrieval burden class"},
            {"Factor": "Spatial pattern", "Symbol": "spatial", "Levels": str(levels["spatial_pattern"]), "Operational interpretation": "Customer spatial distribution"},
            {"Factor": "Drones available", "Symbol": "|D|", "Levels": str(levels["drones_available"]), "Operational interpretation": "Available drone fleet size"},
            {
                "Factor": "Battery-slack threshold",
                "Symbol": "beta_slack",
                "Levels": str(levels["battery_slack_ratio"]),
                "Operational interpretation": "Minimum residual battery ratio for admissible drone assignment",
            },
            {
                "Factor": "Priority-weight profile",
                "Symbol": "(pi_H, pi_M, pi_L)",
                "Levels": str(levels["priority_weight_profile"]),
                "Operational interpretation": "Tardiness weights by priority class",
            },
            {"Factor": "Feasibility policy", "Symbol": "-", "Levels": policy_label, "Operational interpretation": "How infeasible runs are handled in reported objectives"},
        ]
    )
    _save_table(table1, "table_01_experimental_factors", tables_dir, "Experimental factors and levels", "tab:factors")

    table2 = (
        df.groupby(
            [
                "scenario_class",
                "n",
                "spatial_pattern",
                "endurance_level",
                "eligible_share_target",
                "number_of_drones",
            ],
            dropna=False,
        )["instance_id"]
        .nunique()
        .reset_index(name="number_of_generated_instances")
        .sort_values(["n", "scenario_class"])
    )
    _save_table(table2, "table_02_instance_class_summary", tables_dir, "Instance-class summary", "tab:instance_classes")

    by_method_feas = (
        df.groupby("method_name", dropna=False)
        .agg(
            n_runs=("instance_id", "count"),
            feasible_runs=("final_solution_feasibility_flag", "sum"),
            feasibility_rate=("final_solution_feasibility_flag", "mean"),
        )
        .reset_index()
    )
    by_method_report = (
        report_df.groupby("method_name", dropna=False)
        .agg(
            average_objective=("objective_for_reporting", "mean"),
            std_dev=("objective_for_reporting", "std"),
            avg_improvement_vs_truck_only=("improvement_vs_truck_only_reported_pct", "mean"),
            avg_makespan=("makespan_minutes", "mean"),
            avg_cpu_time=("cpu_time_seconds", "mean"),
            avg_drone_service_share=("drone_service_share", "mean"),
            reporting_included_runs=("objective_for_reporting", "count"),
        )
        .reset_index()
    )
    feasible_obj = (
        df[df["final_solution_feasibility_flag"]]
        .groupby("method_name", dropna=False)["objective_total"]
        .mean()
        .reset_index(name="avg_objective_feasible_only")
    )
    infeasible_obj = (
        df[~df["final_solution_feasibility_flag"]]
        .groupby("method_name", dropna=False)["objective_total"]
        .mean()
        .reset_index(name="avg_objective_infeasible_only")
    )
    table3 = by_method_feas.merge(by_method_report, on="method_name", how="left")
    table3 = table3.merge(feasible_obj, on="method_name", how="left")
    table3 = table3.merge(infeasible_obj, on="method_name", how="left")
    table3["reporting_inclusion_rate"] = table3.apply(
        lambda r: _safe_div(float(r["reporting_included_runs"]), float(r["n_runs"])),
        axis=1,
    )
    table3 = table3.sort_values("average_objective", na_position="last")
    _save_table(table3, "table_03_overall_performance", tables_dir, "Overall performance comparison by method", "tab:overall")

    perf_all = df[df["method_name"].isin(["truck_only", "simple_drone", "nils"])].copy()
    perf_report = report_df[report_df["method_name"].isin(["truck_only", "simple_drone", "nils"])].copy()
    table4_feas = (
        perf_all.groupby(["n", "method_name"], dropna=False)
        .agg(
            n_runs=("instance_id", "count"),
            feasible_runs=("final_solution_feasibility_flag", "sum"),
            feasibility_rate=("final_solution_feasibility_flag", "mean"),
            avg_cpu_time=("cpu_time_seconds", "mean"),
        )
        .reset_index()
    )
    table4_obj = (
        perf_report.groupby(["n", "method_name"], dropna=False)
        .agg(
            objective=("objective_for_reporting", "mean"),
            reported_runs=("objective_for_reporting", "count"),
            improvement_vs_truck_only=("improvement_vs_truck_only_reported_pct", "mean"),
        )
        .reset_index()
    )
    table4 = (
        table4_feas.merge(table4_obj, on=["n", "method_name"], how="left")
        .sort_values(["n", "method_name"])
    )
    table4["reporting_inclusion_rate"] = table4.apply(
        lambda r: _safe_div(float(r["reported_runs"]), float(r["n_runs"])),
        axis=1,
    )
    _save_table(table4, "table_04_performance_by_size", tables_dir, "Performance by problem size", "tab:by_size")

    table5 = (
        report_df.groupby("method_name", dropna=False)[
            [
                "truck_travel_cost",
                "drone_travel_cost",
                "waiting_sync_cost",
                "priority_tardiness_cost",
                "objective_for_reporting",
                "total_truck_waiting_time",
                "total_drone_waiting_time",
                "delayed_rendezvous_events",
            ]
        ]
        .mean()
        .reset_index()
        .rename(columns={"objective_for_reporting": "objective_reported"})
    )
    table5["component_sum"] = (
        table5["truck_travel_cost"] + table5["drone_travel_cost"] + table5["waiting_sync_cost"] + table5["priority_tardiness_cost"]
    )
    table5["truck_cost_share_pct"] = table5.apply(
        lambda r: _safe_div(float(r["truck_travel_cost"]) * 100.0, float(r["component_sum"])),
        axis=1,
    )
    table5["drone_cost_share_pct"] = table5.apply(
        lambda r: _safe_div(float(r["drone_travel_cost"]) * 100.0, float(r["component_sum"])),
        axis=1,
    )
    table5["waiting_cost_share_pct"] = table5.apply(
        lambda r: _safe_div(float(r["waiting_sync_cost"]) * 100.0, float(r["component_sum"])),
        axis=1,
    )
    table5["priority_tardiness_share_pct"] = table5.apply(
        lambda r: _safe_div(float(r["priority_tardiness_cost"]) * 100.0, float(r["component_sum"])),
        axis=1,
    )
    table5 = table5.sort_values("objective_reported", na_position="last")
    table5 = table5.merge(by_method_feas[["method_name", "feasibility_rate"]], on="method_name", how="left")
    _save_table(table5, "table_05_objective_components", tables_dir, "Objective-component breakdown", "tab:components")

    table6 = (
        report_df.groupby("method_name", dropna=False)
        .agg(
            drone_service_share=("drone_service_share", "mean"),
            avg_launches=("number_of_drone_launches", "mean"),
            avg_retrievals=("number_of_drone_retrievals", "mean"),
            avg_drone_distance=("drone_flight_distance_km", "mean"),
            avg_truck_distance_saved_pct=("truck_distance_reduction_pct", "mean"),
            avg_truck_waiting=("total_truck_waiting_time", "mean"),
            avg_drone_waiting=("total_drone_waiting_time", "mean"),
        )
        .reset_index()
    )
    table6 = table6.merge(by_method_feas[["method_name", "feasibility_rate"]], on="method_name", how="left")
    _save_table(table6, "table_06_drone_utilization", tables_dir, "Drone utilization and structural behavior", "tab:utilization")

    nils = report_df[report_df["method_name"] == "nils"]
    table7 = (
        nils.groupby(["endurance_level", "speed_ratio"], dropna=False)
        .agg(
            avg_objective=("objective_for_reporting", "mean"),
            avg_improvement=("improvement_vs_truck_only_reported_pct", "mean"),
            avg_drone_share=("drone_service_share", "mean"),
        )
        .reset_index()
        .sort_values(["endurance_level", "speed_ratio"])
    )
    _save_table(table7, "table_07_endurance_speed_sensitivity", tables_dir, "Sensitivity to endurance and speed ratio", "tab:end_speed")

    table8_methods = ["nils", "paired_baseline", "no_unpairing", "simple_drone"]
    t8_pool = report_df[report_df["method_name"].isin(table8_methods)].copy()
    table8 = (
        t8_pool.groupby(["launch_retrieval_time_class", "method_name"], dropna=False)
        .agg(
            n_runs=("instance_id", "count"),
            waiting_cost=("waiting_sync_cost", "mean"),
            avg_truck_waiting_time=("total_truck_waiting_time", "mean"),
            avg_drone_waiting_time=("total_drone_waiting_time", "mean"),
            avg_synchronization_delay=("average_synchronization_delay", "mean"),
            avg_improvement=("improvement_vs_truck_only_reported_pct", "mean"),
            avg_launches=("number_of_drone_launches", "mean"),
            feasibility_rate=("final_solution_feasibility_flag", "mean"),
        )
        .reset_index()
    )
    table8["avg_total_waiting_time"] = table8["avg_truck_waiting_time"] + table8["avg_drone_waiting_time"]
    table8 = table8.sort_values(["launch_retrieval_time_class", "method_name"])
    _save_table(
        table8,
        "table_08_handling_sensitivity",
        tables_dir,
        "Sensitivity to handling/synchronization burden by method (feasibility-aware)",
        "tab:handling",
    )

    table9 = (
        nils.groupby(["eligible_share_target", "spatial_pattern"], dropna=False)
        .agg(
            objective=("objective_for_reporting", "mean"),
            improvement=("improvement_vs_truck_only_reported_pct", "mean"),
            drone_share=("drone_service_share", "mean"),
            truck_distance_reduction=("truck_distance_reduction_pct", "mean"),
        )
        .reset_index()
        .sort_values(["eligible_share_target", "spatial_pattern"])
    )
    _save_table(table9, "table_09_eligibility_spatial_sensitivity", tables_dir, "Sensitivity to eligibility and spatial pattern", "tab:elig_spatial")

    table10_order = [
        "nils",
        "nils_soft_drone_target",
        "paired_baseline",
        "no_unpairing",
        "nils_no_perturbation",
        "nils_no_local_search",
        "nils_no_battery_screening",
        "simple_drone",
        "random_feasible",
    ]
    table10_pool = report_df[report_df["method_name"].isin(table10_order)].copy()
    table10 = (
        table10_pool.groupby("method_name", dropna=False)
        .agg(
            n_runs=("instance_id", "count"),
            avg_objective=("objective_for_reporting", "mean"),
            avg_improvement=("improvement_vs_truck_only_reported_pct", "mean"),
            avg_cpu_time=("cpu_time_seconds", "mean"),
            feasibility_rate=("final_solution_feasibility_flag", "mean"),
        )
        .reset_index()
    )
    table10["method_name"] = pd.Categorical(table10["method_name"], categories=table10_order, ordered=True)
    table10 = table10.sort_values("method_name").reset_index(drop=True)
    _save_table(table10, "table_10_algorithmic_ablation", tables_dir, "Algorithmic ablation", "tab:ablation")

    scenario_method_candidates = report_df[report_df["method_name"].isin(["nils", "paired_baseline", "no_unpairing"])].copy()
    if scenario_method_candidates.empty:
        scenario_analysis_method = "nils"
    else:
        wait_by_method = (
            scenario_method_candidates.groupby("method_name", dropna=False)["waiting_sync_cost"].mean().sort_values(ascending=False)
        )
        scenario_analysis_method = str(wait_by_method.index[0])
    scenario_source = report_df[report_df["method_name"] == scenario_analysis_method].copy()
    scenario_agg = (
        scenario_source.groupby(
            [
                "scenario_id",
                "n",
                "eligible_share_target",
                "number_of_drones",
                "endurance_level",
                "speed_ratio",
                "launch_retrieval_time_class",
                "spatial_pattern",
            ],
            dropna=False,
        )
        .agg(
            avg_improvement=("improvement_vs_truck_only_reported_pct", "mean"),
            drone_service_share=("drone_service_share", "mean"),
            truck_distance_reduction_pct=("truck_distance_reduction_pct", "mean"),
            total_truck_waiting_time=("total_truck_waiting_time", "mean"),
            total_drone_waiting_time=("total_drone_waiting_time", "mean"),
            delayed_rendezvous_events=("delayed_rendezvous_events", "mean"),
            average_battery_usage_ratio=("average_battery_usage_ratio", "mean"),
        )
        .reset_index()
    )
    scenario_agg["analysis_method"] = scenario_analysis_method
    ranked = scenario_agg.sort_values("avg_improvement", ascending=False).reset_index(drop=True)
    k = min(5, max(1, len(ranked) // 2))
    top = ranked.head(k).copy()
    top["segment"] = "best"
    bottom = ranked.tail(k).copy()
    bottom["segment"] = "worst"
    if len(ranked) <= 2 * k:
        top_ids = set(top["scenario_id"])
        bottom = bottom[~bottom["scenario_id"].isin(top_ids)]
    table11 = pd.concat([top, bottom], ignore_index=True)
    table11["dominant_reason"] = table11.apply(_dominant_reason, axis=1)
    _save_table(table11, "table_11_best_worst_scenarios", tables_dir, "Best/worst scenario characterization", "tab:best_worst")

    stats_rows = []
    pivot = report_df.pivot_table(index="instance_id", columns="method_name", values="objective_for_reporting", aggfunc="first")
    if "truck_only" in pivot.columns:
        base = pivot["truck_only"]
        for method in [c for c in pivot.columns if c != "truck_only"]:
            valid = base.notna() & pivot[method].notna()
            if valid.sum() == 0:
                continue
            imp = 100.0 * (base[valid] - pivot.loc[valid, method]) / base[valid]
            ci_low, ci_high = bootstrap_ci(imp.values.tolist(), n_boot=1000, alpha=0.05, seed=2026)
            cmp = paired_comparison(base[valid].values.tolist(), pivot.loc[valid, method].values.tolist())
            stats_rows.append(
                {
                    "method": method,
                    "paired_test": cmp.test,
                    "test_statistic": cmp.statistic,
                    "p_value": cmp.p_value,
                    "effect_size": cmp.effect_size,
                    "effect_size_name": cmp.effect_name,
                    "n_pairs": cmp.n,
                    "mean_improvement_pct": float(np.mean(imp)),
                    "median_improvement_pct": float(np.median(imp)),
                    "ci_low_pct": ci_low,
                    "ci_high_pct": ci_high,
                    "min_improvement_pct": float(np.min(imp)),
                    "max_improvement_pct": float(np.max(imp)),
                }
            )
    if stats_rows:
        table_stats = pd.DataFrame(stats_rows).sort_values("mean_improvement_pct", ascending=False)
        _save_table(table_stats, "table_12_paired_statistics", tables_dir, "Paired statistical comparisons vs truck-only", "tab:paired_stats")

    if "priority_weight_profile" in report_df.columns:
        table13 = (
            report_df.groupby(["priority_weight_profile", "method_name"], dropna=False)
            .agg(
                n_runs=("instance_id", "count"),
                avg_objective=("objective_for_reporting", "mean"),
                avg_improvement_vs_truck_only=("improvement_vs_truck_only_reported_pct", "mean"),
                avg_priority_tardiness_cost=("priority_tardiness_cost", "mean"),
                avg_waiting_cost=("waiting_sync_cost", "mean"),
                avg_drone_service_share=("drone_service_share", "mean"),
                feasibility_rate=("final_solution_feasibility_flag", "mean"),
            )
            .reset_index()
            .sort_values(["priority_weight_profile", "method_name"])
        )
        _save_table(
            table13,
            "table_13_priority_weight_sensitivity",
            tables_dir,
            "Sensitivity to priority-weight profiles by method",
            "tab:priority_weight_sensitivity",
        )

    # Fleet-composition sensitivity under fixed truck fleet (|K| fixed in this study).
    fleet_src = report_df[report_df["method_name"] == "nils"].copy()
    if not fleet_src.empty and {"number_of_drones", "number_of_trucks"}.issubset(set(fleet_src.columns)):
        agg_map = {
            "n_runs": ("instance_id", "count"),
            "number_of_trucks": ("number_of_trucks", "mean"),
            "avg_objective": ("objective_for_reporting", "mean"),
            "avg_improvement_pct": ("improvement_vs_truck_only_reported_pct", "mean"),
            "drone_share": ("drone_service_share", "mean"),
            "avg_cpu_s": ("cpu_time_seconds", "mean"),
            "feasibility_rate": ("final_solution_feasibility_flag", "mean"),
        }
        if "high_priority_tardiness_minutes" in fleet_src.columns:
            agg_map["high_priority_tardiness"] = ("high_priority_tardiness_minutes", "mean")

        table14 = (
            fleet_src.groupby("number_of_drones", dropna=False)
            .agg(**agg_map)
            .reset_index()
            .sort_values("number_of_drones")
        )
        table14["drone_to_truck_ratio"] = table14.apply(
            lambda r: _safe_div(float(r["number_of_drones"]), float(r["number_of_trucks"])),
            axis=1,
        )
        table14["drone_to_truck_ratio"] = table14["drone_to_truck_ratio"].round(2)
        _save_table(
            table14,
            "table_14_drone_fleet_sensitivity",
            tables_dir,
            "Sensitivity of NILS to drone availability under the fixed truck fleet",
            "tab:drone_fleet_sensitivity",
        )

    # Beta-slack sensitivity for NILS.
    slack_src = report_df[report_df["method_name"] == "nils"].copy()
    if not slack_src.empty and "battery_slack_ratio" in slack_src.columns:
        table15 = (
            slack_src.groupby("battery_slack_ratio", dropna=False)
            .agg(
                n_runs=("instance_id", "count"),
                avg_objective=("objective_for_reporting", "mean"),
                avg_improvement_pct=("improvement_vs_truck_only_reported_pct", "mean"),
                drone_share=("drone_service_share", "mean"),
                battery_rejections=("infeasible_battery_attempts_rejected", "mean"),
                avg_cpu_s=("cpu_time_seconds", "mean"),
                feasible_runs_pct=("final_solution_feasibility_flag", "mean"),
                avg_drone_distance=("drone_flight_distance_km", "mean"),
            )
            .reset_index()
            .sort_values("battery_slack_ratio")
        )
        _save_table(
            table15,
            "table_15_beta_slack_sensitivity",
            tables_dir,
            "Sensitivity of NILS to the residual battery-slack threshold",
            "tab:beta_slack_sensitivity",
        )

    # Fixed-n fleet-ratio diagnostics: same n, vary drones (clean fleet-ratio view).
    fleet_methods = [m for m in ["nils", "nils_soft_drone_target"] if m in report_df["method_name"].unique()]
    if fleet_methods:
        fleet_src = report_df[report_df["method_name"].isin(fleet_methods)].copy()
        fleet_group = (
            fleet_src.groupby(["method_name", "n", "number_of_drones", "number_of_trucks"], dropna=False)
            .agg(
                n_runs=("instance_id", "count"),
                avg_objective=("objective_for_reporting", "mean"),
                avg_improvement_pct=("improvement_vs_truck_only_reported_pct", "mean"),
                avg_drone_share=("drone_service_share", "mean"),
                avg_cpu_s=("cpu_time_seconds", "mean"),
                feasibility_rate=("final_solution_feasibility_flag", "mean"),
            )
            .reset_index()
        )
        fleet_group["drone_to_truck_ratio"] = fleet_group.apply(
            lambda r: _safe_div(float(r["number_of_drones"]), float(r["number_of_trucks"])),
            axis=1,
        )

        complete_keys = (
            fleet_group.groupby(["method_name", "n"], dropna=False)["number_of_drones"]
            .nunique()
            .reset_index(name="num_drone_levels")
        )
        complete_keys = complete_keys[complete_keys["num_drone_levels"] >= 3][["method_name", "n"]]
        if not complete_keys.empty:
            fleet_group = fleet_group.merge(complete_keys, on=["method_name", "n"], how="inner")
        fleet_group = fleet_group.sort_values(["method_name", "n", "number_of_drones"])
        _save_table(
            fleet_group,
            "table_21_fleet_ratio_fixed_n",
            tables_dir,
            "Fixed-n fleet-ratio sensitivity (same n, varying number of drones).",
            "tab:fleet_ratio_fixed_n",
        )

        # Marginal gains from adding drones at fixed n.
        pivot = fleet_group.pivot_table(
            index=["method_name", "n"],
            columns="number_of_drones",
            values="avg_objective",
            aggfunc="first",
        )
        if not pivot.empty:
            marg = pivot.reset_index()
            if 1 in marg.columns and 2 in marg.columns:
                marg["objective_improvement_2_vs_1_pct"] = 100.0 * (marg[1] - marg[2]) / marg[1].replace(0.0, np.nan)
            else:
                marg["objective_improvement_2_vs_1_pct"] = float("nan")
            if 2 in marg.columns and 3 in marg.columns:
                marg["objective_improvement_3_vs_2_pct"] = 100.0 * (marg[2] - marg[3]) / marg[2].replace(0.0, np.nan)
            else:
                marg["objective_improvement_3_vs_2_pct"] = float("nan")
            _save_table(
                marg,
                "table_21b_fleet_ratio_marginal_effects",
                tables_dir,
                "Marginal effects of additional drones at fixed n.",
                "tab:fleet_ratio_marginal",
            )

    # Regime-specific gains: favorable vs harsh operating conditions.
    regime_methods = [m for m in ["nils", "nils_soft_drone_target", "paired_baseline", "no_unpairing"] if m in report_df["method_name"].unique()]
    if regime_methods:
        reg = report_df[report_df["method_name"].isin(regime_methods + ["truck_only"])].copy()
        reg["regime_class"] = "neutral"
        favorable_mask = (
            (reg["eligible_share_target"] >= 0.75)
            & (reg["speed_ratio"] >= 2.0)
            & (reg["launch_retrieval_time_class"].astype(str).str.lower() == "short")
            & (reg["endurance_level"].astype(str).str.lower().isin(["medium", "high"]))
        )
        harsh_mask = (
            (reg["eligible_share_target"] <= 0.25)
            & (reg["speed_ratio"] <= 1.25)
            & (reg["launch_retrieval_time_class"].astype(str).str.lower() == "long")
            & (reg["endurance_level"].astype(str).str.lower().isin(["low", "medium"]))
        )
        reg.loc[favorable_mask, "regime_class"] = "favorable"
        reg.loc[harsh_mask, "regime_class"] = "harsh"

        regime_table = (
            reg[reg["method_name"].isin(regime_methods)]
            .groupby(["regime_class", "method_name"], dropna=False)
            .agg(
                n_runs=("instance_id", "count"),
                avg_objective=("objective_for_reporting", "mean"),
                avg_improvement_pct=("improvement_vs_truck_only_reported_pct", "mean"),
                avg_drone_share=("drone_service_share", "mean"),
                avg_cpu_s=("cpu_time_seconds", "mean"),
                feasibility_rate=("final_solution_feasibility_flag", "mean"),
            )
            .reset_index()
            .sort_values(["regime_class", "method_name"])
        )
        _save_table(
            regime_table,
            "table_22_regime_specific_gains",
            tables_dir,
            "Regime-specific gains under favorable/harsh/neutral conditions.",
            "tab:regime_specific_gains",
        )


def _heatmap(
    pivot: pd.DataFrame,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str = "viridis",
) -> None:
    arr = pivot.values.astype(float)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(arr, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(v) for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Avg % improvement vs truck-only")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _drone_sortie_nodes(solution: SolutionData, drone_id: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    launches: List[Tuple[float, int, int]] = []
    recoveries: List[Tuple[float, int, int]] = []
    for (node, drone, truck), val in solution.z1.items():
        if drone != drone_id or val <= 0.5:
            continue
        ts = float(solution.a_drone.get((drone_id, node), solution.a_truck.get((truck, node), float(node))))
        launches.append((ts, int(node), int(truck)))
    for (node, drone, truck), val in solution.z2.items():
        if drone != drone_id or val <= 0.5:
            continue
        ts = float(solution.a_drone.get((drone_id, node), solution.a_truck.get((truck, node), float(node))))
        recoveries.append((ts, int(node), int(truck)))
    launches.sort(key=lambda x: x[0])
    recoveries.sort(key=lambda x: x[0])
    return [(node, truck) for _, node, truck in launches], [(node, truck) for _, node, truck in recoveries]


def _plot_routes(ax, instance: InstanceData, solution: SolutionData, title: str) -> None:
    coords = instance.coordinate_map()
    xs = [coords[i][0] for i in instance.customer_ids]
    ys = [coords[i][1] for i in instance.customer_ids]
    ax.scatter(xs, ys, s=20, c="tab:gray", alpha=0.8, label="Customers")
    ax.scatter([0.0], [0.0], s=80, c="black", marker="s", label="Depot")
    for truck_id, route in solution.truck_routes.items():
        if len(route) < 2:
            continue
        px = [coords[n][0] for n in route]
        py = [coords[n][1] for n in route]
        ax.plot(px, py, linewidth=1.5, alpha=0.9, label=f"Truck {truck_id}")
    for drone_id, route in solution.drone_routes.items():
        customers = [node for node in route if node != 0]
        if len(route) < 2 or not customers:
            continue
        launches, recoveries = _drone_sortie_nodes(solution, drone_id)
        color = plt.cm.tab10((drone_id - 1) % 10)
        if launches or recoveries:
            labeled = False
            for idx, customer in enumerate(customers):
                launch_node, launch_truck = launches[idx] if idx < len(launches) else (launches[-1] if launches else (0, 0))
                recovery_node, recovery_truck = (
                    recoveries[idx] if idx < len(recoveries) else (recoveries[-1] if recoveries else (0, 0))
                )
                if launch_node not in coords or customer not in coords or recovery_node not in coords:
                    continue
                style = ":" if launch_truck > 0 and recovery_truck > 0 and launch_truck != recovery_truck else "--"
                lbl = f"Drone {drone_id} sortie" if not labeled else None
                ax.plot(
                    [coords[launch_node][0], coords[customer][0]],
                    [coords[launch_node][1], coords[customer][1]],
                    linewidth=1.25,
                    alpha=0.9,
                    linestyle=style,
                    color=color,
                    label=lbl,
                )
                ax.plot(
                    [coords[customer][0], coords[recovery_node][0]],
                    [coords[customer][1], coords[recovery_node][1]],
                    linewidth=1.25,
                    alpha=0.9,
                    linestyle=style,
                    color=color,
                )
                ax.scatter([coords[launch_node][0]], [coords[launch_node][1]], s=22, color=color, marker="^", alpha=0.8)
                ax.scatter([coords[recovery_node][0]], [coords[recovery_node][1]], s=22, color=color, marker="v", alpha=0.8)
                labeled = True
        else:
            px = [coords[n][0] for n in route]
            py = [coords[n][1] for n in route]
            ax.plot(px, py, linewidth=1.2, alpha=0.8, linestyle="--", color=color, label=f"Drone {drone_id} route")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.text(
        0.01,
        0.01,
        "Drone sorties: dashed=same-truck/depot recovery, dotted=cross-truck recovery",
        transform=ax.transAxes,
        fontsize=7,
        alpha=0.7,
    )


def _generate_figures(
    df: pd.DataFrame,
    out_dir: Path,
    instance_cache: Dict[str, InstanceData],
    solution_cache: Dict[Tuple[str, str], SolutionData],
) -> None:
    if df.empty:
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_df = df[df["reporting_included"]].copy() if "reporting_included" in df.columns else df.copy()
    if report_df.empty:
        report_df = df.copy()

    fig1 = report_df[report_df["method_name"] != "truck_only"][["method_name", "improvement_vs_truck_only_reported_pct"]].dropna()
    if not fig1.empty:
        methods = sorted(fig1["method_name"].unique())
        data = [fig1[fig1["method_name"] == m]["improvement_vs_truck_only_reported_pct"].values for m in methods]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.boxplot(data, labels=methods)
        ax.set_ylabel("% improvement vs truck-only")
        ax.set_xlabel("Method")
        ax.set_title("Figure 1. Improvement Distribution vs Truck-only")
        fig.tight_layout()
        fig.savefig(fig_dir / "figure_01_improvement_distribution.png")
        plt.close(fig)

    fig2 = report_df.groupby(["n", "method_name"], dropna=False)["objective_for_reporting"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, grp in fig2.groupby("method_name"):
        ax.plot(grp["n"], grp["objective_for_reporting"], marker="o", label=method)
    ax.set_xlabel("Number of customers (n)")
    ax.set_ylabel("Average objective (reporting policy)")
    ax.set_title("Figure 2. Objective vs Problem Size (Feasibility-aware)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "figure_02_objective_vs_size.png")
    plt.close(fig)

    fig3 = df.groupby(["n", "method_name"], dropna=False)["cpu_time_seconds"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, grp in fig3.groupby("method_name"):
        ax.plot(grp["n"], grp["cpu_time_seconds"], marker="o", label=method)
    ax.set_xlabel("Number of customers (n)")
    ax.set_ylabel("Average runtime (s)")
    ax.set_title("Figure 3. CPU Time vs Problem Size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "figure_03_runtime_vs_size.png")
    plt.close(fig)

    nils = report_df[report_df["method_name"] == "nils"]
    if not nils.empty:
        p4 = nils.pivot_table(
            index="endurance_level",
            columns="speed_ratio",
            values="improvement_vs_truck_only_reported_pct",
            aggfunc="mean",
        ).sort_index()
        _heatmap(
            p4,
            fig_dir / "figure_04_endurance_speed_heatmap.png",
            "Figure 4. Improvement by Endurance and Speed Ratio",
            "Speed ratio (v_D / v_T)",
            "Endurance level",
        )

    if not nils.empty:
        p5 = nils.pivot_table(
            index="eligible_share_target",
            columns="spatial_pattern",
            values="improvement_vs_truck_only_reported_pct",
            aggfunc="mean",
        ).sort_index()
        _heatmap(
            p5,
            fig_dir / "figure_05_eligibility_spatial_heatmap.png",
            "Figure 5. Improvement by Eligibility and Spatial Pattern",
            "Spatial pattern",
            "Eligibility share",
        )

    comp = (
        report_df[report_df["method_name"].isin(["truck_only", "simple_drone", "nils"])]
        .groupby("method_name", dropna=False)[
            ["truck_travel_cost", "drone_travel_cost", "waiting_sync_cost", "priority_tardiness_cost"]
        ]
        .mean()
    )
    if not comp.empty:
        fig, (ax_abs, ax_share) = plt.subplots(1, 2, figsize=(13, 5))
        cols = ["truck_travel_cost", "drone_travel_cost", "waiting_sync_cost", "priority_tardiness_cost"]
        x = np.arange(len(comp.index))
        bottom_abs = np.zeros(len(comp))
        for col in cols:
            vals = comp[col].values
            ax_abs.bar(x, vals, bottom=bottom_abs, label=col)
            bottom_abs = bottom_abs + vals
        totals = comp[cols].sum(axis=1).replace(0.0, np.nan)
        comp_share = comp[cols].div(totals, axis=0).fillna(0.0) * 100.0
        bottom_share = np.zeros(len(comp_share))
        for col in cols:
            vals = comp_share[col].values
            ax_share.bar(x, vals, bottom=bottom_share, label=col)
            bottom_share = bottom_share + vals
        ax_abs.set_xticks(x)
        ax_abs.set_xticklabels(comp.index.tolist())
        ax_abs.set_ylabel("Average cost")
        ax_abs.set_title("Absolute")
        ax_share.set_xticks(x)
        ax_share.set_xticklabels(comp.index.tolist())
        ax_share.set_ylabel("Component share (%)")
        ax_share.set_title("Percentage share")
        fig.suptitle("Figure 6. Cost Components (Absolute and Share)")
        ax_abs.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(fig_dir / "figure_06_stacked_cost_components.png")
        plt.close(fig)

    fig7 = report_df[report_df["method_name"] != "truck_only"][["drone_service_share", "truck_distance_reduction_pct"]].dropna()
    if not fig7.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(fig7["drone_service_share"], fig7["truck_distance_reduction_pct"], alpha=0.5)
        ax.set_xlabel("Drone-service share")
        ax.set_ylabel("% truck-distance reduction")
        ax.set_title("Figure 7. Truck-distance Reduction vs Drone-service Share")
        fig.tight_layout()
        fig.savefig(fig_dir / "figure_07_distance_reduction_scatter.png")
        plt.close(fig)

    fig8 = (
        report_df[report_df["method_name"].isin(["simple_drone", "nils", "paired_baseline", "no_unpairing"])]
        .groupby(["launch_retrieval_time_class", "method_name"], dropna=False)["improvement_vs_truck_only_reported_pct"]
        .mean()
        .reset_index()
    )
    if not fig8.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for method, grp in fig8.groupby("method_name"):
            ax.plot(grp["launch_retrieval_time_class"], grp["improvement_vs_truck_only_reported_pct"], marker="o", label=method)
        ax.set_xlabel("Handling time class")
        ax.set_ylabel("% improvement vs truck-only")
        ax.set_title("Figure 8. Synchronization Penalty Effect")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "figure_08_sync_penalty_effect.png")
        plt.close(fig)

    candidate_methods = ["nils", "paired_baseline", "no_unpairing", "simple_drone"]
    method_scores: List[Tuple[int, int, str]] = []
    for idx, method in enumerate(candidate_methods):
        sols = [sol for (iid, m), sol in solution_cache.items() if m == method]
        if not sols:
            continue
        sortie_events = sum(
            sum(1 for v in sol.z1.values() if v > 0.5) + sum(1 for v in sol.z2.values() if v > 0.5) for sol in sols
        )
        method_scores.append((sortie_events, -idx, method))
    route_method = max(method_scores)[2] if method_scores else "nils"

    method_rows = report_df[report_df["method_name"] == route_method][
        ["instance_id", "improvement_vs_truck_only_reported_pct"]
    ].dropna()
    if len(method_rows) >= 3:
        ranked = method_rows.sort_values("improvement_vs_truck_only_reported_pct")
        picks = [
            ranked.iloc[-1]["instance_id"],
            ranked.iloc[len(ranked) // 2]["instance_id"],
            ranked.iloc[0]["instance_id"],
        ]
        titles = [
            f"High-benefit case ({route_method})",
            f"Moderate-benefit case ({route_method})",
            f"Low-benefit case ({route_method})",
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        for ax, iid, title in zip(axes, picks, titles):
            inst = instance_cache.get(iid)
            sol = solution_cache.get((iid, route_method))
            if inst is not None and sol is not None:
                _plot_routes(ax, inst, sol, f"{title}\n{iid}")
            else:
                ax.set_title(f"{title}\n{iid} (missing)")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=4)
        fig.tight_layout()
        fig.savefig(fig_dir / "figure_09_representative_routes.png")
        plt.close(fig)

    fig10 = nils["average_battery_usage_ratio"].dropna()
    if not fig10.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(fig10.values, bins=20, alpha=0.8)
        ax.set_xlabel("Average battery usage ratio")
        ax.set_ylabel("Frequency")
        ax.set_title("Figure 10. Battery Utilization Profile")
        fig.tight_layout()
        fig.savefig(fig_dir / "figure_10_battery_profile.png")
        plt.close(fig)


def _generate_feasibility_disaggregation(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty or "final_solution_feasibility_flag" not in df.columns:
        return
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    work = df.copy()
    work["_feas"] = (
        work["final_solution_feasibility_flag"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0})
        .fillna(0)
        .astype(int)
    )

    # Table A: feasibility by method x scenario class.
    by_scenario = (
        work.groupby(["method_name", "scenario_class"], dropna=False)["_feas"]
        .agg(n_runs="count", feasible_runs="sum")
        .reset_index()
    )
    by_scenario["feasibility_rate"] = by_scenario["feasible_runs"] / by_scenario["n_runs"].clip(lower=1)
    by_scenario = by_scenario.sort_values(["method_name", "feasibility_rate", "n_runs"], ascending=[True, True, False])
    by_scenario.to_csv(tables_dir / "table_19_feasibility_by_scenario_class.csv", index=False)
    dataframe_to_latex(
        by_scenario,
        tables_dir / "table_19_feasibility_by_scenario_class.tex",
        caption="Feasibility rates by method and scenario class.",
        label="tab:feasibility_by_scenario_class",
    )

    # Table B: NILS feasibility by regime (endurance, eligibility share, fleet size).
    nils = work[work["method_name"].astype(str).str.lower() == "nils"].copy()
    if nils.empty:
        return
    regime_cols = [
        "endurance_level",
        "eligible_share_target",
        "n",
        "number_of_drones",
        "priority_weight_profile",
        "battery_slack_ratio",
    ]
    regime_cols = [c for c in regime_cols if c in nils.columns]
    if not regime_cols:
        return
    by_regime = (
        nils.groupby(regime_cols, dropna=False)["_feas"]
        .agg(n_runs="count", feasible_runs="sum")
        .reset_index()
    )
    by_regime["feasibility_rate"] = by_regime["feasible_runs"] / by_regime["n_runs"].clip(lower=1)
    by_regime = by_regime.sort_values(["feasibility_rate", "n_runs"], ascending=[True, False])
    by_regime.to_csv(tables_dir / "table_20_nils_feasibility_by_regime.csv", index=False)
    dataframe_to_latex(
        by_regime,
        tables_dir / "table_20_nils_feasibility_by_regime.tex",
        caption="NILS feasibility by endurance level, eligibility share, customer size, and drone fleet size.",
        label="tab:nils_feasibility_by_regime",
    )


def run_heuristic_study(config: SearchConfig, output_dir: str | None = None) -> pd.DataFrame:
    """Run the full heuristic-only benchmark study with baselines and ablations."""
    exp_cfg = dict(config.experiment or {})
    target_dir = Path(output_dir or str(exp_cfg.get("output_dir", "outputs")))
    target_dir.mkdir(parents=True, exist_ok=True)
    progress_path = _build_progress_writer(target_dir)

    levels = _as_levels(config)
    scenarios = _build_scenarios(levels, config)
    methods = _study_methods(config)
    reps = int(exp_cfg.get("study_reps_per_scenario", exp_cfg.get("study_reps", 20)))
    reps = max(1, reps)

    design = str(exp_cfg.get("study_design", "full_factorial")).strip().lower()
    report_policy = str(exp_cfg.get("study_feasibility_policy", "feasible_only")).strip().lower()
    _log(
        progress_path,
        f"heuristic study start | scenarios={len(scenarios)} | reps={reps} | methods={methods} | "
        f"design={design} | reporting_policy={report_policy}",
    )

    rows: List[Dict[str, object]] = []
    instance_cache: Dict[str, InstanceData] = {}
    solution_cache: Dict[Tuple[str, str], SolutionData] = {}
    regen_max_attempts = max(1, int(exp_cfg.get("study_instance_regen_max_attempts", 60)))
    force_feasible_outputs = bool(exp_cfg.get("study_force_feasible_outputs", True))
    output_repair_steps = max(1, int(exp_cfg.get("study_output_repair_max_steps", 8)))
    raw_probe_initializers = exp_cfg.get("study_preprocess_probe_initializers", ["ortools", "greedy"])
    if not isinstance(raw_probe_initializers, list) or not raw_probe_initializers:
        raw_probe_initializers = ["ortools", "greedy"]
    preprocess_probe_initializers = [str(v).strip().lower() for v in raw_probe_initializers if str(v).strip()]
    if not preprocess_probe_initializers:
        preprocess_probe_initializers = ["ortools", "greedy"]
    preprocess_probe_seed_tries = max(1, int(exp_cfg.get("study_preprocess_probe_seed_tries", 3)))
    preprocess_ortools_time_limit_seconds = max(
        1, int(exp_cfg.get("study_preprocess_ortools_time_limit_seconds", 8))
    )

    raw_dir = target_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for s_idx, scenario in enumerate(scenarios, start=1):
        bar = f"[{s_idx}/{len(scenarios)}]"
        _log(
            progress_path,
            f"{bar} scenario={scenario['scenario_id']} n={scenario['n']} drones={scenario['drones_available']} "
            f"eligible={scenario['eligible_share']} end={scenario['endurance']} ratio={scenario['speed_ratio']} "
            f"handling={scenario['handling_time']} spatial={scenario['spatial_pattern']} "
            f"pweights={scenario.get('priority_weight_profile', 'base')} "
            f"beta={float(scenario.get('battery_slack_ratio', 0.15)):.2f}",
        )
        scenario_cfg = _scenario_config(config, scenario)
        generator = InstanceGenerator.from_search_config(scenario_cfg)
        for rep in range(1, reps + 1):
            base_seed = int(config.seed + s_idx * 100_000 + rep)
            instance_id = f"{scenario['scenario_id']}_R{rep:02d}"
            instance: InstanceData | None = None
            inst_seed = base_seed
            for regen_try in range(regen_max_attempts):
                inst_seed = int(base_seed + regen_try * 1_000_003)
                candidate = generator.generate_single(
                    seed=inst_seed,
                    name=instance_id,
                    overrides={
                        "num_customers": int(scenario["n"]),
                        "num_drones": int(scenario["drones_available"]),
                        "swap_time_s": _handling_seconds(scenario["handling_time"]),
                        "reload_time_s": _handling_seconds(scenario["handling_time"]),
                    },
                )
                _apply_eligibility_mask(candidate, float(scenario["eligible_share"]), inst_seed)
                candidate.metadata.update(
                    {
                        "scenario_id": scenario["scenario_id"],
                        "spatial_pattern": scenario["spatial_pattern"],
                        "endurance_level": scenario["endurance"],
                        "speed_ratio": scenario["speed_ratio"],
                        "handling_time_class": scenario["handling_time"],
                        "drones_available": scenario["drones_available"],
                        "priority_weight_profile": str(scenario.get("priority_weight_profile", "base")).strip().lower(),
                        "battery_slack_ratio": float(scenario.get("battery_slack_ratio", 0.15)),
                        "regen_attempt": int(regen_try),
                    }
                )
                if not _passes_preprocessing_feasibility(candidate):
                    continue

                probe_ok = False
                for init_name in preprocess_probe_initializers:
                    for seed_try in range(preprocess_probe_seed_tries):
                        probe_seed = int(inst_seed + (seed_try + 1) * 7919 + (len(init_name) + 1) * 101)
                        probe = build_initial_solution(
                            candidate,
                            probe_seed,
                            initializer=init_name,
                            ortools_time_limit_seconds=preprocess_ortools_time_limit_seconds,
                        )
                        if is_feasible(candidate, probe):
                            probe_ok = True
                            break
                    if probe_ok:
                        break
                if not probe_ok:
                    legacy_probe = truck_only_baseline(candidate, scenario_cfg).solution
                    if not is_feasible(candidate, legacy_probe):
                        continue
                instance = candidate
                break
            if instance is None:
                raise RuntimeError(
                    f"Unable to generate preprocessing-feasible instance after {regen_max_attempts} attempts "
                    f"for {instance_id}"
                )

            generator.save(instance, raw_dir / f"{instance_id}.json")
            instance_cache[instance_id] = instance

            for method_name in methods:
                _log(progress_path, f"  {instance_id} | method={method_name} | start")
                t0 = time.time()
                try:
                    solution = _run_method(instance, scenario_cfg, method_name)
                    if force_feasible_outputs and not is_feasible(instance, solution):
                        repaired = _repair_infeasible_solution(
                            instance,
                            solution,
                            max_steps=max(output_repair_steps, int(instance.num_customers)),
                        )
                        if repaired is not None and is_feasible(instance, repaired):
                            solution = repaired
                            solution.objective, solution.components = evaluate_solution(instance, solution)
                            solution.status = f"{solution.status}|repaired_output"
                        else:
                            fallback = truck_only_baseline(instance, scenario_cfg).solution
                            fallback.objective, fallback.components = evaluate_solution(instance, fallback)
                            fallback.status = f"{method_name}|fallback_truck_only"
                            solution = fallback
                    solution.run_time_seconds = float(solution.run_time_seconds or (time.time() - t0))
                    row = _collect_row(instance, solution, scenario, method_name)
                    row["cpu_time_seconds"] = float(time.time() - t0)
                    rows.append(row)
                    if method_name in {"nils", "paired_baseline", "no_unpairing", "truck_only", "simple_drone"}:
                        solution_cache[(instance_id, method_name)] = solution
                    _log(
                        progress_path,
                        f"  {instance_id} | method={method_name} | done obj={row['objective_total']:.4f} "
                        f"time={row['cpu_time_seconds']:.2f}s",
                    )
                except Exception as ex:
                    rows.append(
                        {
                            "instance_id": instance_id,
                            "random_seed": inst_seed,
                            "method_name": method_name,
                            "scenario_id": scenario["scenario_id"],
                            "scenario_class": (
                                f"n{scenario['n']}_p{float(scenario['eligible_share']):.2f}_d{scenario['drones_available']}_"
                                f"{scenario['spatial_pattern']}_E{scenario['endurance']}_V{float(scenario['speed_ratio']):.2f}_"
                                f"H{scenario['handling_time']}_PW{str(scenario.get('priority_weight_profile', 'base')).strip().lower()}"
                                f"_B{float(scenario.get('battery_slack_ratio', 0.15)):.2f}"
                            ),
                            "n": int(scenario["n"]),
                            "eligible_share": float(scenario["eligible_share"]),
                            "eligible_share_target": float(scenario["eligible_share"]),
                            "number_of_drones": int(scenario["drones_available"]),
                            "number_of_trucks": int(instance.num_trucks),
                            "endurance_level": str(scenario["endurance"]),
                            "speed_ratio": float(scenario["speed_ratio"]),
                            "spatial_pattern": str(scenario["spatial_pattern"]),
                            "launch_retrieval_time_class": str(scenario["handling_time"]),
                            "priority_weight_profile": str(scenario.get("priority_weight_profile", "base")).strip().lower(),
                            "battery_slack_ratio": float(scenario.get("battery_slack_ratio", 0.15)),
                            "priority_weight_high": float(
                                _priority_penalties_for_profile(
                                    scenario.get("priority_weight_profile", "base"),
                                    dict(instance.priority_penalties),
                                )["high"]
                            ),
                            "priority_weight_medium": float(
                                _priority_penalties_for_profile(
                                    scenario.get("priority_weight_profile", "base"),
                                    dict(instance.priority_penalties),
                                )["medium"]
                            ),
                            "priority_weight_low": float(
                                _priority_penalties_for_profile(
                                    scenario.get("priority_weight_profile", "base"),
                                    dict(instance.priority_penalties),
                                )["low"]
                            ),
                            "depot_position": str(scenario.get("depot_position", "peripheral")),
                            "objective_total": float("nan"),
                            "truck_travel_cost": float("nan"),
                            "drone_travel_cost": float("nan"),
                            "waiting_sync_cost": float("nan"),
                            "penalty_cost": float("nan"),
                            "priority_tardiness_cost": float("nan"),
                            "high_priority_tardiness_minutes": float("nan"),
                            "medium_priority_tardiness_minutes": float("nan"),
                            "low_priority_tardiness_minutes": float("nan"),
                            "total_priority_tardiness_minutes": float("nan"),
                            "high_priority_tardiness_weighted_cost": float("nan"),
                            "medium_priority_tardiness_weighted_cost": float("nan"),
                            "low_priority_tardiness_weighted_cost": float("nan"),
                            "fixed_deployment_cost": 0.0,
                            "makespan_minutes": float("nan"),
                            "truck_route_distance_km": float("nan"),
                            "drone_flight_distance_km": float("nan"),
                            "customers_served_by_truck": float("nan"),
                            "customers_served_by_drone": float("nan"),
                            "drone_service_share": float("nan"),
                            "number_of_drone_launches": float("nan"),
                            "number_of_drone_retrievals": float("nan"),
                            "average_customers_per_drone_sortie": float("nan"),
                            "average_customers_per_drone_route": float("nan"),
                            "total_truck_waiting_time": float("nan"),
                            "total_drone_waiting_time": float("nan"),
                            "average_synchronization_delay": float("nan"),
                            "maximum_synchronization_delay": float("nan"),
                            "delayed_rendezvous_events": float("nan"),
                            "average_battery_usage_ratio": float("nan"),
                            "maximum_battery_usage_ratio": float("nan"),
                            "sorties_near_endurance_limit": float("nan"),
                            "infeasible_battery_attempts_rejected": float("nan"),
                            "eligible_not_assigned_due_battery": float("nan"),
                            "cpu_time_seconds": float(time.time() - t0),
                            "iterations": float("nan"),
                            "improving_moves_accepted": float("nan"),
                            "candidate_reassignments_evaluated": float("nan"),
                            "final_solution_feasibility_flag": False,
                            "status": "error",
                            "run_label": f"failed: {ex}",
                        }
                    )
                    _log(progress_path, f"  {instance_id} | method={method_name} | failed error={ex}")

    df = pd.DataFrame(rows)
    df = _attach_relative_fields(df)
    df = _attach_reporting_fields(df, config)

    tables_dir = target_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "heuristic_study_runs.csv", index=False)

    _generate_tables(df, levels, target_dir)
    _generate_feasibility_disaggregation(df, target_dir)
    _generate_figures(df, target_dir, instance_cache, solution_cache)
    _log(progress_path, "heuristic study complete")
    return df
