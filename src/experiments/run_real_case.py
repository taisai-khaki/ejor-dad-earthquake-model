"""Illustrative real-data case study based on Delivery_Logistics.csv."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data_models import Customer, Drone, InstanceData, ProblemConstants, SolutionData, Truck
from ..feasibility import is_feasible
from ..heuristics import run_nils
from ..heuristics.baselines import (
    no_unpairing_baseline,
    paired_baseline,
    simple_drone_assignment_baseline,
    truck_only_baseline,
    nils_no_local_search_baseline,
    nils_no_perturbation_baseline,
)
from ..heuristics.construction import build_initial_solution
from ..heuristics.nils import _sync_route_assignments, evaluate_solution, move_to_drone, summarize_drone_usage
from ..distance_utils import euclidean_distance
from ..parameters import (
    ConstraintConfig,
    GenerationSettings,
    HeuristicConfig,
    MILPConfig,
    SearchConfig,
    VehicleCatalog,
    VehicleCostConfig,
    VehicleRuntimeConfig,
    VehicleSpec,
    DroneSpec,
)


MODE_PRIORITY = {
    "express": "high",
    "same day": "high",
    "two day": "medium",
    "standard": "low",
}

PRIORITY_WINDOWS = {
    "high": (0.0, 60.0),
    "medium": (0.0, 120.0),
    "low": (0.0, 240.0),
}


def _parse_expected_hours(value: object, fallback: float) -> float:
    text = str(value)
    if "." in text:
        tail = text.rsplit(".", 1)[-1]
        digits = "".join(ch for ch in tail if ch.isdigit())
        if digits:
            return max(1.0, float(int(digits)))
    try:
        return max(1.0, float(value))
    except Exception:
        return max(1.0, float(fallback))


def build_real_case_instance(
    csv_path: str | Path,
    *,
    seed: int = 2026,
    sample_per_mode: int = 20,
    distance_scale: float = 0.1,
    beta_slack_ratio: float = 0.10,
) -> tuple[InstanceData, pd.DataFrame]:
    """Build the illustrative case-study instance from the logistics CSV."""
    raw = pd.read_csv(csv_path)
    raw["delivery_mode_clean"] = raw["delivery_mode"].astype(str).str.strip().str.lower()
    modes = ["same day", "two day", "express", "standard"]
    sampled = []
    for mode in modes:
        part = raw[raw["delivery_mode_clean"] == mode].copy()
        if part.empty:
            continue
        sampled.append(part.sample(n=min(sample_per_mode, len(part)), random_state=seed + len(sampled)))
    if not sampled:
        raise ValueError("No recognized delivery modes found in real-case CSV.")
    data = pd.concat(sampled, ignore_index=True).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    customers: list[Customer] = []
    for idx, row in data.iterrows():
        mode = str(row["delivery_mode_clean"])
        priority = MODE_PRIORITY.get(mode, "low")
        distance = max(0.1, float(row.get("distance_km", 1.0))) * distance_scale
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        x = distance * math.cos(angle)
        y = distance * math.sin(angle)
        expected_h = _parse_expected_hours(row.get("expected_time_hours", 1.0), fallback=1.0)
        lb, default_ub = PRIORITY_WINDOWS[priority]
        ub = min(default_ub, max(15.0, expected_h * 60.0))
        customers.append(
            Customer(
                node_id=idx + 1,
                x=float(x),
                y=float(y),
                demand_kg=float(row.get("package_weight_kg", 1.0)),
                priority=priority,
                lb=lb,
                ub=float(ub),
                service_time_min=1.0,
            )
        )

    trucks = [
        Truck(truck_id=i, capacity_kg=5000.0, speed_kmph=35.0, cost_per_km=0.80)
        for i in range(1, 4)
    ]
    drones = [
        Drone(
            drone_id=i,
            capacity_kg=5.0,
            speed_kmph=65.0,
            max_battery_wh=1200.0,
            energy_per_min_when_empty=15.0,
            energy_per_min_when_loaded=20.0,
            cost_per_km=0.30,
            curb_weight_kg=3.0,
            parcel_weight_kg=2.0,
        )
        for i in range(1, 4)
    ]
    eligible = [c.node_id for c in customers if c.demand_kg <= 5.0]
    instance = InstanceData(
        name="real_case_delivery_logistics",
        seed=seed,
        region="real_delivery_logistics",
        customers=customers,
        trucks=trucks,
        drones=drones,
        priority_penalties={"high": 100.0, "medium": 20.0, "low": 1.0},
        constants=ProblemConstants(swap_time_s=60.0, reload_time_s=60.0, max_shift_minutes=480.0),
        metadata={
            "battery_slack_ratio": float(beta_slack_ratio),
            "drone_eligible_customers": eligible,
            "distance_scale": float(distance_scale),
            "source": str(csv_path),
        },
    )
    return instance, data


def _make_config(seed: int = 2026) -> SearchConfig:
    return SearchConfig(
        seed=seed,
        generation=GenerationSettings(num_customers=80, num_trucks=3, num_drones=3, region="dense_urban"),
        priority_penalties={"high": 100.0, "medium": 20.0, "low": 1.0},
        base_windows=PRIORITY_WINDOWS,
        vehicle_costs=VehicleCostConfig(truck_cost_per_km=0.8, drone_cost_per_km=0.3),
        milp=MILPConfig(enabled=False),
        heuristics=HeuristicConfig(max_outer_iter=25, max_no_improve=8, time_limit_seconds=180, random_seed=seed),
        constraints=ConstraintConfig(t_max_minutes=480.0),
        vehicles=VehicleCatalog(
            truck=VehicleSpec(speed_kmph=35.0, cost_per_km=0.8, capacity_kg=5000.0),
            drone=DroneSpec(speed_kmph=65.0, cost_per_km=0.3, capacity_kg=5.0, max_battery_wh=1200.0),
        ),
        vehicle_runtime=VehicleRuntimeConfig(swap_time_s=60.0, reload_time_s=60.0),
        experiment={"battery_slack_ratio": 0.10},
    )


def _truck_only_ortools(instance: InstanceData) -> SolutionData:
    sol = build_initial_solution(instance, instance.seed, initializer="ortools", ortools_time_limit_seconds=20)
    sol.objective, sol.components = evaluate_solution(instance, sol)
    sol.status = "baseline_truck_only_ortools"
    return sol


def _simple_drone_from_feasible_backbone(instance: InstanceData) -> SolutionData:
    current = _truck_only_ortools(instance)
    eligible = {
        int(cid)
        for cid in instance.metadata.get("drone_eligible_customers", [])
        if int(cid) in instance.customer_ids
    }
    coords = instance.coordinate_map()
    customers = [
        c
        for c in instance.customer_ids
        if c in eligible and any(current.u_truck.get((t, c), 0) == 1 for t in instance.truck_ids())
    ]
    customers.sort(key=lambda cid: (euclidean_distance(coords[0], coords[cid]), cid))
    for customer in customers:
        best_choice = None
        best_distance = float("inf")
        for drone in instance.drone_ids():
            dist = euclidean_distance(coords[0], coords[customer]) * 2.0
            if dist < best_distance:
                best_distance = dist
                best_choice = drone
        if best_choice is None:
            continue
        candidate = move_to_drone(
            instance,
            current,
            customer,
            best_choice,
            enforce_battery=True,
            battery_slack_ratio=float(instance.metadata.get("battery_slack_ratio", 0.10)),
        )
        if candidate is None:
            continue
        _sync_route_assignments(candidate, instance)
        candidate.objective, candidate.components = evaluate_solution(instance, candidate)
        if is_feasible(instance, candidate):
            current = candidate
    _sync_route_assignments(current, instance)
    current.objective, current.components = evaluate_solution(instance, current)
    current.status = "baseline_simple_drone_ortools"
    return current


def _run_methods(instance: InstanceData, config: SearchConfig) -> Dict[str, SolutionData]:
    methods: Dict[str, Callable[[], SolutionData]] = {
        "truck_only": lambda: _truck_only_ortools(instance),
        "simple_drone": lambda: _simple_drone_from_feasible_backbone(instance),
        "paired_baseline": lambda: paired_baseline(instance, config).solution,
        "no_unpairing": lambda: no_unpairing_baseline(instance, config).solution,
        "nils": lambda: run_nils(
            instance,
            seed=config.heuristics.random_seed,
            max_iter=config.heuristics.max_outer_iter,
            max_no_improve=config.heuristics.max_no_improve,
            time_limit=config.heuristics.time_limit_seconds,
            battery_slack_ratio=float(instance.metadata.get("battery_slack_ratio", 0.10)),
        ),
        "nils_no_local_search": lambda: nils_no_local_search_baseline(instance, config).solution,
        "nils_no_perturbation": lambda: nils_no_perturbation_baseline(instance, config).solution,
    }
    results: Dict[str, SolutionData] = {}
    for name, fn in methods.items():
        start = time.time()
        sol = fn()
        sol.run_time_seconds = float(sol.run_time_seconds or (time.time() - start))
        sol.objective, sol.components = evaluate_solution(instance, sol)
        results[name] = sol
    return results


def _priority_counts(instance: InstanceData, sol: SolutionData, priority: str) -> tuple[int, int]:
    cmap = instance.customer_map()
    served = [cid for cid in instance.customer_ids if cmap[cid].priority == priority]
    drone = [cid for cid in sol.served_by_drone() if cid in cmap and cmap[cid].priority == priority]
    return len(served), len(drone)


def _results_table(instance: InstanceData, solutions: Dict[str, SolutionData]) -> pd.DataFrame:
    truck_obj = float(solutions["truck_only"].objective)
    rows = []
    for method, sol in solutions.items():
        usage = summarize_drone_usage(sol)
        high_total, high_drone = _priority_counts(instance, sol, "high")
        med_total, med_drone = _priority_counts(instance, sol, "medium")
        low_total, low_drone = _priority_counts(instance, sol, "low")
        rows.append(
            {
                "method": method,
                "status": sol.status,
                "feasible": bool(is_feasible(instance, sol)),
                "objective": float(sol.objective),
                "runtime_seconds": float(sol.run_time_seconds),
                "truck_cost": float(sol.components.get("truck_cost", 0.0)),
                "drone_cost": float(sol.components.get("drone_cost", 0.0)),
                "waiting_sync_cost": float(sol.components.get("waiting_cost", 0.0)),
                "priority_tardiness_cost": float(sol.components.get("tardiness_cost", 0.0)),
                "drone_served_customers": usage["drone_served_customers"],
                "drone_service_share": usage["drone_served_customers"] / max(1, instance.num_customers),
                "drone_arcs": usage["drone_arcs"],
                "reload_events": usage["reload_events"],
                "battery_swaps": usage["battery_swaps"],
                "nonempty_drone_routes": usage["nonempty_drone_routes"],
                "served_high": high_total,
                "served_medium": med_total,
                "served_low": low_total,
                "drone_served_high": high_drone,
                "drone_served_medium": med_drone,
                "drone_served_low": low_drone,
                "improvement_vs_truck_only_pct": (
                    float("nan") if method == "truck_only" else 100.0 * (truck_obj - float(sol.objective)) / truck_obj
                ),
            }
        )
    order = {
        "truck_only": 1,
        "simple_drone": 2,
        "paired_baseline": 3,
        "no_unpairing": 4,
        "nils": 5,
        "nils_no_local_search": 6,
        "nils_no_perturbation": 7,
    }
    return pd.DataFrame(rows).sort_values("method", key=lambda s: s.map(order)).reset_index(drop=True)


def _plot_routes(instance: InstanceData, solutions: Dict[str, SolutionData], out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    coords = instance.coordinate_map()
    pairs = [("truck_only", "nils"), ("paired_baseline", "no_unpairing")]
    for left, right in pairs:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
        for ax, method in zip(axes, [left, right]):
            sol = solutions[method]
            ax.scatter([coords[i][0] for i in instance.customer_ids], [coords[i][1] for i in instance.customer_ids], s=14, c="#333333")
            ax.scatter([0], [0], s=60, c="#d62728", marker="s")
            for route in sol.truck_routes.values():
                if len(route) > 1:
                    ax.plot([coords[i][0] for i in route], [coords[i][1] for i in route], lw=1.3, alpha=0.75)
            for route in sol.drone_routes.values():
                if len([n for n in route if n != 0]) > 0:
                    ax.plot([coords[i][0] for i in route], [coords[i][1] for i in route], "--", lw=1.0, alpha=0.9)
            ax.set_title(method)
            ax.set_xlabel("x (scaled km)")
            ax.set_ylabel("y (scaled km)")
            ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        fig.savefig(fig_dir / f"real_case_routes_{left}_vs_{right}.png")
        plt.close(fig)


def run_real_case(
    csv_path: str | Path = r"C:\Users\L03128674\Downloads\Delivery_Logistics.csv",
    output_dir: str | Path = "outputs/real_case_delivery_logistics_final",
) -> pd.DataFrame:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    instance, sample = build_real_case_instance(csv_path)
    config = _make_config(seed=instance.seed)
    solutions = _run_methods(instance, config)
    table = _results_table(instance, solutions)
    feasible = table[table["feasible"]].copy()
    table.to_csv(out_dir / "real_case_results.csv", index=False)
    feasible.to_csv(out_dir / "real_case_feasible_results.csv", index=False)
    sample.to_csv(out_dir / "real_case_sample.csv", index=False)
    summary = {
        "source": str(csv_path),
        "sample_size": int(instance.num_customers),
        "sample_rule": "20 per delivery_mode",
        "delivery_mode_counts_sample": sample["delivery_mode_clean"].value_counts().to_dict(),
        "priority_counts": {p: sum(c.priority == p for c in instance.customers) for p in ("high", "medium", "low")},
        "light_package_share_le_5kg": float(sum(c.demand_kg <= 5.0 for c in instance.customers) / instance.num_customers),
        "truck_capacity_each_kg": 5000.0,
        "distance_scale": 0.1,
        "battery_slack_ratio": float(instance.metadata.get("battery_slack_ratio", 0.10)),
        "results_csv": str(out_dir / "real_case_results.csv"),
    }
    (out_dir / "real_case_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_routes(instance, solutions, out_dir)
    return table


if __name__ == "__main__":
    df = run_real_case()
    print(df.to_string(index=False))
