"""Sensitivity analysis experiments."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

from ..heuristics import run_nils
from ..heuristics.baselines import truck_only_baseline
from ..heuristics.nils import summarize_drone_usage
from ..instance_generator import InstanceGenerator
from ..parameters import SearchConfig


def _seed_for(level_index: int, factor_index: int, base_seed: int) -> int:
    return int(base_seed + 10_000 + factor_index * 10_000 + level_index * 1000)


def _study_sizes(config: SearchConfig) -> Tuple[int, ...]:
    explicit = config.experiment.get("sensitivity_sizes")
    if explicit:
        return tuple(int(v) for v in explicit)
    if config.generation.sizes:
        return tuple(config.generation.sizes)
    return (20, 30)


def _study_reps(config: SearchConfig) -> int:
    value = config.experiment.get("sensitivity_reps_per_size")
    if value is not None:
        return max(1, int(value))
    return max(1, int(config.generation.instance_reps_per_size))


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Invalid sensitivity value: {value}")


def _apply_factor(
    config: SearchConfig,
    factor: str,
    value: object,
) -> SearchConfig:
    gen = config.generation
    runtime = config.vehicle_runtime
    vehicles = config.vehicles
    windows = dict(config.base_windows)

    if factor in {"drones", "num_drones"}:
        gen = replace(gen, num_drones=int(_as_float(value)))
    elif factor in {"trucks", "num_trucks"}:
        gen = replace(gen, num_trucks=int(_as_float(value)))
    elif factor in {"customers", "num_customers"}:
        gen = replace(gen, num_customers=int(_as_float(value)))
    elif factor == "swap_time_s":
        runtime = replace(runtime, swap_time_s=_as_float(value))
    elif factor == "reload_time_s":
        runtime = replace(runtime, reload_time_s=_as_float(value))
    elif factor in {"drone_battery", "max_battery_wh"}:
        drone = replace(vehicles.drone, max_battery_wh=_as_float(value))
        vehicles = replace(vehicles, drone=drone)
    elif factor == "drone_speed":
        drone = replace(vehicles.drone, speed_kmph=_as_float(value))
        vehicles = replace(vehicles, drone=drone)
    elif factor == "truck_speed":
        truck = replace(vehicles.truck, speed_kmph=_as_float(value))
        vehicles = replace(vehicles, truck=truck)
    elif factor in {"priority_share", "priority_share_high"}:
        if not isinstance(value, (float, int)):
            raise ValueError(f"priority_share value must be numeric, got {value}")
        high_share = float(value)
        if high_share < 0 or high_share > 1:
            raise ValueError("priority_share must be in [0, 1]")
        if factor == "priority_share":
            # keep existing proportions from base config and stretch low share for remaining
            remaining = max(0.0, 1.0 - high_share)
            medium_share = min(remaining, gen.priority_share[1])
            low_share = remaining - medium_share
            gen = replace(gen, priority_share=(high_share, medium_share, low_share))
        else:
            gen = replace(gen, priority_share=(high_share, (1.0 - high_share) / 2, (1.0 - high_share) / 2))
    elif factor in {"high_window_ub", "class1_ub"}:
        lo, _ = windows["high"]
        windows["high"] = (lo, _as_float(value))
    elif factor == "medium_window_ub":
        lo, _ = windows["medium"]
        windows["medium"] = (lo, _as_float(value))
    elif factor == "low_window_ub":
        lo, _ = windows["low"]
        windows["low"] = (lo, _as_float(value))
    elif factor == "region":
        gen = replace(gen, region=str(value))
    else:
        return config

    return SearchConfig(
        seed=config.seed,
        generation=gen,
        priority_penalties=dict(config.priority_penalties),
        base_windows=windows,
        vehicle_costs=config.vehicle_costs,
        milp=config.milp,
        heuristics=config.heuristics,
        constraints=config.constraints,
        vehicles=vehicles,
        vehicle_runtime=runtime,
        experiment=dict(config.experiment),
        sensitivity=dict(config.sensitivity),
    )


def _run_factor_level(
    *,
    factor: str,
    value: object,
    scenario_seed: int,
    base_config: SearchConfig,
    study_config: SearchConfig,
    sizes: Tuple[int, ...],
    reps: int,
    output_dir: str,
    factor_index: int,
    value_index: int,
) -> List[dict]:
    generator = InstanceGenerator.from_search_config(study_config)
    level_seed = scenario_seed + value_index * 17 + factor_index
    instances = generator.generate_batch(
        seed=level_seed,
        sizes=list(sizes),
        reps_per_size=reps,
        output_dir=output_dir,
        tag=f"sens_{factor}_{value_index}",
    )

    rows: List[dict] = []
    for idx, instance in enumerate(instances):
        seed = base_config.heuristics.random_seed + idx
        try:
            solved = run_nils(
                instance,
                seed=seed,
                max_iter=max(1, int(base_config.heuristics.max_outer_iter)),
                max_no_improve=max(1, int(base_config.heuristics.max_no_improve)),
                time_limit=max(1, int(base_config.heuristics.time_limit_seconds)),
            )
            status = solved.status
            objective = float(solved.objective)
            truck_only_objective = float(truck_only_baseline(instance, base_config).solution.objective)
            truck_cost = float(solved.components.get("truck_cost", 0.0))
            drone_cost = float(solved.components.get("drone_cost", 0.0))
            tardiness_cost = float(solved.components.get("tardiness_cost", 0.0))
            run_time = float(solved.run_time_seconds)
            drone_usage = summarize_drone_usage(solved)
        except Exception:
            status = "failed"
            objective = float("nan")
            truck_only_objective = float("nan")
            truck_cost = float("nan")
            drone_cost = float("nan")
            tardiness_cost = float("nan")
            run_time = 0.0
            drone_usage = {
                "drone_served_customers": float("nan"),
                "drone_arcs": float("nan"),
                "reload_events": float("nan"),
                "battery_swaps": float("nan"),
                "nonempty_drone_routes": float("nan"),
            }

        rows.append(
            {
                "factor": factor,
                "value": value,
                "instance": instance.name,
                "num_customers": instance.num_customers,
                "num_trucks": instance.num_trucks,
                "num_drones": instance.num_drones,
                "seed": instance.seed,
                "objective": objective,
                "truck_only_objective": truck_only_objective,
                "truck_cost": truck_cost,
                "drone_cost": drone_cost,
                "tardiness_cost": tardiness_cost,
                "status": status,
                "run_time_seconds": run_time,
                "vehicle_cost_swap_s": float(study_config.vehicle_runtime.swap_time_s),
                "vehicle_cost_reload_s": float(study_config.vehicle_runtime.reload_time_s),
                "drone_max_battery_wh": float(study_config.vehicles.drone.max_battery_wh),
                "replicate": idx + 1,
                "drone_served_customers": drone_usage["drone_served_customers"],
                "drone_arcs": drone_usage["drone_arcs"],
                "reload_events": drone_usage["reload_events"],
                "battery_swaps": drone_usage["battery_swaps"],
                "nonempty_drone_routes": drone_usage["nonempty_drone_routes"],
            }
        )
    return rows


def run_sensitivity_study(config: SearchConfig, output_dir: str = "outputs") -> pd.DataFrame:
    """Run a reproducible one-factor-at-a-time sensitivity sweep."""
    if not config.sensitivity:
        output_dir = Path(output_dir) / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame(columns=["factor", "value", "instance", "instance_seed"])
        empty.to_csv(output_dir / "sensitivity_summary.csv", index=False)
        return empty

    sizes = _study_sizes(config)
    reps = _study_reps(config)
    rows: List[dict] = []

    for factor_index, (factor, values) in enumerate(config.sensitivity.items()):
        if not isinstance(values, list) or not values:
            continue
        for value_index, raw_value in enumerate(values):
            study_config = _apply_factor(config, factor, raw_value)
            if study_config is config:
                continue
            scenario_seed = _seed_for(value_index, factor_index, int(config.seed))
            rows.extend(
                _run_factor_level(
                    factor=factor,
                    value=raw_value,
                    scenario_seed=scenario_seed,
                    base_config=config,
                    study_config=study_config,
                    sizes=sizes,
                    reps=reps,
                    output_dir=output_dir,
                    factor_index=factor_index,
                    value_index=value_index,
                )
            )

    out = pd.DataFrame(rows)
    out_dir = Path(output_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not out.empty:
        out["gap_to_truck_only"] = out["objective"] - out["truck_only_objective"]
        out.to_csv(out_dir / "sensitivity_summary.csv", index=False)
        out.groupby(["factor", "value"])[["objective", "gap_to_truck_only"]].agg(["mean", "std", "count"]).reset_index().to_csv(
            out_dir / "sensitivity_by_factor.csv",
            index=False,
        )
    return out
