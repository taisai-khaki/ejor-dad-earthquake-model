"""Ablation-style experiment bundle."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ..heuristics import run_nils
from ..heuristics.baselines import run_all_baselines
from ..instance_generator import InstanceGenerator
from ..parameters import SearchConfig
from ..heuristics.nils import summarize_drone_usage


def run_ablation(config: SearchConfig, output_dir: str | None = None) -> pd.DataFrame:
    target_dir = output_dir or str(config.experiment.get("output_dir", "outputs"))
    generator = InstanceGenerator.from_search_config(config)
    sizes = [int(v) for v in (config.generation.sizes or [config.generation.num_customers, 20])]
    reps = max(1, int(config.experiment.get("ablation_reps", 4)))
    instances = generator.generate_batch(config.seed + 777, sizes, reps, target_dir, tag="ablation")

    rows = []
    for inst in instances:
        baseline = run_nils(
            inst,
            seed=config.seed,
            max_iter=config.heuristics.max_outer_iter,
            max_no_improve=config.heuristics.max_no_improve,
            time_limit=config.heuristics.time_limit_seconds,
        )
        baselines = run_all_baselines(inst, config)
        baseline_map: Dict[str, Any] = {item.name: item for item in baselines}

        def _obj(entry) -> float:
            if hasattr(entry, "solution"):
                return float(entry.solution.objective)
            return float(entry.objective)

        def _safe_obj(name: str) -> float:
            return float(_obj(baseline_map[name]))

        def _usage(name: str) -> Dict[str, int]:
            return summarize_drone_usage(baseline_map[name].solution)

        nils_usage = summarize_drone_usage(baseline)
        row: dict[str, Any] = {
            "instance": inst.name,
            "nils": _obj(baseline),
            "truck_only": _safe_obj("truck_only"),
            "paired_drone": _safe_obj("paired_baseline"),
            "no_priority": _safe_obj("no_priority"),
            "no_unpair": _safe_obj("no_unpairing"),
            "nils_drone_served_customers": nils_usage["drone_served_customers"],
            "nils_drone_arcs": nils_usage["drone_arcs"],
            "nils_reload_events": nils_usage["reload_events"],
            "nils_battery_swaps": nils_usage["battery_swaps"],
            "nils_nonempty_drone_routes": nils_usage["nonempty_drone_routes"],
            "truck_only_drone_served_customers": _usage("truck_only")["drone_served_customers"],
            "truck_only_drone_arcs": _usage("truck_only")["drone_arcs"],
            "truck_only_reload_events": _usage("truck_only")["reload_events"],
            "truck_only_battery_swaps": _usage("truck_only")["battery_swaps"],
            "truck_only_nonempty_drone_routes": _usage("truck_only")["nonempty_drone_routes"],
            "paired_drone_drone_served_customers": _usage("paired_baseline")["drone_served_customers"],
            "paired_drone_drone_arcs": _usage("paired_baseline")["drone_arcs"],
            "paired_drone_reload_events": _usage("paired_baseline")["reload_events"],
            "paired_drone_battery_swaps": _usage("paired_baseline")["battery_swaps"],
            "paired_drone_nonempty_drone_routes": _usage("paired_baseline")["nonempty_drone_routes"],
            "no_priority_drone_served_customers": _usage("no_priority")["drone_served_customers"],
            "no_priority_drone_arcs": _usage("no_priority")["drone_arcs"],
            "no_priority_reload_events": _usage("no_priority")["reload_events"],
            "no_priority_battery_swaps": _usage("no_priority")["battery_swaps"],
            "no_priority_nonempty_drone_routes": _usage("no_priority")["nonempty_drone_routes"],
            "no_unpair_drone_served_customers": _usage("no_unpairing")["drone_served_customers"],
            "no_unpair_drone_arcs": _usage("no_unpairing")["drone_arcs"],
            "no_unpair_reload_events": _usage("no_unpairing")["reload_events"],
            "no_unpair_battery_swaps": _usage("no_unpairing")["battery_swaps"],
            "no_unpair_nonempty_drone_routes": _usage("no_unpairing")["nonempty_drone_routes"],
            "run_time_nils": float(baseline.run_time_seconds),
            "num_customers": inst.num_customers,
            "num_trucks": inst.num_trucks,
            "num_drones": inst.num_drones,
        }
        row["gap_truck_only"] = row["nils"] - row["truck_only"]
        row["gap_paired_drone"] = row["nils"] - row["paired_drone"]
        row["gap_no_priority"] = row["nils"] - row["no_priority"]
        row["gap_no_unpair"] = row["nils"] - row["no_unpair"]
        rows.append(row)

    out = pd.DataFrame(rows)
    out_dir = Path(target_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "ablation_summary.csv", index=False)
    return out
