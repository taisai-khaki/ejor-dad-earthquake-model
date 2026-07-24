"""Large instance runner for heuristic-only experiments."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from ..data_models import ExperimentResult
from ..heuristics import run_nils
from ..instance_generator import InstanceGenerator
from ..parameters import SearchConfig
from ..heuristics.nils import summarize_drone_usage


def _to_sizes(config: SearchConfig) -> List[int]:
    explicit = config.experiment.get("run_large_sizes")
    if explicit:
        return [int(v) for v in explicit]
    if config.generation.sizes:
        return [int(v) for v in config.generation.sizes]
    return [100, 150, 200]


def run_large_heuristic(config: SearchConfig, output_dir: str | None = None) -> List[ExperimentResult]:
    exp_cfg = dict(config.experiment or {})
    sizes = _to_sizes(config)
    reps = int(exp_cfg.get("instance_reps_per_size", config.generation.instance_reps_per_size))

    generator = InstanceGenerator.from_search_config(config)
    target_dir = output_dir or str(exp_cfg.get("output_dir", "outputs"))
    instances = generator.generate_batch(config.seed + 1200, sizes, reps, target_dir, tag="large")

    out_rows = []
    results: List[ExperimentResult] = []
    for inst in instances:
        sol = run_nils(
            inst,
            seed=config.heuristics.random_seed,
            max_iter=config.heuristics.max_outer_iter,
            max_no_improve=config.heuristics.max_no_improve,
            time_limit=config.heuristics.time_limit_seconds,
        )
        usage = summarize_drone_usage(sol)
        results.append(
            ExperimentResult(
                instance_name=inst.name,
                method="nils",
                status=sol.status,
                runtime_seconds=sol.run_time_seconds,
                objective=sol.objective,
            )
        )
        out_rows.append(
            {
                "instance": inst.name,
                "objective": sol.objective,
                "status": sol.status,
                "runtime": sol.run_time_seconds,
                "num_customers": inst.num_customers,
                "num_trucks": inst.num_trucks,
                "num_drones": inst.num_drones,
                "size": inst.num_customers,
                "drone_served_customers": usage["drone_served_customers"],
                "drone_arcs": usage["drone_arcs"],
                "reload_events": usage["reload_events"],
                "battery_swaps": usage["battery_swaps"],
                "nonempty_drone_routes": usage["nonempty_drone_routes"],
            }
        )

    out_dir = Path(target_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_dir / "large_heuristic_summary.csv", index=False)
    return results
