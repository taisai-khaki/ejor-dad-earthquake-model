"""Baseline comparison script used in Section 5.2 style ablations."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from ..heuristics.baselines import run_all_baselines
from ..instance_generator import InstanceGenerator
from ..parameters import SearchConfig
from ..heuristics.nils import summarize_drone_usage


def _to_sizes(config: SearchConfig) -> List[int]:
    explicit = config.experiment.get("run_small_sizes")
    if explicit:
        return [int(v) for v in explicit]
    if config.generation.sizes:
        return [int(v) for v in config.generation.sizes]
    return [20, 50]


def run_baseline_comparison(config: SearchConfig, output_dir: str | None = None) -> pd.DataFrame:
    exp_cfg = dict(config.experiment or {})
    sizes = _to_sizes(config)
    reps = int(exp_cfg.get("instance_reps_per_size", config.generation.instance_reps_per_size))

    generator = InstanceGenerator.from_search_config(config)
    target_dir = output_dir or str(exp_cfg.get("output_dir", "outputs"))
    instances = generator.generate_batch(config.seed + 99, sizes, reps, target_dir, tag="baseline")

    rows = []
    for inst in instances:
        for baseline in run_all_baselines(inst, config):
            # Normalize runtime for fair method comparison
            runtime = baseline.solution.run_time_seconds if baseline.solution.run_time_seconds is not None else 0.0
            row = {
                "instance": inst.name,
                "num_customers": inst.num_customers,
                "method": baseline.name,
                "objective": baseline.solution.objective,
                "runtime": runtime,
                "status": baseline.solution.status,
            }
            usage = summarize_drone_usage(baseline.solution)
            for key, value in usage.items():
                row[key] = value
            rows.append(row)

    out = pd.DataFrame(rows)
    out_dir = Path(target_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "baseline_comparison.csv", index=False)
    return out
