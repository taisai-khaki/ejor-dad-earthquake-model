"""Merge disjoint corrected-ALNS shards into one reproducible full-design report."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..parameters import load_and_build_config
from .run_alns_comparison import _with_alns_settings
from .run_full_alns_comparison import (
    ALNS_METHOD,
    _matched_combined_dataframe,
    _write_outputs,
)


def merge_shards(
    *,
    config_path: str | Path,
    source_output_dir: str | Path,
    output_dir: str | Path,
    shard_dirs: list[str | Path],
) -> pd.DataFrame:
    source_dir = Path(source_output_dir)
    output_path = Path(output_dir)
    partial_paths = [Path(shard) / "tables" / "alns_full_alns_runs_partial.csv" for shard in shard_dirs]
    frames = []
    missing = []
    for path in partial_paths:
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing ALNS shard files: " + "; ".join(missing))
    alns = pd.concat(frames, ignore_index=True)
    alns = alns[alns["method_name"].astype(str).eq(ALNS_METHOD)].copy()
    alns = alns.drop_duplicates(["instance_id", "method_name"], keep="last")
    if len(alns) != 720:
        raise ValueError(f"Expected 720 ALNS rows after merging shards, found {len(alns)}")
    if alns["instance_id"].nunique() != 720:
        raise ValueError("Merged ALNS rows do not contain 720 unique instances")
    if not (alns["iterations"].astype(float) == 40.0).all():
        raise ValueError("Corrected ALNS merge contains rows that did not complete 40 iterations")
    if not (alns["candidate_reassignments_evaluated"].astype(float) == 40.0).all():
        raise ValueError("Corrected ALNS merge contains rows without the prescribed destroy-and-repair evaluations")

    source = pd.read_csv(source_dir / "tables" / "heuristic_study_runs.csv")
    config = _with_alns_settings(load_and_build_config(str(config_path)))
    combined = _matched_combined_dataframe(source, alns, config)
    if combined["instance_id"].nunique() != 720:
        raise ValueError("Merged matched report does not contain 720 matched instances")
    tables_dir = output_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    alns.to_csv(tables_dir / "alns_full_alns_runs_partial.csv", index=False)
    return _write_outputs(combined, tables_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-dirs", nargs="+", required=True)
    args = parser.parse_args()
    merge_shards(
        config_path=args.config,
        source_output_dir=args.source_output_dir,
        output_dir=args.output_dir,
        shard_dirs=args.shard_dirs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
