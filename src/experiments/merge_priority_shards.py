"""Merge disjoint matched-wall-clock M5/M9 study shards."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _merge_csv(shard_dirs: list[Path], filename: str, keys: list[str]) -> pd.DataFrame:
    frames = []
    for shard in shard_dirs:
        path = shard / filename
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No {filename} files found in the supplied shards")
    frame = pd.concat(frames, ignore_index=True)
    return frame.drop_duplicates(keys, keep="last").reset_index(drop=True)


def merge_priority_shards(*, output_dir: str | Path, shard_dirs: list[str | Path]) -> pd.DataFrame:
    shards = [Path(path) for path in shard_dirs]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_status = _merge_csv(shards, "run_status_partial.csv", ["instance_id", "method_name"])
    if len(run_status) != 1440:
        raise ValueError(f"Expected 1,440 M5/M9 rows, found {len(run_status)}")
    if set(run_status["method_name"].astype(str)) != {"nils", "nils_r"}:
        raise ValueError("Matched-wall-clock shards must contain only nils and nils_r")
    run_status.to_csv(output / "run_status_full.csv", index=False, float_format="%.10f")

    for filename, keys in [
        ("customer_delivery_observations_partial.csv", ["instance_id", "method_name", "customer_id"]),
        ("customer_service_detail.csv", ["instance_id", "method", "customer_id"]),
        ("drone_decision_log.csv", ["instance_id", "method", "iteration", "customer_id", "action"]),
        ("nils_r_reconstruction_log.csv", ["instance_id", "seed", "reconstruction_attempt"]),
    ]:
        frames = [shard / filename for shard in shards if (shard / filename).exists()]
        if frames:
            merged = _merge_csv(shards, filename, keys)
            out_name = filename.replace("_partial", "_full")
            merged.to_csv(output / out_name, index=False, float_format="%.10f")
    return run_status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-dirs", nargs="+", required=True)
    args = parser.parse_args()
    merge_priority_shards(output_dir=args.output_dir, shard_dirs=args.shard_dirs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
