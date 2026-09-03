from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from critical_revision_common import atomic_json, critical_output_root, write_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-output-dir", default=None)
    parser.add_argument("--output-root", default=str(critical_output_root()))
    parser.add_argument("--start-block", choices=("baseline", "audits", "development", "dependence", "frontier", "all-radii", "synthetic", "equity", "ambiguity", "integrated"), default="baseline")
    parser.add_argument("--stop-after", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    repo = Path(__file__).resolve().parents[1]
    base = Path(args.base_output_dir).resolve() if args.base_output_dir else repo / "data_work" / "noto" / "acute_access_graded_v4"
    root.mkdir(parents=True, exist_ok=True)
    blocks = ["baseline", "audits", "development", "dependence", "frontier", "all-radii", "synthetic", "equity", "ambiguity", "integrated"]
    start = blocks.index(args.start_block)
    if args.stop_after and args.stop_after not in blocks:
        raise ValueError(f"Unknown stop block {args.stop_after}")
    commands = {
        "baseline": [sys.executable, "examples/noto_baseline_regression.py", "--base-output-dir", str(base), "--output-dir", str(root / "baseline")],
        "audits": [[sys.executable, "examples/noto_path_incidence_audit.py", "--base-output-dir", str(base), "--output-dir", str(root / "reproducibility")], [sys.executable, "examples/noto_renovation_cost_audit.py", "--base-output-dir", str(base), "--output-dir", str(root / "reproducibility_cost")]],
        "development": [sys.executable, "examples/noto_continuous_monotone_bb.py", "--base-output-dir", str(base), "--output-dir", str(root / "continuous_bb"), "--rhos", "0,0.10,0.125,0.25", "--initial-partition", "grid_seeded", "--relative-gap", "0.001", "--oracle-epsilon", "1e-6", "--workers", "4", "--resume"],
        "dependence": [sys.executable, "examples/noto_mechanism_cross_evaluation.py", "--base-output-dir", str(base), "--output-dir", str(root / "mechanism_value"), "--rhos", "0,0.10,0.125,0.25", "--resume"],
        "frontier": [sys.executable, "examples/noto_budget_frontier_full_cube.py", "--base-output-dir", str(base), "--output-dir", str(root / "budget_frontier"), "--rhos", "0,0.025,0.05,0.075,0.10,0.125,0.15,0.20,0.25", "--budget-start", "0.20", "--budget-stop", "0.65", "--budget-step", "0.025", "--include-budget", "0.42", "--resume"],
        "all-radii": [sys.executable, "examples/noto_continuous_monotone_bb.py", "--base-output-dir", str(base), "--output-dir", str(root / "continuous_bb"), "--rhos", "0,0.025,0.05,0.075,0.10,0.125,0.15,0.20,0.25", "--initial-partition", "grid_seeded", "--relative-gap", "0.001", "--oracle-epsilon", "1e-6", "--workers", "4", "--resume"],
        "synthetic": [sys.executable, "examples/synthetic_monotone_bb_benchmark.py", "--output-dir", str(root / "synthetic_scaling"), "--dimensions", "5,6,7,8", "--relative-gap", "0.001", "--resume"],
        "equity": [sys.executable, "examples/noto_stage2_robustness_equity.py", "--base-output-dir", str(base), "--output-dir", str(root / "equity"), "--run-base-frontier", "--run-one-factor-sensitivities", "--service-resolution", "1e-4", "--resume"],
        "ambiguity": [sys.executable, "examples/noto_ambiguity_ensemble_calibration.py", "--weights", str(repo / "inputs" / "regime_weight_ensemble.csv"), "--base-output-dir", str(base), "--output-dir", str(root / "ambiguity_anchor"), "--resume"],
        "integrated": [sys.executable, "examples/build_critical_revision_audit.py", "--input-root", str(root), "--output-root", str(repo / "results" / "noto" / "critical_revision_v2")],
    }
    if args.skip_synthetic:
        blocks.remove("synthetic")
    for index in range(start, len(blocks)):
        block = blocks[index]
        command_group = commands[block]
        group = command_group if block == "audits" else [command_group]
        for command in group:
            command = [str(item) for item in command]
            log_path = root / "pipeline_logs" / f"{block}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            status_path = root / "pipeline_status.json"
            atomic_json(status_path, {"status": "running", "current_block": block, "block_index": index + 1, "block_count": len(blocks), "command": command, "log_path": str(log_path), "pid": os.getpid(), "updated_at_epoch": time.time()})
            if args.dry_run:
                print(" ".join(command))
                continue
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n=== START {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                process = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=False)
                log.write(f"=== END exit={process.returncode} ===\n")
            if process.returncode != 0:
                atomic_json(status_path, {"status": "failed", "current_block": block, "returncode": process.returncode, "log_path": str(log_path), "pid": os.getpid(), "updated_at_epoch": time.time()})
                raise SystemExit(process.returncode)
        atomic_json(root / "pipeline_status.json", {"status": "completed_block", "completed_block": block, "block_index": index + 1, "block_count": len(blocks), "pid": os.getpid(), "updated_at_epoch": time.time()})
        if args.stop_after and block == args.stop_after:
            break
    atomic_json(root / "pipeline_status.json", {"status": "completed", "pid": os.getpid(), "updated_at_epoch": time.time(), "output_root": str(root)})


if __name__ == "__main__":
    main()


