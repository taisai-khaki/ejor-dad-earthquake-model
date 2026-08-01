from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noto_correlated_validation_postprocess as validation
import noto_mechanism_full_grid as mechanism
import noto_practical_resilience_experiment as practical
from ejor_dad.checkpoint import CheckpointStore, atomic_write_text
from ejor_dad.fixed_y import evaluate_fixed_y

VERSION = "noto-stage2-joint-full-grid-v2-supplied-spec"
RHOS = (0.0, 0.075, 0.10, 0.125, 0.25)
RELATIVE_TOLERANCES = (0.001, 0.005)
ABSOLUTE_TOLERANCE = 1e-5
BASE_SERVICE_FLOOR = 0.08
MAXIMUM_SERVICE_FLOOR = 1.0
SERVICE_RESOLUTION = 1e-4
_WORKER_INSTANCE = None


def init_worker(instance):
    global _WORKER_INSTANCE
    _WORKER_INSTANCE = instance


def evaluate_service_floor(y, service_floor):
    if _WORKER_INSTANCE is None:
        raise RuntimeError("Stage 2 worker was not initialized.")
    instance = replace(
        _WORKER_INSTANCE,
        minimum_zone_service_fraction=float(service_floor),
    )
    try:
        return evaluate_fixed_y(
            instance,
            np.asarray(y, dtype=float),
            epsilon=1e-5,
            max_iterations=300,
        )
    except RuntimeError as exc:
        if "infeasible" in str(exc).lower():
            return None
        raise


def solve_policy(y, objective_bound):
    started = time.perf_counter()
    base_result = evaluate_service_floor(y, BASE_SERVICE_FLOOR)
    if (
        base_result is None
        or float(base_result.objective) > objective_bound + 2e-5
    ):
        return {
            "status": "inadmissible",
            "y": np.asarray(y, dtype=float).tolist(),
            "runtime_seconds": time.perf_counter() - started,
        }

    low = BASE_SERVICE_FLOOR
    high = MAXIMUM_SERVICE_FLOOR
    best_result = base_result
    steps = 0
    while high - low > SERVICE_RESOLUTION:
        steps += 1
        midpoint = 0.5 * (low + high)
        result = evaluate_service_floor(y, midpoint)
        if (
            result is not None
            and float(result.objective) <= objective_bound + 2e-5
        ):
            low = midpoint
            best_result = result
        else:
            high = midpoint

    return {
        "status": "feasible",
        "y": np.asarray(y, dtype=float).tolist(),
        "service_floor_lower": low,
        "service_floor_upper": high,
        "terminal_service_resolution_gap": high - low,
        "objective": float(best_result.objective),
        "lower_bound": float(best_result.lower_bound),
        "oracle_gap": float(
            best_result.objective - best_result.lower_bound
        ),
        "z": best_result.z.tolist(),
        "w": best_result.w.tolist(),
        "bisection_steps": steps,
        "runtime_seconds": time.perf_counter() - started,
    }


def source_token(rho):
    return f"rho{rho:.2f}_"


def load_stage1_records(output_dir, rho):
    source = output_dir / "correlated_facility_separated_capability_marginal_v1" / "checkpoints"
    records = []
    for path in source.glob(f"*{source_token(rho)}*.json"):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    if len(records) != 996:
        raise RuntimeError(
            f"Expected 996 M4 Stage 1 records at rho={rho}, "
            f"found {len(records)}."
        )
    return records


def stage1_summary(records):
    feasible = [row for row in records if row.get("status") == "feasible"]
    if not feasible:
        raise RuntimeError("No feasible M4 Stage 1 policies.")
    feasible.sort(key=lambda row: (float(row["objective"]), tuple(row["y"])))
    return feasible[0], feasible


def checkpoint_key(rho, relative_tolerance, index):
    return (
        f"stage2_rho{rho:.3f}_eps{relative_tolerance:.3f}_"
        f"grid{index:04d}"
    )


def choose_stage2(records):
    feasible = [row for row in records if row.get("status") == "feasible"]
    if not feasible:
        raise RuntimeError("No Stage 2 feasible policies.")
    maximum_lower = max(float(row["service_floor_lower"]) for row in feasible)
    resolution = max(
        float(row["terminal_service_resolution_gap"]) for row in feasible
    )
    tied = [
        row
        for row in feasible
        if float(row["service_floor_upper"]) >= maximum_lower - 1e-12
    ]
    chosen = min(
        tied,
        key=lambda row: (
            float(row["objective"]),
            tuple(row["y"]),
        ),
    )
    return chosen, tied, resolution


def main(output_dir: Path, workers: int) -> None:
    design = json.loads((output_dir / "run_design.json").read_text())
    args = validation.args_from(design, output_dir)
    root = output_dir / "operational_stage2_joint_separated_capability_marginal_v1"
    (root / "tables").mkdir(parents=True, exist_ok=True)
    cache = CheckpointStore(root / "checkpoints")
    expected_steps = math.ceil(
        math.log2(
            (MAXIMUM_SERVICE_FLOOR - BASE_SERVICE_FLOOR)
            / SERVICE_RESOLUTION
        )
    )
    atomic_write_text(
        root / "run_manifest.json",
        json.dumps(
            {
                "experiment_version": VERSION,
                "rhos": RHOS,
                "relative_tolerances": RELATIVE_TOLERANCES,
                "absolute_tolerance": ABSOLUTE_TOLERANCE,
                "base_service_floor": BASE_SERVICE_FLOOR,
                "maximum_service_floor": MAXIMUM_SERVICE_FLOOR,
                "service_resolution": SERVICE_RESOLUTION,
                "expected_bisection_steps": expected_steps,
                "stage1_source": (
                    "correlated_facility_separated_capability_marginal_v1 full 996-policy M4 grid"
                ),
                "checkpointing": (
                    "one atomic JSON per radius-tolerance-admissible policy"
                ),
                "workers": workers,
                "started": time.time(),
            },
            indent=2,
        ),
    )
    rows = []

    for rho in RHOS:
        source_records = load_stage1_records(output_dir, rho)
        stage1_best, stage1_feasible = stage1_summary(source_records)
        benchmark = float(stage1_best["objective"])
        base, _ = practical.build_instance(rho, args)
        instance = mechanism.build(
            base,
            mechanism.specs(base)["M4"],
            "M4",
        )

        for relative_tolerance in RELATIVE_TOLERANCES:
            sacrifice_allowance = max(
                ABSOLUTE_TOLERANCE,
                relative_tolerance * benchmark,
            )
            objective_bound = benchmark + sacrifice_allowance
            admissible = [
                row
                for row in stage1_feasible
                if float(row["objective"]) <= objective_bound + 1e-8
            ]
            records = []
            pending = []
            for row in admissible:
                index = int(row["candidate_index"])
                key = checkpoint_key(rho, relative_tolerance, index)
                if cache.exists(key):
                    records.append(cache.load(key))
                else:
                    pending.append((index, row["y"], key))

            atomic_write_text(
                root / "status.json",
                json.dumps(
                    {
                        "status": "running",
                        "rho": rho,
                        "relative_tolerance": relative_tolerance,
                        "admissible_policy_count": len(admissible),
                        "completed_policy_bisections": len(records),
                        "pending_policy_bisections": len(pending),
                        "updated": time.time(),
                    },
                    indent=2,
                ),
            )

            if pending:
                with ProcessPoolExecutor(
                    max_workers=min(workers, len(pending)),
                    initializer=init_worker,
                    initargs=(instance,),
                ) as pool:
                    futures = {
                        pool.submit(
                            solve_policy,
                            y,
                            objective_bound,
                        ): (index, key)
                        for index, y, key in pending
                    }
                    for completed, future in enumerate(
                        as_completed(futures),
                        1,
                    ):
                        index, key = futures[future]
                        payload = future.result()
                        payload["candidate_index"] = index
                        payload["rho"] = rho
                        payload["relative_tolerance"] = relative_tolerance
                        payload["objective_bound"] = objective_bound
                        payload["experiment_version"] = VERSION
                        cache.save(key, payload)
                        records.append(payload)
                        atomic_write_text(
                            root / "status.json",
                            json.dumps(
                                {
                                    "status": "running",
                                    "rho": rho,
                                    "relative_tolerance": (
                                        relative_tolerance
                                    ),
                                    "admissible_policy_count": len(admissible),
                                    "completed_policy_bisections": (
                                        len(records)
                                    ),
                                    "pending_policy_bisections": len(pending),
                                    "updated": time.time(),
                                },
                                indent=2,
                            ),
                        )

            chosen, service_ties, resolution = choose_stage2(records)
            objective = float(chosen["objective"])
            realized_sacrifice = objective - benchmark
            rows.append(
                {
                    "experiment_version": VERSION,
                    "rho": rho,
                    "relative_tolerance": relative_tolerance,
                    "absolute_tolerance": ABSOLUTE_TOLERANCE,
                    "stage1_robust_benchmark": benchmark,
                    "stage1_selected_y_json": json.dumps(
                        stage1_best["y"]
                    ),
                    "sacrifice_allowance": sacrifice_allowance,
                    "objective_bound": objective_bound,
                    "stage2_service_floor": float(
                        chosen["service_floor_lower"]
                    ),
                    "stage2_service_upper_bound": float(
                        chosen["service_floor_upper"]
                    ),
                    "stage2_objective": objective,
                    "realized_sacrifice": realized_sacrifice,
                    "realized_sacrifice_percent": (
                        100.0 * realized_sacrifice / benchmark
                    ),
                    "allowed_tolerance_used_percent": (
                        100.0
                        * max(0.0, realized_sacrifice)
                        / sacrifice_allowance
                    ),
                    "selected_y_json": json.dumps(chosen["y"]),
                    "selected_z_json": json.dumps(chosen["z"]),
                    "selected_w_json": json.dumps(chosen["w"]),
                    "stage1_admissible_policy_count": len(admissible),
                    "service_tie_count_within_resolution": len(service_ties),
                    "bisection_steps": int(chosen["bisection_steps"]),
                    "terminal_service_resolution_gap": resolution,
                    "oracle_gap": float(chosen["oracle_gap"]),
                    "solution_scope": (
                        "grid-global up to reported bisection resolution "
                        "over Stage-1-admissible M4 policies"
                    ),
                }
            )
            pd.DataFrame(rows).sort_values(
                ["rho", "relative_tolerance"]
            ).to_csv(
                root
                / "tables"
                / "table_noto_stage2_joint_maxmin_service.csv",
                index=False,
            )

    atomic_write_text(
        root / "status.json",
        json.dumps(
            {
                "status": "completed",
                "rows": len(rows),
                "updated": time.time(),
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    main(Path(arguments.output_dir), arguments.workers)
