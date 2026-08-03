from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import noto_correlated_validation_postprocess as validation
from ejor_dad.certification import (
    budget_intersecting_grid_cells,
    continuous_grid_certificate,
    validate_upper_corner_certificate_instance,
)
from ejor_dad.checkpoint import CheckpointStore, atomic_write_dataframe, atomic_write_text
from ejor_dad.fixed_y import evaluate_fixed_y


GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
RHOS = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25]


def evaluate(instance, y):
    try:
        result = evaluate_fixed_y(
            instance,
            y,
            epsilon=1e-5,
            max_iterations=300,
            enforce_retrofit_budget=False,
        )
    except Exception as error:
        return {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "y": np.asarray(y, dtype=float).tolist(),
        }
    return {
        "status": "feasible",
        "objective": float(result.objective),
        "y": result.y.tolist(),
        "z": result.z.tolist(),
        "w": result.w.tolist(),
    }


def is_valid_payload(payload):
    status = payload.get("status")
    if status in {"failed", "error", "infeasible", "pruned"}:
        return False
    objective = payload.get("objective")
    return objective is not None and math.isfinite(float(objective))


def is_infeasible_payload(payload):
    return payload.get("status") in {"infeasible", "pruned"}


def write_progress(root, *, status, rho, rho_index, completed, total, reused, pending, message=None):
    payload = {
        "status": status,
        "rho": float(rho),
        "rho_index": int(rho_index),
        "rho_count": len(RHOS),
        "completed": int(completed),
        "total": int(total),
        "pending": int(pending),
        "reused": int(reused),
        "pid": os.getpid(),
        "updated": time.time(),
    }
    if message:
        payload["message"] = message
    atomic_write_text(root / "certificate_status.json", json.dumps(payload, indent=2))


def main(out, workers):
    root = out / "correlated_facility_separated_capability_marginal_v2"
    design = json.loads((out / "run_design.json").read_text(encoding="utf-8"))
    args = validation.args_from(design, out)
    summary = pd.read_csv(root / "tables" / "table_noto_correlated_facility.csv")
    cache = CheckpointStore(root / "certificate_checkpoints")
    rows = []

    for rho_index, rho in enumerate(RHOS, start=1):
        instance = validation.build(rho, args)
        validate_upper_corner_certificate_instance(instance)
        cells = budget_intersecting_grid_cells(
            instance.retrofit_costs,
            instance.budget_retrofit,
            GRID,
        )
        grid_payloads = []
        for path in (root / "checkpoints").glob(f"*rho{rho:.2f}_*.json"):
            try:
                grid_payloads.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        lookup = {
            tuple(np.round(payload["y"], 10)): payload
            for payload in grid_payloads
            if "y" in payload
        }
        values = {}
        pending = []
        pruned = []
        reused = 0

        for cell in cells:
            key_tuple = tuple(np.round(cell.upper, 10))
            key = f"cert_rho{rho:.2f}_cell{cell.index:04d}"
            payload = lookup.get(key_tuple)
            if payload is None and cache.exists(key):
                try:
                    payload = cache.load(key)
                except (OSError, json.JSONDecodeError):
                    payload = None
            if payload is not None and is_valid_payload(payload):
                values[cell.index] = float(payload["objective"])
                reused += 1
            elif payload is not None and is_infeasible_payload(payload):
                pruned.append(cell.index)
                reused += 1
            else:
                pending.append((cell, key))

        write_progress(
            root,
            status="running",
            rho=rho,
            rho_index=rho_index,
            completed=len(values) + len(pruned),
            total=len(cells),
            reused=reused,
            pending=len(pending),
            message="Resumed certificate from valid checkpoints.",
        )

        failures = []
        if pending:
            with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = {
                    pool.submit(evaluate, instance, cell.upper): (cell, key)
                    for cell, key in pending
                }
                for count, future in enumerate(as_completed(futures), start=1):
                    cell, key = futures[future]
                    payload = future.result()
                    cache.save(key, payload)
                    if is_valid_payload(payload):
                        values[cell.index] = float(payload["objective"])
                    elif is_infeasible_payload(payload):
                        pruned.append(cell.index)
                    else:
                        failures.append({"cell": cell.index, **payload})
                    completed = len(values) + len(pruned) + len(failures)
                    write_progress(
                        root,
                        status="running",
                        rho=rho,
                        rho_index=rho_index,
                        completed=completed,
                        total=len(cells),
                        reused=reused,
                        pending=len(pending) - count,
                    )

        if failures:
            write_progress(
                root,
                status="failed",
                rho=rho,
                rho_index=rho_index,
                completed=len(values) + len(pruned),
                total=len(cells),
                reused=reused,
                pending=len(failures),
                message=f"{len(failures)} certificate cells failed; failed cells remain resumable.",
            )
            first = failures[0]
            raise RuntimeError(
                f"Certificate evaluation failed for cell {first['cell']}: "
                f"{first.get('error_type', 'Error')}: {first.get('error', '')}"
            )

        feasible_cells = tuple(cell for cell in cells if cell.index not in set(pruned))
        if len(feasible_cells) != len(values):
            missing = sorted(cell.index for cell in feasible_cells if cell.index not in values)
            raise RuntimeError(f"Certificate has missing feasible cell values: {missing[:10]}")
        upper_bound = float(summary.loc[np.isclose(summary.rho, rho), "objective"].iloc[0])
        certificate = continuous_grid_certificate(
            feasible_cells,
            [values[cell.index] for cell in feasible_cells],
            upper_bound,
        )
        rows.append(
            {
                "rho": rho,
                "continuous_lower_bound": certificate.continuous_lower_bound,
                "grid_upper_bound": certificate.grid_upper_bound,
                "absolute_gap": certificate.absolute_gap,
                "relative_gap_percent": certificate.relative_gap_percent,
                "cell_count": len(cells),
                "pruned_infeasible_cells": len(pruned),
                "evaluated_feasible_cells": len(feasible_cells),
                "reused_grid_upper_corners": reused,
                "new_or_cached_overbudget_corners": len(cells) - reused,
                "lower_bound_cell_lower_y_json": json.dumps(certificate.lower_bound_cell.lower.tolist()),
                "lower_bound_cell_upper_y_json": json.dumps(certificate.lower_bound_cell.upper.tolist()),
                "scope": "complete regime-labelled 128-state support; operational Stage 1; monotone upper-corner certificate",
            }
        )
        atomic_write_dataframe(
            pd.DataFrame(rows),
            root / "tables" / "table_noto_correlated_continuous_certificate.csv",
        )
        write_progress(
            root,
            status="running",
            rho=rho,
            rho_index=rho_index,
            completed=len(cells),
            total=len(cells),
            reused=reused,
            pending=0,
            message=f"Completed rho={rho:.3f}.",
        )

    atomic_write_text(
        root / "certificate_status.json",
        json.dumps(
            {
                "status": "completed",
                "rows": len(rows),
                "rho_values": RHOS,
                "pid": os.getpid(),
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
