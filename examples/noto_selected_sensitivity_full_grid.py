from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noto_correlated_claim_sensitivity as claim
import noto_correlated_validation_postprocess as validation
import noto_mechanism_full_grid as mechanism
from ejor_dad.checkpoint import CheckpointStore, atomic_write_text

VERSION = "noto-selected-sensitivity-separated-capability-marginal-v2"
POLICY_B = np.array([0.0, 1.0, 1.0, 0.0, 0.25])
SETTINGS = (
    ("design_S1", {"design": "S1"}, (0.25,)),
    ("facility_mild", {"facility": "mild"}, (0.0, 0.25)),
    ("intensity_0.75", {"intensity": 0.75}, (0.25,)),
    ("residual_0.20", {"residual": 0.20}, (0.25,)),
    ("service_0.00", {"service": 0.0}, (0.25,)),
    ("weights_0.75", {"weights": 0.75}, (0.25,)),
)


def checkpoint_key(setting: str, rho: float, index: int) -> str:
    return f"sensitivity_{setting}_rho{rho:.2f}_grid{index:04d}"


def summarize(setting, parameters, rho, instance, records):
    feasible = [row for row in records if row.get("status") == "feasible"]
    if not feasible:
        raise RuntimeError(f"No feasible policies for {setting}, rho={rho:.2f}.")
    feasible.sort(key=lambda row: (float(row["objective"]), tuple(row["y"])))
    best = feasible[0]
    second = feasible[1]
    best_objective = float(best["objective"])
    selected = np.asarray(best["y"], dtype=float)
    summary = {
        "experiment_version": VERSION,
        "setting": setting,
        "parameters_json": json.dumps(parameters, sort_keys=True),
        "rho": rho,
        "selected_y_json": json.dumps(best["y"]),
        "objective": best_objective,
        "selected_z_json": json.dumps(best["z"]),
        "selected_w_json": json.dumps(best["w"]),
        "second_y_json": json.dumps(second["y"]),
        "second_objective": float(second["objective"]),
        "absolute_margin": float(second["objective"]) - best_objective,
        "margin_percent": 100.0
        * (float(second["objective"]) / best_objective - 1.0),
        "within_0p01_percent": sum(
            float(row["objective"]) <= best_objective * 1.0001
            for row in feasible
        ),
        "within_0p10_percent": sum(
            float(row["objective"]) <= best_objective * 1.001
            for row in feasible
        ),
        "within_0p50_percent": sum(
            float(row["objective"]) <= best_objective * 1.005
            for row in feasible
        ),
        "policy_B_globally_optimal": bool(
            np.allclose(selected, POLICY_B, atol=1e-10, rtol=0.0)
        ),
        "operationally_feasible_count": len(feasible),
        "unacceptable_count": 996 - len(feasible),
        "state_count": len(instance.states),
        "design_basis_count": len(instance.critical_service_state_ids),
    }
    top = [
        {
            "experiment_version": VERSION,
            "setting": setting,
            "parameters_json": json.dumps(parameters, sort_keys=True),
            "rho": rho,
            "rank": rank,
            "objective": float(row["objective"]),
            "gap": float(row["objective"]) - best_objective,
            "gap_percent": 100.0
            * (float(row["objective"]) / best_objective - 1.0),
            "y_json": json.dumps(row["y"]),
            "z_json": json.dumps(row["z"]),
            "w_json": json.dumps(row["w"]),
        }
        for rank, row in enumerate(feasible[:5], 1)
    ]
    return summary, top


def write_tables(root, summaries, top_rows):
    pd.DataFrame(summaries).sort_values(["setting", "rho"]).to_csv(
        root / "tables" / "table_noto_selected_sensitivity_full_grid.csv",
        index=False,
    )
    pd.DataFrame(top_rows).sort_values(["setting", "rho", "rank"]).to_csv(
        root / "tables" / "table_noto_selected_sensitivity_full_grid_top5.csv",
        index=False,
    )


def main(output_dir: Path, workers: int) -> None:
    design = json.loads((output_dir / "run_design.json").read_text())
    args = validation.args_from(design, output_dir)
    root = output_dir / "selected_sensitivity_separated_capability_marginal_v2"
    (root / "tables").mkdir(parents=True, exist_ok=True)
    cache = CheckpointStore(root / "checkpoints")
    atomic_write_text(
        root / "run_manifest.json",
        json.dumps(
            {
                "experiment_version": VERSION,
                "settings": [
                    {
                        "setting": setting,
                        "parameters": parameters,
                        "rhos": rhos,
                    }
                    for setting, parameters, rhos in SETTINGS
                ],
                "grid": mechanism.GRID,
                "expected_grid_count": 996,
                "checkpointing": "one atomic JSON per setting-radius-policy",
                "ascending_radius_warm_start": True,
                "workers": workers,
                "started": time.time(),
            },
            indent=2,
        ),
    )
    summaries = []
    top_rows = []

    for setting, parameters, rhos in SETTINGS:
        instance = claim.construct(min(rhos), args, parameters)
        candidates = mechanism.candidate_grid(instance)
        records_by_rho = {rho: [] for rho in rhos}
        missing = []

        for index, y in candidates:
            missing_rhos = []
            for rho in rhos:
                key = checkpoint_key(setting, rho, index)
                if cache.exists(key):
                    records_by_rho[rho].append(cache.load(key))
                else:
                    missing_rhos.append(rho)
            if missing_rhos:
                missing.append((index, y, tuple(missing_rhos)))

        atomic_write_text(
            root / "status.json",
            json.dumps(
                {
                    "status": "running",
                    "setting": setting,
                    "completed_evaluations": sum(
                        len(rows) for rows in records_by_rho.values()
                    ),
                    "total_evaluations": len(candidates) * len(rhos),
                    "pending_policy_bundles": len(missing),
                    "updated": time.time(),
                },
                indent=2,
            ),
        )

        if missing:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=mechanism.init_worker,
                initargs=(instance,),
            ) as pool:
                futures = {
                    pool.submit(mechanism.solve_radii, y, missing_rhos): (
                        index,
                        missing_rhos,
                    )
                    for index, y, missing_rhos in missing
                }
                for completed, future in enumerate(as_completed(futures), 1):
                    index, _ = futures[future]
                    for payload in future.result():
                        rho = float(payload["rho"])
                        payload["candidate_index"] = index
                        payload["setting"] = setting
                        payload["parameters"] = parameters
                        payload["experiment_version"] = VERSION
                        cache.save(
                            checkpoint_key(setting, rho, index),
                            payload,
                        )
                        records_by_rho[rho].append(payload)
                    if completed % 5 == 0 or completed == len(missing):
                        atomic_write_text(
                            root / "status.json",
                            json.dumps(
                                {
                                    "status": "running",
                                    "setting": setting,
                                    "completed_evaluations": sum(
                                        len(rows)
                                        for rows in records_by_rho.values()
                                    ),
                                    "total_evaluations": (
                                        len(candidates) * len(rhos)
                                    ),
                                    "completed_policy_bundles": completed,
                                    "pending_policy_bundles": len(missing),
                                    "updated": time.time(),
                                },
                                indent=2,
                            ),
                        )

        for rho in rhos:
            if len(records_by_rho[rho]) != len(candidates):
                raise RuntimeError(
                    f"Incomplete {setting}, rho={rho:.2f}: "
                    f"{len(records_by_rho[rho])}/{len(candidates)} records."
                )
            summary, top = summarize(
                setting,
                parameters,
                rho,
                claim.construct(rho, args, parameters),
                records_by_rho[rho],
            )
            summaries.append(summary)
            top_rows.extend(top)
        write_tables(root, summaries, top_rows)

    atomic_write_text(
        root / "status.json",
        json.dumps(
            {
                "status": "completed",
                "rows": len(summaries),
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
