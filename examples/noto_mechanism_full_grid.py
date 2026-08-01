from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noto_correlated_facility_experiment as correlated
import noto_correlated_validation_postprocess as validation
import noto_practical_resilience_experiment as practical
from ejor_dad import HazardRegime, generate_regime_failure_states
from ejor_dad.checkpoint import CheckpointStore, atomic_write_text
from ejor_dad.fixed_y import evaluate_fixed_y

RHOS = (0.0, 0.10, 0.25)
GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
ORDER = ("M4", "M2", "M0", "M1", "M3")
VERSION = "noto-mechanism-separated-capability-marginal-v2"
_WORKER_INSTANCE = None


def specs(base):
    shared = correlated.regimes(base)
    average = {
        link.id: sum(
            regime.probability
            * regime.link_failure_multipliers.get(link.id, 1.0)
            for regime in shared
        )
        for link in base.links
    }
    m0 = [
        HazardRegime(regime.id, regime.probability, (), average)
        for regime in shared
    ]
    m1 = [
        HazardRegime(
            regime.id,
            regime.probability,
            (),
            regime.link_failure_multipliers,
        )
        for regime in shared
    ]
    m2 = [
        HazardRegime(
            regime.id,
            regime.probability,
            regime.failed_centers,
            average,
        )
        for regime in shared
    ]
    m3 = [
        HazardRegime(
            f"{road.id}__{facility.id}",
            road.probability * facility.probability,
            facility.failed_centers,
            road.link_failure_multipliers,
        )
        for road in shared
        for facility in shared
    ]
    return {"M0": m0, "M1": m1, "M2": m2, "M3": m3, "M4": shared}


def build(base, regimes, model):
    states = generate_regime_failure_states(base.links, regimes)
    if model == "M3":
        critical = {
            state.id
            for state in states
            if state.hazard_regime_id.split("__")[1]
            in {"normal", "north", "central"}
            and len(state.failed_links) <= 1
        }
    else:
        critical = {
            state.id
            for state in states
            if state.hazard_regime_id in {"normal", "north", "central"}
            and len(state.failed_links) <= 1
        }
    return replace(
        base,
        states=states,
        hazard_regimes=regimes,
        critical_service_state_ids=critical,
        minimum_protected_population=(
            0.10 * base.protected_population_coefficients.sum()
        ),
        minimum_zone_service_fraction=0.08,
    )


def init_worker(instance):
    global _WORKER_INSTANCE
    _WORKER_INSTANCE = instance


def solve_radii(y, rhos):
    if _WORKER_INSTANCE is None:
        raise RuntimeError("Mechanism worker was not initialized.")
    y_vec = np.asarray(y, dtype=float)
    cuts = None
    records = []
    operationally_infeasible = False
    for rho in sorted(float(value) for value in rhos):
        if operationally_infeasible:
            records.append(
                {
                    "status": "infeasible",
                    "rho": rho,
                    "y": y_vec.tolist(),
                    "message": "Operational infeasibility propagated across radii.",
                }
            )
            continue
        instance = replace(_WORKER_INSTANCE, ambiguity_radius=rho)
        started = time.perf_counter()
        try:
            result = evaluate_fixed_y(
                instance,
                y_vec,
                epsilon=1e-5,
                max_iterations=300,
                initial_cuts=cuts,
            )
        except RuntimeError as exc:
            if "infeasible" not in str(exc).lower():
                raise
            operationally_infeasible = True
            records.append(
                {
                    "status": "infeasible",
                    "rho": rho,
                    "y": y_vec.tolist(),
                    "message": str(exc),
                    "runtime_seconds": time.perf_counter() - started,
                }
            )
            continue
        cuts = result.cuts
        records.append(
            {
                "status": "feasible",
                "rho": rho,
                "objective": float(result.objective),
                "lower_bound": float(result.lower_bound),
                "oracle_gap": float(result.objective - result.lower_bound),
                "y": result.y.tolist(),
                "z": result.z.tolist(),
                "w": result.w.tolist(),
                "iterations": int(result.iterations),
                "runtime_seconds": time.perf_counter() - started,
            }
        )
    return records


def candidate_grid(instance):
    candidates = []
    for index, values in enumerate(product(GRID, repeat=len(instance.links)), 1):
        y = np.asarray(values, dtype=float)
        if instance.retrofit_costs @ y <= instance.budget_retrofit + 1e-9:
            candidates.append((index, y))
    if len(candidates) != 996:
        raise RuntimeError(
            f"Expected 996 budget-feasible policies, found {len(candidates)}."
        )
    return candidates


def key_for(model, rho, index):
    return f"mechanism_{model}_rho{rho:.2f}_grid{index:04d}"


def summarize(model, rho, instance, records):
    feasible = [row for row in records if row.get("status") == "feasible"]
    if not feasible:
        raise RuntimeError(f"No feasible policies for {model}, rho={rho:.2f}.")
    feasible.sort(key=lambda row: (float(row["objective"]), tuple(row["y"])))
    best = float(feasible[0]["objective"])
    second = float(feasible[1]["objective"])
    summary = {
        "experiment_version": VERSION,
        "model": model,
        "rho": rho,
        "selected_y_json": json.dumps(feasible[0]["y"]),
        "objective": best,
        "selected_z_json": json.dumps(feasible[0]["z"]),
        "selected_w_json": json.dumps(feasible[0]["w"]),
        "second_y_json": json.dumps(feasible[1]["y"]),
        "second_objective": second,
        "absolute_margin": second - best,
        "margin_percent": 100.0 * (second / best - 1.0),
        "within_0p01_percent": sum(
            float(row["objective"]) <= best * 1.0001 for row in feasible
        ),
        "within_0p10_percent": sum(
            float(row["objective"]) <= best * 1.001 for row in feasible
        ),
        "within_0p50_percent": sum(
            float(row["objective"]) <= best * 1.005 for row in feasible
        ),
        "operationally_feasible_count": len(feasible),
        "unacceptable_count": 996 - len(feasible),
        "state_count": len(instance.states),
        "design_basis_count": len(instance.critical_service_state_ids),
        "marginal_matching": (
            "M0/M1 road marginals matched; M2/M3/M4 facility marginals "
            "matched; M3/M4 road marginals matched"
        ),
    }
    top = [
        {
            "experiment_version": VERSION,
            "model": model,
            "rho": rho,
            "rank": rank,
            "objective": row["objective"],
            "gap": float(row["objective"]) - best,
            "gap_percent": 100.0 * (float(row["objective"]) / best - 1.0),
            "y_json": json.dumps(row["y"]),
            "z_json": json.dumps(row["z"]),
            "w_json": json.dumps(row["w"]),
        }
        for rank, row in enumerate(feasible[:5], 1)
    ]
    return summary, top


def write_tables(root, summaries, top_rows):
    summaries = sorted(summaries, key=lambda row: (row["model"], row["rho"]))
    top_rows = sorted(
        top_rows,
        key=lambda row: (row["model"], row["rho"], row["rank"]),
    )
    pd.DataFrame(summaries).to_csv(
        root / "tables" / "table_noto_mechanism_ablation_full_grid.csv",
        index=False,
    )
    pd.DataFrame(top_rows).to_csv(
        root / "tables" / "table_noto_mechanism_ablation_full_grid_top5.csv",
        index=False,
    )


def main(output_dir, workers):
    design = json.loads((output_dir / "run_design.json").read_text())
    args = validation.args_from(design, output_dir)
    base, _ = practical.build_instance(0.0, args)
    root = output_dir / "mechanism_separated_capability_marginal_v2"
    (root / "tables").mkdir(parents=True, exist_ok=True)
    cache = CheckpointStore(root / "checkpoints")
    atomic_write_text(
        root / "run_manifest.json",
        json.dumps(
            {
                "experiment_version": VERSION,
                "rhos": RHOS,
                "models": ORDER,
                "grid": GRID,
                "expected_grid_count": 996,
                "specification_sources": [
                    "user-supplied joint-state/model validation attachment",
                    "user-supplied remaining_computation_guideline.md",
                    "user-supplied remaining-analysis attachment",
                ],
                "checkpointing": "one atomic JSON per model-radius-policy",
                "ascending_radius_warm_start": True,
                "workers": workers,
                "started": time.time(),
            },
            indent=2,
        ),
    )
    summaries = []
    top_rows = []
    all_specs = specs(base)

    for model in ORDER:
        instance = build(base, all_specs[model], model)
        candidates = candidate_grid(instance)
        records_by_rho = {rho: [] for rho in RHOS}
        missing = []
        for index, y in candidates:
            missing_rhos = []
            for rho in RHOS:
                key = key_for(model, rho, index)
                payload = None
                if cache.exists(key):
                    payload = cache.load(key)
                if payload is None:
                    missing_rhos.append(rho)
                else:
                    records_by_rho[rho].append(payload)
            if missing_rhos:
                missing.append((index, y, tuple(missing_rhos)))

        existing_count = sum(len(rows) for rows in records_by_rho.values())
        atomic_write_text(
            root / "status.json",
            json.dumps(
                {
                    "status": "running",
                    "model": model,
                    "completed_evaluations": existing_count,
                    "total_evaluations": len(candidates) * len(RHOS),
                    "pending_policy_bundles": len(missing),
                    "updated": time.time(),
                },
                indent=2,
            ),
        )

        if missing:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=init_worker,
                initargs=(instance,),
            ) as pool:
                futures = {
                    pool.submit(solve_radii, y, rhos): (index, y, rhos)
                    for index, y, rhos in missing
                }
                for completed, future in enumerate(as_completed(futures), 1):
                    index, _, _ = futures[future]
                    for payload in future.result():
                        rho = float(payload["rho"])
                        payload["candidate_index"] = index
                        payload["model"] = model
                        payload["experiment_version"] = VERSION
                        cache.save(key_for(model, rho, index), payload)
                        records_by_rho[rho].append(payload)
                    if completed % 5 == 0 or completed == len(missing):
                        completed_count = sum(
                            len(rows) for rows in records_by_rho.values()
                        )
                        atomic_write_text(
                            root / "status.json",
                            json.dumps(
                                {
                                    "status": "running",
                                    "model": model,
                                    "completed_evaluations": completed_count,
                                    "total_evaluations": len(candidates) * len(RHOS),
                                    "completed_policy_bundles": completed,
                                    "pending_policy_bundles": len(missing),
                                    "updated": time.time(),
                                },
                                indent=2,
                            ),
                        )

        for rho in RHOS:
            if len(records_by_rho[rho]) != len(candidates):
                raise RuntimeError(
                    f"Incomplete {model}, rho={rho:.2f}: "
                    f"{len(records_by_rho[rho])}/{len(candidates)} records."
                )
            summary, top = summarize(
                model,
                rho,
                replace(instance, ambiguity_radius=rho),
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
                "models": ORDER,
                "rhos": RHOS,
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
