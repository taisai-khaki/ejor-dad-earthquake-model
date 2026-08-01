from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import noto_access_experiment as noto
import noto_correlated_facility_experiment as correlated
from ejor_dad import generate_regime_failure_states
from ejor_dad.checkpoint import CheckpointStore, atomic_write_dataframe, atomic_write_text
from ejor_dad.fixed_y import evaluate_fixed_y


VERSION = "noto-joint-sensitivity-separated-capability-marginal-v1"
GRID_LEVELS = (0.0, 0.25, 0.50, 0.75, 1.0)
DENSITY_CAPS = (1.5, 3.0)
DENSITY_CAP_RHOS = (0.075, 0.10, 0.125, 0.25)
RESPONSE_RHOS = (0.10, 0.25)
RESPONSE_PROFILES = {
    "lower_timely_credit": ((0.0, 1.0), (30.0, 1.0), (60.0, 0.70), (120.0, 0.20), (180.0, 0.0)),
    "base": ((0.0, 1.0), (30.0, 1.0), (60.0, 0.75), (120.0, 0.25), (180.0, 0.0)),
    "higher_timely_credit": ((0.0, 1.0), (30.0, 1.0), (60.0, 0.90), (120.0, 0.40), (180.0, 0.0)),
}
FIXED_Y_EPSILON = 1e-5
FIXED_Y_MAX_ITERATIONS = 240


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restart-safe final Noto joint-model density-cap and graded-response sensitivity enumeration."
    )
    parser.add_argument(
        "--base-output-dir",
        default="data_work/noto/acute_access_graded_v4",
        help="Frozen base-output directory used to construct the final joint model.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Sensitivity output directory; defaults to joint_sensitivity_separated_capability_marginal_v1 below the base output directory.",
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true", help="Recompute existing candidate checkpoints.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write a frozen manifest without solving.")
    parser.add_argument("--only-response", action="store_true", help="Run only the graded-response sensitivity block.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least one.")
    return args


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")


def write_status(output_dir: Path, **payload: Any) -> None:
    payload["updated_at_epoch"] = time.time()
    atomic_write_text(output_dir / "status.json", json.dumps(payload, indent=2, sort_keys=True))


def specification_payload(base_output_dir: Path, design: dict[str, Any], only_response: bool) -> dict[str, Any]:
    payload = {
        "version": VERSION,
        "base_output_dir": str(base_output_dir.resolve()),
        "base_run_design_sha256": hashlib.sha256((base_output_dir / "run_design.json").read_bytes()).hexdigest(),
        "model_scope": "128-state correlated road-facility regime model with graded timely-access response",
        "road_grid": list(GRID_LEVELS),
        "budget_feasible_candidate_count": 996,
        "density_cap_scenarios": ([] if only_response else [
            {"density_cap": density_cap, "rho_values": list(DENSITY_CAP_RHOS)}
            for density_cap in DENSITY_CAPS
        ]),
        "requested_blocks": ["graded_response"] if only_response else ["density_cap", "graded_response"],
        "response_scenarios": [
            {
                "response_profile": profile,
                "density_cap": float(design["density_cap"]),
                "rho_values": list(RESPONSE_RHOS),
                "knots": [list(knot) for knot in knots],
            }
            for profile, knots in RESPONSE_PROFILES.items()
        ],
        "fixed_y_epsilon": FIXED_Y_EPSILON,
        "fixed_y_max_iterations": FIXED_Y_MAX_ITERATIONS,
        "checkpointing": "one atomic JSON checkpoint per scenario-radius-policy",
        "base_results": "Frozen kappa=2, base-response full-grid rows are reused for comparisons only.",
    }
    payload["specification_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def validate_design(design: dict[str, Any]) -> None:
    expected_base = RESPONSE_PROFILES["base"]
    observed = tuple(tuple(float(value) for value in knot) for knot in design.get("graded_response_knots", []))
    if not design.get("graded_response"):
        raise RuntimeError("Frozen base output is not configured for graded response.")
    if not np.isclose(float(design.get("density_cap", np.nan)), 2.0):
        raise RuntimeError("Frozen base output must use density_cap=2.0.")
    if observed != expected_base:
        raise RuntimeError("Frozen base response knots do not match the pre-specified base response profile.")


def build_joint_instance(
    design: dict[str, Any],
    rho: float,
    density_cap: float,
    response_knots: Sequence[Sequence[float]],
) -> Any:
    base, _ = noto.build_noto_instance(
        rho,
        residual_failure_ratio=float(design["residual_failure_ratio"]),
        failure_delay_reduction=float(design["failure_delay_reduction"]),
        time_sensitive_fraction=float(design["time_sensitive_fraction"]),
        immediate_loss_fraction=float(design["immediate_loss_fraction"]),
        capacity_throughput_per_bed=float(design["capacity_throughput_per_bed"]),
        response_curve_knots=response_knots,
    )
    base = replace(
        base,
        ambiguity_density_cap=float(density_cap),
        budget_retrofit=float(design["retrofit_budget_scale"]) * base.budget_retrofit,
    )
    hazard_regimes = correlated.regimes(base)
    states = generate_regime_failure_states(base.links, hazard_regimes)
    critical_states = {
        state.id
        for state in states
        if state.hazard_regime_id in {"normal", "north", "central"} and len(state.failed_links) <= 1
    }
    return replace(
        base,
        states=states,
        hazard_regimes=hazard_regimes,
        critical_service_state_ids=critical_states,
        minimum_protected_population=0.10 * base.protected_population_coefficients.sum(),
        minimum_zone_service_fraction=0.08,
    )


def candidate_grid(instance: Any) -> list[tuple[int, np.ndarray]]:
    candidates = []
    for index, values in enumerate(product(GRID_LEVELS, repeat=len(instance.links)), start=1):
        y = np.asarray(values, dtype=float)
        if instance.retrofit_costs @ y <= instance.budget_retrofit + 1e-9:
            candidates.append((index, y))
    if len(candidates) != 996:
        raise RuntimeError(f"Expected 996 budget-feasible candidates; found {len(candidates)}.")
    return candidates


def evaluate_candidate(instance: Any, y: np.ndarray) -> dict[str, Any]:
    try:
        result = evaluate_fixed_y(
            instance,
            y,
            epsilon=FIXED_Y_EPSILON,
            max_iterations=FIXED_Y_MAX_ITERATIONS,
        )
        return {
            "status": "feasible",
            "objective": float(result.objective),
            "lower_bound": float(result.lower_bound),
            "y": result.y.tolist(),
            "z": result.z.tolist(),
            "w": result.w.tolist(),
            "iterations": int(result.iterations),
        }
    except RuntimeError as error:
        if "infeasible" in str(error).lower():
            return {"status": "infeasible", "y": np.asarray(y, dtype=float).tolist()}
        raise


def scenario_id(scenario_type: str, density_cap: float, response_profile: str) -> str:
    cap = f"{density_cap:.2f}".replace(".", "p")
    return f"{scenario_type}__k{cap}__{response_profile}"


def checkpoint_key(identifier: str, rho: float, candidate_index: int) -> str:
    rho_tag = f"{rho:.3f}".replace(".", "p")
    return f"{VERSION}__{identifier}__rho{rho_tag}__grid{candidate_index:04d}"


def scenarios(base_density_cap: float) -> list[dict[str, Any]]:
    output = [
        {
            "scenario_type": "density_cap",
            "density_cap": density_cap,
            "response_profile": "base",
            "response_knots": RESPONSE_PROFILES["base"],
            "rho_values": DENSITY_CAP_RHOS,
        }
        for density_cap in DENSITY_CAPS
    ]
    output.extend(
        {
            "scenario_type": "graded_response",
            "density_cap": base_density_cap,
            "response_profile": profile,
            "response_knots": knots,
            "rho_values": RESPONSE_RHOS,
        }
        for profile, knots in RESPONSE_PROFILES.items()
        if profile != "base"
    )
    for scenario in output:
        scenario["scenario_id"] = scenario_id(
            scenario["scenario_type"], scenario["density_cap"], scenario["response_profile"]
        )
    return output


def upsert(rows: list[dict[str, Any]], row: dict[str, Any]) -> list[dict[str, Any]]:
    key = (row["scenario_id"], float(row["rho"]))
    return [item for item in rows if (item["scenario_id"], float(item["rho"])) != key] + [row]


def load_table(path: Path) -> list[dict[str, Any]]:
    return [] if not path.exists() else pd.read_csv(path).to_dict("records")


def summarize(
    scenario: dict[str, Any], rho: float, instance: Any, records: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    feasible = [record for record in records if record.get("status") == "feasible"]
    if not feasible:
        raise RuntimeError(f"No feasible policies for {scenario['scenario_id']} at rho={rho:.3f}.")
    feasible.sort(key=lambda record: (float(record["objective"]), tuple(record["y"])))
    best = feasible[0]
    second = feasible[1] if len(feasible) > 1 else None
    tolerance = max(1e-8, 1e-10 * float(best["objective"]))
    ties = [record for record in feasible if float(record["objective"]) <= float(best["objective"]) + tolerance]
    return (
        {
            "experiment_version": VERSION,
            "scenario_id": scenario["scenario_id"],
            "scenario_type": scenario["scenario_type"],
            "density_cap": float(scenario["density_cap"]),
            "response_profile": scenario["response_profile"],
            "response_knots_json": json.dumps([list(knot) for knot in scenario["response_knots"]]),
            "rho": float(rho),
            "objective": float(best["objective"]),
            "lower_bound": float(best["lower_bound"]),
            "selected_y_json": json.dumps(best["y"]),
            "selected_z_json": json.dumps(best["z"]),
            "selected_w_json": json.dumps(best["w"]),
            "second_y_json": "" if second is None else json.dumps(second["y"]),
            "second_objective": np.nan if second is None else float(second["objective"]),
            "absolute_margin": np.nan if second is None else float(second["objective"]) - float(best["objective"]),
            "margin_percent": np.nan
            if second is None
            else 100.0 * (float(second["objective"]) / float(best["objective"]) - 1.0),
            "tie_count": len(ties),
            "operationally_feasible_count": len(feasible),
            "operationally_infeasible_count": len(records) - len(feasible),
            "state_count": len(instance.states),
            "critical_state_count": len(instance.critical_service_state_ids),
            "grid_scope": "full {0,0.25,0.50,0.75,1}^5 intersect road budget",
            "source": "new_sensitivity_run",
        },
        best,
    )


def selected_diagnostic(
    store: CheckpointStore,
    identifier: str,
    rho: float,
    instance: Any,
    selected_y: Sequence[float],
) -> dict[str, Any]:
    key = f"{identifier}__rho{rho:.3f}".replace(".", "p")
    y = np.asarray(selected_y, dtype=float)
    if store.exists(key):
        cached = store.load(key)
        if np.allclose(np.asarray(cached.get("selected_y", []), dtype=float), y):
            return cached
    result = evaluate_fixed_y(instance, y, epsilon=1e-8, max_iterations=300)
    nominal = np.asarray(result.nominal_distribution, dtype=float)
    worst_case = np.asarray(result.worst_case_distribution, dtype=float)
    positive = nominal > 1e-12
    density_ratio = np.divide(worst_case, nominal, out=np.zeros_like(worst_case), where=positive)
    widespread = np.asarray([state.hazard_regime_id == "widespread" for state in instance.states])
    widespread_ratios = density_ratio[widespread & positive]
    payload = {
        "scenario_id": identifier,
        "rho": float(rho),
        "selected_y": y.tolist(),
        "diagnostic_objective": float(result.objective),
        "actual_tv_movement": float(0.5 * np.abs(worst_case - nominal).sum()),
        "max_density_ratio": float(density_ratio[positive].max()),
        "density_cap_binding": bool(
            np.isclose(density_ratio[positive].max(), instance.ambiguity_density_cap, atol=1e-7)
        ),
        "widespread_nominal_mass": float(nominal[widespread].sum()),
        "widespread_worst_case_mass": float(worst_case[widespread].sum()),
        "widespread_mass_shift": float((worst_case - nominal)[widespread].sum()),
        "widespread_max_density_ratio": float(widespread_ratios.max()),
        "widespread_saturated_state_count": int(
            np.isclose(widespread_ratios, instance.ambiguity_density_cap, atol=1e-7).sum()
        ),
        "widespread_state_count": int(widespread.sum()),
    }
    store.save(key, payload)
    return payload


def frozen_rows(base_output_dir: Path, rhos: Sequence[float]) -> list[dict[str, Any]]:
    path = base_output_dir / "correlated_facility_separated_capability_marginal_v1" / "tables" / "table_noto_correlated_facility.csv"
    frame = pd.read_csv(path)
    rows = []
    for rho in rhos:
        matches = frame[np.isclose(frame["rho"].astype(float), rho)]
        if len(matches) != 1:
            raise RuntimeError(f"Frozen final-model result is missing rho={rho:.3f}.")
        row = matches.iloc[0]
        rows.append(
            {
                "scenario_id": "frozen_base_k2p00",
                "scenario_type": "frozen_baseline",
                "density_cap": 2.0,
                "response_profile": "base",
                "response_knots_json": json.dumps([list(knot) for knot in RESPONSE_PROFILES["base"]]),
                "rho": float(rho),
                "objective": float(row["objective"]),
                "selected_y_json": row["selected_y_json"],
                "selected_z_json": row["selected_z_json"],
                "selected_w_json": row["selected_w_json"],
                "tie_count": int(row["tie_count"]),
                "operationally_feasible_count": int(row["feasible_count"]),
                "operationally_infeasible_count": int(row["infeasible_count"]),
                "state_count": int(row["state_count"]),
                "critical_state_count": int(row["critical_state_count"]),
                "grid_scope": row["grid_scope"],
                "source": "frozen_base_archive",
            }
        )
    return rows


def write_reporting_tables(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    include_density_cap: bool = True,
) -> None:
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    if summaries:
        atomic_write_dataframe(
            pd.DataFrame(summaries).sort_values(["scenario_type", "density_cap", "response_profile", "rho"]),
            tables / "table_noto_joint_sensitivity_full_grid.csv",
        )
    merged = pd.DataFrame(diagnostics)
    baseline = pd.DataFrame(base_rows)
    if merged.empty:
        return
    baseline_diagnostics = merged[merged["scenario_type"] == "frozen_baseline"]
    cap = pd.concat(
        [baseline_diagnostics[baseline_diagnostics["rho"].isin(DENSITY_CAP_RHOS)], merged[merged["scenario_type"] == "density_cap"]],
        ignore_index=True,
    )
    if include_density_cap and not cap.empty:
        cap["scenario_type"] = "density_cap"
        reference = cap[np.isclose(cap["density_cap"].astype(float), 2.0)][["rho", "selected_y_json"]].rename(
            columns={"selected_y_json": "base_k2_selected_y_json"}
        )
        cap = cap.merge(reference, on="rho", how="left")
        cap["policy_differs_from_k2"] = cap["selected_y_json"] != cap["base_k2_selected_y_json"]
        atomic_write_dataframe(cap.sort_values(["rho", "density_cap"]), tables / "table_noto_density_cap_sensitivity.csv")
    response = pd.concat(
        [baseline_diagnostics[baseline_diagnostics["rho"].isin(RESPONSE_RHOS)], merged[merged["scenario_type"] == "graded_response"]],
        ignore_index=True,
    )
    if not response.empty:
        response["scenario_type"] = "graded_response"
        atomic_write_dataframe(
            response.sort_values(["rho", "response_profile"]),
            tables / "table_noto_graded_response_sensitivity.csv",
        )


def solve_scenario(
    output_dir: Path,
    design: dict[str, Any],
    scenario: dict[str, Any],
    workers: int,
    force: bool,
    summaries: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    diagnostic_store: CheckpointStore,
    log_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    store = CheckpointStore(output_dir / "checkpoints" / scenario["scenario_id"])
    for rho in scenario["rho_values"]:
        instance = build_joint_instance(design, rho, scenario["density_cap"], scenario["response_knots"])
        candidates = candidate_grid(instance)
        records = []
        pending = []
        for candidate_index, y in candidates:
            key = checkpoint_key(scenario["scenario_id"], rho, candidate_index)
            if not force and store.exists(key):
                records.append(store.load(key))
            else:
                pending.append((candidate_index, y, key))
        append_log(
            log_path,
            f"{scenario['scenario_id']} rho={rho:.3f}: reused={len(records)}, pending={len(pending)}.",
        )
        write_status(
            output_dir,
            status="running",
            scenario_id=scenario["scenario_id"],
            scenario_type=scenario["scenario_type"],
            density_cap=scenario["density_cap"],
            response_profile=scenario["response_profile"],
            rho=rho,
            completed=len(records),
            pending=len(pending),
            total=len(candidates),
            log_path=str(log_path),
        )
        if pending:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(evaluate_candidate, instance, y): (candidate_index, key)
                    for candidate_index, y, key in pending
                }
                for completed, future in enumerate(as_completed(futures), start=1):
                    candidate_index, key = futures[future]
                    payload = future.result()
                    payload["candidate_index"] = candidate_index
                    store.save(key, payload)
                    records.append(payload)
                    if completed % 10 == 0 or completed == len(pending):
                        write_status(
                            output_dir,
                            status="running",
                            scenario_id=scenario["scenario_id"],
                            scenario_type=scenario["scenario_type"],
                            density_cap=scenario["density_cap"],
                            response_profile=scenario["response_profile"],
                            rho=rho,
                            completed=len(candidates) - len(pending) + completed,
                            pending=len(pending) - completed,
                            total=len(candidates),
                            log_path=str(log_path),
                        )
        if len(records) != len(candidates):
            raise RuntimeError(
                f"Incomplete {scenario['scenario_id']} rho={rho:.3f}: {len(records)}/{len(candidates)} records."
            )
        summary, best = summarize(scenario, rho, instance, records)
        diagnostic = selected_diagnostic(diagnostic_store, scenario["scenario_id"], rho, instance, best["y"])
        summaries = upsert(summaries, summary)
        diagnostics = upsert(diagnostics, {**summary, **diagnostic})
        atomic_write_dataframe(
            pd.DataFrame(summaries).sort_values(["scenario_type", "density_cap", "response_profile", "rho"]),
            output_dir / "tables" / "table_noto_joint_sensitivity_full_grid.csv",
        )
        append_log(
            log_path,
            f"{scenario['scenario_id']} rho={rho:.3f}: objective={summary['objective']:.6f}, y={summary['selected_y_json']}.",
        )
    return summaries, diagnostics


def write_reproducibility(output_dir: Path) -> None:
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {package: metadata.version(package) for package in ("numpy", "pandas", "scipy")},
        "solver": "scipy.optimize.linprog(method='highs')",
        "fixed_y_epsilon": FIXED_Y_EPSILON,
        "fixed_y_max_iterations": FIXED_Y_MAX_ITERATIONS,
    }
    atomic_write_text(output_dir / "reproducibility.json", json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    base_output_dir = Path(args.base_output_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else base_output_dir / "joint_sensitivity_separated_capability_marginal_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    log_path = output_dir / "logs" / f"joint_sensitivity_{time.strftime('%Y%m%d_%H%M%S')}.log"
    design_path = base_output_dir / "run_design.json"
    if not design_path.exists():
        raise RuntimeError(f"Frozen run design not found: {design_path}")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    validate_design(design)
    specification = specification_payload(base_output_dir, design, args.only_response)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("specification_sha256") != specification["specification_sha256"]:
            raise RuntimeError("Existing sensitivity directory has a different frozen specification.")
    else:
        atomic_write_text(manifest_path, json.dumps(specification, indent=2))
    write_reproducibility(output_dir)
    run_scenarios = scenarios(float(design["density_cap"]))
    if args.only_response:
        run_scenarios = [scenario for scenario in run_scenarios if scenario["scenario_type"] == "graded_response"]
    expected_evaluations = sum(996 * len(scenario["rho_values"]) for scenario in run_scenarios)
    if args.dry_run:
        write_status(
            output_dir,
            status="ready",
            expected_new_evaluations=expected_evaluations,
            scenario_count=len(run_scenarios),
            message="Inputs, frozen specification, and restart-safe checkpoint layout validated.",
            log_path=str(log_path),
        )
        append_log(log_path, "Dry run completed successfully.")
        return
    started = time.time()
    append_log(log_path, f"Started {VERSION} with {args.workers} worker processes.")
    write_status(
        output_dir,
        status="running",
        expected_new_evaluations=expected_evaluations,
        scenario_count=len(run_scenarios),
        message="Restart-safe final joint-model sensitivity enumeration started.",
        log_path=str(log_path),
    )
    summaries = load_table(output_dir / "tables" / "table_noto_joint_sensitivity_full_grid.csv")
    diagnostics = []
    diagnostic_store = CheckpointStore(output_dir / "selected_diagnostics")
    try:
        for scenario in run_scenarios:
            summaries, diagnostics = solve_scenario(
                output_dir,
                design,
                scenario,
                args.workers,
                args.force,
                summaries,
                diagnostics,
                diagnostic_store,
                log_path,
            )
        baseline_rhos = tuple(sorted(set(DENSITY_CAP_RHOS) | set(RESPONSE_RHOS)))
        baseline = frozen_rows(base_output_dir, baseline_rhos)
        for row in baseline:
            instance = build_joint_instance(design, row["rho"], 2.0, RESPONSE_PROFILES["base"])
            diagnostic = selected_diagnostic(
                diagnostic_store,
                row["scenario_id"],
                row["rho"],
                instance,
                json.loads(row["selected_y_json"]),
            )
            diagnostics.append({**row, **diagnostic})
        write_reporting_tables(output_dir, summaries, diagnostics, baseline, include_density_cap=not args.only_response)
        runtime = time.time() - started
        atomic_write_text(output_dir / "runtime_summary.json", json.dumps({"runtime_sec": runtime}, indent=2))
        append_log(log_path, f"Completed in {runtime:.3f} seconds.")
        write_status(
            output_dir,
            status="completed",
            expected_new_evaluations=expected_evaluations,
            completed_summary_rows=len(summaries),
            message="Density-cap and graded-response final joint-model sensitivity enumeration completed.",
            runtime_sec=runtime,
            log_path=str(log_path),
        )
    except Exception as error:
        append_log(log_path, f"FAILED: {type(error).__name__}: {error}")
        write_status(
            output_dir,
            status="failed",
            expected_new_evaluations=expected_evaluations,
            message=str(error),
            log_path=str(log_path),
        )
        raise


if __name__ == "__main__":
    main()



