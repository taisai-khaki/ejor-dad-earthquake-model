from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critical_revision_common import (
    BASE_RHOS,
    RESPONSE_PROFILES,
    atomic_json,
    base_output_root,
    build_m4_instance,
    candidate_grid,
    finish_run_metadata,
    json_string,
    load_source_records,
    model_hash,
    save_table,
    write_progress_log,
    write_run_metadata,
    write_status,
)
from ejor_dad.checkpoint import CheckpointStore
from ejor_dad.fixed_y import evaluate_fixed_plan, evaluate_fixed_y
from ejor_dad.recourse import solve_capability


BASE_TAUS = (0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01)
SENSITIVITY_RHOS = (0.0, 0.10, 0.125, 0.25)
SERVICE_RESOLUTION = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-output-dir", default=str(base_output_root()))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-base-frontier", action="store_true")
    parser.add_argument("--run-one-factor-sensitivities", action="store_true")
    parser.add_argument("--service-resolution", type=float, default=SERVICE_RESOLUTION)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def scenario_id(cap: float, profile: str) -> str:
    return f"k{cap:.2f}".replace('.', 'p') + f"__{profile}"


def build_instance(base_output_dir: Path, rho: float, cap: float, profile: str):
    return build_m4_instance(base_output_dir, rho, density_cap=cap, response_profile=profile)


def stage1_records(args: argparse.Namespace, base_output_dir: Path, output_dir: Path, rho: float, cap: float, profile: str) -> list[dict]:
    identifier = scenario_id(cap, profile)
    if cap == 2.0 and profile == "base":
        records = load_source_records(base_output_dir, rho, model_name="M4")
        if records:
            return records
    instance = build_instance(base_output_dir, rho, cap, profile)
    cache = CheckpointStore(output_dir / "stage1_checkpoints" / identifier / f"rho_{rho:.3f}".replace('.', 'p'))
    records = []
    candidates = candidate_grid(instance)
    for completed, (index, y) in enumerate(candidates, start=1):
        key = f"grid{index:04d}"
        if args.resume and cache.exists(key):
            payload = cache.load(key)
        else:
            try:
                result = evaluate_fixed_y(instance, y, epsilon=1e-6, max_iterations=500)
                payload = {"status": "feasible", "objective": result.objective, "lower_bound": result.lower_bound, "oracle_gap": result.objective - result.lower_bound, "y": result.y.tolist(), "z": result.z.tolist(), "w": result.w.tolist(), "candidate_index": index, "rho": rho, "density_cap": cap, "response_profile": profile}
            except RuntimeError as error:
                payload = {"status": "infeasible" if "infeasible" in str(error).lower() or "master lp failed" in str(error).lower() else "failed", "candidate_index": index, "rho": rho, "y": y.tolist(), "density_cap": cap, "response_profile": profile, "error": str(error)}
            cache.save(key, payload)
        if payload.get("status") == "feasible":
            records.append(payload)
        if completed % 20 == 0 or completed == len(candidates):
            write_status(output_dir / "status.json", status="running", block="equity_stage1", scenario_id=identifier, rho=rho, completed=completed, total=len(candidates), feasible=len(records))
    if not records:
        raise RuntimeError(f"No feasible stage-1 policies for {identifier}, rho={rho}.")
    return records


def service_check(instance, z: np.ndarray, w: np.ndarray, y: np.ndarray, service_floor: float) -> bool:
    candidate = replace(instance, minimum_zone_service_fraction=float(service_floor))
    for state in candidate.states:
        if np.any(candidate.service_fractions_for_state(state) > 1e-12) and not solve_capability(candidate, state, z, w, y=y).feasible:
            return False
    return True


def maximum_fixed_service(instance, z: Sequence[float], w: Sequence[float], y: Sequence[float], resolution: float, cache: CheckpointStore | None = None, key: str | None = None) -> tuple[float, float, int]:
    if cache is not None and key is not None and cache.exists(key):
        payload = cache.load(key)
        return float(payload["lower"]), float(payload["upper"]), int(payload["steps"])
    z = np.asarray(z, dtype=float)
    w = np.asarray(w, dtype=float)
    y = np.asarray(y, dtype=float)
    lower = 0.0
    upper = 1.0
    steps = 0
    if not service_check(instance, z, w, y, lower):
        result = (0.0, 0.0, 0)
    else:
        while upper - lower > resolution:
            midpoint = 0.5 * (lower + upper)
            steps += 1
            if service_check(instance, z, w, y, midpoint):
                lower = midpoint
            else:
                upper = midpoint
        result = (lower, upper, steps)
    if cache is not None and key is not None:
        cache.save(key, {"lower": result[0], "upper": result[1], "steps": result[2]})
    return result


def evaluate_reoptimized_service_floor(instance, y: Sequence[float], service_floor: float):
    candidate = replace(instance, minimum_zone_service_fraction=float(service_floor))
    try:
        return evaluate_fixed_y(
            candidate,
            np.asarray(y, dtype=float),
            epsilon=1e-6,
            max_iterations=500,
        )
    except RuntimeError as error:
        message = str(error).lower()
        if "infeasible" in message or "master lp failed" in message:
            return None
        raise


def maximum_reoptimized_service(
    instance,
    y: Sequence[float],
    objective_bound: float,
    resolution: float,
    cache: CheckpointStore | None = None,
    key: str | None = None,
) -> dict | None:
    if cache is not None and key is not None and cache.exists(key):
        payload = cache.load(key)
        if payload.get("status") == "feasible":
            return payload
        if payload.get("status") == "inadmissible":
            return None

    y = np.asarray(y, dtype=float)
    lower = 0.0
    upper = 1.0
    steps = 0
    best_result = evaluate_reoptimized_service_floor(instance, y, lower)
    if best_result is None or float(best_result.objective) > objective_bound + 1e-8:
        payload = {
            "status": "inadmissible",
            "y": y.tolist(),
            "objective_bound": float(objective_bound),
            "lower": 0.0,
            "upper": 0.0,
            "steps": 0,
        }
        if cache is not None and key is not None:
            cache.save(key, payload)
        return None

    while upper - lower > resolution:
        midpoint = 0.5 * (lower + upper)
        steps += 1
        result = evaluate_reoptimized_service_floor(instance, y, midpoint)
        if result is not None and float(result.objective) <= objective_bound + 1e-8:
            lower = midpoint
            best_result = result
        else:
            upper = midpoint

    payload = {
        "status": "feasible",
        "y": y.tolist(),
        "objective_bound": float(objective_bound),
        "lower": lower,
        "upper": upper,
        "steps": steps,
        "objective": float(best_result.objective),
        "lower_bound": float(best_result.lower_bound),
        "oracle_gap": float(best_result.objective - best_result.lower_bound),
        "z": best_result.z.tolist(),
        "w": best_result.w.tolist(),
    }
    if cache is not None and key is not None:
        cache.save(key, payload)
    return payload


def stage2_frontier(args, base_output_dir: Path, output_dir: Path, cap: float, profile: str, rhos: tuple[float, ...], taus: tuple[float, ...]) -> list[dict]:
    rows = []
    service_cache = CheckpointStore(output_dir / "service_checkpoints_reoptimized_v2" / scenario_id(cap, profile))
    for rho in rhos:
        instance = build_instance(base_output_dir, rho, cap, profile)
        records = stage1_records(args, base_output_dir, output_dir, rho, cap, profile)
        records.sort(key=lambda row: (float(row["objective"]), tuple(row["y"])))
        benchmark = float(records[0]["objective"])
        for tau in taus:
            bound = benchmark + 1e-5 if tau == 0.0 else benchmark * (1.0 + tau)
            admissible = [
                record
                for record in records
                if float(record["objective"]) <= bound + 1e-8
            ]
            candidates = []
            for position, record in enumerate(admissible, start=1):
                index = int(record.get("candidate_index", position))
                service_result = maximum_reoptimized_service(
                    instance,
                    record["y"],
                    bound,
                    args.service_resolution,
                    service_cache,
                    f"rho{rho:.3f}_tau{tau:.6f}_grid{index:04d}",
                )
                if service_result is not None:
                    candidates.append((service_result, record))
                if position % 10 == 0 or position == len(admissible):
                    write_status(
                        output_dir / "status.json",
                        status="running",
                        block="equity_stage2",
                        scenario_id=scenario_id(cap, profile),
                        rho=rho,
                        tau=tau,
                        service_evaluated=position,
                        service_total=len(admissible),
                    )
            if not candidates:
                raise RuntimeError(
                    f"No service-evaluable policies for {scenario_id(cap, profile)}, "
                    f"rho={rho}, tau={tau}."
                )
            candidates.sort(
                key=lambda item: (
                    -float(item[0]["lower"]),
                    float(item[0]["objective"]),
                    tuple(item[1]["y"]),
                )
            )
            selected_service, selected = candidates[0]
            ties = sum(
                float(item[0]["upper"]) >= float(selected_service["lower"]) - 1e-12
                for item in candidates
            )
            objective = float(selected_service["objective"])
            rows.append(
                {
                    "scenario_id": scenario_id(cap, profile),
                    "density_cap": cap,
                    "response_profile": profile,
                    "rho": rho,
                    "tau": tau,
                    "stage1_benchmark": benchmark,
                    "objective_bound": bound,
                    "stage2_service_lower": float(selected_service["lower"]),
                    "stage2_service_upper": float(selected_service["upper"]),
                    "stage2_objective": objective,
                    "stage2_objective_lower_bound": float(selected_service["lower_bound"]),
                    "realized_sacrifice": objective - benchmark,
                    "realized_sacrifice_percent": 100.0 * (objective - benchmark) / max(1.0, abs(benchmark)),
                    "selected_y_json": json_string(selected["y"]),
                    "selected_z_json": json_string(selected_service["z"]),
                    "selected_w_json": json_string(selected_service["w"]),
                    "admissible_policy_count": len(admissible),
                    "service_evaluable_policy_count": len(candidates),
                    "service_tie_count": ties,
                    "bisection_steps": int(selected_service["steps"]),
                    "bisection_resolution": args.service_resolution,
                    "maximum_oracle_gap": float(selected_service["oracle_gap"]),
                    "optimization_scope": "z,w reoptimized by evaluate_fixed_y at every service-floor bisection step",
                }
            )
            save_table(rows, output_dir / "tables" / "table_noto_stage2_working.csv", ["scenario_id", "rho", "tau"])
    return rows

def fixed_plan_decomposition(args, base_output_dir: Path, output_dir: Path) -> list[dict]:
    rows = []
    cache = CheckpointStore(output_dir / "decomposition_service_checkpoints")
    base_summary = pd.read_csv(base_output_dir / "correlated_facility_separated_capability_marginal_v2" / "tables" / "table_noto_correlated_facility.csv")
    plans = {}
    for rho in BASE_RHOS:
        row = base_summary.iloc[np.argmin(np.abs(base_summary.rho.to_numpy(dtype=float) - rho))]
        plans[rho] = {"z": np.asarray(json.loads(row.selected_z_json), dtype=float), "w": np.asarray(json.loads(row.selected_w_json), dtype=float), "y": np.asarray(json.loads(row.selected_y_json), dtype=float)}
    baseline = plans[0.0]
    for rho in BASE_RHOS:
        instance = build_instance(base_output_dir, rho, 2.0, "base")
        selected = plans[rho]
        hybrids = {"baseline": (baseline["z"], baseline["w"], baseline["y"]), "road_only": (baseline["z"], baseline["w"], selected["y"]), "capacity_only": (baseline["z"], selected["w"], baseline["y"]), "complete": (selected["z"], selected["w"], selected["y"])}
        for name, (z, w, y) in hybrids.items():
            key = f"rho{rho:.3f}_{name}"
            service_lower, service_upper, steps = maximum_fixed_service(instance, z, w, y, args.service_resolution, cache, key)
            plan = evaluate_fixed_plan(instance, z, w, y)
            rows.append({"rho": rho, "hybrid_plan": name, "maximum_fixed_plan_service_lower": service_lower, "maximum_fixed_plan_service_upper": service_upper, "robust_loss": plan.objective, "budget_feasible": True, "baseline_8_percent_feasible": service_check(instance, z, w, y, 0.08), "bisection_steps": steps, "y_json": json_string(y), "z_json": json_string(z), "w_json": json_string(w)})
    return rows


def make_figures(frontier: pd.DataFrame, metrics: pd.DataFrame, decomposition: pd.DataFrame, output_dir: Path) -> None:
    base = frontier[(frontier.density_cap == 2.0) & (frontier.response_profile == "base")]
    fig, ax = plt.subplots(figsize=(7, 4))
    for rho, group in base.groupby("rho"):
        ax.plot(group.tau * 100, group.stage2_service_lower, marker=".", label=f"rho={rho:g}")
    ax.set_xlabel("Robust-loss sacrifice (%)")
    ax.set_ylabel("Psi lower service bound")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "psi_vs_tau_base.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for tau, group in metrics[metrics.tau.isin([0.001, 0.005])].groupby("tau"):
        ax.plot(group.rho, group.complementarity_DID, marker="o", label=f"tau={tau*100:g}%")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Ambiguity radius")
    ax.set_ylabel("Complementarity DID")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "complementarity_DID_vs_rho.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, group in decomposition.groupby("hybrid_plan"):
        ax.plot(group.rho, group.maximum_fixed_plan_service_lower, marker=".", label=name)
    ax.set_xlabel("Ambiguity radius")
    ax.set_ylabel("Fixed-plan service lower bound")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "fixed_plan_service_decomposition.png", dpi=180)
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    base_output_dir = Path(args.base_output_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    for directory in (output_dir, output_dir / "tables", output_dir / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    if not (output_dir / "run_manifest.json").exists():
        write_run_metadata(output_dir, experiment="robustness_equity_frontier", parameters=vars(args), expected_work={"base_rhos": BASE_RHOS, "base_taus": BASE_TAUS, "sensitivity_rhos": SENSITIVITY_RHOS})
    rows = []
    if args.run_base_frontier or args.run_one_factor_sensitivities:
        rows.extend(stage2_frontier(args, base_output_dir, output_dir, 2.0, "base", BASE_RHOS, BASE_TAUS))
    if args.run_one_factor_sensitivities:
        for cap in (1.5, 3.0):
            rows.extend(stage2_frontier(args, base_output_dir, output_dir, cap, "base", SENSITIVITY_RHOS, (0.0, 0.001, 0.005)))
        for profile in ("lower_timely_credit", "higher_timely_credit"):
            rows.extend(stage2_frontier(args, base_output_dir, output_dir, 2.0, profile, SENSITIVITY_RHOS, (0.0, 0.001, 0.005)))
        rows.extend(stage2_frontier(args, base_output_dir, output_dir, 2.0, "base", SENSITIVITY_RHOS, (0.0, 0.001, 0.005)))
    frame = pd.DataFrame(rows).drop_duplicates(["scenario_id", "rho", "tau"], keep="last")
    save_table(frame.to_dict("records"), output_dir / "tables" / "table_noto_stage2_full_frontier.csv", ["scenario_id", "rho", "tau"])
    save_table(frame[frame.density_cap != 2.0].to_dict("records"), output_dir / "tables" / "table_noto_stage2_density_sensitivity.csv", ["scenario_id", "rho", "tau"])
    save_table(frame[frame.response_profile != "base"].to_dict("records"), output_dir / "tables" / "table_noto_stage2_response_sensitivity.csv", ["scenario_id", "rho", "tau"])
    metrics = []
    for scenario_id, group in frame.groupby("scenario_id"):
        group = group.sort_values(["rho", "tau"])
        for rho in sorted(float(value) for value in group.rho.unique() if float(value) > 0.0):
            for tau in sorted(float(value) for value in group.tau.unique() if float(value) > 0.0):
                current = group[np.isclose(group.rho, rho) & np.isclose(group.tau, tau)]
                nominal = group[np.isclose(group.rho, 0.0) & np.isclose(group.tau, tau)]
                current0 = group[np.isclose(group.rho, rho) & np.isclose(group.tau, 0.0)]
                nominal0 = group[np.isclose(group.rho, 0.0) & np.isclose(group.tau, 0.0)]
                if current.empty or nominal.empty or current0.empty or nominal0.empty:
                    continue
                current = current.iloc[0]
                nominal = nominal.iloc[0]
                current0 = current0.iloc[0]
                nominal0 = nominal0.iloc[0]
                metrics.append({"scenario_id": scenario_id, "density_cap": current.density_cap, "response_profile": current.response_profile, "rho": rho, "tau": tau, "Psi_rho": current.stage2_service_lower, "Psi_0": nominal.stage2_service_lower, "co_benefit": current.stage2_service_lower - nominal.stage2_service_lower, "incremental_equity_value": current.stage2_service_lower - current0.stage2_service_lower, "complementarity_DID": (current.stage2_service_lower - current0.stage2_service_lower) - (nominal.stage2_service_lower - nominal0.stage2_service_lower)})
    metrics_frame = pd.DataFrame(metrics)
    save_table(metrics_frame.to_dict("records"), output_dir / "tables" / "table_noto_robustness_equity_metrics.csv", ["scenario_id", "rho", "tau"])
    decomposition = pd.DataFrame(fixed_plan_decomposition(args, base_output_dir, output_dir))
    save_table(decomposition.to_dict("records"), output_dir / "tables" / "table_noto_fixed_plan_service_decomposition.csv", ["rho", "hybrid_plan"])
    base_metrics = metrics_frame[(metrics_frame.scenario_id == "k2p00__base") & (metrics_frame.rho.isin([0.125, 0.25])) & (metrics_frame.tau.isin([0.001, 0.005]))]
    base_sign_ok = len(base_metrics) == 4 and bool((base_metrics.complementarity_DID > args.service_resolution).all())
    nonbase = metrics_frame[(metrics_frame.scenario_id != "k2p00__base") & (metrics_frame.rho.isin([0.125, 0.25])) & (metrics_frame.tau.isin([0.001, 0.005]))]
    positive_counts = {str(tau): int((nonbase[np.isclose(nonbase.tau, tau)].complementarity_DID > args.service_resolution).sum()) for tau in (0.001, 0.005)}
    one_factor_sign_ok = all(value >= 6 for value in positive_counts.values())
    cover_path = output_dir.parent / "continuous_bb" / "tables" / "table_noto_continuous_policy_cover.csv"
    cover_unresolved = True
    if cover_path.exists():
        cover_table = pd.read_csv(cover_path)
        cover_unresolved = cover_table.empty or bool((cover_table.policy_class_status.astype(str) == "unresolved").any())
    complementarity_supported = bool(base_sign_ok and one_factor_sign_ok and not cover_unresolved)
    atomic_json(output_dir / "equity_gate.json", {"complementarity_supported": complementarity_supported, "base_sign_ok": base_sign_ok, "one_factor_positive_counts": positive_counts, "one_factor_sign_ok": one_factor_sign_ok, "continuous_cover_unresolved": cover_unresolved, "criterion": "DID positive at base transition radii, at least six of eight non-base comparisons per tested tolerance, and no unresolved continuous-policy certificate", "service_resolution": args.service_resolution})
    make_figures(frame, metrics_frame, decomposition, output_dir)
    write_status(output_dir / "status.json", status="completed", block="equity_frontier", rows=len(frame), complementarity_supported=complementarity_supported)
    finish_run_metadata(output_dir, status="completed", runtime_seconds=time.perf_counter() - started, extra={"rows": len(frame), "complementarity_supported": complementarity_supported})


if __name__ == "__main__":
    main()

