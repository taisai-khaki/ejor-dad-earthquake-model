from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critical_revision_common import atomic_json, finish_run_metadata, json_string, save_table, write_run_metadata, write_status
from ejor_dad import AidCenter, DADInstance, HazardRegime, Link, PiecewiseLinearResponseParams, Zone, generate_regime_failure_states
from ejor_dad.checkpoint import CheckpointStore
from ejor_dad.fixed_y import evaluate_fixed_y
from ejor_dad.monotone_bb import OracleEvaluation, run_monotone_box_bb

try:
    import psutil
except ImportError:
    psutil = None


SEEDS = (17, 29, 43)
PROFILES = ((0.0, 1.0), (30.0, 1.0), (60.0, 0.8), (100.0, 0.35), (180.0, 0.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dimensions", default="5,6,7,8")
    parser.add_argument("--relative-gap", type=float, default=0.001)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--enumeration-time-limit", type=float, default=7200.0)
    return parser.parse_args()


def generate_instance(L: int, seed: int) -> tuple[DADInstance, dict]:
    rng = np.random.default_rng(seed)
    links = []
    for index in range(L):
        base = float(rng.uniform(0.10, 0.30))
        links.append(Link(id=f"link_{index+1:02d}", tail=f"n{index}", head=f"n{index+1}", baseline_failure_probability=base, residual_failure_probability=0.15 * base, retrofit_cost=float(rng.uniform(0.8, 1.6)), travel_time=float(rng.uniform(8.0, 18.0)), failure_delay_reduction=float(rng.uniform(0.35, 0.70))))
    zones = [Zone(id=f"zone_{index+1:02d}", population=float(rng.uniform(850.0, 1400.0)), collapse_fraction=float(rng.uniform(0.08, 0.22)), renovation_cost=float(rng.uniform(0.8, 1.7)), node=f"n{index % max(2, L)}", region="synthetic", time_sensitive_fraction=0.8, immediate_loss_fraction=0.1) for index in range(5)]
    centers = [AidCenter(id=f"center_{index+1:02d}", node=f"n{(index + 1) % max(2, L)}", existing_capacity=2200.0 + 350.0 * index, capacity_unit_cost=1.0 + 0.1 * index) for index in range(3)]
    normal_multiplier = {link.id: float(rng.uniform(0.70, 1.25)) for link in links}
    severe_multiplier = {link.id: float(rng.uniform(1.45, 1.85)) for link in links}
    regimes = [HazardRegime(id="normal", probability=0.80, link_failure_multipliers=normal_multiplier), HazardRegime(id="severe", probability=0.20, failed_centers=("center_01",), link_failure_multipliers=severe_multiplier)]
    states = generate_regime_failure_states(links, regimes)
    intact = np.zeros((len(centers), len(zones)), dtype=float)
    for center_index in range(len(centers)):
        for zone_index in range(len(zones)):
            intact[center_index, zone_index] = 30.0 + 5.0 * center_index + 7.0 * zone_index + float(rng.uniform(-2.0, 2.0))
    penalty_matrices = {}
    for link_index, link in enumerate(links):
        matrix = np.zeros_like(intact)
        for center_index in range(len(centers)):
            for zone_index in range(len(zones)):
                coverage = 0.25 + 0.65 * (((link_index + 2 * center_index + zone_index) % 4) / 3.0)
                matrix[center_index, zone_index] = coverage * float(rng.uniform(10.0, 35.0))
        penalty_matrices[link.id] = matrix.tolist()
    precomputed = {}
    for state in states:
        matrix = intact.copy()
        for link in state.failed_links:
            matrix += np.asarray(penalty_matrices[link], dtype=float)
        precomputed[state.id] = matrix.tolist()
    critical = {state.id for state in states if state.hazard_regime_id == "normal" or len(state.failed_links) <= 1}
    instance = DADInstance(zones=zones, links=links, centers=centers, budget_renovation=0.35 * sum(zone.renovation_cost for zone in zones), budget_retrofit=0.38 * sum(link.retrofit_cost for link in links), budget_capacity=0.40 * sum(center.existing_capacity for center in centers), ambiguity_radius=0.10, ambiguity_density_cap=2.0, states=states, hazard_regimes=regimes, survival=PiecewiseLinearResponseParams(knots=PROFILES), precomputed_travel_times=precomputed, intact_travel_times=intact.tolist(), failure_penalty_matrices=penalty_matrices, minimum_protected_population=0.10 * sum(zone.at_risk for zone in zones), minimum_zone_service_fraction=0.05, critical_service_state_ids=critical)
    metadata = {"L": L, "R": 2, "state_count": len(states), "seed": seed, "budget_fraction": 0.38, "link_costs": [link.retrofit_cost for link in links], "link_fragilities": [link.baseline_failure_probability for link in links], "od_pairs_by_link": {key: int(np.count_nonzero(np.asarray(value) > 0.0)) for key, value in penalty_matrices.items()}, "regimes": [asdict(regime) for regime in regimes], "response_knots": PROFILES}
    return instance, metadata


def result_payload(result) -> dict:
    return {"status": "feasible", "objective": result.objective, "lower_bound": result.lower_bound, "oracle_gap": result.objective - result.lower_bound, "y": result.y.tolist(), "z": result.z.tolist(), "w": result.w.tolist(), "iterations": result.iterations}


def run_instance(args: argparse.Namespace, output: Path, L: int, seed: int):
    instance, metadata = generate_instance(L, seed)
    instance_id = f"synthetic_L{L}_seed{seed}"
    atomic_json(output / "inputs" / f"{instance_id}.json", metadata)
    cube_size = 5 ** L
    budget_count = sum(float(instance.retrofit_costs @ np.asarray(values)) <= instance.budget_retrofit + 1e-10 for values in product((0.0, 0.25, 0.50, 0.75, 1.0), repeat=L))
    cache = CheckpointStore(output / "checkpoints" / instance_id)
    enumeration_rows = []
    started = time.perf_counter()
    best_enum = None
    total = cube_size
    for completed, values in enumerate(product((0.0, 0.25, 0.50, 0.75, 1.0), repeat=L), start=1):
        y = np.asarray(values, dtype=float)
        key = "enum_" + "_".join(f"{value:.2f}" for value in values)
        if args.resume and cache.exists(key):
            payload = cache.load(key)
        else:
            try:
                payload = result_payload(evaluate_fixed_y(instance, y, epsilon=1e-6, max_iterations=500, enforce_retrofit_budget=False))
            except Exception as error:
                payload = {"status": "failed", "error": str(error), "y": y.tolist()}
            cache.save(key, payload)
        if payload.get("status") == "feasible":
            row = {"instance_id": instance_id, "algorithm": "tier_enumeration", "candidate_index": completed, "objective": float(payload["objective"]), "lower_bound": float(payload["lower_bound"]), "oracle_gap": float(payload["oracle_gap"]), "y_json": json_string(payload["y"]), "runtime_seconds": time.perf_counter() - started}
            enumeration_rows.append(row)
            if best_enum is None or row["objective"] < best_enum["objective"]:
                best_enum = row
        if completed % 100 == 0 or completed == total:
            write_status(output / "status.json", status="running", block="synthetic_enumeration", instance_id=instance_id, completed=completed, total=total, feasible=len(enumeration_rows))
        if time.perf_counter() - started > args.enumeration_time_limit:
            break
    enum_complete = len(enumeration_rows) == total
    if best_enum is None:
        return [], [], [{"instance_id": instance_id, "seed": seed, "L": L, "R": 2, "state_count": len(instance.states), "tier_cube_size": cube_size, "budget_feasible_tier_count": budget_count, "algorithm": "tier_enumeration", "converged": False, "termination_reason": "no_feasible_evaluation", "enumeration_complete": False}]
    seed_y = np.full(L, min(0.40, instance.budget_retrofit / max(1e-9, instance.retrofit_costs.sum())), dtype=float)
    while float(instance.retrofit_costs @ seed_y) > instance.budget_retrofit:
        seed_y *= 0.95
    try:
        seed_result = evaluate_fixed_y(instance, seed_y, epsilon=1e-6, max_iterations=500, enforce_retrofit_budget=True)
    except Exception:
        seed_result = evaluate_fixed_y(instance, np.zeros(L), epsilon=1e-6, max_iterations=500, enforce_retrofit_budget=True)
    corner_cache = {}
    checkpoint_calls = {"count": 0}
    bb_started = time.perf_counter()
    def oracle(y: np.ndarray) -> OracleEvaluation:
        key = "bb_" + "_".join(f"{value:.12f}" for value in np.round(y, 12))
        cache_key = tuple(np.round(y, 12).tolist())
        if cache_key in corner_cache:
            return corner_cache[cache_key]
        if args.resume and cache.exists(key):
            payload = cache.load(key)
        else:
            try:
                result = evaluate_fixed_y(instance, y, epsilon=1e-6, max_iterations=500, enforce_retrofit_budget=False)
                payload = result_payload(result)
            except Exception as error:
                try:
                    result = evaluate_fixed_y(instance, y, epsilon=1e-6, max_iterations=1000, enforce_retrofit_budget=False)
                    payload = result_payload(result)
                except Exception as retry_error:
                    payload = {"status": "failed", "error": f"{error}; retry: {retry_error}", "y": y.tolist()}
            cache.save(key, payload)
        checkpoint_calls["count"] += 1
        write_status(output / "status.json", status="running", block="synthetic_bnb", instance_id=instance_id, oracle_calls=checkpoint_calls["count"])
        if payload.get("status") != "feasible":
            raise RuntimeError(payload.get("error", "synthetic oracle failed"))
        evaluation = OracleEvaluation(status="feasible", objective=float(payload["objective"]), lower_bound=float(payload["lower_bound"]), y=np.asarray(payload["y"], dtype=float), z=np.asarray(payload["z"], dtype=float), w=np.asarray(payload["w"], dtype=float), iterations=int(payload["iterations"]), oracle_gap=float(payload["oracle_gap"]), payload=payload)
        corner_cache[cache_key] = evaluation
        return evaluation
    events = []
    def progress(record):
        events.append({"instance_id": instance_id, **dict(record)})
        if len(events) % 50 == 0:
            save_table(events, output / "tables" / "table_synthetic_gap_trajectories.csv", ["instance_id", "event_index"])
    bb_result = run_monotone_box_bb(initial_boxes=[(np.zeros(L), np.ones(L))], costs=instance.retrofit_costs, budget=instance.budget_retrofit, oracle=oracle, incumbent=OracleEvaluation(status="feasible", objective=float(seed_result.objective), lower_bound=float(seed_result.lower_bound), y=seed_result.y, z=seed_result.z, w=seed_result.w, iterations=seed_result.iterations, oracle_gap=seed_result.objective - seed_result.lower_bound), rel_gap_target=args.relative_gap, progress_callback=progress)
    save_table(events, output / "tables" / "table_synthetic_gap_trajectories.csv", ["instance_id", "event_index"])
    peak_memory_mb = None if psutil is None else psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    bnb_row = {"instance_id": instance_id, "seed": seed, "L": L, "R": 2, "state_count": len(instance.states), "tier_cube_size": cube_size, "budget_feasible_tier_count": budget_count, "algorithm": "adaptive_monotone_bb", "target_gap_percent": args.relative_gap * 100.0, "achieved_gap_percent": bb_result.relative_gap_percent, "converged": bb_result.converged, "incumbent_objective": bb_result.incumbent.objective, "global_lower_bound": bb_result.global_lower_bound, "unique_oracle_calls": bb_result.unique_oracle_calls, "nodes_processed": bb_result.nodes_processed, "nodes_pruned": bb_result.nodes_budget_pruned + bb_result.nodes_capability_pruned + bb_result.nodes_bound_pruned + bb_result.nodes_epsilon_pruned, "runtime_seconds": time.perf_counter() - bb_started, "peak_memory_mb": peak_memory_mb, "termination_reason": bb_result.termination_reason}
    enum_summary = {"instance_id": instance_id, "seed": seed, "L": L, "R": 2, "state_count": len(instance.states), "tier_cube_size": cube_size, "budget_feasible_tier_count": budget_count, "algorithm": "tier_enumeration", "target_gap_percent": 0.0, "achieved_gap_percent": 100.0 * float(best_enum["objective"] - best_enum["lower_bound"]) / max(1.0, abs(float(best_enum["objective"]))), "converged": enum_complete, "incumbent_objective": best_enum["objective"], "global_lower_bound": best_enum["lower_bound"], "unique_oracle_calls": len(enumeration_rows), "nodes_processed": len(enumeration_rows), "nodes_pruned": 0, "runtime_seconds": float(best_enum["runtime_seconds"]), "peak_memory_mb": peak_memory_mb, "termination_reason": "complete_cube" if enum_complete else "time_limit", "enumeration_complete": enum_complete, "best_y_json": best_enum["y_json"]}
    compare = {"instance_id": instance_id, "seed": seed, "L": L, "R": 2, "state_count": len(instance.states), "tier_cube_size": cube_size, "budget_feasible_tier_count": budget_count, "enumeration_complete": enum_complete, "enumeration_objective": best_enum["objective"], "bnb_objective": bb_result.incumbent.objective, "bnb_lower_bound": bb_result.global_lower_bound, "bnb_not_worse_than_enumeration": bool(bb_result.incumbent.objective <= best_enum["objective"] + 1e-6), "bnb_bound_valid": bool(bb_result.global_lower_bound <= bb_result.incumbent.objective + 1e-7)}
    return [bnb_row, enum_summary], events, [compare]


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    output = Path(args.output_dir).resolve()
    for directory in (output, output / "inputs", output / "checkpoints", output / "tables"):
        directory.mkdir(parents=True, exist_ok=True)
    dimensions = tuple(int(value) for value in args.dimensions.split(",") if value.strip())
    if not (output / "run_manifest.json").exists():
        write_run_metadata(output, experiment="synthetic_monotone_bb_scaling", parameters=vars(args), expected_work={"dimensions": dimensions, "seeds": SEEDS, "state_formula": "R*2^L"})
    rows, trajectories, comparisons = [], [], []
    for L in dimensions:
        seeds = SEEDS if L < 8 else (SEEDS[0],)
        for seed in seeds:
            result_rows, events, comparison_rows = run_instance(args, output, L, seed)
            rows.extend(result_rows)
            trajectories.extend(events)
            comparisons.extend(comparison_rows)
            save_table(rows, output / "tables" / "table_synthetic_scalability.csv", ["L", "seed", "algorithm"])
            save_table(comparisons, output / "tables" / "table_synthetic_enumeration_comparison.csv", ["L", "seed"])
    save_table(rows, output / "tables" / "table_synthetic_instance_manifest.csv", ["L", "seed", "algorithm"])
    save_table(trajectories, output / "tables" / "table_synthetic_gap_trajectories.csv", ["instance_id", "event_index"])
    save_table(comparisons, output / "tables" / "table_synthetic_enumeration_comparison.csv", ["L", "seed"])
    write_status(output / "status.json", status="completed", block="synthetic_scaling", rows=len(rows), comparisons=len(comparisons))
    finish_run_metadata(output, status="completed", runtime_seconds=time.perf_counter() - started, extra={"rows": len(rows), "comparisons": len(comparisons)})


if __name__ == "__main__":
    main()

