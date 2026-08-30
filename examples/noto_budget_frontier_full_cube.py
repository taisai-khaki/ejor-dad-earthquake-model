from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critical_revision_common import (
    BASE_RHOS,
    GRID_LEVELS,
    atomic_json,
    base_output_root,
    build_m4_instance,
    candidate_grid,
    finish_run_metadata,
    json_string,
    model_hash,
    save_table,
    source_commit,
    write_progress_log,
    write_run_metadata,
    write_status,
)
from ejor_dad.checkpoint import CheckpointStore
from ejor_dad.fixed_y import evaluate_fixed_y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-output-dir", default=str(base_output_root()))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rhos", default=','.join(str(value) for value in BASE_RHOS))
    parser.add_argument("--budget-start", type=float, default=0.20)
    parser.add_argument("--budget-stop", type=float, default=0.65)
    parser.add_argument("--budget-step", type=float, default=0.025)
    parser.add_argument("--include-budget", type=float, default=0.42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def budget_fractions(args: argparse.Namespace) -> list[float]:
    values = []
    current = args.budget_start
    while current <= args.budget_stop + 1e-10:
        values.append(round(current, 10))
        current += args.budget_step
    if all(abs(value - args.include_budget) > 1e-10 for value in values):
        values.append(float(args.include_budget))
    return sorted(set(values))


def load_existing_checkpoint(base_output_dir: Path, rho: float, index: int, y: np.ndarray) -> dict | None:
    path = base_output_dir / "correlated_facility_separated_capability_marginal_v2" / "checkpoints" / f"noto-correlated-facility-separated-capability-marginal-v2_rho{rho:.2f}_grid{index:04d}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") != "feasible" or not np.allclose(np.asarray(payload.get("y"), dtype=float), y, atol=1e-12, rtol=0.0):
        return None
    return payload


def evaluate_cube(args: argparse.Namespace, base_output_dir: Path, output_dir: Path, rho: float) -> list[dict]:
    instance = build_m4_instance(base_output_dir, rho)
    digest = model_hash(base_output_dir)
    cache = CheckpointStore(output_dir / "checkpoints" / f"rho_{rho:.3f}".replace('.', 'p'))
    rows = []
    candidates = candidate_grid(instance, enforce_budget=False)
    for completed, (index, y) in enumerate(candidates, start=1):
        key = f"grid{index:04d}_model{digest[:16]}"
        payload = cache.load(key) if args.resume and cache.exists(key) else load_existing_checkpoint(base_output_dir, rho, index, y)
        if payload is None:
            try:
                result = evaluate_fixed_y(instance, y, epsilon=1e-6, max_iterations=500, enforce_retrofit_budget=False)
                payload = {"status": "feasible", "objective": result.objective, "lower_bound": result.lower_bound, "oracle_gap": result.objective - result.lower_bound, "y": result.y.tolist(), "z": result.z.tolist(), "w": result.w.tolist(), "iterations": result.iterations}
            except RuntimeError as error:
                payload = {"status": "infeasible" if "infeasible" in str(error).lower() or "master lp failed" in str(error).lower() else "failed", "y": y.tolist(), "error": str(error)}
            cache.save(key, payload)
        row = {"rho": rho, "candidate_index": index, "y_json": json_string(y), "status": payload.get("status"), "objective": payload.get("objective"), "lower_bound": payload.get("lower_bound"), "oracle_gap": payload.get("oracle_gap"), "z_json": json.dumps(payload.get("z")) if payload.get("z") is not None else None, "w_json": json.dumps(payload.get("w")) if payload.get("w") is not None else None, "iterations": payload.get("iterations"), "road_cost_base": float(instance.retrofit_costs @ y), "road_cost_fraction_base": float(instance.retrofit_costs @ y / instance.retrofit_costs.sum()), "state_count": len(instance.states), "model_hash": digest}
        rows.append(row)
        if completed % 25 == 0 or completed == len(candidates):
            save_table(rows, output_dir / "tables" / "table_noto_full_tier_cube.csv", ["rho", "candidate_index"])
            write_status(output_dir / "status.json", status="running", block="budget_frontier", rho=rho, completed=completed, total=len(candidates), total_rho_count=len(args.rhos.split(',')))
    return rows


def summarize_budget(rows: list[dict], instance, fractions: list[float], base_budget_fraction: float) -> list[dict]:
    frame = pd.DataFrame(rows)
    feasible = frame[frame.status == "feasible"].copy()
    output = []
    for fraction in fractions:
        subset = feasible[feasible.road_cost_fraction_base <= fraction + 1e-10].sort_values(["objective", "candidate_index"])
        if subset.empty:
            continue
        best = subset.iloc[0]
        second = subset.iloc[1] if len(subset) > 1 else best
        best_value = float(best.objective)
        y = np.asarray(json.loads(best.y_json), dtype=float)
        output.append({"rho": float(best.rho), "budget_fraction": fraction, "absolute_budget": fraction * float(instance.retrofit_costs.sum()), "objective": best_value, "second_objective": float(second.objective), "absolute_margin": float(second.objective - best.objective), "margin_percent": 100.0 * float(second.objective - best.objective) / max(1.0, abs(best_value)), "selected_y_json": best.y_json, "selected_z_json": best.z_json, "selected_w_json": best.w_json, "road_budget_used": float(instance.retrofit_costs @ y), "road_budget_slack": fraction * float(instance.retrofit_costs.sum()) - float(instance.retrofit_costs @ y), "feasible_tier_count": len(subset), "within_0p01_percent": int((subset.objective <= best_value * 1.0001).sum()), "within_0p10_percent": int((subset.objective <= best_value * 1.001).sum()), "within_0p50_percent": int((subset.objective <= best_value * 1.005).sum()), "corridor_2_full": bool(y[1] >= 1.0 - 1e-12), "corridor_3_full": bool(y[2] >= 1.0 - 1e-12), "residual_split_two_corridors": bool((y > 1e-12).sum() >= 2 and not (y[1] >= 1.0 - 1e-12 and y[2] >= 1.0 - 1e-12)), "base_budget_fraction": base_budget_fraction})
    return output


def cost_perturbations(rows: list[dict], instance, base_budget: float) -> list[dict]:
    frame = pd.DataFrame(rows)
    feasible = frame[frame.status == "feasible"].copy()
    output = []
    for rho in sorted(feasible.rho.unique()):
        current = feasible[feasible.rho == rho]
        for corridor in range(len(instance.links)):
            for multiplier in (0.80, 0.90, 1.10, 1.20):
                selected = []
                costs = instance.retrofit_costs.copy()
                costs[corridor] *= multiplier
                for _, row in current.iterrows():
                    y = np.asarray(json.loads(row.y_json), dtype=float)
                    if float(costs @ y) <= base_budget + 1e-10:
                        selected.append((float(row.objective), row, y))
                selected.sort(key=lambda item: (item[0], int(item[1].candidate_index)))
                if not selected:
                    continue
                best_value, best_row, y = selected[0]
                second = selected[1][0] if len(selected) > 1 else best_value
                output.append({"rho": rho, "perturbed_corridor_index": corridor + 1, "corridor_id": instance.links[corridor].id, "cost_multiplier": multiplier, "objective": best_value, "second_objective": second, "margin_percent": 100.0 * (second - best_value) / max(1.0, abs(best_value)), "selected_y_json": json_string(y), "perturbed_budget_used": float(costs @ y), "perturbed_budget_slack": base_budget - float(costs @ y), "feasible_tier_count": len(selected), "corridor_2_full": bool(y[1] >= 1.0 - 1e-12), "corridor_3_full": bool(y[2] >= 1.0 - 1e-12)})
    return output


def change_points(frontier: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rho, group in frontier.groupby("rho"):
        group = group.sort_values("budget_fraction")
        previous = None
        for _, row in group.iterrows():
            policy = row.selected_y_json
            if policy != previous:
                rows.append({"rho": rho, "budget_fraction": row.budget_fraction, "selected_y_json": policy, "change_type": "budget_change_point"})
                previous = policy
    for fraction, group in frontier.groupby("budget_fraction"):
        group = group.sort_values("rho")
        previous = None
        for _, row in group.iterrows():
            policy = row.selected_y_json
            if policy != previous:
                rows.append({"budget_fraction": fraction, "rho": row.rho, "selected_y_json": policy, "change_type": "ambiguity_change_point"})
                previous = policy
    return pd.DataFrame(rows)


def make_figures(frontier: pd.DataFrame, output_dir: Path) -> None:
    for value, name, ylabel in (("objective", "robust_loss_vs_budget", "Robust loss"), ("margin_percent", "policy_margin_vs_budget", "First-versus-second margin (%)")):
        fig, ax = plt.subplots(figsize=(7, 4))
        for rho, group in frontier.groupby("rho"):
            ax.plot(group.budget_fraction, group[value], marker=".", label=f"rho={rho:g}")
        ax.axvline(0.42, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Road-budget fraction")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "figures" / f"{name}.png", dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for index in range(5):
        for rho, group in frontier.groupby("rho"):
            group = group.sort_values("budget_fraction")
            levels = [json.loads(value)[index] for value in group.selected_y_json]
            ax.plot(group.budget_fraction, levels, label=f"y{index+1}, rho={rho:g}", alpha=0.7)
    ax.set_xlabel("Road-budget fraction")
    ax.set_ylabel("Selected retrofit level")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "selected_road_levels_vs_budget.png", dpi=180)
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    base_output_dir = Path(args.base_output_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    for directory in (output_dir, output_dir / "tables", output_dir / "figures", output_dir / "checkpoints"):
        directory.mkdir(parents=True, exist_ok=True)
    rhos = tuple(float(value) for value in args.rhos.split(',') if value.strip())
    fractions = budget_fractions(args)
    if not (output_dir / "run_manifest.json").exists():
        write_run_metadata(output_dir, experiment="noto_road_budget_and_cost_frontier", parameters=vars(args), expected_work={"rhos": rhos, "tier_cube": 5 ** 5, "budget_fractions": fractions, "cost_perturbations": 5 * 4})
    all_rows: list[dict] = []
    frontier_rows: list[dict] = []
    perturbation_rows: list[dict] = []
    for rho in rhos:
        instance = build_m4_instance(base_output_dir, rho)
        existing_path = output_dir / "tables" / "table_noto_full_tier_cube.csv"
        if args.resume and existing_path.exists():
            prior = pd.read_csv(existing_path)
            prior = prior[np.isclose(prior.rho, rho)]
            rows = prior.to_dict("records") if len(prior) == 3125 else evaluate_cube(args, base_output_dir, output_dir, rho)
        else:
            rows = evaluate_cube(args, base_output_dir, output_dir, rho)
        all_rows = [row for row in all_rows if float(row["rho"]) != rho] + rows
        base_fraction = float(instance.budget_retrofit / instance.retrofit_costs.sum())
        frontier = summarize_budget(rows, instance, fractions, base_fraction)
        frontier_rows = [row for row in frontier_rows if float(row["rho"]) != rho] + frontier
        perturbation_rows.extend(cost_perturbations(rows, instance, instance.budget_retrofit))
        save_table(all_rows, output_dir / "tables" / "table_noto_full_tier_cube.csv", ["rho", "candidate_index"])
        save_table(frontier_rows, output_dir / "tables" / "table_noto_budget_frontier.csv", ["rho", "budget_fraction"])
        save_table(perturbation_rows, output_dir / "tables" / "table_noto_cost_perturbation.csv", ["rho", "perturbed_corridor_index", "cost_multiplier"])
        write_status(output_dir / "status.json", status="running", block="budget_frontier", completed_rho=rho, completed_cube_rows=len(all_rows), total_cube_rows=len(rhos) * 3125)
    frontier_frame = pd.DataFrame(frontier_rows)
    cube_frame = pd.DataFrame(all_rows)
    exact_rows = []
    for rho, group in cube_frame[cube_frame.status == "feasible"].groupby("rho"):
        total_cost = float(group.road_cost_base.max() / max(1e-12, group.road_cost_fraction_base.max()))
        for cost in sorted(group.road_cost_base.unique()):
            subset = group[group.road_cost_base <= cost + 1e-10].sort_values(["objective", "candidate_index"])
            if subset.empty:
                continue
            exact_rows.append({"rho": rho, "budget_fraction": float(cost / total_cost), "absolute_budget": float(cost), "selected_y_json": subset.iloc[0].y_json, "change_type": "tier_cost_breakpoint", "tier_cost": float(cost), "feasible_tier_count": int(len(subset))})
    breakpoints = pd.concat([change_points(frontier_frame), pd.DataFrame(exact_rows)], ignore_index=True, sort=False)
    save_table(breakpoints.to_dict("records"), output_dir / "tables" / "table_noto_budget_breakpoints.csv", ["rho", "budget_fraction"])
    summaries = []
    for rho, group in frontier_frame.groupby("rho"):
        local = group[(group.budget_fraction >= 0.37) & (group.budget_fraction <= 0.47)]
        remote = group[(group.budget_fraction < 0.37) | (group.budget_fraction > 0.47)]
        policy_count = local.selected_y_json.nunique()
        if policy_count == 1:
            classification = "current_budget_locally_forced"
        elif policy_count > 1:
            classification = "tradeoffs_emerge_under_nearby_budgets"
        else:
            classification = "tradeoffs_only_emerge_under_remote_budgets"
        current = group.iloc[np.argmin(np.abs(group.budget_fraction.to_numpy(dtype=float) - 0.42))]
        full_pair = group[(group.corridor_2_full) & (group.corridor_3_full)]
        first_full_pair = float(full_pair.budget_fraction.min()) if not full_pair.empty else None
        summaries.append({"rho": rho, "nearby_policy_count": policy_count, "nearby_distinct_policies_json": json.dumps(sorted(local.selected_y_json.unique().tolist())), "path_distinct_policy_count": int(group.selected_y_json.nunique()), "path_distinct_policies_json": json.dumps(sorted(group.selected_y_json.unique().tolist())), "remote_distinct_policy_count": int(remote.selected_y_json.nunique()), "current_budget_selected_y_json": current.selected_y_json, "first_full_pair_budget_fraction": first_full_pair, "current_budget_residual_split_two_corridors": bool(current.residual_split_two_corridors), "evidence_classification": classification, "corridors_2_3_full_in_nearby": bool(local.corridor_2_full.all() and local.corridor_3_full.all())})
    save_table(summaries, output_dir / "tables" / "table_noto_budget_policy_summary.csv", ["rho"])
    make_figures(frontier_frame, output_dir)
    write_status(output_dir / "status.json", status="completed", block="budget_frontier", cube_rows=len(all_rows), frontier_rows=len(frontier_rows), rhos=rhos)
    finish_run_metadata(output_dir, status="completed", runtime_seconds=time.perf_counter() - started, extra={"cube_rows": len(all_rows), "frontier_rows": len(frontier_rows), "rhos": rhos})


if __name__ == "__main__":
    main()
