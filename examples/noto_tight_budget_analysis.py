from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ejor_dad.checkpoint import atomic_write_text

import noto_access_experiment as noto


DEFAULT_RHOS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
DEFAULT_BUDGET_MULTIPLIERS = [0.40, 0.50]
EXPERIMENT_VERSION = "noto-tight-budget-v1"


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    table_dir = output_dir / "tables"
    config_dir = output_dir / "configs"
    table_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    write_status(output_dir, "running", "Loading completed fixed-y checkpoints.")
    started = time.time()

    try:
        rho_values = parse_float_list(args.rho_values, minimum=0.0, maximum=1.0)
        if 0.0 not in rho_values:
            raise ValueError("rho=0 must be included to define the nominal reference policy.")
        budget_multipliers = parse_float_list(args.budget_multipliers, minimum=0.0, maximum=1.0)
        instance, _ = noto.build_noto_instance(0.0)
        grid = noto.GRID_LEVELS["full"]
        policies = enumerate_archived_policies(instance, grid)
        archive = load_archive(
            source_dir=source_dir,
            policies=policies,
            rho_values=rho_values,
            density_cap=args.density_cap,
        )
        critical_indices = identify_critical_links(archive[0.0])

        requested_summary, requested_top, requested_near = analyze_requested_budgets(
            instance=instance,
            archive=archive,
            rho_values=rho_values,
            budget_multipliers=budget_multipliers,
            critical_indices=critical_indices,
            density_cap=args.density_cap,
        )
        frontier, phases = analyze_budget_frontier(
            instance=instance,
            archive=archive,
            rho_values=rho_values,
            critical_indices=critical_indices,
        )
        active_case = select_mechanism_active_case(frontier, rho_values)

        noto.write_table(requested_summary, output_dir, "table_noto_tight_budget_requested")
        noto.write_table(requested_top, output_dir, "table_noto_tight_budget_top10")
        noto.write_table(requested_near, output_dir, "table_noto_tight_budget_near_optimal")
        noto.write_table(frontier, output_dir, "table_noto_budget_frontier")
        noto.write_table(phases, output_dir, "table_noto_budget_policy_phases")
        noto.write_table(active_case, output_dir, "table_noto_budget_mechanism_active_case")

        config = {
            "experiment_version": EXPERIMENT_VERSION,
            "source_checkpoint_directory": str(source_dir.resolve()),
            "density_cap": args.density_cap,
            "rho_values": rho_values,
            "requested_budget_multipliers": budget_multipliers,
            "baseline_B_Y": instance.budget_retrofit,
            "retrofit_costs": instance.retrofit_costs.tolist(),
            "link_ids": [link.id for link in instance.links],
            "critical_link_indices_zero_based": critical_indices,
            "critical_link_ids": [instance.links[index].id for index in critical_indices],
            "grid_levels": grid.tolist(),
            "archived_feasible_policies_per_rho": len(policies),
            "reuse_justification": (
                "B_Y is a separate road-feasibility constraint. Tightening it only removes y vectors; "
                "the exact fixed-y objective, z, w, nominal distribution, and worst-case distribution "
                "for every retained vector are unchanged."
            ),
            "mechanism_active_case_selection": (
                "Ex post diagnostic selected by the largest Delta_rho over the exhaustive archived "
                "budget-threshold frontier; it is not a recalibrated baseline."
            ),
            "runtime_sec": time.time() - started,
        }
        atomic_write_text(config_dir / "noto_tight_budget_design.json", json.dumps(config, indent=2))
        write_status(
            output_dir,
            "completed",
            "Tight-budget and exhaustive budget-frontier analyses completed from exact checkpoints.",
            extra={
                "runtime_sec": time.time() - started,
                "requested_rows": len(requested_summary),
                "frontier_rows": len(frontier),
                "phase_rows": len(phases),
            },
        )
    except Exception as exc:
        write_status(output_dir, "failed", str(exc))
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reanalyze exact capped-TV Noto checkpoints under tighter road budgets."
    )
    parser.add_argument(
        "--source-dir",
        default="data_work/noto/support_preserving_full/checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        default="data_work/noto/support_preserving_tight_budget",
    )
    parser.add_argument(
        "--budget-multipliers",
        default=",".join(str(value) for value in DEFAULT_BUDGET_MULTIPLIERS),
    )
    parser.add_argument(
        "--rho-values",
        default=",".join(str(value) for value in DEFAULT_RHOS),
    )
    parser.add_argument("--density-cap", type=float, default=2.0)
    args = parser.parse_args()
    if args.density_cap < 1.0:
        parser.error("--density-cap must be at least 1.")
    return args


def parse_float_list(value: str, minimum: float, maximum: float) -> list[float]:
    values = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not values:
        raise ValueError("Expected a nonempty comma-separated numeric list.")
    if any(item < minimum or item > maximum for item in values):
        raise ValueError(f"Values must lie in [{minimum}, {maximum}].")
    return values


def enumerate_archived_policies(instance: Any, grid: np.ndarray) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for candidate_index, values in enumerate(product(grid, repeat=len(instance.links)), start=1):
        y = np.asarray(values, dtype=float)
        budget_used = float(np.dot(instance.retrofit_costs, y))
        if budget_used > instance.budget_retrofit + 1e-9:
            continue
        policies.append(
            {
                "candidate_index": candidate_index,
                "y": tuple(float(value) for value in y),
                "budget_used": budget_used,
                "hash": noto.hash_array(y),
            }
        )
    return policies


def load_archive(
    source_dir: Path,
    policies: list[dict[str, Any]],
    rho_values: list[float],
    density_cap: float,
) -> dict[float, list[dict[str, Any]]]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {source_dir}")
    archive: dict[float, list[dict[str, Any]]] = {}
    cap_label = f"{density_cap:.2f}".replace(".", "p")
    for rho in rho_values:
        experiment_id = f"noto_support_m5_full_k{cap_label}_rho{rho:.2f}"
        rows: list[dict[str, Any]] = []
        missing: list[str] = []
        for policy in policies:
            key = noto.versioned_key(
                f"{experiment_id}__grid_{policy['candidate_index']:04d}_{policy['hash']}"
            )
            checkpoint_path = source_dir / f"{key}.json"
            if not checkpoint_path.exists():
                missing.append(str(checkpoint_path))
                continue
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            stored_y = tuple(float(value) for value in payload["y"])
            if not np.allclose(stored_y, policy["y"], atol=1e-12, rtol=0.0):
                raise ValueError(f"Checkpoint policy mismatch: {checkpoint_path}")
            rows.append(
                policy
                | {
                    "objective": float(payload["objective"]),
                    "lower_bound": float(payload["lower_bound"]),
                    "z": tuple(float(value) for value in payload["z"]),
                    "w": tuple(float(value) for value in payload["w"]),
                    "nominal_distribution": np.asarray(payload["nominal_distribution"], dtype=float),
                    "worst_case_distribution": np.asarray(
                        payload["worst_case_distribution"], dtype=float
                    ),
                    "state_losses": np.asarray(payload["state_losses"], dtype=float),
                    "iterations": int(payload["iterations"]),
                    "eval_runtime_sec": float(payload.get("eval_runtime_sec", np.nan)),
                }
            )
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} required checkpoints for rho={rho:.2f}; first: {missing[0]}"
            )
        archive[rho] = rows
    return archive


def identify_critical_links(rho_zero_rows: list[dict[str, Any]]) -> list[int]:
    best = min(rho_zero_rows, key=lambda row: row["objective"])
    fully_hardened = [index for index, value in enumerate(best["y"]) if value >= 1.0 - 1e-9]
    if len(fully_hardened) != 2:
        raise ValueError(
            "Expected the baseline nominal grid optimum to identify exactly two fully hardened critical links."
        )
    return fully_hardened


def analyze_requested_budgets(
    instance: Any,
    archive: dict[float, list[dict[str, Any]]],
    rho_values: list[float],
    budget_multipliers: list[float],
    critical_indices: list[int],
    density_cap: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    near_rows: list[dict[str, Any]] = []
    for multiplier in budget_multipliers:
        budget = multiplier * instance.budget_retrofit
        feasible_by_rho = {
            rho: sorted(
                (row for row in archive[rho] if row["budget_used"] <= budget + 1e-9),
                key=lambda row: row["objective"],
            )
            for rho in rho_values
        }
        if not feasible_by_rho[0.0]:
            raise RuntimeError(f"No feasible grid policy for B_Y multiplier {multiplier}.")
        reference_y = feasible_by_rho[0.0][0]["y"]
        for rho in rho_values:
            feasible = feasible_by_rho[rho]
            best = feasible[0]
            reference = find_policy(feasible, reference_y)
            diagnostics = policy_diagnostics(instance, best, critical_indices)
            reference_diagnostics = prefixed(
                policy_diagnostics(instance, reference, critical_indices),
                "rho0_policy_",
            )
            summary_rows.append(
                {
                    "budget_multiplier": multiplier,
                    "B_Y": budget,
                    "rho": rho,
                    "density_cap": density_cap,
                    "total_grid_candidates": 3125,
                    "feasible_grid_candidates": len(feasible),
                    "best_objective": best["objective"],
                    "best_y_json": json.dumps(best["y"]),
                    "best_z_json": json.dumps(best["z"]),
                    "best_w_json": json.dumps(best["w"]),
                    "budget_used": best["budget_used"],
                    "budget_utilization_percent": 100.0 * best["budget_used"] / budget,
                    "rho0_policy_y_json": json.dumps(reference_y),
                    "objective_using_rho0_policy": reference["objective"],
                    "delta_rho_value": reference["objective"] - best["objective"],
                    "policy_changed_from_rho0": best["y"] != reference_y,
                    "second_best_gap": feasible[1]["objective"] - best["objective"]
                    if len(feasible) > 1
                    else np.nan,
                }
                | diagnostics
                | reference_diagnostics
            )
            for rank, row in enumerate(feasible[:10], start=1):
                top_rows.append(
                    {
                        "budget_multiplier": multiplier,
                        "B_Y": budget,
                        "rho": rho,
                        "rank": rank,
                        "objective": row["objective"],
                        "gap_to_best": row["objective"] - best["objective"],
                        "gap_percent": 100.0
                        * (row["objective"] - best["objective"])
                        / max(1.0, abs(best["objective"])),
                        "selected_y_json": json.dumps(row["y"]),
                        "budget_used": row["budget_used"],
                    }
                )
            for threshold in [0.01, 0.05, 0.10, 0.50]:
                count = sum(
                    100.0 * (row["objective"] - best["objective"])
                    / max(1.0, abs(best["objective"]))
                    <= threshold + 1e-12
                    for row in feasible
                )
                near_rows.append(
                    {
                        "budget_multiplier": multiplier,
                        "B_Y": budget,
                        "rho": rho,
                        "threshold_percent": threshold,
                        "near_optimal_policy_count": count,
                        "feasible_grid_candidates": len(feasible),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(top_rows), pd.DataFrame(near_rows)


def analyze_budget_frontier(
    instance: Any,
    archive: dict[float, list[dict[str, Any]]],
    rho_values: list[float],
    critical_indices: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = sorted({round(row["budget_used"], 12) for row in archive[0.0]})
    frontier_rows: list[dict[str, Any]] = []
    signatures: list[tuple[tuple[float, ...], ...]] = []
    threshold_payloads: list[dict[str, Any]] = []
    for threshold in thresholds:
        feasible_by_rho = {
            rho: [row for row in archive[rho] if row["budget_used"] <= threshold + 1e-9]
            for rho in rho_values
        }
        best_by_rho = {
            rho: min(feasible_by_rho[rho], key=lambda row: row["objective"]) for rho in rho_values
        }
        reference_y = best_by_rho[0.0]["y"]
        signature = tuple(best_by_rho[rho]["y"] for rho in rho_values)
        signatures.append(signature)
        deltas: list[float] = []
        for rho in rho_values:
            best = best_by_rho[rho]
            reference = find_policy(feasible_by_rho[rho], reference_y)
            delta = reference["objective"] - best["objective"]
            deltas.append(delta)
            frontier_rows.append(
                {
                    "B_Y": threshold,
                    "budget_multiplier": threshold / instance.budget_retrofit,
                    "rho": rho,
                    "feasible_grid_candidates": len(feasible_by_rho[rho]),
                    "best_objective": best["objective"],
                    "best_y_json": json.dumps(best["y"]),
                    "rho0_policy_y_json": json.dumps(reference_y),
                    "objective_using_rho0_policy": reference["objective"],
                    "delta_rho_value": delta,
                    "policy_changed_from_rho0": best["y"] != reference_y,
                }
                | policy_diagnostics(instance, best, critical_indices)
                | prefixed(
                    policy_diagnostics(instance, reference, critical_indices),
                    "rho0_policy_",
                )
            )
        threshold_payloads.append(
            {
                "reference_y": reference_y,
                "best_by_rho": best_by_rho,
                "deltas": deltas,
            }
        )

    phase_rows: list[dict[str, Any]] = []
    phase_start = 0
    for index in range(1, len(thresholds) + 1):
        if index < len(thresholds) and signatures[index] == signatures[phase_start]:
            continue
        payload = threshold_payloads[phase_start]
        phase_upper = (
            thresholds[index]
            if index < len(thresholds)
            else instance.budget_retrofit + 1e-12
        )
        changed_rhos = [
            rho
            for rho, best_y in zip(rho_values, signatures[phase_start])
            if best_y != payload["reference_y"]
        ]
        max_delta_index = int(np.argmax(payload["deltas"]))
        max_delta_rho = rho_values[max_delta_index]
        phase_rows.append(
            {
                "B_Y_lower_inclusive": thresholds[phase_start],
                "B_Y_upper_exclusive": phase_upper,
                "budget_multiplier_lower_inclusive": thresholds[phase_start]
                / instance.budget_retrofit,
                "budget_multiplier_upper_exclusive": phase_upper / instance.budget_retrofit,
                "nominal_y_json": json.dumps(payload["reference_y"]),
                "policy_by_rho_json": json.dumps(
                    {
                        f"{rho:.2f}": list(payload["best_by_rho"][rho]["y"])
                        for rho in rho_values
                    }
                ),
                "dda_policy_active": bool(changed_rhos),
                "first_policy_change_rho": min(changed_rhos) if changed_rhos else np.nan,
                "max_delta_rho_value": max(payload["deltas"]),
                "max_delta_rho_percent": 100.0
                * max(payload["deltas"])
                / max(
                    1.0,
                    abs(payload["best_by_rho"][max_delta_rho]["objective"]),
                ),
            }
        )
        phase_start = index
    return pd.DataFrame(frontier_rows), pd.DataFrame(phase_rows)


def select_mechanism_active_case(frontier: pd.DataFrame, rho_values: list[float]) -> pd.DataFrame:
    active = frontier[frontier["policy_changed_from_rho0"]].copy()
    if active.empty:
        return pd.DataFrame(
            columns=list(frontier.columns)
            + ["selection_rule", "selected_budget_multiplier"]
        )
    selected_index = active["delta_rho_value"].idxmax()
    selected_budget = float(active.loc[selected_index, "B_Y"])
    rows = frontier[np.isclose(frontier["B_Y"], selected_budget, atol=1e-10)].copy()
    rows = rows[rows["rho"].isin(rho_values)].copy()
    rows["selection_rule"] = (
        "Ex post maximum Delta_rho over exhaustive archived budget thresholds; sensitivity only"
    )
    rows["selected_budget_multiplier"] = float(rows.iloc[0]["budget_multiplier"])
    return rows


def find_policy(rows: list[dict[str, Any]], y: tuple[float, ...]) -> dict[str, Any]:
    for row in rows:
        if row["y"] == y:
            return row
    raise KeyError(f"Policy is absent from the feasible archive: {y}")


def policy_diagnostics(
    instance: Any,
    row: dict[str, Any],
    critical_indices: list[int],
) -> dict[str, Any]:
    nominal = row["nominal_distribution"]
    worst_case = row["worst_case_distribution"]
    positive_nominal = nominal > 1e-12
    zero_nominal = ~positive_nominal
    positive_shift = np.maximum(worst_case - nominal, 0.0)
    critical_ids = {instance.links[index].id for index in critical_indices}
    joint_failure_indices = [
        state_index
        for state_index, state in enumerate(instance.states)
        if critical_ids.issubset(set(state.failed_links))
    ]
    return {
        "mass_added_to_zero_nominal_states": float(positive_shift[zero_nominal].sum()),
        "max_realized_density_ratio": float(
            np.max(worst_case[positive_nominal] / nominal[positive_nominal])
        ),
        "both_critical_failure_nominal_probability": float(
            nominal[joint_failure_indices].sum()
        ),
        "both_critical_failure_worst_case_probability": float(
            worst_case[joint_failure_indices].sum()
        ),
        "both_critical_failure_probability_shift": float(
            (worst_case[joint_failure_indices] - nominal[joint_failure_indices]).sum()
        ),
    }


def prefixed(values: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def write_status(
    output_dir: Path,
    status: str,
    message: str,
    extra: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "message": message,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if extra:
        payload.update(extra)
    atomic_write_text(output_dir / "run_status.json", json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
