from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ejor_dad.checkpoint import atomic_write_dataframe, atomic_write_text


EXPERIMENT_VERSION = "noto-practical-resilience-v1"
NEAR_OPTIMAL_THRESHOLDS = (0.01, 0.05, 0.10, 0.50)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    log_path = output_dir / "logs" / f"noto_practical_postprocess_{time.strftime('%Y%m%d_%H%M%S')}.log"
    write_status(output_dir, "running", "Loading completed practical-grid checkpoints.", log_path)
    started = time.time()
    try:
        design = load_json(output_dir / "run_design.json")
        require_completed_run(output_dir, args.allow_partial)
        rho_values = [float(value) for value in design["rho_values"]]
        source_summary = load_source_summary(output_dir)
        heuristic_table = load_optional_table(output_dir / "tables" / "table_noto_practical_heuristics.csv")
        run_experiment_version = str(design.get("experiment_version", EXPERIMENT_VERSION))
        by_rho = {
            rho: load_grid_payloads(output_dir / "checkpoints", rho, run_experiment_version)
            for rho in rho_values
        }
        summary, top_policies, discrete_portfolio, regime_diagnostics, objective_components = analyze_archive(
            by_rho=by_rho,
            rho_values=rho_values,
            density_cap=float(design["density_cap"]),
            source_summary=source_summary,
            heuristic_table=heuristic_table,
        )
        recommendation = recommend_follow_up(summary, discrete_portfolio, regime_diagnostics, design)
        write_tables(
            output_dir,
            summary,
            top_policies,
            discrete_portfolio,
            regime_diagnostics,
            objective_components,
        )
        write_decision_memo(
            output_dir,
            summary,
            discrete_portfolio,
            regime_diagnostics,
            objective_components,
            recommendation,
        )
        atomic_write_text(
            output_dir / "follow_up_recommendation.json",
            json.dumps(recommendation, indent=2, ensure_ascii=False),
        )
        runtime = time.time() - started
        atomic_write_text(output_dir / "postprocess_runtime.json", json.dumps({"runtime_sec": runtime}, indent=2))
        write_status(
            output_dir,
            "completed",
            "Post-run diagnostic completed; no follow-up optimization was launched automatically.",
            log_path,
            runtime_sec=runtime,
        )
    except Exception as exc:
        write_status(output_dir, "failed", str(exc), log_path)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a completed practical Noto grid archive, quantify ambiguity activity, "
            "and extract its exact discrete-project portfolio subset."
        )
    )
    parser.add_argument("--output-dir", default="data_work/noto/practical_resilience")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Analyze available checkpoints without requiring a completed all-rho run.",
    )
    return parser.parse_args()


def require_completed_run(output_dir: Path, allow_partial: bool) -> None:
    status_path = output_dir / "run_status.json"
    if not status_path.exists():
        raise FileNotFoundError(f"Missing practical-run status file: {status_path}")
    status = load_json(status_path)
    if status.get("status") == "completed" or allow_partial:
        return
    raise RuntimeError(
        "The practical Noto grid is not complete. The queued postprocessor must wait for all rho values; "
        "pass --allow-partial only for a provisional diagnostic."
    )


def load_source_summary(output_dir: Path) -> dict[float, dict[str, Any]]:
    table_path = output_dir / "tables" / "table_noto_practical_summary.csv"
    if not table_path.exists():
        return {}
    frame = pd.read_csv(table_path)
    return {float(row.rho): row._asdict() for row in frame.itertuples(index=False)}


def load_optional_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_grid_payloads(
    checkpoint_dir: Path,
    rho: float,
    run_experiment_version: str,
) -> list[dict[str, Any]]:
    rho_key = f"rho{rho:.2f}".replace(".", "p")
    payloads: list[dict[str, Any]] = []
    for path in checkpoint_dir.glob("*__grid_*.json"):
        if run_experiment_version not in path.name or rho_key not in path.name:
            continue
        payload = load_json(path)
        if "candidate_index" not in payload or "y" not in payload or "objective" not in payload:
            continue
        payload["_source_path"] = str(path)
        payloads.append(payload)
    payloads.sort(key=lambda row: int(row["candidate_index"]))
    if not payloads:
        raise RuntimeError(f"No grid checkpoints found for rho={rho:.2f}.")
    candidate_ids = [int(row["candidate_index"]) for row in payloads]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError(f"Duplicate grid checkpoint candidate indices detected for rho={rho:.2f}.")
    return payloads


def analyze_archive(
    by_rho: dict[float, list[dict[str, Any]]],
    rho_values: Iterable[float],
    density_cap: float,
    source_summary: dict[float, dict[str, Any]],
    heuristic_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered_rhos = list(sorted(rho_values))
    reference_rows = by_rho[ordered_rhos[0]]
    reference_best = min(reference_rows, key=lambda row: float(row["objective"]))
    reference_y = canonical_y(reference_best["y"])
    reference_binary_rows = [row for row in reference_rows if is_binary_policy(row["y"])]
    if not reference_binary_rows:
        raise RuntimeError("The rho=0 archive contains no binary project portfolios.")
    reference_binary_y = canonical_y(
        min(reference_binary_rows, key=lambda row: float(row["objective"]))["y"]
    )
    summary_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    discrete_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    objective_component_rows: list[dict[str, Any]] = []

    for rho in ordered_rhos:
        rows = sorted(by_rho[rho], key=lambda row: float(row["objective"]))
        best = rows[0]
        best_y = canonical_y(best["y"])
        reference_row = find_policy(rows, reference_y)
        if reference_row is None:
            raise RuntimeError(f"The rho=0 policy is missing from rho={rho:.2f}'s completed archive.")
        binary_rows = [row for row in rows if is_binary_policy(row["y"])]
        if not binary_rows:
            raise RuntimeError(f"No binary project portfolios found for rho={rho:.2f}.")
        binary_best = min(binary_rows, key=lambda row: float(row["objective"]))
        density = distribution_diagnostics(best, density_cap)
        source = source_summary.get(rho, {})
        heuristic = best_heuristic_row(heuristic_table, rho)
        best_objective = float(best["objective"])
        reference_objective = float(reference_row["objective"])
        binary_objective = float(binary_best["objective"])
        summary_rows.append(
            {
                "rho": rho,
                "grid_checkpoint_count": len(rows),
                "best_grid_objective": best_objective,
                "best_grid_y_json": json.dumps(best_y),
                "rho0_policy_objective": reference_objective,
                "delta_rho_value": reference_objective - best_objective,
                "delta_rho_percent": relative_percent(reference_objective - best_objective, best_objective),
                "policy_changed_from_rho0": best_y != reference_y,
                "y_l1_distance_from_rho0": l1_distance(best_y, reference_y),
                "binary_project_count": len(binary_rows),
                "best_binary_objective": binary_objective,
                "best_binary_y_json": json.dumps(canonical_y(binary_best["y"])),
                "binary_gap_to_grid": binary_objective - best_objective,
                "binary_gap_percent": relative_percent(binary_objective - best_objective, best_objective),
                "best_heuristic": heuristic.get("heuristic") if heuristic else None,
                "best_heuristic_objective": heuristic.get("objective") if heuristic else np.nan,
                "best_heuristic_gap_percent": (
                    relative_percent(float(heuristic["objective"]) - best_objective, best_objective)
                    if heuristic
                    else np.nan
                ),
                **density,
                "source_best_z_json": source.get("best_z_json"),
                "source_best_w_json": source.get("best_w_json"),
                "source_road_value_over_no_retrofit": source.get("road_value_over_no_retrofit"),
            }
        )
        objective_component_rows.append(objective_decomposition(best, rho))
        for rank, row in enumerate(rows[:10], start=1):
            top_rows.append(
                {
                    "rho": rho,
                    "rank": rank,
                    "objective": float(row["objective"]),
                    "gap_to_grid_best": float(row["objective"]) - best_objective,
                    "gap_percent": relative_percent(float(row["objective"]) - best_objective, best_objective),
                    "y_json": json.dumps(canonical_y(row["y"])),
                    "candidate_index": int(row["candidate_index"]),
                }
            )
        for threshold in NEAR_OPTIMAL_THRESHOLDS:
            near_count = sum(
                relative_percent(float(row["objective"]) - best_objective, best_objective) <= threshold + 1e-12
                for row in rows
            )
            discrete_rows.append(
                {
                    "rho": rho,
                    "record_type": "near_optimal_count",
                    "threshold_percent": threshold,
                    "policy_count": near_count,
                    "share_of_grid_policies": near_count / len(rows),
                    "objective": np.nan,
                    "y_json": None,
                }
            )
        for label, row in [
            ("binary_grid_best", binary_best),
            ("rho0_binary_policy", find_policy(binary_rows, reference_binary_y)),
        ]:
            if row is None:
                continue
            discrete_rows.append(
                {
                    "rho": rho,
                    "record_type": label,
                    "threshold_percent": np.nan,
                    "policy_count": np.nan,
                    "share_of_grid_policies": np.nan,
                    "objective": float(row["objective"]),
                    "y_json": json.dumps(canonical_y(row["y"])),
                }
            )
        regime_rows.append(
            {
                "rho": rho,
                "policy_y_json": json.dumps(best_y),
                "policy_changed_from_rho0": best_y != reference_y,
                "rho0_policy_gap_percent": relative_percent(reference_objective - best_objective, best_objective),
                **density,
            }
        )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(top_rows),
        pd.DataFrame(discrete_rows),
        pd.DataFrame(regime_rows),
        pd.DataFrame(objective_component_rows),
    )


def objective_decomposition(payload: dict[str, Any], rho: float) -> dict[str, Any]:
    nominal = np.asarray(payload["nominal_distribution"], dtype=float)
    worst_case = np.asarray(payload["worst_case_distribution"], dtype=float)
    losses = np.asarray(payload["state_losses"], dtype=float)
    survivors = np.asarray(payload["state_survivors"], dtype=float)
    nominal_expected_loss = float(nominal @ losses)
    robust_expected_loss = float(worst_case @ losses)
    common_loss_floor = float(losses.min())
    state_loss_range = float(losses.max() - common_loss_floor)
    moved_mass = 0.5 * float(np.abs(worst_case - nominal).sum())
    ambiguity_premium = robust_expected_loss - nominal_expected_loss
    modelled_loss_exposure = float(np.mean(losses + survivors))
    return {
        "rho": rho,
        "modelled_loss_exposure_after_renovation": modelled_loss_exposure,
        "minimum_state_loss_common_floor": common_loss_floor,
        "common_floor_share_percent": 100.0 * common_loss_floor / max(1.0, robust_expected_loss),
        "nominal_expected_loss": nominal_expected_loss,
        "nominal_state_dispersion_above_floor": nominal_expected_loss - common_loss_floor,
        "robust_expected_loss": robust_expected_loss,
        "ambiguity_premium": ambiguity_premium,
        "ambiguity_premium_percent": 100.0 * ambiguity_premium / max(1.0, robust_expected_loss),
        "state_loss_range": state_loss_range,
        "tv_moved_mass": moved_mass,
        "tv_range_upper_bound": moved_mass * state_loss_range,
        "fraction_of_tv_range_bound_used": (
            ambiguity_premium / max(1e-12, moved_mass * state_loss_range)
        ),
        "nominal_response_credits": modelled_loss_exposure - nominal_expected_loss,
        "worst_case_response_credits": modelled_loss_exposure - robust_expected_loss,
    }


def best_heuristic_row(table: pd.DataFrame, rho: float) -> dict[str, Any] | None:
    if table.empty or "rho" not in table.columns:
        return None
    subset = table[np.isclose(table["rho"].astype(float), rho)]
    if subset.empty:
        return None
    return subset.loc[subset["objective"].astype(float).idxmin()].to_dict()


def distribution_diagnostics(payload: dict[str, Any], density_cap: float) -> dict[str, Any]:
    nominal = np.asarray(payload["nominal_distribution"], dtype=float)
    worst_case = np.asarray(payload["worst_case_distribution"], dtype=float)
    shifts = worst_case - nominal
    ratios = np.divide(worst_case, nominal, out=np.zeros_like(worst_case), where=nominal > 1e-12)
    return {
        "tv_moved_mass": 0.5 * float(np.abs(shifts).sum()),
        "maximum_density_ratio": float(ratios.max()),
        "density_cap_saturated_state_count": int(
            np.sum((nominal > 1e-12) & (ratios >= density_cap - 1e-7))
        ),
        "positive_shift_state_count": int(np.sum(shifts > 1e-9)),
        "largest_probability_shift": float(shifts.max()),
        "largest_shift_state_index": int(np.argmax(shifts)),
    }


def recommend_follow_up(
    summary: pd.DataFrame,
    discrete_portfolio: pd.DataFrame,
    regime: pd.DataFrame,
    design: dict[str, Any],
) -> dict[str, Any]:
    changed = summary["policy_changed_from_rho0"].astype(bool)
    meaningful_change = changed & (summary["delta_rho_percent"].astype(float) >= 0.10)
    binary_gap = summary["binary_gap_percent"].astype(float)
    cap_saturated = regime["density_cap_saturated_state_count"].astype(int) > 0
    near_flat = discrete_portfolio[
        (discrete_portfolio["record_type"] == "near_optimal_count")
        & np.isclose(discrete_portfolio["threshold_percent"].astype(float), 0.10)
    ]
    high_flatness = bool((near_flat["share_of_grid_policies"].astype(float) >= 0.10).any())

    if bool(meaningful_change.any()):
        primary = (
            "Keep the two-channel practical specification and run targeted sensitivity analysis; "
            "the completed finite-grid evidence shows a meaningful ambiguity-sensitive policy change."
        )
    elif bool(changed.any()):
        primary = (
            "Report ambiguity-sensitive tie-breaking rather than a strong policy-reversal claim; "
            "the selected policy changes, but its value effect is below the declared 0.10% materiality screen."
        )
    elif high_flatness:
        primary = (
            "Do not launch a new model automatically. Report a flat near-optimal policy set and add a practitioner "
            "selection criterion only if it is substantively justified."
        )
    else:
        primary = (
            "Treat the selected policy as an empirically robust no-regret plan over the tested ambiguity radii; "
            "a different model is not required merely because the policy is stable."
        )

    actions = [primary]
    if bool((binary_gap <= 0.10).all()):
        actions.append(
            "The binary-project subset is within 0.10% of the five-level grid at every rho; formalizing a discrete "
            "project portfolio is an empirically supported simplification for a follow-up paper or robustness section."
        )
    else:
        actions.append(
            "Retain multi-level retrofit decisions in the primary empirical analysis because the binary-project subset "
            "has a material loss relative to the five-level grid for at least one rho."
        )
    if bool(cap_saturated.any()):
        actions.append(
            "Run a predeclared density-cap sensitivity before attributing stability or switching solely to rho, because "
            "the support cap is active in at least one selected solution."
        )
    else:
        actions.append(
            "The density cap is not saturated at the selected solutions; prioritize consequence heterogeneity and budget "
            "sensitivity over an immediate cap-sensitivity rerun."
        )
    return {
        "experiment_version": str(design.get("experiment_version", EXPERIMENT_VERSION)),
        "automatic_rerun_started": False,
        "primary_recommendation": primary,
        "actions": actions,
        "decision_rules": {
            "meaningful_policy_value_change_percent": 0.10,
            "binary_portfolio_gap_percent": 0.10,
            "flatness_share_at_0_10_percent": 0.10,
        },
        "design": {
            "density_cap": design.get("density_cap"),
            "residual_failure_ratio": design.get("residual_failure_ratio"),
            "failure_delay_reduction": design.get("failure_delay_reduction"),
            "time_sensitive_fraction": design.get("time_sensitive_fraction"),
            "immediate_loss_fraction": design.get("immediate_loss_fraction"),
            "capacity_throughput_per_bed": design.get("capacity_throughput_per_bed"),
            "response_threshold_minutes": design.get("response_threshold_minutes"),
        },
    }


def write_tables(
    output_dir: Path,
    summary: pd.DataFrame,
    top_policies: pd.DataFrame,
    discrete_portfolio: pd.DataFrame,
    regime: pd.DataFrame,
    objective_components: pd.DataFrame,
) -> None:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for stem, frame in {
        "table_noto_practical_postrun_summary": summary,
        "table_noto_practical_postrun_top10": top_policies,
        "table_noto_practical_discrete_portfolio": discrete_portfolio,
        "table_noto_practical_regime_diagnostics": regime,
        "table_noto_practical_objective_decomposition": objective_components,
    }.items():
        atomic_write_dataframe(frame, table_dir / f"{stem}.csv")
        atomic_write_dataframe(frame, table_dir / f"{stem}.tex", kind="latex", escape=True)


def write_decision_memo(
    output_dir: Path,
    summary: pd.DataFrame,
    discrete_portfolio: pd.DataFrame,
    regime: pd.DataFrame,
    objective_components: pd.DataFrame,
    recommendation: dict[str, Any],
) -> None:
    lines = [
        "# Noto Practical Post-Run Decision Memo",
        "",
        "## Recommendation",
        "",
        recommendation["primary_recommendation"],
        "",
        "## Follow-up Actions",
        "",
    ]
    lines.extend(f"- {action}" for action in recommendation["actions"])
    lines.extend(
        [
            "",
            "## Exact Grid Summary",
            "",
            "```text",
            summary.to_csv(index=False).strip(),
            "```",
            "",
            "## Regime Diagnostics",
            "",
            "```text",
            regime.to_csv(index=False).strip(),
            "```",
            "",
            "## Objective Decomposition",
            "",
            "```text",
            objective_components.to_csv(index=False).strip(),
            "```",
            "",
            "The binary-project table and all values are extracted from the completed five-level archive; this postprocessor does not rerun or alter the optimization model.",
        ]
    )
    atomic_write_text(output_dir / "noto_practical_decision_memo.md", "\n".join(lines) + "\n")


def find_policy(rows: Iterable[dict[str, Any]], target_y: tuple[float, ...]) -> dict[str, Any] | None:
    return next((row for row in rows if canonical_y(row["y"]) == target_y), None)


def canonical_y(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(float(round(float(value), 8)) for value in values)


def is_binary_policy(values: Iterable[float]) -> bool:
    return all(abs(value) <= 1e-8 or abs(value - 1.0) <= 1e-8 for value in canonical_y(values))


def l1_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return float(sum(abs(a - b) for a, b in zip(left, right)))


def relative_percent(value: float, baseline: float) -> float:
    return 100.0 * float(value) / max(1.0, abs(float(baseline)))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_status(
    output_dir: Path,
    status: str,
    message: str,
    log_path: Path,
    runtime_sec: float | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "message": message,
        "updated_at_epoch": time.time(),
        "log_path": str(log_path.resolve()),
    }
    if runtime_sec is not None:
        payload["runtime_sec"] = runtime_sec
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    atomic_write_text(output_dir / "postprocess_status.json", json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()