from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ejor_dad.checkpoint import atomic_write_text

import noto_access_experiment as noto
import noto_tight_budget_analysis as tight


DEFAULT_RHOS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
DEFAULT_COST_MULTIPLIERS = [0.80, 0.90, 1.00, 1.10, 1.20]
EXPERIMENT_VERSION = "noto-mechanism-robustness-v1"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    config_dir = output_dir / "configs"
    for path in [table_dir, figure_dir, config_dir]:
        path.mkdir(parents=True, exist_ok=True)
    write_status(output_dir, "running", "Loading exact capped-TV fixed-y archive.")
    started = time.time()

    try:
        rho_values = tight.parse_float_list(args.rho_values, 0.0, 1.0)
        cost_multipliers = tight.parse_float_list(args.cost_multipliers, 0.01, 10.0)
        instance, _ = noto.build_noto_instance(0.0)
        policies = tight.enumerate_archived_policies(instance, noto.GRID_LEVELS["full"])
        archive = tight.load_archive(
            Path(args.source_dir),
            policies,
            rho_values,
            args.density_cap,
        )
        critical_indices = tight.identify_critical_links(archive[0.0])
        active_budget = args.active_budget_multiplier * instance.budget_retrofit

        state_shifts, critical_patterns = build_distribution_diagnostics(
            instance,
            archive,
            rho_values,
            active_budget,
            args.paper_rho,
            critical_indices,
        )
        cost_surface, cost_summary, exact_phases = build_cost_budget_sensitivity(
            instance,
            archive,
            rho_values,
            cost_multipliers,
            args.budget_min_multiplier,
            args.budget_max_multiplier,
            args.budget_step,
            critical_indices,
        )

        noto.write_table(state_shifts, output_dir, "table_noto_active_state_probability_shifts")
        noto.write_table(critical_patterns, output_dir, "table_noto_active_critical_failure_patterns")
        noto.write_table(cost_surface, output_dir, "table_noto_cost_budget_surface")
        noto.write_table(cost_summary, output_dir, "table_noto_cost_budget_summary")
        noto.write_table(exact_phases, output_dir, "table_noto_cost_budget_exact_phases")

        plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.22})
        make_distribution_figure(critical_patterns, figure_dir, args.paper_rho)
        make_state_shift_figure(state_shifts, figure_dir, args.paper_rho)
        make_cost_budget_figure(cost_summary, cost_multipliers, figure_dir)
        make_baseline_frontier_figure(cost_surface, rho_values, figure_dir)

        config = {
            "experiment_version": EXPERIMENT_VERSION,
            "source_checkpoint_directory": str(Path(args.source_dir).resolve()),
            "density_cap": args.density_cap,
            "rho_values": rho_values,
            "paper_rho": args.paper_rho,
            "active_budget_multiplier": args.active_budget_multiplier,
            "active_budget": active_budget,
            "cost_multipliers": cost_multipliers,
            "budget_multiplier_range": [
                args.budget_min_multiplier,
                args.budget_max_multiplier,
            ],
            "budget_step": args.budget_step,
            "budget_normalization": "absolute B_Y divided by the original calibrated B_Y",
            "critical_link_indices_zero_based": critical_indices,
            "critical_link_ids": [instance.links[index].id for index in critical_indices],
            "cost_sensitivity_interpretation": (
                "Only retrofit-feasibility costs change. Exact fixed-y objectives and probability "
                "distributions are reused without approximation."
            ),
            "runtime_sec": time.time() - started,
        }
        atomic_write_text(
            config_dir / "noto_mechanism_robustness_design.json",
            json.dumps(config, indent=2),
        )
        write_status(
            output_dir,
            "completed",
            "Distribution and cost-budget mechanism diagnostics completed.",
            {
                "runtime_sec": time.time() - started,
                "state_shift_rows": len(state_shifts),
                "cost_surface_rows": len(cost_surface),
                "exact_phase_rows": len(exact_phases),
            },
        )
    except Exception as exc:
        write_status(output_dir, "failed", str(exc))
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Noto worst-case distribution and cost-budget mechanism diagnostics."
    )
    parser.add_argument(
        "--source-dir",
        default="data_work/noto/support_preserving_full/checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        default="data_work/noto/support_preserving_mechanism_analysis",
    )
    parser.add_argument("--rho-values", default=",".join(str(value) for value in DEFAULT_RHOS))
    parser.add_argument(
        "--cost-multipliers",
        default=",".join(str(value) for value in DEFAULT_COST_MULTIPLIERS),
    )
    parser.add_argument("--density-cap", type=float, default=2.0)
    parser.add_argument("--paper-rho", type=float, default=0.25)
    parser.add_argument("--active-budget-multiplier", type=float, default=0.4694201989666984)
    parser.add_argument("--budget-min-multiplier", type=float, default=0.30)
    parser.add_argument("--budget-max-multiplier", type=float, default=0.75)
    parser.add_argument("--budget-step", type=float, default=0.005)
    args = parser.parse_args()
    if args.density_cap < 1.0:
        parser.error("--density-cap must be at least 1.")
    if not 0.0 <= args.paper_rho <= 1.0:
        parser.error("--paper-rho must lie in [0, 1].")
    if not 0.0 < args.active_budget_multiplier <= 1.0:
        parser.error("--active-budget-multiplier must lie in (0, 1].")
    if not 0.0 <= args.budget_min_multiplier < args.budget_max_multiplier <= 1.0:
        parser.error("The budget multiplier range must be ordered inside [0, 1].")
    if args.budget_step <= 0.0:
        parser.error("--budget-step must be positive.")
    return args


def build_distribution_diagnostics(
    instance: Any,
    archive: dict[float, list[dict[str, Any]]],
    rho_values: list[float],
    active_budget: float,
    paper_rho: float,
    critical_indices: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if paper_rho not in rho_values:
        raise ValueError("The paper rho must be included in the archived rho sweep.")
    feasible_zero = [row for row in archive[0.0] if row["budget_used"] <= active_budget + 1e-9]
    feasible_robust = [row for row in archive[paper_rho] if row["budget_used"] <= active_budget + 1e-9]
    nominal_best = min(feasible_zero, key=lambda row: row["objective"])
    nominal_policy_at_rho = tight.find_policy(feasible_robust, nominal_best["y"])
    robust_best = min(feasible_robust, key=lambda row: row["objective"])
    evaluations = [
        ("nominal policy", nominal_policy_at_rho),
        ("ambiguity-dependent policy", robust_best),
    ]
    link_labels = {link.id: f"L{index + 1}" for index, link in enumerate(instance.links)}
    critical_ids = {instance.links[index].id for index in critical_indices}
    state_rows: list[dict[str, Any]] = []
    pattern_rows: list[dict[str, Any]] = []
    for policy_label, row in evaluations:
        for state_index, state in enumerate(instance.states):
            nominal_probability = float(row["nominal_distribution"][state_index])
            worst_case_probability = float(row["worst_case_distribution"][state_index])
            failed = set(state.failed_links)
            state_rows.append(
                {
                    "rho": paper_rho,
                    "B_Y": active_budget,
                    "budget_multiplier": active_budget / instance.budget_retrofit,
                    "policy_label": policy_label,
                    "selected_y_json": json.dumps(row["y"]),
                    "state_id": state.id,
                    "failed_links_json": json.dumps(list(state.failed_links)),
                    "failed_link_labels": ", ".join(link_labels[item] for item in state.failed_links)
                    if state.failed_links
                    else "intact",
                    "state_loss": float(row["state_losses"][state_index]),
                    "nominal_probability": nominal_probability,
                    "worst_case_probability": worst_case_probability,
                    "probability_shift": worst_case_probability - nominal_probability,
                    "density_ratio": worst_case_probability / nominal_probability
                    if nominal_probability > 1e-12
                    else np.nan,
                    "both_critical_links_failed": critical_ids.issubset(failed),
                }
            )
        for pattern, required_failed, required_survived in critical_failure_patterns(
            instance,
            critical_indices,
        ):
            selected_states = [
                state_index
                for state_index, state in enumerate(instance.states)
                if required_failed.issubset(set(state.failed_links))
                and required_survived.isdisjoint(set(state.failed_links))
            ]
            for distribution_label, probabilities in [
                ("nominal", row["nominal_distribution"]),
                ("worst case", row["worst_case_distribution"]),
            ]:
                pattern_rows.append(
                    {
                        "rho": paper_rho,
                        "B_Y": active_budget,
                        "policy_label": policy_label,
                        "selected_y_json": json.dumps(row["y"]),
                        "distribution": distribution_label,
                        "critical_failure_pattern": pattern,
                        "probability": float(np.asarray(probabilities)[selected_states].sum()),
                    }
                )
    state_frame = pd.DataFrame(state_rows).sort_values(
        ["policy_label", "probability_shift"],
        ascending=[True, False],
    )
    return state_frame, pd.DataFrame(pattern_rows)


def critical_failure_patterns(
    instance: Any,
    critical_indices: list[int],
) -> list[tuple[str, set[str], set[str]]]:
    first = instance.links[critical_indices[0]].id
    second = instance.links[critical_indices[1]].id
    return [
        ("Neither critical link fails", set(), {first, second}),
        ("L2 only", {first}, {second}),
        ("L3 only", {second}, {first}),
        ("Both critical links fail", {first, second}, set()),
    ]


def build_cost_budget_sensitivity(
    instance: Any,
    archive: dict[float, list[dict[str, Any]]],
    rho_values: list[float],
    cost_multipliers: list[float],
    budget_min_multiplier: float,
    budget_max_multiplier: float,
    budget_step: float,
    critical_indices: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy_rows = archive[0.0]
    y_matrix = np.asarray([row["y"] for row in policy_rows], dtype=float)
    archive_lookup = {
        rho: {row["y"]: row for row in archive[rho]}
        for rho in rho_values
    }
    objective_matrix = np.column_stack(
        [
            np.asarray(
                [archive_lookup[rho][row["y"]]["objective"] for row in policy_rows]
            )
            for rho in rho_values
        ]
    )
    budgets = np.arange(
        budget_min_multiplier,
        budget_max_multiplier + 0.5 * budget_step,
        budget_step,
    )
    surface_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    for first_multiplier in cost_multipliers:
        for second_multiplier in cost_multipliers:
            modified_costs = np.asarray(instance.retrofit_costs, dtype=float).copy()
            modified_costs[critical_indices[0]] *= first_multiplier
            modified_costs[critical_indices[1]] *= second_multiplier
            policy_costs = y_matrix @ modified_costs
            phases = exact_policy_phases(
                instance,
                policy_rows,
                objective_matrix,
                policy_costs,
                rho_values,
                budget_min_multiplier,
                budget_max_multiplier,
                first_multiplier,
                second_multiplier,
            )
            phase_rows.extend(phases)
            active_phases = [phase for phase in phases if phase["dda_policy_active"]]
            active_width = sum(
                phase["budget_multiplier_upper_exclusive"]
                - phase["budget_multiplier_lower_inclusive"]
                for phase in active_phases
            )
            summary_rows.append(
                {
                    "L2_cost_multiplier": first_multiplier,
                    "L3_cost_multiplier": second_multiplier,
                    "num_policy_phases": len(phases),
                    "num_dda_active_phases": len(active_phases),
                    "active_budget_width": active_width,
                    "active_budget_share_percent": 100.0
                    * active_width
                    / (budget_max_multiplier - budget_min_multiplier),
                    "max_delta_rho_value": max(
                        (phase["max_delta_rho_value"] for phase in phases),
                        default=0.0,
                    ),
                    "max_delta_rho_percent": max(
                        (phase["max_delta_rho_percent"] for phase in phases),
                        default=0.0,
                    ),
                    "active_budget_min": min(
                        (phase["budget_multiplier_lower_inclusive"] for phase in active_phases),
                        default=np.nan,
                    ),
                    "active_budget_max": max(
                        (phase["budget_multiplier_upper_exclusive"] for phase in active_phases),
                        default=np.nan,
                    ),
                }
            )
            for budget_multiplier in budgets:
                budget = budget_multiplier * instance.budget_retrofit
                feasible = policy_costs <= budget + 1e-9
                if not feasible.any():
                    continue
                masked = np.where(feasible[:, None], objective_matrix, np.inf)
                best_indices = np.argmin(masked, axis=0)
                reference_index = int(best_indices[0])
                deltas = objective_matrix[reference_index, :] - objective_matrix[
                    best_indices,
                    np.arange(len(rho_values)),
                ]
                changed = [
                    policy_rows[int(best_index)]["y"] != policy_rows[reference_index]["y"]
                    for best_index in best_indices
                ]
                surface_rows.append(
                    {
                        "L2_cost_multiplier": first_multiplier,
                        "L3_cost_multiplier": second_multiplier,
                        "budget_multiplier": budget_multiplier,
                        "B_Y": budget,
                        "num_feasible_grid_policies": int(feasible.sum()),
                        "dda_policy_active": any(changed[1:]),
                        "first_policy_change_rho": min(
                            (rho for rho, is_changed in zip(rho_values, changed) if is_changed),
                            default=np.nan,
                        ),
                        "max_delta_rho_value": float(np.max(deltas)),
                        "delta_at_rho_0p25": float(deltas[-1]),
                        "delta_by_rho_json": json.dumps(
                            {f"{rho:.2f}": float(delta) for rho, delta in zip(rho_values, deltas)}
                        ),
                        "nominal_y_json": json.dumps(policy_rows[reference_index]["y"]),
                        "policy_by_rho_json": json.dumps(
                            {
                                f"{rho:.2f}": list(policy_rows[int(index)]["y"])
                                for rho, index in zip(rho_values, best_indices)
                            }
                        ),
                        "policy_at_rho_0p25_json": json.dumps(
                            policy_rows[int(best_indices[-1])]["y"]
                        ),
                    }
                )
    return pd.DataFrame(surface_rows), pd.DataFrame(summary_rows), pd.DataFrame(phase_rows)


def exact_policy_phases(
    instance: Any,
    policy_rows: list[dict[str, Any]],
    objective_matrix: np.ndarray,
    policy_costs: np.ndarray,
    rho_values: list[float],
    lower_multiplier: float,
    upper_multiplier: float,
    first_cost_multiplier: float,
    second_cost_multiplier: float,
) -> list[dict[str, Any]]:
    lower_budget = lower_multiplier * instance.budget_retrofit
    upper_budget = upper_multiplier * instance.budget_retrofit
    order = np.argsort(policy_costs, kind="stable")
    rounded_costs = np.round(policy_costs[order], 12)
    best_indices = np.full(len(rho_values), -1, dtype=int)
    best_objectives = np.full(len(rho_values), np.inf)
    cursor = 0

    def add_group(group_indices: np.ndarray) -> None:
        nonlocal best_indices, best_objectives
        for policy_index in group_indices:
            values = objective_matrix[int(policy_index), :]
            improved = values < best_objectives - 1e-12
            best_objectives[improved] = values[improved]
            best_indices[improved] = int(policy_index)

    while cursor < len(order) and policy_costs[order[cursor]] <= lower_budget + 1e-9:
        group_cost = rounded_costs[cursor]
        end = cursor + 1
        while end < len(order) and rounded_costs[end] == group_cost:
            end += 1
        add_group(order[cursor:end])
        cursor = end
    if np.any(best_indices < 0):
        raise RuntimeError("The lower cost-budget boundary has no feasible policy.")

    phase_rows: list[dict[str, Any]] = []
    phase_start = lower_multiplier
    phase_indices = best_indices.copy()
    while cursor < len(order) and policy_costs[order[cursor]] < upper_budget - 1e-9:
        group_cost = rounded_costs[cursor]
        end = cursor + 1
        while end < len(order) and rounded_costs[end] == group_cost:
            end += 1
        add_group(order[cursor:end])
        boundary = float(policy_costs[order[cursor]] / instance.budget_retrofit)
        if tuple(best_indices) != tuple(phase_indices):
            phase_rows.append(
                phase_payload(
                    policy_rows,
                    objective_matrix,
                    rho_values,
                    phase_start,
                    boundary,
                    phase_indices,
                    first_cost_multiplier,
                    second_cost_multiplier,
                )
            )
            phase_start = boundary
            phase_indices = best_indices.copy()
        cursor = end
    phase_rows.append(
        phase_payload(
            policy_rows,
            objective_matrix,
            rho_values,
            phase_start,
            upper_multiplier,
            phase_indices,
            first_cost_multiplier,
            second_cost_multiplier,
        )
    )
    return phase_rows


def phase_payload(
    policy_rows: list[dict[str, Any]],
    objective_matrix: np.ndarray,
    rho_values: list[float],
    lower: float,
    upper: float,
    best_indices: np.ndarray,
    first_cost_multiplier: float,
    second_cost_multiplier: float,
) -> dict[str, Any]:
    reference_index = int(best_indices[0])
    deltas = objective_matrix[reference_index, :] - objective_matrix[
        best_indices,
        np.arange(len(rho_values)),
    ]
    changed = [
        policy_rows[int(index)]["y"] != policy_rows[reference_index]["y"]
        for index in best_indices
    ]
    max_index = int(np.argmax(deltas))
    return {
        "L2_cost_multiplier": first_cost_multiplier,
        "L3_cost_multiplier": second_cost_multiplier,
        "budget_multiplier_lower_inclusive": lower,
        "budget_multiplier_upper_exclusive": upper,
        "nominal_y_json": json.dumps(policy_rows[reference_index]["y"]),
        "policy_by_rho_json": json.dumps(
            {
                f"{rho:.2f}": list(policy_rows[int(index)]["y"])
                for rho, index in zip(rho_values, best_indices)
            }
        ),
        "dda_policy_active": any(changed[1:]),
        "first_policy_change_rho": min(
            (rho for rho, is_changed in zip(rho_values, changed) if is_changed),
            default=np.nan,
        ),
        "max_delta_rho_value": float(deltas[max_index]),
        "max_delta_rho_percent": 100.0
        * float(deltas[max_index])
        / max(1.0, abs(float(objective_matrix[int(best_indices[max_index]), max_index]))),
    }


def make_distribution_figure(
    patterns: pd.DataFrame,
    figure_dir: Path,
    paper_rho: float,
) -> None:
    pattern_order = [
        "Neither critical link fails",
        "L2 only",
        "L3 only",
        "Both critical links fail",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    for axis, policy_label in zip(
        axes,
        ["nominal policy", "ambiguity-dependent policy"],
    ):
        subset = patterns[patterns["policy_label"] == policy_label]
        nominal = subset[subset["distribution"] == "nominal"].set_index(
            "critical_failure_pattern"
        )["probability"].reindex(pattern_order)
        worst = subset[subset["distribution"] == "worst case"].set_index(
            "critical_failure_pattern"
        )["probability"].reindex(pattern_order)
        positions = np.arange(len(pattern_order))
        width = 0.36
        axis.bar(positions - width / 2, nominal, width, label="Nominal", color="#94A3B8")
        axis.bar(positions + width / 2, worst, width, label="Worst case", color="#DC2626")
        axis.set_xticks(positions, ["Neither", "L2 only", "L3 only", "Both"])
        axis.set_title(policy_label.capitalize())
        axis.set_xlabel("Critical-link failure pattern")
    axes[0].set_ylabel("Probability")
    axes[1].legend(frameon=False)
    figure.suptitle(f"Worst-case redistribution in the active budget regime ($\\rho={paper_rho:.2f}$)")
    save_figure(figure, figure_dir, "fig_noto_mechanism_01_distribution_patterns")


def make_state_shift_figure(
    state_shifts: pd.DataFrame,
    figure_dir: Path,
    paper_rho: float,
) -> None:
    data = state_shifts[state_shifts["policy_label"] == "nominal policy"].copy()
    data = data.reindex(data["probability_shift"].abs().sort_values(ascending=False).index).head(12)
    data = data.sort_values("probability_shift")
    colors = np.where(data["probability_shift"] >= 0.0, "#DC2626", "#2563EB")
    figure, axis = plt.subplots(figsize=(8.0, 5.4))
    axis.barh(data["failed_link_labels"], data["probability_shift"], color=colors)
    axis.axvline(0.0, color="#111827", linewidth=0.8)
    axis.set_xlabel("Worst-case probability minus nominal probability")
    axis.set_ylabel("Failed links in state")
    axis.set_title(f"Largest state-level probability shifts under the nominal policy ($\\rho={paper_rho:.2f}$)")
    save_figure(figure, figure_dir, "fig_noto_mechanism_02_state_shifts")


def make_cost_budget_figure(
    summary: pd.DataFrame,
    cost_multipliers: list[float],
    figure_dir: Path,
) -> None:
    active = summary.pivot(
        index="L3_cost_multiplier",
        columns="L2_cost_multiplier",
        values="active_budget_share_percent",
    ).reindex(index=cost_multipliers, columns=cost_multipliers)
    value = summary.pivot(
        index="L3_cost_multiplier",
        columns="L2_cost_multiplier",
        values="max_delta_rho_percent",
    ).reindex(index=cost_multipliers, columns=cost_multipliers)
    figure, axes = plt.subplots(1, 2, figsize=(10.7, 4.6))
    for axis, matrix, title, color_map, color_label in [
        (axes[0], active, "Share of budget range that is DDA-active", "viridis", "Percent"),
        (axes[1], value, "Largest policy-switch value", "magma", "Percent of objective"),
    ]:
        image = axis.imshow(matrix.values, origin="lower", aspect="auto", cmap=color_map)
        axis.set_xticks(range(len(cost_multipliers)), [f"{value:.1f}" for value in cost_multipliers])
        axis.set_yticks(range(len(cost_multipliers)), [f"{value:.1f}" for value in cost_multipliers])
        axis.set_xlabel("L2 cost multiplier")
        axis.set_ylabel("L3 cost multiplier")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.82, label=color_label)
    save_figure(figure, figure_dir, "fig_noto_mechanism_03_cost_budget_robustness")


def make_baseline_frontier_figure(
    surface: pd.DataFrame,
    rho_values: list[float],
    figure_dir: Path,
) -> None:
    data = surface[
        np.isclose(surface["L2_cost_multiplier"], 1.0)
        & np.isclose(surface["L3_cost_multiplier"], 1.0)
    ].copy()
    budgets = np.sort(data["budget_multiplier"].unique())
    matrix = np.zeros((len(rho_values), len(budgets)))
    for column, budget in enumerate(budgets):
        row = data[np.isclose(data["budget_multiplier"], budget)].iloc[0]
        deltas = json.loads(row["delta_by_rho_json"])
        for row_index, rho in enumerate(rho_values):
            matrix[row_index, column] = float(deltas[f"{rho:.2f}"])
    figure, axis = plt.subplots(figsize=(8.5, 4.4))
    image = axis.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        extent=[budgets.min(), budgets.max(), min(rho_values), max(rho_values)],
        cmap="YlOrRd",
    )
    axis.set_xlabel("Road-budget multiplier relative to calibrated $B_Y$")
    axis.set_ylabel("Ambiguity radius $\\rho$")
    axis.set_title("Budget regions with ambiguity-dependent policy value")
    figure.colorbar(image, ax=axis, label="Policy-switch value $\\Delta_\\rho$")
    save_figure(figure, figure_dir, "fig_noto_mechanism_04_baseline_budget_frontier")


def save_figure(figure: plt.Figure, figure_dir: Path, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(figure_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(figure_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


def write_status(
    output_dir: Path,
    status: str,
    message: str,
    extra: dict[str, Any] | None = None,
) -> None:
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
