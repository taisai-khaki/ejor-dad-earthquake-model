from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ejor_dad.checkpoint import atomic_write_dataframe


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    fine_dir = root / "support_preserving_active_fine_grid" / "tables"
    tight_dir = root / "support_preserving_tight_budget" / "tables"
    sbb_dir = root / "support_preserving_active_m2_sbb" / "tables"
    output_dir = root / "support_preserving_active_validation"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    fine_summary = read_complete_table(
        fine_dir / "table_noto_active_fine_grid_summary.csv",
        "fine-grid summary",
    )
    fine_candidates = read_complete_table(
        fine_dir / "table_noto_active_fine_grid_candidates.csv",
        "fine-grid candidates",
        require_rhos=False,
    )
    coarse = read_complete_table(
        tight_dir / "table_noto_budget_mechanism_active_case.csv",
        "coarse-grid active case",
    )
    comparison = build_resolution_comparison(coarse, fine_summary)
    write_table(comparison, table_dir, "table_noto_active_resolution_comparison")
    write_table(fine_summary, table_dir, "table_noto_active_fine_grid_summary")

    sbb_path = sbb_dir / "table_noto_active_critical_m2_sbb.csv"
    sbb = pd.DataFrame()
    if sbb_path.exists():
        sbb = read_complete_table(sbb_path, "active critical-link SBB", require_rhos=False)
        write_table(sbb, table_dir, "table_noto_active_critical_m2_sbb")
    elif args.require_sbb:
        raise FileNotFoundError(f"Required completed SBB table is missing: {sbb_path}")

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.22})
    make_resolution_figure(comparison, figure_dir)
    make_fine_surface_figure(fine_candidates, fine_summary, figure_dir)
    make_policy_trajectory_figure(fine_summary, figure_dir)
    if not sbb.empty:
        make_sbb_gap_figure(sbb, figure_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Postprocess coarse, fine-grid, and reduced-SBB active-budget diagnostics."
    )
    parser.add_argument("--root", default="data_work/noto")
    parser.add_argument("--require-sbb", action="store_true")
    return parser.parse_args()


def read_complete_table(path: Path, label: str, require_rhos: bool = True) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required {label} table is missing: {path}")
    dataframe = pd.read_csv(path)
    if require_rhos:
        expected = np.asarray([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])
        observed = np.sort(dataframe["rho"].unique())
        if observed.shape != expected.shape or not np.allclose(observed, expected):
            raise RuntimeError(f"{label} is incomplete: expected {expected.tolist()}, found {observed.tolist()}")
    return dataframe


def build_resolution_comparison(coarse: pd.DataFrame, fine: pd.DataFrame) -> pd.DataFrame:
    coarse = coarse.sort_values("rho").reset_index(drop=True)
    fine = fine.sort_values("rho").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for method, dataframe, objective_column, policy_column in [
        ("five-level full-grid sensitivity", coarse, "best_objective", "best_y_json"),
        ("0.05 critical-link grid", fine, "best_objective", "best_y_json"),
    ]:
        for row in dataframe.itertuples(index=False):
            rows.append(
                {
                    "method": method,
                    "rho": float(row.rho),
                    "budget_multiplier": float(row.budget_multiplier),
                    "grid_step": 0.25 if method.startswith("five-level") else float(row.grid_step),
                    "objective": float(getattr(row, objective_column)),
                    "selected_y_json": getattr(row, policy_column),
                    "objective_using_rho0_policy": float(row.objective_using_rho0_policy),
                    "delta_rho_value": float(row.delta_rho_value),
                    "policy_changed_from_rho0": bool(row.policy_changed_from_rho0),
                    "scope": (
                        "all five y variables on the five-level grid"
                        if method.startswith("five-level")
                        else "full five-link state/recourse model; only L2 and L3 refined and other y fixed zero"
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_resolution_figure(comparison: pd.DataFrame, figure_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.7))
    styles = {
        "five-level full-grid sensitivity": ("#DC2626", "o", "Five-level grid"),
        "0.05 critical-link grid": ("#2563EB", "s", "0.05 critical-link grid"),
    }
    for method, group in comparison.groupby("method", sort=False):
        color, marker, label = styles[method]
        group = group.sort_values("rho")
        axis.plot(
            group["rho"],
            group["delta_rho_value"],
            color=color,
            marker=marker,
            linewidth=2.1,
            label=label,
        )
    axis.axhline(0.0, color="#111827", linewidth=0.8)
    axis.set_xlabel("Ambiguity radius $\\rho$")
    axis.set_ylabel("Policy-switch value $\\Delta_\\rho$")
    axis.set_title("Policy activation under coarse and refined retrofit grids")
    axis.legend(frameon=False)
    save_figure(figure, figure_dir, "fig_noto_active_validation_01_resolution_comparison")


def make_fine_surface_figure(
    candidates: pd.DataFrame,
    summary: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.7, 4.7), sharex=True, sharey=True)
    for axis, rho in zip(axes, [0.0, 0.25]):
        data = candidates[np.isclose(candidates["rho"], rho)].copy()
        pivot = data.pivot(
            index="critical_link_1_y",
            columns="critical_link_2_y",
            values="gap_to_best",
        ).sort_index().sort_index(axis=1)
        image = axis.imshow(
            pivot.values,
            origin="lower",
            aspect="auto",
            extent=[
                pivot.columns.min(),
                pivot.columns.max(),
                pivot.index.min(),
                pivot.index.max(),
            ],
            cmap="viridis",
        )
        best_y = json.loads(summary[np.isclose(summary["rho"], rho)].iloc[0]["best_y_json"])
        axis.scatter(best_y[2], best_y[1], marker="*", s=170, color="#DC2626", edgecolor="white")
        axis.set_title(f"$\\rho={rho:.2f}$")
        axis.set_xlabel("L3 retrofit level")
        figure.colorbar(image, ax=axis, shrink=0.82, label="Objective gap")
    axes[0].set_ylabel("L2 retrofit level")
    figure.suptitle("Refined critical-link objective landscape")
    save_figure(figure, figure_dir, "fig_noto_active_validation_02_fine_grid_surface")


def make_policy_trajectory_figure(summary: pd.DataFrame, figure_dir: Path) -> None:
    data = summary.sort_values("rho").copy()
    policies = np.asarray([json.loads(value) for value in data["best_y_json"]], dtype=float)
    figure, axis = plt.subplots(figsize=(7.4, 4.7))
    axis.plot(data["rho"], policies[:, 1], marker="o", linewidth=2.1, label="L2")
    axis.plot(data["rho"], policies[:, 2], marker="s", linewidth=2.1, label="L3")
    axis.set_ylim(-0.03, 1.05)
    axis.set_xlabel("Ambiguity radius $\\rho$")
    axis.set_ylabel("Selected retrofit level")
    axis.set_title("Refined-grid critical-link policy trajectory")
    axis.legend(frameon=False)
    save_figure(figure, figure_dir, "fig_noto_active_validation_03_policy_trajectory")


def make_sbb_gap_figure(sbb: pd.DataFrame, figure_dir: Path) -> None:
    data = sbb.sort_values("rho")
    figure, axis = plt.subplots(figsize=(7.4, 4.7))
    axis.semilogy(data["rho"], np.maximum(data["absolute_gap"], 1e-10), marker="o", linewidth=2.1)
    axis.axhline(
        float(data["certificate_absolute_tolerance"].iloc[0]),
        color="#DC2626",
        linestyle="--",
        label="Declared tolerance",
    )
    axis.set_xlabel("Ambiguity radius $\\rho$")
    axis.set_ylabel("SBB absolute gap (log scale)")
    axis.set_title("Reduced critical-link SBB gap diagnostic")
    axis.legend(frameon=False)
    save_figure(figure, figure_dir, "fig_noto_active_validation_04_sbb_gap")


def write_table(dataframe: pd.DataFrame, table_dir: Path, stem: str) -> None:
    atomic_write_dataframe(dataframe, table_dir / f"{stem}.csv")
    atomic_write_dataframe(dataframe, table_dir / f"{stem}.tex", kind="latex", escape=True)


def save_figure(figure: plt.Figure, figure_dir: Path, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(figure_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(figure_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
