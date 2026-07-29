from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = ["#1D4ED8", "#059669", "#D97706", "#7C3AED", "#DC2626"]


def main() -> None:
    args = parse_args()
    output_name = "access_experiment_pilot" if args.mode == "pilot" else "access_experiment"
    output_dir = Path("data_work/noto") / output_name
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})

    make_sector_figure(table_dir, figure_dir, args.paper_rho)
    make_dda_figure(table_dir, figure_dir)
    make_road_value_figure(table_dir, figure_dir)
    make_link_policy_figure(table_dir, figure_dir)
    make_near_optimal_figure(table_dir, figure_dir)
    make_probability_shift_figure(table_dir, figure_dir)
    make_support_expansion_figure(table_dir, figure_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create publication-ready Noto empirical figures.")
    parser.add_argument("--mode", choices=["pilot", "full"], default="full")
    parser.add_argument("--paper-rho", type=float, default=0.10)
    return parser.parse_args()


def make_sector_figure(table_dir: Path, figure_dir: Path, paper_rho: float) -> None:
    data = pd.read_csv(table_dir / "table_noto_access_main_results.csv")
    data = data[np.isclose(data["rho"], paper_rho)].copy()
    labels = [
        "No investment",
        "Exposure only",
        "Capacity only",
        "Exposure + capacity",
        "All sectors",
    ]
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    bars = axis.barh(labels, data["reduction_from_no_investment"], color=COLORS)
    axis.invert_yaxis()
    axis.set_xlabel("Reduction in worst-case expected loss")
    axis.set_title(f"Sector contribution in the Noto instance ($\\rho={paper_rho:.2f}$)")
    axis.bar_label(bars, fmt="%.1f", padding=4, fontsize=8)
    save_figure(figure, figure_dir, "fig_noto_01_sector_contribution")


def make_dda_figure(table_dir: Path, figure_dir: Path) -> None:
    data = pd.read_csv(table_dir / "table_noto_access_sector_comparison.csv")
    selected = data[data["comparison"].isin(["exposure + capacity; no road retrofit", "all-sector discretized"])]
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    for comparison, label, color in [
        ("exposure + capacity; no road retrofit", "No road retrofit", COLORS[2]),
        ("all-sector discretized", "All sectors", COLORS[0]),
    ]:
        line = selected[selected["comparison"] == comparison].sort_values("rho")
        axis.plot(line["rho"], line["objective"], marker="o", linewidth=2.2, label=label, color=color)
    axis.set_xlabel("Ambiguity radius $\\rho$")
    axis.set_ylabel("Worst-case expected loss")
    axis.set_title("Ambiguity sensitivity and road-retrofit value")
    axis.legend(frameon=False)
    save_figure(figure, figure_dir, "fig_noto_02_ambiguity_sweep")


def make_road_value_figure(table_dir: Path, figure_dir: Path) -> None:
    data = pd.read_csv(table_dir / "table_noto_access_claim_diagnostics.csv").sort_values("rho")
    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    axis.plot(data["rho"], data["road_value_over_no_retrofit"], marker="o", linewidth=2.2, color=COLORS[1])
    axis.set_xlabel("Ambiguity radius $\\rho$")
    axis.set_ylabel("Objective gain over no road retrofit")
    axis.set_title("Marginal value of the road-retrofit sector")
    save_figure(figure, figure_dir, "fig_noto_03_road_value")


def make_link_policy_figure(table_dir: Path, figure_dir: Path) -> None:
    data = pd.read_csv(table_dir / "table_noto_access_link_decisions.csv")
    pivot = data.pivot(index="corridor", columns="rho", values="selected_retrofit_y")
    corridor_order = data.drop_duplicates("corridor")["corridor"].tolist()
    pivot = pivot.reindex(corridor_order)
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    image = axis.imshow(pivot.to_numpy(), vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    axis.set_xticks(np.arange(len(pivot.columns)), [f"{rho:.2f}" for rho in pivot.columns])
    axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
    axis.set_xlabel("Ambiguity radius $\\rho$")
    axis.set_title("Best discretized corridor-retrofit policy")
    for row_index in range(len(pivot.index)):
        for column_index in range(len(pivot.columns)):
            value = pivot.iloc[row_index, column_index]
            text_color = "white" if value >= 0.65 else "black"
            axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)
    figure.colorbar(image, ax=axis, label="Retrofit ratio $y_{ij}$")
    save_figure(figure, figure_dir, "fig_noto_04_link_policy")


def make_near_optimal_figure(table_dir: Path, figure_dir: Path) -> None:
    data = pd.read_csv(table_dir / "table_noto_access_near_optimal.csv")
    pivot = data.pivot(index="threshold_percent", columns="rho", values="near_optimal_policy_count")
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    image = axis.imshow(pivot.to_numpy(), cmap="Oranges", aspect="auto")
    axis.set_xticks(np.arange(len(pivot.columns)), [f"{rho:.2f}" for rho in pivot.columns])
    axis.set_yticks(np.arange(len(pivot.index)), [f"{threshold:.2f}%" for threshold in pivot.index])
    axis.set_xlabel("Ambiguity radius $\\rho$")
    axis.set_ylabel("Near-optimality threshold")
    axis.set_title("Number of near-optimal retrofit policies")
    for row_index in range(len(pivot.index)):
        for column_index in range(len(pivot.columns)):
            axis.text(column_index, row_index, f"{int(pivot.iloc[row_index, column_index])}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Policy count")
    save_figure(figure, figure_dir, "fig_noto_05_near_optimal_sets")


def make_probability_shift_figure(table_dir: Path, figure_dir: Path) -> None:
    data = pd.read_csv(table_dir / "table_noto_access_dda_summary.csv").sort_values("rho")
    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    axis.plot(data["rho"], data["adversarial_target_mass_shift"], marker="o", linewidth=2.2, color=COLORS[3])
    axis.set_xlabel("Ambiguity radius $\\rho$")
    axis.set_ylabel("Mass shifted to the adverse state")
    axis.set_title("Worst-case probability redistribution")
    save_figure(figure, figure_dir, "fig_noto_06_probability_shift")


def make_support_expansion_figure(table_dir: Path, figure_dir: Path) -> None:
    data = pd.read_csv(table_dir / "table_noto_access_ambiguity_support_diagnostic.csv").sort_values("rho")
    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    axis.bar(
        data["rho"],
        data["mass_added_to_zero_nominal_states"],
        width=0.035,
        color=COLORS[4],
        label="Added to zero-nominal states",
    )
    axis.plot(
        data["rho"],
        data["total_positive_probability_shift"],
        marker="o",
        linewidth=1.8,
        color="black",
        label="Total positive shift",
    )
    axis.set_xlabel("Ambiguity radius $\\rho$")
    axis.set_ylabel("Probability mass")
    axis.set_title("Support expansion under total-variation ambiguity")
    axis.legend(frameon=False)
    save_figure(figure, figure_dir, "fig_noto_07_support_expansion")


def save_figure(figure: plt.Figure, figure_dir: Path, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(figure_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(figure_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
