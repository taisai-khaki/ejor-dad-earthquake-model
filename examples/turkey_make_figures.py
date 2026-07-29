from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


TABLE_DIR = Path("data_work/turkey/paper_tables")
FIGURE_DIR = Path("data_work/turkey/paper_figures")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    make_main_results()
    make_policy_comparison()
    make_ambiguity_sensitivity()
    make_budget_design()
    make_candidate_link_sensitivity()
    make_zone_decisions()
    make_link_decisions()
    make_worst_case_shift()
    print(f"Figures written to: {FIGURE_DIR.resolve()}")


def savefig(name: str) -> None:
    png = FIGURE_DIR / f"{name}.png"
    svg = FIGURE_DIR / f"{name}.svg"
    plt.tight_layout()
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")
    plt.close()


def make_main_results() -> None:
    data = pd.read_csv(TABLE_DIR / "table_03_main_results.csv")
    plt.figure(figsize=(7.2, 4.5))
    ax = sns.barplot(data=data, x="Plan", y="Worst-case expected deaths", color="#3B82F6")
    ax.set_title("Worst-case expected deaths by investment plan")
    ax.set_xlabel("")
    ax.set_ylabel("Worst-case expected deaths")
    ax.tick_params(axis="x", rotation=15)
    add_bar_labels(ax)
    savefig("fig_01_main_results")


def make_policy_comparison() -> None:
    data = pd.read_csv(TABLE_DIR / "table_11_policy_comparison.csv")
    data = data.sort_values("Worst-case expected deaths", ascending=False)
    plt.figure(figsize=(8.2, 5.0))
    ax = sns.barplot(data=data, y="Policy", x="Worst-case expected deaths", color="#10B981")
    ax.set_title("Policy comparison across mitigation sectors")
    ax.set_xlabel("Worst-case expected deaths")
    ax.set_ylabel("")
    savefig("fig_02_policy_comparison")


def make_ambiguity_sensitivity() -> None:
    data = pd.read_csv(TABLE_DIR / "table_05_ambiguity_sensitivity.csv")
    plt.figure(figsize=(6.8, 4.4))
    ax = sns.lineplot(data=data, x="rho", y="Worst-case expected deaths", marker="o", linewidth=2.2, color="#7C3AED")
    ax.set_title("Sensitivity to total-variation ambiguity radius")
    ax.set_xlabel("Ambiguity radius $\\rho$")
    ax.set_ylabel("Worst-case expected deaths")
    savefig("fig_03_ambiguity_sensitivity")


def make_budget_design() -> None:
    data = pd.read_csv(TABLE_DIR / "table_12_budget_design.csv")
    data = data.sort_values("Worst-case expected deaths", ascending=True)
    plt.figure(figsize=(8.8, 5.4))
    ax = sns.barplot(data=data, y="Design", x="Worst-case expected deaths", color="#F97316")
    ax.set_title("Budget allocation design comparison")
    ax.set_xlabel("Worst-case expected deaths")
    ax.set_ylabel("")
    savefig("fig_04_budget_design")


def make_candidate_link_sensitivity() -> None:
    data = pd.read_csv(TABLE_DIR / "table_13_candidate_link_sensitivity.csv")
    plt.figure(figsize=(6.8, 4.4))
    ax = sns.lineplot(data=data, x="candidate_links", y="Road-retrofit gain", marker="o", linewidth=2.2, color="#DC2626")
    ax.set_title("Road-retrofit gain as candidate network expands")
    ax.set_xlabel("Candidate road links")
    ax.set_ylabel("Gain over no-retrofit objective")
    savefig("fig_05_candidate_link_sensitivity")


def make_zone_decisions() -> None:
    data = pd.read_csv(TABLE_DIR / "table_07_zone_decisions.csv")
    data = data.sort_values("at_risk_population", ascending=False).head(12)
    plt.figure(figsize=(8.4, 5.2))
    ax = sns.scatterplot(
        data=data,
        x="at_risk_population",
        y="renovation_ratio_z",
        size="total_buildings",
        hue="collapse_fraction",
        palette="mako_r",
        sizes=(50, 320),
        legend="brief",
    )
    ax.set_title("Zone renovation decisions by exposure and damage")
    ax.set_xlabel("Baseline at-risk population")
    ax.set_ylabel("Renovation ratio $z_{rl}$")
    ax.set_ylim(-0.05, 1.05)
    savefig("fig_06_zone_renovation")


def make_link_decisions() -> None:
    data = pd.read_csv(TABLE_DIR / "table_08_link_decisions.csv")
    data = data.sort_values("Phi_ij", ascending=False)
    data["link_short"] = [f"link {idx + 1}" for idx in range(len(data))]
    fig, ax1 = plt.subplots(figsize=(7.2, 4.5))
    sns.barplot(data=data, x="link_short", y="Phi_ij", color="#60A5FA", ax=ax1)
    ax1.set_ylabel("Failure probability $\\Phi_{ij}$")
    ax1.set_xlabel("")
    ax2 = ax1.twinx()
    sns.lineplot(data=data, x="link_short", y="retrofit_ratio_y", marker="o", color="#EF4444", ax=ax2)
    ax2.set_ylabel("Retrofit ratio $y_{ij}$")
    ax2.set_ylim(-0.05, 1.05)
    ax1.set_title("Hazard-calibrated road risks and retrofit decisions")
    savefig("fig_07_link_risk_retrofit")


def make_worst_case_shift() -> None:
    data = pd.read_csv(TABLE_DIR / "table_10_worst_case_states.csv")
    data = data.sort_values("probability_shift", ascending=True)
    data["state_short"] = [f"s{idx + 1}" for idx in range(len(data))]
    plt.figure(figsize=(8.2, 5.0))
    ax = sns.barplot(data=data, y="state_short", x="probability_shift", color="#8B5CF6")
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_title("Worst-case probability shifts across network states")
    ax.set_xlabel("Worst-case probability - nominal probability")
    ax.set_ylabel("Network state")
    savefig("fig_08_worst_case_shift")


def add_bar_labels(ax) -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=8)


if __name__ == "__main__":
    main()
