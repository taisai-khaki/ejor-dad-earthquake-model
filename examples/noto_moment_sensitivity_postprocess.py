from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import noto_access_experiment as noto


ROOT = Path("data_work/noto")
MAIN_OUTPUT = ROOT / "moment_constrained_active_fine_grid"
PROFILE_DIRS = {
    "Tight": ROOT / "moment_constrained_tight_rho025",
    "Moderate": MAIN_OUTPUT,
    "Loose": ROOT / "moment_constrained_loose_rho025",
}


def main() -> None:
    rows = build_rows()
    table = pd.DataFrame(rows)
    noto.write_table(table, MAIN_OUTPUT, "table_noto_moment_tolerance_sensitivity")
    make_figure(table, MAIN_OUTPUT / "figures")


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    moderate_row: pd.Series | None = None
    for order, (profile, directory) in enumerate(PROFILE_DIRS.items(), start=1):
        summary = pd.read_csv(directory / "tables" / "table_noto_moment_envelope_summary.csv")
        row = summary[np.isclose(summary["rho"], 0.25)].iloc[0]
        design = json.loads((directory / "run_design.json").read_text(encoding="utf-8"))
        envelope = design["failure_moment_envelope"]
        rows.append(
            {
                "order": order,
                "profile": profile,
                "marginal_relative_tolerance": envelope["marginal_relative_tolerance"],
                "marginal_absolute_tolerance": envelope["marginal_absolute_tolerance"],
                "joint_relative_tolerance": envelope["joint_relative_tolerance"],
                "joint_absolute_tolerance": envelope["joint_absolute_tolerance"],
                "count_mean_absolute_tolerance": envelope["count_mean_absolute_tolerance"],
                "count_second_moment_relative_tolerance": envelope[
                    "count_second_moment_relative_tolerance"
                ],
                "count_second_moment_absolute_tolerance": envelope[
                    "count_second_moment_absolute_tolerance"
                ],
                "rho": 0.25,
                "best_objective": float(row["best_objective"]),
                "best_y_json": row["best_y_json"],
                "worst_failure_count_mean": float(row["worst_failure_count_mean"]),
                "worst_failure_count_sd": float(row["worst_failure_count_sd"]),
                "full_policy_delta_rho_value": float(row["full_policy_delta_rho_value"]),
                "second_best_gap": float(row["second_best_gap"]),
            }
        )
        if profile == "Moderate":
            moderate_row = row
    if moderate_row is None:
        raise RuntimeError("Moderate profile is required for the capped-TV comparison.")
    rows.append(
        {
            "order": 4,
            "profile": "Capped TV",
            "marginal_relative_tolerance": np.nan,
            "marginal_absolute_tolerance": np.nan,
            "joint_relative_tolerance": np.nan,
            "joint_absolute_tolerance": np.nan,
            "count_mean_absolute_tolerance": np.nan,
            "count_second_moment_relative_tolerance": np.nan,
            "count_second_moment_absolute_tolerance": np.nan,
            "rho": 0.25,
            "best_objective": float(moderate_row["baseline_capped_tv_objective"]),
            "best_y_json": moderate_row["baseline_best_y_json"],
            "worst_failure_count_mean": float(moderate_row["baseline_worst_failure_count_mean"]),
            "worst_failure_count_sd": float(moderate_row["baseline_worst_failure_count_sd"]),
            "full_policy_delta_rho_value": float(
                moderate_row["baseline_full_policy_delta_rho_value"]
            ),
            "second_best_gap": 14.982168285357147,
        }
    )
    return rows


def make_figure(table: pd.DataFrame, output_dir: Path) -> None:
    data = table.sort_values("order")
    labels = data["profile"].tolist()
    positions = np.arange(len(data))
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    axes[0].bar(positions, data["best_objective"], color="#2878B5")
    axes[0].set_ylabel("Best worst-case modeled loss")
    axes[0].set_title("Robust objective")
    axes[1].bar(positions, data["worst_failure_count_mean"], color="#F28E2B")
    axes[1].set_ylabel("Expected failed links")
    axes[1].set_title("Worst-case failure mean")
    axes[2].bar(positions, data["full_policy_delta_rho_value"], color="#3A923A")
    axes[2].set_ylabel("Modeled-loss units")
    axes[2].set_title("Value of capacity adaptation")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=20, ha="right")
    figure.suptitle(r"Moment-envelope sensitivity at $\rho=0.25$")
    figure.tight_layout()
    figure.savefig(output_dir / "fig_noto_moment_06_tolerance_sensitivity.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / "fig_noto_moment_06_tolerance_sensitivity.svg", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
