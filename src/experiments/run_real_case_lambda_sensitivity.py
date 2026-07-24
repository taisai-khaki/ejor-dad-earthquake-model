"""Distance-normalization sensitivity for the illustrative real-data case."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .run_real_case import (
    build_real_case_instance,
    _make_config,
    _plot_routes,
    _results_table,
    _run_methods,
)


def run_lambda_sensitivity(
    csv_path: str | Path = r"C:\Users\L03128674\Downloads\Delivery_Logistics.csv",
    output_dir: str | Path = "outputs/real_case_lambda_sensitivity_beta010",
    lambdas: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> pd.DataFrame:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []

    for lam in lambdas:
        case_dir = out_dir / f"lambda_{lam:.2f}".replace(".", "p")
        case_dir.mkdir(parents=True, exist_ok=True)
        instance, sample = build_real_case_instance(csv_path, distance_scale=float(lam), beta_slack_ratio=0.10)
        config = _make_config(seed=instance.seed)
        solutions = _run_methods(instance, config)
        table = _results_table(instance, solutions)
        table.insert(0, "lambda_distance_scale", float(lam))
        table.to_csv(case_dir / "real_case_results.csv", index=False)
        sample.to_csv(case_dir / "real_case_sample.csv", index=False)
        _plot_routes(instance, solutions, case_dir)
        rows.append(table)

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(out_dir / "real_case_lambda_sensitivity_full.csv", index=False)

    compact = combined[
        [
            "lambda_distance_scale",
            "method",
            "feasible",
            "objective",
            "improvement_vs_truck_only_pct",
            "runtime_seconds",
            "drone_served_customers",
            "drone_service_share",
            "drone_served_high",
            "drone_served_medium",
            "drone_served_low",
            "waiting_sync_cost",
        ]
    ].copy()
    compact["drone_service_share_pct"] = 100.0 * compact["drone_service_share"]
    compact = compact.drop(columns=["drone_service_share"])
    compact.to_csv(out_dir / "table_real_case_lambda_sensitivity_beta010.csv", index=False)
    (out_dir / "table_real_case_lambda_sensitivity_beta010.tex").write_text(
        compact.to_latex(index=False, escape=False, na_rep="--", float_format="%.2f"),
        encoding="utf-8",
    )
    return compact


if __name__ == "__main__":
    df = run_lambda_sensitivity()
    print(df.to_string(index=False))
