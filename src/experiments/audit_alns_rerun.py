"""Audit submitted versus corrected ALNS rows on matched instance IDs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def audit(*, submitted_path: str | Path, corrected_path: str | Path, output_dir: str | Path) -> None:
    submitted = pd.read_csv(submitted_path)
    corrected = pd.read_csv(corrected_path)
    submitted = submitted[submitted["method_name"].eq("feasibility_aware_alns")].copy()
    corrected = corrected[corrected["method_name"].eq("feasibility_aware_alns")].copy()
    columns = [
        "instance_id", "n", "objective_total", "iterations", "candidate_reassignments_evaluated",
        "cpu_time_seconds", "drone_service_share", "final_solution_feasibility_flag",
    ]
    left = submitted[columns].rename(columns={column: f"submitted_{column}" for column in columns if column != "instance_id"})
    right = corrected[columns].rename(columns={column: f"corrected_{column}" for column in columns if column != "instance_id"})
    detail = left.merge(right, on="instance_id", how="outer", validate="one_to_one")
    detail["objective_difference_corrected_minus_submitted"] = detail["corrected_objective_total"] - detail["submitted_objective_total"]
    detail["objective_changed"] = detail["objective_difference_corrected_minus_submitted"].abs() > 1e-6
    detail["submitted_zero_drone_share"] = detail["submitted_drone_service_share"].fillna(0).abs() <= 1e-12
    detail["corrected_zero_drone_share"] = detail["corrected_drone_service_share"].fillna(0).abs() <= 1e-12
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    detail.sort_values(["corrected_n", "instance_id"]).to_csv(output / "submitted_vs_corrected_alns_instance_detail.csv", index=False, float_format="%.10f")

    rows = []
    for size_label, subset in [("All", detail), *[(str(size), detail[detail["corrected_n"].eq(size)]) for size in sorted(detail["corrected_n"].dropna().unique())]]:
        rows.append(
            {
                "size": size_label,
                "matched_rows": int(len(subset)),
                "objective_changed_rows": int(subset["objective_changed"].sum()),
                "submitted_mean_iterations": float(subset["submitted_iterations"].mean()),
                "corrected_mean_iterations": float(subset["corrected_iterations"].mean()),
                "submitted_mean_candidate_evaluations": float(subset["submitted_candidate_reassignments_evaluated"].mean()),
                "corrected_mean_candidate_evaluations": float(subset["corrected_candidate_reassignments_evaluated"].mean()),
                "submitted_zero_drone_share_rate": float(subset["submitted_zero_drone_share"].mean()),
                "corrected_zero_drone_share_rate": float(subset["corrected_zero_drone_share"].mean()),
                "submitted_mean_objective": float(subset["submitted_objective_total"].mean()),
                "corrected_mean_objective": float(subset["corrected_objective_total"].mean()),
                "mean_objective_difference_corrected_minus_submitted": float(subset["objective_difference_corrected_minus_submitted"].mean()),
                "max_abs_objective_difference": float(subset["objective_difference_corrected_minus_submitted"].abs().max()),
            }
        )
    pd.DataFrame(rows).to_csv(output / "submitted_vs_corrected_alns_by_size.csv", index=False, float_format="%.10f")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submitted", required=True)
    parser.add_argument("--corrected", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    audit(submitted_path=args.submitted, corrected_path=args.corrected, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
