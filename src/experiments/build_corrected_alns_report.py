"""Build corrected-ALNS runtime and matched-wall-clock comparison tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ALNS_METHOD = "feasibility_aware_alns"
METHODS = {"nils": "M5 NILS", "nils_r": "M9 NILS-R"}
TOL = 1e-6


def _bool_series(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return frame[column].map(lambda value: str(value).strip().lower() in {"true", "1", "1.0"})
    raise KeyError(f"None of the feasibility columns exist: {columns}")


def _wilcoxon_p(left: np.ndarray, right: np.ndarray) -> float:
    differences = left - right
    differences = differences[np.isfinite(differences)]
    if differences.size == 0 or np.all(np.abs(differences) <= TOL):
        return 1.0
    try:
        return float(wilcoxon(left, right, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def _pair_table(alns: pd.DataFrame, competitor: pd.DataFrame, method: str, budget: float) -> pd.DataFrame:
    left = competitor[["instance_id", "n", "objective", "cpu_time_seconds"]].copy()
    left["feasible_method"] = _bool_series(competitor, ["feasible", "final_solution_feasibility_flag", "is_feasible_run"])
    right = alns[["instance_id", "n", "objective_total", "cpu_time_seconds"]].copy()
    right["feasible_alns"] = _bool_series(alns, ["final_solution_feasibility_flag", "is_feasible_run", "feasible"])
    right = right.rename(columns={"objective_total": "objective_alns", "cpu_time_seconds": "cpu_alns"})
    merged = left.rename(columns={"objective": "objective_method", "cpu_time_seconds": "cpu_method"}).merge(
        right, on=["instance_id", "n"], how="inner", validate="one_to_one"
    )
    rows = []
    for size_label, subset in [("All", merged), *[(str(size), merged[merged["n"].eq(size)]) for size in sorted(merged["n"].unique())]]:
        pair = subset[subset["feasible_method"] & subset["feasible_alns"]].copy()
        left_obj = pair["objective_method"].to_numpy(dtype=float)
        right_obj = pair["objective_alns"].to_numpy(dtype=float)
        diff = left_obj - right_obj
        improvement = 100.0 * (right_obj - left_obj) / np.maximum(np.abs(right_obj), 1e-12)
        rows.append(
            {
                "method": METHODS[method],
                "size": size_label,
                "pairs": int(len(subset)),
                "feasible_pairs": int(len(pair)),
                "method_wins": int(np.sum(diff < -TOL)),
                "alns_wins": int(np.sum(diff > TOL)),
                "ties": int(np.sum(np.abs(diff) <= TOL)),
                "mean_objective_method": float(np.mean(left_obj)) if left_obj.size else np.nan,
                "mean_objective_alns": float(np.mean(right_obj)) if right_obj.size else np.nan,
                "median_objective_method": float(np.median(left_obj)) if left_obj.size else np.nan,
                "median_objective_alns": float(np.median(right_obj)) if right_obj.size else np.nan,
                "mean_improvement_method_vs_alns_pct": float(np.mean(improvement)) if improvement.size else np.nan,
                "median_improvement_method_vs_alns_pct": float(np.median(improvement)) if improvement.size else np.nan,
                "wilcoxon_p": _wilcoxon_p(left_obj, right_obj),
                "mean_cpu_method": float(subset["cpu_method"].mean()),
                "median_cpu_method": float(subset["cpu_method"].median()),
                "mean_cpu_alns": float(subset["cpu_alns"].mean()),
                "median_cpu_alns": float(subset["cpu_alns"].median()),
                "method_feasibility_rate": float(subset["feasible_method"].mean()),
                "alns_feasibility_rate": float(subset["feasible_alns"].mean()),
                "common_wall_clock_budget_seconds": float(budget),
            }
        )
    return pd.DataFrame(rows)


def build_report(*, alns_dir: str | Path, matched_dir: str | Path, output_dir: str | Path, budget: float) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    alns_path = Path(alns_dir) / "tables" / "alns_full_alns_runs_partial.csv"
    alns = pd.read_csv(alns_path)
    alns = alns[alns["method_name"].eq(ALNS_METHOD)].drop_duplicates("instance_id", keep="last")
    if len(alns) != 720:
        raise ValueError(f"Expected 720 corrected ALNS rows, found {len(alns)}")
    if not np.allclose(alns["iterations"].to_numpy(dtype=float), 40.0):
        raise ValueError("Corrected ALNS rows do not all report 40 iterations")

    runtime = (
        alns.assign(feasible=_bool_series(alns, ["final_solution_feasibility_flag", "is_feasible_run"]))
        .groupby("n", as_index=False)
        .agg(
            runs=("instance_id", "count"),
            mean_iterations=("iterations", "mean"),
            min_iterations=("iterations", "min"),
            max_iterations=("iterations", "max"),
            feasible_runs=("feasible", "sum"),
            mean_wall_seconds=("cpu_time_seconds", "mean"),
            median_wall_seconds=("cpu_time_seconds", "median"),
            max_wall_seconds=("cpu_time_seconds", "max"),
            mean_objective=("objective_total", "mean"),
            mean_drone_share=("drone_service_share", "mean"),
        )
    )
    runtime.insert(0, "size", runtime.pop("n"))
    runtime.to_csv(output / "corrected_alns_runtime_by_size.csv", index=False, float_format="%.10f")

    matched = pd.read_csv(Path(matched_dir) / "run_status_full.csv")
    for method in METHODS:
        table = _pair_table(alns, matched[matched["method_name"].eq(method)], method, budget)
        table.to_csv(output / f"table_{method}_vs_corrected_alns_matched_wallclock_by_size.csv", index=False, float_format="%.10f")

    protocol = {
        "alns_seed": 2237,
        "base_seed": 2026,
        "alns_max_iterations": 40,
        "alns_time_limit_seconds": 0,
        "common_wall_clock_budget_seconds": budget,
        "corrected_alns_rows": int(len(alns)),
        "corrected_alns_iterations": sorted(set(alns["iterations"].astype(float))),
        "matched_rows_by_method": matched.groupby("method_name")["instance_id"].nunique().to_dict(),
    }
    (output / "corrected_alns_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alns-dir", required=True)
    parser.add_argument("--matched-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget", type=float, required=True)
    args = parser.parse_args()
    build_report(alns_dir=args.alns_dir, matched_dir=args.matched_dir, output_dir=args.output_dir, budget=args.budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
