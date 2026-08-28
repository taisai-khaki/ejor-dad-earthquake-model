"""Aggregate matched NILS-R comparisons from the priority-study run file."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .statistics import paired_comparison


ALNS_METHOD = "feasibility_aware_alns"
NILSR_METHOD = "nils_r"
NILS_METHOD = "nils"
TIE_TOLERANCE = 1e-6


def _safe_mean(values: pd.Series | list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _safe_median(values: pd.Series | list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def _as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes"})


def _pair_frame(run_status: pd.DataFrame, comparison_method: str) -> pd.DataFrame:
    keys = ["instance_id", "scenario_id", "n"]
    columns = keys + ["method_name", "feasible", "objective", "cpu_time_seconds"]
    available = [column for column in columns if column in run_status.columns]
    rows = run_status[available].copy()
    rows["feasible"] = _as_bool(rows["feasible"])
    for column in ["objective", "cpu_time_seconds"]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.drop_duplicates(keys + ["method_name"], keep="last")
    left = rows[rows["method_name"].eq(NILSR_METHOD)][keys + ["feasible", "objective", "cpu_time_seconds"]]
    right = rows[rows["method_name"].eq(comparison_method)][keys + ["feasible", "objective", "cpu_time_seconds"]]
    left = left.rename(
        columns={
            "feasible": "feasible_nilsr",
            "objective": "objective_nilsr",
            "cpu_time_seconds": "cpu_time_nilsr",
        }
    )
    right = right.rename(
        columns={
            "feasible": "feasible_other",
            "objective": "objective_other",
            "cpu_time_seconds": "cpu_time_other",
        }
    )
    pairs = left.merge(right, on=keys, how="outer")
    pairs["feasible_nilsr"] = pairs["feasible_nilsr"].fillna(False).astype(bool)
    pairs["feasible_other"] = pairs["feasible_other"].fillna(False).astype(bool)
    pairs["usable_nilsr"] = pairs["feasible_nilsr"] & np.isfinite(pairs["objective_nilsr"])
    pairs["usable_other"] = pairs["feasible_other"] & np.isfinite(pairs["objective_other"])
    pairs["pair_feasible"] = pairs["usable_nilsr"] & pairs["usable_other"]
    pairs["objective_diff_nilsr_minus_other"] = pairs["objective_nilsr"] - pairs["objective_other"]
    pairs["improvement_nilsr_vs_other_pct"] = 100.0 * (
        pairs["objective_other"] - pairs["objective_nilsr"]
    ) / pairs["objective_other"].replace(0.0, np.nan)
    return pairs


def _summary(pairs: pd.DataFrame, size_label: str | int) -> dict[str, object]:
    feasible = pairs[pairs["pair_feasible"]].copy()
    difference = feasible["objective_diff_nilsr_minus_other"]
    wins = int((difference < -TIE_TOLERANCE).sum())
    losses = int((difference > TIE_TOLERANCE).sum())
    ties = int((difference.abs() <= TIE_TOLERANCE).sum())
    comparison = paired_comparison(
        feasible["objective_nilsr"], feasible["objective_other"]
    ) if not feasible.empty else paired_comparison([], [])
    return {
        "size": str(size_label),
        "pairs": int(len(feasible)),
        "NILSR_wins": wins,
        "NILSR_losses": losses,
        "ties": ties,
        "mean_Z_NILSR": _safe_mean(feasible["objective_nilsr"]),
        "mean_Z_other": _safe_mean(feasible["objective_other"]),
        "median_Z_NILSR": _safe_median(feasible["objective_nilsr"]),
        "median_Z_other": _safe_median(feasible["objective_other"]),
        "mean_imp_NILSR_vs_other": _safe_mean(feasible["improvement_nilsr_vs_other_pct"]),
        "median_imp_NILSR_vs_other": _safe_median(feasible["improvement_nilsr_vs_other_pct"]),
        "wilcoxon_p": float(comparison.p_value),
        "mean_cpu_NILSR": _safe_mean(pairs["cpu_time_nilsr"]),
        "mean_cpu_other": _safe_mean(pairs["cpu_time_other"]),
        "feasibility_rate_NILSR": float(np.mean(pairs["usable_nilsr"])) if len(pairs) else float("nan"),
        "feasibility_rate_other": float(np.mean(pairs["usable_other"])) if len(pairs) else float("nan"),
    }


def _table(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = [_summary(pairs, "All")]
    for size, subset in pairs.groupby("n", sort=True):
        rows.append(_summary(subset, int(size)))
    return pd.DataFrame(rows)


def _rename_for_output(table: pd.DataFrame, comparison_method: str) -> pd.DataFrame:
    other_label = "NILS" if comparison_method == NILS_METHOD else "ALNS"
    renamed = table.rename(
        columns={
            "mean_Z_other": f"mean_Z_{other_label}",
            "median_Z_other": f"median_Z_{other_label}",
            "mean_cpu_other": f"mean_cpu_{other_label}",
            "feasibility_rate_other": f"feasibility_rate_{other_label}",
            "NILSR_losses": f"{other_label}_wins",
            "mean_imp_NILSR_vs_other": f"mean_imp_NILSR_vs_{other_label}",
            "median_imp_NILSR_vs_other": f"median_imp_NILSR_vs_{other_label}",
        }
    )
    columns = [
        "size",
        "pairs",
        "NILSR_wins",
        f"{other_label}_wins",
        "ties",
        "mean_Z_NILSR",
        f"mean_Z_{other_label}",
        f"mean_imp_NILSR_vs_{other_label}",
        f"median_imp_NILSR_vs_{other_label}",
        "wilcoxon_p",
        "mean_cpu_NILSR",
        f"mean_cpu_{other_label}",
        "feasibility_rate_NILSR",
        f"feasibility_rate_{other_label}",
    ]
    return renamed[columns]


def aggregate(run_status_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_status = pd.read_csv(run_status_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables: list[pd.DataFrame] = []
    for comparison_method, filename, detail_filename in [
        (NILS_METHOD, "table_nils_r_vs_nils_by_size.csv", "nils_r_vs_nils_pairwise_detail.csv"),
        (ALNS_METHOD, "table_nils_r_vs_alns_by_size.csv", "nils_r_vs_alns_pairwise_detail.csv"),
    ]:
        pairs = _pair_frame(run_status, comparison_method)
        table = _rename_for_output(_table(pairs), comparison_method)
        table.to_csv(output_dir / filename, index=False, float_format="%.10f")
        detail = pairs.copy()
        detail["winner"] = "not_pair_feasible"
        detail.loc[detail["pair_feasible"] & (detail["objective_diff_nilsr_minus_other"] < -TIE_TOLERANCE), "winner"] = "nils_r"
        detail.loc[detail["pair_feasible"] & (detail["objective_diff_nilsr_minus_other"] > TIE_TOLERANCE), "winner"] = comparison_method
        detail.loc[detail["pair_feasible"] & (detail["objective_diff_nilsr_minus_other"].abs() <= TIE_TOLERANCE), "winner"] = "tie"
        detail.to_csv(output_dir / detail_filename, index=False, float_format="%.10f")
        tables.append(table)
    return tables[0], tables[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-status", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    aggregate(arguments.run_status, arguments.output_dir)
