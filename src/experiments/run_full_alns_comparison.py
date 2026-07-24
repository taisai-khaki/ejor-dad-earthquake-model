"""Full-design matched NILS--ALNS comparison for reviewer response.

This runner reuses the already completed NILS/truck-only rows from the full
720-instance heuristic study and runs the independent feasibility-aware ALNS
only for missing instances. Outputs are refreshed after every ALNS run so the
experiment can be resumed safely.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from ..instance_generator import load_instance_json
from ..parameters import SearchConfig, load_and_build_config
from ..reporting.latex_export import dataframe_to_latex
from .heuristic_study import (
    _attach_relative_fields,
    _attach_reporting_fields,
    _collect_row,
    _run_method,
    _scenario_config,
)
from .run_alns_comparison import _scenario_from_row, _with_alns_settings
from .statistics import paired_comparison


ALNS_METHOD = "feasibility_aware_alns"
CORE_METHODS = ["truck_only", "simple_drone", "nils"]
TIE_TOLERANCE = 1e-6


def _log(path: Path, msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _safe_mean(values: Iterable[float]) -> float:
    arr = np.array([float(v) for v in values if pd.notna(v) and math.isfinite(float(v))], dtype=float)
    return float(np.mean(arr)) if arr.size else float("nan")


def _safe_median(values: Iterable[float]) -> float:
    arr = np.array([float(v) for v in values if pd.notna(v) and math.isfinite(float(v))], dtype=float)
    return float(np.median(arr)) if arr.size else float("nan")


def _safe_rate(values: Iterable[object]) -> float:
    arr = np.array([bool(v) for v in values], dtype=bool)
    return 100.0 * float(np.mean(arr)) if arr.size else float("nan")

def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if pd.isna(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _load_existing_alns(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = pd.read_csv(path)
    if rows.empty:
        return rows
    rows = rows[rows["method_name"].astype(str).eq(ALNS_METHOD)].copy()
    rows = rows.drop_duplicates(["instance_id", "method_name"], keep="last")
    return rows.reset_index(drop=True)


def _matched_combined_dataframe(source: pd.DataFrame, alns_rows: pd.DataFrame, config: SearchConfig) -> pd.DataFrame:
    core = source[source["method_name"].isin(CORE_METHODS)].copy()
    if not alns_rows.empty:
        missing_cols = [column for column in core.columns if column not in alns_rows.columns]
        for column in missing_cols:
            alns_rows[column] = np.nan
        extra_cols = [column for column in alns_rows.columns if column not in core.columns]
        if extra_cols:
            core = core.copy()
            for column in extra_cols:
                core[column] = np.nan
        alns_rows = alns_rows[core.columns]
        combined = pd.concat([core, alns_rows], ignore_index=True)
    else:
        combined = core

    combined = combined.drop_duplicates(["instance_id", "method_name"], keep="last")
    required = {"nils", ALNS_METHOD}
    complete_ids = []
    for instance_id, sub in combined.groupby("instance_id"):
        if required.issubset(set(sub["method_name"])):
            complete_ids.append(instance_id)
    combined = combined[combined["instance_id"].isin(complete_ids)].copy()
    derived_columns = [
        "truck_only_objective",
        "truck_only_distance",
        "simple_drone_objective",
        "improvement_vs_truck_only_pct",
        "improvement_vs_simple_drone_pct",
        "truck_distance_reduction_pct",
        "is_feasible_run",
        "reporting_policy",
        "reporting_penalty_value",
        "objective_for_reporting",
        "reporting_included",
        "truck_only_objective_reported",
        "simple_drone_objective_reported",
        "improvement_vs_truck_only_reported_pct",
        "improvement_vs_simple_drone_reported_pct",
    ]
    combined = combined.drop(columns=[column for column in derived_columns if column in combined.columns])
    combined = _attach_relative_fields(combined)
    combined = _attach_reporting_fields(combined, config)
    return combined.reset_index(drop=True)


def _pair_frame(df: pd.DataFrame) -> pd.DataFrame:
    needed = df[df["method_name"].isin(["nils", ALNS_METHOD])].copy()
    cols = [
        "instance_id",
        "scenario_id",
        "n",
        "method_name",
        "objective_for_reporting",
        "objective_total",
        "reporting_included",
        "final_solution_feasibility_flag",
        "is_feasible_run",
        "cpu_time_seconds",
    ]
    available_cols = [column for column in cols if column in needed.columns]
    pivot = needed[available_cols].pivot_table(
        index=["instance_id", "scenario_id", "n"],
        columns="method_name",
        values=[
            "objective_for_reporting",
            "objective_total",
            "reporting_included",
            "final_solution_feasibility_flag",
            "is_feasible_run",
            "cpu_time_seconds",
        ],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{method}" for metric, method in pivot.columns]
    out = pivot.reset_index()
    for method in ["nils", ALNS_METHOD]:
        included_col = f"reporting_included_{method}"
        if included_col not in out.columns:
            out[included_col] = False
        out[included_col] = out[included_col].fillna(False).astype(bool)
    out["pair_feasible"] = out["reporting_included_nils"] & out[f"reporting_included_{ALNS_METHOD}"]
    out["objective_diff_nils_minus_alns"] = out["objective_for_reporting_nils"] - out[f"objective_for_reporting_{ALNS_METHOD}"]
    out["nils_improvement_vs_alns_pct"] = 100.0 * (
        out[f"objective_for_reporting_{ALNS_METHOD}"] - out["objective_for_reporting_nils"]
    ) / out[f"objective_for_reporting_{ALNS_METHOD}"].replace(0.0, np.nan)
    return out


def _summary_for_pairs(pairs: pd.DataFrame, size_label: str | int) -> Dict[str, object]:
    feasible = pairs[pairs["pair_feasible"]].copy()
    diff = feasible["objective_diff_nils_minus_alns"]
    nils_wins = int((diff < -TIE_TOLERANCE).sum())
    alns_wins = int((diff > TIE_TOLERANCE).sum())
    ties = int((diff.abs() <= TIE_TOLERANCE).sum())

    if len(feasible) > 0:
        cmp = paired_comparison(feasible["objective_for_reporting_nils"], feasible[f"objective_for_reporting_{ALNS_METHOD}"])
    else:
        cmp = paired_comparison([], [])

    return {
        "size": str(size_label),
        "matched_instances": int(len(pairs)),
        "paired_feasible_instances": int(len(feasible)),
        "nils_feasible_pct": _safe_rate(pairs["reporting_included_nils"]),
        "alns_feasible_pct": _safe_rate(pairs[f"reporting_included_{ALNS_METHOD}"]),
        "nils_wins": nils_wins,
        "alns_wins": alns_wins,
        "ties": ties,
        "mean_nils_objective": _safe_mean(feasible["objective_for_reporting_nils"]),
        "median_nils_objective": _safe_median(feasible["objective_for_reporting_nils"]),
        "mean_alns_objective": _safe_mean(feasible[f"objective_for_reporting_{ALNS_METHOD}"]),
        "median_alns_objective": _safe_median(feasible[f"objective_for_reporting_{ALNS_METHOD}"]),
        "mean_nils_improvement_vs_alns_pct": _safe_mean(feasible["nils_improvement_vs_alns_pct"]),
        "median_nils_improvement_vs_alns_pct": _safe_median(feasible["nils_improvement_vs_alns_pct"]),
        "mean_nils_cpu_s": _safe_mean(pairs["cpu_time_seconds_nils"]),
        "median_nils_cpu_s": _safe_median(pairs["cpu_time_seconds_nils"]),
        "mean_alns_cpu_s": _safe_mean(pairs[f"cpu_time_seconds_{ALNS_METHOD}"]),
        "median_alns_cpu_s": _safe_median(pairs[f"cpu_time_seconds_{ALNS_METHOD}"]),
        "wilcoxon_test": cmp.test,
        "wilcoxon_statistic": cmp.statistic,
        "wilcoxon_p_value": cmp.p_value,
        "wilcoxon_effect_size": cmp.effect_size,
        "wilcoxon_effect_name": cmp.effect_name,
    }


def _full_matched_table(df: pd.DataFrame) -> pd.DataFrame:
    pairs = _pair_frame(df)
    rows = [_summary_for_pairs(pairs, "All")]
    for size, sub in pairs.groupby("n", sort=True):
        rows.append(_summary_for_pairs(sub, int(size)))
    return pd.DataFrame(rows)


def _wins_losses_detail(df: pd.DataFrame) -> pd.DataFrame:
    pairs = _pair_frame(df)
    out = pairs[
        [
            "instance_id",
            "scenario_id",
            "n",
            "pair_feasible",
            "objective_for_reporting_nils",
            f"objective_for_reporting_{ALNS_METHOD}",
            "objective_diff_nils_minus_alns",
            "nils_improvement_vs_alns_pct",
            "cpu_time_seconds_nils",
            f"cpu_time_seconds_{ALNS_METHOD}",
        ]
    ].copy()
    out["winner"] = "tie"
    out.loc[out["objective_diff_nils_minus_alns"] < -TIE_TOLERANCE, "winner"] = "nils"
    out.loc[out["objective_diff_nils_minus_alns"] > TIE_TOLERANCE, "winner"] = "alns"
    out.loc[~out["pair_feasible"], "winner"] = "not_pair_feasible"
    return out.sort_values(["n", "scenario_id", "instance_id"]).reset_index(drop=True)


def _write_outputs(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "alns_full_matched_runs.csv", index=False)
    pair_detail = _wins_losses_detail(df)
    full_table = _full_matched_table(df)

    pair_detail.to_csv(tables_dir / "alns_full_pairwise_detail.csv", index=False)
    full_table.to_csv(tables_dir / "table_15_full_matched_nils_alns.csv", index=False)
    dataframe_to_latex(
        full_table,
        tables_dir / "table_15_full_matched_nils_alns.tex",
        "Full matched NILS--ALNS comparison over the 720-instance design.",
        "tab:full_matched_nils_alns",
    )

    overall = full_table[full_table["size"].eq("All")].copy()
    by_size = full_table[~full_table["size"].eq("All")].copy()
    overall.to_csv(tables_dir / "table_15a_full_matched_nils_alns_overall.csv", index=False)
    by_size.to_csv(tables_dir / "table_15b_full_matched_nils_alns_by_size.csv", index=False)
    dataframe_to_latex(
        overall,
        tables_dir / "table_15a_full_matched_nils_alns_overall.tex",
        "Overall full-design matched comparison of NILS and feasibility-aware ALNS.",
        "tab:full_matched_nils_alns_overall",
    )
    dataframe_to_latex(
        by_size,
        tables_dir / "table_15b_full_matched_nils_alns_by_size.tex",
        "Size-stratified full-design matched comparison of NILS and feasibility-aware ALNS.",
        "tab:full_matched_nils_alns_by_size",
    )

    summary = [
        "# Full 720-Instance NILS--ALNS Comparison",
        "",
        "Lower objective values are better. Positive improvement means NILS improves over ALNS.",
        "Wilcoxon tests are paired on instances where both methods are feasible/reportable.",
        "Negative rank-biserial effect sizes favor NILS because the test is applied to NILS minus ALNS objectives.",
        "",
        _markdown_table(full_table),
        "",
    ]
    (tables_dir / "table_15_full_matched_nils_alns.md").write_text("\n".join(summary), encoding="utf-8")
    return full_table


def run_full_alns_comparison(
    config: SearchConfig,
    *,
    source_output_dir: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
) -> pd.DataFrame:
    config = _with_alns_settings(config)
    source_dir = Path(source_output_dir)
    out_dir = Path(output_dir)
    tables_dir = out_dir / "tables"
    logs_dir = out_dir / "logs"
    raw_dir = source_dir / "raw"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    progress = logs_dir / f"alns_full_720_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    source_path = source_dir / "tables" / "heuristic_study_runs.csv"
    source = pd.read_csv(source_path)
    meta = (
        source.sort_values(["scenario_id", "instance_id"])
        .drop_duplicates(["scenario_id", "instance_id"])
        .reset_index(drop=True)
    )
    if limit is not None and limit > 0:
        meta = meta.head(int(limit)).copy()

    partial_path = tables_dir / "alns_full_alns_runs_partial.csv"
    alns_rows = _load_existing_alns(partial_path)
    completed = set(alns_rows["instance_id"].astype(str)) if not alns_rows.empty else set()
    _log(progress, f"Full ALNS comparison start | target_instances={len(meta)} | completed_alns={len(completed)}")

    rows: List[Dict[str, object]] = alns_rows.to_dict("records") if not alns_rows.empty else []
    for idx, row in meta.iterrows():
        instance_id = str(row["instance_id"])
        if instance_id in completed:
            continue

        scenario = _scenario_from_row(row)
        scenario_cfg = _with_alns_settings(_scenario_config(config, scenario))
        instance = load_instance_json(raw_dir / f"{instance_id}.json")
        _log(
            progress,
            (
                f"[{idx + 1}/{len(meta)}] {instance_id} | n={scenario['n']} "
                f"d={scenario['drones_available']} E={scenario['endurance']} V={scenario['speed_ratio']} "
                f"H={scenario['handling_time']} | ALNS start"
            ),
        )
        start = time.time()
        solution = _run_method(instance, scenario_cfg, ALNS_METHOD)
        solution.run_time_seconds = float(solution.run_time_seconds or (time.time() - start))
        out_row = _collect_row(instance, solution, scenario, ALNS_METHOD)
        out_row["cpu_time_seconds"] = float(time.time() - start)
        rows.append(out_row)
        completed.add(instance_id)

        alns_df = pd.DataFrame(rows).drop_duplicates(["instance_id", "method_name"], keep="last")
        alns_df.to_csv(partial_path, index=False)
        current = _matched_combined_dataframe(source, alns_df, config)
        full_table = _write_outputs(current, tables_dir)
        all_row = full_table[full_table["size"].eq("All")].iloc[0]
        _log(
            progress,
            (
                f"[{idx + 1}/{len(meta)}] {instance_id} | ALNS done "
                f"obj={out_row['objective_total']:.4f} time={out_row['cpu_time_seconds']:.2f}s | "
                f"matched={int(all_row['matched_instances'])}"
            ),
        )

    alns_df = pd.DataFrame(rows).drop_duplicates(["instance_id", "method_name"], keep="last")
    alns_df.to_csv(partial_path, index=False)
    current = _matched_combined_dataframe(source, alns_df, config)
    full_table = _write_outputs(current, tables_dir)
    _log(progress, f"Full ALNS comparison complete | matched_instances={int(full_table.iloc[0]['matched_instances'])}")
    return full_table


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ALNS on the full 720-instance design and build matched NILS--ALNS tables.")
    parser.add_argument("--config", default="config/experiment_alns_comparison.yaml")
    parser.add_argument("--source-output-dir", default="outputs/doptimal_full_beta010_tables5_14_rerun2_20260601_144538")
    parser.add_argument("--output-dir", default="outputs/alns_comparison_beta010_full720")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test cap on number of instances.")
    args = parser.parse_args(argv)
    config = load_and_build_config(args.config)
    run_full_alns_comparison(
        config,
        source_output_dir=args.source_output_dir,
        output_dir=args.output_dir,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


