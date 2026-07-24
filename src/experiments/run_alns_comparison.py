"""Matched-subset comparison against an independent feasibility-aware ALNS."""
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
from .statistics import paired_comparison


METHOD_LABELS = {
    "truck_only": "M1 Truck-only",
    "paired_baseline": "M3 Paired baseline",
    "no_unpairing": "M4 No-unpairing",
    "nils": "M5 NILS",
    "feasibility_aware_alns": "M8 Feasibility-aware ALNS",
}


def _log(path: Path, msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _scenario_from_row(row: pd.Series) -> Dict[str, object]:
    return {
        "scenario_id": str(row["scenario_id"]),
        "n": int(row["n"]),
        "eligible_share": float(row["eligible_share_target"]),
        "drones_available": int(row["number_of_drones"]),
        "endurance": str(row["endurance_level"]),
        "speed_ratio": float(row["speed_ratio"]),
        "spatial_pattern": str(row["spatial_pattern"]),
        "handling_time": str(row["launch_retrieval_time_class"]),
        "priority_weight_profile": str(row["priority_weight_profile"]),
        "battery_slack_ratio": float(row["battery_slack_ratio"]),
        "depot_position": str(row.get("depot_position", "peripheral")),
    }


def _select_instances(
    existing: pd.DataFrame,
    *,
    scenario_limit: int,
    reps_per_scenario: int,
    seed: int,
) -> pd.DataFrame:
    meta = (
        existing.sort_values(["scenario_id", "instance_id"])
        .drop_duplicates(["scenario_id", "instance_id"])
        .copy()
    )
    scenarios = sorted(meta["scenario_id"].unique())
    if scenario_limit > 0 and scenario_limit < len(scenarios):
        indices = np.linspace(0, len(scenarios) - 1, scenario_limit, dtype=int)
        scenarios = [scenarios[int(i)] for i in indices]
    selected = []
    for sid in scenarios:
        sub = meta[meta["scenario_id"] == sid].sort_values("instance_id").head(max(1, reps_per_scenario))
        selected.append(sub)
    if not selected:
        return pd.DataFrame()
    out = pd.concat(selected, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _with_alns_settings(config: SearchConfig) -> SearchConfig:
    exp = dict(config.experiment or {})
    exp.setdefault("alns_max_iter", 80)
    exp.setdefault("alns_time_limit_seconds", 20)
    exp.setdefault("alns_max_positions_per_route", 5)
    exp.setdefault("alns_max_destroy", 8)
    exp.setdefault("battery_slack_ratio", 0.10)
    return replace(config, experiment=exp)


def _overall_table(df: pd.DataFrame) -> pd.DataFrame:
    report = df[df["reporting_included"]].copy()
    report["relative_objective_index"] = 100.0 * report["objective_for_reporting"] / report["truck_only_objective_reported"]
    out = (
        report.groupby("method_name", dropna=False)
        .agg(
            n_runs=("instance_id", "count"),
            avg_objective=("objective_for_reporting", "mean"),
            std_objective=("objective_for_reporting", "std"),
            avg_relative_objective_index=("relative_objective_index", "mean"),
            avg_improvement_pct=("improvement_vs_truck_only_reported_pct", "mean"),
            avg_drone_share=("drone_service_share", "mean"),
            avg_cpu_s=("cpu_time_seconds", "mean"),
            feasibility_rate=("final_solution_feasibility_flag", "mean"),
        )
        .reset_index()
    )
    out["method_label"] = out["method_name"].map(METHOD_LABELS).fillna(out["method_name"])
    order = {m: i for i, m in enumerate(METHOD_LABELS)}
    out["method_order"] = out["method_name"].map(order).fillna(999)
    return out.sort_values("method_order").drop(columns=["method_order"])


def _size_table(df: pd.DataFrame) -> pd.DataFrame:
    report = df[df["reporting_included"]].copy()
    report["relative_objective_index"] = 100.0 * report["objective_for_reporting"] / report["truck_only_objective_reported"]
    out = (
        report.groupby(["n", "method_name"], dropna=False)
        .agg(
            n_runs=("instance_id", "count"),
            avg_relative_objective_index=("relative_objective_index", "mean"),
            avg_improvement_pct=("improvement_vs_truck_only_reported_pct", "mean"),
            avg_objective=("objective_for_reporting", "mean"),
            avg_cpu_s=("cpu_time_seconds", "mean"),
        )
        .reset_index()
    )
    out["method_label"] = out["method_name"].map(METHOD_LABELS).fillna(out["method_name"])
    return out.sort_values(["n", "method_name"])


def _paired_table(df: pd.DataFrame) -> pd.DataFrame:
    report = df[df["reporting_included"]].copy()
    pivot = report.pivot_table(index="instance_id", columns="method_name", values="objective_for_reporting", aggfunc="first")
    rows = []
    if "nils" not in pivot.columns:
        return pd.DataFrame()
    for competitor in ["feasibility_aware_alns", "paired_baseline", "no_unpairing"]:
        if competitor not in pivot.columns:
            continue
        sub = pivot[["nils", competitor]].dropna()
        if sub.empty:
            continue
        cmp = paired_comparison(sub["nils"], sub[competitor])
        improvements = 100.0 * (sub[competitor] - sub["nils"]) / sub[competitor].replace(0.0, np.nan)
        rows.append(
            {
                "comparison": f"M5 NILS vs {METHOD_LABELS.get(competitor, competitor)}",
                "n_pairs": int(len(sub)),
                "mean_nils_improvement_pct": float(improvements.mean()),
                "median_nils_improvement_pct": float(improvements.median()),
                "test": cmp.test,
                "statistic": cmp.statistic,
                "p_value": cmp.p_value,
                "effect_size": cmp.effect_size,
                "effect_name": cmp.effect_name,
            }
        )
    return pd.DataFrame(rows)


def _completed_subset(raw: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    """Keep only instances for which every requested method has finished."""
    if raw.empty:
        return raw
    required = set(methods)
    complete_ids = []
    for instance_id, sub in raw.groupby("instance_id"):
        if required.issubset(set(sub["method_name"])):
            complete_ids.append(instance_id)
    if not complete_ids:
        return raw.iloc[0:0].copy()
    out = raw[raw["instance_id"].isin(complete_ids)].copy()
    out = out.drop_duplicates(["instance_id", "method_name"], keep="last")
    return out.reset_index(drop=True)


def _write_current_outputs(raw_rows: List[Dict[str, object]], config: SearchConfig, methods: List[str], tables_dir: Path) -> pd.DataFrame:
    """Persist raw progress and refresh current complete-instance tables."""
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(tables_dir / "alns_comparison_runs_partial.csv", index=False)
    complete = _completed_subset(raw, methods)
    if complete.empty:
        return complete
    df = _attach_relative_fields(complete)
    df = _attach_reporting_fields(df, config)
    df.to_csv(tables_dir / "alns_comparison_runs.csv", index=False)
    overall = _overall_table(df)
    size = _size_table(df)
    paired = _paired_table(df)
    overall.to_csv(tables_dir / "table_alns_comparison_overall.csv", index=False)
    size.to_csv(tables_dir / "table_alns_comparison_by_size.csv", index=False)
    paired.to_csv(tables_dir / "table_alns_pairwise_tests.csv", index=False)
    dataframe_to_latex(overall, tables_dir / "table_alns_comparison_overall.tex", "Comparison with feasibility-aware ALNS", "tab:alns_comparison")
    dataframe_to_latex(size, tables_dir / "table_alns_comparison_by_size.tex", "ALNS comparison by problem size", "tab:alns_by_size")
    dataframe_to_latex(paired, tables_dir / "table_alns_pairwise_tests.tex", "Paired tests for NILS against ALNS and hybrid benchmarks", "tab:alns_pairwise")
    return df


def run_alns_comparison(
    config: SearchConfig,
    *,
    source_output_dir: str | Path,
    output_dir: str | Path,
    scenario_limit: int = 24,
    reps_per_scenario: int = 1,
    methods: List[str] | None = None,
) -> pd.DataFrame:
    """Run M8 and selected competitors on matched final-study raw instances."""
    config = _with_alns_settings(config)
    source_dir = Path(source_output_dir)
    out_dir = Path(output_dir)
    tables_dir = out_dir / "tables"
    logs_dir = out_dir / "logs"
    raw_dir = source_dir / "raw"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    progress = logs_dir / f"alns_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    existing_path = source_dir / "tables" / "heuristic_study_runs.csv"
    existing = pd.read_csv(existing_path)
    selected = _select_instances(
        existing,
        scenario_limit=int(scenario_limit),
        reps_per_scenario=int(reps_per_scenario),
        seed=int(config.seed),
    )
    if methods is None:
        raw_methods = config.experiment.get("alns_comparison_methods", ["truck_only", "nils", "feasibility_aware_alns"])
        methods = [str(v) for v in raw_methods] if isinstance(raw_methods, list) and raw_methods else ["truck_only", "nils", "feasibility_aware_alns"]
    _log(progress, f"ALNS comparison start | instances={len(selected)} | methods={methods}")

    partial_path = tables_dir / "alns_comparison_runs_partial.csv"
    if partial_path.exists():
        partial = pd.read_csv(partial_path)
        rows = partial.to_dict("records")
        completed_keys = set(zip(partial["instance_id"].astype(str), partial["method_name"].astype(str)))
        _log(progress, f"Resuming from partial rows={len(rows)}")
    else:
        rows = []
        completed_keys = set()
    for idx, meta in selected.iterrows():
        instance_id = str(meta["instance_id"])
        scenario = _scenario_from_row(meta)
        instance = load_instance_json(raw_dir / f"{instance_id}.json")
        scenario_cfg = _with_alns_settings(_scenario_config(config, scenario))
        _log(
            progress,
            (
                f"[{idx + 1}/{len(selected)}] {instance_id} | n={scenario['n']} "
                f"d={scenario['drones_available']} E={scenario['endurance']} V={scenario['speed_ratio']} "
                f"H={scenario['handling_time']}"
            ),
        )
        for method in methods:
            key = (instance_id, method)
            if key in completed_keys:
                _log(progress, f"  {instance_id} | {method} skip existing")
                continue
            t0 = time.time()
            _log(progress, f"  {instance_id} | {method} start")
            solution = _run_method(instance, scenario_cfg, method)
            solution.run_time_seconds = float(solution.run_time_seconds or (time.time() - t0))
            row = _collect_row(instance, solution, scenario, method)
            row["cpu_time_seconds"] = float(time.time() - t0)
            rows.append(row)
            completed_keys.add(key)
            _write_current_outputs(rows, config, methods, tables_dir)
            _log(progress, f"  {instance_id} | {method} done obj={row['objective_total']:.4f} time={row['cpu_time_seconds']:.2f}s")

    df = _write_current_outputs(rows, config, methods, tables_dir)

    _log(progress, "ALNS comparison complete")
    return df


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run matched ALNS comparison subset.")
    parser.add_argument("--config", default="config/experiment_alns_comparison.yaml")
    parser.add_argument("--source-output-dir", default="outputs/doptimal_full_beta010_tables5_14_rerun2_20260601_144538")
    parser.add_argument("--output-dir", default="outputs/alns_comparison_beta010")
    parser.add_argument("--scenario-limit", type=int, default=24)
    parser.add_argument("--reps-per-scenario", type=int, default=1)
    parser.add_argument("--methods", default=None, help="Comma-separated method list. Defaults to config.")
    args = parser.parse_args(argv)
    config = load_and_build_config(args.config)
    run_alns_comparison(
        config,
        source_output_dir=args.source_output_dir,
        output_dir=args.output_dir,
        scenario_limit=args.scenario_limit,
        reps_per_scenario=args.reps_per_scenario,
        methods=[m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
