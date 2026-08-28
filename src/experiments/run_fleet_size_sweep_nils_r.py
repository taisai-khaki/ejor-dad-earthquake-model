from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..feasibility import is_feasible
from ..heuristics.nils import _repair_infeasible_solution, evaluate_solution
from ..heuristics.nils_r import compute_priority_metrics
from ..instance_generator import load_instance_json
from ..parameters import load_and_build_config
from .heuristic_study import _run_method, _scenario_config
from .run_alns_comparison import _scenario_from_row, _with_alns_settings
from .run_priority_service_speed_by_method import _delivery_times


OUTPUT_COLUMNS = [
    "instance_id",
    "scenario_class_id",
    "seed",
    "size",
    "available_drones",
    "active_drones_used",
    "drone_served_customers",
    "drone_share",
    "objective",
    "priority_tardiness_cost",
    "high_priority_tardiness",
    "p90_high_priority_tardiness",
    "high_priority_on_time_rate",
    "mean_high_priority_delivery_time",
    "cpu",
    "feasible",
    "error",
    "reused_original_fleet_run",
]


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _variant(instance, config, available_drones: int):
    template = instance.drones[0]
    drones = [replace(template, drone_id=drone_id) for drone_id in range(1, available_drones + 1)]
    metadata = dict(instance.metadata)
    metadata["number_of_drones"] = float(available_drones)
    return replace(instance, drones=drones, metadata=metadata)


def _high_priority_summary(instance, solution) -> tuple[float, float]:
    delivered = _delivery_times(instance, solution)
    high = [customer for customer in instance.customers if customer.priority == "high"]
    values = [float(delivered[customer.node_id]) for customer in high if customer.node_id in delivered]
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.mean([
        float(delivered[customer.node_id]) <= float(customer.ub) + 1e-9
        for customer in high
        if customer.node_id in delivered
    ]))


def _active_drones(solution) -> int:
    return int(sum(1 for route in solution.drone_routes.values() if len(route) > 2))


def _run_variant(instance, scenario, base_config, available_drones: int) -> dict[str, object]:
    config = _scenario_config(base_config, {**scenario, "drones_available": max(1, available_drones)})
    variant = _variant(instance, config, available_drones)
    start = time.time()
    method_name = "truck_only" if available_drones == 0 else "nils_r"
    solution = _run_method(variant, config, method_name)
    if not is_feasible(variant, solution):
        repaired = _repair_infeasible_solution(
            variant,
            solution,
            max_steps=max(1, int(config.experiment.get("study_output_repair_max_steps", 8))),
        )
        if repaired is not None and is_feasible(variant, repaired):
            solution = repaired
    solution.objective, solution.components = evaluate_solution(variant, solution)
    feasible = bool(is_feasible(variant, solution))
    elapsed = time.time() - start
    metrics = compute_priority_metrics(variant, solution) if feasible else {}
    mean_delivery, on_time = _high_priority_summary(variant, solution) if feasible else (float("nan"), float("nan"))
    return {
        "available_drones": available_drones,
        "active_drones_used": _active_drones(solution) if feasible else 0,
        "drone_served_customers": len(solution.served_by_drone()) if feasible else 0,
        "drone_share": len(solution.served_by_drone()) / max(1, variant.num_customers) if feasible else float("nan"),
        "objective": float(solution.objective) if feasible else float("nan"),
        "priority_tardiness_cost": float(metrics.get("priority_tardiness_cost", float("nan"))),
        "high_priority_tardiness": float(metrics.get("high_priority_tardiness", float("nan"))),
        "p90_high_priority_tardiness": float(metrics.get("p90_high_priority_tardiness", float("nan"))),
        "high_priority_on_time_rate": on_time,
        "mean_high_priority_delivery_time": mean_delivery,
        "cpu": elapsed,
        "feasible": feasible,
        "error": "",
        "reused_original_fleet_run": False,
    }


def _load_partial(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    frame = pd.read_csv(path)
    for column in OUTPUT_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    return frame[OUTPUT_COLUMNS].copy()


def run_pipeline(
    *,
    main_config_path: Path,
    source_output_dir: Path,
    output_dir: Path,
    original_report_dir: Path | None = None,
    start_index: int = 0,
    end_index: int | None = None,
) -> pd.DataFrame:
    config = load_and_build_config(str(main_config_path))
    source = pd.read_csv(source_output_dir / "tables" / "heuristic_study_runs.csv")
    meta = source.sort_values(["scenario_id", "instance_id"]).drop_duplicates(["scenario_id", "instance_id"]).reset_index(drop=True)
    meta = meta.iloc[int(start_index):int(end_index) if end_index is not None else None].reset_index(drop=True)
    raw_dir = source_output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fleet_size_sweep_nils_r_partial.csv"
    frame = _load_partial(output_path)
    completed = set(zip(frame["instance_id"].astype(str), frame["available_drones"].astype(str)))
    original_status = None
    original_observations = None
    original_detail = None
    if original_report_dir is not None:
        original_status = pd.read_csv(original_report_dir / "run_status_full.csv")
        original_status = original_status[original_status["method_name"].eq("nils_r")].copy()
        original_observations = pd.read_csv(original_report_dir / "customer_delivery_observations.csv")
        original_observations = original_observations[original_observations["method_name"].eq("nils_r")].copy()
        original_detail = pd.read_csv(original_report_dir / "customer_service_detail.csv")
        original_detail = original_detail[original_detail["method"].eq("M9 NILS-R")].copy()

    for index, source_row in meta.iterrows():
        instance_id = str(source_row["instance_id"])
        instance = load_instance_json(raw_dir / f"{instance_id}.json")
        scenario = _scenario_from_row(source_row)
        rows: list[dict[str, object]] = []
        original_drones = int(source_row["number_of_drones"])
        for available_drones in range(4):
            key = (instance_id, str(available_drones))
            if key in completed:
                continue
            if available_drones == original_drones and original_status is not None:
                status = original_status[original_status["instance_id"].astype(str).eq(instance_id)]
                obs = original_observations[original_observations["instance_id"].astype(str).eq(instance_id)]
                detail = original_detail[original_detail["instance_id"].astype(str).eq(instance_id)]
                if len(status) == 1:
                    status_row = status.iloc[0]
                    high_obs = obs
                    high_obs = high_obs[high_obs["customer_id"].isin(
                        [customer.node_id for customer in instance.customers if customer.priority == "high"]
                    )]
                    active = detail[detail["is_drone_served"].astype(str).str.lower().isin({"true", "1"})]["drone_id"].nunique()
                    rows.append({
                        "available_drones": available_drones,
                        "active_drones_used": int(active),
                        "drone_served_customers": int(status_row["drone_served_customers"]),
                        "drone_share": float(status_row["drone_share"]),
                        "objective": float(status_row["objective"]),
                        "priority_tardiness_cost": float(status_row["priority_tardiness_cost"]),
                        "high_priority_tardiness": float(status_row["high_priority_tardiness"]),
                        "p90_high_priority_tardiness": float(status_row["p90_high_priority_tardiness"]),
                        "high_priority_on_time_rate": float(high_obs["on_time"].mean()),
                        "mean_high_priority_delivery_time": float(high_obs["delivery_time"].mean()),
                        "cpu": float(status_row["cpu_time_seconds"]),
                        "feasible": _bool(status_row["feasible"]),
                        "error": "",
                        "reused_original_fleet_run": True,
                    })
                    continue
            try:
                rows.append(_run_variant(instance, scenario, config, available_drones))
            except Exception as error:
                rows.append({
                    "available_drones": available_drones,
                    "active_drones_used": 0,
                    "drone_served_customers": 0,
                    "drone_share": float("nan"),
                    "objective": float("nan"),
                    "priority_tardiness_cost": float("nan"),
                    "high_priority_tardiness": float("nan"),
                    "p90_high_priority_tardiness": float("nan"),
                    "high_priority_on_time_rate": float("nan"),
                    "mean_high_priority_delivery_time": float("nan"),
                    "cpu": float("nan"),
                    "feasible": False,
                    "error": repr(error),
                    "reused_original_fleet_run": False,
                })
        if rows:
            for row in rows:
                row.update({
                    "instance_id": instance_id,
                    "scenario_class_id": str(source_row["scenario_class"]),
                    "seed": int(instance.seed),
                    "size": int(instance.num_customers),
                })
            frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
            frame = frame.drop_duplicates(["instance_id", "available_drones"], keep="last")
            frame.to_csv(output_path, index=False, float_format="%.10f")
            print(f"[{index + 1}/{len(meta)}] {instance_id} rows={len(rows)}", flush=True)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-config", type=Path, default=Path("config/experiment_doptimal_beta010_tables5_14.yaml"))
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original-report-dir", type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    args = parser.parse_args()
    frame = run_pipeline(
        main_config_path=args.main_config,
        source_output_dir=args.source_output_dir,
        output_dir=args.output_dir,
        original_report_dir=args.original_report_dir,
        start_index=args.start_index,
        end_index=args.end_index,
    )
    print(f"Fleet sweep complete | rows={len(frame)} | output={args.output_dir}")


if __name__ == "__main__":
    main()


