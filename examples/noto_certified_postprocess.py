from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd

import noto_access_experiment as noto
import noto_practical_resilience_experiment as practical
from ejor_dad.certification import (
    ContinuousGridCertificate,
    GridCell,
    budget_intersecting_grid_cells,
    continuous_grid_certificate,
    validate_upper_corner_certificate_instance,
)
from ejor_dad.channels import decompose_road_retrofit_channels
from ejor_dad.checkpoint import CheckpointStore, atomic_write_text
from ejor_dad.fixed_y import evaluate_fixed_y
from ejor_dad.reproducibility import reproducibility_metadata
from ejor_dad.tv import capped_tv_profile


CERTIFICATE_VERSION = "noto-continuous-certificate-v2"
REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    design_path = output_dir / "run_design.json"
    if not design_path.is_file():
        raise FileNotFoundError(f"Run design is missing: {design_path}")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    model_args, rho_values = model_args_from_design(design, output_dir, args.workers)
    if design.get("experiment_version") != practical.EXPERIMENT_VERSION:
        raise ValueError(
            "The grid run was created by a different experiment version. "
            "Run the corrected grid sweep before postprocessing it."
        )

    for directory in [output_dir / "tables", output_dir / "logs", output_dir / "checkpoints", output_dir / "configs"]:
        directory.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "logs" / f"noto_certificate_{time.strftime('%Y%m%d_%H%M%S')}.log"
    cache = CheckpointStore(output_dir / "checkpoints")
    write_reproducibility(output_dir, design_path)
    write_status(output_dir, "running", "startup", "Continuous certificate postprocess started.", log_path)

    started = time.time()
    try:
        run_postprocess(
            output_dir=output_dir,
            cache=cache,
            log_path=log_path,
            model_args=model_args,
            rho_values=rho_values,
            workers=args.workers,
            force=args.force,
        )
        runtime = time.time() - started
        atomic_write_text(output_dir / "certificate_runtime_summary.json", json.dumps({"runtime_sec": runtime}, indent=2))
        write_status(
            output_dir,
            "completed",
            "complete",
            "Continuous certificate postprocess completed.",
            log_path,
            exit_code=0,
            extra={"certificate_version": CERTIFICATE_VERSION, "runtime_sec": runtime},
        )
    except Exception as exc:
        append_log(log_path, f"FAILED: {exc}")
        write_status(
            output_dir,
            "failed",
            "error",
            str(exc),
            log_path,
            exit_code=1,
            extra={"certificate_version": CERTIFICATE_VERSION},
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create restart-safe continuous-grid certificates and diagnostics "
            "from a corrected practical Noto grid sweep."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true", help="Recompute stored certificate upper-corner evaluations.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1.")
    return args


def model_args_from_design(
    design: dict[str, Any], output_dir: Path, workers: int
) -> tuple[argparse.Namespace, list[float]]:
    required = [
        "mode",
        "rho_values",
        "density_cap",
        "residual_failure_ratio",
        "failure_delay_reduction",
        "retrofit_budget_scale",
        "time_sensitive_fraction",
        "immediate_loss_fraction",
        "capacity_throughput_per_bed",
        "response_threshold_minutes",
        "graded_response",
        "graded_response_knots",
    ]
    missing = [name for name in required if name not in design]
    if missing:
        raise ValueError(f"Run design is missing required corrected-model fields: {missing}")
    rho_values = [float(value) for value in design["rho_values"]]
    if not rho_values or 0.0 not in rho_values:
        raise ValueError("The run design must include rho=0.")
    return (
        argparse.Namespace(
            mode=str(design["mode"]),
            density_cap=float(design["density_cap"]),
            residual_failure_ratio=float(design["residual_failure_ratio"]),
            failure_delay_reduction=float(design["failure_delay_reduction"]),
            time_sensitive_fraction=float(design["time_sensitive_fraction"]),
            immediate_loss_fraction=float(design["immediate_loss_fraction"]),
            capacity_throughput_per_bed=design["capacity_throughput_per_bed"],
            response_threshold_minutes=design["response_threshold_minutes"],
            graded_response=bool(design["graded_response"]),
            retrofit_budget_scale=float(design["retrofit_budget_scale"]),
            output_dir=str(output_dir),
            workers=workers,
            force=False,
        ),
        sorted(set(rho_values)),
    )


def run_postprocess(
    *,
    output_dir: Path,
    cache: CheckpointStore,
    log_path: Path,
    model_args: argparse.Namespace,
    rho_values: Sequence[float],
    workers: int,
    force: bool,
) -> None:
    grid = noto.GRID_LEVELS[model_args.mode]
    template, _ = practical.build_instance(0.0, model_args)
    total_grid = int(grid.size ** len(template.links))
    certificate_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []

    for rho in rho_values:
        block_started = time.time()
        instance, _ = practical.build_instance(rho, model_args)
        validate_upper_corner_certificate_instance(instance)
        experiment_id = practical.experiment_key(model_args, rho)
        append_log(log_path, f"rho={rho:.2f}: loading exact finite-grid evaluations.")
        results, infeasible = noto.evaluate_grid(
            instance=instance,
            grid=grid,
            experiment_id=experiment_id,
            cache=cache,
            force=False,
            out=output_dir,
            log_path=log_path,
            mode=model_args.mode,
            rho=rho,
            total_grid=total_grid,
            workers=workers,
        )
        if not results:
            raise RuntimeError(f"No grid-feasible policies were available for rho={rho:.2f}.")
        results.sort(key=lambda result: result.objective)
        best = results[0]
        certificate, cells, reused_grid_evaluations, newly_evaluated = evaluate_certificate_cells(
            instance=instance,
            grid=grid,
            grid_results=results,
            grid_upper_bound=float(best.objective),
            cache=cache,
            experiment_id=experiment_id,
            output_dir=output_dir,
            log_path=log_path,
            mode=model_args.mode,
            rho=rho,
            workers=workers,
            force=force,
        )
        certificate_rows.append(certificate_row(certificate, rho, len(results), infeasible, reused_grid_evaluations, newly_evaluated))
        channel_rows.append(channel_row(instance, best, rho))
        profile_rows.extend(radius_profile_rows(instance, best, rho))
        replay_rows.append(replay_row(instance, best, rho))
        write_tables(output_dir, certificate_rows, channel_rows, profile_rows, replay_rows)
        append_log(
            log_path,
            f"rho={rho:.2f}: certificate complete; continuous LB={certificate.continuous_lower_bound:.6f}, "
            f"grid UB={certificate.grid_upper_bound:.6f}.",
        )
        write_status(
            output_dir,
            "running",
            "rho_complete",
            f"Completed continuous certificate for rho={rho:.2f}.",
            log_path,
            extra={
                "certificate_version": CERTIFICATE_VERSION,
                "rho": rho,
                "completed_rho_values": [row["rho"] for row in certificate_rows],
                "runtime_sec_current_rho": time.time() - block_started,
            },
        )


def evaluate_certificate_cells(
    *,
    instance: Any,
    grid: np.ndarray,
    grid_results: Sequence[SimpleNamespace],
    grid_upper_bound: float,
    cache: CheckpointStore,
    experiment_id: str,
    output_dir: Path,
    log_path: Path,
    mode: str,
    rho: float,
    workers: int,
    force: bool,
) -> tuple[ContinuousGridCertificate, tuple[GridCell, ...], int, int]:
    cells = budget_intersecting_grid_cells(instance.retrofit_costs, instance.budget_retrofit, grid)
    results_by_y = {noto.hash_array(np.asarray(result.y, dtype=float)): result for result in grid_results}
    results_by_cell: dict[int, SimpleNamespace] = {}
    pending: list[tuple[GridCell, str]] = []
    reused_grid_evaluations = 0

    for cell in cells:
        y_hash = noto.hash_array(cell.upper)
        if y_hash in results_by_y:
            results_by_cell[cell.index] = results_by_y[y_hash]
            reused_grid_evaluations += 1
            continue
        key = noto.versioned_key(
            f"{experiment_id}__certificate_upper_{cell.index:04d}_{y_hash}"
        )
        if not force and cache.exists(key):
            results_by_cell[cell.index] = noto.payload_to_result(cache.load(key))
        else:
            pending.append((cell, key))

    if workers == 1:
        for cell, key in pending:
            payload = evaluate_certificate_upper_payload(instance, cell)
            cache.save(key, payload)
            results_by_cell[cell.index] = noto.payload_to_result(payload)
            report_certificate_progress(
                output_dir, log_path, mode, rho, len(results_by_cell), len(cells), workers
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(evaluate_certificate_upper_payload, instance, cell): (cell, key)
                for cell, key in pending
            }
            for future in as_completed(futures):
                cell, key = futures[future]
                payload = future.result()
                cache.save(key, payload)
                results_by_cell[cell.index] = noto.payload_to_result(payload)
                report_certificate_progress(
                    output_dir, log_path, mode, rho, len(results_by_cell), len(cells), workers
                )

    ordered_objectives = [float(results_by_cell[cell.index].objective) for cell in cells]
    certificate = continuous_grid_certificate(cells, ordered_objectives, grid_upper_bound)
    return certificate, cells, reused_grid_evaluations, len(pending)


def evaluate_certificate_upper_payload(instance: Any, cell: GridCell) -> dict[str, Any]:
    started = time.time()
    result = evaluate_fixed_y(
        instance,
        cell.upper,
        epsilon=1e-5,
        max_iterations=160,
        enforce_retrofit_budget=False,
    )
    return noto.fixed_result_payload(result) | {
        "certificate_cell_index": cell.index,
        "certificate_lower_y": cell.lower.tolist(),
        "certificate_upper_y": cell.upper.tolist(),
        "certificate_lower_budget_used": cell.lower_budget_used,
        "certificate_upper_budget_used": cell.upper_budget_used,
        "certificate_eval_runtime_sec": time.time() - started,
    }


def report_certificate_progress(
    output_dir: Path,
    log_path: Path,
    mode: str,
    rho: float,
    completed: int,
    total: int,
    workers: int,
) -> None:
    if completed % 25 != 0 and completed != total:
        return
    message = f"rho={rho:.2f}: certified {completed}/{total} upper-corner cells with {workers} workers."
    append_log(log_path, message)
    write_status(
        output_dir,
        "running",
        "continuous_certificate",
        message,
        log_path,
        extra={"certificate_version": CERTIFICATE_VERSION, "rho": rho, "evaluated_cells": completed, "total_cells": total},
    )


def certificate_row(
    certificate: ContinuousGridCertificate,
    rho: float,
    feasible_grid_policies: int,
    infeasible_grid_policies: int,
    reused_grid_evaluations: int,
    newly_evaluated: int,
) -> dict[str, Any]:
    cell = certificate.lower_bound_cell
    return {
        "rho": rho,
        "grid_upper_bound": certificate.grid_upper_bound,
        "continuous_lower_bound": certificate.continuous_lower_bound,
        "absolute_gap": certificate.absolute_gap,
        "relative_gap_percent": certificate.relative_gap_percent,
        "evaluated_cell_count": certificate.evaluated_cell_count,
        "feasible_grid_policies": feasible_grid_policies,
        "infeasible_grid_policies": infeasible_grid_policies,
        "reused_feasible_grid_corner_evaluations": reused_grid_evaluations,
        "new_or_replayed_upper_corner_evaluations": newly_evaluated,
        "lower_bound_cell_index": cell.index,
        "lower_bound_cell_lower_y_json": json.dumps(cell.lower.tolist()),
        "lower_bound_cell_upper_y_json": json.dumps(cell.upper.tolist()),
        "lower_bound_cell_lower_budget_used": cell.lower_budget_used,
        "lower_bound_cell_upper_budget_used": cell.upper_budget_used,
        "certificate_scope": "complete no-tail independent states; monotone timely-access response; capped-TV without moment envelope",
    }


def channel_row(instance: Any, best: SimpleNamespace, rho: float) -> dict[str, Any]:
    channels = decompose_road_retrofit_channels(instance, best.z, best.w, best.y)
    return {
        "rho": rho,
        "selected_y_json": json.dumps(best.y.tolist()),
        "selected_z_json": json.dumps(best.z.tolist()),
        "selected_w_json": json.dumps(best.w.tolist()),
        "no_retrofit_fixed_zw_objective": channels.no_retrofit.objective,
        "conditional_consequence_only_objective": channels.conditional_consequence.objective,
        "decision_dependent_probability_objective": channels.decision_dependent_probability.objective,
        "conditional_consequence_improvement": channels.conditional_consequence_improvement,
        "probability_channel_improvement": channels.probability_channel_improvement,
        "total_fixed_zw_road_improvement": channels.total_road_improvement,
        "decomposition_error": channels.total_road_improvement
        - channels.conditional_consequence_improvement
        - channels.probability_channel_improvement,
    }


def radius_profile_rows(instance: Any, best: SimpleNamespace, rho: float) -> list[dict[str, Any]]:
    profile = capped_tv_profile(
        best.nominal_distribution,
        best.state_losses,
        maximize=True,
        density_cap=instance.ambiguity_density_cap,
    )
    profile_value = profile.evaluate(rho).value
    rows: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(profile.segments, start=1):
        rows.append(
            {
                "rho_selected": rho,
                "plan_y_json": json.dumps(best.y.tolist()),
                "plan_z_json": json.dumps(best.z.tolist()),
                "plan_w_json": json.dumps(best.w.tolist()),
                "segment_index": segment_index,
                "rho_start": segment.start_radius,
                "rho_end": segment.end_radius,
                "source_state": instance.states[segment.source_index].id,
                "target_state": instance.states[segment.target_index].id,
                "value_slope": segment.value_slope,
                "value_start": segment.start_value,
                "value_end": segment.end_value,
                "terminal_radius": profile.terminal_radius,
                "terminal_value": profile.terminal_value,
                "profile_value_at_selected_rho": profile_value,
                "fixed_y_objective": best.objective,
                "profile_match_error": profile_value - best.objective,
                "interpretation": "Fixed complete plan; this profile does not assert reoptimized values between reported radii.",
            }
        )
    if rows:
        return rows
    return [
        {
            "rho_selected": rho,
            "plan_y_json": json.dumps(best.y.tolist()),
            "plan_z_json": json.dumps(best.z.tolist()),
            "plan_w_json": json.dumps(best.w.tolist()),
            "segment_index": 0,
            "rho_start": 0.0,
            "rho_end": 0.0,
            "source_state": None,
            "target_state": None,
            "value_slope": 0.0,
            "value_start": profile.terminal_value,
            "value_end": profile.terminal_value,
            "terminal_radius": profile.terminal_radius,
            "terminal_value": profile.terminal_value,
            "profile_value_at_selected_rho": profile_value,
            "fixed_y_objective": best.objective,
            "profile_match_error": profile_value - best.objective,
            "interpretation": "Fixed complete plan; this profile does not assert reoptimized values between reported radii.",
        }
    ]


def replay_row(instance: Any, best: SimpleNamespace, rho: float) -> dict[str, Any]:
    replay = evaluate_fixed_y(instance, best.y, epsilon=1e-8, max_iterations=240)
    return {
        "rho": rho,
        "selected_y_json": json.dumps(best.y.tolist()),
        "cached_grid_objective": best.objective,
        "fresh_replay_objective": replay.objective,
        "absolute_objective_difference": abs(replay.objective - best.objective),
        "fresh_replay_lower_bound": replay.lower_bound,
        "fresh_replay_gap": replay.objective - replay.lower_bound,
        "fresh_replay_iterations": replay.iterations,
    }


def write_reproducibility(output_dir: Path, design_path: Path) -> None:
    prepared = REPO_ROOT / "data_work" / "noto" / "prepared"
    metadata = reproducibility_metadata(
        input_files=[
            design_path,
            prepared / "noto_zones.csv",
            prepared / "noto_centers.csv",
            prepared / "noto_corridors.csv",
        ],
        source_files=[
            Path(__file__),
            Path(noto.__file__),
            Path(practical.__file__),
            REPO_ROOT / "src" / "ejor_dad" / "model.py",
            REPO_ROOT / "src" / "ejor_dad" / "recourse.py",
            REPO_ROOT / "src" / "ejor_dad" / "fixed_y.py",
            REPO_ROOT / "src" / "ejor_dad" / "tv.py",
            REPO_ROOT / "src" / "ejor_dad" / "certification.py",
            REPO_ROOT / "src" / "ejor_dad" / "channels.py",
        ],
    )
    metadata["certificate_version"] = CERTIFICATE_VERSION
    metadata["certificate_method"] = "budget-intersecting grid cells evaluated at monotonic upper corners"
    atomic_write_text(output_dir / "reproducibility.json", json.dumps(metadata, indent=2, ensure_ascii=False))


def write_tables(
    output_dir: Path,
    certificate_rows: list[dict[str, Any]],
    channel_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
) -> None:
    noto.write_table(pd.DataFrame(certificate_rows), output_dir, "table_noto_continuous_certificate")
    noto.write_table(pd.DataFrame(channel_rows), output_dir, "table_noto_road_channels")
    noto.write_table(pd.DataFrame(profile_rows), output_dir, "table_noto_fixed_plan_radius_profile")
    noto.write_table(pd.DataFrame(replay_rows), output_dir, "table_noto_fresh_replay_verification")


def append_log(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def write_status(
    output_dir: Path,
    status: str,
    block: str,
    message: str,
    log_path: Path,
    exit_code: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "block": block,
        "message": message,
        "pid": os.getpid(),
        "updated_at_epoch": time.time(),
        "log_path": str(log_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "checkpoint_dir": str((output_dir / "checkpoints").resolve()),
    }
    if exit_code is not None:
        payload["exit_code"] = exit_code
    if extra:
        payload.update(extra)
    atomic_write_text(output_dir / "certificate_status.json", json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
