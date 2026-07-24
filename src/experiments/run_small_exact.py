"""Small instance runner: exact MILP + heuristic for comparison."""
from __future__ import annotations

import math
import multiprocessing as mp
import queue as pyqueue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from ..data_models import ExperimentResult, SolutionData
from ..feasibility import is_feasible
from ..heuristics import run_nils
from ..instance_generator import InstanceGenerator
from ..milp import solve_instance
from ..parameters import SearchConfig
from ..heuristics.nils import summarize_drone_usage


def _build_progress_writer(output_dir: str | None) -> Path:
    out_dir = Path(output_dir or "outputs")
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"run_small_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def _append_progress(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{line}\n")


def _progress_bar(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[------------------------] 0/0"
    safe_total = max(1, int(total))
    ratio = min(1.0, max(0.0, current / safe_total))
    fill = int(ratio * width)
    return "[" + "#" * fill + "-" * (width - fill) + f"] {current}/{safe_total}"


def _exact_worker(instance: object, config: SearchConfig, out_queue) -> None:
    def _progress(stage: str) -> None:
        out_queue.put({"kind": "stage", "stage": stage, "ts": time.time()})

    try:
        sol = solve_instance(instance, config, progress_hook=_progress)
        out_queue.put({"kind": "result", "solution": sol})
    except Exception as ex:  # pragma: no cover - exercised in runtime only
        out_queue.put({"kind": "error", "error": str(ex)})


def _safe_solve_exact(
    instance: object,
    config: SearchConfig,
    progress_path: Path,
    exp_cfg: Dict[str, object],
):
    if not config.milp.enabled:
        return None

    watchdog_enabled = bool(exp_cfg.get("small_exact_watchdog_enabled", True))
    wall_timeout_seconds = int(
        exp_cfg.get(
            "small_exact_wall_timeout_seconds",
            max(180, int(config.milp.time_limit_seconds) + 60),
        )
    )
    heartbeat_seconds = max(2, int(exp_cfg.get("small_exact_heartbeat_seconds", 20)))

    main_file = str(getattr(__import__("__main__"), "__file__", ""))
    spawn_safe = bool(main_file) and "<" not in main_file and "stdin" not in main_file.lower()

    if watchdog_enabled and not spawn_safe:
        _append_progress(
            progress_path,
            (
                f"exact_watchdog_spawn_unavailable | {getattr(instance, 'name', 'unknown')} "
                f"| main={main_file or 'none'} | using_direct_mode=1"
            ),
        )
        watchdog_enabled = False

    if not watchdog_enabled:
        stage_holder = {"stage": "startup"}
        stop_heartbeat = threading.Event()
        start = time.time()

        def _heartbeat_loop() -> None:
            while not stop_heartbeat.wait(heartbeat_seconds):
                elapsed = time.time() - start
                _append_progress(
                    progress_path,
                    (
                        f"exact_heartbeat | {getattr(instance, 'name', 'unknown')} "
                        f"| elapsed={elapsed:.1f}s | stage={stage_holder['stage']} | mode=direct"
                    ),
                )

        beat = threading.Thread(target=_heartbeat_loop, daemon=True)
        beat.start()

        def _progress(stage: str) -> None:
            stage_holder["stage"] = stage
            _append_progress(progress_path, f"exact_stage | {getattr(instance, 'name', 'unknown')} | {stage}")

        try:
            sol = solve_instance(instance, config, progress_hook=_progress)
            return sol
        except RuntimeError as ex:
            message = str(ex).lower()
            if "gurobi is not" in message or "unavailable" in message or "not installed" in message:
                return None
            if "no module named 'cplex'" in message or "cplex_py: not available" in message:
                return None
            raise
        finally:
            stop_heartbeat.set()
            beat.join(timeout=1)
            elapsed = time.time() - start
            _append_progress(
                progress_path,
                (
                    f"exact_direct_done | {getattr(instance, 'name', 'unknown')} "
                    f"| elapsed={elapsed:.2f}s | final_stage={stage_holder['stage']}"
                ),
            )

    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    proc = ctx.Process(target=_exact_worker, args=(instance, config, out_queue), daemon=True)
    proc.start()

    start = time.time()
    last_heartbeat = start
    last_stage = "spawned"
    solution = None
    error_message = None
    _append_progress(
        progress_path,
        (
            f"exact_start | {getattr(instance, 'name', 'unknown')} | solver={config.milp.solver_backend} "
            f"| wall_timeout={wall_timeout_seconds}s | pid={proc.pid}"
        ),
    )

    while True:
        got_message = False
        while True:
            try:
                msg = out_queue.get_nowait()
            except pyqueue.Empty:
                break

            got_message = True
            kind = str(msg.get("kind", ""))
            if kind == "stage":
                last_stage = str(msg.get("stage", "unknown"))
                _append_progress(progress_path, f"exact_stage | {getattr(instance, 'name', 'unknown')} | {last_stage}")
            elif kind == "result":
                solution = msg.get("solution")
            elif kind == "error":
                error_message = str(msg.get("error", "unknown error"))

        if solution is not None or error_message is not None:
            break

        if not proc.is_alive() and not got_message:
            break

        now = time.time()
        elapsed = now - start
        if elapsed > wall_timeout_seconds:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
            _append_progress(
                progress_path,
                (
                    f"exact_watchdog_timeout | {getattr(instance, 'name', 'unknown')} "
                    f"| elapsed={elapsed:.1f}s | stage={last_stage}"
                ),
            )
            return SolutionData(
                instance_name=getattr(instance, "name", "unknown"),
                status="exact_watchdog_timeout",
                objective=float("inf"),
                components={"watchdog_timeout": 1.0},
                run_time_seconds=elapsed,
            )

        if now - last_heartbeat >= heartbeat_seconds:
            _append_progress(
                progress_path,
                (
                    f"exact_heartbeat | {getattr(instance, 'name', 'unknown')} "
                    f"| elapsed={elapsed:.1f}s | stage={last_stage} | pid={proc.pid}"
                ),
            )
            last_heartbeat = now

        time.sleep(0.25)

    if proc.is_alive():
        proc.join(timeout=2)

    if solution is not None:
        return solution

    if error_message:
        message = error_message.lower()
        if (
            "gurobi is not" in message
            or "unavailable" in message
            or "not installed" in message
            or "no module named 'cplex'" in message
            or "cplex_py: not available" in message
        ):
            _append_progress(
                progress_path,
                f"exact_backend_unavailable | {getattr(instance, 'name', 'unknown')} | {error_message}",
            )
            return None
        raise RuntimeError(error_message)

    _append_progress(
        progress_path,
        f"exact_no_result | {getattr(instance, 'name', 'unknown')} | stage={last_stage}",
    )
    return SolutionData(
        instance_name=getattr(instance, "name", "unknown"),
        status="exact_no_result",
        objective=float("inf"),
        components={"no_result": 1.0},
        run_time_seconds=time.time() - start,
    )


def _to_sizes(config: SearchConfig) -> List[int]:
    explicit = config.experiment.get("run_small_sizes")
    if explicit:
        return [int(v) for v in explicit]
    if config.generation.sizes:
        return [int(v) for v in config.generation.sizes]
    return [3, 4, 5, 6, 7, 8, 9]


def _nils_seed_for_start(
    *,
    base_seed: int,
    instance_seed: int,
    restart_index: int,
    seed_stride: int,
    use_instance_seed: bool,
) -> int:
    anchor = int(instance_seed) if use_instance_seed else 0
    return int(base_seed + anchor + restart_index * seed_stride)


def _run_nils_multistart(
    instance,
    config: SearchConfig,
    exp_cfg: Dict[str, object],
    progress_path: Path,
) -> Tuple[object, Dict[str, float]]:
    restarts = max(1, int(exp_cfg.get("small_nils_restarts", 1)))
    seed_stride = max(1, int(exp_cfg.get("small_nils_seed_stride", 10_000)))
    use_instance_seed = bool(exp_cfg.get("small_nils_use_instance_seed", True))
    feasible_first = bool(exp_cfg.get("small_nils_select_feasible_first", True))
    base_seed = int(config.heuristics.random_seed)
    battery_slack_ratio = float(exp_cfg.get("small_battery_slack_ratio", exp_cfg.get("battery_slack_ratio", 0.10)))

    best_any = None
    best_feasible = None
    feasible_count = 0

    for restart in range(restarts):
        start_seed = _nils_seed_for_start(
            base_seed=base_seed,
            instance_seed=int(instance.seed),
            restart_index=restart,
            seed_stride=seed_stride,
            use_instance_seed=use_instance_seed,
        )
        candidate = run_nils(
            instance,
            seed=start_seed,
            max_iter=int(config.heuristics.max_outer_iter),
            max_no_improve=int(config.heuristics.max_no_improve),
            time_limit=int(config.heuristics.time_limit_seconds),
            strict_candidate_check_cap=int(exp_cfg.get("small_strict_candidate_check_cap", 24)),
            full_feasibility_check_interval=int(exp_cfg.get("small_full_feasibility_check_interval", 1)),
            require_feasible_incumbent=bool(exp_cfg.get("small_require_feasible_incumbent", True)),
            repair_infeasible_candidates=bool(exp_cfg.get("small_repair_infeasible_candidates", True)),
            repair_max_steps=int(exp_cfg.get("small_repair_max_steps", 4)),
            battery_slack_ratio=battery_slack_ratio,
        )
        candidate_feasible = is_feasible(instance, candidate)
        if candidate_feasible:
            feasible_count += 1
            if best_feasible is None or candidate.objective < best_feasible.objective:
                best_feasible = candidate
        if best_any is None or candidate.objective < best_any.objective:
            best_any = candidate
        _append_progress(
            progress_path,
            (
                f"heur_start_done | {instance.name} | start={restart + 1}/{restarts} "
                f"| seed={start_seed} | feasible={candidate_feasible} "
                f"| obj={candidate.objective:.4f} | time={candidate.run_time_seconds:.2f}s"
            ),
        )

    selected = best_feasible if (feasible_first and best_feasible is not None) else best_any
    if selected is None:
        raise RuntimeError("NILS multistart did not produce any candidate solution.")

    selected_feasible = is_feasible(instance, selected)
    selected.components = dict(selected.components)
    selected.components["small_nils_restarts"] = float(restarts)
    selected.components["small_nils_feasible_starts"] = float(feasible_count)
    selected.components["small_nils_selected_feasible"] = 1.0 if selected_feasible else 0.0
    selected.components["small_nils_seed_stride"] = float(seed_stride)
    selected.components["small_nils_base_seed"] = float(base_seed)
    selected.components["small_battery_slack_ratio"] = float(battery_slack_ratio)

    return selected, {
        "restarts": float(restarts),
        "feasible_starts": float(feasible_count),
        "selected_feasible": 1.0 if selected_feasible else 0.0,
    }


def run_small_exact(config: SearchConfig, output_dir: str | None = None) -> List[ExperimentResult]:
    exp_cfg = dict(config.experiment or {})
    sizes = _to_sizes(config)
    reps = int(exp_cfg.get("instance_reps_per_size", config.generation.instance_reps_per_size))

    generator = InstanceGenerator.from_search_config(config)
    target_dir = output_dir or str(exp_cfg.get("output_dir", "outputs"))
    instances = generator.generate_batch(config.seed, sizes, reps, target_dir, tag="small")
    progress_path = _build_progress_writer(target_dir)
    _append_progress(progress_path, f"run_small_exact start | instances={len(instances)} | output_dir={target_dir}")

    rows: List[ExperimentResult] = []
    table_rows = []

    for idx, inst in enumerate(instances, start=1):
        bar = _progress_bar(idx, len(instances))
        _append_progress(
            progress_path,
            f"{bar} | instance={inst.name} | customers={inst.num_customers} | drones={inst.num_drones} | trucks={inst.num_trucks}",
        )
        heuristic, heur_diag = _run_nils_multistart(
            inst,
            config=config,
            exp_cfg=exp_cfg,
            progress_path=progress_path,
        )
        heur_feasible = bool(heur_diag.get("selected_feasible", 0.0) >= 0.5)
        heuristic_usage = summarize_drone_usage(heuristic)
        _append_progress(
            progress_path,
            (
                f"heuristic_done | {inst.name} | status={heuristic.status} | obj={heuristic.objective:.4f} "
                f"| feasible={heur_feasible} | starts={int(heur_diag.get('restarts', 1.0))} "
                f"| feasible_starts={int(heur_diag.get('feasible_starts', 0.0))} "
                f"| time={heuristic.run_time_seconds:.2f}s"
            ),
        )

        exact = _safe_solve_exact(inst, config, progress_path, exp_cfg)
        exact_feasible = bool(exact and exact.status != "no_solution" and is_feasible(inst, exact))
        if exact is not None:
            _append_progress(
                progress_path,
                (
                    f"exact_done | {inst.name} | status={exact.status} | obj={exact.objective:.4f} "
                    f"| feasible={exact_feasible} | time={exact.run_time_seconds:.2f}s"
                ),
            )
        else:
            _append_progress(progress_path, f"exact_unavailable | {inst.name} | status={'exact_unavailable'}")
        exact_gap_raw = None
        exact_gap = None
        if exact is not None and math.isfinite(exact.objective) and abs(exact.objective) > 1e-9 and math.isfinite(heuristic.objective):
            exact_gap_raw = (heuristic.objective - exact.objective) / abs(exact.objective)
        if exact_gap_raw is not None and exact_feasible and heur_feasible:
            exact_gap = exact_gap_raw

        rows.append(
            ExperimentResult(
                instance_name=inst.name,
                method="heuristic",
                status=heuristic.status,
                runtime_seconds=heuristic.run_time_seconds,
                objective=float(heuristic.objective),
                gap_to_baseline=exact_gap if exact is not None else None,
                notes=(
                    "exact_enabled"
                    if (config.milp.enabled and exact is not None)
                    else ("exact_disabled" if not config.milp.enabled else "exact_unavailable")
                ),
                stats={
                    "drone_served_customers": heuristic_usage["drone_served_customers"],
                    "drone_arcs": heuristic_usage["drone_arcs"],
                    "reload_events": heuristic_usage["reload_events"],
                    "battery_swaps": heuristic_usage["battery_swaps"],
                    "nonempty_drone_routes": heuristic_usage["nonempty_drone_routes"],
                    "small_nils_restarts": float(heur_diag.get("restarts", 1.0)),
                    "small_nils_feasible_starts": float(heur_diag.get("feasible_starts", 0.0)),
                    "small_nils_selected_feasible": float(heur_diag.get("selected_feasible", 0.0)),
                    "heuristic_tardiness_cost": float(heuristic.components.get("tardiness_cost", 0.0)),
                    "heuristic_feasible": 1.0 if heur_feasible else 0.0,
                    "exact_objective": float(exact.objective) if exact else math.nan,
                    "exact_feasible": 1.0 if exact_feasible else 0.0,
                    "exact_gap_raw_pct": float(exact_gap_raw * 100.0) if exact_gap_raw is not None else math.nan,
                    "exact_gap_feasible_pct": float(exact_gap * 100.0) if exact_gap is not None else math.nan,
                },
            )
        )

        if exact is not None:
            exact_usage = summarize_drone_usage(exact)
            rows.append(
                ExperimentResult(
                    instance_name=inst.name,
                    method="exact",
                    status=exact.status,
                    runtime_seconds=exact.run_time_seconds,
                    objective=float(exact.objective),
                    stats={
                        "drone_served_customers": exact_usage["drone_served_customers"],
                        "drone_arcs": exact_usage["drone_arcs"],
                        "reload_events": exact_usage["reload_events"],
                        "battery_swaps": exact_usage["battery_swaps"],
                        "nonempty_drone_routes": exact_usage["nonempty_drone_routes"],
                        "heuristic_objective": float(heuristic.objective),
                        "exact_gap_raw_pct": float(exact_gap_raw * 100.0) if exact_gap_raw is not None else float("nan"),
                        "exact_gap_feasible_pct": float(exact_gap * 100.0) if exact_gap is not None else float("nan"),
                        "exact_feasible": 1.0 if exact_feasible else 0.0,
                    },
                )
            )
        else:
            unknown = {
                "drone_served_customers": math.nan,
                "drone_arcs": math.nan,
                "reload_events": math.nan,
                "battery_swaps": math.nan,
                "nonempty_drone_routes": math.nan,
            }
            rows.append(
                ExperimentResult(
                    instance_name=inst.name,
                    method="exact",
                    status="disabled" if not config.milp.enabled else "unavailable",
                    runtime_seconds=0.0,
                    objective=float("nan"),
                    notes="milp_disabled" if not config.milp.enabled else "milp_unavailable",
                    stats=unknown,
                )
            )

        if exact is not None:
            exact_usage = summarize_drone_usage(exact)
        else:
            exact_usage = {
                "drone_served_customers": math.nan,
                "drone_arcs": math.nan,
                "reload_events": math.nan,
                "battery_swaps": math.nan,
                "nonempty_drone_routes": math.nan,
            }

        table_rows.append(
            {
                "instance": inst.name,
                "method": "exact" if exact else "exact_unavailable",
                "exact_obj": float(exact.objective) if exact else math.nan,
                "heur_obj": float(heuristic.objective),
                "seed": inst.seed,
                "size": inst.num_customers,
                "feasible_exact": exact_feasible,
                "feasible_heur": heur_feasible,
                "comparison_valid_for_gap": bool(exact_gap is not None),
                "gap_exact_minus_heur": float(heuristic.objective - exact.objective) if exact else math.nan,
                "gap_pct_vs_exact_raw": float(exact_gap_raw * 100.0) if exact_gap_raw is not None else math.nan,
                "gap_pct_vs_exact_feasible_only": float(exact_gap * 100.0) if exact_gap is not None else math.nan,
                "runtime_exact": float(exact.run_time_seconds) if exact else 0.0,
                "runtime_heuristic": float(heuristic.run_time_seconds),
                "exact_available": bool(config.milp.enabled and exact is not None),
                "exact_status": exact.status if exact else "unavailable",
                "exact_solver_gap": float(exact.components.get("optimality_gap", math.nan)) if exact else math.nan,
                "exact_best_bound": float(exact.components.get("best_bound", math.nan)) if exact else math.nan,
                "exact_incumbent_objective": float(exact.components.get("incumbent_objective", math.nan)) if exact else math.nan,
                "nils_restarts": int(heur_diag.get("restarts", 1.0)),
                "nils_feasible_starts": int(heur_diag.get("feasible_starts", 0.0)),
                "nils_selected_feasible": bool(heur_diag.get("selected_feasible", 0.0) >= 0.5),
                "heur_drone_served_customers": heuristic_usage["drone_served_customers"],
                "heur_drone_arcs": heuristic_usage["drone_arcs"],
                "heur_reload_events": heuristic_usage["reload_events"],
                "heur_battery_swaps": heuristic_usage["battery_swaps"],
                "heur_nonempty_drone_routes": heuristic_usage["nonempty_drone_routes"],
                "exact_drone_served_customers": exact_usage["drone_served_customers"],
                "exact_drone_arcs": exact_usage["drone_arcs"],
                "exact_reload_events": exact_usage["reload_events"],
                "exact_battery_swaps": exact_usage["battery_swaps"],
                "exact_nonempty_drone_routes": exact_usage["nonempty_drone_routes"],
            }
        )
    _append_progress(progress_path, "run_small_exact complete")

    out_dir = Path(target_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table_rows).to_csv(out_dir / "small_exact_summary.csv", index=False)
    return rows
