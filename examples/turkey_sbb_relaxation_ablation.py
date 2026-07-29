from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ejor_dad.checkpoint import CheckpointStore, atomic_write_dataframe, atomic_write_text
from ejor_dad.sbb import (
    build_variable_index,
    root_node,
    solve_node_with_cut_separation,
    valid_objective_lower_bound,
)

import turkey_revised_experiments as revised


EXPERIMENT_VERSION = "turkey-sbb-relaxation-ablation-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare valid SBB probability relaxations on the static Turkey no-tail "
            "instance. Each root LP is checkpointed independently."
        )
    )
    parser.add_argument("--base", type=Path, default=Path("data_work/turkey"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_work/turkey/sbb_relaxation_ablation"),
    )
    parser.add_argument("--rho", type=float, default=0.10)
    parser.add_argument(
        "--link-counts",
        default="2,3,4,5",
        help="Comma-separated ladder rungs. The m=5 row runs all variants.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_link_counts(value: str) -> list[int]:
    counts = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not counts or any(count < 1 for count in counts):
        raise ValueError("--link-counts must contain positive integers.")
    return counts


def write_status(output: Path, phase: str, message: str, **extra: Any) -> None:
    payload = {
        "experiment_version": EXPERIMENT_VERSION,
        "status": "running",
        "phase": phase,
        "message": message,
        "updated_at_epoch": time.time(),
        **extra,
    }
    atomic_write_text(output / "status.json", json.dumps(payload, indent=2))


def grid_reference(base: Path, rho: float) -> dict[str, Any]:
    path = base / "revised_experiments" / "tables" / "table_discretized_m5_summary.csv"
    if not path.exists():
        return {
            "available": False,
            "source": str(path),
            "message": "No existing m=5 grid enumeration was found.",
        }
    data = pd.read_csv(path)
    matches = data[np.isclose(data["rho"].astype(float), rho)]
    if matches.empty:
        return {
            "available": False,
            "source": str(path),
            "message": f"No existing m=5 grid row was found for rho={rho:.2f}.",
        }
    row = matches.iloc[0]
    return {
        "available": True,
        "source": str(path),
        "objective": float(row["best_discretized_objective"]),
        "y_json": str(row["best_y_json"]),
        "feasible_grid_policies": int(row["feasible_evaluated_candidates"]),
    }


def root_payload(
    instance,
    probability_relaxation: str,
    link_count: int,
    rho: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    node = root_node(instance, probability_relaxation=probability_relaxation)
    cut_sets = {state_index: [] for state_index in range(len(instance.states))}
    relaxation = solve_node_with_cut_separation(
        instance,
        node,
        cut_sets,
        epsilon_cut=1e-7,
        max_cut_iterations=100,
        probability_relaxation=probability_relaxation,
    )
    elapsed = time.perf_counter() - started
    index = build_variable_index(instance, probability_relaxation=probability_relaxation)
    raw_bound = float(relaxation.lower_bound) if relaxation.success else np.nan
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "link_count": link_count,
        "num_states": len(instance.states),
        "rho": rho,
        "probability_relaxation": probability_relaxation,
        "success": bool(relaxation.success),
        "raw_root_lower_bound": raw_bound,
        "reported_root_lower_bound": (
            valid_objective_lower_bound(raw_bound) if relaxation.success else np.nan
        ),
        "root_bound_materially_positive": bool(
            relaxation.success and valid_objective_lower_bound(raw_bound) > 0.01 * instance.d_max
        ),
        "node_lp_variables": int(index.size),
        "product_tree_auxiliaries": int(len(index.v)),
        "recourse_cut_count": int(sum(len(cuts) for cuts in cut_sets.values())),
        "cut_iterations": int(relaxation.cut_iterations),
        "runtime_sec": float(elapsed),
        "solver_message": relaxation.message,
    }


def write_report(
    output: Path,
    table: pd.DataFrame,
    reference: dict[str, Any],
    link_counts: list[int],
    rho: float,
) -> None:
    lines = [
        "# Static SBB relaxation ablation",
        "",
        f"- Version: {EXPERIMENT_VERSION}.",
        f"- Static no-tail Turkey benchmark at rho={rho:.2f}.",
        "- This assesses continuous-SBB bounds only; it does not certify the newer conditional-delay model.",
        "- product_tree is V1, corner_boxes is V2, and corner_link_cuts is V3.",
        "",
        "## Grid reference",
        "",
    ]
    if reference["available"]:
        lines.extend(
            [
                f"- Existing m=5 grid objective: {reference['objective']:.6f}.",
                f"- Existing grid policy: {reference['y_json']}.",
                f"- Feasible grid policies: {reference['feasible_grid_policies']}.",
            ]
        )
    else:
        lines.append(f"- {reference['message']}")
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "- A nonnegative root bound only removes the sign error.",
            "- A materially positive V3 root bound is necessary before an equal-time five-link SBB rerun is worthwhile.",
            "- V2/V3 are valid relaxations but are not automatically tighter than the product tree; compare the recorded bounds.",
            "",
            "## Completed tasks",
            "",
        ]
    )
    for _, row in table.sort_values(["link_count", "probability_relaxation"]).iterrows():
        lines.append(
            "- "
            f"m={int(row.link_count)}, {row.probability_relaxation}: "
            f"root LB={row.reported_root_lower_bound:.6f}, "
            f"raw={row.raw_root_lower_bound:.6f}, "
            f"variables={int(row.node_lp_variables)}, "
            f"seconds={row.runtime_sec:.3f}."
        )
    missing = [count for count in link_counts if count not in table["link_count"].tolist()]
    if missing:
        lines.append(f"- Pending ladder rungs: {missing}.")
    atomic_write_text(output / "report.md", "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    link_counts = parse_link_counts(args.link_counts)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    cache = CheckpointStore(output / "checkpoints")
    reference = grid_reference(args.base, args.rho)

    atomic_write_text(
        output / "run_config.json",
        json.dumps(
            {
                "experiment_version": EXPERIMENT_VERSION,
                "base": str(args.base),
                "output": str(output),
                "rho": args.rho,
                "link_counts": link_counts,
                "python": sys.version,
                "platform": platform.platform(),
                "static_model_only": True,
                "notes": (
                    "Product tree is the current SBB relaxation. Corner boxes and "
                    "corner link cuts are benchmark ablations, not replacement claims."
                ),
            },
            indent=2,
        ),
    )
    write_status(output, "startup", "Loading cached Turkey context.", completed=0)

    context = revised.load_context(
        args.base,
        args.base / "revised_experiments",
        force_routes=False,
    )
    rows: list[dict[str, Any]] = []
    tasks = [(count, "product_tree") for count in link_counts]
    if 5 in link_counts:
        tasks.extend([(5, "corner_boxes"), (5, "corner_link_cuts")])

    for task_index, (link_count, probability_relaxation) in enumerate(tasks, start=1):
        key = (
            f"{EXPERIMENT_VERSION}__root_m{link_count}_rho{args.rho:.2f}_"
            f"{probability_relaxation}"
        )
        write_status(
            output,
            "root_relaxation",
            f"Solving root LP for m={link_count}, {probability_relaxation}.",
            current_task=task_index,
            total_tasks=len(tasks),
            completed=len(rows),
            link_count=link_count,
            probability_relaxation=probability_relaxation,
        )
        bundle = revised.build_bundle(
            context,
            "exposure_dominant",
            "full_state_no_tail",
            rho=args.rho,
            link_count=link_count,
            experiment_id=f"root_m{link_count}_{probability_relaxation}",
        )
        payload = cache.get_or_compute(
            key,
            lambda bundle=bundle, probability_relaxation=probability_relaxation: root_payload(
                bundle.instance,
                probability_relaxation,
                link_count,
                args.rho,
            ),
            force=args.force,
        )
        rows.append(payload)
        table = pd.DataFrame(rows)
        atomic_write_dataframe(table, output / "root_bounds.csv")
        write_report(output, table, reference, link_counts, args.rho)

    write_status(
        output,
        "complete",
        "All requested root relaxations completed.",
        completed=len(rows),
        total_tasks=len(tasks),
    )
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    status["status"] = "completed"
    atomic_write_text(output / "status.json", json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
