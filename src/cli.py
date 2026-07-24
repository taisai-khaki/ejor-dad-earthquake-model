"""Single command entry-point for the full reproducibility pipeline."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .parameters import SearchConfig, load_and_build_config
from .experiments import (
    run_ablation,
    run_baseline_comparison,
    run_heuristic_study,
    run_large_heuristic,
    run_sensitivity_study,
    run_small_exact,
)
from .reporting.latex_export import dataframe_to_latex



def _now_stamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _timestamped_log_path(output_dir: Path) -> Path:
    output_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"study_run_{stamp}.json"


def _load_config(path: str | None) -> SearchConfig:
    if path is None:
        return load_and_build_config(Path("config") / "default.yaml")
    return load_and_build_config(path)


def _persist_rows(name: str, payload: Any, output_dir: Path) -> None:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "to_csv"):
        payload.to_csv(table_dir / f"{name}.csv", index=False)
    elif isinstance(payload, list):
        import pandas as pd

        rows = [asdict(item) if not isinstance(item, dict) else item for item in payload]
        pd.DataFrame(rows).to_csv(table_dir / f"{name}.csv", index=False)


def _write_log(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_command_log(command: str, status: str, *, runtime_seconds: float | None = None, error: str | None = None) -> Dict[str, Any]:
    return {
        "run_id": _now_stamp(),
        "command": command,
        "status": status,
        "runtime_seconds": runtime_seconds,
        "error": error,
    }


def run_pipeline(config: SearchConfig, output_dir: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    small = run_small_exact(config, output_dir=str(output_dir))
    baseline = run_baseline_comparison(config, output_dir=str(output_dir))
    large = run_large_heuristic(config, output_dir=str(output_dir))
    sensitivity = run_sensitivity_study(config, output_dir=str(output_dir))
    ablation = run_ablation(config, output_dir=str(output_dir))

    _persist_rows("small_experiment", small, output_dir)
    _persist_rows("baseline_comparison", baseline, output_dir)
    _persist_rows("large_heuristic", large, output_dir)
    _persist_rows("sensitivity_summary", sensitivity, output_dir)
    _persist_rows("ablation_summary", ablation, output_dir)

    try:
        import pandas as pd

        small_rows = [asdict(row) for row in small]
        small_df = pd.DataFrame(small_rows)
        agg = small_df.groupby("instance_name")["objective"].agg(["mean", "std", "count"]).reset_index()
        agg.to_csv(output_dir / "tables" / "small_objective_summary.csv", index=False)
        dataframe_to_latex(
            agg,
            output_dir / "tables" / "small_objective_summary.tex",
            caption="Small-instance objective summary",
            label="tab:small",
        )
    except Exception:
        pass

    results = {
        "small": small,
        "baseline": baseline,
        "large": large,
        "sensitivity": sensitivity,
        "ablation": ablation,
    }

    log = {
        "run_id": _now_stamp(),
        "command": "run-all",
        "seed": config.seed,
        "status": "success",
        "runtime_seconds": time.perf_counter() - started,
        "results": {
            "small_rows": int(len(small)),
            "baseline_rows": int(len(baseline.index)),
            "large_rows": int(len(large)),
            "sensitivity_rows": int(len(sensitivity.index)),
            "ablation_rows": int(len(ablation.index)),
        },
    }
    _write_log(_timestamped_log_path(output_dir), log)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce the full UTDRP-DP study.")
    parser.add_argument(
        "command",
        nargs="?",
        default="run-all",
        choices=[
            "run-all",
            "run-small",
            "run-large",
            "run-baseline",
            "run-sensitivity",
            "run-ablation",
            "run-heuristic-study",
        ],
    )
    parser.add_argument("--config", default=None, help="Path to YAML configuration.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    args = parser.parse_args(argv)

    # Default to the compact small-study configuration for run-small unless explicitly overridden.
    if args.command == "run-small" and args.config is None:
        config = _load_config("config/experiment_small.yaml")
    else:
        config = _load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config.experiment.get("output_dir", "outputs"))

    start = time.perf_counter()
    try:
        if args.command == "run-all":
            run_pipeline(config, output_dir)
        elif args.command == "run-small":
            _persist_rows("small_experiment", run_small_exact(config, output_dir=str(output_dir)), output_dir)
        elif args.command == "run-large":
            _persist_rows("large_heuristic", run_large_heuristic(config, output_dir=str(output_dir)), output_dir)
        elif args.command == "run-baseline":
            _persist_rows("baseline_comparison", run_baseline_comparison(config, output_dir=str(output_dir)), output_dir)
        elif args.command == "run-sensitivity":
            _persist_rows(
                "sensitivity_summary",
                run_sensitivity_study(config, output_dir=str(output_dir)),
                output_dir,
            )
        elif args.command == "run-ablation":
            _persist_rows("ablation_summary", run_ablation(config, output_dir=str(output_dir)), output_dir)
        elif args.command == "run-heuristic-study":
            _persist_rows("heuristic_study_runs", run_heuristic_study(config, output_dir=str(output_dir)), output_dir)

        if args.command != "run-all":
            runtime = time.perf_counter() - start
            _write_log(
                _timestamped_log_path(output_dir),
                _build_command_log(args.command, status="success", runtime_seconds=runtime),
            )
        return 0
    except Exception as ex:
        runtime = time.perf_counter() - start
        _write_log(
            _timestamped_log_path(output_dir),
            _build_command_log(
                args.command,
                status="failure",
                runtime_seconds=runtime,
                error=str(ex),
            ),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
