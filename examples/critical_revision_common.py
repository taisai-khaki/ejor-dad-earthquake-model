from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

import noto_access_experiment as noto
import noto_correlated_facility_experiment as correlated
import noto_mechanism_full_grid as mechanism
import noto_practical_resilience_experiment as practical
from ejor_dad import generate_regime_failure_states
from ejor_dad.checkpoint import CheckpointStore, atomic_write_dataframe, atomic_write_text
from ejor_dad.reproducibility import sha256_file


BASE_VERSION = "acute_access_graded_v4"
NEW_VERSION = "critical_revision_v1"
BASE_RHOS = (0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25)
DEVELOPMENT_RHOS = (0.0, 0.10, 0.125, 0.25)
GRID_LEVELS = (0.0, 0.25, 0.50, 0.75, 1.0)
RESPONSE_PROFILES = {
    "lower_timely_credit": ((0.0, 1.0), (30.0, 1.0), (60.0, 0.7), (120.0, 0.2), (180.0, 0.0)),
    "base": ((0.0, 1.0), (30.0, 1.0), (60.0, 0.75), (120.0, 0.25), (180.0, 0.0)),
    "higher_timely_credit": ((0.0, 1.0), (30.0, 1.0), (60.0, 0.9), (120.0, 0.4), (180.0, 0.0)),
}
BASELINE_OBJECTIVES = {
    0.0: 1070.808480778403,
    0.025: 1097.391425585094,
    0.05: 1121.899848445411,
    0.075: 1143.340829617140,
    0.10: 1159.710299213850,
    0.125: 1174.574248900284,
    0.15: 1185.046723379786,
    0.20: 1204.519775281224,
    0.25: 1222.787875223676,
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def base_output_root() -> Path:
    return repository_root() / "data_work" / "noto" / BASE_VERSION


def critical_output_root() -> Path:
    return repository_root() / "data_work" / "noto" / NEW_VERSION


def load_design(base_output_dir: Path) -> dict[str, Any]:
    return json.loads((base_output_dir / "run_design.json").read_text(encoding="utf-8"))


def model_args(design: dict[str, Any], output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        mode=design["mode"],
        density_cap=float(design["density_cap"]),
        residual_failure_ratio=float(design["residual_failure_ratio"]),
        failure_delay_reduction=float(design["failure_delay_reduction"]),
        retrofit_budget_scale=float(design["retrofit_budget_scale"]),
        time_sensitive_fraction=float(design["time_sensitive_fraction"]),
        immediate_loss_fraction=float(design["immediate_loss_fraction"]),
        capacity_throughput_per_bed=design.get("capacity_throughput_per_bed"),
        response_threshold_minutes=design.get("response_threshold_minutes"),
        graded_response=bool(design.get("graded_response", True)),
        output_dir=str(output_dir),
        workers=int(design.get("workers", 1)),
        force=False,
    )


def build_m4_instance(
    base_output_dir: Path,
    rho: float,
    *,
    density_cap: float | None = None,
    response_profile: str = "base",
    budget_scale: float | None = None,
) -> Any:
    design = load_design(base_output_dir)
    knots = RESPONSE_PROFILES[response_profile]
    base, _ = noto.build_noto_instance(
        float(rho),
        residual_failure_ratio=float(design["residual_failure_ratio"]),
        failure_delay_reduction=float(design["failure_delay_reduction"]),
        time_sensitive_fraction=float(design["time_sensitive_fraction"]),
        immediate_loss_fraction=float(design["immediate_loss_fraction"]),
        capacity_throughput_per_bed=design.get("capacity_throughput_per_bed"),
        response_threshold_minutes=design.get("response_threshold_minutes"),
        response_curve_knots=knots if design.get("graded_response", True) else None,
    )
    base = replace(
        base,
        ambiguity_density_cap=float(design["density_cap"] if density_cap is None else density_cap),
        budget_retrofit=float(base.budget_retrofit) * float(
            design["retrofit_budget_scale"] if budget_scale is None else budget_scale
        ),
    )
    regimes = correlated.regimes(base)
    states = generate_regime_failure_states(base.links, regimes)
    critical = {
        state.id
        for state in states
        if state.hazard_regime_id in {"normal", "north", "central"}
        and len(state.failed_links) <= 1
    }
    return replace(
        base,
        states=states,
        hazard_regimes=regimes,
        critical_service_state_ids=critical,
        minimum_protected_population=0.10 * float(base.protected_population_coefficients.sum()),
        minimum_zone_service_fraction=0.08,
    )


def build_mechanism_instances(base_output_dir: Path, rho: float) -> dict[str, Any]:
    base = build_m4_instance(base_output_dir, rho)
    specifications = mechanism.specs(base)
    return {name: mechanism.build(base, regimes, name) for name, regimes in specifications.items()}


def candidate_grid(instance: Any, *, enforce_budget: bool = True) -> list[tuple[int, np.ndarray]]:
    candidates = []
    for index, values in enumerate(product(GRID_LEVELS, repeat=len(instance.links)), start=1):
        y = np.asarray(values, dtype=float)
        if not enforce_budget or float(instance.retrofit_costs @ y) <= instance.budget_retrofit + 1e-10:
            candidates.append((index, y))
    return candidates


def atomic_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_status(path: Path, **payload: Any) -> None:
    payload.setdefault("updated_at_epoch", time.time())
    payload.setdefault("pid", os.getpid())
    atomic_json(path, payload)


def write_progress_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def source_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(repository_root()), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def source_dirty() -> bool:
    return bool(subprocess.run(
        ["git", "-C", str(repository_root()), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip())


def file_digest(path: Path) -> str:
    return sha256_file(path)


def model_hash(base_output_dir: Path, *, density_cap: float = 2.0, response_profile: str = "base") -> str:
    design_path = base_output_dir / "run_design.json"
    digest = hashlib.sha256()
    digest.update(design_path.read_bytes())
    digest.update(json.dumps({"density_cap": density_cap, "response_profile": response_profile}, sort_keys=True).encode())
    for relative in ("noto_zones.csv", "noto_centers.csv", "noto_corridors.csv"):
        path = base_output_dir / "tables" / relative
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def system_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cores": multiprocessing.cpu_count(),
    }
    try:
        import scipy
        metadata["scipy"] = scipy.__version__
    except Exception:
        metadata["scipy"] = None
    try:
        import psutil
        metadata["physical_cores"] = psutil.cpu_count(logical=False)
        metadata["memory_gb"] = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        metadata["physical_cores"] = None
        metadata["memory_gb"] = None
    return metadata


def write_run_metadata(output_dir: Path, *, experiment: str, parameters: dict[str, Any], expected_work: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = base_output_root()
    input_hashes = {"run_design.json": file_digest(base / "run_design.json")}
    for input_path in sorted((base / "tables").glob("*.csv")):
        input_hashes[str(input_path.relative_to(base)).replace("\\", "/")] = file_digest(input_path)
    manifest = {
        "experiment": experiment,
        "starting_commit": source_commit(),
        "starting_worktree_dirty": source_dirty(),
        "authoritative_model": "noto-mechanism-separated-capability-marginal-v2",
        "release": "noto-corrected-v2",
        "base_output_dir": str(base),
        "base_run_design_sha256": file_digest(base / "run_design.json"),
        "input_hashes": input_hashes,
        "parameters": parameters,
        "expected_work": expected_work,
        "tolerances": {
            "fixed_y_epsilon": 1e-6,
            "fixed_y_max_iterations": 500,
            "budget_tolerance": 1e-10,
            "probability_tolerance": 1e-10,
            "box_width_tolerance": 1e-8,
        },
        "system": system_metadata(),
        "started_at_epoch": time.time(),
        "status": "running",
    }
    atomic_json(output_dir / "run_manifest.json", manifest)
    atomic_json(output_dir / "reproducibility.json", {"source_commit": manifest["starting_commit"], "system": manifest["system"], "input_hashes": manifest["input_hashes"]})


def finish_run_metadata(output_dir: Path, *, status: str, runtime_seconds: float, extra: dict[str, Any] | None = None) -> None:
    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.update({"status": status, "finished_at_epoch": time.time(), "runtime_seconds": runtime_seconds})
    if extra:
        manifest.update(extra)
    atomic_json(manifest_path, manifest)
    atomic_json(output_dir / "runtime_summary.json", {"status": status, "runtime_seconds": runtime_seconds, **(extra or {})})


def load_source_records(base_output_dir: Path, rho: float, *, model_name: str = "M4") -> list[dict[str, Any]]:
    if model_name == "M4":
        root = base_output_dir / "correlated_facility_separated_capability_marginal_v2" / "checkpoints"
        prefix = "noto-correlated-facility-separated-capability-marginal-v2"
    else:
        root = base_output_dir / "mechanism_separated_capability_marginal_v2" / "checkpoints"
        prefix = f"mechanism_{model_name}"
    records = []
    for path in root.glob(f"*rho{rho:.2f}_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "feasible":
            records.append(payload)
    return records


def validate_plan(instance: Any, z: Sequence[float], w: Sequence[float], y: Sequence[float]) -> None:
    if float(instance.renovation_costs @ np.asarray(z)) > instance.budget_renovation + 1e-7:
        raise RuntimeError("Transferred plan violates renovation budget.")
    if float(instance.capacity_costs @ np.asarray(w)) > instance.budget_capacity + 1e-7:
        raise RuntimeError("Transferred plan violates capacity budget.")
    if float(instance.retrofit_costs @ np.asarray(y)) > instance.budget_retrofit + 1e-7:
        raise RuntimeError("Transferred plan violates retrofit budget.")


def save_table(rows: Iterable[dict[str, Any]], path: Path, sort_by: Sequence[str] | None = None) -> None:
    frame = pd.DataFrame(list(rows))
    if sort_by and not frame.empty:
        frame = frame.sort_values(list(sort_by))
    atomic_write_dataframe(frame, path)


def json_string(value: Sequence[float] | np.ndarray) -> str:
    return json.dumps(np.asarray(value, dtype=float).tolist(), separators=(",", ":"))
