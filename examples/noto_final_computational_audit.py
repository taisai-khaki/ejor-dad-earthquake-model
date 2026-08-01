from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noto_correlated_validation_postprocess as validation
import noto_mechanism_full_grid as mechanism
import noto_practical_resilience_experiment as practical
from ejor_dad.checkpoint import atomic_write_text
from ejor_dad.fixed_y import evaluate_fixed_y
from ejor_dad.recourse import solve_recourse, solve_recourse_dual
from ejor_dad.states import nominal_probabilities

VERSION = "noto-final-computational-audit-v1-supplied-spec"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def hash_files(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def json_hash(payload) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def repository_revision(repo: Path) -> dict:
    if not (repo / ".git").exists():
        return {
            "commit": None,
            "status": (
                "Project directory is not an independent Git repository; "
                "the source-bundle SHA-256 is the reproducibility identifier."
            ),
        }
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {"commit": commit, "status": "available"}


def selected_rows(output_dir: Path) -> pd.DataFrame:
    path = (
        output_dir
        / "mechanism_separated_capability_marginal_v1"
        / "tables"
        / "table_noto_mechanism_ablation_full_grid.csv"
    )
    table = pd.read_csv(path)
    expected = {(model, rho) for model in mechanism.ORDER for rho in mechanism.RHOS}
    observed = {
        (str(row.model), float(row.rho))
        for row in table.itertuples(index=False)
    }
    if observed != expected:
        raise RuntimeError(
            "Mechanism table does not contain the complete 15-row grid."
        )
    return table


def probability_audit(base, model_specs, table):
    rows = []
    marginal_rows = []
    max_sum_error = 0.0
    minimum_probability = 1.0
    maximum_marginal_error = 0.0
    for row in table.itertuples(index=False):
        model = str(row.model)
        rho = float(row.rho)
        y = np.asarray(json.loads(row.selected_y_json), dtype=float)
        instance = mechanism.build(base, model_specs[model], model)
        nominal = nominal_probabilities(
            instance.links,
            instance.states,
            y,
            instance.hazard_regimes,
        )
        if any(state.is_tail for state in instance.states):
            raise RuntimeError("Nominal marginal preservation requires explicit no-tail support.")
        indicators = np.asarray(
            [
                [link.id in state.failed_links for state in instance.states]
                for link in instance.links
            ],
            dtype=float,
        )
        targets = np.asarray(
            [
                link.failure_probability(float(level))
                for link, level in zip(instance.links, y)
            ],
            dtype=float,
        )
        computed = indicators @ nominal
        for link_index, link in enumerate(instance.links):
            error = float(computed[link_index] - targets[link_index])
            maximum = abs(error)
            maximum_marginal_error = max(maximum_marginal_error, maximum)
            marginal_rows.append(
                {
                    "model": model,
                    "rho": rho,
                    "corridor": link.id,
                    "y_level": float(y[link_index]),
                    "target_nominal_marginal": float(targets[link_index]),
                    "computed_nominal_marginal": float(computed[link_index]),
                    "absolute_error": maximum,
                }
            )
        sum_error = abs(float(nominal.sum()) - 1.0)
        max_sum_error = max(max_sum_error, sum_error)
        minimum_probability = min(minimum_probability, float(nominal.min()))
        regime_masses = {}
        for state, probability in zip(instance.states, nominal):
            regime = state.hazard_regime_id or "unlabelled"
            regime_masses[regime] = (
                regime_masses.get(regime, 0.0) + float(probability)
            )
        rows.append(
            {
                "model": model,
                "rho": rho,
                "selected_y_json": json.dumps(y.tolist()),
                "probability_sum": float(nominal.sum()),
                "probability_sum_error": sum_error,
                "minimum_probability": float(nominal.min()),
                "maximum_probability": float(nominal.max()),
                "regime_masses_json": json.dumps(
                    regime_masses,
                    sort_keys=True,
                ),
            }
        )
    return rows, marginal_rows, max_sum_error, minimum_probability, maximum_marginal_error


def primal_dual_value(instance, state, z, w, alpha, beta):
    demand = instance.demand_after_renovation(z)
    capacity = (
        instance.capacity_after_investment(w)
        * instance.facility_availability(state)
    )
    return float(
        capacity @ alpha
        + demand @ beta
    )


def numerical_replay(output_dir, args, table):
    rows = []
    worst_case_rows = []
    maximum_primal_dual_error = 0.0
    maximum_replay_error = 0.0
    m4 = table[table["model"] == "M4"].sort_values("rho")
    for row in m4.itertuples(index=False):
        rho = float(row.rho)
        y = np.asarray(json.loads(row.selected_y_json), dtype=float)
        stored_z = np.asarray(json.loads(row.selected_z_json), dtype=float)
        stored_w = np.asarray(json.loads(row.selected_w_json), dtype=float)
        stored_objective = float(row.objective)
        base, _ = practical.build_instance(rho, args)
        instance = mechanism.build(
            base,
            mechanism.specs(base)["M4"],
            "M4",
        )
        state_maximum = 0.0
        for state in instance.states:
            primal = solve_recourse(
                instance,
                state,
                stored_z,
                stored_w,
                y=y,
            )
            alpha, beta = solve_recourse_dual(
                instance,
                state,
                stored_z,
                stored_w,
                survival=primal.survival,
                y=y,
            )
            dual_value = primal_dual_value(
                instance,
                state,
                stored_z,
                stored_w,
                alpha,
                beta,
            )
            state_maximum = max(
                state_maximum,
                abs(primal.survivors - dual_value),
            )
        replay = evaluate_fixed_y(
            instance,
            y,
            epsilon=1e-5,
            max_iterations=300,
        )
        replay_error = abs(float(replay.objective) - stored_objective)
        indicators = np.asarray(
            [
                [link.id in state.failed_links for state in instance.states]
                for link in instance.links
            ],
            dtype=float,
        )
        worst_case_marginals = indicators @ replay.worst_case_distribution
        for link_index, link in enumerate(instance.links):
            worst_case_rows.append(
                {
                    "model": "M4",
                    "rho": rho,
                    "corridor": link.id,
                    "y_level": float(y[link_index]),
                    "worst_case_marginal": float(worst_case_marginals[link_index]),
                }
            )
        maximum_primal_dual_error = max(
            maximum_primal_dual_error,
            state_maximum,
        )
        maximum_replay_error = max(maximum_replay_error, replay_error)
        rows.append(
            {
                "rho": rho,
                "stored_objective": stored_objective,
                "replayed_objective": float(replay.objective),
                "replay_discrepancy": replay_error,
                "replayed_oracle_gap": float(
                    replay.objective - replay.lower_bound
                ),
                "maximum_state_primal_dual_discrepancy": state_maximum,
                "nominal_probability_sum_error": abs(
                    float(replay.nominal_distribution.sum()) - 1.0
                ),
                "worst_case_probability_sum_error": abs(
                    float(replay.worst_case_distribution.sum()) - 1.0
                ),
            }
        )
    return rows, worst_case_rows, maximum_primal_dual_error, maximum_replay_error


def monotonicity_audit(output_dir: Path) -> list[dict[str, float | int]]:
    checkpoint_root = (
        output_dir
        / "mechanism_separated_capability_marginal_v1"
        / "checkpoints"
    )
    rows = []
    for rho in mechanism.RHOS:
        records = {}
        pattern = f"mechanism_M4_rho{rho:.2f}_grid*.json"
        for path in checkpoint_root.glob(pattern):
            payload = json.loads(path.read_text(encoding="utf-8"))
            records[tuple(np.round(payload["y"], 12))] = payload
        if len(records) != 996:
            raise RuntimeError(
                f"Expected 996 M4 checkpoints at rho={rho:.2f}, found {len(records)}."
            )
        comparable_pairs = 0
        acceptability_violations = 0
        objective_violations = 0
        largest_violation = 0.0
        items = list(records.items())
        for lower_y, lower_payload in items:
            for upper_y, upper_payload in items:
                if lower_y == upper_y or not np.all(
                    np.asarray(upper_y) >= np.asarray(lower_y) - 1e-10
                ):
                    continue
                comparable_pairs += 1
                if lower_payload.get("status") != "feasible":
                    continue
                if upper_payload.get("status") != "feasible":
                    acceptability_violations += 1
                    continue
                violation = float(upper_payload["objective"]) - float(lower_payload["objective"])
                if violation > 1e-7:
                    objective_violations += 1
                    largest_violation = max(largest_violation, violation)
        rows.append(
            {
                "rho": float(rho),
                "comparable_pair_count": comparable_pairs,
                "acceptability_nesting_violations": acceptability_violations,
                "objective_monotonicity_violations": objective_violations,
                "largest_objective_violation": largest_violation,
            }
        )
    return rows


def experiment_runtime_and_checkpoints(output_dir: Path):
    definitions = (
        ("mechanism_full_grid", "mechanism_separated_capability_marginal_v1"),
        (
            "selected_sensitivity_full_grid",
            "selected_sensitivity_separated_capability_marginal_v1",
        ),
        ("stage2_joint_full_grid", "operational_stage2_joint_separated_capability_marginal_v1"),
    )
    rows = []
    for name, directory in definitions:
        root = output_dir / directory
        manifest = json.loads(
            (root / "run_manifest.json").read_text(encoding="utf-8-sig")
        )
        status = json.loads(
            (root / "status.json").read_text(encoding="utf-8-sig")
        )
        if status.get("status") != "completed":
            raise RuntimeError(f"{name} is not complete.")
        started = float(manifest["started"])
        completed = float(status["updated"])
        checkpoint_root = root / "checkpoints"
        checkpoint_count = (
            len(list(checkpoint_root.glob("*.json")))
            if checkpoint_root.exists()
            else 0
        )
        rows.append(
            {
                "experiment": name,
                "started_epoch": started,
                "completed_epoch": completed,
                "wall_runtime_seconds": completed - started,
                "checkpoint_count": checkpoint_count,
                "status": status["status"],
            }
        )
    return rows


def run_tests(repo: Path):
    started = time.perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout + "\n" + completed.stderr).strip()
    passed = None
    for token in output.replace("\n", " ").split():
        if token.isdigit():
            continue
        if token.endswith("passed"):
            break
    import re

    match = re.search(r"(\d+)\s+passed", output)
    if match:
        passed = int(match.group(1))
    return {
        "return_code": completed.returncode,
        "passed": passed,
        "runtime_seconds": time.perf_counter() - started,
        "output": output,
    }


def main(output_dir: Path) -> None:
    repo = Path(__file__).resolve().parent.parent
    root = output_dir / "final_computational_audit_v1"
    root.mkdir(parents=True, exist_ok=True)
    design_path = output_dir / "run_design.json"
    design = json.loads(design_path.read_text(encoding="utf-8-sig"))
    args = validation.args_from(design, output_dir)
    base, _ = practical.build_instance(0.0, args)
    model_specs = mechanism.specs(base)
    table = selected_rows(output_dir)

    state_manifest = {}
    for model, regimes in model_specs.items():
        instance = mechanism.build(base, regimes, model)
        state_manifest[model] = {
            "state_ids": [state.id for state in instance.states],
            "design_basis_ids": sorted(
                instance.critical_service_state_ids
            ),
            "regimes": [
                {
                    "id": regime.id,
                    "probability": regime.probability,
                    "failed_centers": list(regime.failed_centers),
                    "link_failure_multipliers": dict(
                        regime.link_failure_multipliers
                    ),
                }
                for regime in regimes
            ],
        }
    candidates = mechanism.candidate_grid(
        mechanism.build(base, model_specs["M4"], "M4")
    )
    grid_manifest = {
        "levels": mechanism.GRID,
        "road_order": base.link_ids,
        "retrofit_costs": base.retrofit_costs.tolist(),
        "retrofit_budget": float(base.budget_retrofit),
        "candidate_count": len(candidates),
        "candidates": [
            {"index": index, "y": y.tolist()} for index, y in candidates
        ],
    }
    atomic_write_text(
        root / "state_manifest.json",
        json.dumps(state_manifest, indent=2, sort_keys=True),
    )
    atomic_write_text(
        root / "grid_manifest.json",
        json.dumps(grid_manifest, indent=2, sort_keys=True),
    )

    (
        probability_rows,
        marginal_rows,
        max_sum_error,
        minimum_probability,
        max_marginal_error,
    ) = probability_audit(base, model_specs, table)
    replay_rows, worst_case_rows, max_primal_dual, max_replay = numerical_replay(
        output_dir,
        args,
        table,
    )
    monotonicity_rows = monotonicity_audit(output_dir)
    if max_marginal_error > 1e-10:
        raise RuntimeError(
            f"Nominal marginal preservation failed with maximum error {max_marginal_error:.6g}."
        )
    if any(
        row["acceptability_nesting_violations"]
        or row["objective_monotonicity_violations"]
        for row in monotonicity_rows
    ):
        raise RuntimeError("M4 exhaustive monotonicity audit failed.")
    runtime_rows = experiment_runtime_and_checkpoints(output_dir)
    tests = run_tests(repo)
    if tests["return_code"] != 0:
        raise RuntimeError(
            "Final test suite failed. See final_test_output.txt."
        )

    pd.DataFrame(probability_rows).to_csv(
        root / "table_probability_audit.csv",
        index=False,
    )
    pd.DataFrame(marginal_rows).to_csv(
        root / "table_nominal_marginal_audit.csv",
        index=False,
    )
    pd.DataFrame(replay_rows).to_csv(
        root / "table_numerical_replay_audit.csv",
        index=False,
    )
    pd.DataFrame(worst_case_rows).to_csv(
        root / "table_worst_case_marginal_audit.csv",
        index=False,
    )
    pd.DataFrame(monotonicity_rows).to_csv(
        root / "table_monotonicity_audit.csv",
        index=False,
    )
    pd.DataFrame(runtime_rows).to_csv(
        root / "table_runtime_checkpoint_audit.csv",
        index=False,
    )
    atomic_write_text(root / "final_test_output.txt", tests["output"])

    source_files = list((repo / "src" / "ejor_dad").glob("*.py"))
    source_files.extend(
        [
            repo / "examples" / "noto_mechanism_full_grid.py",
            repo / "examples" / "noto_selected_sensitivity_full_grid.py",
            repo / "examples" / "noto_stage2_joint_full_grid.py",
            repo / "examples" / "noto_final_computational_audit.py",
        ]
    )
    revision = repository_revision(repo)
    manifest = {
        "audit_version": VERSION,
        "generated_epoch": time.time(),
        "model_version": mechanism.VERSION,
        "code_revision": revision,
        "source_bundle_sha256": hash_files(repo, source_files),
        "parameter_file": str(design_path),
        "parameter_file_sha256": sha256_file(design_path),
        "state_manifest_sha256": json_hash(state_manifest),
        "grid_manifest_sha256": json_hash(grid_manifest),
        "regime_definition": state_manifest["M4"]["regimes"],
        "design_basis_definition": (
            "normal, north, and central regimes with at most one failed road"
        ),
        "design_basis_state_count": len(
            state_manifest["M4"]["design_basis_ids"]
        ),
        "road_grid_definition": (
            "{0,.25,.50,.75,1}^5 intersect retrofit budget"
        ),
        "road_grid_candidate_count": len(candidates),
        "solver": {
            "linear_programming_method": "scipy.optimize.linprog(method='highs')",
            "scipy_version": scipy.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "tolerances": {
            "fixed_y_cut_gap": 1e-5,
            "fixed_y_max_iterations": 300,
            "stage2_service_resolution": SERVICE_RESOLUTION,
            "lp_primal_dual_tolerances": "SciPy/HiGHS defaults",
            "near_optimal_relative_thresholds": [0.0001, 0.001, 0.005],
        },
        "probability_audit": {
            "maximum_probability_sum_error": max_sum_error,
            "minimum_probability": minimum_probability,
            "maximum_nominal_marginal_error": max_marginal_error,
        },
        "monotonicity_audit": monotonicity_rows,
        "numerical_audit": {
            "maximum_primal_dual_discrepancy": max_primal_dual,
            "maximum_replay_discrepancy": max_replay,
        },
        "experiment_runtime_and_checkpoints": runtime_rows,
        "final_test_suite": tests,
        "priority_gate": {
            "status": "not_run",
            "reason": (
                "The supplied computation requirements do not specify the "
                "cascade allocation rule, sector floors and ceilings, "
                "baseline envelope, or schedule tie tolerance."
            ),
        },
    }
    atomic_write_text(
        root / "master_manifest.json",
        json.dumps(manifest, indent=2),
    )


if __name__ == "__main__":
    from noto_stage2_joint_full_grid import SERVICE_RESOLUTION

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()
    main(Path(arguments.output_dir))
