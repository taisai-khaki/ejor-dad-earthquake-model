from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

from critical_revision_common import atomic_json, finish_run_metadata, save_table, write_run_metadata, write_status


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def classify(input_root: Path) -> dict:
    continuous = read_csv(input_root / "continuous_bb" / "tables" / "table_noto_continuous_monotone_bb.csv")
    cover = read_csv(input_root / "continuous_bb" / "tables" / "table_noto_continuous_policy_cover.csv")
    dependence = read_csv(input_root / "mechanism_value" / "tables" / "table_noto_value_shared_dependence.csv")
    frontier = read_csv(input_root / "budget_frontier" / "tables" / "table_noto_budget_policy_summary.csv")
    equity_gate = {}
    gate_path = input_root / "equity" / "equity_gate.json"
    if gate_path.exists():
        equity_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    ambiguity_status = "blocked"
    ambiguity_path = input_root / "ambiguity_anchor" / "status.json"
    if ambiguity_path.exists():
        ambiguity_status = json.loads(ambiguity_path.read_text(encoding="utf-8")).get("status", "unknown")
    path_status = json.loads((input_root / "reproducibility" / "audit_summary.json").read_text(encoding="utf-8")).get("status", "missing") if (input_root / "reproducibility" / "audit_summary.json").exists() else "missing"
    cost_status = json.loads((input_root / "reproducibility_cost" / "audit_summary.json").read_text(encoding="utf-8")).get("status", "missing") if (input_root / "reproducibility_cost" / "audit_summary.json").exists() else "missing"
    continuous_status = "not_run"
    policy_status = "unresolved"
    if not continuous.empty:
        continuous_status = "certified" if bool(continuous.converged.all()) else "not_certified"
    if not cover.empty and "policy_class_status" in cover:
        statuses = set(cover.policy_class_status.astype(str))
        policy_status = next(iter(statuses)) if len(statuses) == 1 else "unresolved"
    dependence_status = "missing"
    if not dependence.empty:
        if {"source_model", "evaluation_model", "status"}.issubset(dependence.columns):
            selected = dependence[(dependence.source_model.isin(["M2", "M3"])) & (dependence.evaluation_model == "M4")]
            dependence_status = "; ".join(f"{row.source_model}->{row.evaluation_model}:{row.status}" for _, row in selected.iterrows())
        elif {"contrast", "transfer_type", "status"}.issubset(dependence.columns):
            selected = dependence[dependence.contrast.isin(["M2->M4", "M3->M4"])]
            dependence_status = "; ".join(f"{row.contrast}:{row.transfer_type}:{row.status}" for _, row in selected.iterrows())
    budget_status = "missing" if frontier.empty else "; ".join(sorted(set(frontier.evidence_classification.astype(str))))
    equity_status = "missing" if not equity_gate else ("complementarity_supported" if equity_gate.get("complementarity_supported") else "specification-dependent co-benefit or not supported")
    return {"continuous_status": continuous_status, "policy_status": policy_status, "dependence_status": dependence_status, "budget_status": budget_status, "equity_status": equity_status, "ambiguity_status": ambiguity_status, "path_audit_status": path_status, "cost_audit_status": cost_status, "continuous_rows": len(continuous), "dependence_rows": len(dependence), "frontier_rows": len(frontier)}


def write_decision_tables(input_root: Path, output: Path) -> None:
    target = output / "manuscript_decision_tables"
    target.mkdir(parents=True, exist_ok=True)
    mappings = {
        "table_A_continuous_certificate.csv": input_root / "continuous_bb" / "tables" / "table_noto_continuous_monotone_bb.csv",
        "table_B_value_of_dependence.csv": input_root / "mechanism_value" / "tables" / "table_noto_value_shared_dependence.csv",
        "table_C_budget_frontier_key_points.csv": input_root / "budget_frontier" / "tables" / "table_noto_budget_policy_summary.csv",
        "table_D_equity_frontier_key_points.csv": input_root / "equity" / "tables" / "table_noto_robustness_equity_metrics.csv",
        "table_E_scalability.csv": input_root / "synthetic_scaling" / "tables" / "table_synthetic_scalability.csv",
        "table_F_ambiguity_anchor.csv": input_root / "ambiguity_anchor" / "tables" / "table_noto_anchored_ambiguity_results.csv",
    }
    for name, source in mappings.items():
        if name == "table_F_ambiguity_anchor.csv" and not source.exists():
            continue
        if source.exists():
            shutil.copy2(source, target / name)
        else:
            pd.DataFrame([{"status": "not_available", "source": str(source)}]).to_csv(target / name, index=False)
    audit_rows = []
    for label, source in (("path_incidence", input_root / "reproducibility" / "audit_summary.json"), ("renovation_cost", input_root / "reproducibility_cost" / "audit_summary.json")):
        payload = json.loads(source.read_text(encoding="utf-8")) if source.exists() else {"status": "missing"}
        audit_rows.append({"audit": label, **payload})
    pd.DataFrame(audit_rows).to_csv(target / "table_G_reproducibility_audits.csv", index=False)


def report(summary: dict, output: Path) -> None:
    lines = [
        "# Critical revision computation report", "", 
        "1. Reproducibility gate: " + ("pass" if summary["path_audit_status"] == "passed" and summary["cost_audit_status"] == "passed" else "incomplete or failed"),
        f"2. Continuous problem: {summary['continuous_status']}; optimizer cover status: {summary['policy_status']}",
        f"3. Residual crossover: {summary['policy_status']}",
        f"4. Shared-dependence value: {summary['dependence_status']}",
        f"5. Budget geometry: {summary['budget_status']}",
        f"6. Robustness–equity result: {summary['equity_status']}",
        "7. Search scalability: see the full synthetic table; state support is reported as R*2^L and no extrapolation is made.",
        f"8. Ambiguity calibration: {summary['ambiguity_status']}",
        "9. Claims the manuscript may retain: only numerical facts supported by completed tables, with continuous policy claims restricted to resolved optimizer-cover classes.",
        "10. Claims the manuscript must remove: any unsupported claim of continuous-policy crossover, shared-dependence value, complementarity, or large-network scalability.",
        "", "The report separates numerical facts from model-conditional inferences. Missing or blocked blocks are not treated as negative empirical evidence.",
    ]
    (output / "critical_revision_computation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def manifest(output: Path, source_root: Path) -> None:
    files = []
    generator_sources = [
        (
            str(candidate.relative_to(Path(__file__).resolve().parents[1]).as_posix()),
            candidate.read_text(encoding="utf-8-sig", errors="ignore"),
        )
        for candidate in sorted(Path(__file__).resolve().parent.glob("*.py"))
    ]
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name in {"master_manifest.json", "SHA256SUMS.txt"}:
            continue
        relative = path.relative_to(output).as_posix()
        generator = "unknown"
        if relative.startswith("manuscript_decision_tables/") or relative.endswith("critical_revision_computation_report.md"):
            generator = "examples/build_critical_revision_audit.py"
        else:
            for candidate_path, candidate_source in generator_sources:
                if path.name in candidate_source:
                    generator = candidate_path
                    break
        files.append({"path": relative, "sha256": sha256(path), "generating_script": generator})
    atomic_json(output / "master_manifest.json", {"source_root": str(source_root), "files": files})
    (output / "SHA256SUMS.txt").write_text("\n".join(f"{item['sha256']}  {item['path']}" for item in files) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--test-output", default=None)
    parser.add_argument("--skip-copy", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    input_root = Path(args.input_root).resolve()
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "run_manifest.json").exists():
        write_run_metadata(output, experiment="critical_revision_integrated_audit", parameters=vars(args), expected_work={"input_root": str(input_root), "decision_tables": 7, "manifest": True})
    if input_root.exists() and not args.skip_copy:
        for source in input_root.iterdir():
            destination = output / source.name
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            elif source.name not in {"run_manifest.json", "runtime_summary.json"}:
                shutil.copy2(source, destination)
    summary = classify(output)
    atomic_json(output / "critical_revision_summary.json", summary)
    write_decision_tables(output, output)
    report(summary, output)
    if args.test_output and Path(args.test_output).exists():
        shutil.copy2(args.test_output, output / "test_output.txt")
    elif not (output / "test_output.txt").exists():
        (output / "test_output.txt").write_text("Full test output is produced by the pipeline test step.\n", encoding="utf-8")
    manifest(output, input_root)
    write_status(output / "status.json", status="completed", block="integrated_audit", summary=summary)
    finish_run_metadata(output, status="completed", runtime_seconds=time.perf_counter() - started, extra=summary)


if __name__ == "__main__":
    main()
