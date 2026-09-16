from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from critical_revision_common import atomic_json, base_output_root, finish_run_metadata, save_table, write_run_metadata, write_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-output-dir", default=str(base_output_root()))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    base = Path(args.base_output_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.joinpath("tables").mkdir(parents=True, exist_ok=True)
    if not (output / "run_manifest.json").exists():
        write_run_metadata(output, experiment="noto_renovation_cost_audit", parameters=vars(args), expected_work={"zones": 5, "budget_fraction": 0.12})
    zones = pd.read_csv(base / "tables" / "table_noto_zones.csv")
    rows = []
    for zone in zones.itertuples(index=False):
        raw = float(zone.at_risk_population)
        raw_cost = raw / 1000.0
        final = max(raw_cost, 1.0)
        rows.append({"zone": zone.zone_id, "at_risk_population": raw, "raw_cost_A_over_1000": raw_cost, "cost_floor": 1.0, "final_cost": final})
    total = sum(row["final_cost"] for row in rows)
    budget = 0.12 * total
    for row in rows:
        row.update({"total_cost": total, "renovation_budget": budget, "budget_fraction": budget / total})
    save_table(rows, output / "tables" / "table_noto_renovation_cost_audit.csv", ["zone"])
    raw_total = sum(row["raw_cost_A_over_1000"] for row in rows)
    summary = {"status": "passed", "implemented_total_cost": total, "raw_sum_without_floor": raw_total, "implemented_budget": budget, "implemented_budget_fraction": budget / total, "incorrect_raw_only_fraction": 0.12 * total / raw_total, "formula": "C_r^Z=max(A_r/1000,1.0)"}
    atomic_json(output / "audit_summary.json", summary)
    write_status(output / "status.json", **summary, block="renovation_cost_audit")
    finish_run_metadata(output, status="passed", runtime_seconds=time.perf_counter() - started, extra=summary)


if __name__ == "__main__":
    main()
