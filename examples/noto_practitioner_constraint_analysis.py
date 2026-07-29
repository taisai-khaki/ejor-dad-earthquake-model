from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def load_candidates(output_dir: Path) -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"_rho(?P<rho>\d+)p(?P<decimal>\d+).*__grid_")
    for path in (output_dir / "checkpoints").glob("*__grid_*.json"):
        match = pattern.search(path.name)
        if not match:
            continue
        rho = float(f"{int(match.group('rho'))}.{match.group('decimal')}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "rho": rho, "objective": float(payload["objective"]),
            "y": np.asarray(payload["y"], dtype=float),
            "candidate_index": int(payload.get("candidate_index", -1)),
        })
    if not rows:
        raise RuntimeError("No grid checkpoints found.")
    return pd.DataFrame(rows)


def analyse(output_dir: Path) -> None:
    candidates = load_candidates(output_dir)
    corridors = pd.read_csv(output_dir / "tables" / "table_noto_corridors.csv")
    costs = corridors["retrofit_cost"].to_numpy(float)
    criticality = (
        corridors["baseline_failure_probability"].to_numpy(float)
        * corridors["failure_penalty_minutes"].to_numpy(float)
        * np.sqrt(corridors["route_length_km"].to_numpy(float))
    )
    criticality /= criticality.sum()
    scenarios = [
        ("coverage_basic", 2, 1.0, 0.75),
        ("coverage_balanced", 3, 1.5, 0.60),
        ("coverage_strong", 3, 2.0, 0.50),
    ]
    epsilons = [0.001, 0.005, 0.01]
    rows = []
    for rho, block in candidates.groupby("rho"):
        robust_optimum = float(block["objective"].min())
        for epsilon in epsilons:
            near = block[block["objective"] <= robust_optimum * (1.0 + epsilon)].copy()
            for scenario, minimum_active, northern_coverage, max_concentration in scenarios:
                feasible = []
                for row in near.itertuples(index=False):
                    y = row.y
                    spend = costs * y
                    total_spend = float(spend.sum())
                    active = int(np.count_nonzero(y >= 0.25 - 1e-10))
                    north = float(y[2] + y[3] + y[4])
                    concentration = float(spend.max() / total_spend) if total_spend > 0 else 1.0
                    score = float(criticality @ y)
                    if active >= minimum_active and north >= northern_coverage - 1e-10 and concentration <= max_concentration + 1e-10:
                        feasible.append((score, tuple(-value for value in y), row))
                if feasible:
                    score, _, selected = max(feasible, key=lambda item: (item[0], item[1]))
                    selected_y = selected.y
                    selected_objective = float(selected.objective)
                    status = "feasible"
                else:
                    selected_y = np.full(len(costs), np.nan)
                    selected_objective = np.nan
                    score = np.nan
                    status = "infeasible"
                rows.append({
                    "rho": rho, "epsilon_fraction": epsilon, "scenario": scenario,
                    "status": status, "robust_optimum": robust_optimum,
                    "admissible_near_optimal_count": len(near),
                    "operationally_feasible_count": len(feasible),
                    "selected_objective": selected_objective,
                    "robust_loss_sacrifice": selected_objective - robust_optimum if status == "feasible" else np.nan,
                    "robust_loss_sacrifice_percent": 100.0 * (selected_objective / robust_optimum - 1.0) if status == "feasible" else np.nan,
                    "selected_y_json": json.dumps(selected_y.tolist()),
                    "criticality_score": score,
                    "minimum_active_links": minimum_active,
                    "minimum_northern_branch_coverage": northern_coverage,
                    "maximum_single_corridor_spend_share": max_concentration,
                    "metric_definition": "criticality proportional to baseline failure probability * disruption minutes * sqrt(route length)",
                })
    result = pd.DataFrame(rows).sort_values(["scenario", "epsilon_fraction", "rho"])
    result.to_csv(output_dir / "tables" / "table_noto_epsilon_constrained_practitioner_policies.csv", index=False)
    design = {
        "method": "lexicographic epsilon-constrained robust optimization",
        "primary_requirement": "robust objective <= (1+epsilon) times discretized robust optimum",
        "secondary_objective": "maximize normalized corridor criticality coverage",
        "constraints": ["minimum active retrofits", "minimum northern branch coverage", "maximum single-corridor spending share"],
        "warning": "scenario thresholds are policy diagnostics and require practitioner validation before prescriptive use",
    }
    (output_dir / "practitioner_layer_design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    analyse(Path(parser.parse_args().output_dir))
