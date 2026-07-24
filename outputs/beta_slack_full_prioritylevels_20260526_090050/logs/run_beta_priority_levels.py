import subprocess
from pathlib import Path
from datetime import datetime
import sys

PROJECT = Path(r"C:\Users\L03128674\projects\utdrp_priority_sensitive_unpaired_truck_drone")
ROOT_REL = r"outputs\\beta_slack_full_prioritylevels_20260526_090050"
ROOT = PROJECT / ROOT_REL
MASTER = ROOT / "logs" / "beta_priority_master.log"

BETAS = [
    ("0p05", "0.05"),
    ("0p10", "0.10"),
    ("0p15", "0.15"),
    ("0p20", "0.20"),
]

def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    MASTER.parent.mkdir(parents=True, exist_ok=True)
    with MASTER.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)

cfg_template = """seed: 2026

generation:
  num_customers: 50
  num_trucks: 3
  num_drones: 2
  region: dense_urban
  instance_reps_per_size: 10
  sizes: [15, 50, 75, 100]

base_class_windows:
  high: [0, 60]
  medium: [0, 120]
  low: [0, 240]

milp:
  enabled: false
  solver: cplex
  time_limit_seconds: 120
  mip_gap: 0.001
  threads: 1

heuristics:
  nils:
    max_outer_iter: 25
    max_no_improve: 8
    time_limit_seconds: 180
    random_seed: 2026

experiment:
  output_dir: {run_rel}
  save_json: true
  save_csv: true
  study_methods:
    - truck_only
    - nils
  study_design: d_optimal
  study_d_optimal_points: 72
  study_d_optimal_restarts: 10
  study_reps_per_scenario: 10
  study_feasibility_policy: feasible_only

  study_exclude_conditions:
    - {{spatial_pattern: clustered, n_min: 75}}
    - {{spatial_pattern: clustered, number_of_drones_min: 3}}
    - {{spatial_pattern: clustered, handling_time: long}}

  study_grid:
    n: [15, 50, 75, 100]
    eligible_share: [0.25, 0.75]
    endurance: [low, medium, high]
    speed_ratio: [1.25, 2.0]
    handling_time: [short, medium, long]
    spatial_pattern: [uniform, clustered]
    drones_available: [1, 2, 3]
    priority_weight_profile: [flat, moderate, base, strict]
    battery_slack_ratio: [{beta}]
  study_conditional_n_by_drones:
    1: [15, 50]
    2: [50, 75]
    3: [75, 100]
"""

log(f"Start beta full run with priority-level metrics | root={ROOT_REL}")

for tag, val in BETAS:
    run_rel = f"{ROOT_REL}\\beta_{tag}"
    run_dir = PROJECT / run_rel
    run_logs = run_dir / "logs"
    run_logs.mkdir(parents=True, exist_ok=True)

    cfg = PROJECT / "config" / f"tmp_beta_slack_priority_{tag}.yaml"
    cfg.write_text(cfg_template.format(run_rel=run_rel, beta=val), encoding="utf-8")

    live = run_logs / f"beta_{tag}_live.log"
    log(f"Start beta={val} | out={run_rel}")

    cmd = [
        r"C:\\Python314\\python.exe", "-u", "-m", "src.cli", "run-heuristic-study",
        "--config", str(cfg), "--output-dir", run_rel,
    ]

    with live.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen(cmd, cwd=str(PROJECT), stdout=out, stderr=subprocess.STDOUT, text=True)
        code = proc.wait()

    if code != 0:
        log(f"FAILED beta={val} exit={code}")
        sys.exit(code)

    log(f"Done beta={val}")

log("All beta blocks finished.")
