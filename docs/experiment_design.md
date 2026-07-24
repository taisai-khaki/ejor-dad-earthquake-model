# Experiment design (operationalized)

This project reproduces paper-style sections through structured experiment modules.

## A) Small-instance exact validation
- Generate benchmarks with controlled seeds from `config/experiment_small.yaml`.
- Compare exact and heuristic outputs on the same instances.
- Collect:
  - exact objective,
  - heuristic objective,
  - MIP runtime and MIP gap,
  - solution feasibility flags,
  - objective gap (`(heuristic - exact) / exact`).

## B) Baseline comparison
- Compare proposed method against:
  - truck_only_baseline,
  - paired/fixed coordination baseline,
  - no_priority baseline,
  - no_unpairing baseline.
- Store per-instance CSV table and summary metrics.

## C) Medium/large-scale heuristic study
- Run `python -m src.cli --config config/experiment_heuristic_study.yaml run-heuristic-study`.
- Methods:
  - `truck_only`,
  - `simple_drone`,
  - `random_feasible`,
  - `paired_baseline`,
  - `no_unpairing`,
  - `nils`,
  - `nils_no_local_search`,
  - `nils_no_perturbation`,
  - `nils_no_battery_screening`.
- Scenario factors:
  - `n`,
  - eligible share,
  - endurance level,
  - speed ratio,
  - launch/retrieval handling time,
  - spatial pattern,
  - number of drones.
- Outputs:
  - per-run file `outputs/tables/heuristic_study_runs.csv`,
  - publication tables `table_01_...` through `table_12_...`,
  - figures `figure_01_...` through `figure_10_...`,
  - progress logs `outputs/logs/heuristic_study_progress_*.log`.

## D) Sensitivity sweeps
- One-factor-at-a-time sweep over configuration keys provided in `config.experiment`.
- Default sweep supports:
  - number of drones,
  - number of trucks,
  - swap/reload times,
  - high-priority window upper bounds,
  - customer class shares,
  - battery and speed parameters.
- For each factor value, generate fresh instances with fixed seeds and run heuristic + truck-only baseline.

## E) Ablation and structural checks
- Compare full heuristic with baseline variants on repeated synthetic instances.
- Track gaps to each baseline and runtime.

## F) Statistical layer
- Grouped summaries by method/factor, bootstrap confidence intervals, paired comparisons.
- Effect sizes and robust means stored from `src/experiments/statistics.py`.

## Output structure
- `outputs/tables` for CSV summaries.
- `outputs/figures` for visual summaries.
- `outputs/logs` for each pipeline run with time-stamped metadata.
