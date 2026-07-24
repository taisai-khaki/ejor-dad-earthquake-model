# Unpaired Truck-Drone Routing with Delivery Priority (UTDRP-DP)

This repository provides a research-grade Python implementation for the manuscript on
priority-sensitive unpaired truck-drone routing.

## Implemented modules

- `data_models.py`: typed data structures for customers, vehicles, instances, and solutions.
- `parameters.py`: YAML config parser and typed experiment settings.
- `instance_generator.py`: stochastic benchmark generator with reproducible seeds.
- `milp/`: exact formulation and solver wrapper.
- `heuristics/`: construction, NILS, local search, perturbation, acceptance, and baselines.
- `experiments/`: exact benchmark, heuristic benchmark, baselines, ablations, sensitivity.
- `reporting/`: CSV, LaTeX, and plotting utilities.
- `feasibility.py`: strict solution validator.

## Setup

```bash
pip install -r requirements.txt
```

## One command to run the full study

```bash
python -m src.cli --config config/default.yaml run-all
```

This executes:
- small exact/heuristic comparison,
- large heuristic runs,
- baseline comparison,
- sensitivity sweep,
- ablation.

## Reproducibility

- Every generator and solver call is seed-controlled.
- Outputs are written with time-stamped logging under `outputs/logs`.
- CSV summaries are written under `outputs/tables`.
- Intermediate generated instances are written under `data/raw`.

## Individual commands

```bash
python -m src.cli --config config/experiment_small.yaml run-small
python -m src.cli --config config/experiment_large.yaml run-large
python -m src.cli --config config/experiment_large.yaml run-baseline
python -m src.cli --config config/experiment_large.yaml run-sensitivity
python -m src.cli --config config/experiment_large.yaml run-ablation
python -m src.cli --config config/experiment_heuristic_study.yaml run-heuristic-study
```

`config/experiment_small.yaml` is calibrated for exact-vs-heuristic comparison on tiny instances
(`n = 3..9`) with:
- shared instance seeds,
- CPLEX exact solve,
- multi-start NILS selection,
- feasible-only optimality-gap reporting fields in `outputs/tables/small_exact_summary.csv`.

## Heuristic-Only Benchmark Study (Recommended for large-scale runs)

`run-heuristic-study` executes:
- strong baselines (`truck_only`, `simple_drone`, `random_feasible`, `paired_baseline`, `no_unpairing`),
- full NILS,
- ablation variants (`nils_no_local_search`, `nils_no_perturbation`, `nils_no_battery_screening`),
- multi-factor scenario grid experiments,
- publication tables and figures (`outputs/tables`, `outputs/figures`),
- per-run progress logs (`outputs/logs/heuristic_study_progress_*.log`).

For pilot runs, set `experiment.study_max_scenarios` to a small number in
`config/experiment_heuristic_study.yaml`.

## Manuscript model audit

- `docs/model_audit.md`: complete sets/parameters/variables/constraints mapping and ambiguity log.
- `docs/experiment_design.md`: planned protocol mapping to experiment scripts.
- `docs/assumptions.md`: all implementation assumptions.
- `IMPORTANT_QUESTIONS_FOR_AUTHOR.md`: unresolved assumptions to confirm.

## Exact vs Heuristic Modes

- Exact (`run-small`) can be enabled via `milp.enabled: true` and a compatible MILP backend.
- Heuristic study mode (`run-heuristic-study`) is designed to run with `milp.enabled: false`.
