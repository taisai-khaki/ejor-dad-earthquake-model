# EJOR DR-DAD Earthquake Model

> **Public replication release.** The versioned code, frozen configuration,
> paper-ready Noto results, and output checksums are in this repository. Raw
> third-party source data, restart checkpoints, and machine-specific logs are
> deliberately excluded. See `docs/PUBLIC_RELEASE.md` and
> `results/noto/acute_access_graded_v4/README.md` before reproducing results.

This package implements the model in Sections 3-4 of `DAD_EJOR (2).pdf`:

- first-stage renovation `z_rl`, road retrofitting `y_ij`, and capacity expansion `w_k`;
- second-stage relief dispatch LP `Q(s,z,w)`;
- decision-dependent nominal network-state probabilities `p_hat_s(y)`;
- unrestricted or support-preserving total-variation ambiguity with exact worst-case reweighting;
- fixed-retrofitting distribution-generation evaluation;
- a spatial branch-and-bound scaffold with McCormick node relaxations for small/reduced networks.

## Install

```powershell
cd C:\Users\L03128674\projects\ejor_dad_nepal
python -m pip install -e . pytest
```

## Run the demo

```powershell
cd C:\Users\L03128674\projects\ejor_dad_nepal
python .\examples\synthetic_run.py
```

## Turkey validation workflow

Use Zenodo + HOTOSM/HDX + OSM as the core replacement stack:

1. Building damage: Zenodo record `18437501` and HDX `hotosm_tur_destroyed_buildings`.
2. Building denominator: HDX `hotosm_tur_buildings` or Zenodo `GBA_building_footprint`.
3. Roads: OSM roads from a local extract, Overpass, OSMnx, or Geofabrik.
4. Hazards: Zenodo `PGV`, `DEM`, `Fault`, `Lithology`, and `Epicenter` rasters.
5. Population/facilities: WorldPop/GHS-POP/census plus OSM hospitals and emergency facilities.

Run the first lightweight ingestion step:

```powershell
cd C:\Users\L03128674\projects\ejor_dad_nepal
$env:PYTHONPATH = "C:\Users\L03128674\projects\ejor_dad_nepal\src"
python .\examples\turkey_first_step.py
```

This writes metadata tables and grid-level destroyed-building counts to `data_work\turkey`. It does not download the full 585 MB Zenodo archive or the large all-buildings export.

Generate paper tables with checkpoints:

```powershell
cd C:\Users\L03128674\projects\ejor_dad_nepal
powershell -ExecutionPolicy Bypass -File .\scripts\run_turkey_paper_results_resumable.ps1
```

Start the same computation detached from Codex/Desktop so it continues if the app is closed:

```powershell
cd C:\Users\L03128674\projects\ejor_dad_nepal
powershell -ExecutionPolicy Bypass -File .\scripts\start_turkey_paper_results_detached.ps1
```

Check status:

```powershell
cd C:\Users\L03128674\projects\ejor_dad_nepal
powershell -ExecutionPolicy Bypass -File .\scripts\status_turkey_paper_results.ps1
```

Checkpoint files are written under `data_work\turkey\paper_tables\checkpoints`; rerunning skips completed scenarios unless `--force` is passed to `run_turkey_paper_results_resumable.ps1`.

## Noto validation workflow

The 2024 Noto Peninsula case uses official Japanese sources for municipality population and households, dwelling damage, daily road-restoration GIS, observed intercity travel times, and pre-event hospital beds. Link probabilities, operational capacity shares, costs, and budgets remain explicitly constructed or scenario-calibrated.

Prepare the compact model inputs after downloading the raw files listed in `data_work\noto\prepared\noto_source_manifest.csv`:

```powershell
cd C:\Users\L03128674\projects\ejor_dad_nepal
$env:PYTHONPATH = "$PWD\src"
python .\examples\noto_prepare_data.py
```

Run the coarse pilot independently of Codex/Desktop:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_noto_access_experiment_detached.ps1 -Mode pilot -Workers 4
powershell -ExecutionPolicy Bypass -File .\scripts\status_noto_access_experiment.ps1 -Mode pilot
```

Run the five-level exhaustive grid after reviewing the pilot:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_noto_access_experiment_detached.ps1 -Mode full -Workers 4
powershell -ExecutionPolicy Bypass -File .\scripts\status_noto_access_experiment.ps1 -Mode full
```

Create the paper-ready decision, claim-diagnostic, numerical-audit, and figure files after a run completes:

```powershell
$env:PYTHONPATH = "$PWD\src"
python .\examples\noto_postprocess_results.py --mode full --paper-rho 0.10
python .\examples\noto_make_figures.py --mode full --paper-rho 0.10
```

Every completed fixed-`y` evaluation is atomically checkpointed. Restarting the same mode skips existing checkpoints, so closing the app or interrupting the wrapper does not discard completed policies. The status command reports exact evaluations completed, rolling throughput, ETA, active process identifiers, and the current log.

Run the repaired support-preserving ambiguity analysis independently of Codex/Desktop:

```powershell
# Four-cap coarse diagnostic
powershell -ExecutionPolicy Bypass -File .\scripts\start_noto_support_preserving_detached.ps1 -Mode pilot -DensityCaps "1.5,2,5,10" -Workers 4
powershell -ExecutionPolicy Bypass -File .\scripts\status_noto_support_preserving.ps1 -Mode pilot

# Primary five-level grid with the predeclared cap kappa=2
powershell -ExecutionPolicy Bypass -File .\scripts\start_noto_support_preserving_detached.ps1 -Mode full -DensityCaps 2 -Workers 4
powershell -ExecutionPolicy Bypass -File .\scripts\status_noto_support_preserving.ps1 -Mode full

# Two-link/four-state tolerance certificate; rho>0.05 follows by monotonicity
powershell -ExecutionPolicy Bypass -File .\scripts\start_noto_support_m3_detached.ps1 -NumLinks 2 -DensityCap 2 -MaxNodes 5000 -TimeLimitSec 3600 -MonotonicityAnchor 0.05
powershell -ExecutionPolicy Bypass -File .\scripts\status_noto_support_m3.ps1 -NumLinks 2

# Requested three-link/eight-state SBB diagnostic
powershell -ExecutionPolicy Bypass -File .\scripts\start_noto_support_m3_detached.ps1 -NumLinks 3 -DensityCap 2 -MaxNodes 100
powershell -ExecutionPolicy Bypass -File .\scripts\status_noto_support_m3.ps1 -NumLinks 3

# Final comparison tables and figures after all runs complete
$env:PYTHONPATH = "$PWD\src"
python .\examples\noto_support_postprocess.py

# Reuse the completed exact fixed-y archive for tighter B_Y sensitivity
$env:PYTHONPATH = "$PWD\src"
python .\examples\noto_tight_budget_analysis.py --budget-multipliers "0.4,0.5"

# State-shift and 5x5 cost-budget robustness diagnostics
python .\examples\noto_mechanism_robustness.py

# Restart-safe 0.05 critical-link resolution check
powershell -ExecutionPolicy Bypass -File .\scripts\start_noto_active_fine_grid_detached.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\status_noto_active_fine_grid.ps1

# Reduced four-state continuous SBB endpoint diagnostic
powershell -ExecutionPolicy Bypass -File .\scripts\start_noto_active_m2_sbb_detached.ps1 -RhoValues "0,0.25"
powershell -ExecutionPolicy Bypass -File .\scripts\status_noto_active_m2_sbb.ps1

# Coarse-versus-fine and SBB comparison figures
python .\examples\noto_active_validation_postprocess.py

# Restart-safe 0.05 grid with link-marginal, joint-failure, and count-moment bounds
powershell -ExecutionPolicy Bypass -File .\scripts\start_noto_moment_ambiguity_detached.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\status_noto_moment_ambiguity.ps1
python .\examples\noto_moment_sensitivity_postprocess.py
```

The repaired ambiguity set adds `p_s <= kappa * pi_s(y)`. Consequently, a state with zero decision-dependent nominal probability must retain zero adversarial probability. Omitting `ambiguity_density_cap` preserves the original unrestricted-TV behavior for reproducibility.

The optional `FailureMomentEnvelope` further bounds link-failure marginals, pairwise joint failures, the expected number of failed links, and the fixed-center second moment around their decision-dependent nominal values. The fixed-`y` adversary is then solved as a linear program. The current continuous SBB relaxation does not include these extra decision-dependent constraints and therefore rejects moment-constrained instances explicitly; use exact fixed-`y` enumeration for this specification.

The tight-budget analysis does not rerun the optimization oracle. Because `B_Y` is a separate road-feasibility constraint, every policy feasible under a smaller `B_Y` is already present in the completed full-grid archive with the same exact objective, `z`, `w`, and probability distributions. The script checks archive completeness, writes results atomically, evaluates the requested budget multipliers, and scans every archived budget threshold so any mechanism-active interval is reported rather than selected informally.

The interpretation and paper-claim boundaries for this diagnostic are recorded in `docs\noto_tight_budget_sensitivity.md`.

## Practical residual-risk Noto workflow

The practical extension preserves a residual link-failure floor and lets road retrofit reduce the conditional access delay after a failed link. It uses support-preserving capped-TV ambiguity and exact fixed-`y` enumeration on the declared grid; continuous SBB intentionally rejects this performance-adjusted extension until its relaxation is extended.

```powershell
# Default: floor = 10% of baseline Phi; full retrofit removes 50% of failed-link delay.
powershell -ExecutionPolicy Bypass -File .\scripts\start_noto_practical_resilience_detached.ps1 -Mode full -Workers 4
powershell -ExecutionPolicy Bypass -File .\scripts\status_noto_practical_resilience.ps1
```

Completed policies are atomically checkpointed in `data_work\noto\practical_resilience\checkpoints`, so restarting after an app or terminal interruption reuses the existing exact evaluations. See `docs\noto_practical_resilience.md` for the formulation, tables, and claim boundaries.
## Nepal validation workflow

The Kathmandu Living Labs / Nepal earthquake portal provides the survey layer that can calibrate exposure and building damage. The code expects local CSV exports, because the portal URL may need browser interaction.

1. Aggregate survey rows with `zone_exposure_from_survey()` to estimate `P_rl` and `q_rl`.
2. Build OSM/HOT road links externally or with GeoPandas/OSMnx, then score `Phi_ij` with `road_failure_probability_score()`.
3. Choose facilities from OSM/humanitarian layers and set `w_k^0` via observed capacity or `capacity_from_facility_type()`.
4. Build `zones.csv`, `links.csv`, `centers.csv`, and optional `travel_times.csv` following `data_schema`.
5. Use `build_instance_from_tables()` and run `evaluate_fixed_y()` or `solve_global_sbb()` on a reduced candidate network.

Use the survey to validate the exposure/damage layer, not the whole model: link-failure probabilities, capacity, costs, budgets, and ambiguity radius remain scenario-calibrated unless separate ground-truth data are added.

## Data columns

- `zones.csv`: `zone_id`, `population`, `collapse_fraction`, `renovation_cost`, optional `region`, optional `node`.
- `links.csv`: `link_id`, `tail`, `head`, `failure_probability`, `retrofit_cost`, optional `travel_time`.
- `centers.csv`: `center_id`, `node`, `existing_capacity`, `capacity_unit_cost`.
- `travel_times.csv`: `state_id`, `center_id`, `zone_id`, `travel_time`.

