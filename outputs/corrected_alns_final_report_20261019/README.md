# Corrected ALNS Diagnostic and Fair Comparison

This report corrects the earlier ALNS benchmark without changing the ALNS algorithm.

- Base seed: `2026`; ALNS seed: `2237` (`base seed + 211`).
- ALNS protocol: 40 destroy-and-repair iterations, `alns_time_limit_seconds: 0`.
- Coverage: 720 unique instances; all rows completed 40 iterations, recorded 40 candidate-reassignment evaluations, and were feasible.
- Observed ALNS wall time: maximum `213.621624` seconds; the final matched comparison uses a common 240-second wall-clock budget.
- The M5/M9 results are reused from the same-source, same-seed full run because their recorded maximum runtimes are below 240 seconds (`121.716647` seconds for M9).

## Files

- `corrected_alns_runtime_by_size.csv`: corrected ALNS iterations, feasibility, objective, and wall time by size.
- `table_nils_vs_corrected_alns_matched_wallclock_by_size.csv`: final M5--ALNS comparison.
- `table_nils_r_vs_corrected_alns_matched_wallclock_by_size.csv`: final M9--ALNS comparison.
- `corrected_alns_protocol.json`: machine-readable protocol and validation record.

The corrected ALNS raw rows are stored in `../alns_comparison_beta010_full720_corrected_40iter_20261019/tables/alns_full_alns_runs_partial.csv`.
