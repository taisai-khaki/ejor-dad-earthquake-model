# Frozen Noto Paper Results

This directory is the curated, public subset of the final Noto computation.
It is generated from the local run archive by
`scripts/build_public_release.ps1`.

## Contents

- `frozen_inputs/`: the exact run design, reproducibility metadata, and
  practitioner-layer configuration.
- `base_tables/`: paper-ready tables and LaTeX exports for the frozen final
  joint model.
- `correlated_facility/`: exhaustive correlated-facility grid tables and
  configuration snapshots.
- `operational_stage2/`: epsilon-constrained max-min service results.
- `sensitivity/density_cap/`: completed kappa-bar sensitivity tables.
- `sensitivity/graded_response/`: completed lower/base/higher response-knot
  sensitivity tables, diagnostics, and run metadata.
- `figures/maps/`: publication maps, captions, and their generation manifest.
- `SHA256SUMS.txt`: SHA-256 hashes for integrity checking.

The archived density-cap and graded-response experiments each enumerate the
full five-level road grid subject to the road budget, which yields 996
candidate policies per scenario. The density-cap release contains eight
completed grids; the graded-response release contains four completed grids
plus the frozen base rows at rho equal to 0.10 and 0.25.

An early, infeasible lower-response pilot is intentionally excluded. It was
superseded by the documented lower-credit response curve in the completed
graded-response release.

Raw source layers, solver checkpoints, and execution logs are not included.
See `docs/PUBLIC_RELEASE.md` for the data-access policy and rebuild steps.
