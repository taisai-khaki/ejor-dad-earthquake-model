# Public Reproducibility Release

## Scope

This repository is the public computational companion for the paper's final
joint Noto Peninsula analysis. It contains the model implementation, final
paper-ready tables and maps, frozen parameter files, and scripts used to
build the public artifact bundle.

The repository does not redistribute raw third-party data, intermediate
checkpoints, process identifiers, or solver logs. Those files are both large
and subject to the source providers' terms. Data provenance and acquisition
requirements are documented in `docs/noto_methods_and_data.md` and
`data_schema/turkey_sources.md`; the final Noto source manifest is preserved
in the released results.

## Included Final Analyses

- the correlated-facility, 128-state, five-link exhaustive Noto grid;
- the operational Stage-2 max-min service analysis;
- the density-cap sensitivity for kappa-bar equal to 1.5 and 3.0 at rho equal
  to 0.075, 0.10, 0.125, and 0.25;
- the lower, base, and higher graded-response sensitivity at rho equal to
  0.10 and 0.25; and
- the maps and tables used to report the final configuration.

## Rebuild the Public Artifact Bundle

After completing the local Noto analyses and placing their output under
`data_work/noto/acute_access_graded_v4`, run:

    powershell -ExecutionPolicy Bypass -File .\scripts\build_public_release.ps1

The script copies only the frozen inputs, final tables, selected diagnostics,
and publication maps into `results/noto/acute_access_graded_v4`. It does not
copy raw data, checkpoints, or logs. It then creates `SHA256SUMS.txt` for the
released artifacts.

## Software Setup

Install the package and its test dependencies with:

    python -m pip install -e .[test]

Use the optional `geo` dependency group only when rebuilding the spatial input
layers. The final enumeration scripts use the dependencies declared in
`pyproject.toml`.

## Interpretation Boundary

The sensitivity release supports reproducible evaluation of how the density
cap and graded-response curve affect robust values and policy rankings. The
two leading road policies are often close in value, so the results should be
reported as a robustness and near-optimality diagnostic rather than as evidence
of a universally unique retrofit prescription.
