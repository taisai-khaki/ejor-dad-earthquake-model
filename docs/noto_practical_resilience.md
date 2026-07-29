# Practical Residual-Risk Noto Experiment

## Purpose

This experiment tests a practitioner-oriented extension of the DR-DAD road-retrofit layer. It does not assume that a fully strengthened road becomes impossible to fail, and it allows strengthening to reduce the access delay when a failure nevertheless occurs. The experiment uses the same five Noto candidate corridors, 32 no-tail failure states, and support-preserving capped total-variation ambiguity set as the repaired Noto analysis.

## Revised road mechanism

For every candidate corridor `(i,j)`, let `Phi_ij` be its baseline scenario-calibrated failure probability and let `floor_ij` be a residual failure floor. The nominal failure probability is

`phi_ij(y_ij) = floor_ij + (Phi_ij - floor_ij) (1 - y_ij)`.

Thus `y_ij = 1` leaves `floor_ij`, rather than forcing the modeled failure probability to zero. The default experiment sets `floor_ij = 0.10 Phi_ij`; this is a transparent sensitivity parameter, not an observed engineering reliability value.

For a state `s`, precomputed intact travel time is augmented by the delay penalties of failed corridors:

`tau_kr^s(y) = tau_kr^intact + sum_(i,j in F_s) Delta_tau_kr,ij (1 - gamma_ij y_ij)`.

`Delta_tau_kr,ij` is the observed/scenario-calibrated Noto disruption penalty already used by the access network. The default `gamma_ij = 0.50` means full retrofit removes one half of that conditional delay. Intact-state travel time is unchanged.

This formulation creates two physically distinct road-retrofit effects:

- It lowers the frequency of link failure, but does not assume perfect elimination.
- It reduces conditional disruption severity or recovery delay when the link fails.

## Ambiguity and solution method

The adversary uses support-preserving capped total variation:

`sum_s |p_s - pi_s(y)| <= 2 rho`, `p_s <= kappa pi_s(y)`, and `sum_s p_s = 1`.

The run uses `kappa = 2`. The density cap prevents the adversary from assigning mass to a nominally impossible state and limits amplification of feasible nominal states.

For each discrete road plan `y`, Algorithm 1 still solves the `z,w` recourse problem exactly because both the state probabilities and the performance-adjusted travel-time matrices are fixed. The five-link calculation exhaustively evaluates every grid-feasible `y` in `{0, .25, .50, .75, 1}^5`; its conclusion is globally optimal **on that declared grid**, not a continuous global optimum.

The continuous SBB implementation explicitly rejects instances with retrofit-dependent conditional delays. Its current relaxation only models the probability products, not the new `y`-dependent state-loss surfaces. This is an honest certification boundary, not an error in the fixed-`y` enumeration.

## Outputs and interpretation

The experiment writes restart-safe checkpoints and these paper-ready tables in `data_work/noto/practical_resilience/tables`:

- `table_noto_practical_summary`: best exact grid policy, `z`, `w`, effective link probabilities, value of the nominal policy at every `rho`, and the ambiguity diagnostic `Delta_rho`.
- `table_noto_practical_top10`: the ten best road policies and their objective gaps.
- `table_noto_practical_near_optimal`: counts within 0.01%, 0.05%, 0.10%, and 0.50% of the grid optimum.
- `table_noto_practical_heuristics`: exact values of least-cost-first, baseline-failure-per-cost, and failure-delay-per-cost policies.
- `table_noto_practical_probability_shifts`: nominal and worst-case state probabilities.
- `table_noto_practical_sector_comparison`: no-investment, no-road-retrofit, and all-sector values.

A policy change across ambiguity radii is not required for the formulation to be useful. The practical evidence is stronger when the exact model has a meaningful loss reduction relative to transparent heuristics, identifies a small near-optimal set, and reports when ambiguity leaves a robust policy unchanged. If many policies are near-equivalent, the result should be reported as policy flatness rather than as a uniquely prescriptive recommendation.

## Restart and monitoring

Run the detached calculation:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_noto_practical_resilience_detached.ps1 -Mode full -Workers 4
```

Monitor it at any time:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\status_noto_practical_resilience.ps1
```

Every fixed-`y` result is atomically checkpointed. Restarting the same command reuses completed evaluations, including after the app or terminal closes.

## Calibration limits

The Noto population, dwelling damage, facility beds, roads, and observed travel-time inputs are documented in `docs/noto_methods_and_data.md`. Link probabilities, residual-risk ratios, delay-reduction factors, costs, operational capacity shares, and budgets are still scenario-calibrated. The paper should therefore present sensitivity analysis over the residual floor, delay-reduction factor, link cost, and road budget before making a policy-specific claim.