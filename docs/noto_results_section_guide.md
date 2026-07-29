# Noto Empirical Results: Final Author Guide

## Readiness and Bottom Line

The repaired Noto computation is complete and ready for the paper. No computation was lost when Codex usage stopped: the detached jobs completed from atomic checkpoints.

The final evidence supports three claims:

1. Integrated exposure, capacity, and road mitigation materially lowers worst-case modeled loss.
2. Road retrofit has substantial incremental value under observed Noto access disruption.
3. The support-preserving ambiguity repair removes the unrestricted-TV support artifact.

The primary base-budget specification does **not** support a claim that DDA changes the selected `y`, `z`, or `w`. Policy stability remains after repairing the ambiguity set, testing four density caps, globally bounding the two-link continuous instance, and exhaustively evaluating the five-link grid. That stability is therefore an empirical result, not a solver failure. The separate scarcity extension reported later changes `w` but not `y` or `z`.

## Headline Result at `rho=0.10`

The primary specification is support-preserving TV with `kappa=2`.

| Policy | Worst-case modeled loss | Reduction from no investment | Reduction percent |
|---|---:|---:|---:|
| No investment | 9,367.190 | 0.000 | 0.00% |
| Exposure only | 7,875.130 | 1,492.060 | 15.93% |
| Capacity only | 7,246.554 | 2,120.636 | 22.64% |
| Exposure + capacity; no road | 5,794.712 | 3,572.478 | 38.14% |
| All sectors | 4,985.955 | 4,381.235 | 46.77% |

Road retrofit improves the optimized no-road plan by `808.756` modeled-loss units, or `13.96%` of the no-road objective. This is the paper's strongest practical result.

Use **worst-case modeled loss**, not predicted deaths, unless the survival parameters are independently validated against observed mortality.

## Selected Policy

The globally best policy on the declared five-level road grid is unchanged across all six ambiguity radii:

`y = [0.25, 1.00, 1.00, 0.00, 0.00]`.

Corridor order:

1. Kanazawa-Nanao: `0.25`
2. Nanao-Anamizu: `1.00`
3. Anamizu-Wajima: `1.00`
4. Anamizu-Noto: `0.00`
5. Anamizu-Suzu: `0.00`

At `rho=0.10`, the exposure plan renovates `38.47%` of Suzu. Capacity expansion adds `2,202.926` emergency-throughput units in Suzu and `330.394` in Noto. These are scenario throughput units, not observed added hospital beds.

## Repaired DDA Sweep

| `rho` | Best objective | Increase from `rho=0` | Road value | `delta_rho` |
|---:|---:|---:|---:|---:|
| 0.00 | 4,978.652 | 0.000 | 691.445 | 0.000 |
| 0.05 | 4,982.360 | 3.708 | 750.176 | 0.000 |
| 0.10 | 4,985.955 | 7.303 | 808.756 | 0.000 |
| 0.15 | 4,986.227 | 7.575 | 869.628 | 0.000 |
| 0.20 | 4,986.413 | 7.761 | 924.266 | 0.000 |
| 0.25 | 4,986.599 | 7.947 | 963.472 | 0.000 |

The `y`, `z`, and `w` norms relative to `rho=0` are zero up to numerical noise. The largest `z/w` norm is `1.66e-11`.

For the primary base-budget specification, the correct claim is: **DDA changes evaluation but not policy.**

## Why the Repair Matters

The repaired ambiguity set is:

`P_kappa(y) = {p >= 0: sum_s p_s=1, 0.5 sum_s |p_s-pi_s(y)| <= rho, p_s <= kappa pi_s(y)}`.

Under unrestricted TV, all positive probability shift entered zero-nominal states, including failures of fully retrofitted links. With `kappa=2`:

- mass added to zero-nominal states is exactly zero at every radius;
- no fully retrofitted link fails in an added-probability state;
- the maximum realized density ratio never exceeds `2`;
- total positive shift remains equal to the requested radius in the reported full-grid rows.

At `rho=0.10`, the ambiguity penalty falls from `127.258` under unrestricted TV to `7.303` under support-preserving TV, a `94.26%` reduction. At `rho=0.25`, it falls from `318.146` to `7.947`, a `97.50%` reduction. Most of the old ambiguity penalty was therefore a support-expansion artifact.

## Density-Cap Sensitivity

The coarse pilot tests `kappa in {1.5,2,5,10}` at all six radii. Every cap:

- preserves nominal support;
- respects its density limit;
- selects the same coarse road policy at every radius;
- gives `delta_rho=0`.

At `rho=0.25`, the coarse-grid ambiguity increase is `5.146`, `10.211`, `18.277`, and `18.651` for `kappa=1.5`, `2`, `5`, and `10`, respectively. The cap changes the evaluation penalty but not the optimizer. Do not select `kappa` to force policy switching.

## Near-Optimal Policies

At `rho=0.10`:

- one policy is within `0.01%` of the best;
- five policies are within `0.05%`, `0.10%`, and `0.50%`;
- the second policy is `[0,1,1,0.5,0]`, with gap `1.804` or `0.0362%`.

At `rho>=0.15`, only one policy remains within `0.05%`. The support repair slightly sharpens policy separation, but not enough to change the best policy.

## Scarcity and Resolution Check

The exhaustive five-level budget frontier initially produces a favorable-looking switch. At the baseline link costs, six coarse-grid phases are DDA-active, and 18 of 25 Link-2/Link-3 cost pairs contain at least one active phase. The strongest baseline-cost phase is:

| Quantity | Coarse five-level result |
|---|---:|
| Budget-multiplier interval | `[0.469420,0.479479)` |
| `rho=0` road vector | `[0,0.5,0.75,0,0]` |
| Positive-`rho` road vector | `[0,1,0,0,0]` |
| `delta_0.25` | `105.873` |
| Relative switching value | `1.951%` |

Do not use this result alone. A follow-up `0.05` grid on the two critical links, evaluated at the midpoint of that interval with the full five-link probability and recourse model, scans `313` feasible policies per radius and `1,878` exact policies overall. It selects `[0,1,0.2,0,0]` for all six radii and gives road-policy `delta_rho=0` in every row. Runtime is `3,056.25` seconds (`50.94` minutes), the maximum Algorithm 1 gap is below `1.7e-9`, and no support leakage occurs.

Therefore:

- the budget and cost analyses are valid coarse-resolution diagnostics;
- their apparent policy-switch region does not survive the refined critical-link grid;
- the scientifically defensible finding remains road-retrofit stability under scarcity;
- any claim of DDA-induced policy switching requires a refined-grid or continuous confirmation.

The reduced four-state continuous SBB endpoint diagnostic also selects `[1,0.208025]` at both `rho=0` and `rho=0.25`. Its 5,000-node absolute gaps are `2.736602` (`0.052521%`) and `1.665559` (`0.031661%`), respectively. These are small relative gaps but exceed the declared `0.1`-unit certificate tolerance. Report them as node- or time-limited diagnostics, not global certificates and not full five-link results.

## Failure-Moment Extension

The optional moment-aware ambiguity set adds bands on link marginals, pairwise joint failures, failed-link-count mean, and a fixed-center second moment. On the same `0.05` scarcity grid:

- all 1,878 policies are evaluated exactly;
- road `y=[0,1,0.2,0,0]` remains unchanged;
- exposure `z` remains unchanged;
- the `rho=0.25` objective falls from `5,370.685` to `5,305.867`;
- the ambiguity penalty falls by `64.818`, or `45.91%`;
- capacity `w` moves `221.586` units from Noto to Wajima for every positive radius;
- the capacity-adaptation value reaches `1.917` units (`0.0361%`) at `rho=0.25`.

This supports a narrower claim than a road-policy switch: **moment-aware DDA changes operational capacity deployment while the long-lived road plan remains stable.** The moment tolerances are scenario-calibrated and require a hazard ensemble before being presented as empirically estimated bounds.

At `rho=0.25`, tight, moderate, loose, and capped-TV profiles all select `[0,1,0.2,0,0]`. Their objectives are `5,261.947`, `5,305.867`, `5,367.556`, and `5,370.685`, respectively. Use this as tolerance sensitivity for evaluation and capacity adaptation, not as evidence that any one declared envelope is empirically correct.

## Global and Grid Certificates

### Two-link continuous instance

The nested two-link/four-state instance is globally bounded at a declared absolute tolerance of `0.1`, equivalent to less than `0.002%` relative gap.

- `rho=0`: exact SBB gap `0`.
- `rho=0.05`: direct 5,000-node SBB gives objective `4,971.667386`, lower bound `4,971.597003`, and gap `0.070383` (`0.001416%`).
- `rho=0.10` through `0.25`: the same feasible objective is obtained. Since the optimal value is nondecreasing in `rho`, the `rho=0.05` lower bound applies and yields the same `0.070383` certificate gap.

The certified road policy is `[0.924508,1]` and is stable over the sweep.

### Three-link continuous instance

The requested three-link/eight-state diagnostic is retained:

- `rho=0`: exact gap `0`;
- `rho=0.05`: exact gap `0` after 13 nodes;
- `rho=0.10` through `0.25`: 100-node objective `4,982.415078`, lower bound `4,981.634760`, gap `0.780318` (`0.015661%`).

Do not call the positive-radius `m=3` rows globally certified. Their gaps are small but node-limited.

### Five-link empirical grid

- Grid vectors scanned per radius: `3,125`.
- Budget-feasible policies evaluated exactly per radius: `996`.
- Total exact evaluations: `5,976`.
- Runtime with four workers: `6,442.33` seconds (`107.37` minutes).
- The selected road vector is globally best over the declared five-level grid, not over continuous `[0,1]^5`.

## Numerical Audit

Across all `5,976` fixed-`y` evaluations:

- maximum Algorithm 1 objective/lower-bound gap: `9.99e-7`;
- maximum Algorithm 1 iterations: `9`;
- maximum nominal probability-sum error: `2.22e-16`;
- maximum worst-case probability-sum error: `4.44e-16`;
- minimum reported probability: `0`;
- maximum single-policy runtime: `12.22` seconds.

## Data and Variable Status

Observed or directly anchored:

- municipality population and households for `P_rl`;
- fully destroyed dwellings for the severe-damage proxy `q_rl`;
- hospital beds;
- official corridor geometry, disrupted/recovered travel times, and restoration locations.

Constructed from observations:

- corridor failure scores `Phi_ij`;
- the five-corridor network abstraction and state travel penalties.

Scenario-calibrated:

- costs and sector budgets;
- operational capacity shares and throughput multiplier;
- retrofit effectiveness;
- `rho` and `kappa`.

The objective is a comparative modeled-loss measure. The data do not validate actual deaths, causal retrofit effectiveness, or the full latent model.

## Final Tables and Figures

The repair package creates seven summary table types in CSV and LaTeX:

1. `table_noto_support_sector_comparison`
2. `table_noto_support_model_comparison`
3. `table_noto_support_cap_sensitivity`
4. `table_noto_support_claim_diagnostics`
5. `table_noto_support_m2_certification`
6. `table_noto_support_m3_certification`
7. `table_noto_support_numerical_audit`

It creates five figures in PNG and SVG:

1. `fig_noto_support_01_penalty_comparison`
2. `fig_noto_support_02_support_leakage`
3. `fig_noto_support_03_cap_sensitivity`
4. `fig_noto_support_04_road_value`
5. `fig_noto_support_05_sector_contribution`

Recommended main-text items: sector comparison, model comparison, claim diagnostics, penalty figure, support-leakage figure, and road-value figure. Put cap sensitivity, full top-10/near-optimal tables, numerical audit, and `m=2/m=3` details in the appendix.

## Claims to Use

- Support-preserving TV eliminates probability mass on zero-nominal states.
- Integrated mitigation reduces the `rho=0.10` objective by `46.77%` relative to no investment.
- Road retrofit adds a `13.96%` improvement over the optimized no-road plan.
- The empirical road policy is globally best on the declared five-level grid.
- The two-link continuous instance is globally bounded within `0.1` objective units.
- In the primary base-budget Noto specification, DDA changes worst-case evaluation but not the selected policy.
- Under the moment-aware scarcity diagnostic, DDA changes capacity deployment but not road retrofit or renovation.

## Claims to Avoid

- For the primary base-budget specification, do not claim DDA changes `y`, `z`, or `w`; for the separate moment-aware scarcity diagnostic, report the observed `w` reallocation explicitly.
- Do not claim a meaningful road-retrofit switching value; the road-policy `delta_rho=0`.
- Do not call the `m=3`, `rho>=0.10` rows globally certified.
- Do not claim continuous global optimality for the five-link empirical problem.
- Do not call `Phi_ij`, costs, budgets, operational shares, `rho`, or `kappa` observed.
- Do not call the objective a validated prediction of deaths.
- Do not tune the ambiguity set merely to create a favorable policy switch.
- Do not present the coarse scarcity switch without its failed `0.05`-grid replication.
- Do not call the active-budget reduced SBB endpoint rows globally certified.
