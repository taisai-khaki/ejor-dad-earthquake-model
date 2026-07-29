# Noto DDA Mechanism: Additional Analyses and Final Interpretation

## Purpose

These analyses test whether the support-preserving decision-dependent ambiguity mechanism changes the Noto road-retrofit decision when the road budget is scarce. They also test whether any apparent policy change is robust to link-cost perturbations, decision-grid resolution, and a reduced continuous SBB formulation.

All computations use support-preserving capped total variation with `kappa=2`:

`P_kappa(y) = {p >= 0: sum_s p_s=1, 0.5 sum_s |p_s-pi_s(y)| <= rho, p_s <= kappa pi_s(y)}`.

## 1. Coarse Budget Frontier

The completed five-level archive is filtered at every distinct feasible road-budget threshold. The declared road grid is `y_ij in {0,0.25,0.50,0.75,1}` and the ambiguity sweep is `rho in {0,0.05,0.10,0.15,0.20,0.25}`.

At the baseline link costs, six policy phases are DDA-active. The strongest phase has road-budget multiplier interval `[0.469420,0.479479)`:

| `rho` | Coarse best `y` | Best objective | Objective using `rho=0` policy | `Delta_rho` |
|---:|---|---:|---:|---:|
| 0.00 | `[0,0.5,0.75,0,0]` | 5,280.690 | 5,280.690 | 0.000 |
| 0.05 | `[0,1,0,0,0]` | 5,317.397 | 5,339.412 | 22.016 |
| 0.10 | `[0,1,0,0,0]` | 5,346.068 | 5,389.883 | 43.815 |
| 0.15 | `[0,1,0,0,0]` | 5,374.557 | 5,440.038 | 65.481 |
| 0.20 | `[0,1,0,0,0]` | 5,402.348 | 5,490.171 | 87.824 |
| 0.25 | `[0,1,0,0,0]` | 5,427.630 | 5,533.503 | 105.873 |

At `rho=0.25`, the switching value is 105.873 modeled-loss units, or 1.951% of the optimized robust objective. This is a meaningful effect on the five-level grid.

## 2. Worst-Case Distribution Shift

At `rho=0.25`, the coarse nominal policy `[0,0.5,0.75,0,0]` produces the following critical-link pattern probabilities:

| Critical-link pattern | Nominal probability | Worst-case probability | Change |
|---|---:|---:|---:|
| Neither critical link fails | 0.663168 | 0.413168 | -0.250000 |
| Link 2 only | 0.200895 | 0.401789 | +0.200895 |
| Link 3 only | 0.104332 | 0.121832 | +0.017500 |
| Both critical links fail | 0.031605 | 0.063211 | +0.031605 |

The capped adversary doubles the Link-2-only and joint-failure pattern probabilities where allowed by nominal support. State-level results show that the largest additions are assigned broadly to high-loss states containing Link 2; nature does not target only the joint Link-2/Link-3 failure state.

Under the coarse ambiguity-dependent policy `[0,1,0,0,0]`, Link 2 is fully hardened. Link-2-only and joint-failure states then have zero nominal and zero worst-case probability. Nature instead raises the Link-3-only pattern from 0.543750 to 0.793750. In every case, mass added to zero-nominal states is exactly zero.

## 3. Link-Cost by Budget Sensitivity

The Link 2 and Link 3 costs are independently multiplied by `{0.8,0.9,1.0,1.1,1.2}`. For each of the 25 cost pairs, all distinct five-level budget phases over the multiplier range `[0.30,0.75]` are evaluated from the exact fixed-`y` archive.

- 18 of 25 cost pairs contain at least one coarse-grid DDA-active phase.
- The median active share of the scanned budget range is 13.411%.
- The maximum active share is 53.290%.
- At the baseline cost pair `(1,1)`, six active phases span 13.411% of the scanned range.

These findings show that the coarse-grid activation region shifts with relative link costs. They do not establish resolution-robust policy activation because only the baseline cost cell is replicated on the finer decision grid.

## 4. Refined Critical-Link Grid

The strongest coarse phase is retested at its budget midpoint, multiplier `0.4744494336`, corresponding to `B_Y=1.0896860397`. The two critical road variables use increments of `0.05`; the other three road variables are fixed at zero. The full five-link failure-state and recourse model is retained.

- Feasible policies per radius: 313.
- Exact evaluations: 1,878.
- Runtime: 3,056.25 seconds, or 50.94 minutes.
- Maximum Algorithm 1 gap over all evaluations: below `1.7e-9`.
- Mass added to zero-nominal states: zero.

| `rho` | Refined-grid best objective | Refined-grid best `y` | `Delta_rho` | Second-best gap |
|---:|---:|---|---:|---:|
| 0.00 | 5,229.491 | `[0,1,0.2,0,0]` | 0.000 | 4.883 |
| 0.05 | 5,259.039 | `[0,1,0.2,0,0]` | 0.000 | 14.589 |
| 0.10 | 5,287.596 | `[0,1,0.2,0,0]` | 0.000 | 14.618 |
| 0.15 | 5,316.086 | `[0,1,0.2,0,0]` | 0.000 | 14.618 |
| 0.20 | 5,343.837 | `[0,1,0.2,0,0]` | 0.000 | 14.982 |
| 0.25 | 5,370.685 | `[0,1,0.2,0,0]` | 0.000 | 14.982 |

The refined nominal optimizer already fully hardens Link 2 and partially hardens Link 3. Ambiguity therefore changes the objective but not the policy. The favorable-looking five-level switch is a discretization artifact.

## 5. Reduced Continuous SBB Diagnostic

A separate four-state SBB diagnostic retains only the two critical links and treats excluded links as intact. It uses the same active budget midpoint and warm-starts from `[1,0.2]`. Each endpoint is allowed up to 5,000 nodes.

| `rho` | Incumbent objective | Lower bound | Absolute gap | Relative gap | Continuous `y` | Nodes | Status |
|---:|---:|---:|---:|---:|---|---:|---|
| 0.00 | 5,210.459 | 5,207.722 | 2.737 | 0.0525% | `[1,0.208025]` | 5,000 | time-limited |
| 0.25 | 5,260.569 | 5,258.903 | 1.666 | 0.0317% | `[1,0.208025]` | 5,000 | node-limited |

The SBB policy is unchanged to numerical precision and its switching value is below `2e-11`. Both gaps exceed the declared 0.1-unit certificate tolerance, so these rows are diagnostics rather than global certificates. They nevertheless corroborate the refined-grid finding: the reduced continuous incumbent is stable between the nominal and high-ambiguity endpoints.

The reduced SBB instance is not the full five-link empirical model. Its results must not be used to claim continuous global optimality for the empirical instance.

## Final Decision for the Paper

The proposed statement that DDA changes the Noto road-retrofit policy under budget scarcity is not supported after resolution checking. The defensible conclusion for road retrofit is:

> A coarse five-level budget frontier identifies narrow regions in which ambiguity appears to change the preferred retrofit vector. However, the strongest region does not survive a 0.05-grid replication on the critical links, and a reduced continuous SBB diagnostic selects the same incumbent at `rho=0` and `rho=0.25`. In the Noto instance, support-preserving decision-dependent ambiguity changes worst-case evaluation but does not robustly change the retrofit policy, including under the tested scarcity regime. The exercise demonstrates why apparent optimizer changes must be accompanied by resolution and bound diagnostics.

A later moment-constrained extension preserves this road-policy conclusion but reveals a small capacity response: for positive `rho`, 221.586 expansion units move from Noto to Wajima. Its value reaches 1.917 units or 0.0361% at `rho=0.25`. Therefore, use “road-policy stability,” not “complete policy stability,” when discussing the scarcity experiment.

This is still a useful scientific result. It separates three questions that should not be conflated:

1. Does the ambiguity set represent physically possible states? Yes, after support preservation.
2. Does ambiguity change the worst-case value and distribution? Yes.
3. Does ambiguity robustly change the Noto investment policy? No, under the tested specifications.

## Reproducibility and Monitoring

The fine-grid and SBB jobs are restart-safe. Each completed fixed-`y` evaluation and each completed SBB radius is atomically checkpointed. Detached launchers write separate process and progress logs, avoiding the Windows shared-log lock that interrupted the first fine-grid attempt. Status scripts report the worker PID, checkpoint count, current radius, node count, incumbent, and lower bound.

Key commands:

```powershell
scripts\start_noto_active_fine_grid_detached.ps1
scripts\status_noto_active_fine_grid.ps1
scripts\start_noto_active_m2_sbb_detached.ps1 -RhoValues '0,0.25' -MaxNodes 5000 -TimeLimitSec 1800
scripts\status_noto_active_m2_sbb.ps1
```

## Output Files

- Distribution diagnostics: `data_work/noto/support_preserving_mechanism_analysis/tables/table_noto_active_critical_failure_patterns.csv`
- Cost-budget summary: `data_work/noto/support_preserving_mechanism_analysis/tables/table_noto_cost_budget_summary.csv`
- Fine-grid summary: `data_work/noto/support_preserving_active_fine_grid/tables/table_noto_active_fine_grid_summary.csv`
- Resolution comparison: `data_work/noto/support_preserving_active_validation/tables/table_noto_active_resolution_comparison.csv`
- Reduced SBB table: `data_work/noto/support_preserving_active_validation/tables/table_noto_active_critical_m2_sbb.csv`
- Consolidated figures: `data_work/noto/support_preserving_active_validation/figures`
