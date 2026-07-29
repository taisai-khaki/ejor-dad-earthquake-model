# Noto Failure-Moment Ambiguity Results

## Research Question

The support-preserving capped-TV set bounds probability mass state by state but does not explicitly keep link-failure marginals, joint failures, or failed-link-count moments near their decision-dependent nominal values. This experiment tests whether adding those restrictions changes the Noto result, especially the apparent stability of the road-retrofit decision.

The original finite state support already makes every mean and standard deviation mathematically finite. The extension addresses empirical plausibility rather than mathematical unboundedness.

## Model Extension

Let `F_ls` equal one when link `l` fails in state `s`, and let `N_s=sum_l F_ls`. The ambiguity set intersects capped TV with the following fixed-`y` linear bands:

`|E_p[F_l]-E_pi[F_l]| <= a_M+r_M E_pi[F_l]`,

`|E_p[F_l F_r]-E_pi[F_l F_r]| <= a_J+r_J E_pi[F_l F_r]`,

`|E_p[N]-E_pi[N]| <= epsilon_mu`,

and a band on `E_p[(N-E_pi[N])^2]` around its nominal value.

The primary diagnostic uses:

- marginal tolerances `(r_M,a_M)=(0.25,0.02)`;
- joint-failure tolerances `(r_J,a_J)=(0.50,0.01)`;
- failed-link-count mean tolerance `epsilon_mu=0.15`;
- fixed-center second-moment tolerance `0.25 M_2(y)+0.05`;
- support-preserving density cap `kappa=2`.

These are declared scenario tolerances, not estimated earthquake moments. Their purpose is to test the mechanism, not to claim statistical identification from one event.

For fixed `y`, the adversarial problem remains a finite LP. The continuous SBB relaxation has not been extended with the new decision-dependent moment products, so the implementation rejects moment-constrained SBB and uses exact fixed-`y` enumeration.

## Computational Design

The experiment uses the strongest scarcity interval identified in the earlier budget diagnostic. Its midpoint is road-budget multiplier `0.4744494336`, giving `B_Y=1.0896860397`. Links 2 and 3 use a `0.05` grid while the other three road variables are fixed at zero. The full five-link state and recourse model is retained.

- Feasible road vectors per radius: 313.
- Ambiguity radii: `{0,0.05,0.10,0.15,0.20,0.25}`.
- Exact fixed-`y` evaluations: 1,878.
- Four-worker detached runtime: approximately 2,161 seconds, or 36.02 minutes.
- Every evaluation is atomically checkpointed and survives application closure.

## Objective and Road Policy

| `rho` | Capped-TV best objective | Moment-constrained best objective | Reduction from moment bounds | Best road `y` | Worst expected failed links |
|---:|---:|---:|---:|---|---:|
| 0.00 | 5,229.491 | 5,229.491 | 0.000 | `[0,1,0.2,0,0]` | 1.261587 |
| 0.05 | 5,259.039 | 5,258.957 | 0.082 | `[0,1,0.2,0,0]` | 1.402201 |
| 0.10 | 5,287.596 | 5,287.430 | 0.166 | `[0,1,0.2,0,0]` | 1.411587 |
| 0.15 | 5,316.086 | 5,304.187 | 11.898 | `[0,1,0.2,0,0]` | 1.411587 |
| 0.20 | 5,343.837 | 5,305.451 | 38.386 | `[0,1,0.2,0,0]` | 1.411587 |
| 0.25 | 5,370.685 | 5,305.867 | 64.818 | `[0,1,0.2,0,0]` | 1.411587 |

At `rho=0.25`, capped TV raises the objective by 141.194 relative to `rho=0`. The moment-constrained increase is 76.376. Explicit moment control therefore removes 64.818 units, or 45.91% of the capped-TV ambiguity penalty.

The road vector remains `[0,1,0.2,0,0]` at every radius, so the road-policy switching value is zero. This is not caused by a flat road landscape: at `rho=0.25`, the second-best vector `[0,1,0.15,0,0]` is 17.954 units or 0.3384% worse, and only one vector lies within 0.10% of the optimum.

## Tolerance Sensitivity

The `rho=0.25` endpoint is reoptimized over all 313 road vectors under tight and loose moment envelopes. The nominal `rho=0` reference is common to all profiles.

| Profile | Mean tolerance | Best objective | Worst expected failed links | Best road `y` | Full-policy adaptation value |
|---|---:|---:|---:|---|---:|
| Tight | 0.075 | 5,261.947 | 1.336587 | `[0,1,0.2,0,0]` | 0.640 |
| Moderate | 0.150 | 5,305.867 | 1.411587 | `[0,1,0.2,0,0]` | 1.917 |
| Loose | 0.300 | 5,367.556 | 1.561587 | `[0,1,0.2,0,0]` | 6.602 |
| Capped TV only | none | 5,370.685 | 1.648671 | `[0,1,0.2,0,0]` | 3.941 |

The road result is invariant across the tolerance bracket. Evaluation and the value of capacity adaptation are sensitive to the envelope, and the latter is not monotone. The moment tolerances should therefore be calibrated from hazard evidence rather than selected to obtain a preferred policy or adaptation value.

## What Changes by Decision Layer

The moment extension does not imply that every decision is invariant.

- Road retrofit `y`: unchanged over all six radii.
- Exposure renovation `z`: unchanged to numerical tolerance; Suzu remains at `0.384685`.
- Capacity expansion `w`: changes once ambiguity becomes positive.

At `rho=0`, expansion is allocated as follows:

- Wajima: 108.808;
- Suzu: 2,202.926;
- Noto: 221.586.

For every positive radius, 221.586 units move from Noto to Wajima:

- Wajima: 330.394;
- Suzu: 2,202.926;
- Noto: 0.

Holding the complete `rho=0` plan `(y,z,w)` fixed costs 0.685 units at `rho=0.05` and 1.917 units at `rho=0.25`. The latter is 0.0361% of the optimized robust objective. Thus, DDA changes the capacity-deployment decision, but its measured value is small. Under capped TV without moment bounds, the corresponding `rho=0.25` capacity-adaptation value is 3.941; moment control reduces rather than creates this effect.

## Behavior of Nature

The mean bound becomes active at `rho=0.10`. For the selected policy, nominal expected failed links equal 1.261587. At `rho=0.25`, capped TV raises this to 1.648671, whereas the moment-constrained adversary is capped at 1.411587.

At `rho=0.25`, the nontrivial binding restrictions are:

- upper marginal bound for Link 1;
- upper marginal bound for Link 3;
- lower marginal bound for Link 5;
- upper Link-3/Link-4 joint-failure bound;
- upper failed-link-count mean bound.

Nature still uses the complete TV radius at every positive `rho`. The extension therefore does not make ambiguity inactive; it redirects probability within a physically narrower moment envelope.

## Numerical Audit

Across all 1,878 exact evaluations:

- maximum Algorithm 1 gap: `6.78e-6`;
- maximum nominal and worst-case probability-sum error: `2.22e-16`;
- maximum support leakage: `0`;
- maximum density ratio: `2.0000000000000013`;
- maximum TV-radius excess: `1.67e-16`;
- maximum moment-bound violation: `1.11e-15`;
- maximum Algorithm 1 iterations: 14.

## Paper Interpretation

The moment extension successfully addresses the criticism that the adversary's aggregate failure behavior is insufficiently controlled. It materially changes worst-case evaluation at large radii and yields a small but genuine capacity-policy response.

It does not make the long-lived road-retrofit decision ambiguity-sensitive. This is evidence that road-policy stability is structural in the tested Noto scarcity instance, not an artifact of unconstrained failure moments or of the solution method.

Recommended claim:

> Adding decision-dependent failure-moment bounds reduces the high-radius ambiguity penalty and changes emergency-capacity deployment, while the preferred road-retrofit vector remains stable. The result distinguishes operational adaptation from robust long-lived infrastructure investment.

Do not claim that the moment extension changes the Noto road-retrofit policy. Also do not present the declared moment tolerances as statistically estimated until they are calibrated from a hazard ensemble or repeated-event evidence.

## Reproduction and Outputs

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_noto_moment_ambiguity_detached.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\status_noto_moment_ambiguity.ps1
```

- Main summary: `data_work/noto/moment_constrained_active_fine_grid/tables/table_noto_moment_envelope_summary.csv`
- Active bounds: `data_work/noto/moment_constrained_active_fine_grid/tables/table_noto_moment_envelope_bounds.csv`
- Link marginals: `data_work/noto/moment_constrained_active_fine_grid/tables/table_noto_moment_envelope_marginals.csv`
- Numerical audit: `data_work/noto/moment_constrained_active_fine_grid/tables/table_noto_moment_envelope_numerical_audit.csv`
- Tolerance sensitivity: `data_work/noto/moment_constrained_active_fine_grid/tables/table_noto_moment_tolerance_sensitivity.csv`
- Figures: `data_work/noto/moment_constrained_active_fine_grid/figures`
