# Assessment of the reviewer recommendations

## Scope

This note separates mathematically valid improvements from claims that the
available computations cannot support. The probability-relaxation ablation was
run on the static Turkey no-tail benchmark at rho = 0.10. It is not a
certificate for the later model with retrofit-dependent conditional delays,
which the continuous SBB implementation deliberately rejects.

The restart-safe output is stored in:

- data_work/turkey/sbb_relaxation_ablation/root_bounds.csv
- data_work/turkey/sbb_relaxation_ablation/status.json
- data_work/turkey/sbb_relaxation_ablation/checkpoints

Every completed root task is atomically checkpointed. The six-link rung was
also launched as a detached Windows process with separate stdout and stderr
logs, so the calculation did not depend on the desktop app remaining open.

## Root-bound ablation

The reviewer correctly asked for an ablation rather than assuming that a
reduced-space probability representation is tighter. The completed results
are:

| Candidate links | States | Variant | Root raw lower bound | Valid root lower bound | LP variables |
|---:|---:|---|---:|---:|---:|
| 2 | 4 | Product tree | -60,684.736 | 0.000 | 38 |
| 3 | 8 | Product tree | -141,342.025 | 0.000 | 63 |
| 4 | 16 | Product tree | -172,760.166 | 0.000 | 120 |
| 5 | 32 | Product tree | -184,026.202 | 0.000 | 249 |
| 5 | 32 | Corner boxes | -184,026.202 | 0.000 | 121 |
| 5 | 32 | Corner boxes plus link cuts | -184,026.202 | 0.000 | 121 |
| 6 | 64 | Product tree | -190,701.508 | 0.000 | 538 |

The three five-link root values agree to numerical precision. Corner boxes
and the affine link cuts therefore reduce the LP dimension, but do not improve
the root lower bound in this instance. This is a decisive negative result for
the proposed root-bound repair: it does not repair the weak five-link
certificate.

The equality of the root bounds does not prove that the variants have
identical branch-and-bound performance. It does show that a long equal-time
five-link rerun is not justified by the reviewer's stated root-bound gate:
the valid root lower bound remains zero. The earlier two-link product-tree
run did close at 106,328.472, so the small-scale SBB implementation remains a
valid diagnostic; its root lower bound is simply weak before branching.

## Recommendations to retain

1. **Support-preserving capped total variation.** Keep the statewise cap
   p_s <= kappa pi_s(y). It prevents the adversary from assigning mass to a
   nominally impossible state and gives a meaningful likelihood-ratio
   interpretation.

2. **Residual link-failure floors.** Keep an engineering residual floor when
   full retrofit means strengthening rather than complete replacement. This
   improves physical interpretation. With strictly positive floors, however,
   most nominal state probabilities remain positive, so the support argument
   is a bound on amplification rather than an elimination claim.

3. **Exact fixed-y evaluation and exhaustive implementable-grid search.**
   These are the reliable empirical solver. They give an exact optimum over
   the declared finite project grid and remain valid for the conditional-delay
   extension.

4. **A transparent SBB ladder and ablation.** Include it only as a method
   diagnostic. Report the two-link closure, the zero root bounds, the
   node/time limits, and the V1/V2/V3 result honestly. It is useful evidence
   about the method's scalability, not evidence of a continuous five-link
   optimum.

5. **Grid warm starts.** Use them in any future SBB runs because they give a
   reproducible upper bound. They cannot repair the lower bound: a warm start
   changes the incumbent only, while certification requires a stronger valid
   lower bound.

## Recommendations not to adopt as written

1. **Do not call corner bounds exact as a relaxation.** The interval extrema
   for each individual state probability are exact over a node box, but
   independent state boxes allow jointly inconsistent probabilities. The
   product tree retains y--pi coupling. Link cuts validly strengthen corner
   boxes but do not make them exact and did not improve this root bound.

2. **Correct the rare-state transfer statement.** For a receiver set T, the
   extra mass is bounded by
   min{rho, (kappa - 1) pi(T), 1 - pi(T)}.
   The final term is necessary because mass must be removed from outside T.

3. **Do not claim adversarial non-reversal whenever a retrofit merely reduces
   failure probability.** The cap gives p_s = 0 only when pi_s(y) = 0.
   Positive residual failure floors normally prevent that condition.

4. **Do not apply the static SBB certificate to the practical conditional-delay
   model.** In the current model, road retrofit affects both nominal state
   probabilities and conditional travel time. The static SBB relaxation
   contains only the probability products and explicitly rejects that
   extension.

5. **Do not use a constructed mechanism-active calibration as evidence that
   ambiguity changes the empirical policy.** The earlier apparent policy
   switch was resolution-sensitive. The Turkey, Nepal, and practical Noto
   diagnostics instead show that ambiguity can change the robust value while
   leaving the meaningful policy stable or tied.

## What this resolves

The reviewer recommendations improve the paper's mathematical consistency,
reproducibility, and reporting discipline. They do not make decision-dependent
ambiguity automatically policy-active and they do not fix continuous
five-link SBB certification. The central practical result should therefore
be framed as follows:

- the model selects an exact, implementable-grid policy under an explicitly
  calibrated ambiguity set;
- capped ambiguity changes worst-case evaluation without assigning mass to
  unsupported states;
- policy changes are reported only when they survive grid-resolution and
  complete-policy diagnostics;
- continuous SBB is a reduced-instance verification tool, not the empirical
  five-link solver.

To obtain a continuous five-link certificate would require a materially
stronger global formulation, not simply a smaller probability lift. For the
paper's practitioner goal, the defensible route is to make the finite
implementation grid explicit and report its exact enumeration certificate.
