# Noto Acute-Response Configuration

## Purpose

The legacy empirical run uses the full structurally at-risk population, `P_r q_r`, as
if every person requires time-critical facility access. That is not an appropriate
interpretation of the Noto destroyed-dwelling layer. This configuration changes the
*decision outcome* to timely access for a declared share of the post-damage population.
It does not estimate earthquake mortality.

The legacy run is retained as a sensitivity benchmark. It must not be used alone to
argue that decision-dependent ambiguity is inactive merely because its total loss is
dominated by an intact-state travel-time loss.

## Outcome definition

For zone `r`, let `D_r(z) = P_r q_r (1-z_r)`. The exact fixed-`y` evaluator now uses

```text
E_r(z) = eta_r D_r(z)
L_s(z, w, y) = sum_r delta_r D_r(z) + sum_r E_r(z) - survivors_s(z, w, y)
```

where:

- `eta_r` is the time-sensitive fraction of the structurally affected population;
- `delta_r` is an immediate-loss fraction that is not changed by facility access; and
- `survivors_s` is the recourse result. With a response threshold, it counts timely
  facility access rather than clinical survival.

The unmodeled remainder, `(1 - eta_r - delta_r) D_r(z)`, is not treated as a death or
as a patient who needs emergency transport. This prevents the model from turning the
building-damage proxy into an unsupported mortality estimate.

## Pilot design

The restart-safe pilot at `data_work/noto/acute_access_pilot` declares:

| Quantity | Value | Interpretation |
|---|---:|---|
| `eta_r` | 0.25 for every zone | Scenario fraction requiring time-sensitive facility access |
| `delta_r` | 0.00 | Direct losses are reported separately, not folded into the access objective |
| Capacity throughput | 1.0 per operational bed | One acute response window, not four turnover cycles |
| Response threshold | 60 minutes | Timely-access planning threshold, not a universal clinical survival curve |
| Residual link risk | 10% of baseline `Phi_ij` | Existing residual-risk scenario |
| Failed-link delay reduction | 50% at full retrofit | Existing conditional-performance scenario |
| Ambiguity set | capped TV, `kappa=2` | Support-preserving decision-dependent ambiguity |
| Radius screen | `rho in {0, 0.25}` | Fast exact mechanism screen before a full sweep |

The values for `eta_r`, the capacity window, and the threshold are **scenario
parameters**, not observations. They must be accompanied by the prespecified
sensitivity grid `eta in {0.15, 0.25, 0.35}` and threshold in `{30, 60, 90}` minutes
before they are presented as an empirical practical recommendation.

## Reporting requirements

1. Label the primary metric as **time-sensitive people without timely access**, not
   deaths or total earthquake casualties.
2. Report the raw structural exposure `sum_r P_r q_r` separately from the modeled
   access-responsive exposure `sum_r eta_r P_r q_r`.
3. Replace aggregate-only Table 4 with the objective decomposition table. At a minimum
   show the common state-loss floor, nominal loss, robust loss, ambiguity premium,
   state-loss range, TV mass moved, and the selected `y`, `z`, and `w`.
4. Report `rho=0` policy loss under every positive radius and the near-optimal policy
   counts. A policy switch without a material value difference is not a substantive
   decision change.
5. Do not choose the threshold, demand fraction, or capacity horizon after observing a
   favorable retrofit switch.
