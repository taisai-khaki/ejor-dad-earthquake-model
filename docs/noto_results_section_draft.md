# Draft Empirical Results Section: 2024 Noto Peninsula Earthquake

## Empirical Setting and Data

We evaluate the model using the 2024 Noto Peninsula earthquake in Ishikawa Prefecture, Japan. The case is selected because public official data jointly describe municipality population, dwelling damage, pre-event hospital beds, intercity road geometry, restoration locations, and disrupted and recovered travel times. This combination provides unusually direct evidence for the transportation-access mechanism that was only proxy-calibrated in the Turkey case.

The empirical stack combines 2022 municipality population and household counts from Ishikawa Prefecture, fully destroyed dwelling counts from Ishikawa damage report No. 162, road-restoration GIS and intercity travel-time observations from the Ministry of Land, Infrastructure, Transport and Tourism, and FY2023 hospital beds from the Ministry of Health, Labour and Welfare. Five municipalities—Nanao, Wajima, Suzu, Anamizu, and Noto—define demand zones. Kanazawa is retained as an external high-capacity center, producing six candidate aid centers.

For municipality `rl`, `P_rl` is observed pre-event population. The severe-damage proxy `q_rl` is fully destroyed dwellings divided by pre-event households. Consequently, `P_rl q_rl` assumes the same mean household size among destroyed and non-destroyed dwellings. The five zones contain 106,022 residents and 12,108.462 baseline at-risk persons under this proxy. These data calibrate exposure and severe dwelling damage; they do not provide person-level casualties conditional on building damage.

## Road and Capacity Construction

The road abstraction contains five official intercity corridors: Kanazawa-Nanao, Nanao-Anamizu, Anamizu-Wajima, Anamizu-Noto, and Anamizu-Suzu. Recovered travel times are 45, 30, 30, 40, and 50 minutes, respectively. Post-earthquake disrupted times are 50, 100, 80, 70, and 90 minutes. State-specific failures therefore add observed penalties of 5, 70, 50, 30, and 40 minutes to affected routes.

Baseline link-failure probabilities are constructed from observed delay ratios and initial restoration-point exposure using a scoring rule fixed before optimization. The resulting values are 0.1350, 0.4650, 0.5438, 0.3157, and 0.3759. They are scenario probabilities rather than repeated-event frequencies. Retrofit reduces the decision-dependent nominal probability through `Phi_ij(y)=Phi_ij(1-y_ij)`.

The hospital source reports 8,420 pre-event beds across the six municipalities. Existing emergency throughput combines observed beds with declared operational shares and a four-person throughput multiplier, yielding 8,444.4 modeled units. Capacity expansion represents emergency or temporary throughput, not observed additional hospital beds. Costs, budgets, operational shares, throughput, retrofit effectiveness, and ambiguity parameters remain scenario-calibrated.

## Ambiguity-Set Repair

The unrestricted total-variation set used in the baseline analysis could assign positive probability to states having zero decision-dependent nominal probability. In the Noto instance, this allowed the adversary to fail fully retrofitted corridors. We therefore use the support-preserving set

`P_kappa(y) = {p >= 0: sum_s p_s=1, 0.5 sum_s |p_s-pi_s(y)| <= rho, p_s <= kappa pi_s(y) for every s}`.

The primary specification sets `kappa=2`, meaning that no state can receive more than twice its nominal probability. This value is a declared scenario restriction, not an estimated parameter. Sensitivity analysis also considers `kappa in {1.5,5,10}`. The restriction preserves nominal support because `pi_s(y)=0` implies `p_s=0`.

For fixed `y`, the capped-TV adversarial problem remains linear and is solved exactly through probability-mass transfer within the nominal support. The SBB formulation uses its equivalent mass-transfer dual and McCormick envelopes for the remaining decision-dependent products. The implementation also uses true single-state loss bounds computed with `rho=0`, interval probability tightening, the valid bound `LB>=0`, loss-dominance cuts when verified by the survival matrices, branching on the TV threshold, and child look-ahead bounds.

## Computational Design

The five links generate all 32 no-tail failure states. Road retrofit is evaluated on `y_ij in {0,0.25,0.50,0.75,1}`. At each ambiguity radius, all 3,125 vectors are scanned and the 996 budget-feasible vectors are solved exactly with fixed-`y` Algorithm 1. The sweep uses `rho in {0,0.05,0.10,0.15,0.20,0.25}`, giving 5,976 exact evaluations. This establishes global optimality over the declared discrete grid, not over continuous `[0,1]^5`.

The detached four-worker computation required 6,442.33 seconds, or 107.37 minutes. Each policy was atomically checkpointed, so closing the application did not discard completed evaluations. Across all policies, the largest Algorithm 1 objective/lower-bound gap was `9.99e-7`, no policy required more than nine iterations, and the largest probability-sum error was `4.44e-16`.

## Main Policy Results

At the reference radius `rho=0.10`, the no-investment objective is 9,367.190. Exposure-only investment lowers it to 7,875.130, a reduction of 1,492.060 or 15.93%. Capacity-only investment lowers it to 7,246.554, a reduction of 2,120.636 or 22.64%. Combining exposure and capacity without road retrofit lowers the objective to 5,794.712, a reduction of 3,572.478 or 38.14%.

The all-sector solution lowers the objective further to 4,985.955. This is a total reduction of 4,381.235, or 46.77%, relative to no investment. Relative to the optimized exposure-and-capacity plan, road retrofit contributes an additional reduction of 808.756, or 13.96% of the no-road objective. This result supports the practical importance of transportation access when corridor disruption is calibrated from observed travel-time and restoration data.

The best grid policy is `y=[0.25,1,1,0,0]`. It fully retrofits Nanao-Anamizu and Anamizu-Wajima, applies a 25% retrofit to Kanazawa-Nanao, and does not retrofit the Anamizu-Noto or Anamizu-Suzu corridors. The exposure decision renovates 38.47% of Suzu. Capacity expansion adds 2,202.926 throughput units in Suzu and 330.394 in Noto.

## Ambiguity Effects and Policy Stability

The support-preserving objective increases from 4,978.652 at `rho=0` to 4,986.599 at `rho=0.25`. The ambiguity increase is 7.303 at `rho=0.10` and 7.947 at `rho=0.25`. Road value remains material and rises from 691.445 at `rho=0` to 963.472 at `rho=0.25` because the optimized no-road benchmark deteriorates faster than the selected all-sector policy.

The repair materially changes the interpretation of DDA. Under unrestricted TV, the ambiguity increase was 127.258 at `rho=0.10` and 318.146 at `rho=0.25`. Support preservation reduces these penalties by 94.26% and 97.50%, respectively. It also eliminates all probability mass added to zero-nominal states and prevents adversarial failures of fully retrofitted links. The unrestricted penalty was therefore dominated by support expansion rather than by plausible reweighting within the decision-dependent state law.

At the primary base budget, ambiguity does not change the selected policy. The `y`, `z`, and `w` decisions remain identical across the six radii to numerical tolerance, and `delta_rho=0` in every row. The four-cap pilot reaches the same conclusion. Thus, the corrected interpretation for the primary specification is that DDA changes evaluation but not policy. Changing the solver or tuning the cap would not justify a different claim.

The near-optimality analysis identifies a small policy cluster. At `rho=0.10`, one policy lies within 0.01% of the best and five policies lie within 0.05%, 0.10%, and 0.50%. The second-ranked vector is `[0,1,1,0.5,0]`, with an objective gap of 1.804 or 0.0362%. At `rho>=0.15`, only the selected policy remains within 0.05%.

## Budget-Frontier and Resolution Diagnostics

We also vary the road budget over every distinct threshold represented in the five-level archive and perturb the costs of the two critical corridors on a `5 x 5` design with multipliers in `{0.8,0.9,1,1.1,1.2}`. At the baseline costs, the coarse archive contains six DDA-active budget phases spanning 13.41% of the scanned budget range. Across the 25 cost pairs, 18 contain at least one coarse-grid policy switch. In the strongest baseline-cost phase, with budget multiplier in `[0.469420,0.479479)`, the five-level optimizer changes from `[0,0.5,0.75,0,0]` at `rho=0` to `[0,1,0,0,0]` at positive radii. At `rho=0.25`, this coarse switch has a value of 105.873, or 1.951% of the optimized robust objective.

The associated worst-case distribution follows the intended support-preserving mechanism. Under the coarse nominal policy, the probabilities of neither critical link failing, Link 2 only, Link 3 only, and both links failing change from `(0.663168,0.200895,0.104332,0.031605)` to `(0.413168,0.401789,0.121832,0.063211)` at `rho=0.25`. The largest additions are assigned to high-loss states containing Link 2; joint failure is only one component. No mass enters a zero-nominal state.

However, the apparent switch is not robust to decision resolution. At the midpoint of the strongest coarse interval, a `0.05` grid on the two critical links retains the full five-link state and recourse model while fixing the other three retrofit variables at zero. It evaluates 313 feasible policies at each radius, or 1,878 exact Algorithm 1 solves in total. The best vector is `[0,1,0.2,0,0]` at every radius, and the road-policy `delta_rho=0` throughout. The objective rises from 5,229.491 at `rho=0` to 5,370.685 at `rho=0.25`, while the second-best gap rises from 4.883 to 14.982. Thus, the five-level budget-frontier switch is a discretization artifact, and the cost-budget surface is reported only as a coarse-resolution diagnostic. It does not support a claim that DDA changes the empirical Noto road-retrofit policy under scarcity.

## Failure-Moment Robustness Extension

We next intersect support-preserving TV with decision-dependent bands on link marginals, pairwise joint failures, the expected number of failed links, and its fixed-center second moment. At the same scarcity budget, the exact `0.05` grid again evaluates 1,878 policies. The selected road vector remains `[0,1,0.2,0,0]` for every radius, and the renovation vector is unchanged. Thus, explicit moment control does not activate the long-lived infrastructure decision.

The extension materially changes evaluation. At `rho=0.25`, the objective falls from 5,370.685 under capped TV to 5,305.867 with moment bounds. This removes 64.818 units, or 45.91% of the capped-TV ambiguity penalty relative to `rho=0`. Expected failed links are limited to 1.411587 rather than 1.648671. Nature nevertheless uses the full TV radius, so the ambiguity mechanism remains active inside the narrower moment envelope.

Capacity deployment does respond. At `rho=0`, expansion assigns 108.808 units to Wajima, 2,202.926 to Suzu, and 221.586 to Noto. At every positive radius, the Noto allocation moves to Wajima, giving 330.394 units in Wajima, 2,202.926 in Suzu, and zero in Noto. Relative to holding the complete nominal plan fixed, this adaptation is worth 1.917 units or 0.0361% at `rho=0.25`. The correct interpretation is therefore that moment-aware DDA changes operational capacity deployment but not road retrofit or exposure renovation in the tested scarcity instance.

Endpoint sensitivity uses tight, moderate, and loose failed-link-count mean tolerances of 0.075, 0.150, and 0.300. At `rho=0.25`, the corresponding objectives are 5,261.947, 5,305.867, and 5,367.556, compared with 5,370.685 under capped TV alone. Every profile selects `[0,1,0.2,0,0]`. Thus, the evaluation level and capacity-adaptation value depend on the declared moment envelope, but road-policy stability does not.

## Reduced Continuous Certificates

We use reduced instances to audit the continuous SBB method without extending a weak certificate to the five-link model. In the two-link/four-state instance, `rho=0` closes exactly. At `rho=0.05`, a 5,000-node run gives objective 4,971.667386 and lower bound 4,971.597003, an absolute gap of 0.070383 or 0.001416%. This satisfies the declared 0.1-unit certificate tolerance. The same feasible objective occurs for all larger radii. Since the optimal robust value is nondecreasing in `rho`, the `rho=0.05` lower bound certifies the larger-radius rows to the same tolerance. The continuous policy `[0.924508,1]` remains stable.

The requested three-link/eight-state diagnostic is also retained. It closes exactly for `rho=0` and `rho=0.05`; the latter requires 13 nodes. At `rho>=0.10`, the 100-node run gives objective 4,982.415078 and lower bound 4,981.634760, a gap of 0.780318 or 0.015661%. These rows are reported as node-limited diagnostics, not global certificates.

At the tight-budget midpoint, a separate four-state SBB diagnostic retains only the two critical links and treats excluded links as intact. At `rho=0`, its incumbent is 5,210.458965 with lower bound 5,207.722363, an absolute gap of 2.736602 or 0.052521%. At `rho=0.25`, its incumbent is 5,260.568986 with lower bound 5,258.903427, an absolute gap of 1.665559 or 0.031661%. Both 5,000-node rows select `[1,0.208025]` to numerical precision and have switching value below `2e-11`. Because both gaps exceed the declared 0.1-unit tolerance, they are node- or time-limited diagnostics, not global certificates. The reduced incumbent stability corroborates, but does not certify, the full-state refined-grid result.

## Interpretation and Limitations

The Noto case strengthens the paper's empirical access mechanism. Population, dwelling damage, hospital beds, corridor geometry, and travel disruption are tied to official data. The 13.96% road-sector improvement demonstrates that transportation access can be a practically important mitigation lever.

The repaired ambiguity model also resolves the central structural criticism: the adversary can no longer create failures that the decision-dependent nominal model assigns zero probability. However, the repair does not manufacture a policy switch. The stable policy reflects strong corridor dominance and a small ambiguity penalty within the feasible support.

Several inputs remain constructed or scenario-calibrated, including failure probabilities, operational shares, throughput, costs, budgets, retrofit effectiveness, `rho`, and `kappa`. The five-corridor network is an aggregate representation and does not dynamically reroute traffic on a complete road network. The objective is a comparative modeled-loss measure, not a validated prediction of Noto mortality. Accordingly, we claim discrete-grid optimality for the empirical road policy, a tolerance certificate for the two-link continuous instance, and a transparent node-limited gap for the three-link instance; we do not claim continuous global optimality for the five-link problem or a DDA-induced change in `y`, `z`, or `w`.
