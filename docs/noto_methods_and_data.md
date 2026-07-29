# Noto Empirical Case: Data, Preparation, and Model Adjustments

## Why This Case

The 2024 Noto Peninsula earthquake is used as a second empirical case because it supplies unusually direct public evidence for the paper's transportation-access mechanism. The case combines municipality exposure and dwelling damage, pre-event hospital beds, official intercity road geometry, post-earthquake road-restoration points, and observed disrupted and recovered travel times. This is a stronger basis for `L` and `tau` than the OSM-only proxy layer used in the Turkey case.

The case does not validate every latent model input. It is best described as an observed-data calibration of exposure, damage, road access, and pre-event health capacity, combined with transparent constructed and scenario-calibrated parameters.

## Source Stack

| Layer | Source | Empirical status | Model use |
|---|---|---|---|
| Population and households | [Ishikawa Prefecture Statistics, 2022](https://toukei.pref.ishikawa.lg.jp/library/2022.html) | Observed | `P_rl` and the household denominator for `q_rl` |
| Dwelling damage | [Ishikawa damage report No. 162, October 1, 2024](https://www.pref.ishikawa.lg.jp/saigai/documents/higaihou_162_1001_1400.pdf) | Observed | Fully destroyed dwellings by municipality |
| Roads and travel times | [MLIT Noto road-restoration GIS and travel-time tables](https://www.mlit.go.jp/road/r6noto/index2.html) | Observed | Corridor geometry, disrupted travel time, recovered travel time, and restoration points |
| Hospital beds | [MHLW FY2023 Hospital Function Report](https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/open_data_00016.html) | Observed | Pre-event beds by municipality |
| Hospital operability | [MHLW Noto response report](https://kouseikyoku.mhlw.go.jp/tokaihokuriku/000391260.pdf) | Documentary anchor | Scenario operational shares reflecting water and intake constraints |

The local source manifest records file paths and SHA-256 checksums in `data_work/noto/prepared/noto_source_manifest.csv`.

## Spatial and Network Units

- Demand zones are the five core municipalities: Nanao, Wajima, Suzu, Anamizu, and Noto.
- Kanazawa is retained as an external high-capacity center, giving six modeled centers.
- The access network contains five observed intercity corridors: Kanazawa-Nanao, Nanao-Anamizu, Anamizu-Wajima, Anamizu-Noto, and Anamizu-Suzu.
- All combinations of the five link states are represented, producing 32 no-tail states.
- The full road-policy grid uses `y_ij in {0, 0.25, 0.50, 0.75, 1}`. It scans all `5^5=3,125` vectors per ambiguity radius; 996 satisfy the declared road budget and receive an exact fixed-`y` evaluation.

## Observed Exposure and Damage

For municipality `rl`, population is the observed 2022 population:

`P_rl = municipality population`.

The severe-damage proxy is:

`q_rl = fully destroyed dwellings / pre-event households`.

The multiplication `P_rl q_rl` assumes that destroyed and non-destroyed households have the same average household size. It is therefore an observed damage-rate proxy, not a person-level casualty observation.

The five zones contain 106,022 residents and 12,108.462 baseline at-risk persons under this proxy. The municipality damage fractions are 0.0250 for Nanao, 0.2375 for Wajima, 0.3226 for Suzu, 0.1208 for Anamizu, and 0.0390 for Noto.

## Road Access Construction

Normal and disrupted corridor times are locked before optimization from the MLIT observations:

| Corridor | Recovered time | Disrupted time | Failure penalty |
|---|---:|---:|---:|
| Kanazawa-Nanao | 45 | 50 | 5 |
| Nanao-Anamizu | 30 | 100 | 70 |
| Anamizu-Wajima | 30 | 80 | 50 |
| Anamizu-Noto | 40 | 70 | 30 |
| Anamizu-Suzu | 50 | 90 | 40 |

`Phi_ij` is constructed from observations rather than directly estimated as a repeated-event failure frequency:

`Phi_ij = clip(0.08 + 0.55 delay_ratio_ij + 0.12 recovery_score_ij, 0.08, 0.62)`.

The resulting baseline values are 0.1350, 0.4650, 0.5438, 0.3157, and 0.3759 in corridor order. Retrofit acts through the paper's declared relationship `Phi_ij(y)=Phi_ij(1-y_ij)`.

When a corridor fails, the state-specific route receives its observed disruption penalty. This is an aggregated corridor model; it does not dynamically reroute traffic on a complete street-level network.

## Capacity Construction

The source data report 8,420 pre-event hospital beds across the six municipalities. Existing modeled emergency throughput is:

`w_k^0 = reported beds x operational share x 4 persons per bed`.

Reported beds are observed, while the operational shares and throughput multiplier are declared scenario parameters. The modeled existing capacity totals 8,444.4. Expansion `w_k` is interpreted as emergency throughput, including temporary or surge capacity; it must not be described as observed added hospital beds.

## Costs, Budgets, and Ambiguity

The following variables remain scenario-calibrated:

- Zone renovation costs are normalized from baseline at-risk population, with a minimum unit cost.
- Corridor retrofit costs equal `0.50 + route_length_km/50`.
- Capacity expansion unit costs are normalized to one.
- Sector budgets are fixed before optimization at `B_Z=1.513774`, `B_Y=2.296738`, and `B_X=2,533.32`.
- The ambiguity sweep uses `rho in {0, 0.05, 0.10, 0.15, 0.20, 0.25}`.
- The primary support-preserving specification uses density cap `kappa=2`; pilot sensitivity uses `kappa in {1.5, 2, 5, 10}`.

These normalized costs support policy comparison and sensitivity analysis; they are not engineering bid estimates.

## Support-Preserving Ambiguity Adjustment

The unrestricted total-variation set can assign positive probability to a state for which the decision-dependent nominal probability is zero. In this application, that allowed the adversary to fail a fully retrofitted corridor even though the model assigns that failure zero nominal probability. The repaired ambiguity set is:

`P_kappa(y) = {p >= 0: sum_s p_s = 1, 0.5 sum_s |p_s-pi_s(y)| <= rho, p_s <= kappa pi_s(y) for every s}`.

The density-ratio restriction has three consequences:

- `pi_s(y)=0` implies `p_s=0`, so nominal support is preserved;
- the fixed-`y` adversarial problem remains a finite linear program and is solved exactly by capped probability-mass transfer;
- the ambiguity radius remains active, but the adversary cannot create states ruled out by the retrofit decision.

For the capped SBB formulation, the fixed-state robust term is represented by the equivalent mass-transfer dual:

`sum_s pi_s theta_s + rho gamma + (kappa-1) sum_s pi_s u_s + sum_s pi_s v_s`,

subject to `lambda + gamma + u_s >= theta_s`, `v_s + theta_s >= lambda`, and `gamma,u_s,v_s >= 0`. McCormick envelopes are applied to the three decision-dependent products `pi_s theta_s`, `pi_s u_s`, and `pi_s v_s`. The relaxation is strengthened with true single-state loss bounds computed at `rho=0`, probability-simplex and product interval tightening, the valid objective lower bound `LB >= 0`, survival-verified loss-dominance cuts, branching on the TV threshold, and child look-ahead bounds before queue insertion.

The value `kappa=2` is a declared scenario restriction: no state may receive more than twice its nominal probability. It is not estimated from repeated earthquake observations. Results for several caps are therefore reported as sensitivity analysis rather than used to select a cap that produces a preferred policy.

## Failure-Moment Ambiguity Adjustment

Support preservation prevents impossible states but does not directly control how far link-failure marginals or dependence may move within the nominal support. The optional moment-constrained specification intersects `P_kappa(y)` with linear bands. Let `F_ls` equal one when link `l` fails in state `s`, and let `N_s=sum_l F_ls`. For each link, the marginal restriction is

`|sum_s p_s F_ls - sum_s pi_s(y) F_ls| <= a_M + r_M sum_s pi_s(y) F_ls`.

The same form is applied to each pairwise indicator `F_ls F_rs`. The failed-link count is constrained by

`|sum_s p_s N_s - mu_N(y)| <= epsilon_mu`,

and its fixed-center second moment is kept within a declared band around

`M_2(y)=sum_s pi_s(y) [N_s-mu_N(y)]^2`.

Using the fixed nominal center keeps every added condition linear in `p` for fixed `y`. The mean and second-moment bands jointly bound failed-link-count dispersion without imposing a nonconvex endogenous-variance constraint. The Noto primary diagnostic uses scenario tolerances `(r_M,a_M)=(0.25,0.02)` for marginals, `(0.50,0.01)` for joint failures, `epsilon_mu=0.15`, and second-moment tolerance `0.25 M_2(y)+0.05`. These values are sensitivity parameters, not estimated earthquake moments.

Algorithm 1 remains exact for fixed `y`: its adversarial step becomes a finite LP containing TV, density-cap, marginal, joint, mean, and second-moment inequalities. The empirical grid remains restart-safe and atomically checkpointed. Continuous SBB is not used for this extension because its current relaxation does not contain the additional decision-dependent moment products; the implementation raises an explicit error rather than returning an invalid bound.

## Computational Adjustment

The empirical solver is exhaustive discretized search rather than continuous spatial branch-and-bound. For each budget-feasible road vector, Algorithm 1 solves the fixed-`y` renovation, capacity, and recourse problem to numerical tolerance. Therefore:

- the selected road plan is globally best over the declared five-level grid;
- it is not a certificate for the continuous `y in [0,1]^5` problem;
- continuous SBB is evaluated on nested two-link/four-state and three-link/eight-state mechanism instances, with a global claim made only for rows whose gap is closed directly or by the stated monotonicity argument;
- the five-link empirical solver remains exhaustive five-level enumeration because it provides a transparent finite-grid certificate;
- the five-link empirical SBB gap is no longer used to support an optimality claim.

The `m=2` instance is bounded at the declared `0.1`-unit absolute tolerance. At `rho=0.05`, the direct 5,000-node SBB gap is `0.070383` (`0.001416%`). The same feasible objective occurs for all larger radii; monotonicity of the optimal robust value transfers the anchor lower bound and gives the same tolerance certificate. The `m=3` instance closes exactly for `rho<=0.05`; its 100-node gap for `rho>=0.10` is `0.780318` (`0.015661%`) and is reported as node-limited rather than certified.

The support-preserving five-link sweep evaluates `5,976` exact fixed-`y` policies in `6,442.33` seconds with four workers. The largest Algorithm 1 gap is `9.99e-7`, no run requires more than nine iterations, and the largest probability-sum error is `4.44e-16`.

Every fixed-`y` result is atomically checkpointed. The detached process survives closing Codex, and rerunning the same command skips completed policies. The status command reports process identifiers, progress, throughput, ETA, checkpoints, and log location.

## Claims Supported

- The Noto data provide observed calibration for municipality exposure, severe dwelling damage, pre-event beds, corridor geometry, and post-earthquake travel disruption.
- Integrated exposure, capacity, and road mitigation can be compared on a fully enumerated empirical policy grid.
- A nonzero road-sector gain is practically meaningful when it remains material relative to the optimized no-road benchmark.
- DDA may be reported as changing worst-case evaluation even if it does not change the selected discretized policy.
- The support-preserving repair eliminates probability leakage into zero-nominal states.

## Claims Not Supported

- Do not describe `Phi_ij`, costs, budgets, operational shares, `rho`, or `kappa` as directly observed.
- Do not interpret the objective as a validated prediction of actual Noto deaths.
- Do not claim the data validate the entire model or causal retrofit effectiveness.
- Do not claim continuous global optimality for the five-link empirical problem.
- Do not claim DDA changes policy unless `delta_rho` is positive and the exhaustive sweep selects a different `y`.
- Do not treat a stable `y`, `z`, or `w` as a solver failure once the relevant reduced-instance gap is disclosed and the five-link grid is exhaustively enumerated.
