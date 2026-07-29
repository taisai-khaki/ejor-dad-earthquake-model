# Draft Empirical Results Section: 2023 Türkiye Earthquake Case

## Empirical Setting

The original validation plan considered the 2015 Nepal/Gorkha earthquake survey; however, the required survey data were not accessible. We therefore construct a reproducible empirical instance using public data from the 2023 Türkiye earthquake. This case is suitable for validating the computational workflow because it combines spatially explicit post-event building damage, population surfaces, emergency-facility locations, and OpenStreetMap road data. These layers provide the information needed to calibrate exposure, damage, facility access, and road-disruption scenarios in the DR-DAD model.

The empirical instance uses a public data stack composed of Zenodo building-damage datasets, HDX/HOTOSM destroyed-building and health-facility layers, OpenStreetMap roads, and WorldPop population rasters. The Zenodo data provide building-level damage labels and hazard-related raster covariates. The HOTOSM destroyed-building layer provides an additional GIS-ready post-event damage layer. WorldPop is used to estimate exposed population at the zone level, and HOTOSM health facilities define candidate emergency centers. OpenStreetMap roads define the transportation network and baseline travel times.

## Instance Construction

The study area is partitioned into regular grid zones. Each grid cell is treated as one demand zone in the model. This choice makes the empirical instance reproducible and avoids dependence on unavailable household-survey or administrative-boundary files. For each zone `rl`, population `P_rl` is estimated by aggregating WorldPop raster cells to the grid. The collapse or severe-damage fraction `q_rl` is derived from Zenodo building damage labels, where severe damage is defined using the upper damage-grade categories. The baseline at-risk population is then computed as `D(0)=sum_rl P_rl q_rl`.

Emergency centers are selected from HOTOSM health-facility records near the damaged area. Because observed emergency treatment capacity is not available in the open data, existing capacity `w_k^0` and expansion unit cost `lambda_k` are assigned by facility type. Hospitals receive higher baseline capacity than clinics and health posts. The road network is derived from OpenStreetMap roads. Candidate road links are selected from frequently used shortest paths between emergency centers and demand zones.

Link failure probabilities `Phi_ij` are constructed as scenario-calibrated hazard-exposure scores. They are not observed road-failure probabilities. The score combines OpenStreetMap road class, bridge or critical-road indicators, proximity to destroyed-building clusters, and Zenodo hazard rasters including PGV, fault distance, and epicenter distance. This produces a defensible road-disruption layer for computational experiments while preserving the distinction between observed data and scenario calibration.

## Computational Design

The empirical instance contains 12 demand zones, 6 emergency centers, 5 candidate road links, and 17 road-failure states. The modeled zone population is 323,476.54, and the baseline at-risk population is 223,255.15. The mean collapse fraction across modeled zones is 0.666. The main budget setting uses renovation budget `B_Z=4568.2`, road-retrofit budget `B_Y=0.8097`, capacity budget `B_X=408.0`, and ambiguity radius `rho=0.10`.

The optimization is solved as a scenario-calibrated DR-DAD instance. For a fixed road-retrofit vector, the renovation/capacity/recourse problem is evaluated exactly over the constructed road-failure states. The reported all-sector road solution is a feasible incumbent found by heuristic candidate search, not a certified global optimum. This distinction is important for the paper: the empirical results demonstrate the value of integrated mitigation under the constructed scenario set, while the global continuous road-retrofit certificate remains outside the current computational claim.

## Main Results

Table 3 reports the main policy comparison. Under no investment, the worst-case expected deaths are 222,656.11. Renovation and capacity investments reduce the value to 107,105.15, a reduction of 115,550.96. Adding road retrofits further reduces the value to 106,582.31, a total reduction of 116,073.79 relative to no investment.

The results show that building renovation is the dominant mitigation lever in this empirical instance. This is expected because the modeled zones have high collapse fractions and large exposed populations. Capacity expansion alone produces only a small reduction, indicating that dispatch capacity is not the primary bottleneck when building exposure remains high. Road retrofit contributes a smaller but meaningful marginal improvement once the road-failure layer is enriched with hazard and damaged-corridor information.

## Sector Contribution and Robustness

The sector-ablation results reinforce this interpretation. Building-only investment reduces worst-case expected deaths by 115,386.15, while capacity-only investment reduces them by 164.80. Combining building and capacity investments yields a reduction of 115,550.96. The all-sector incumbent improves the reduction to 116,073.79, indicating that the road layer adds value after the primary exposure reduction is achieved.

The ambiguity sensitivity shows that the worst-case objective increases monotonically as `rho` grows from 0.00 to 0.25, while the selected road-retrofit pattern remains stable in the tested range. This supports the robustness of the policy structure under moderate ambiguity changes. The budget-design experiments show that building-heavy and balanced-high designs produce the largest reductions, again confirming that the exposure layer is the main driver of loss reduction in this instance.

Candidate-link sensitivity is particularly important for the transportation component. When the candidate road set increases from 3 to 5 to 8 links, the road-retrofit gain increases from approximately 206.32 to 522.84 to 678.86. This pattern supports the paper's mechanism: road retrofits matter more when the model has a richer representation of critical disrupted links.

## Interpretation

The Turkey case should be presented as a reproducible empirical implementation and scenario-calibrated validation exercise. The open data directly support the exposure and building-damage layer through `P_rl` and `q_rl`. The remaining variables, including link failure probabilities, facility capacities, mitigation costs, and budgets, are constructed from open geospatial proxies and sensitivity-tested calibration choices. This means the empirical results validate the model workflow and demonstrate policy behavior, but they do not validate every latent operational parameter as observed truth.

The results are favorable to the paper's argument. Integrated mitigation substantially reduces worst-case expected losses, building renovation is essential where collapse exposure is high, and road retrofits provide an additional measurable benefit when disrupted-road candidates are hazard-informed. The paper should emphasize this layered interpretation rather than claiming full empirical observability of all model parameters.

## Suggested Captions

- Table 1: Public data sources and their mapping to model inputs.
- Table 2: Summary of the empirical Turkey DR-DAD instance.
- Table 3: Main worst-case expected-loss comparison across investment plans.
- Table 4: Sector-ablation analysis for building, capacity, and road investments.
- Table 5: Sensitivity to ambiguity radius.
- Table 8: Candidate road-link risk and retrofit decisions.
- Table 12: Budget-design sensitivity.
- Table 13: Sensitivity to candidate road-network richness.
- Figure 1: Worst-case expected deaths by investment plan.
- Figure 5: Road-retrofit gain under increasing candidate-link sets.
- Figure 8: Nominal-to-worst-case probability shift across disruption states.
