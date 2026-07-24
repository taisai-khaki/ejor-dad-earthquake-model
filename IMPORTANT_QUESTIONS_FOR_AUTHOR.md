# Important Questions for Author

Please confirm the following to remove implementation ambiguity:

1. Synchronization semantics
- Is `z1`/`z2` a hard equality, precedence inequality, or both with waiting allowances?
- Should the model enforce both a launch and recovery pair at the same customer if both are used in one visit?

2. Priority handling
- Are high/medium/low precedence constraints hard constraints, soft penalties, or only objective weighting?
- Should precedence compare completion time, arrival time, or departure time?

3. Drone energy and load
- Is drone battery consumption arc-dependent on payload, altitude, wind, or fixed by speed/distance only?
- Should loaded energy be separate from unloaded energy in all constraints, or only in objective.

4. Subtour elimination
- Is MTZ the intended exact formulation or are additional vehicle-specific path constraints expected?
- Do drones require subtour control identical to truck subtours?

5. Time-window and big-M forms
- Are time windows active on departure or arrival variables?
- Should late arrivals be truncated with waiting or considered infeasible beyond `ub`?

6. Mission completion
- Is returning to depot required for each truck/drone that is launched?
- Are drones allowed to end route without returning when no further customers are assigned?

7. Endogenous data in the manuscript
- What is the exact unit convention for costs and distances in the paper examples (time in min vs sec)?
- Should the objective include waiting explicitly as a cost component?

8. Drone eligibility modeling in benchmark design
- Is drone eligibility defined by parcel weight/capacity only, or by an exogenous eligibility flag per customer?
- For computational experiments, should eligibility-share levels be enforced by sampled flags (current implementation) or derived from demand distributions only?

9. Endurance and handling level mappings
- Please confirm numerical mappings for endurance classes (`low/medium/high`) used in sensitivity runs.
- Please confirm numerical mappings for handling-time classes (`short/medium/long`) for launch/retrieval and synchronization penalties.

10. Depot-location factor
- The study grid includes depot-position concepts; current geometry fixes depot at `(0,0)` (peripheral for positive-coordinate regions).
- Should we implement explicit central/peripheral depot relocation as a controlled factor?

Please confirm these items before final experimental replication.
