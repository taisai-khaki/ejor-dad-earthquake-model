# Reviewer response: flexible versus fully unpaired truck-drone operations

## Conceptual clarification
We now distinguish three operating policies. In a fixed paired policy, each drone remains assigned to a specific truck. In a flexible same-truck policy, sorties can be chosen flexibly, but a drone cannot be recovered by a different truck; recovery is limited to the launch truck route or depot. In the fully unpaired policy, launch and recovery can be associated with different truck routes whenever the timing, endurance, capacity, and synchronization constraints remain feasible.

## Numerical evidence from the full design
Using the existing full 720-instance design, fully unpaired NILS and the flexible same-truck comparator are both feasible on all 720 matched instances. Fully unpaired NILS wins/ties/losses against flexible same-truck operations are 654/1/65. The mean objective decreases from 95268.939 to 48038.472, and the median objective decreases from 6027.091 to 2836.167. The mean and median improvements are 33.313% and 41.529%, respectively. A two-sided paired Wilcoxon signed-rank test gives p=5.015e-86.

The same conclusion holds relative to the fixed paired baseline: fully unpaired NILS wins/ties/losses are 638/0/82, with mean and median improvements of 27.191% and 41.669%.

## Recommended manuscript changes
1. Add Table `tab:unpairing_definitions` near Table 1 to define fixed paired, flexible same-truck, and fully unpaired operations.
2. Add Table `tab:value_full_unpairing_overall` to the numerical section to report the matched full-design comparison.
3. Add Table `tab:value_full_unpairing_by_size` to show that the benefit persists by instance size.
4. Add Table `tab:full_unpairing_mechanism` if space allows, because it explains why fully unpaired operations improve the objective rather than only reporting statistical dominance.
