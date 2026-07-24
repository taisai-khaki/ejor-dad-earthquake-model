# Assumptions

## Modeling assumptions
- One customer is served exactly once by either truck or drone.
- Depot is indexed as node 0 and is the only explicit start/end location.
- Vehicles may stay unused; arc binaries for that vehicle remain zero.
- Priority constraints are implemented as a precedence-style soft control, not as strict lexicographic constraints.

## Feasibility assumptions
- Truck load is reduced by served demand along served arcs.
- Drone battery decay is evaluated with conservative or loaded-aware rate depending on arc context.
- Waiting is modeled via auxiliary nonnegative variables and does not currently carry explicit cost unless enabled in objective.
- Synchronization is represented by two binary flags at customer nodes (`z1`, `z2`) for reload/parcels.

## Computational assumptions
- Small instances use exact MILP if `milp.enabled` is true.
- Medium and large instances use heuristic NILS by default.
- All generators and solvers are seeded by configuration for reproducibility.

## Practical assumptions used for implementation
- Unpaired coordination is represented by explicit truck-drone sync decisions.
- Subtour elimination uses MTZ constraints for tractability.
- If a field is absent in YAML input, typed default values are used (see `src/parameters.py`).
- If ambiguity is found in manuscript notation, the study logs the unresolved item and continues with a documented default.

## Heuristic-study assumptions
- `eligible_share` is implemented via a sampled per-instance customer eligibility mask stored in instance metadata.
- Default drone energy burn rates are `15 Wh/min` (empty) and `20 Wh/min` (loaded) unless overridden in experiment config.
- Endurance levels are mapped as battery multipliers: `low=0.70`, `medium=1.00`, `high=1.30` (relative to base battery).
- Handling-time levels are mapped as both swap and reload times: `short=30s`, `medium=60s`, `long=120s`.
- Spatial pattern `uniform` maps to `dispersed`, and `clustered` maps to `dense_urban` in the generator.
- Depot remains at `(0,0)`; when customer coordinates are in positive ranges this behaves like a peripheral depot.
