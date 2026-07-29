# Turkey Core Data Stack

Use these sources for the Turkey replacement workflow:

| Layer | Source | Model role |
|---|---|---|
| Building damage | Zenodo `18437501` | `q_rl` severe/collapse fraction and validation labels |
| Destroyed buildings | HDX `hotosm_tur_destroyed_buildings` | GIS-ready destroyed-building numerator |
| Building denominator | HDX `hotosm_tur_buildings` or Zenodo GBA footprints | denominator for `q_rl` |
| Roads | OSM roads, Geofabrik/PBF, Overpass, or OSMnx | road links `L` and travel time `tau` |
| Hazard rasters | Zenodo `PGV`, `DEM`, `Fault`, `Lithology`, `Epicenter` | constructed link failure score `Phi_ij` |
| Population | WorldPop, GHS-POP, or official census | zone population `P_rl` |
| Facilities | OSM hospitals, health posts, police/army, Red Crescent | emergency centers `K` and baseline capacity `w_k^0` |

The first runnable step is `examples/turkey_first_step.py`, which fetches source metadata, downloads the small HDX destroyed-buildings CSV, assigns destroyed points to a regular grid, and writes destroyed counts by grid zone.
