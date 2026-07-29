from __future__ import annotations

from pathlib import Path

from ejor_dad.turkey import (
    HDX_ALL_BUILDINGS_PACKAGE,
    HDX_DESTROYED_BUILDINGS_PACKAGE,
    HDX_HEALTH_FACILITIES_PACKAGE,
    WORLDPOP_TUR_2020_UNADJ_CONSTRAINED_URL,
    aggregate_destroyed_counts,
    aggregate_worldpop_to_zones,
    aggregate_zenodo_damage_by_zone,
    assign_points_to_existing_grid,
    build_regular_grid_from_points,
    download_hdx_health_facilities_geojson,
    download_zenodo_this_study_city,
    download_hdx_destroyed_buildings_csv,
    fetch_hdx_package,
    fetch_zenodo_record,
    hdx_resource_table,
    load_hotosm_destroyed_buildings_csv,
    load_hotosm_health_facilities_geojson,
    load_zenodo_this_study_buildings,
    centers_model_table,
    select_emergency_centers,
    download_url,
    zenodo_file_table,
)


def main() -> None:
    output_dir = Path("data_work/turkey")
    output_dir.mkdir(parents=True, exist_ok=True)

    zenodo = fetch_zenodo_record()
    zenodo_files = zenodo_file_table(zenodo)
    zenodo_files.to_csv(output_dir / "zenodo_18437501_files.csv", index=False)

    destroyed_package = fetch_hdx_package(HDX_DESTROYED_BUILDINGS_PACKAGE)
    hdx_resource_table(destroyed_package).to_csv(output_dir / "hdx_destroyed_buildings_resources.csv", index=False)

    all_buildings_package = fetch_hdx_package(HDX_ALL_BUILDINGS_PACKAGE)
    hdx_resource_table(all_buildings_package).to_csv(output_dir / "hdx_all_buildings_resources.csv", index=False)

    health_package = fetch_hdx_package(HDX_HEALTH_FACILITIES_PACKAGE)
    hdx_resource_table(health_package).to_csv(output_dir / "hdx_health_facilities_resources.csv", index=False)
    health_geojson = download_hdx_health_facilities_geojson(output_dir)
    health_facilities = load_hotosm_health_facilities_geojson(health_geojson)

    (output_dir / "worldpop_tur_2020_url.txt").write_text(WORLDPOP_TUR_2020_UNADJ_CONSTRAINED_URL, encoding="utf-8")
    worldpop_tif = output_dir / "tur_ppp_2020_UNadj_constrained.tif"
    if not worldpop_tif.exists():
        download_url(WORLDPOP_TUR_2020_UNADJ_CONSTRAINED_URL, worldpop_tif)

    destroyed_csv = download_hdx_destroyed_buildings_csv(output_dir)
    destroyed = load_hotosm_destroyed_buildings_csv(destroyed_csv)
    assigned, zones = build_regular_grid_from_points(destroyed, cell_size_km=2.0, padding_km=1.0)
    destroyed_counts = aggregate_destroyed_counts(assigned)

    for city in ("Antakya", "Nurdagi"):
        download_zenodo_this_study_city(city, output_dir / "zenodo_this_study")
    zenodo_buildings = load_zenodo_this_study_buildings(output_dir / "zenodo_this_study", cities=("Antakya", "Nurdagi"))
    zenodo_assigned, zenodo_zones = build_regular_grid_from_points(zenodo_buildings, cell_size_km=2.0, padding_km=1.0)
    zenodo_damage = aggregate_zenodo_damage_by_zone(zenodo_assigned, severe_threshold=3.0, destroyed_threshold=4.0)
    population = aggregate_worldpop_to_zones(worldpop_tif, zenodo_zones)
    zones_model = (
        zenodo_damage.merge(population, on="zone_id", how="left")
        .assign(
            renovation_cost=lambda frame: frame["total_buildings"].astype(float),
            node=lambda frame: frame["zone_id"],
            region="turkey_grid",
        )
        [
            [
                "zone_id",
                "region",
                "population",
                "collapse_fraction",
                "renovation_cost",
                "node",
                "total_buildings",
                "severe_buildings",
                "destroyed_buildings",
                "mean_damage_level",
                "centroid_lon",
                "centroid_lat",
            ]
        ]
    )

    hdx_on_zenodo_grid = assign_points_to_existing_grid(destroyed, zenodo_zones)
    hdx_on_zenodo_counts = aggregate_destroyed_counts(hdx_on_zenodo_grid.dropna(subset=["zone_id"]))
    selected_centers = select_emergency_centers(health_facilities, zenodo_zones, max_centers=12, max_distance_km=25.0)
    centers_model = centers_model_table(selected_centers)

    assigned.to_csv(output_dir / "hotosm_destroyed_buildings_with_grid.csv", index=False)
    zones.to_csv(output_dir / "turkey_damage_grid_zones.csv", index=False)
    destroyed_counts.to_csv(output_dir / "turkey_destroyed_counts_by_grid.csv", index=False)
    zenodo_assigned.to_csv(output_dir / "zenodo_this_study_buildings_with_grid.csv", index=False)
    zenodo_zones.to_csv(output_dir / "turkey_zenodo_grid_zones.csv", index=False)
    zenodo_damage.to_csv(output_dir / "turkey_zenodo_q_by_grid.csv", index=False)
    population.to_csv(output_dir / "turkey_worldpop_population_by_zenodo_grid.csv", index=False)
    zones_model.to_csv(output_dir / "turkey_zones_model_input.csv", index=False)
    hdx_on_zenodo_counts.to_csv(output_dir / "turkey_hdx_destroyed_counts_on_zenodo_grid.csv", index=False)
    health_facilities.to_csv(output_dir / "turkey_health_facilities_raw.csv", index=False)
    centers_model.to_csv(output_dir / "turkey_centers_model_input.csv", index=False)

    print("Zenodo files:")
    print(zenodo_files.to_string(index=False))
    print("\nDestroyed-building rows:", len(destroyed))
    print("Grid zones with destroyed buildings:", len(destroyed_counts))
    print("Zenodo classified buildings:", len(zenodo_buildings))
    print("Zenodo grid zones with q_rl:", len(zenodo_damage))
    print("WorldPop population in model zones:", round(float(zones_model["population"].sum()), 2))
    print("HDX health facilities:", len(health_facilities))
    print("Selected emergency centers:", len(centers_model))
    print("WorldPop source:", WORLDPOP_TUR_2020_UNADJ_CONSTRAINED_URL)
    print("Outputs written to:", output_dir.resolve())


if __name__ == "__main__":
    main()
