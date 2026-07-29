from __future__ import annotations

import json
import math
import struct
import urllib.parse
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image


ZENODO_TURKEY_RECORD_ID = "18437501"
HDX_DESTROYED_BUILDINGS_PACKAGE = "hotosm_tur_destroyed_buildings"
HDX_ALL_BUILDINGS_PACKAGE = "hotosm_tur_buildings"
HDX_HEALTH_FACILITIES_PACKAGE = "hotosm_tur_health_facilities"
WORLDPOP_TUR_2020_UNADJ_CONSTRAINED_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/"
    "2020/BSGM/TUR/tur_ppp_2020_UNadj_constrained.tif"
)
ZENODO_INFLUENCING_FACTORS = ("PGV", "Fault", "Epicenter", "Lithology")


@dataclass(frozen=True)
class DownloadedResource:
    name: str
    path: Path
    extracted_files: tuple[Path, ...] = ()


@dataclass(frozen=True)
class RemoteZipEntry:
    name: str
    compressed_size: int
    uncompressed_size: int
    compression_method: int
    local_header_offset: int
    flags: int


def fetch_zenodo_record(record_id: str = ZENODO_TURKEY_RECORD_ID) -> dict[str, Any]:
    """Fetch Zenodo metadata for the Turkey building-damage record."""
    return read_json_url(f"https://zenodo.org/api/records/{record_id}")


def zenodo_file_table(record: Mapping[str, Any]) -> pd.DataFrame:
    files = record.get("files", [])
    rows = []
    for item in files:
        rows.append(
            {
                "key": item.get("key"),
                "size": item.get("size"),
                "checksum": item.get("checksum"),
                "download_url": item.get("links", {}).get("self"),
            }
        )
    return pd.DataFrame(rows)


def zenodo_archive_url(record: Mapping[str, Any], key: str = "2023Turkey_earthquake_data.zip") -> str:
    for item in record.get("files", []):
        if item.get("key") == key:
            return item["links"]["self"]
    raise ValueError(f"Zenodo archive {key} was not found in record.")


def list_remote_zip(url: str) -> list[RemoteZipEntry]:
    """List a remote ZIP archive using HTTP range requests."""
    tail = http_range(url, suffix=1024 * 1024)
    position = tail.rfind(b"PK\x05\x06")
    if position < 0:
        raise ValueError("ZIP end-of-central-directory record not found.")
    fields = struct.unpack("<4s4H2LH", tail[position : position + 22])
    _, _, _, _, total_entries, central_size, central_offset, _ = fields
    central_directory = http_range(url, start=central_offset, end=central_offset + central_size - 1)
    entries: list[RemoteZipEntry] = []
    cursor = 0
    for _ in range(total_entries):
        if central_directory[cursor : cursor + 4] != b"PK\x01\x02":
            raise ValueError(f"Unexpected central-directory signature at byte {cursor}.")
        values = struct.unpack("<4s2H4H3L5H2L", central_directory[cursor : cursor + 46])
        (
            _,
            _,
            _,
            flags,
            method,
            _,
            _,
            _,
            compressed_size,
            uncompressed_size,
            name_len,
            extra_len,
            comment_len,
            _,
            _,
            _,
            local_offset,
        ) = values
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = central_directory[cursor + 46 : cursor + 46 + name_len].decode(encoding, errors="replace")
        entries.append(
            RemoteZipEntry(
                name=name,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                compression_method=method,
                local_header_offset=local_offset,
                flags=flags,
            )
        )
        cursor += 46 + name_len + extra_len + comment_len
    return entries


def download_remote_zip_members(
    url: str,
    output_dir: str | Path,
    patterns: Sequence[str],
    overwrite: bool = False,
) -> list[Path]:
    """Download selected members from a remote ZIP archive without fetching it all."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    entries = list_remote_zip(url)
    matched = [entry for entry in entries if any(fnmatch(entry.name, pattern) for pattern in patterns)]
    downloaded: list[Path] = []
    for entry in matched:
        destination = output_path / entry.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not destination.exists():
            destination.write_bytes(read_remote_zip_entry(url, entry))
        downloaded.append(destination)
    return downloaded


def read_remote_zip_entry(url: str, entry: RemoteZipEntry) -> bytes:
    header = http_range(url, start=entry.local_header_offset, end=entry.local_header_offset + 4096)
    if header[:4] != b"PK\x03\x04":
        raise ValueError(f"Unexpected local-header signature for {entry.name}.")
    local = struct.unpack("<4s5H3L2H", header[:30])
    name_len = local[-2]
    extra_len = local[-1]
    data_start = entry.local_header_offset + 30 + name_len + extra_len
    compressed = http_range(url, start=data_start, end=data_start + entry.compressed_size - 1)
    if entry.compression_method == 0:
        return compressed
    if entry.compression_method == 8:
        return zlib.decompress(compressed, -15)
    raise NotImplementedError(f"ZIP compression method {entry.compression_method} is not supported.")


def download_zenodo_this_study_city(
    city: str,
    output_dir: str | Path,
    record_id: str = ZENODO_TURKEY_RECORD_ID,
    overwrite: bool = False,
) -> list[Path]:
    record = fetch_zenodo_record(record_id)
    archive_url = zenodo_archive_url(record)
    return download_remote_zip_members(
        archive_url,
        output_dir,
        patterns=[f"*/Building_damage_data/This_study/Global_GBA_{city}.*"],
        overwrite=overwrite,
    )


def download_zenodo_influencing_factor(
    factor_name: str,
    output_dir: str | Path,
    record_id: str = ZENODO_TURKEY_RECORD_ID,
    overwrite: bool = False,
) -> Path:
    """Download one Zenodo influencing-factor GeoTIFF without downloading the full archive."""
    factor = factor_name.strip()
    record = fetch_zenodo_record(record_id)
    archive_url = zenodo_archive_url(record)
    files = download_remote_zip_members(
        archive_url,
        output_dir,
        patterns=[f"*/Influencing_factors/{factor}.tif"],
        overwrite=overwrite,
    )
    tif_files = [path for path in files if path.name.lower() == f"{factor.lower()}.tif"]
    if not tif_files:
        raise FileNotFoundError(f"{factor}.tif was not found in Zenodo record {record_id}.")
    return tif_files[0]


def download_zenodo_influencing_factors(
    output_dir: str | Path,
    factor_names: Sequence[str] = ZENODO_INFLUENCING_FACTORS,
    record_id: str = ZENODO_TURKEY_RECORD_ID,
    overwrite: bool = False,
) -> dict[str, Path]:
    paths = {}
    for factor in factor_names:
        paths[factor] = download_zenodo_influencing_factor(factor, output_dir, record_id=record_id, overwrite=overwrite)
    return paths


def load_zenodo_this_study_buildings(
    folder: str | Path,
    cities: Sequence[str] = ("Antakya", "Nurdagi"),
) -> pd.DataFrame:
    """Load Zenodo This_study damage shapefiles as centroid points plus damage labels."""
    folder = Path(folder)
    frames: list[pd.DataFrame] = []
    for city in cities:
        base = next(folder.rglob(f"Global_GBA_{city}.shp"), None)
        if base is None:
            raise FileNotFoundError(f"Global_GBA_{city}.shp not found under {folder}.")
        dbf = base.with_suffix(".dbf")
        records = read_dbf(dbf)
        centroids = read_shp_centroids(base)
        if len(records) != len(centroids):
            raise ValueError(f"{city}: DBF records ({len(records)}) and SHP records ({len(centroids)}) differ.")
        frame = pd.DataFrame(records)
        frame["longitude"] = [lon for lon, _ in centroids]
        frame["latitude"] = [lat for _, lat in centroids]
        frame["city"] = city
        frame["damage_lev"] = pd.to_numeric(frame["damage_lev"], errors="coerce")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def fetch_hdx_package(package_id: str) -> dict[str, Any]:
    """Fetch public CKAN metadata from HDX without browser/JavaScript access."""
    url = "https://data.humdata.org/api/3/action/package_show?" + urllib.parse.urlencode({"id": package_id})
    payload = read_json_url(url)
    if not payload.get("success"):
        raise RuntimeError(f"HDX package lookup failed for {package_id}: {payload}")
    return payload["result"]


def hdx_resource_table(package: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for resource in package.get("resources", []):
        rows.append(
            {
                "name": resource.get("name"),
                "format": resource.get("format"),
                "size": resource.get("size"),
                "url": resource.get("url"),
                "last_modified": resource.get("last_modified"),
            }
        )
    return pd.DataFrame(rows)


def select_hdx_resource(package: Mapping[str, Any], preferred_format: str) -> Mapping[str, Any]:
    preferred = preferred_format.lower()
    resources = package.get("resources", [])
    for resource in resources:
        if str(resource.get("format", "")).lower() == preferred:
            return resource
    raise ValueError(f"No {preferred_format} resource found for {package.get('name')}.")


def download_hdx_resource(
    package_id: str,
    preferred_format: str,
    output_dir: str | Path,
    overwrite: bool = False,
) -> DownloadedResource:
    package = fetch_hdx_package(package_id)
    resource = select_hdx_resource(package, preferred_format)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = resource["name"]
    local_zip = output_path / filename
    if overwrite or not local_zip.exists():
        download_url(resource["url"], local_zip)
    extracted: list[Path] = []
    if zipfile.is_zipfile(local_zip):
        with zipfile.ZipFile(local_zip) as archive:
            archive.extractall(output_path)
            extracted = [output_path / name for name in archive.namelist()]
    return DownloadedResource(name=filename, path=local_zip, extracted_files=tuple(extracted))


def download_hdx_destroyed_buildings_csv(
    output_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    downloaded = download_hdx_resource(HDX_DESTROYED_BUILDINGS_PACKAGE, "CSV", output_dir, overwrite=overwrite)
    csv_files = [path for path in downloaded.extracted_files if path.suffix.lower() == ".csv"]
    if not csv_files:
        raise FileNotFoundError(f"No CSV found after extracting {downloaded.path}.")
    return csv_files[0]


def download_hdx_health_facilities_geojson(
    output_dir: str | Path,
    overwrite: bool = False,
    geometry: str = "points",
) -> Path:
    package = fetch_hdx_package(HDX_HEALTH_FACILITIES_PACKAGE)
    resources = package.get("resources", [])
    selected = None
    for resource in resources:
        if str(resource.get("format", "")).lower() == "geojson" and geometry.lower() in str(resource.get("name", "")).lower():
            selected = resource
            break
    if selected is None:
        raise ValueError(f"No health-facility {geometry} GeoJSON resource found.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    local_zip = output_path / selected["name"]
    if overwrite or not local_zip.exists():
        download_url(selected["url"], local_zip)
    with zipfile.ZipFile(local_zip) as archive:
        archive.extractall(output_path)
        geojson_files = [output_path / name for name in archive.namelist() if name.lower().endswith(".geojson")]
    if not geojson_files:
        raise FileNotFoundError(f"No GeoJSON found after extracting {local_zip}.")
    return geojson_files[0]


def load_hotosm_destroyed_buildings_csv(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"osm_id", "destroyed_building", "damage_date", "longitude", "latitude"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Destroyed-building CSV is missing columns: {sorted(missing)}")
    data = data.copy()
    data["destroyed_indicator"] = data["destroyed_building"].astype(str).str.lower().eq("yes").astype(int)
    data["damage_date"] = pd.to_datetime(data["damage_date"], errors="coerce")
    data["zone_source"] = "hotosm_destroyed_buildings"
    return data


def load_hotosm_health_facilities_geojson(path: str | Path) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for feature in payload.get("features", []):
        properties = dict(feature.get("properties") or {})
        lon, lat = geojson_feature_centroid(feature)
        properties["longitude"] = lon
        properties["latitude"] = lat
        rows.append(properties)
    data = pd.DataFrame(rows)
    if data.empty:
        return data
    data["facility_type"] = data.apply(classify_health_facility, axis=1)
    data["existing_capacity"] = data["facility_type"].map(
        {
            "hospital": 500.0,
            "clinic": 150.0,
            "pharmacy": 30.0,
            "dentist": 20.0,
            "doctors": 50.0,
            "healthcare": 80.0,
            "other": 60.0,
        }
    ).fillna(60.0)
    data["capacity_unit_cost"] = data["facility_type"].map(
        {
            "hospital": 1.0,
            "clinic": 1.2,
            "pharmacy": 1.4,
            "dentist": 1.5,
            "doctors": 1.3,
            "healthcare": 1.2,
            "other": 1.5,
        }
    ).fillna(1.5)
    return data


def select_emergency_centers(
    facilities: pd.DataFrame,
    zones: pd.DataFrame,
    max_centers: int = 12,
    bbox_padding_degrees: float = 0.05,
    max_distance_km: float = 40.0,
    keep_types: Sequence[str] = ("hospital", "clinic", "doctors", "healthcare"),
) -> pd.DataFrame:
    """Select candidate emergency centers near the empirical zones.

    OSM health-facility exports contain many pharmacies. For response-capacity
    modeling, this keeps higher-capacity health facilities first and only falls
    back to other types if the AOI has too few candidates.
    """
    required = {"west", "east", "south", "north"}
    missing = required - set(zones.columns)
    if missing:
        raise ValueError(f"zones missing columns: {sorted(missing)}")
    if facilities.empty:
        return facilities.copy()
    west = float(zones["west"].min()) - bbox_padding_degrees
    east = float(zones["east"].max()) + bbox_padding_degrees
    south = float(zones["south"].min()) - bbox_padding_degrees
    north = float(zones["north"].max()) + bbox_padding_degrees
    candidates = facilities[
        facilities["longitude"].between(west, east) & facilities["latitude"].between(south, north)
    ].copy()
    if candidates.empty:
        return candidates
    zone_points = zones[["centroid_lon", "centroid_lat"]].dropna().to_numpy(dtype=float)
    if len(zone_points):
        candidates["nearest_zone_km"] = [
            min(haversine_km(lon, lat, zone_lon, zone_lat) for zone_lon, zone_lat in zone_points)
            for lon, lat in zip(candidates["longitude"], candidates["latitude"])
        ]
        candidates = candidates[candidates["nearest_zone_km"] <= max_distance_km].copy()
    if candidates.empty:
        return candidates
    candidates["_type_rank"] = candidates["facility_type"].map(
        {"hospital": 0, "clinic": 1, "doctors": 2, "healthcare": 3, "other": 4, "pharmacy": 5, "dentist": 6}
    ).fillna(9)
    preferred = candidates[candidates["facility_type"].isin(keep_types)].copy()
    if len(preferred) < max_centers:
        preferred = pd.concat([preferred, candidates[~candidates.index.isin(preferred.index)]], ignore_index=False)
    selected = (
        preferred.sort_values(["_type_rank", "nearest_zone_km", "existing_capacity"], ascending=[True, True, False])
        .head(max_centers)
        .copy()
        .reset_index(drop=True)
    )
    selected["center_id"] = [f"center_{index:03d}" for index in range(len(selected))]
    selected["node"] = selected["center_id"]
    return selected.drop(columns=["_type_rank"], errors="ignore")


def centers_model_table(centers: pd.DataFrame) -> pd.DataFrame:
    required = {"center_id", "node", "existing_capacity", "capacity_unit_cost", "longitude", "latitude"}
    missing = required - set(centers.columns)
    if missing:
        raise ValueError(f"centers missing columns: {sorted(missing)}")
    columns = ["center_id", "node", "existing_capacity", "capacity_unit_cost", "longitude", "latitude"]
    optional = [column for column in ["name", "facility_type", "amenity", "healthcare"] if column in centers.columns]
    return centers[columns + optional].copy()


def classify_health_facility(row: pd.Series) -> str:
    values = " ".join(str(value).lower() for value in row.dropna().tolist())
    for label in ("hospital", "clinic", "pharmacy", "dentist", "doctors"):
        if label in values:
            return label
    if "health" in values or "medical" in values:
        return "healthcare"
    return "other"


def geojson_feature_centroid(feature: Mapping[str, Any]) -> tuple[float, float]:
    geometry = feature.get("geometry") or {}
    coordinates = geometry.get("coordinates")
    geometry_type = geometry.get("type")
    if geometry_type == "Point":
        return float(coordinates[0]), float(coordinates[1])
    points = flatten_geojson_coordinates(coordinates)
    if not points:
        return np.nan, np.nan
    xs, ys = zip(*points)
    return float(np.mean(xs)), float(np.mean(ys))


def flatten_geojson_coordinates(coordinates: Any) -> list[tuple[float, float]]:
    if coordinates is None:
        return []
    if isinstance(coordinates, (list, tuple)) and len(coordinates) >= 2 and all(
        isinstance(value, (int, float)) for value in coordinates[:2]
    ):
        return [(float(coordinates[0]), float(coordinates[1]))]
    points: list[tuple[float, float]] = []
    if isinstance(coordinates, (list, tuple)):
        for item in coordinates:
            points.extend(flatten_geojson_coordinates(item))
    return points


def build_regular_grid_from_points(
    points: pd.DataFrame,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    cell_size_km: float = 2.0,
    padding_km: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign point data to an approximate lat/lon regular grid.

    This is a dependency-light first pass. For publication-grade zones, replace
    it with administrative polygons or a projected equal-area grid.
    """
    if points.empty:
        raise ValueError("Cannot build a grid from an empty point table.")
    mean_lat = float(points[lat_col].mean())
    lat_step = km_to_lat_degrees(cell_size_km)
    lon_step = km_to_lon_degrees(cell_size_km, mean_lat)
    lat_padding = km_to_lat_degrees(padding_km)
    lon_padding = km_to_lon_degrees(padding_km, mean_lat)
    lon_min = float(points[lon_col].min()) - lon_padding
    lon_max = float(points[lon_col].max()) + lon_padding
    lat_min = float(points[lat_col].min()) - lat_padding
    lat_max = float(points[lat_col].max()) + lat_padding
    assigned = points.copy()
    assigned["_grid_col"] = np.floor((assigned[lon_col] - lon_min) / lon_step).astype(int)
    assigned["_grid_row"] = np.floor((assigned[lat_col] - lat_min) / lat_step).astype(int)
    assigned["zone_id"] = [
        f"grid_r{row:04d}_c{col:04d}"
        for row, col in zip(assigned["_grid_row"].astype(int), assigned["_grid_col"].astype(int))
    ]
    rows = []
    for zone_id, group in assigned.groupby("zone_id"):
        row = int(group["_grid_row"].iloc[0])
        col = int(group["_grid_col"].iloc[0])
        west = lon_min + col * lon_step
        east = west + lon_step
        south = lat_min + row * lat_step
        north = south + lat_step
        rows.append(
            {
                "zone_id": zone_id,
                "west": west,
                "east": east,
                "south": south,
                "north": north,
                "centroid_lon": 0.5 * (west + east),
                "centroid_lat": 0.5 * (south + north),
                "cell_size_km": cell_size_km,
            }
        )
    zones = pd.DataFrame(rows).sort_values("zone_id").reset_index(drop=True)
    return assigned.drop(columns=["_grid_row", "_grid_col"]), zones


def assign_points_to_existing_grid(
    points: pd.DataFrame,
    zones: pd.DataFrame,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
) -> pd.DataFrame:
    """Assign points to a grid created by build_regular_grid_from_points."""
    required = {"zone_id", "west", "east", "south", "north"}
    missing = required - set(zones.columns)
    if missing:
        raise ValueError(f"zones missing columns: {sorted(missing)}")
    lon_step = float((zones["east"] - zones["west"]).median())
    lat_step = float((zones["north"] - zones["south"]).median())
    lon_min = float(zones["west"].min())
    lat_min = float(zones["south"].min())
    valid_zones = set(zones["zone_id"])
    assigned = points.copy()
    cols = np.floor((assigned[lon_col] - lon_min) / lon_step).astype(int)
    rows = np.floor((assigned[lat_col] - lat_min) / lat_step).astype(int)
    assigned["zone_id"] = [f"grid_r{row:04d}_c{col:04d}" for row, col in zip(rows, cols)]
    assigned.loc[~assigned["zone_id"].isin(valid_zones), "zone_id"] = pd.NA
    return assigned


def aggregate_destroyed_counts(destroyed_buildings: pd.DataFrame, zone_col: str = "zone_id") -> pd.DataFrame:
    if zone_col not in destroyed_buildings.columns:
        raise ValueError(f"{zone_col} is required. Run a spatial/grid assignment first.")
    output = (
        destroyed_buildings.groupby(zone_col, dropna=False)
        .agg(
            destroyed_buildings=("destroyed_indicator", "sum"),
            destroyed_records=("osm_id", "count"),
            centroid_lon=("longitude", "mean"),
            centroid_lat=("latitude", "mean"),
        )
        .reset_index()
    )
    return output


def aggregate_zenodo_damage_by_zone(
    buildings: pd.DataFrame,
    zone_col: str = "zone_id",
    severe_threshold: float = 3.0,
    destroyed_threshold: float = 4.0,
) -> pd.DataFrame:
    if zone_col not in buildings.columns:
        raise ValueError(f"{zone_col} is required. Run a spatial/grid assignment first.")
    data = buildings.dropna(subset=[zone_col]).copy()
    data["severe_building"] = data["damage_lev"] >= severe_threshold
    data["destroyed_building"] = data["damage_lev"] >= destroyed_threshold
    output = (
        data.groupby(zone_col)
        .agg(
            total_buildings=("damage_lev", "count"),
            severe_buildings=("severe_building", "sum"),
            destroyed_buildings=("destroyed_building", "sum"),
            mean_damage_level=("damage_lev", "mean"),
            centroid_lon=("longitude", "mean"),
            centroid_lat=("latitude", "mean"),
        )
        .reset_index()
    )
    output["collapse_fraction"] = np.where(
        output["total_buildings"] > 0,
        output["severe_buildings"] / output["total_buildings"],
        0.0,
    )
    output["collapse_fraction"] = output["collapse_fraction"].clip(0.0, 1.0)
    return output


def aggregate_worldpop_to_zones(
    tif_path: str | Path,
    zones: pd.DataFrame,
    zone_col: str = "zone_id",
) -> pd.DataFrame:
    """Aggregate a WorldPop WGS84 population-count GeoTIFF to rectangular grid zones."""
    required = {zone_col, "west", "east", "south", "north"}
    missing = required - set(zones.columns)
    if missing:
        raise ValueError(f"zones missing columns: {sorted(missing)}")
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(tif_path) as image:
        transform = geotiff_transform_from_pil(image)
        nodata = image.tag_v2.get(42113)
        nodata_value = float(nodata) if nodata is not None else None
        rows = []
        for zone in zones.itertuples(index=False):
            west = float(getattr(zone, "west"))
            east = float(getattr(zone, "east"))
            south = float(getattr(zone, "south"))
            north = float(getattr(zone, "north"))
            left, upper = lonlat_to_pixel(transform, west, north)
            right, lower = lonlat_to_pixel(transform, east, south)
            left = max(0, math.floor(left))
            upper = max(0, math.floor(upper))
            right = min(image.width, math.ceil(right))
            lower = min(image.height, math.ceil(lower))
            population = 0.0
            if right > left and lower > upper:
                array = np.asarray(image.crop((left, upper, right, lower)), dtype=float)
                if nodata_value is not None:
                    array = array[array != nodata_value]
                array = array[np.isfinite(array)]
                population = float(array[array > 0].sum()) if array.size else 0.0
            rows.append({zone_col: getattr(zone, zone_col), "population": population})
    return pd.DataFrame(rows)


def sample_geotiff_points(
    tif_path: str | Path,
    points: pd.DataFrame,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    output_col: str | None = None,
) -> pd.Series:
    """Sample a WGS84 single-band GeoTIFF at point coordinates."""
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(tif_path) as image:
        transform = geotiff_transform_from_pil(image)
        nodata = image.tag_v2.get(42113)
        nodata_value = float(nodata) if nodata is not None else None
        values = []
        for row in points.itertuples(index=False):
            lon = float(getattr(row, lon_col))
            lat = float(getattr(row, lat_col))
            col, pixel_row = lonlat_to_pixel(transform, lon, lat)
            col_index = int(round(col))
            row_index = int(round(pixel_row))
            if col_index < 0 or row_index < 0 or col_index >= image.width or row_index >= image.height:
                values.append(np.nan)
                continue
            value = float(image.getpixel((col_index, row_index)))
            if nodata_value is not None and value == nodata_value:
                values.append(np.nan)
            else:
                values.append(value)
    return pd.Series(values, index=points.index, name=output_col or Path(tif_path).stem)


def sample_geotiff_to_dataframe(
    dataframe: pd.DataFrame,
    tif_paths: Mapping[str, str | Path],
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    prefix: str = "hazard",
) -> pd.DataFrame:
    output = dataframe.copy()
    for name, path in tif_paths.items():
        output[f"{prefix}_{name.lower()}"] = sample_geotiff_points(path, output, lon_col=lon_col, lat_col=lat_col)
    return output


def geotiff_transform_from_pil(image: Image.Image) -> dict[str, float]:
    scale = image.tag_v2.get(33550)
    tiepoint = image.tag_v2.get(33922)
    if scale is None or tiepoint is None:
        raise ValueError("GeoTIFF is missing ModelPixelScaleTag or ModelTiepointTag.")
    return {
        "origin_x": float(tiepoint[3]),
        "origin_y": float(tiepoint[4]),
        "pixel_width": float(scale[0]),
        "pixel_height": float(scale[1]),
    }


def lonlat_to_pixel(transform: Mapping[str, float], lon: float, lat: float) -> tuple[float, float]:
    col = (lon - transform["origin_x"]) / transform["pixel_width"]
    row = (transform["origin_y"] - lat) / transform["pixel_height"]
    return col, row


def aggregate_total_building_counts(buildings: pd.DataFrame, zone_col: str = "zone_id") -> pd.DataFrame:
    if zone_col not in buildings.columns:
        raise ValueError(f"{zone_col} is required. Run a spatial/grid assignment first.")
    id_col = "osm_id" if "osm_id" in buildings.columns else buildings.columns[0]
    output = (
        buildings.groupby(zone_col, dropna=False)
        .agg(
            total_buildings=(id_col, "count"),
        )
        .reset_index()
    )
    return output


def estimate_q_from_building_counts(
    destroyed_counts: pd.DataFrame,
    total_building_counts: pd.DataFrame,
    zone_col: str = "zone_id",
) -> pd.DataFrame:
    """Compute q_rl = destroyed/severe buildings divided by all buildings."""
    required_destroyed = {zone_col, "destroyed_buildings"}
    required_total = {zone_col, "total_buildings"}
    missing_destroyed = required_destroyed - set(destroyed_counts.columns)
    missing_total = required_total - set(total_building_counts.columns)
    if missing_destroyed:
        raise ValueError(f"destroyed_counts missing columns: {sorted(missing_destroyed)}")
    if missing_total:
        raise ValueError(f"total_building_counts missing columns: {sorted(missing_total)}")
    output = total_building_counts.merge(destroyed_counts[[zone_col, "destroyed_buildings"]], on=zone_col, how="left")
    output["destroyed_buildings"] = output["destroyed_buildings"].fillna(0.0)
    output["collapse_fraction"] = np.where(
        output["total_buildings"] > 0,
        output["destroyed_buildings"] / output["total_buildings"],
        0.0,
    )
    output["collapse_fraction"] = output["collapse_fraction"].clip(0.0, 1.0)
    return output


def score_turkey_link_failure_probability(
    roads: pd.DataFrame,
    pgv_col: str | None = None,
    near_destroyed_col: str | None = None,
    slope_col: str | None = None,
    bridge_col: str | None = None,
    highway_col: str | None = "highway",
    base: float = 0.02,
    pgv_weight: float = 0.12,
    damage_proximity_weight: float = 0.12,
    slope_weight: float = 0.08,
    bridge_weight: float = 0.06,
    critical_road_weight: float = 0.04,
    cap: float = 0.45,
) -> pd.Series:
    """Construct Phi_ij as a hazard-exposure score for Turkey road links."""
    score = pd.Series(base, index=roads.index, dtype=float)
    if pgv_col:
        score += pgv_weight * normalize_01(roads[pgv_col])
    if near_destroyed_col:
        score += damage_proximity_weight * roads[near_destroyed_col].astype(bool).astype(float)
    if slope_col:
        score += slope_weight * normalize_01(roads[slope_col])
    if bridge_col:
        score += bridge_weight * roads[bridge_col].astype(bool).astype(float)
    if highway_col and highway_col in roads.columns:
        critical = roads[highway_col].astype(str).str.lower().isin(
            {"motorway", "trunk", "primary", "secondary", "tertiary"}
        )
        score += critical_road_weight * critical.astype(float)
    return score.clip(0.0, cap)


def overpass_roads_bbox(
    south: float,
    west: float,
    north: float,
    east: float,
    highway_types: Sequence[str] | None = None,
    timeout_seconds: int = 180,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download OSM road ways inside a bbox through Overpass.

    Returns node and directed edge tables. For large regions, prefer a local
    Geofabrik/PBF extract or OSMnx.
    """
    if highway_types is None:
        highway_types = ("motorway", "trunk", "primary", "secondary", "tertiary", "residential", "service")
    highway_regex = "|".join(map(str, highway_types))
    query = f"""
    [out:json][timeout:{timeout_seconds}];
    (
      way["highway"~"^({highway_regex})$"]({south},{west},{north},{east});
    );
    (._;>;);
    out body;
    """
    url = "https://overpass-api.de/api/interpreter"
    payload = urllib.parse.urlencode({"data": query}).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"User-Agent": "ejor-dad-turkey/0.1"})
    with urllib.request.urlopen(request, timeout=timeout_seconds + 30) as response:
        data = json.loads(response.read().decode("utf-8"))
    return osm_json_to_nodes_edges(data)


def osm_json_to_nodes_edges(data: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_rows = []
    ways = []
    for element in data.get("elements", []):
        if element.get("type") == "node":
            node_rows.append({"node": str(element["id"]), "lon": element["lon"], "lat": element["lat"]})
        elif element.get("type") == "way":
            ways.append(element)
    nodes = pd.DataFrame(node_rows)
    if nodes.empty:
        return nodes, pd.DataFrame()
    node_lookup = nodes.set_index("node")[["lon", "lat"]].to_dict("index")
    edge_rows = []
    for way in ways:
        tags = way.get("tags", {})
        way_nodes = [str(node_id) for node_id in way.get("nodes", [])]
        oneway = str(tags.get("oneway", "")).lower() in {"yes", "true", "1"}
        for tail, head in zip(way_nodes[:-1], way_nodes[1:]):
            if tail not in node_lookup or head not in node_lookup:
                continue
            distance_km = haversine_km(
                node_lookup[tail]["lon"],
                node_lookup[tail]["lat"],
                node_lookup[head]["lon"],
                node_lookup[head]["lat"],
            )
            edge_rows.append(osm_edge_row(way, tail, head, distance_km, tags))
            if not oneway:
                edge_rows.append(osm_edge_row(way, head, tail, distance_km, tags))
    edges = pd.DataFrame(edge_rows)
    return nodes, edges


def osm_edge_row(way: Mapping[str, Any], tail: str, head: str, distance_km: float, tags: Mapping[str, Any]) -> dict[str, Any]:
    highway = str(tags.get("highway", "road"))
    speed_kmh = default_speed_kmh(highway)
    return {
        "link_id": f"osm_way_{way['id']}_{tail}_{head}",
        "tail": tail,
        "head": head,
        "osm_way_id": str(way["id"]),
        "highway": highway,
        "bridge": str(tags.get("bridge", "")).lower() in {"yes", "true", "1"},
        "distance_km": distance_km,
        "travel_time": 60.0 * distance_km / speed_kmh if speed_kmh > 0 else np.inf,
        "retrofit_cost": max(distance_km, 0.01),
    }


def default_speed_kmh(highway: str) -> float:
    return {
        "motorway": 90.0,
        "trunk": 75.0,
        "primary": 60.0,
        "secondary": 50.0,
        "tertiary": 40.0,
        "residential": 25.0,
        "service": 15.0,
    }.get(str(highway).lower(), 30.0)


def normalize_01(values: Sequence[float]) -> pd.Series:
    series = pd.to_numeric(pd.Series(values), errors="coerce").astype(float)
    lower = float(series.min(skipna=True))
    upper = float(series.max(skipna=True))
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        return pd.Series(0.0, index=series.index)
    return ((series - lower) / (upper - lower)).fillna(0.0).clip(0.0, 1.0)


def km_to_lat_degrees(km: float) -> float:
    return float(km) / 110.574 if km else 0.0


def km_to_lon_degrees(km: float, latitude: float) -> float:
    denominator = 111.320 * math.cos(math.radians(latitude))
    if abs(denominator) < 1e-9:
        raise ValueError("Longitude degree conversion is unstable near the poles.")
    return float(km) / denominator if km else 0.0


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    return 2.0 * radius * math.asin(math.sqrt(a))


def read_json_url(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "ejor-dad-turkey/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download_url(url: str, path: str | Path) -> Path:
    path = Path(path)
    request = urllib.request.Request(url, headers={"User-Agent": "ejor-dad-turkey/0.1"})
    with urllib.request.urlopen(request, timeout=180) as response, path.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    return path


def http_range(url: str, start: int | None = None, end: int | None = None, suffix: int | None = None) -> bytes:
    if suffix is not None:
        range_header = f"bytes=-{suffix}"
    elif start is not None and end is not None:
        range_header = f"bytes={start}-{end}"
    elif start is not None:
        range_header = f"bytes={start}-"
    else:
        raise ValueError("Provide either suffix or start/end.")
    request = urllib.request.Request(url, headers={"Range": range_header, "User-Agent": "ejor-dad-turkey/0.1"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def read_dbf(path: str | Path) -> list[dict[str, Any]]:
    data = Path(path).read_bytes()
    record_count = struct.unpack("<I", data[4:8])[0]
    header_length = struct.unpack("<H", data[8:10])[0]
    record_length = struct.unpack("<H", data[10:12])[0]
    fields = []
    cursor = 32
    while cursor + 32 <= len(data) and data[cursor] != 0x0D:
        name = data[cursor : cursor + 11].split(b"\x00", 1)[0].decode("ascii", errors="replace")
        field_type = chr(data[cursor + 11])
        field_length = data[cursor + 16]
        decimal_count = data[cursor + 17]
        fields.append((name, field_type, field_length, decimal_count))
        cursor += 32
    records: list[dict[str, Any]] = []
    cursor = header_length
    for _ in range(record_count):
        record = data[cursor : cursor + record_length]
        cursor += record_length
        if not record or record[:1] == b"*":
            continue
        row: dict[str, Any] = {}
        position = 1
        for name, field_type, field_length, _ in fields:
            raw = record[position : position + field_length].decode("latin1", errors="replace").strip()
            position += field_length
            if field_type in {"N", "F"}:
                try:
                    row[name] = float(raw) if raw else np.nan
                except ValueError:
                    row[name] = raw
            else:
                row[name] = raw
        records.append(row)
    return records


def read_shp_centroids(path: str | Path) -> list[tuple[float, float]]:
    data = Path(path).read_bytes()
    centroids: list[tuple[float, float]] = []
    cursor = 100
    while cursor + 8 <= len(data):
        _, content_words = struct.unpack(">2i", data[cursor : cursor + 8])
        cursor += 8
        content_bytes = content_words * 2
        content = data[cursor : cursor + content_bytes]
        cursor += content_bytes
        if len(content) < 4:
            continue
        shape_type = struct.unpack("<i", content[:4])[0]
        if shape_type == 0:
            centroids.append((np.nan, np.nan))
        elif shape_type == 1:
            x, y = struct.unpack("<2d", content[4:20])
            centroids.append((x, y))
        elif shape_type in {3, 5}:
            if len(content) < 44:
                centroids.append((np.nan, np.nan))
                continue
            num_parts, num_points = struct.unpack("<2i", content[36:44])
            points_offset = 44 + 4 * num_parts
            points = [
                struct.unpack("<2d", content[points_offset + 16 * index : points_offset + 16 * (index + 1)])
                for index in range(num_points)
            ]
            centroids.append(polygon_centroid(points))
        else:
            centroids.append((np.nan, np.nan))
    return centroids


def polygon_centroid(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return np.nan, np.nan
    if len(points) < 3:
        xs, ys = zip(*points)
        return float(np.mean(xs)), float(np.mean(ys))
    area = 0.0
    centroid_x = 0.0
    centroid_y = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
        cross = x0 * y1 - x1 * y0
        area += cross
        centroid_x += (x0 + x1) * cross
        centroid_y += (y0 + y1) * cross
    area *= 0.5
    if abs(area) < 1e-12:
        xs, ys = zip(*points)
        return float(np.mean(xs)), float(np.mean(ys))
    return centroid_x / (6.0 * area), centroid_y / (6.0 * area)
