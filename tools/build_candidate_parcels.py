"""Build a unified land/building candidate GeoPackage without altering polygons.

Example (from the project root):
    python tools/build_candidate_parcels.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LAND_PATH = Path(r"C:\Users\User\Downloads\후보지_지적Polygon_면적포함_532건.gpkg")
DEFAULT_BUILDING_PATH = Path(r"C:\Users\User\Downloads\Building_Test_Chungcheong_VWorld_건물Polygon.gpkg")
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "data" / "candidate_parcels.gpkg"
TARGET_CRS = "EPSG:5179"

# Source columns used only to create the unified common fields.  They are not
# copied again under their old, conflicting names (addr/address_ml, etc.).
LAND_ALIAS_COLUMNS = {"addr", "query_longitude", "query_latitude", "parcel_area_m2"}
BUILDING_ALIAS_COLUMNS = {
    "address_ml",
    "candidate_longitude",
    "candidate_latitude",
    "building_area_m2",
}
COMMON_COLUMNS = [
    "candidate_id",
    "candidate_type",
    "source_id_ml",
    "address",
    "longitude",
    "latitude",
    "candidate_area_m2",
    "geometry",
]


def _candidate_ids(gdf: gpd.GeoDataFrame, candidate_type: str) -> pd.Series:
    """Make IDs unique even if land and building source IDs overlap."""
    fallback = gdf.get("pnu", pd.Series(gdf.index.astype(str), index=gdf.index)).astype(str)
    source_id = gdf["source_id_ml"].where(gdf["source_id_ml"].notna(), fallback).astype(str)
    # source_id_ml is not guaranteed unique within a source file, so retain the
    # source row position as a deterministic final component.
    row_position = pd.Series(range(len(gdf)), index=gdf.index).astype(str)
    return candidate_type + ":" + source_id + ":" + row_position


def _prepare_land(land: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    result = land.copy()
    result["candidate_id"] = _candidate_ids(result, "land")
    result["candidate_type"] = "land"
    result["address"] = result["addr"]
    result["longitude"] = result["query_longitude"]
    result["latitude"] = result["query_latitude"]
    result["candidate_area_m2"] = result["parcel_area_m2"]
    return result.drop(columns=LAND_ALIAS_COLUMNS, errors="ignore")


def _prepare_building(building: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # to_crs changes coordinate values only; it preserves every polygon's shape
    # and feature boundaries (no buffer, simplify, dissolve, or other geometry edit).
    result = building.to_crs(TARGET_CRS) if building.crs != TARGET_CRS else building.copy()
    result["candidate_id"] = _candidate_ids(result, "building")
    result["candidate_type"] = "building"
    result["address"] = result["address_ml"]
    result["longitude"] = result["candidate_longitude"]
    result["latitude"] = result["candidate_latitude"]
    result["candidate_area_m2"] = result["building_area_m2"]
    return result.drop(columns=BUILDING_ALIAS_COLUMNS, errors="ignore")


def build_candidate_parcels(
    land_path: Path, building_path: Path, output_path: Path
) -> gpd.GeoDataFrame:
    land = gpd.read_file(land_path)
    building = gpd.read_file(building_path)

    if land.crs is None:
        raise ValueError("Land GPKG has no CRS; EPSG:5179 is required.")
    if land.crs.to_string() != TARGET_CRS:
        land = land.to_crs(TARGET_CRS)
    if building.crs is None:
        raise ValueError("Building GPKG has no CRS; EPSG:4326 is required for conversion.")

    land = _prepare_land(land)
    building = _prepare_building(building)

    # concat creates the union of source-specific columns.  Values that belong
    # only to the other candidate type naturally remain NULL in the GeoPackage.
    merged = gpd.GeoDataFrame(
        pd.concat([land, building], ignore_index=True, sort=False),
        geometry="geometry",
        crs=TARGET_CRS,
    )
    remaining = [column for column in merged.columns if column not in COMMON_COLUMNS]
    merged = merged[COMMON_COLUMNS[:-1] + remaining + ["geometry"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(output_path, layer="candidate_parcels", driver="GPKG")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--land", type=Path, default=DEFAULT_LAND_PATH)
    parser.add_argument("--building", type=Path, default=DEFAULT_BUILDING_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    merged = build_candidate_parcels(args.land, args.building, args.output)
    print(f"Saved {len(merged):,} candidates to {args.output} (CRS: {merged.crs})")
    print(merged["candidate_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
