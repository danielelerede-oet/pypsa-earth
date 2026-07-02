# SPDX-FileCopyrightText: PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Classify existing PV polygons as rooftop or non-rooftop.

This script reads selected Global PV Mapping 2022 raster tiles, polygonizes the
PV pixels, and classifies each resulting PV polygon using its spatial overlap
with OpenStreetMap building footprints.

The output contains all detected PV polygons and adds a Boolean ``is_rooftop``
column. Downstream steps can then keep only rooftop PV while retaining the full
classified dataset for QA and threshold calibration.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from _helpers import configure_logging, create_logger
from rasterio.features import shapes
from shapely.geometry import shape

logger = create_logger(__name__)

YEAR = 2022
PV_VALUE = 1
ROOFTOP_OVERLAP_THRESHOLD = 0.30
AREA_CRS = "EPSG:6933"


def read_buildings(path: Path) -> gpd.GeoDataFrame:
    """
    Read OSM building footprints and keep polygon geometries only.

    Parameters
    ----------
    path : pathlib.Path
        Path to the OSM buildings GeoJSON file.

    Returns
    -------
    geopandas.GeoDataFrame
        Building footprint polygons projected to the area CRS.
    """
    buildings = gpd.read_file(path)

    if buildings.empty:
        raise ValueError(f"OSM buildings file is empty: {path}")

    buildings = buildings[
        buildings.geometry.notna()
        & buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    if buildings.empty:
        raise ValueError(f"No polygon building footprints found in {path}")

    if buildings.crs is None:
        raise ValueError(f"OSM buildings file has no CRS: {path}")

    buildings = buildings.to_crs(AREA_CRS)
    buildings = buildings[["geometry"]].reset_index(drop=True)

    logger.info("Loaded %d OSM building polygons.", len(buildings))
    return buildings


def polygonize_pv_tile(tile_path: Path, tile_id: int) -> gpd.GeoDataFrame:
    """
    Polygonize PV pixels from one raster tile.

    Parameters
    ----------
    tile_path : pathlib.Path
        Path to the Global PV Mapping GeoTIFF tile.
    tile_id : int
        Tile identifier from the global sub-zoning grid.

    Returns
    -------
    geopandas.GeoDataFrame
        Polygonized PV features in the raster CRS.
    """
    with rasterio.open(tile_path) as src:
        data = src.read(1)
        mask = data == PV_VALUE

        if not mask.any():
            return gpd.GeoDataFrame(
                columns=["tile_id", "tile_file", "geometry"],
                geometry="geometry",
                crs=src.crs,
            )

        geometries = [
            shape(geom)
            for geom, value in shapes(data, mask=mask, transform=src.transform)
            if int(value) == PV_VALUE
        ]

        if not geometries:
            return gpd.GeoDataFrame(
                columns=["tile_id", "tile_file", "geometry"],
                geometry="geometry",
                crs=src.crs,
            )

        if src.crs is None:
            raise ValueError(f"Raster tile has no CRS: {tile_path}")

        pv = gpd.GeoDataFrame(geometry=geometries, crs=src.crs)
        pv["tile_id"] = tile_id
        pv["tile_file"] = tile_path.name

    return pv


def select_local_buildings(
    buildings: gpd.GeoDataFrame,
    pv: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Select building footprints intersecting the PV tile extent.

    Parameters
    ----------
    buildings : geopandas.GeoDataFrame
        Building footprints in area CRS.
    pv : geopandas.GeoDataFrame
        PV polygons in area CRS.

    Returns
    -------
    geopandas.GeoDataFrame
        Local building footprints near the PV tile.
    """
    bounds = pv.total_bounds
    candidate_idx = list(buildings.sindex.intersection(bounds))

    if not candidate_idx:
        return buildings.iloc[[]].copy()

    candidates = buildings.iloc[candidate_idx].copy()
    tile_area = pv.union_all()

    return candidates[candidates.intersects(tile_area)].copy()


def classify_pv_polygons(
    pv: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Classify PV polygons using overlap with building footprints.

    Parameters
    ----------
    pv : geopandas.GeoDataFrame
        PV polygons in area CRS.
    buildings : geopandas.GeoDataFrame
        Building footprints in area CRS.

    Returns
    -------
    geopandas.GeoDataFrame
        Classified PV polygons.
    """
    pv = pv.reset_index(drop=True).reset_index(names="pv_id")
    pv["pv_area_m2"] = pv.geometry.area

    local_buildings = select_local_buildings(buildings, pv)

    if local_buildings.empty:
        pv["building_overlap_m2"] = 0.0
    else:
        intersections = gpd.overlay(
            pv[["pv_id", "geometry"]],
            local_buildings[["geometry"]],
            how="intersection",
        )

        if intersections.empty:
            pv["building_overlap_m2"] = 0.0
        else:
            intersections["overlap_area_m2"] = intersections.geometry.area
            overlap = intersections.groupby("pv_id")["overlap_area_m2"].sum()
            pv["building_overlap_m2"] = pv["pv_id"].map(overlap).fillna(0.0)

    pv["overlap_share"] = pv["building_overlap_m2"] / pv["pv_area_m2"]
    pv["is_rooftop"] = pv["overlap_share"] >= ROOFTOP_OVERLAP_THRESHOLD

    return pv.drop(columns=["pv_id"])


def classify_tile(
    tile_path: Path,
    tile_id: int,
    buildings: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Polygonize and classify one Global PV Mapping tile.

    Parameters
    ----------
    tile_path : pathlib.Path
        Path to the GeoTIFF tile.
    tile_id : int
        Tile identifier.
    buildings : geopandas.GeoDataFrame
        Building footprints in area CRS.

    Returns
    -------
    geopandas.GeoDataFrame
        Classified PV polygons.
    """
    logger.info("Classifying PV tile %s.", tile_path.name)

    pv = polygonize_pv_tile(tile_path, tile_id)

    if pv.empty:
        logger.info("No PV pixels found in %s.", tile_path.name)
        return pv

    pv = pv.to_crs(AREA_CRS)
    classified = classify_pv_polygons(pv, buildings)

    logger.info(
        "Classified %d PV polygons in %s; %d rooftop.",
        len(classified),
        tile_path.name,
        int(classified["is_rooftop"].sum()),
    )

    return classified


def build_empty_output() -> gpd.GeoDataFrame:
    """
    Build an empty classified PV GeoDataFrame.

    Returns
    -------
    geopandas.GeoDataFrame
        Empty output with the expected schema.
    """
    return gpd.GeoDataFrame(
        {
            "tile_id": pd.Series(dtype="int64"),
            "tile_file": pd.Series(dtype="str"),
            "pv_area_m2": pd.Series(dtype="float64"),
            "building_overlap_m2": pd.Series(dtype="float64"),
            "overlap_share": pd.Series(dtype="float64"),
            "is_rooftop": pd.Series(dtype="bool"),
        },
        geometry=gpd.GeoSeries([], crs=AREA_CRS),
        crs=AREA_CRS,
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("classify_rooftop_pv")

    configure_logging(snakemake)

    selected_tiles = gpd.read_file(snakemake.input.selected_tiles)
    buildings = read_buildings(Path(snakemake.input.buildings))

    tile_dir = Path(snakemake.input.tile_dir)
    output_path = Path(snakemake.output.classified_pv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for _, tile in selected_tiles.iterrows():
        tile_file = tile["tile_file"]
        tile_id = int(tile["tile_id"])
        tile_path = tile_dir / tile_file

        if not tile_path.exists():
            raise FileNotFoundError(f"Selected tile is missing: {tile_path}")

        classified_tile = classify_tile(
            tile_path=tile_path,
            tile_id=tile_id,
            buildings=buildings,
        )

        if not classified_tile.empty:
            results.append(classified_tile)

    if results:
        classified = pd.concat(results, ignore_index=True)
        classified = gpd.GeoDataFrame(classified, geometry="geometry", crs=AREA_CRS)
    else:
        classified = build_empty_output()

    logger.info(
        "Classified %d PV polygons in total; %d rooftop.",
        len(classified),
        int(classified["is_rooftop"].sum()) if len(classified) else 0,
    )

    classified.to_file(output_path, layer="pv", driver="GPKG")
    logger.info("Saved classified PV polygons to %s.", output_path)
