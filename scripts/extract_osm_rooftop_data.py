# SPDX-FileCopyrightText: PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Extract OSM data required for rooftop PV classification.

This script retrieves OpenStreetMap building footprints and solar power assets
for the configured countries. Building footprints are used to identify rooftop
PV candidates, while OSM solar plants and generators are used to avoid
classifying utility-scale solar farms as rooftop PV.
"""

import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
from _helpers import BASE_DIR, configure_logging, create_logger
from download_osm_data import country_list_to_geofk
from earth_osm import eo

logger = create_logger(__name__)


def run_earth_osm(
    countries: list[str],
    primary_name: str,
    feature_list: list[str],
    out_dir: Path,
    data_dir: Path,
    progress_bar: bool,
) -> None:
    """
    Run earth-osm extraction.

    Parameters
    ----------
    countries : list of str
        Geofabrik country identifiers.
    primary_name : str
        Main OSM tag key.
    feature_list : list of str
        Requested feature values.
    out_dir : pathlib.Path
        Output directory used by earth-osm.
    data_dir : pathlib.Path
        Directory containing downloaded OSM PBF files and cache files.
    progress_bar : bool
        Whether to show earth-osm progress bars.

    Returns
    -------
    None
    """
    eo.save_osm_data(
        region_list=countries,
        primary_name=primary_name,
        feature_list=feature_list,
        update=False,
        mp=True,
        data_dir=data_dir,
        out_dir=out_dir,
        out_format=["csv", "geojson"],
        out_aggregate=True,
        progress_bar=progress_bar,
    )


def move_single_output(out_dir: Path, csv_target: Path, geojson_target: Path) -> None:
    """
    Move a single earth-osm output pair to stable workflow filenames.

    Parameters
    ----------
    out_dir : pathlib.Path
        Directory used by earth-osm.
    csv_target : pathlib.Path
        Stable CSV output path.
    geojson_target : pathlib.Path
        Stable GeoJSON output path.

    Returns
    -------
    None
    """
    out_path = out_dir / "out"
    targets = {
        "csv": csv_target,
        "geojson": geojson_target,
    }

    for fmt, target in targets.items():
        files = sorted(out_path.glob(f"*.{fmt}"))

        if len(files) != 1:
            raise RuntimeError(
                f"Expected exactly one {fmt} file in {out_path}, found {len(files)}."
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Move %s to %s.", files[0], target)
        shutil.move(files[0], target)


def read_earth_osm_geojson_outputs(out_dir: Path) -> gpd.GeoDataFrame:
    """
    Read all GeoJSON outputs produced by earth-osm.

    Parameters
    ----------
    out_dir : pathlib.Path
        Directory used by earth-osm.

    Returns
    -------
    geopandas.GeoDataFrame
        Combined GeoDataFrame.
    """
    files = sorted((out_dir / "out").glob("*.geojson"))

    if not files:
        raise RuntimeError(f"No GeoJSON files found in {out_dir / 'out'}.")

    frames = []
    for path in files:
        logger.info("Reading %s.", path)
        frames.append(gpd.read_file(path))

    if len(frames) == 1:
        return frames[0]

    return gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=frames[0].crs,
    )


def filter_solar_power_assets(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Keep polygonal OSM power assets related to solar PV.
    """
    if gdf.empty:
        return gdf

    gdf = gdf[
        gdf.geometry.notna() & gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    if gdf.empty:
        return gdf

    def norm(column: str) -> pd.Series:
        if column not in gdf.columns:
            return pd.Series("", index=gdf.index)
        return gdf[column].fillna("").astype(str).str.lower()

    solar = (
        norm("tags.generator:source").eq("solar")
        | norm("tags.plant:source").eq("solar")
        | norm("tags.generator:method").str.contains("photovoltaic", na=False)
        | norm("tags.plant:method").str.contains("photovoltaic", na=False)
    )

    solar_gdf = gdf[solar].copy()

    logger.info(
        "Filtered %d solar power polygons from %d OSM power polygon features.",
        len(solar_gdf),
        len(gdf),
    )

    return solar_gdf


def save_solar_outputs(
    solar: gpd.GeoDataFrame,
    csv_target: Path,
    geojson_target: Path,
) -> None:
    """
    Save filtered solar power assets to stable workflow files.

    Parameters
    ----------
    solar : geopandas.GeoDataFrame
        Filtered solar power assets.
    csv_target : pathlib.Path
        CSV output path.
    geojson_target : pathlib.Path
        GeoJSON output path.

    Returns
    -------
    None
    """
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    geojson_target.parent.mkdir(parents=True, exist_ok=True)

    solar.to_file(geojson_target, driver="GeoJSON")
    solar.to_csv(csv_target, index=False)

    logger.info("Saved %d solar power polygons to %s.", len(solar), geojson_target)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("extract_osm_rooftop_data")

    configure_logging(snakemake)

    countries = country_list_to_geofk(snakemake.params.countries)
    data_dir = Path(BASE_DIR) / "data" / "osm"
    progress_bar = snakemake.config.get("enable", {}).get("progress_bar", True)

    buildings_out_dir = Path(snakemake.output.buildings_geojson).parent
    solar_out_dir = Path(snakemake.output.solar_geojson).parent

    logger.info("Extracting OSM buildings for countries: %s.", countries)
    run_earth_osm(
        countries=countries,
        primary_name="building",
        feature_list=["ALL"],
        out_dir=buildings_out_dir,
        data_dir=data_dir,
        progress_bar=progress_bar,
    )
    move_single_output(
        out_dir=buildings_out_dir,
        csv_target=Path(snakemake.output.buildings_csv),
        geojson_target=Path(snakemake.output.buildings_geojson),
    )

    logger.info("Extracting OSM solar power assets for countries: %s.", countries)
    run_earth_osm(
        countries=countries,
        primary_name="power",
        feature_list=["plant", "generator"],
        out_dir=solar_out_dir,
        data_dir=data_dir,
        progress_bar=progress_bar,
    )

    raw_solar = read_earth_osm_geojson_outputs(solar_out_dir)
    solar = filter_solar_power_assets(raw_solar)

    save_solar_outputs(
        solar=solar,
        csv_target=Path(snakemake.output.solar_csv),
        geojson_target=Path(snakemake.output.solar_geojson),
    )
