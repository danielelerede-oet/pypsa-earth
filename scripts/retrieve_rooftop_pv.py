# SPDX-FileCopyrightText: PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Retrieve the global PV 2022 dataset for the model region.

This script downloads the 2022 global PV dataset from Zenodo together with the
associated sub-zoning grid. The model onshore regions are intersected with the
sub-zoning grid to identify the required 4x4 degree tiles. Only the selected
GeoTIFF files are extracted from the yearly archive and stored locally for
subsequent rooftop PV classification.
"""

import hashlib
import shutil
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import requests
from _helpers import configure_logging, create_logger, progress_retrieve

logger = create_logger(__name__)

ZENODO_RECORD_ID = "10684793"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
YEAR = 2022

PV_ZIP_NAME = f"{YEAR}.zip"
ZONING_PREFIX = "sub-zoning_ID."


def get_zenodo_files() -> dict:
    """
    Retrieve metadata for files available in the Zenodo record.

    Returns
    -------
    dict
        Dictionary keyed by filename. Values are the corresponding file
        metadata entries returned by the Zenodo API.
    """
    response = requests.get(ZENODO_API_URL, timeout=120)
    response.raise_for_status()

    record = response.json()
    return {item["key"]: item for item in record["files"]}


def verify_checksum(path: Path, checksum: str | None) -> None:
    """
    Verify the checksum of a downloaded file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the downloaded file.
    checksum : str or None
        Checksum string returned by the Zenodo API. Only MD5 checksums are
        currently supported.

    Returns
    -------
    None
    """
    if not checksum:
        return

    algorithm, expected = checksum.split(":", 1)

    if algorithm.lower() != "md5":
        logger.warning("Unsupported checksum algorithm %s for %s.", algorithm, path)
        return

    md5 = hashlib.md5()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            md5.update(chunk)

    actual = md5.hexdigest()

    if actual != expected:
        path.unlink(missing_ok=True)
        raise ValueError(
            f"Checksum mismatch for {path}: expected {expected}, got {actual}."
        )


def download_zenodo_file(files: dict, filename: str, output_path: Path) -> None:
    """
    Download a Zenodo file if it is not already available locally.

    Parameters
    ----------
    files : dict
        Dictionary returned by :func:`get_zenodo_files`.
    filename : str
        Name of the file to download.
    output_path : pathlib.Path
        Local destination path.

    Returns
    -------
    None
    """
    if filename not in files:
        raise FileNotFoundError(f"File {filename} not found in Zenodo record.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_info = files[filename]
    checksum = file_info.get("checksum")

    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info("Using cached file: %s", output_path)
        verify_checksum(output_path, checksum)
        return

    url = file_info["links"]["self"]
    logger.info("Downloading %s to %s.", filename, output_path)
    progress_retrieve(url, output_path)
    verify_checksum(output_path, checksum)


def download_dataset_files(data_dir: Path) -> Path:
    """
    Download the yearly PV archive and sub-zoning shapefile components.

    Files already available locally are reused after checksum verification.

    Parameters
    ----------
    data_dir : pathlib.Path
        Directory where the downloaded files are stored.

    Returns
    -------
    pathlib.Path
        Path to the downloaded yearly PV zip archive.
    """
    files = get_zenodo_files()

    zip_path = data_dir / PV_ZIP_NAME
    download_zenodo_file(files, PV_ZIP_NAME, zip_path)

    zoning_files = sorted(
        filename for filename in files if filename.startswith(ZONING_PREFIX)
    )

    if not zoning_files:
        raise FileNotFoundError(
            f"No files starting with {ZONING_PREFIX} found in Zenodo record."
        )

    for filename in zoning_files:
        download_zenodo_file(files, filename, data_dir / filename)

    return zip_path


def select_tiles(zoning_path: Path, regions_path: Path) -> gpd.GeoDataFrame:
    """
    Identify PV tiles intersecting the model onshore regions.

    Parameters
    ----------
    zoning_path : pathlib.Path
        Path to the sub-zoning shapefile.
    regions_path : pathlib.Path
        Path to the model onshore regions.

    Returns
    -------
    geopandas.GeoDataFrame
        Selected sub-zoning tiles with tile IDs and raster filenames.
    """
    zoning = gpd.read_file(zoning_path).to_crs("EPSG:4326")
    regions = gpd.read_file(regions_path).to_crs("EPSG:4326")

    expected_countries = set(snakemake.config.get("countries", []))

    if expected_countries and "name" in regions.columns:
        shape_countries = set(regions["name"].astype(str))

        if not expected_countries.issubset(shape_countries):
            raise ValueError(
                "Country shapes do not match the configured countries. "
                f"Configured countries: {sorted(expected_countries)}. "
                f"Countries in {regions_path}: {sorted(shape_countries)}. "
                "Remove stale files in resources/shapes and rerun the shape-building rule."
            )

    if zoning.empty:
        raise ValueError(f"Sub-zoning grid is empty: {zoning_path}")

    if regions.empty:
        raise ValueError(f"Onshore regions are empty: {regions_path}")

    region_union = regions.union_all()
    selected = zoning[zoning.intersects(region_union)].copy()

    if selected.empty:
        raise ValueError(f"No PV tiles intersect regions in {regions_path}.")

    selected["tile_id"] = selected["Id"].astype(int)
    selected["tile_file"] = selected["tile_id"].map(lambda i: f"ty{i}_{YEAR}.tif")

    logger.info("Selected %d rooftop PV tiles.", len(selected))
    return selected


def extract_selected_tiles(
    zip_path: Path,
    selected_tiles: gpd.GeoDataFrame,
    tile_dir: Path,
) -> gpd.GeoDataFrame:
    """
    Extract only required GeoTIFF tiles from the yearly archive.

    Returns
    -------
    geopandas.GeoDataFrame
        Selected tiles filtered to those available in the archive.
    """
    tile_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(selected_tiles["tile_file"])

    with ZipFile(zip_path, "r") as archive:
        archive_by_basename = {Path(name).name: name for name in archive.namelist()}
        available = wanted & set(archive_by_basename)
        unavailable = sorted(wanted - available)

        if unavailable:
            logger.warning(
                "%d selected rooftop PV tiles are not available in the archive "
                "and will be skipped: %s",
                len(unavailable),
                ", ".join(unavailable[:20]) + (" ..." if len(unavailable) > 20 else ""),
            )

        selected_tiles = selected_tiles[
            selected_tiles["tile_file"].isin(available)
        ].copy()

        if selected_tiles.empty:
            raise FileNotFoundError(
                "No selected PV tiles are available in the yearly archive."
            )

        existing = {path.name for path in tile_dir.glob(f"ty*_{YEAR}.tif")}
        missing = available - existing

        if not missing:
            logger.info("All available rooftop PV tiles are already extracted.")
            return selected_tiles

        logger.info("Extracting %d selected rooftop PV tiles.", len(missing))

        for filename in sorted(missing):
            member = archive_by_basename[filename]
            output_path = tile_dir / filename

            with archive.open(member) as src, open(output_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            logger.info("Extracted %s.", output_path)

    return selected_tiles


def save_selected_tiles(selected_tiles: gpd.GeoDataFrame, output_path: Path) -> None:
    """
    Save selected tile metadata.

    Parameters
    ----------
    selected_tiles : geopandas.GeoDataFrame
        Selected sub-zoning tiles.
    output_path : pathlib.Path
        Path to the output GeoJSON file.

    Returns
    -------
    None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_tiles[["Id", "tile_id", "tile_file", "geometry"]].to_file(
        output_path,
        driver="GeoJSON",
    )

    logger.info("Saved selected tile metadata to %s.", output_path)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_rooftop_pv")

    configure_logging(snakemake)

    data_dir = Path(snakemake.output.data_dir)
    tile_dir = data_dir / str(YEAR)
    zoning_path = data_dir / "sub-zoning_ID.shp"

    zip_path = download_dataset_files(data_dir)

    selected_tiles = select_tiles(
        zoning_path=zoning_path,
        regions_path=Path(snakemake.input.country_shapes),
    )

    selected_tiles = extract_selected_tiles(
        zip_path=zip_path,
        selected_tiles=selected_tiles,
        tile_dir=tile_dir,
    )

    save_selected_tiles(
        selected_tiles=selected_tiles,
        output_path=Path(snakemake.output.selected_tiles),
    )

    logger.info(
        "Retrieved %d rooftop PV tiles (%d raster files available).",
        len(selected_tiles),
        len(list(tile_dir.glob(f"ty*_{YEAR}.tif"))),
    )
