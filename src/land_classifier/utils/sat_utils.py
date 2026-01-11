"""Satellite data utilities for extraction and preprocessing."""

from __future__ import annotations

import planetary_computer
import pystac_client
import stackstac
import xarray as xr
from shapely.geometry import shape

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
DATE_RANGE = "2022-01-01/2024-12-31"


def get_satellite_cube(
    geometry: dict | None,
    bbox: tuple[float, float, float, float] | None = None,
    epsg: int = 4326,
) -> xr.DataArray | None:
    """Fetch a lazily stacked STAC cube."""
    catalog = pystac_client.Client.open(
        STAC_URL, modifier=planetary_computer.sign_inplace
    )

    if bbox is None and geometry is not None:
        bbox = shape(geometry).bounds

    search_kwargs = {
        "collections": [COLLECTION],
        "datetime": DATE_RANGE,
        "query": {"eo:cloud_cover": {"lt": 20}},
    }

    if geometry is not None:
        search_kwargs["intersects"] = geometry
    else:
        search_kwargs["bbox"] = bbox

    items = catalog.search(**search_kwargs).item_collection()
    if len(items) == 0:
        return None

    gdal_env = stackstac.DEFAULT_GDAL_ENV.updated(
        always=dict(
            GDAL_HTTP_MAX_RETRY=3,
            GDAL_HTTP_RETRY_DELAY=5,
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
            GDAL_NUM_THREADS="ALL_CPUS",
        )
    )

    return stackstac.stack(
        items,
        assets=["B02", "B03", "B04", "B08", "B11", "B12", "SCL"],
        chunksize=512,
        resolution=10,
        bounds_latlon=bbox,
        epsg=epsg,
        gdal_env=gdal_env,
    )


def mask_clouds(cube: xr.DataArray) -> xr.DataArray:
    """Mask cloud and invalid pixels using Sentinel-2 SCL band."""
    scl = cube.sel(band="SCL")
    mask = (scl == 0) | (scl == 1) | (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10)
    return cube.where(~mask)


def calculate_indices(cube: xr.DataArray) -> xr.DataArray:
    """Compute NDVI, EVI, and BSI indices and return with raw bands."""
    # Extract raw bands and drop the 'band' coordinate to avoid concat conflicts
    # We use errors="ignore" in case some bands already had it dropped
    blue = (
        cube.sel(band="B02").drop_vars("band", errors="ignore").astype("float32")
        / 10000.0
    )
    green = (
        cube.sel(band="B03").drop_vars("band", errors="ignore").astype("float32")
        / 10000.0
    )
    red = (
        cube.sel(band="B04").drop_vars("band", errors="ignore").astype("float32")
        / 10000.0
    )
    nir = (
        cube.sel(band="B08").drop_vars("band", errors="ignore").astype("float32")
        / 10000.0
    )
    swir1 = (
        cube.sel(band="B11").drop_vars("band", errors="ignore").astype("float32")
        / 10000.0
    )
    swir2 = (
        cube.sel(band="B12").drop_vars("band", errors="ignore").astype("float32")
        / 10000.0
    )

    epsilon = 1e-8
    ndvi = (nir - red) / (nir + red + epsilon)
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + epsilon)
    bsi = ((swir2 + red) - (nir + blue)) / ((swir2 + red) + (nir + blue) + epsilon)

    # Ensure indices also don't have the 'band' coordinate
    ndvi = ndvi.drop_vars("band", errors="ignore")
    evi = evi.drop_vars("band", errors="ignore")
    bsi = bsi.drop_vars("band", errors="ignore")

    # Combine raw bands and indices
    features = xr.concat(
        [blue, green, red, nir, swir1, swir2, ndvi, evi, bsi], dim="feature"
    )
    return features.assign_coords(
        feature=["B2", "B3", "B4", "B8", "B11", "B12", "ndvi", "evi", "bsi"]
    )


def preprocess_timeseries(da: xr.DataArray) -> xr.DataArray:
    """Monthly resampling with interpolation and edge filling."""
    monthly = da.resample(time="1MS").median()
    monthly = monthly.chunk({"time": -1})
    filled = monthly.interpolate_na(dim="time", method="linear", use_coordinate=False)
    return filled.bfill(dim="time").ffill(dim="time")


__all__ = [
    "STAC_URL",
    "COLLECTION",
    "DATE_RANGE",
    "get_satellite_cube",
    "mask_clouds",
    "calculate_indices",
    "preprocess_timeseries",
]
