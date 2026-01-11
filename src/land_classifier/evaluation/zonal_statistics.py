from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import RasterioIOError

if TYPE_CHECKING:
    import geopandas as gpd

CLASS_MAP = {
    0: "Abandoned",
    1: "Active",
    2: "Fallow",
    3: "Vegetation",
    4: "Other",
}
PIXEL_AREA_HA = 0.01


def calculate_land_distribution(
    tif_path: str | Path,
    output_csv: str | Path | None = None,
) -> pd.DataFrame | None:
    """Calculate class distribution and area from a classification raster.

    Args:
        tif_path: Path to the classification GeoTIFF.
        output_csv: Optional CSV path for saving the summary table.

    Returns:
        DataFrame with class counts, area in hectares, and percentages, or
        None if the raster is missing or has no valid data.
    """
    raster_path = Path(tif_path)
    try:
        with rasterio.open(raster_path) as src:
            raster_band = src.read(1)
            nodata_value = src.nodata
    except RasterioIOError:
        print(f"Error: {raster_path} not found.")
        return None

    unique_values, value_counts = np.unique(raster_band, return_counts=True)
    class_records: list[dict[str, float | int | str]] = []
    for class_id, count in zip(unique_values, value_counts):
        if nodata_value is not None and class_id == nodata_value:
            continue
        if class_id < 0:
            continue

        class_name = CLASS_MAP.get(int(class_id), "Unknown")
        area_ha = round(count * PIXEL_AREA_HA, 2)
        class_records.append(
            {
                "Class_ID": int(class_id),
                "Class_Name": class_name,
                "Pixel_Count": int(count),
                "Area_Ha": area_ha,
            }
        )

    if not class_records:
        print("No valid classification data found in raster.")
        return None

    summary_frame = pd.DataFrame(class_records).sort_values(by="Class_ID")
    total_area = summary_frame["Area_Ha"].sum()
    summary_frame["Percentage"] = round(
        (summary_frame["Area_Ha"] / total_area) * 100, 2
    )

    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_frame.to_csv(output_path, index=False)
        print(f"Global statistics saved to: {output_path}")

    return summary_frame


def run_advanced_zonal_stats(
    tif_path: str | Path,
    vector_path: str | Path,
    output_json: str | Path | None = None,
) -> "gpd.GeoDataFrame | None":
    """Compute zonal statistics for a vector geometry layer.

    Args:
        tif_path: Path to the classification GeoTIFF.
        vector_path: Path to the vector file (e.g., GeoJSON/Shapefile).
        output_json: Optional GeoJSON output path with embedded stats.

    Returns:
        GeoDataFrame with zonal statistics, or None on failure.
    """
    try:
        import geopandas as gpd
        from rasterstats import zonal_stats
    except ImportError:
        print("Advanced stats skipped: 'rasterstats' or 'geopandas' not installed.")
        return None

    raster_path = Path(tif_path)
    vector_file = Path(vector_path)
    print(f"Calculating zonal statistics for vector: {vector_file}...")

    try:
        zonal_results = zonal_stats(
            vector_file, raster_path, categorical=True, stats="count"
        )
        geo_dataframe = gpd.read_file(vector_file)
    except Exception as exc:
        print(f"Error: {exc}")
        return None

    for idx, row_stats in enumerate(zonal_results):
        geo_dataframe.at[idx, "zonal_stats"] = json.dumps(row_stats)

    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        geo_dataframe.to_file(output_path, driver="GeoJSON")
        print(f"Advanced zonal stats saved to: {output_path}")

    return geo_dataframe


if __name__ == "__main__":
    input_raster = Path("models/exports/khajura_classification.tif")
    output_csv = Path("outputs/predictions/land_distribution_stats.csv")

    print("\n=== Calculating Global Land Use Statistics ===")
    results = calculate_land_distribution(input_raster, output_csv)
    if results is not None:
        print(results.to_string(index=False))
