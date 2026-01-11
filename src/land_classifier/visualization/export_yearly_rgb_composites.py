from __future__ import annotations

import argparse
from pathlib import Path

import ee
import geemap
from nepal_palika_finder.locator import PalikaLocator

from land_classifier.visualization.config import (
    DEFAULT_CONFIG_PATH,
    load_config,
    require,
    resolve_path,
)


def initialize_ee(project_id: str) -> None:
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


def mask_s2(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    return image.updateMask(mask).divide(10000).select("B.*")


def export_yearly_rgb(config_path: Path) -> None:
    cfg = load_config(config_path)
    gee_cfg = cfg.get("gee", {})
    composites_cfg = cfg.get("yearly_composites", {})

    project_id = str(require(gee_cfg.get("project_id"), "gee.project_id"))
    palika_name = str(require(gee_cfg.get("palika_name"), "gee.palika_name"))
    years = [int(year) for year in (composites_cfg.get("years") or [])]
    if not years:
        print("No years configured for yearly composites.")
        return

    output_dir = resolve_path(
        require(composites_cfg.get("output_dir"), "yearly_composites.output_dir")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cloud_pct = int(composites_cfg.get("cloud_pct", 30))
    scale = int(composites_cfg.get("scale", 20))
    rgb_bands = list(composites_cfg.get("rgb_bands", ["B4", "B3", "B2"]))
    filename_template = str(
        composites_cfg.get("filename_template", "{palika_lower}_rgb_{year}.tif")
    )

    initialize_ee(project_id)

    locator = PalikaLocator()
    try:
        match = locator.search_palikas_by_name(palika_name)[0]
    except IndexError:
        print(f"No palika match found for {palika_name}.")
        return
    geom_raw = locator.get_palika_geometry_by_name(
        match["GaPa_NaPa"] or match["PALIKA"], match["DISTRICT"]
    )
    aoi = ee.Geometry(
        {"type": geom_raw["type"], "coordinates": geom_raw["coordinates"]}
    )

    for year in years:
        print(f"Processing RGB Composite for {year}...")
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
            .map(mask_s2)
        )

        median = collection.median().select(rgb_bands)

        visual = median.unitScale(0, 0.3).multiply(255).uint8()

        visual = visual.clip(aoi)

        output_file = output_dir / filename_template.format(
            palika_lower=palika_name.lower(), year=year
        )
        print(f"Exporting to {output_file}...")

        geemap.ee_export_image(
            visual,
            filename=str(output_file),
            scale=scale,
            region=aoi,
            file_per_band=False,
        )
        print(f"Successfully saved {year} composite.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export yearly RGB composites from Sentinel-2."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to visualization config YAML.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_yearly_rgb(args.config)
