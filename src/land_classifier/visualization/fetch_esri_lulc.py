from __future__ import annotations

import argparse
import json
from pathlib import Path

import planetary_computer
import pystac_client
import rioxarray
import stackstac
from nepal_palika_finder.locator import PalikaLocator
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from land_classifier.visualization.config import (
    DEFAULT_CONFIG_PATH,
    load_config,
    require,
    resolve_path,
)


def get_palika_geometry(palika_name: str) -> tuple[BaseGeometry, str]:
    locator = PalikaLocator()
    matches = locator.search_palikas_by_name(palika_name)
    if not matches:
        raise ValueError(f"No palika found for name: {palika_name}")
    match = matches[0]
    palika_label = match["GaPa_NaPa"] or match["PALIKA"]
    geom_raw = locator.get_palika_geometry_by_name(palika_label, match["DISTRICT"])
    return shape(geom_raw), palika_label.replace(" ", "_")


def pick_asset(item) -> str:
    for key, asset in item.assets.items():
        if asset.roles and "data" in asset.roles:
            return key
    for key in ("data", "classification", "lulc", "map"):
        if key in item.assets:
            return key
    for key, asset in item.assets.items():
        if asset.media_type and "tiff" in asset.media_type:
            return key
    return next(iter(item.assets))


def fetch_latest_item(
    geometry: BaseGeometry, *, stac_url: str, collection: str
):
    catalog = pystac_client.Client.open(
        stac_url, modifier=planetary_computer.sign_inplace
    )
    search = catalog.search(
        collections=[collection],
        intersects=geometry.__geo_interface__,
    )
    items = list(search.item_collection())
    if not items:
        raise RuntimeError(f"No items found for collection: {collection}")

    def get_sort_key(item):
        dt = item.datetime or item.properties.get("datetime")
        if dt:
            return str(dt)
        for key in ("start_datetime", "end_datetime", "io:year"):
            if value := item.properties.get(key):
                return str(value)
        return ""

    items.sort(key=get_sort_key)
    return items[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch ESRI LULC raster for a palika.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to visualization config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    esri_cfg = cfg.get("esri_lulc", {})

    stac_url = str(require(esri_cfg.get("stac_url"), "esri_lulc.stac_url"))
    collection = str(require(esri_cfg.get("collection"), "esri_lulc.collection"))
    palika_name = str(
        require(esri_cfg.get("palika_name"), "esri_lulc.palika_name")
    )
    output_dir = resolve_path(
        require(esri_cfg.get("output_dir"), "esri_lulc.output_dir")
    )
    output_template = str(
        esri_cfg.get("output_filename_template", "{palika_slug}_esri_lulc.tif")
    )
    legend_template = str(
        esri_cfg.get("legend_filename_template", "{palika_slug}_esri_lulc.json")
    )
    resolution = float(esri_cfg.get("resolution", 0.00009))
    classes_raw = esri_cfg.get("classes", {})
    if not classes_raw:
        raise ValueError("esri_lulc.classes must be defined in the config.")
    esri_classes = {int(k): str(v) for k, v in classes_raw.items()}

    geometry, palika_slug = get_palika_geometry(palika_name)
    item = fetch_latest_item(geometry, stac_url=stac_url, collection=collection)
    asset_name = pick_asset(item)

    print(f"Loading {palika_slug} for LULC...")

    cube = stackstac.stack(
        [item],
        assets=[asset_name],
        bounds_latlon=geometry.bounds,
        epsg=4326,
        resolution=resolution,
    )

    da = cube.squeeze("time", drop=True).squeeze("band", drop=True)
    if not da.rio.crs:
        da = da.rio.write_crs("EPSG:4326")

    clipped = da.rio.clip([mapping(geometry)], "EPSG:4326", drop=True)
    clipped.attrs["class_names"] = [esri_classes[key] for key in sorted(esri_classes)]
    clipped.attrs["class_values"] = sorted(esri_classes)
    if clipped.rio.nodata is None:
        clipped.rio.write_nodata(0, inplace=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_template.format(palika_slug=palika_slug)
    clipped.rio.to_raster(out_path)

    legend_path = output_dir / legend_template.format(palika_slug=palika_slug)
    with legend_path.open("w", encoding="utf-8") as handle:
        json.dump(esri_classes, handle, indent=4)

    print(f"Wrote {out_path}")
    print(f"Wrote legend to {legend_path}")


if __name__ == "__main__":
    main()
