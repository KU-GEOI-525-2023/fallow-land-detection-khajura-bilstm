from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from land_classifier.visualization.config import (
    DEFAULT_CONFIG_PATH,
    load_config,
    require,
    resolve_path,
)


def _resolve_band_indices(
    band_order: list[str | int], band_values: list[object] | None
) -> list[int]:
    if not band_order:
        raise ValueError("band_order must be defined in the config.")

    if all(isinstance(item, int) for item in band_order):
        return [int(item) for item in band_order]

    if not band_values:
        raise ValueError("Band labels not found. Use integer band_order indices.")

    band_lookup = [str(value) for value in band_values]
    try:
        return [band_lookup.index(str(item)) for item in band_order]
    except ValueError as exc:
        raise ValueError(
            f"Band order {band_order} not present in dataset bands {band_lookup}."
        ) from exc


def save_rgb_visual(
    input_path: Path,
    output_path: Path,
    *,
    band_order: list[str | int],
    stretch_min: float,
    stretch_max: float,
) -> None:
    print(f"Loading composite from {input_path}...")
    with xr.open_dataset(input_path, engine="rasterio") as ds:
        ds = ds.squeeze()
        if "band_data" not in ds:
            raise KeyError("Expected 'band_data' variable in raster dataset.")

        band_values = ds.coords.get("band")
        band_values_list = (
            band_values.values.tolist() if band_values is not None else None
        )
        indices = _resolve_band_indices(band_order, band_values_list)
        data = ds.band_data.values
        rgb_stack = data[indices, :, :]

        rgb_stack = np.nan_to_num(rgb_stack, nan=0.0)
        rgb_scaled = np.clip(
            (rgb_stack - stretch_min) / (stretch_max - stretch_min),
            0,
            1,
        )
        rgb_uint8 = (rgb_scaled * 255).astype(np.uint8)

        rgb_xr = xr.DataArray(
            rgb_uint8,
            coords={"band": [1, 2, 3], "y": ds.y, "x": ds.x},
            dims=("band", "y", "x"),
        )
        if ds.rio.crs:
            rgb_xr = rgb_xr.rio.write_crs(ds.rio.crs)

    print(f"Saving RGB visual to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_xr.rio.to_raster(output_path, dtype="uint8")
    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a true-color RGB visualization from a composite raster."
    )
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
    rgb_cfg = cfg.get("rgb_visual", {})

    composite_path = resolve_path(
        require(rgb_cfg.get("composite_path"), "rgb_visual.composite_path")
    )
    output_path = resolve_path(
        require(rgb_cfg.get("output_path"), "rgb_visual.output_path")
    )
    band_order = list(rgb_cfg.get("band_order", ["B4", "B3", "B2"]))
    stretch_min = float(rgb_cfg.get("stretch_min", 0.0))
    stretch_max = float(rgb_cfg.get("stretch_max", 0.3))

    try:
        save_rgb_visual(
            composite_path,
            output_path,
            band_order=band_order,
            stretch_min=stretch_min,
            stretch_max=stretch_max,
        )
    except FileNotFoundError:
        print(f"Error: {composite_path} not found. Run inference first.")


if __name__ == "__main__":
    main()
