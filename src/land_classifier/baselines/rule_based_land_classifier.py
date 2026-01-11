from __future__ import annotations

import argparse
import time
from pathlib import Path

import ee
import geemap
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from nepal_palika_finder.locator import PalikaLocator
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT / "configs/baselines/rule_based.yaml"


def _require(value: object, name: str) -> object:
    if value is None:
        raise ValueError(f"Missing required config value: {name}")
    return value


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> DictConfig:
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    return cfg


def initialize_ee(project_id: str | None = None) -> None:
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
    except Exception:
        ee.Authenticate()
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()


def mask_s2(image: ee.Image) -> ee.Image:
    """Scaled reflectance and masking (QA60 + SCL) matching project conventions."""
    scaled = image.divide(10000).select("B.*")
    scl = image.select("SCL")
    keep = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    qa = image.select("QA60")
    qa_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return scaled.updateMask(keep).updateMask(qa_mask).copyProperties(
        image, ["system:time_start"]
    )


def add_ndvi(image: ee.Image) -> ee.Image:
    return image.addBands(image.normalizedDifference(["B8", "B4"]).rename("ndvi"))


def get_palika_geometry(palika_name: str) -> tuple[ee.Geometry, str]:
    locator = PalikaLocator()
    matches = locator.search_palikas_by_name(palika_name)
    if not matches:
        raise ValueError(f"No palika found for name: {palika_name}")
    match = matches[0]
    palika_label = match["GaPa_NaPa"] or match["PALIKA"]
    geom_raw = locator.get_palika_geometry_by_name(palika_label, match["DISTRICT"])
    geometry = {"type": geom_raw["type"], "coordinates": geom_raw["coordinates"]}
    return ee.Geometry(geometry), palika_label.replace(" ", "_")


def build_ndvi_collection(
    aoi: ee.Geometry, start_date: str, end_date: str, cloud_filter: int
) -> ee.ImageCollection:
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_filter))
        .map(mask_s2)
        .map(add_ndvi)
        .select(["ndvi"])
    )


def annual_ndvi_max(aoi: ee.Geometry, year: int, cloud_filter: int) -> ee.Image:
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    col = build_ndvi_collection(aoi, start, end, cloud_filter)
    return col.max().rename(f"ndvi_max_{year}")


def multi_year_ndvi_stats(
    aoi: ee.Geometry,
    start_year: int,
    end_year_exclusive: int,
    cloud_filter: int,
) -> tuple[ee.Image, ee.Image, ee.Image]:
    start = f"{start_year}-01-01"
    end = f"{end_year_exclusive}-01-01"
    col = build_ndvi_collection(aoi, start, end, cloud_filter)
    return (
        col.mean().rename("ndvi_mean"),
        col.min().rename("ndvi_min"),
        col.count().rename("ndvi_count"),
    )


def build_rule_based_classification(
    aoi: ee.Geometry,
    prev_year: int,
    target_year: int,
    next_year: int,
    cloud_filter: int,
    peak_threshold: float,
    abandoned_mean_low: float,
    abandoned_mean_high: float,
    other_mean_low: float,
    nonveg_max_threshold: float,
    forest_min_threshold: float,
    majority_filter: bool,
    output_schema: str,
    nodata_value: int,
) -> ee.Image:
    """
    Decision rules derived from 'Preparing Traning Data.pdf':
    - Active: max NDVI in target year > peak_threshold
    - Fallow: max NDVI in target year <= peak_threshold AND (prev OR next year has peak)
    - Abandoned: no peaks across prev/target/next AND mean NDVI indicates persistent low veg
    - Vegetation/Other: threshold-based buckets for persistent forest and non-vegetated surfaces
    """
    max_prev = annual_ndvi_max(aoi, prev_year, cloud_filter)
    max_t = annual_ndvi_max(aoi, target_year, cloud_filter)
    max_next = annual_ndvi_max(aoi, next_year, cloud_filter)

    ndvi_mean, ndvi_min, ndvi_count = multi_year_ndvi_stats(
        aoi, prev_year, next_year + 1, cloud_filter
    )

    max_3yr = max_prev.max(max_t).max(max_next).rename("ndvi_max_3yr")
    valid = ndvi_count.gt(0)

    active = max_t.gt(peak_threshold)
    fallow = max_t.lte(peak_threshold).And(
        max_prev.gt(peak_threshold).Or(max_next.gt(peak_threshold))
    )

    no_peaks_3yr = (
        max_prev.lte(peak_threshold)
        .And(max_t.lte(peak_threshold))
        .And(max_next.lte(peak_threshold))
    )

    other_low = ndvi_mean.lt(other_mean_low)
    abandoned_ecology = ndvi_mean.gte(abandoned_mean_low).And(
        ndvi_mean.lte(abandoned_mean_high)
    )
    abandoned = no_peaks_3yr.And(abandoned_ecology).And(other_low.Not())

    vegetation = ndvi_min.gt(forest_min_threshold)
    nonveg = max_3yr.lt(nonveg_max_threshold).Or(other_low)

    if output_schema == "project5":
        cls_abandoned, cls_active, cls_fallow, cls_veg, cls_other = 0, 1, 2, 3, 4
    elif output_schema == "ag4":
        # Active=0, Fallow=1, Abandoned=2, Other=3
        cls_active, cls_fallow, cls_abandoned, cls_other = 0, 1, 2, 3
        cls_veg = cls_other
    else:
        raise ValueError("output_schema must be 'project5' or 'ag4'")

    # Default everything to Other, then overwrite by priority.
    cls = ee.Image.constant(cls_other).toInt16()
    cls = cls.where(vegetation, cls_veg)
    cls = cls.where(nonveg, cls_other)
    cls = cls.where(abandoned, cls_abandoned)
    cls = cls.where(fallow, cls_fallow)
    cls = cls.where(active, cls_active)

    # Optional 3x3 mode filter (salt-and-pepper cleanup).
    if majority_filter:
        cls = cls.focal_mode(radius=1, units="pixels")

    # Mask pixels with no valid observations, then fill with nodata.
    return cls.updateMask(valid).unmask(nodata_value).toInt16().rename("class_id")


def _to_2d(tile: np.ndarray) -> np.ndarray:
    if tile.ndim == 3 and tile.shape[-1] == 1:
        return tile[:, :, 0]
    if tile.ndim == 2:
        return tile
    raise ValueError(f"Unexpected tile shape: {tile.shape}")


def download_classification_tiled(
    class_img: ee.Image,
    aoi: ee.Geometry,
    output_raster: Path,
    tile_size_deg: float,
    nodata_value: int,
    cache_dir: Path | None,
    scale: int = 10,
) -> None:
    bbox = aoi.bounds().getInfo()["coordinates"][0]
    lons = [p[0] for p in bbox]
    lats = [p[1] for p in bbox]
    west, east, south, north = min(lons), max(lons), min(lats), max(lats)

    lon_grid = np.arange(west, east, tile_size_deg)
    lat_grid = np.arange(south, north, tile_size_deg)

    stitched_rows: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None

    start_time = time.time()
    total_tiles = len(lon_grid) * len(lat_grid)
    completed = 0

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for lat_idx, lat in enumerate(reversed(lat_grid)):
        row_tiles: list[np.ndarray] = []
        for lon_idx, lon in enumerate(lon_grid):
            tile_aoi = ee.Geometry.Rectangle(
                [lon, lat, lon + tile_size_deg, lat + tile_size_deg], proj="EPSG:4326"
            )

            cache_file = None
            if cache_dir is not None:
                cache_file = cache_dir / f"tile_{lat_idx}_{lon_idx}.npy"
                try:
                    tile = np.load(cache_file)
                except FileNotFoundError:
                    tile = None
                else:
                    row_tiles.append(tile)
                    completed += 1
                    continue

            tile_raw = geemap.ee_to_numpy(class_img, region=tile_aoi, scale=scale)
            if tile_raw.size == 0:
                if expected_shape is None:
                    expected_shape = (100, 100)
                tile = np.full(expected_shape, nodata_value, dtype=np.int16)
            else:
                tile_2d = _to_2d(tile_raw).astype(np.int16, copy=False)
                if expected_shape is None:
                    expected_shape = tile_2d.shape
                tile = tile_2d

            if cache_file is not None:
                np.save(cache_file, tile)

            row_tiles.append(tile)
            completed += 1

            if completed % 10 == 0 or completed == total_tiles:
                pct = (completed / total_tiles) * 100
                elapsed = time.time() - start_time
                print(f"Progress: {completed}/{total_tiles} tiles ({pct:.1f}%), {elapsed:.1f}s")

        stitched_rows.append(np.concatenate(row_tiles, axis=1))

    final_map = np.concatenate(stitched_rows, axis=0)
    h, w = final_map.shape

    result = (
        xr.DataArray(
            final_map.astype(np.int16, copy=False),
            coords={"y": np.linspace(north, south, h), "x": np.linspace(west, east, w)},
            dims=("y", "x"),
        )
        .rio.write_crs("EPSG:4326")
        .rio.write_nodata(nodata_value)
    )

    try:
        result = result.rio.clip([aoi.getInfo()], "EPSG:4326")
    except Exception:
        pass

    output_raster.parent.mkdir(parents=True, exist_ok=True)
    result.rio.to_raster(output_raster)


def build_arg_parser(
    cfg: DictConfig, parents: list[argparse.ArgumentParser] | None = None
) -> argparse.ArgumentParser:
    rule_cfg = cfg.get("rule_based", cfg)
    thresholds_cfg = rule_cfg.get("thresholds", {})

    output_path = _resolve_path(
        _require(rule_cfg.get("output"), "rule_based.output")
    )
    cache_dir = rule_cfg.get("cache_dir")
    cache_dir = _resolve_path(cache_dir) if cache_dir else None

    parser = argparse.ArgumentParser(
        description="Rule-based temporal land status classifier (NDVI decision matrix).",
        parents=parents or [],
    )
    project_id = _require(rule_cfg.get("project_id"), "rule_based.project_id")
    parser.add_argument(
        "--project-id",
        type=str,
        default=project_id,
        help="GEE project id.",
    )
    parser.add_argument(
        "--palika",
        type=str,
        default=rule_cfg.get("palika", "Khajura"),
        help="Palika name.",
    )
    parser.add_argument(
        "--target-year",
        type=int,
        default=int(rule_cfg.get("target_year", 2023)),
        help="Target year (T).",
    )
    parser.add_argument(
        "--cloud-filter",
        type=int,
        default=int(rule_cfg.get("cloud_filter", 60)),
        help="Cloudy pixel percentage cutoff.",
    )
    parser.add_argument(
        "--peak-threshold",
        type=float,
        default=float(thresholds_cfg.get("peak", 0.6)),
        help="NDVI threshold for a crop 'peak' (default 0.6).",
    )
    parser.add_argument(
        "--abandoned-mean-low",
        type=float,
        default=float(thresholds_cfg.get("abandoned_mean_low", 0.3)),
        help="Lower mean NDVI bound for abandoned ecology filter.",
    )
    parser.add_argument(
        "--abandoned-mean-high",
        type=float,
        default=float(thresholds_cfg.get("abandoned_mean_high", 0.5)),
        help="Upper mean NDVI bound for abandoned ecology filter.",
    )
    parser.add_argument(
        "--other-mean-low",
        type=float,
        default=float(thresholds_cfg.get("other_mean_low", 0.2)),
        help="Mean NDVI below this is treated as built/water-like Other.",
    )
    parser.add_argument(
        "--nonveg-max-threshold",
        type=float,
        default=float(thresholds_cfg.get("nonveg_max", 0.3)),
        help="Max NDVI below this is treated as non-vegetated Other.",
    )
    parser.add_argument(
        "--forest-min-threshold",
        type=float,
        default=float(thresholds_cfg.get("forest_min", 0.7)),
        help="Min NDVI above this is treated as permanent Vegetation/Forest.",
    )
    parser.add_argument(
        "--majority-filter",
        action="store_true",
        default=bool(rule_cfg.get("majority_filter", False)),
        help="Apply 3x3 majority (mode) filter on final classes.",
    )
    parser.add_argument(
        "--no-majority-filter",
        action="store_false",
        dest="majority_filter",
        help="Disable majority filter.",
    )
    parser.add_argument(
        "--output-schema",
        choices=["project5", "ag4"],
        default=rule_cfg.get("output_schema", "project5"),
        help="Class ID schema: project5(0 Abandoned,1 Active,2 Fallow,3 Vegetation,4 Other) or ag4(0 Active,1 Fallow,2 Abandoned,3 Other).",
    )
    parser.add_argument(
        "--nodata",
        type=int,
        default=int(rule_cfg.get("nodata", -1)),
        help="Nodata value used in output raster.",
    )
    parser.add_argument(
        "--tile-size-deg",
        type=float,
        default=float(rule_cfg.get("tile_size_deg", 0.01)),
        help="Tile size in degrees used for downloading (default 0.01).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=cache_dir,
        help="Optional cache directory for downloaded tiles.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=output_path,
        help="Output GeoTIFF path.",
    )
    parser.add_argument(
        "--download-scale",
        type=int,
        default=int(rule_cfg.get("download_scale", 10)),
        help="Pixel scale (m) for downloading tiles.",
    )
    return parser


def parse_args() -> tuple[argparse.Namespace, DictConfig]:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to rule-based config YAML.",
    )
    base_args, remaining = base_parser.parse_known_args()
    cfg = load_config(base_args.config)
    parser = build_arg_parser(cfg, parents=[base_parser])
    args = parser.parse_args(remaining)
    args.config = base_args.config
    return args, cfg


def main() -> None:
    args, _ = parse_args()
    initialize_ee(args.project_id)
    aoi, palika_slug = get_palika_geometry(args.palika)

    prev_year = args.target_year - 1
    next_year = args.target_year + 1

    print(f"Building rule-based classification for {palika_slug}...")
    print(f"Years: {prev_year}, {args.target_year}, {next_year}")

    class_img = build_rule_based_classification(
        aoi=aoi,
        prev_year=prev_year,
        target_year=args.target_year,
        next_year=next_year,
        cloud_filter=args.cloud_filter,
        peak_threshold=args.peak_threshold,
        abandoned_mean_low=args.abandoned_mean_low,
        abandoned_mean_high=args.abandoned_mean_high,
        other_mean_low=args.other_mean_low,
        nonveg_max_threshold=args.nonveg_max_threshold,
        forest_min_threshold=args.forest_min_threshold,
        majority_filter=args.majority_filter,
        output_schema=args.output_schema,
        nodata_value=args.nodata,
    )

    download_classification_tiled(
        class_img=class_img,
        aoi=aoi,
        output_raster=args.output,
        tile_size_deg=args.tile_size_deg,
        nodata_value=args.nodata,
        cache_dir=args.cache_dir,
        scale=args.download_scale,
    )

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
