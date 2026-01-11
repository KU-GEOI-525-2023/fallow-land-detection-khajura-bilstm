from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import ee
import geemap
import numpy as np
import rioxarray  # noqa: F401
import torch
import xarray as xr
from nepal_palika_finder.locator import PalikaLocator
from omegaconf import DictConfig, OmegaConf

from land_classifier.models import BiLSTM
from land_classifier.utils import get_logger

log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT / "configs/inference/defaults.yaml"


def _require(value: object, name: str) -> object:
    if value is None:
        raise ValueError(f"Missing required config value: {name}")
    return value


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def initialize_ee(project_id: str) -> None:
    try:
        ee.Initialize(project=project_id)
        log.info("Earth Engine Initialized: %s", project_id)
    except Exception:
        log.info("Requesting EE Authentication...")
        ee.Authenticate()
        ee.Initialize(project=project_id)


def mask_s2(image: ee.Image) -> ee.Image:
    """Scaled reflectance and masking matching training."""
    scaled = image.divide(10000).select("B.*")
    scl = image.select("SCL")
    keep = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    qa = image.select("QA60")
    qa_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return (
        scaled.updateMask(keep)
        .updateMask(qa_mask)
        .copyProperties(image, ["system:time_start"])
    )


def add_indices(image: ee.Image) -> ee.Image:
    """Spectral indices matching extraction logic."""
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("ndvi")
    evi = image.expression(
        "2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))",
        {"B8": image.select("B8"), "B4": image.select("B4"), "B2": image.select("B2")},
    ).rename("evi")
    bsi = image.expression(
        "((B12 + B4) - (B8 + B2)) / ((B12 + B4) + (B8 + B2))",
        {
            "B12": image.select("B12"),
            "B4": image.select("B4"),
            "B8": image.select("B8"),
            "B2": image.select("B2"),
        },
    ).rename("bsi")
    return image.addBands([ndvi, evi, bsi])


def get_stacked_image(
    aoi: ee.Geometry,
    *,
    start_date: str,
    end_date: str,
    context_length: int,
    step_days: int,
    nodata: float,
    cloud_pct: int,
) -> ee.Image:
    """Build the time-series stack for the requested window."""
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(mask_s2)
        .map(add_indices)
        .select(["B2", "B3", "B4", "B8", "B11", "B12", "ndvi", "evi", "bsi"])
    )

    start = ee.Date(start_date)
    band_names = ["B2", "B3", "B4", "B8", "B11", "B12", "ndvi", "evi", "bsi"]

    def make_step(n: int) -> ee.Image:
        n_num = ee.Number(n)
        t1 = start.advance(n_num.multiply(step_days), "day")
        t2 = t1.advance(step_days, "day")
        subset = collection.filterDate(t1, t2)

        empty_img = (
            ee.Image.constant([nodata] * 9)
            .rename(band_names)
            .cast({b: "float" for b in band_names})
        )

        comp = subset.median().select(band_names)
        return ee.Algorithms.If(subset.size().gt(0), comp, empty_img)

    img_list = ee.List.sequence(0, context_length - 1).map(make_step)
    return ee.ImageCollection.fromImages(img_list).toBands()


def _env_path(name: str, default: Path) -> Path:
    return Path(value) if (value := os.getenv(name)) else default


def _env_value(name: str, default: object, cast: type) -> object:
    if (value := os.getenv(name)) is None:
        return default
    return cast(value)


def load_config(config_path: Path | None = None) -> DictConfig:
    resolved_path = config_path or DEFAULT_CONFIG_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"Inference config not found: {resolved_path}")
    cfg = OmegaConf.load(resolved_path)
    OmegaConf.resolve(cfg)
    return cfg


def build_arg_parser(
    cfg: DictConfig, parents: list[argparse.ArgumentParser] | None = None
) -> argparse.ArgumentParser:
    cfg = OmegaConf.create(cfg)
    paths_cfg = cfg.get("paths", {})
    inf_cfg = cfg.get("inference", {})
    model_cfg = cfg.get("model", {})

    model_path = _resolve_path(
        ROOT, _require(paths_cfg.get("model_path"), "paths.model_path")
    )
    output_raster = _resolve_path(
        ROOT, _require(paths_cfg.get("output_raster"), "paths.output_raster")
    )
    output_composite = _resolve_path(
        ROOT, _require(paths_cfg.get("output_composite"), "paths.output_composite")
    )

    parser = argparse.ArgumentParser(
        description="Run deep learning inference and export GeoTIFFs.",
        parents=parents or [],
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=_env_path("MODEL_PATH", model_path),
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--output-raster",
        type=Path,
        default=_env_path("OUTPUT_RASTER", output_raster),
        help="Output path for the classified GeoTIFF.",
    )
    parser.add_argument(
        "--output-composite",
        type=Path,
        default=_env_path("OUTPUT_COMPOSITE", output_composite),
        help="Output path for the cloud-free composite GeoTIFF.",
    )
    parser.add_argument(
        "--palika",
        type=str,
        default=os.getenv("PALIKA_NAME", inf_cfg.get("palika_name", "Khajura")),
        help="Palika name to run inference for.",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=_env_value("INPUT_DIM", model_cfg.get("input_dim", 9), int),
        help="Number of input features per timestep.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=_env_value("HIDDEN_DIM", model_cfg.get("hidden_dim", 64), int),
        help="Hidden dimension size for the BiLSTM.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=_env_value("NUM_LAYERS", model_cfg.get("num_layers", 2), int),
        help="Number of BiLSTM layers.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=_env_value("NUM_CLASSES", model_cfg.get("num_classes", 5), int),
        help="Number of output classes.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=_env_value("DROPOUT", model_cfg.get("dropout", 0.0), float),
        help="Dropout rate used when constructing the model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_env_value("BATCH_SIZE", inf_cfg.get("batch_size", 8192), int),
        help="Inference batch size for per-tile prediction.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default=os.getenv("INFERENCE_DEVICE", inf_cfg.get("device", "auto")),
        help="Device to run inference on.",
    )
    return parser


def parse_args() -> tuple[argparse.Namespace, DictConfig]:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to inference config YAML.",
    )
    base_args, remaining = base_parser.parse_known_args()
    cfg = load_config(base_args.config)
    parser = build_arg_parser(cfg, parents=[base_parser])
    args = parser.parse_args(remaining)
    args.config = base_args.config
    return args, cfg


def apply_cli_overrides(
    cfg: DictConfig, args: argparse.Namespace
) -> DictConfig:
    overrides = {
        "paths": {
            "model_path": str(args.model_path),
            "output_raster": str(args.output_raster),
            "output_composite": str(args.output_composite),
        },
        "inference": {
            "palika_name": args.palika,
            "batch_size": args.batch_size,
            "device": args.device,
        },
        "model": {
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_classes": args.num_classes,
            "dropout": args.dropout,
        },
    }
    return OmegaConf.merge(cfg, overrides)


def run_inference(cfg: DictConfig | dict) -> None:
    cfg = OmegaConf.create(cfg)
    gee_cfg = cfg.get("gee", {})
    inf_cfg = cfg.get("inference", {})
    model_cfg = cfg.get("model", {})
    paths_cfg = cfg.get("paths", {})

    project_id = str(_require(gee_cfg.get("project_id"), "gee.project_id"))
    start_date = str(_require(gee_cfg.get("start_date"), "gee.start_date"))
    end_date = str(_require(gee_cfg.get("end_date"), "gee.end_date"))
    cloud_pct = int(gee_cfg.get("cloud_pct", 60))
    scale = int(gee_cfg.get("scale", 10))

    palika_name = str(_require(inf_cfg.get("palika_name"), "inference.palika_name"))
    context_length = int(
        _require(inf_cfg.get("context_length"), "inference.context_length")
    )
    step_days = int(_require(inf_cfg.get("step_days"), "inference.step_days"))
    nodata = float(_require(inf_cfg.get("nodata"), "inference.nodata"))
    tile_size_deg = float(inf_cfg.get("tile_size_deg", 0.01))
    batch_size = int(_require(inf_cfg.get("batch_size"), "inference.batch_size"))
    device = str(inf_cfg.get("device", "auto"))

    model_path = _resolve_path(
        ROOT, _require(paths_cfg.get("model_path"), "paths.model_path")
    )
    output_raster = _resolve_path(
        ROOT, _require(paths_cfg.get("output_raster"), "paths.output_raster")
    )
    output_composite = _resolve_path(
        ROOT, _require(paths_cfg.get("output_composite"), "paths.output_composite")
    )
    cache_dir = _resolve_path(ROOT, paths_cfg.get("cache_dir", "data/processed/cache"))

    input_dim = int(_require(model_cfg.get("input_dim"), "model.input_dim"))
    hidden_dim = int(_require(model_cfg.get("hidden_dim"), "model.hidden_dim"))
    num_layers = int(_require(model_cfg.get("num_layers"), "model.num_layers"))
    num_classes = int(_require(model_cfg.get("num_classes"), "model.num_classes"))
    dropout = float(model_cfg.get("dropout", 0.0))

    start_time = time.time()
    initialize_ee(project_id)

    locator = PalikaLocator()
    try:
        match = locator.search_palikas_by_name(palika_name)[0]
    except IndexError:
        log.error("No palika match found for %s.", palika_name)
        return

    palika_label = match["GaPa_NaPa"] or match["PALIKA"]
    palika_slug = palika_label.replace(" ", "_")
    try:
        geom_raw = locator.get_palika_geometry_by_name(palika_label, match["DISTRICT"])
    except Exception as exc:
        log.error("Failed to fetch palika geometry for %s: %s", palika_name, exc)
        return
    geometry = {"type": geom_raw["type"], "coordinates": geom_raw["coordinates"]}
    aoi = ee.Geometry(geometry)

    cache_path = cache_dir / f"{palika_slug}_{start_date}_{end_date}_{scale}m"
    cache_path.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"
    device = torch.device(device)
    if not model_path.exists():
        log.error("Model path %s not found. Please train first.", model_path)
        return

    model = BiLSTM(
        input_dim, hidden_dim, num_layers, num_classes, dropout=dropout
    ).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(
        ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    )
    model.eval()

    stats_path = model_path.parent / "normalization_stats.pt"
    if not stats_path.exists():
        log.error("Stats path %s not found.", stats_path)
        return
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    means, stds = stats["mean"], stats["std"]

    log.info(
        "Fetching multi-temporal data from GEE at %dm scale (Tiled for Memory)...",
        scale,
    )
    stacked = get_stacked_image(
        aoi,
        start_date=start_date,
        end_date=end_date,
        context_length=context_length,
        step_days=step_days,
        nodata=nodata,
        cloud_pct=cloud_pct,
    )

    bounds = aoi.bounds().getInfo()["coordinates"][0]
    lons, lats = zip(*bounds)
    west, east = min(lons), max(lons)
    south, north = min(lats), max(lats)

    lon_grid = np.arange(west, east, tile_size_deg)
    lat_grid = np.arange(south, north, tile_size_deg)

    log.info("Tiling AOI into %dx%d grid...", len(lon_grid), len(lat_grid))

    stitched_preds = []
    stitched_composites = []
    total_tiles = len(lon_grid) * len(lat_grid)
    completed = 0
    fallback_shape: tuple[int, int, int] | None = None

    for lat_idx, lat in enumerate(reversed(lat_grid)):
        row_preds = []
        row_composites = []
        row_shape: tuple[int, int, int] | None = None
        for lon_idx, lon in enumerate(lon_grid):
            tile_aoi = ee.Geometry.Rectangle(
                [lon, lat, lon + tile_size_deg, lat + tile_size_deg]
            )
            cache_file = cache_path / f"tile_{lat_idx}_{lon_idx}.npy"

            try:
                try:
                    data_tile = np.load(cache_file)
                except FileNotFoundError:
                    data_tile = geemap.ee_to_numpy(
                        stacked, region=tile_aoi, scale=scale
                    )
                    np.save(cache_file, data_tile)

                if data_tile.size == 0 or np.all(data_tile == nodata):
                    if data_tile.ndim == 3:
                        h_t, w_t, _ = data_tile.shape
                        row_shape = row_shape or data_tile.shape
                        fallback_shape = fallback_shape or data_tile.shape
                    else:
                        shape = row_shape or fallback_shape or (
                            100,
                            100,
                            input_dim * context_length,
                        )
                        h_t, w_t = shape[:2]
                    row_preds.append(np.full((h_t, w_t), -1.0, dtype=np.float32))
                    row_composites.append(
                        np.full((h_t, w_t, input_dim), nodata, dtype=np.float32)
                    )
                    completed += 1
                    continue

                h_t, w_t, _ = data_tile.shape
                row_shape = row_shape or data_tile.shape
                fallback_shape = fallback_shape or data_tile.shape
                tile_reshaped = data_tile.reshape(
                    h_t, w_t, context_length, input_dim
                )

                composite_mask = np.where(
                    tile_reshaped == nodata, np.nan, tile_reshaped
                )
                median_tile = np.nanmedian(composite_mask, axis=2)
                row_composites.append(
                    np.nan_to_num(median_tile, nan=nodata).astype(np.float32)
                )

                tile_np = tile_reshaped.transpose(3, 2, 0, 1)

                tile_xr = xr.DataArray(tile_np, dims=("feature", "time", "y", "x"))
                tile_xr = tile_xr.where(tile_xr != nodata, np.nan).interpolate_na(
                    dim="time", method="linear"
                )
                tile_xr = tile_xr.bfill("time").ffill("time").fillna(0.0)

                feature_count, time_steps, height, width = tile_xr.shape
                pixel_series = tile_xr.values.reshape(
                    feature_count, time_steps, -1
                ).transpose(2, 1, 0)

                normalized = (np.nan_to_num(pixel_series) - means) / (stds + 1e-8)

                batch_preds = []
                with torch.no_grad():
                    for i in range(0, normalized.shape[0], batch_size):
                        batch = (
                            torch.from_numpy(normalized[i : i + batch_size])
                            .float()
                            .to(device)
                        )
                        batch_preds.append(model(batch).argmax(dim=1).cpu().numpy())

                tile_map = np.concatenate(batch_preds).reshape(height, width).astype(
                    np.float32
                )

                valid_mask = np.any(tile_xr.values != 0, axis=(0, 1))
                tile_map[~valid_mask] = -1.0

                row_preds.append(tile_map)

                completed += 1
                if completed % 10 == 0 or completed == total_tiles:
                    log.info(
                        "Progress: %d/%d tiles (%.1f%%)",
                        completed,
                        total_tiles,
                        (completed / total_tiles) * 100,
                    )

                del (
                    data_tile,
                    tile_reshaped,
                    tile_xr,
                    pixel_series,
                    normalized,
                    batch_preds,
                    tile_map,
                    composite_mask,
                    median_tile,
                )
                gc.collect()

            except Exception as e:
                log.error("Tile at %d_%d failed: %s", lat_idx, lon_idx, e)
                shape = row_shape or fallback_shape
                if shape is None:
                    shape = (10, 10, input_dim * context_length)
                h_t, w_t, _ = shape
                row_preds.append(np.full((h_t, w_t), -1.0, dtype=np.float32))
                row_composites.append(
                    np.full((h_t, w_t, input_dim), nodata, dtype=np.float32)
                )

        stitched_preds.append(np.concatenate(row_preds, axis=1))
        stitched_composites.append(np.concatenate(row_composites, axis=1))
        del row_preds, row_composites
        gc.collect()

    final_map = np.concatenate(stitched_preds, axis=0)
    final_composite = np.concatenate(stitched_composites, axis=0)
    height, width = final_map.shape
    log.info("Classification map and composite stitched: %dx%d", height, width)

    result = (
        xr.DataArray(
            final_map,
            coords={
                "y": np.linspace(north, south, height),
                "x": np.linspace(west, east, width),
            },
            dims=("y", "x"),
        )
        .rio.write_crs("EPSG:4326")
        .rio.write_nodata(-1.0)
    )

    comp_result = (
        xr.DataArray(
            final_composite.transpose(2, 0, 1),
            coords={
                "band": ["B2", "B3", "B4", "B8", "B11", "B12", "ndvi", "evi", "bsi"],
                "y": np.linspace(north, south, height),
                "x": np.linspace(west, east, width),
            },
            dims=("band", "y", "x"),
        )
        .rio.write_crs("EPSG:4326")
        .rio.write_nodata(nodata)
    )

    try:
        result = result.rio.clip([geometry], "EPSG:4326")
        comp_result = comp_result.rio.clip([geometry], "EPSG:4326")
    except Exception as e:
        log.warning("Final clip failed: %s", e)

    log.info("Exporting classified GeoTIFF to %s", output_raster)
    output_raster.parent.mkdir(parents=True, exist_ok=True)
    result.rio.to_raster(output_raster)

    log.info("Exporting cloud-free composite to %s", output_composite)
    comp_result.rio.to_raster(output_composite)

    log.info("Inference complete in %.1f seconds.", time.time() - start_time)


if __name__ == "__main__":
    args, cfg = parse_args()
    cfg = apply_cli_overrides(cfg, args)
    run_inference(cfg)
