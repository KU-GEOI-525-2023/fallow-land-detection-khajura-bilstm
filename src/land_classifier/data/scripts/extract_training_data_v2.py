from __future__ import annotations

import argparse
from pathlib import Path

import ee
import geemap
import geopandas as gpd
import numpy as np
from omegaconf import DictConfig, OmegaConf
from shapely.geometry import mapping
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = ROOT / "configs/data/dataset_v2.yaml"


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


def initialize_ee(project_id: str) -> None:
    print("--- Initializing Earth Engine ---")
    try:
        ee.Initialize(project=project_id)
        print("Earth Engine Initialized successfully.")
    except Exception as exc:
        print(f"Initialization failed: {exc}")
        print("Attempting to authenticate...")
        ee.Authenticate()
        ee.Initialize(project=project_id)


def mask_s2(image: ee.Image, mask_mode: str) -> ee.Image:
    """Scaled reflectance with optional masking."""
    scaled = image.divide(10000).select("B.*")

    match mask_mode:
        case "none":
            return scaled.copyProperties(image, ["system:time_start"])
        case "qa60":
            qa = image.select("QA60")
            mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
            return scaled.updateMask(mask).copyProperties(image, ["system:time_start"])
        case "scl":
            scl = image.select("SCL")
            keep = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
            qa = image.select("QA60")
            qa_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
            return (
                scaled.updateMask(keep)
                .updateMask(qa_mask)
                .copyProperties(image, ["system:time_start"])
            )
        case _:
            return scaled.copyProperties(image, ["system:time_start"])


def add_indices(image: ee.Image) -> ee.Image:
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


def build_s2_collection(
    aoi: ee.Geometry,
    cloud_filter: int,
    *,
    start_date: str,
    end_date: str,
    mask_mode: str,
    bands: list[str],
) -> ee.ImageCollection:
    raw = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_filter))
    )

    try:
        print(
            f"DEBUG: Raw images found (cloud<{cloud_filter}%): {raw.size().getInfo()}"
        )
    except Exception:
        pass

    def _mask(img: ee.Image) -> ee.Image:
        return mask_s2(img, mask_mode)

    return raw.map(_mask).map(add_indices).select(bands)


def get_step_timeseries(
    aoi: ee.Geometry,
    cloud_filter: int,
    *,
    start_date: str,
    end_date: str,
    step_days: int,
    nodata: float,
    mask_mode: str,
    bands: list[str],
    sample_scale: int,
) -> ee.ImageCollection:
    """
    20-day median composites.

    Key fix: avoid filter(notNull(["ndvi"])) since notNull checks properties, not bands.
    """
    collection = build_s2_collection(
        aoi,
        cloud_filter,
        start_date=start_date,
        end_date=end_date,
        mask_mode=mask_mode,
        bands=bands,
    )

    try:
        if collection.size().getInfo() == 0:
            return ee.ImageCollection([])
    except Exception:
        pass

    first = ee.Image(collection.first())
    native_proj = first.select("ndvi").projection()

    # Debug: centroid value in first image
    try:
        centroid = aoi.centroid(1)
        test_val = (
            first.select("ndvi")
            .reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=centroid,
                scale=sample_scale,
                bestEffort=True,
                maxPixels=1e9,
            )
            .getInfo()
        )
        print("DEBUG: first-image NDVI at AOI centroid:", test_val)
    except Exception:
        pass

    start = ee.Date(start_date)
    end = ee.Date(end_date)
    n_steps = end.difference(start, "day").divide(step_days).ceil()

    def make_step(n):
        n = ee.Number(n)
        t1 = start.advance(n.multiply(step_days), "day")
        t2 = t1.advance(step_days, "day")
        subset = collection.filterDate(t1, t2)

        empty_img = (
            ee.Image.constant([nodata] * len(bands))
            .rename(bands)
            .cast({band: "float" for band in bands})
            .set("system:time_start", t1.millis())
        )

        comp = (
            subset.median()
            .select(bands)
            .set("system:time_start", t1.millis())
        )

        return ee.Algorithms.If(subset.size().gt(0), comp, empty_img)

    img_list = ee.List.sequence(0, n_steps.subtract(1)).map(make_step)

    ic = ee.ImageCollection.fromImages(img_list)

    def finalize(img):
        return ee.Image(img).setDefaultProjection(native_proj)

    return ic.map(finalize)


def build_step_collection_with_fallback(
    aoi: ee.Geometry,
    *,
    cloud_filters: list[int],
    start_date: str,
    end_date: str,
    step_days: int,
    nodata: float,
    mask_mode: str,
    bands: list[str],
    sample_scale: int,
) -> tuple[ee.ImageCollection, int]:
    for cf in cloud_filters:
        step_col = get_step_timeseries(
            aoi,
            cf,
            start_date=start_date,
            end_date=end_date,
            step_days=step_days,
            nodata=nodata,
            mask_mode=mask_mode,
            bands=bands,
            sample_scale=sample_scale,
        )
        size = step_col.size().getInfo()
        print(f"DEBUG: Step composites available with cloud<{cf}%: {size}")
        if size > 0:
            return step_col, cf
    return ee.ImageCollection([]), cloud_filters[-1] if cloud_filters else 0


def _coerce_value(value: object) -> float:
    if value is None:
        return 0.0
    try:
        if np.isnan(value):
            return 0.0
    except TypeError:
        pass
    return float(value)


def extract_data(cfg: DictConfig | dict) -> None:
    cfg = OmegaConf.create(cfg)
    extract_cfg = cfg.get("extract")
    if extract_cfg is None:
        extract_cfg = cfg
    project_id = str(
        _require(extract_cfg.get("project", {}).get("id"), "project.id")
    )
    mask_mode = str(extract_cfg.get("mask_mode", "scl"))

    dates_cfg = extract_cfg.get("dates", {})
    start_date = dates_cfg.get("start_date")
    end_date = dates_cfg.get("end_date")
    target_year = dates_cfg.get("target_year")
    if not start_date or not end_date:
        if target_year is None:
            raise ValueError("dates.start_date/end_date or dates.target_year required.")
        target_year = int(target_year)
        start_date = f"{target_year - 1}-01-01"
        end_date = f"{target_year + 1}-12-31"
    start_date = str(start_date)
    end_date = str(end_date)

    cloud_filters = [int(cf) for cf in extract_cfg.get("cloud_filters", [])]
    if not cloud_filters:
        raise ValueError("cloud_filters must be defined in the config.")

    step_days = int(_require(extract_cfg.get("step_days"), "step_days"))
    nodata = float(_require(extract_cfg.get("nodata"), "nodata"))

    paths_cfg = extract_cfg.get("paths", {})
    data_dir = _resolve_path(_require(paths_cfg.get("data_dir"), "paths.data_dir"))
    output_file_x = _resolve_path(
        _require(paths_cfg.get("output_file_x"), "paths.output_file_x")
    )
    output_file_y = _resolve_path(
        _require(paths_cfg.get("output_file_y"), "paths.output_file_y")
    )
    output_file_sources = _resolve_path(
        _require(paths_cfg.get("output_file_sources"), "paths.output_file_sources")
    )
    processed_dir_value = paths_cfg.get("processed_dir")
    if processed_dir_value is None:
        processed_dir = output_file_x.parent
    else:
        processed_dir = _resolve_path(processed_dir_value)

    classes = extract_cfg.get("classes", {})
    class_map = extract_cfg.get("class_map", {})
    if not classes or not class_map:
        raise ValueError("classes and class_map must be defined in the config.")

    bands = list(extract_cfg.get("bands", []))
    if not bands:
        raise ValueError("bands must be defined in the config.")

    sampling_cfg = extract_cfg.get("sampling", {})
    sample_scale = int(sampling_cfg.get("scale", 10))
    tile_scale = int(sampling_cfg.get("tile_scale", 4))
    probe_buffer_m = int(sampling_cfg.get("probe_buffer_m", 20))

    initialize_ee(project_id)
    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    source_batches: list[np.ndarray] = []
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"DEBUG: MASK_MODE = {mask_mode}")
    print(f"DEBUG: NODATA sentinel = {nodata}")
    print(f"DEBUG: STEP_DAYS = {step_days}, date range {start_date} → {end_date}")
    print(f"DEBUG: CLOUD_FILTERS fallback = {cloud_filters}")

    start_ee = ee.Date(start_date)
    end_ee = ee.Date(end_date)
    step_count = int(
        end_ee.difference(start_ee, "day").divide(step_days).ceil().getInfo()
    )

    for class_name, rel_path in classes.items():
        print(f"\n--- Processing Class: {class_name} ---")
        shp_path = (data_dir / rel_path).resolve()

        try:
            gdf = gpd.read_file(shp_path)
            if gdf.empty:
                print("Empty shapefile.")
                continue

            gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]

            if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")

            gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
            if gdf.empty:
                print("No polygon geometries after filtering.")
                continue

            ee_features = [
                ee.Feature(ee.Geometry(mapping(geom))).set("poly_idx", int(idx))
                for idx, geom in enumerate(gdf.geometry.reset_index(drop=True))
                if geom is not None
            ]
            if not ee_features:
                print("No valid polygons after filtering.")
                continue

            ee_fc = ee.FeatureCollection(ee_features)
            aoi = ee_fc.geometry()

            step_col, used_cf = build_step_collection_with_fallback(
                aoi,
                cloud_filters=cloud_filters,
                start_date=start_date,
                end_date=end_date,
                step_days=step_days,
                nodata=nodata,
                mask_mode=mask_mode,
                bands=bands,
                sample_scale=sample_scale,
            )
            if (step_size := step_col.size().getInfo()) == 0:
                print(
                    "WARNING: No step composites available even after fallback. Skipping class."
                )
                continue

            print("DEBUG: Probing first feature geometry...")
            first_geom = ee.Feature(ee_fc.first()).geometry()
            probe_idx = min(5, step_size - 1)
            probe_img = ee.Image(step_col.toList(step_col.size()).get(probe_idx))
            probe_val = probe_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=first_geom.buffer(probe_buffer_m),
                scale=sample_scale,
                bestEffort=True,
                maxPixels=1e9,
            ).getInfo()
            print(
                f"DEBUG PROBE (Step {probe_idx} mean in 20m buffer, cloud<{used_cf}%): {probe_val}"
            )

            print(f"DEBUG: Feature collection size: {ee_fc.size().getInfo()}")

            stacked = step_col.toBands()
            print(f"DEBUG: Stacked image bands: {len(stacked.bandNames().getInfo())}")

            print("DEBUG: Sampling regions...")
            sample_fc = stacked.sampleRegions(
                collection=ee_fc,
                scale=sample_scale,
                geometries=False,
                tileScale=tile_scale,
            )

            sample_size = sample_fc.size().getInfo()
            print(f"DEBUG: Sample size: {sample_size}")

            if sample_size == 0:
                print(
                    "WARNING: sampleRegions returned 0 features. Trying reduceRegions as fallback..."
                )
                sample_fc = stacked.reduceRegions(
                    collection=ee_fc,
                    reducer=ee.Reducer.first(),
                    scale=sample_scale,
                    tileScale=tile_scale,
                )
                sample_size = sample_fc.size().getInfo()
                print(f"DEBUG: reduceRegions sample size: {sample_size}")

            samples_df = geemap.ee_to_df(sample_fc)
            if samples_df.empty:
                print("DataFrame empty.")
                continue

            feature_cols = [
                c
                for c in samples_df.columns
                if c.endswith(tuple(f"_{band}" for band in bands))
            ]
            if not feature_cols:
                print("No feature columns found (unexpected).")
                continue

            valid_mask = (samples_df[feature_cols] != nodata).any(axis=1)
            df_clean = samples_df.loc[valid_mask].copy()

            dropped = len(samples_df) - len(df_clean)
            if dropped > 0:
                print(
                    f"INFO: Dropped {dropped} rows with all-NODATA. Remaining: {len(df_clean)}"
                )

            if df_clean.empty:
                print("WARNING: All samples were NODATA after sampling.")
                continue

            df_clean[feature_cols] = df_clean[feature_cols].replace(nodata, 0.0)

            features = [
                [
                    [
                        _coerce_value(row.get(f"{t}_{band}", 0.0))
                        for band in bands
                    ]
                    for t in range(step_count)
                ]
                for _, row in tqdm(df_clean.iterrows(), total=len(df_clean))
            ]

            feature_array = np.array(features, dtype=np.float32)
            label_array = np.full(
                len(feature_array),
                int(class_map[class_name]),
                dtype=np.int64,
            )

            feature_batches.append(feature_array)
            label_batches.append(label_array)
            if "poly_idx" in df_clean.columns:
                source_batches.append(df_clean["poly_idx"].astype(int).to_numpy())
            else:
                source_batches.append(np.full(len(feature_array), -1, dtype=np.int64))

            print(
                f"DEBUG: Collected {len(feature_array)} samples for {class_name} with T={step_count}"
            )
        except Exception as exc:
            print(f"Error processing {class_name}: {exc}")
            continue

    if feature_batches:
        features_final = np.concatenate(feature_batches, axis=0)
        labels_final = np.concatenate(label_batches, axis=0)

        print(f"\nSUCCESS! Extracted {len(features_final)} samples.")
        print(f"X shape: {features_final.shape}  (N, T, F)")
        print(f"y shape: {labels_final.shape}")

        np.save(output_file_x, features_final)
        np.save(output_file_y, labels_final)
        sources_final = (
            np.concatenate(source_batches)
            if source_batches
            else np.array([], dtype=np.int64)
        )
        np.save(output_file_sources, sources_final)

        print(f"Saved: {output_file_x}")
        print(f"Saved: {output_file_y}")
        print(f"Saved: {output_file_sources}")
    else:
        print("\nFailed to extract any data.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract training data from GEE time series."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to extraction config YAML.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_data(load_config(args.config))
