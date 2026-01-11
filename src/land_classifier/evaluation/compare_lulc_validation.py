from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
from rasterio.enums import Resampling

MODEL_CLASS_MAP = {
    0: "Abandoned",
    1: "Active",
    2: "Fallow",
    3: "Vegetation",
    4: "Other",
}

ESRI_CLASS_MAP = {
    0: "nodata",
    1: "water",
    2: "trees",
    3: "grass",
    4: "flooded veg",
    5: "crops",
    6: "scrub",
    7: "built area",
    8: "bare",
    9: "snow/ice",
    10: "clouds",
}


def _load_single_band(path: Path) -> rioxarray.DataArray:
    da = rioxarray.open_rasterio(path)
    if "band" in da.dims and da.sizes.get("band", 1) == 1:
        da = da.squeeze("band", drop=True)
    return da


def _align_esri_to_model(
    model_da: rioxarray.DataArray, esri_da: rioxarray.DataArray
) -> rioxarray.DataArray:
    if not esri_da.rio.crs:
        raise ValueError("ESRI raster is missing CRS.")
    if not model_da.rio.crs:
        raise ValueError("Model raster is missing CRS.")

    return esri_da.rio.reproject_match(
        model_da, resampling=Resampling.nearest
    ).astype(np.int32)


def _build_valid_mask(
    model_da: rioxarray.DataArray,
    esri_da: rioxarray.DataArray,
    model_nodata: float | int | None,
    esri_nodata: float | int | None,
) -> np.ndarray:
    model_data = model_da.values
    esri_data = esri_da.values

    valid = np.ones(model_data.shape, dtype=bool)

    if model_nodata is not None:
        valid &= model_data != model_nodata
    if esri_nodata is not None:
        valid &= esri_data != esri_nodata

    valid &= model_data >= 0
    valid &= esri_data >= 0
    return valid


def _build_crosstab(model_vals: np.ndarray, esri_vals: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"esri": esri_vals, "model": model_vals})
    return pd.crosstab(df["esri"], df["model"])


def _label_crosstab(crosstab: pd.DataFrame) -> pd.DataFrame:
    crosstab = crosstab.rename(index=ESRI_CLASS_MAP, columns=MODEL_CLASS_MAP)
    crosstab.index.name = "esri_class"
    crosstab.columns.name = "model_class"
    return crosstab


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def _plot_heatmap(
    data: pd.DataFrame,
    output_path: Path,
    title: str,
    cmap: str = "viridis",
    fmt: str = "d",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    values = data.values
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(values, cmap=cmap)

    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticklabels(data.index)
    ax.set_xlabel("Model Classes")
    ax.set_ylabel("ESRI Classes")
    ax.set_title(title)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j,
                i,
                format(values[i, j], fmt),
                ha="center",
                va="center",
                color="white" if values[i, j] > values.max() * 0.6 else "black",
                fontsize=8,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_bar(
    data: pd.Series,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(data.index.astype(str), data.values, color="#2a6f97")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(data.index)))
    ax.set_xticklabels(data.index.astype(str), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def compare_lulc(
    model_path: Path,
    esri_path: Path,
    output_dir: Path,
    pixel_area_ha: float,
    figure_dir: Path | None = None,
) -> None:
    model_da = _load_single_band(model_path)
    esri_da = _load_single_band(esri_path)
    esri_aligned = _align_esri_to_model(model_da, esri_da)

    model_nodata = model_da.rio.nodata
    esri_nodata = esri_da.rio.nodata
    valid = _build_valid_mask(model_da, esri_aligned, model_nodata, esri_nodata)

    model_vals = model_da.values[valid].astype(np.int32)
    esri_vals = esri_aligned.values[valid].astype(np.int32)

    crosstab = _build_crosstab(model_vals, esri_vals)
    labeled = _label_crosstab(crosstab)

    area_ha = labeled * pixel_area_ha
    area_ha = area_ha.round(2)

    comparison_csv = output_dir / "comparison_stats.csv"
    area_csv = output_dir / "comparison_area_ha.csv"

    _write_csv(labeled, comparison_csv)
    _write_csv(area_ha, area_csv)

    crops_mask = esri_vals == 5
    crops_model = model_vals[crops_mask]
    crops_counts = pd.Series(crops_model).value_counts().sort_index()
    crops_df = crops_counts.rename(index=MODEL_CLASS_MAP).to_frame("pixel_count")
    crops_df["area_ha"] = (crops_df["pixel_count"] * pixel_area_ha).round(2)
    crops_df["percentage"] = (
        (crops_df["pixel_count"] / crops_df["pixel_count"].sum()) * 100
    ).round(2)
    crops_df.index.name = "model_class"

    crops_csv = output_dir / "crops_distribution.csv"
    _write_csv(crops_df, crops_csv)

    if figure_dir is not None:
        figure_dir.mkdir(parents=True, exist_ok=True)

        _plot_heatmap(
            labeled,
            figure_dir / "lulc_crosstab_counts.png",
            "ESRI vs Model Cross-Tab (Counts)",
            cmap="viridis",
            fmt="d",
        )

        labeled_pct = labeled.div(labeled.sum(axis=1).replace(0, np.nan), axis=0) * 100
        _plot_heatmap(
            labeled_pct.fillna(0),
            figure_dir / "lulc_crosstab_percent.png",
            "ESRI vs Model Cross-Tab (Row %)",
            cmap="magma",
            fmt=".1f",
        )

        _plot_heatmap(
            labeled_pct.fillna(0),
            figure_dir / "lulc_confusion_matrix.png",
            "ESRI vs Model Confusion Matrix (Row %)",
            cmap="plasma",
            fmt=".1f",
        )

        _plot_bar(
            crops_df["percentage"],
            figure_dir / "esri_crops_model_split.png",
            "ESRI Crops Split Across Model Classes",
            "Model Class",
            "Percentage (%)",
        )

    total_pixels = int(valid.sum())
    total_area = round(total_pixels * pixel_area_ha, 2)
    print("Aligned comparison complete.")
    print(f"Valid pixels compared: {total_pixels}")
    print(f"Total area (Ha): {total_area}")
    print(f"Saved: {comparison_csv}")
    print(f"Saved: {area_csv}")
    print(f"Saved: {crops_csv}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare ESRI LULC vs model classification for Khajura."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/exports/khajura_classification.tif"),
        help="Path to model classification GeoTIFF.",
    )
    parser.add_argument(
        "--esri",
        type=Path,
        default=Path("outputs/landuse/Khajura_esri_lulc.tif"),
        help="Path to ESRI LULC GeoTIFF.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/landuse"),
        help="Output directory for CSV summaries.",
    )
    parser.add_argument(
        "--pixel-area-ha",
        type=float,
        default=0.01,
        help="Pixel area in hectares (default: 0.01 for 10m pixels).",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Generate figure outputs (heatmaps and bar chart).",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("outputs/landuse/figures"),
        help="Output directory for figures.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    figure_dir = args.figure_dir if args.figures else None
    compare_lulc(args.model, args.esri, args.output_dir, args.pixel_area_ha, figure_dir)


if __name__ == "__main__":
    main()
