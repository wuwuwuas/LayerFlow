"""Evaluate GOES LayerFlow pressure-assigned AMVs against ERA5.

Report satellite viewpoints and geometry variants separately."""

from __future__ import annotations

import os
import argparse

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASE_NAME = os.environ.get("AMV_CASE", "goes_g16_g18_E20240804T120506_W20240804T120508")
SATELLITES = ("G16", "G18")
GEOMETRY_MODES = ("surface", "cloud_height")
TRACKER_TEMPLATE = "pressure_conditioned_flow_{satellite_lower}"
CHUNK_SIZE = 50_000
SAVE_ERA5_CACHE_NPZ = True

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from validation.compare_era5 import save_era5_cache_npz  # noqa: E402
from runtime_options import read_case_inputs  # noqa: E402
from validation.compare_era5_goes import (  # noqa: E402
    build_goes_amv_vs_era5_table,
    compare_goes_amv_to_era5,
    discover_goes_era5_regions,
)


def _scalar_text(data: Any, key: str) -> str:
    if key not in data.files:
        return ""
    values = np.asarray(data[key])
    return str(values.ravel()[0]) if values.size else ""


def _product_time(path: Path) -> pd.Timestamp:
    with np.load(path, allow_pickle=False) as data:
        text = _scalar_text(data, "observation_time")
        if text:
            return pd.Timestamp(text)
        if "metadata_json" in data.files:
            document = json.loads(_scalar_text(data, "metadata_json") or "{}")
            value = document.get("centre_scan_midpoint")
            if value:
                return pd.Timestamp(value)
    raise RuntimeError(f"Could not resolve observation time from {path}")


def evaluation_options(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=CASE_NAME, help="Retrieval output case name")
    parser.add_argument("--data-dir", type=Path, help="Observation directory containing era5/<region>/")
    parser.add_argument("--satellites", nargs="+", choices=["G16", "G18"], help="Satellites to evaluate; defaults to the retrieval selection")
    parser.add_argument("--geometry-modes", nargs="+", choices=["surface", "cloud_height"], help="Geometry products to evaluate")
    args = parser.parse_args(argv)
    if not args.case or Path(args.case).name != args.case or any(c in args.case for c in '/\\') or args.case in {'.', '..'}:
        parser.error("--case must be a single folder name")
    record = read_case_inputs(PROJECT_ROOT, args.case)
    args.data_dir = (args.data_dir or Path(record.get("data_dir", PROJECT_ROOT / "data" / args.case))).expanduser().resolve()
    product_dir = PROJECT_ROOT / "outputs" / args.case / "pressure_assignment"
    available = [sat for sat in SATELLITES if any(product_dir.glob(f"pressure_conditioned_flow_{sat.lower()}_*_final_amv_goes_pressure.npz"))]
    args.satellites = args.satellites or record.get("satellites") or available or list(SATELLITES)
    args.geometry_modes = args.geometry_modes or record.get("geometry_modes") or list(GEOMETRY_MODES)
    return args


def main(argv: list[str] | None = None) -> int:
    args = evaluation_options(argv)
    case_dir = args.data_dir
    output_dir = PROJECT_ROOT / "outputs" / args.case / "goes_pcf_vs_era5"
    output_dir.mkdir(parents=True, exist_ok=True)
    regions = discover_goes_era5_regions(case_dir)
    metric_tables: list[pd.DataFrame] = []
    attrition_rows: list[dict[str, Any]] = []

    print("=" * 112)
    print("GOES PRESSURE-CONDITIONED FLOW VS ERA5")
    print("=" * 112)
    print("case:", args.case)
    print("satellites:", args.satellites)
    print("geometry modes:", args.geometry_modes)
    print("ERA5 evaluation pressure range is defined by downloaded levels")
    print("ERA5 regions:")
    for region in regions:
        print(" ", region.name, region.path)

    for satellite in args.satellites:
        tracker = TRACKER_TEMPLATE.format(satellite_lower=satellite.lower())
        for geometry_mode in args.geometry_modes:
            product = (
                PROJECT_ROOT
                / "outputs"
                / args.case
                / "pressure_assignment"
                / f"{tracker}_{geometry_mode}_final_amv_goes_pressure.npz"
            )
            if not product.is_file():
                raise FileNotFoundError(
                    f"GOES PCF pressure product not found: {product}"
                )
            valid_time = _product_time(product)
            print(f"\n[{satellite} | {geometry_mode}] product: {product}")
            print(
                f"[{satellite} | {geometry_mode}] ERA5 valid time: {valid_time}"
            )
            table = build_goes_amv_vs_era5_table(
                product,
                regions,
                valid_time=valid_time,
                chunk_size=CHUNK_SIZE,
                progress=True,
            )
            collocated_count = int(table["era5_valid_mask"].sum())
            attrition_rows.append(
                {
                    "satellite": satellite,
                    "geometry_mode": geometry_mode,
                    "pressure_product_points": int(len(table)),
                    "era5_collocated": collocated_count,
                    "era5_collocation_fraction": (
                        collocated_count / len(table) if len(table) else np.nan
                    ),
                }
            )
            metrics = compare_goes_amv_to_era5(
                table,
                comparison="goes_pcf_vs_era5",
            )
            metrics.insert(0, "geometry_mode", geometry_mode)
            metrics.insert(0, "satellite", satellite)
            metrics.insert(0, "tracker", tracker)
            metric_tables.append(metrics)

            if SAVE_ERA5_CACHE_NPZ:
                cache_path = (
                    output_dir
                    / f"{tracker}_{geometry_mode}_era5_collocated.npz"
                )
                save_era5_cache_npz(
                    table,
                    cache_path,
                    pressure_source=(
                        "goes_abi_cloud_top_pressure_center_frame"
                    ),
                    amv_geometry_source=(
                        f"{satellite}_pcf_{geometry_mode}_geometry"
                    ),
                    metadata={
                        "satellite": satellite,
                        "tracker": tracker,
                        "geometry_mode": geometry_mode,
                        "valid_time": str(valid_time),
                    },
                    compress=True,
                )
                print(
                    f"[{satellite} | {geometry_mode}] saved ERA5 cache: "
                    f"{cache_path}"
                )

    metrics_all = pd.concat(metric_tables, ignore_index=True)
    attrition = pd.DataFrame(attrition_rows)
    metrics_path = output_dir / "goes_pcf_vs_era5_metrics.csv"
    attrition_path = output_dir / "goes_pcf_vs_era5_attrition.csv"
    metrics_all.to_csv(metrics_path, index=False)
    attrition.to_csv(attrition_path, index=False)

    columns = [
        "tracker",
        "satellite",
        "geometry_mode",
        "subset",
        "sample_count",
        "mean_vector_difference",
        "mean_absolute_speed_error",
        "vector_difference_rmse",
        "mean_absolute_direction_difference",
        "speed_correlation",
        "u_bias_amv_minus_era5",
        "v_bias_amv_minus_era5",
        "speed_bias_amv_minus_era5",
    ]
    print("\n" + "=" * 112)
    print("METRICS")
    print("=" * 112)
    available = [name for name in columns if name in metrics_all.columns]
    print(metrics_all[available].to_string(index=False))
    print("\nAttrition:")
    print(attrition.to_string(index=False))
    print("\nmetrics:", metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
