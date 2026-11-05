"""Evaluate GOES traditional optical-flow products using existing PCF caches.

No ERA5 interpolation is performed.  For each satellite/geometry combination,
all traditional trackers reuse the ERA5 collocation cache previously written
by ``scripts/evaluate_era5_layerflow_goes.py``.
"""

from __future__ import annotations

import os

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASE_NAME = os.environ.get("AMV_CASE", "goes_g16_g18_E20240804T120506_W20240804T120508")
SATELLITES = ("G16", "G18")
GEOMETRY_MODES = ("cloud_height",)
SKIP_MISSING_PRODUCTS = True

# Names exclude the satellite suffix; the evaluator appends _g16/_g18.
# TRACKER_BASE_NAMES = (
#     "raw_farneback_opencv",
#     "raw_dis",
#     "raw_deepflow_opencv",
#     "raw_dualtvl1",
#     "layerwise_farneback_opencv",
#     "layerwise_dis",
#     "layerwise_deepflow_opencv",
#     "layerwise_dualtvl1",
# )
# Same method set as the SEVIRI Table I (Original / H/P-conditioned rows),
# FlowDiffuser, and the layer-constrained PCF row.  All products share the same
# GOES centre output mask and reuse the PCF ERA5 cache.
TRACKER_BASE_NAMES = (
    # Original images
    # "raw_farneback_opencv",
    # "raw_deepflow_opencv",
    # "raw_dis",
    # "raw_tvl1",
    # "raw_dualtvl1",
    # "raw_ilk",
    # "raw_pcaflow",
    "nn_raw_flowdiffuser",
    "nn_raw_windflow_inv",
    # Pressure-conditioned images
    # "layerwise_farneback_opencv",
    # "layerwise_deepflow_opencv",
    # "layerwise_dis",
    # "layerwise_tvl1",
    # "layerwise_dualtvl1",
    "layerwise_ilk",
    "layerwise_pcaflow",
    "nn_layerwise_flowdiffuser",
    "nn_layerwise_windflow_inv",
    # Layer-constrained optical flow (PCF)
    # "pressure_conditioned_flow",
)
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from validation.compare_era5_goes import compare_goes_amv_to_era5  # noqa: E402
from validation.goes_existing_era5_cache import (  # noqa: E402
    build_amv_vs_existing_goes_era5_cache,
    resolve_existing_pcf_cache,
)


def main() -> int:
    output_dir = PROJECT_ROOT / "outputs" / CASE_NAME / "goes_traditional_vs_era5"
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_tables: list[pd.DataFrame] = []
    attrition_rows: list[dict[str, Any]] = []
    reuse_rows: list[dict[str, Any]] = []

    print("=" * 120)
    print("GOES TRADITIONAL OPTICAL FLOW VS ERA5 — REUSE EXISTING PCF CACHE")
    print("=" * 120)
    print("case:", CASE_NAME)
    print("satellites:", SATELLITES)
    print("geometry modes:", GEOMETRY_MODES)
    print("ERA5 interpolation rerun: no")
    print("new ERA5 cache files: no")

    shared_caches: dict[tuple[str, str], Path] = {}
    for satellite in SATELLITES:
        for geometry_mode in GEOMETRY_MODES:
            cache = resolve_existing_pcf_cache(
                PROJECT_ROOT,
                CASE_NAME,
                satellite,
                geometry_mode,
            )
            shared_caches[(satellite, geometry_mode)] = cache
            print(f"shared cache [{satellite} | {geometry_mode}]: {cache}")

    for satellite in SATELLITES:
        for base_name in TRACKER_BASE_NAMES:
            tracker = f"{base_name}_{satellite.lower()}"
            for geometry_mode in GEOMETRY_MODES:
                product = (
                    PROJECT_ROOT
                    / "outputs"
                    / CASE_NAME
                    / "pressure_assignment"
                    / f"{tracker}_{geometry_mode}_final_amv_goes_pressure.npz"
                )
                if not product.is_file():
                    message = f"Missing product: {product}"
                    if SKIP_MISSING_PRODUCTS:
                        print(message, "-- skip")
                        continue
                    raise FileNotFoundError(message)

                cache = shared_caches[(satellite, geometry_mode)]
                print(f"\n[{satellite} | {tracker} | {geometry_mode}]")
                print("product:", product)
                print("reuse cache:", cache)
                try:
                    table, reuse_info = build_amv_vs_existing_goes_era5_cache(
                        product,
                        cache,
                        amv_source=f"{tracker}_{geometry_mode}",
                    )
                except ValueError as exc:
                    # e.g. product pixels absent from the PCF cache.  Report and
                    # continue so one product does not stop the whole table.
                    print(f"[{tracker} | {geometry_mode}] cache reuse failed: {exc} -- skip")
                    continue
                print("cache alignment:", reuse_info["alignment"])
                print(
                    "geometry max differences: "
                    f"lon={reuse_info['max_abs_longitude_difference_deg']:.3g} deg, "
                    f"lat={reuse_info['max_abs_latitude_difference_deg']:.3g} deg, "
                    f"pressure={reuse_info['max_abs_pressure_difference_hpa']:.3g} hPa"
                )

                collocated_count = int(table["era5_valid_mask"].sum())
                attrition_rows.append(
                    {
                        "tracker": tracker,
                        "satellite": satellite,
                        "geometry_mode": geometry_mode,
                        "pressure_product_points": int(len(table)),
                        "era5_collocated": collocated_count,
                        "era5_collocation_fraction": (
                            collocated_count / len(table) if len(table) else np.nan
                        ),
                    }
                )
                reuse_rows.append(
                    {
                        "tracker": tracker,
                        "satellite": satellite,
                        "geometry_mode": geometry_mode,
                        **reuse_info,
                    }
                )
                metrics = compare_goes_amv_to_era5(
                    table,
                    comparison="goes_traditional_vs_era5",
                )
                metrics.insert(0, "geometry_mode", geometry_mode)
                metrics.insert(0, "satellite", satellite)
                metrics.insert(0, "tracker", tracker)
                metric_tables.append(metrics)

    if not metric_tables:
        print("No traditional GOES products were found; nothing to evaluate.")
        return 1

    metrics_all = pd.concat(metric_tables, ignore_index=True)
    attrition = pd.DataFrame(attrition_rows)
    reuse = pd.DataFrame(reuse_rows)
    metrics_path = output_dir / "goes_traditional_vs_era5_metrics.csv"
    attrition_path = output_dir / "goes_traditional_vs_era5_attrition.csv"
    reuse_path = output_dir / "goes_traditional_era5_cache_reuse.csv"
    metrics_all.to_csv(metrics_path, index=False)
    attrition.to_csv(attrition_path, index=False)
    reuse.to_csv(reuse_path, index=False)

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
    print("\n" + "=" * 120)
    print("METRICS")
    print("=" * 120)
    available = [name for name in columns if name in metrics_all.columns]
    print(metrics_all[available].to_string(index=False))
    print("\nAttrition:")
    print(attrition.to_string(index=False))
    print("\nCache reuse:")
    print(
        reuse[
            [
                "tracker",
                "satellite",
                "geometry_mode",
                "alignment",
                "product_points",
                "cache_points",
                "era5_interpolation_rerun",
            ]
        ].to_string(index=False)
    )
    print("\nmetrics:", metrics_path)
    print("cache reuse report:", reuse_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
