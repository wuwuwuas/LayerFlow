#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the validated PCF algorithm on GOES ABI C14.

For each satellite:

* C1: centre -> previous (signed dt < 0)
* C2: centre -> next     (signed dt > 0)

Both dense components are defined on the centre grid and both are required.
The same pixel flows are converted twice without rerunning PCF:

* surface geometry: nominal reference ellipsoid;
* cloud-height geometry: centre CTH parallax correction.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
import sys

import cv2


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASE_NAME = os.environ.get("AMV_CASE", "goes_g16_g18_E20240804T120506_W20240804T120508")
SATELLITES = ("G16", "G18")
TRACKER_TEMPLATE = "pressure_conditioned_flow_{satellite_lower}"

DATA_CASE_DIR = PROJECT_ROOT / "data" / CASE_NAME
OUTPUT_DIR = PROJECT_ROOT / "outputs" / CASE_NAME
TRACKER_ROOT_DIR = OUTPUT_DIR / "trackers"
PRESSURE_OUTPUT_DIR = OUTPUT_DIR / "pressure_assignment"

GEOMETRY_MODES = ("surface", "cloud_height")
GEOMETRY_CHUNK_ROWS = 256

SAVE_NATIVE_NPZ = True
SAVE_COMPONENT_NPZ = False
SAVE_GOES_PRESSURE_PRODUCT = True
SAVE_DIAGNOSTIC_FIGURES = False

PRIVATE_OPENCV_INSTALL_DIR = None
PRIVATE_OPENCV_THREADS = None

# Exact source-pyramid reuse only. If pair-wise normalization makes the centre
# source image differ between backward and forward components, the runner
# preserves the established numerical definition and rebuilds that source.
SOURCE_PYRAMID_REUSE_POLICY = "exact"


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

cv2.setUseOptimized(True)
cv2.setNumThreads(max(1, (os.cpu_count() or 4) - 1))

from data_io.goes import discover_goes_frames, frame_for, load_goes_frame  # noqa: E402
from pipeline.goes_pressure_conditioned_flow_runner import (  # noqa: E402
    run_goes_pressure_conditioned_flow_satellite,
)
from preprocessing.goes import GoesPcfMaskConfig  # noqa: E402
from preprocessing.layerwise_image import (  # noqa: E402
    LayerwiseImageConfig,
    PressureLayer,
)
from tracking.pressure_conditioned_flow import (  # noqa: E402
    PressureConditionedFlowConfig,
)


# -----------------------------------------------------------------------------
# Validated PCF configuration retained from the Ubuntu SEVIRI baseline
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Pressure-conditioned states: experiment configuration belongs in this script.
# The tracking implementation receives this object and contains no GOES-specific
# pressure-boundary choice.
# -----------------------------------------------------------------------------

PRESSURE_LAYERS = (
    PressureLayer("high_050_300", 50.0, 300.0),
    PressureLayer("upper_mid_300_500", 300.0, 500.0),
    PressureLayer("lower_mid_500_700", 500.0, 700.0),
    PressureLayer("low_700_1100", 700.0, 1100.0),
)

LAYER_CONFIG = LayerwiseImageConfig(
    layers=PRESSURE_LAYERS,
    mode="hard_soft_boundary",
    soft_sigma_hpa=50.0,
    soft_min_weight=0.01,
    background_sigma_pixels=35.0,
    output_erode_pixels=0,
    min_weight_sum=0.01,
    require_pressure_in_both_frames=False,
    hard_fallback_mode="nearest_valid_layer",
)

SOLVER_CONFIG = PressureConditionedFlowConfig(
    alpha=1.0,
    delta=0.5,
    gamma=5.0,
    sigma=0.6,
    min_size=25,
    downscale_factor=0.95,
    fixed_point_iterations=5,
    sor_iterations=25,
    omega=1.6,
    max_layers=200,
    max_pyramid_levels=None,
    interpolation=cv2.INTER_LINEAR,
    input_dtype="uint8_0_255",
    solver_active_threshold=0.01,
    normalized_blur_epsilon=1.0e-4,
    target_invalid_weight=0.0,
    pyramid_storage_dtype="float32",
    opencv_num_threads=PRIVATE_OPENCV_THREADS,
    native_install_dir=PRIVATE_OPENCV_INSTALL_DIR,
    use_native_uv_interface=True,
    collect_native_profile=True,
    verbose=True,
    print_every_levels=10,
    collect_level_diagnostics=False,
)

MASK_CONFIG = GoesPcfMaskConfig(
    minimum_tracking_pressure_hpa=50.0,
    maximum_tracking_pressure_hpa=1100.0,
    minimum_cloud_mask_value=2,
    require_image_dqf_zero=True,
    require_pressure_dqf_zero=True,
    require_cloud_mask_dqf_zero=False,
    minimum_height_m=-500.0,
    maximum_height_m=25000.0,
    require_height_dqf_zero=True,
)


def _print_pressure_layers() -> None:
    print("Pressure-conditioned states:")
    for index, layer in enumerate(LAYER_CONFIG.layers):
        print(
            f"  {index}: {layer.name:22s} "
            f"[{layer.p_min_hpa:g}, {layer.p_max_hpa:g}) hPa"
        )
    print("  mode:", LAYER_CONFIG.mode)
    print("  soft sigma:", LAYER_CONFIG.soft_sigma_hpa, "hPa")
    print(
        "  require pressure in both frames:",
        LAYER_CONFIG.require_pressure_in_both_frames,
    )


def main(argv: list[str] | None = None) -> None:
    if argv is not None:
        from runtime_options import configure_retrieval
        configure_retrieval(globals(), argv, sensor="goes")
    print("=" * 112)
    print("GOES PRESSURE-CONDITIONED FLOW CONFIGURATION")
    print("=" * 112)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("CASE_NAME:", CASE_NAME)
    print("DATA_CASE_DIR:", DATA_CASE_DIR)
    print("SATELLITES:", SATELLITES)
    print("GEOMETRY_MODES:", GEOMETRY_MODES)
    print("OpenCV Python threads:", cv2.getNumThreads())
    print("Temporal components:")
    print("  C1: center/Pcenter -> previous/Pprevious; signed dt < 0")
    print("  C2: center/Pcenter -> next/Pnext; signed dt > 0")
    print("  both dense components are required")
    print("Output validity: centre I0 only; no destination-product hard filter")
    print("ERA5 evaluation range remains 100--1000 hPa downstream")
    print("Pressure-state execution: sequential build -> solve -> release")
    print("Source-pyramid reuse policy:", SOURCE_PYRAMID_REUSE_POLICY)
    _print_pressure_layers()

    frame_paths = discover_goes_frames(DATA_CASE_DIR)
    for satellite in SATELLITES:
        tracker_name = TRACKER_TEMPLATE.format(
            satellite_lower=satellite.lower()
        )
        print("\n" + "=" * 112)
        print(f"LOADING {satellite}")
        print("=" * 112)
        frames_by_role = {}
        for role in ("previous", "center", "next"):
            paths = frame_for(frame_paths, satellite, role)
            frame = load_goes_frame(paths, apply_quality_mask=False)
            frames_by_role[role] = frame
            print(
                f"{role:8s} midpoint={frame.scan_midpoint}; "
                f"shape={frame.brightness_temperature_k.shape}"
            )
            print("  image:", paths.image)
            print("  height:", paths.height)
            print("  pressure:", paths.pressure)
            print("  cloud mask:", paths.cloud_mask)

        result = run_goes_pressure_conditioned_flow_satellite(
            frames_by_role=frames_by_role,
            tracker_root_dir=TRACKER_ROOT_DIR,
            pressure_output_dir=PRESSURE_OUTPUT_DIR,
            tracker_name=tracker_name,
            layer_config=LAYER_CONFIG,
            solver_config=SOLVER_CONFIG,
            mask_config=MASK_CONFIG,
            geometry_modes=GEOMETRY_MODES,
            geometry_chunk_rows=GEOMETRY_CHUNK_ROWS,
            save_native_npz=SAVE_NATIVE_NPZ,
            save_component_npz=SAVE_COMPONENT_NPZ,
            save_pressure_product=SAVE_GOES_PRESSURE_PRODUCT,
            save_diagnostic_figures=SAVE_DIAGNOSTIC_FIGURES,
            source_reuse_policy=SOURCE_PYRAMID_REUSE_POLICY,
        )

        print("\nDone:", satellite)
        print("Final native products:", result.get("final_paths"))
        print("GOES-pressure AMV products:", result.get("pressure_paths"))
        print("Point counts:", result.get("pressure_point_counts"))
        print("Runtime summary:", result.get("summary_path"))
        print(
            "Total runtime minutes:",
            float(result["runtime_seconds"]) / 60.0,
        )
        del result, frames_by_role
        gc.collect()

    from runtime_options import save_case_inputs
    save_case_inputs(globals(), sensor="goes")


if __name__ == "__main__":
    main(sys.argv[1:])
