#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the validated pressure-conditioned dense AMV baseline.

This example performs both stages required by downstream processing:

1. saves the standard dense PIV-native product;
2. saves the standard compact OCA-pressure AMV product.

QC, ERA5 comparison and IGRA comparison are intentionally left to the existing
common scripts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASE_NAME = os.environ.get("AMV_CASE", "test_2018-01-12T11")
TRACKER_NAME = "pressure_conditioned_flow"

# Formal runtime input: four original SEVIRI frames and four independent
# original OCA products. No tracking-input bundle is read.
DATA_CASE_DIR = PROJECT_ROOT / "data" / CASE_NAME.removeprefix("test_")
OCA_TOLERANCE_MINUTES = 7.5

OUTPUT_DIR = PROJECT_ROOT / "outputs" / CASE_NAME
TRACKER_ROOT_DIR = OUTPUT_DIR / "trackers"
PRESSURE_OUTPUT_DIR = OUTPUT_DIR / "pressure_assignment"

FRAME_INTERVAL_MINUTES = 15.0
FRAME_TIMES = None
AOI = None
MIN_VALID_COMPONENTS = 2
MIN_VALID_PRESSURE_COMPONENTS = 2

SAVE_NATIVE_NPZ = True
SAVE_COMPONENT_NPZ = False
SAVE_OCA_PRESSURE_PRODUCT = True
INCLUDE_STANDARD_ATMOSPHERE_HEIGHT = False

# None lets the native loader search repository defaults and the legacy working
# installation. An explicit example is:
# PRIVATE_OPENCV_INSTALL_DIR = (
#     PROJECT_ROOT
#     / "_third_party/opencv_install_baseline"
# )
PRIVATE_OPENCV_INSTALL_DIR = None

# None retains the private OpenCV default thread count.
PRIVATE_OPENCV_THREADS = None

COMPONENT_PAIRS = ((0, 1), (1, 2), (2, 3))


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

cv2.setUseOptimized(True)
cv2.setNumThreads(max(1, (os.cpu_count() or 4) - 1))

from data_io.case_observations import load_case_observations  # noqa: E402
from pipeline.pressure_conditioned_flow_runner import (  # noqa: E402
    run_pressure_conditioned_flow_case,
)
from tracking.pressure_conditioned_flow import (  # noqa: E402
    PressureConditionedFlowConfig,
    default_pressure_layer_config,
)


# -----------------------------------------------------------------------------
# Exact validated configuration
# -----------------------------------------------------------------------------

LAYER_CONFIG = default_pressure_layer_config()

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
    verbose=True,
    print_every_levels=10,
    collect_level_diagnostics=False,
)


def main(argv: list[str] | None = None) -> None:
    if argv is not None:
        from runtime_options import configure_retrieval
        configure_retrieval(globals(), argv, sensor="seviri")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("CASE_NAME:", CASE_NAME)
    print("TRACKER_NAME:", TRACKER_NAME)
    print("DATA_CASE_DIR:", DATA_CASE_DIR)
    print("OpenCV Python threads:", cv2.getNumThreads())

    observations = load_case_observations(
        case_name=CASE_NAME,
        data_case_dir=DATA_CASE_DIR,
        frame_count=4,
        frame_times=FRAME_TIMES,
        interval_minutes=FRAME_INTERVAL_MINUTES,
        aoi=AOI,
        channel="IR10.8",
        load_oca_data=True,
        oca_time_tolerance_minutes=OCA_TOLERANCE_MINUTES,
    )
    if observations.frame_pressure_hpa is None:
        raise RuntimeError("Original OCA pressure frames P0..P3 were not loaded")

    images = observations.images
    valid_masks = observations.valid_masks
    pressure = observations.frame_pressure_hpa
    cloud = observations.frame_cloud_probability

    print("Images:", images.shape, images.dtype)
    print("Valid masks:", valid_masks.shape, valid_masks.dtype)
    print("Pressure: original OCA P0..P3", pressure.shape, pressure.dtype)
    print("Cloud probability:", None if cloud is None else cloud.shape)
    print("Lon/lat:", observations.lon_grid.shape, observations.lat_grid.shape)
    print("Frame times:", observations.frame_times)
    print("Original frame mapping:")
    for index, (image_path, pressure_path) in enumerate(
        zip(observations.seviri_paths, observations.oca_paths)
    ):
        print(f"  I{index}: {image_path}")
        print(f"  P{index}: {pressure_path}")
    print("Pressure mapping:")
    print("  C1: I0/P0 -> I1/P1; final pressure component P0")
    print("  C2: I1/P1 -> I2/P2; final pressure component P1")
    print("  C3: I2/P2 -> I3/P3; final pressure component P2")

    result = run_pressure_conditioned_flow_case(
        images=images,
        valid_masks=valid_masks,
        pressure_hpa=pressure,
        cloud_probability=cloud,
        lon_grid=observations.lon_grid,
        lat_grid=observations.lat_grid,
        frame_times=observations.frame_times,
        tracker_root_dir=TRACKER_ROOT_DIR,
        pressure_output_dir=PRESSURE_OUTPUT_DIR,
        tracker_name=TRACKER_NAME,
        component_pairs=COMPONENT_PAIRS,
        layer_config=LAYER_CONFIG,
        solver_config=SOLVER_CONFIG,
        frame_interval_minutes=FRAME_INTERVAL_MINUTES,
        min_valid_components=MIN_VALID_COMPONENTS,
        min_valid_pressure_components=MIN_VALID_PRESSURE_COMPONENTS,
        include_height_m=INCLUDE_STANDARD_ATMOSPHERE_HEIGHT,
        save_native_npz=SAVE_NATIVE_NPZ,
        save_component_npz=SAVE_COMPONENT_NPZ,
        save_pressure_product=SAVE_OCA_PRESSURE_PRODUCT,
        pressure_component_source="original_oca_frame_start_grids",
    )

    print("\nDone.")
    from runtime_options import save_case_inputs
    save_case_inputs(globals(), sensor="seviri")
    print("Final native product:", result.get("final_path"))
    print("OCA-pressure AMV product:", result.get("pressure_path"))
    print("Runtime summary:", result.get("summary_path"))
    print(
        "Total runtime minutes:",
        float(result["runtime_seconds"]) / 60.0,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
