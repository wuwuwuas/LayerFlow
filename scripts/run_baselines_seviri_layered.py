"""Run pressure-layerwise PIV / optical-flow trackers.

This script only runs tracking. It reads four original SEVIRI frames and four
independent original OCA pressure grids, builds pressure-layer images, runs
normal tracker functions on each layer, merges layer flows, then saves the usual
PIV-native NPZ.

It does not perform OCA pressure assignment, QC, ERA5 comparison, IGRA
comparison, or TTO/refinement.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
CASE_NAME = os.environ.get("AMV_CASE", "test_2018-01-12T11")
DATA_CASE_DIR = PROJECT_ROOT / "data" / CASE_NAME.removeprefix("test_")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / CASE_NAME
TRACKER_ROOT_DIR = OUTPUT_DIR / "trackers"

FRAME_INTERVAL_MINUTES = 15
MIN_VALID_COMPONENTS = 2
RUN_TRACKERS = True
SAVE_NATIVE_NPZ = True
SAVE_COMPONENT_NPZ = False

# Pressure pairing used to construct the two layer images for each component:
#   independent   : C1 P0/P1, C2 P1/P2, C3 P2/P3
#   source_shared : C1 P0/P0, C2 P1/P1, C3 P2/P2
#   target_shared : C1 P1/P1, C2 P2/P2, C3 P3/P3
#
# The source-shared mode explicitly reproduces the earlier source-pressure
# behavior without pretending that P3 is missing. All four original OCA frames
# are still loaded and validated.
PRESSURE_PAIR_MODE = "independent"  # Pressure-conditioned baseline.

# Non-independent modes receive an output-name suffix so existing independent
# products are never overwritten.
PRESSURE_PAIR_TRACKER_SUFFIX = {
    "independent": "",
    "source_shared": "_source_pressure",
    "target_shared": "_target_pressure",
}

# Backend switches for self-written replacements.
# backend = implementation source; device = where torch-capable implementations run.

TRACKER_COMPONENT_PAIRS = [(0, 1), (1, 2), (2, 3)]
# TRACKER_COMPONENT_PAIRS = [(0, 1)]  # quick test

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import cv2

    cv2.setUseOptimized(True)
    cv2.setNumThreads(max(1, (os.cpu_count() or 4) - 50))
    print("OpenCV threads:", cv2.getNumThreads())
except Exception as exc:
    print("OpenCV thread setup skipped:", repr(exc))

from data_io.case_observations import load_case_observations
from preprocessing.layerwise_image import LayerwiseImageConfig, PressureLayer
from tracking import run_deepflow, run_dis, run_dualtvl1, run_farneback, run_ilk, run_pcaflow, run_tvl1
from pipeline.tracker_runner_layerwise import (
    PRESSURE_PAIR_MODES,
    run_requested_layerwise_trackers,
)


def output_tracker_name(base_name: str) -> str:
    """Return a non-overwriting tracker name for the selected pressure mode."""

    mode = str(PRESSURE_PAIR_MODE).strip().lower()
    if mode not in PRESSURE_PAIR_MODES:
        raise ValueError(
            f"PRESSURE_PAIR_MODE must be one of {PRESSURE_PAIR_MODES}, "
            f"got {PRESSURE_PAIR_MODE!r}"
        )
    return f"{base_name}{PRESSURE_PAIR_TRACKER_SUFFIX[mode]}"


# -----------------------------------------------------------------------------
# Load original SEVIRI and four independent OCA frames
# -----------------------------------------------------------------------------

from runtime_options import configure_retrieval, save_case_inputs
FRAME_TIMES = None
AOI = None
configure_retrieval(globals(), sys.argv[1:], sensor="seviri")

observations = load_case_observations(
    case_name=CASE_NAME,
    data_case_dir=DATA_CASE_DIR,
    frame_count=4,
    frame_times=FRAME_TIMES,
    aoi=AOI,
    interval_minutes=FRAME_INTERVAL_MINUTES,
    channel="IR10.8",
    load_oca_data=True,
)
if observations.frame_pressure_hpa is None:
    raise RuntimeError("Original OCA pressure frames P0..P3 were not loaded")

TRACKER_ROOT_DIR.mkdir(parents=True, exist_ok=True)
images_stack = observations.images
valid_masks_stack = observations.valid_masks
pressure_stack = observations.frame_pressure_hpa
images = [images_stack[i] for i in range(images_stack.shape[0])]
valid_masks = [valid_masks_stack[i] for i in range(valid_masks_stack.shape[0])]
frame_pressure_hpa = [pressure_stack[i] for i in range(pressure_stack.shape[0])]
lon_grid = observations.lon_grid
lat_grid = observations.lat_grid
frame_times_loaded = list(observations.frame_times)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("CASE_NAME:", CASE_NAME)
print("DATA_CASE_DIR:", DATA_CASE_DIR)
print("Image stack:", images_stack.shape, images_stack.dtype)
print("Valid masks:", valid_masks_stack.shape, valid_masks_stack.dtype)
print("Pressure stack: original OCA P0..P3", pressure_stack.shape, pressure_stack.dtype)
print("Lon/lat:", lon_grid.shape, lat_grid.shape)
print("Frame times:", frame_times_loaded)
print("Original frame mapping:")
for index, (image_path, pressure_path) in enumerate(
    zip(observations.seviri_paths, observations.oca_paths)
):
    print(f"  I{index}: {image_path}")
    print(f"  P{index}: {pressure_path}")

print("Pressure pair mode:", PRESSURE_PAIR_MODE)
if PRESSURE_PAIR_MODE == "independent":
    print("  C1: I0/P0 -> I1/P1")
    print("  C2: I1/P1 -> I2/P2")
    print("  C3: I2/P2 -> I3/P3")
elif PRESSURE_PAIR_MODE == "source_shared":
    print("  C1: I0/P0 -> I1/P0")
    print("  C2: I1/P1 -> I2/P1")
    print("  C3: I2/P2 -> I3/P2")
elif PRESSURE_PAIR_MODE == "target_shared":
    print("  C1: I0/P1 -> I1/P1")
    print("  C2: I1/P2 -> I2/P2")
    print("  C3: I2/P3 -> I3/P3")
else:
    raise ValueError(
        f"PRESSURE_PAIR_MODE must be one of {PRESSURE_PAIR_MODES}, "
        f"got {PRESSURE_PAIR_MODE!r}"
    )

# -----------------------------------------------------------------------------
# Layer config
# -----------------------------------------------------------------------------

LAYER_CONFIG = LayerwiseImageConfig(
    mode="hard_soft_boundary",          # "hard", "soft", or "hard_soft_boundary"
    soft_sigma_hpa=50.0,
    soft_min_weight=0.01,
    background_sigma_pixels=35.0,
    output_erode_pixels=0,
    min_weight_sum=0.01,
    require_pressure_in_both_frames=False,
    hard_fallback_mode="nearest_valid_layer",
    layers=(
    PressureLayer("high_050_300", 50.0, 300.0),
    PressureLayer("upper_mid_300_500", 300.0, 500.0),
    PressureLayer("lower_mid_500_700", 500.0, 700.0),
    PressureLayer("low_700_1100", 700.0, 1100.0),
),
)
# ISCCP 3-layer
    # layers=(
    #     PressureLayer("high_050_440", 50.0, 440.0),
    #     PressureLayer("middle_440_680", 440.0, 680.0),
    #     PressureLayer("low_680_1100", 680.0, 1100.0),
    # )

# layers=(
#     PressureLayer("high_050_300", 50.0, 300.0),
#     PressureLayer("upper_mid_300_500", 300.0, 500.0),
#     PressureLayer("lower_mid_500_700", 500.0, 700.0),
#     PressureLayer("low_700_1100", 700.0, 1100.0),
# ),

# -----------------------------------------------------------------------------
# Tracker definitions: one best/default parameter set per algorithm
# -----------------------------------------------------------------------------


FARNEBACK_PARAMS_OPENCV = {
    "backend": "opencv",
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 25,
    "iterations": 3,
    "poly_n": 7,
    "poly_sigma": 1.1,
    "flags": 0,
}
DEEPFLOW_PARAMS_OPENCV = {
    "backend": "opencv",
    "alpha": 1.0, #
    "delta": 0.5,
    "gamma": 5.0,
    "sigma": 0.6,
    "min_size": 25,
    "downscale_factor": 0.95,
    "fixed_point_iterations": 5,
    "sor_iterations": 25,
    "omega": 1.6,
    "input_dtype": "uint8_0_255",
    "verbose": False,
}

DIS_PARAMS = {"preset": "medium"}
TVL1_PARAMS = {"attachment": 15.0, "tightness": 0.3, "num_warp": 5, "num_iter": 10, "tol": 1e-4, "prefilter": False}
DUALTVL1_PARAMS = {"lambda": 0.10, "theta": 0.30, "nscales": 5, "warps": 5, "epsilon": 0.01, "medianFiltering": 5}
ILK_PARAMS = {"radius": 11, "num_warp": 5, "gaussian": True, "prefilter": True}
PCAFLOW_PARAMS = {}


TRACKER_CONFIGS = {output_tracker_name('layerwise_farneback_opencv'): (run_farneback, FARNEBACK_PARAMS_OPENCV), output_tracker_name('layerwise_deepflow_opencv'): (run_deepflow, DEEPFLOW_PARAMS_OPENCV), output_tracker_name('layerwise_dis'): (run_dis, DIS_PARAMS), output_tracker_name('layerwise_tvl1'): (run_tvl1, TVL1_PARAMS), output_tracker_name('layerwise_dualtvl1'): (run_dualtvl1, DUALTVL1_PARAMS), output_tracker_name('layerwise_ilk'): (run_ilk, ILK_PARAMS), output_tracker_name('layerwise_pcaflow'): (run_pcaflow, PCAFLOW_PARAMS)}

# # Keep this short for first tests. NN layerwise costs roughly num_layers times raw NN.
TRACKERS_TO_RUN = [output_tracker_name('layerwise_farneback_opencv'), output_tracker_name('layerwise_deepflow_opencv'), output_tracker_name('layerwise_dis'), output_tracker_name('layerwise_tvl1'), output_tracker_name('layerwise_dualtvl1'), output_tracker_name('layerwise_ilk'), output_tracker_name('layerwise_pcaflow')]


# TRACKER_CONFIGS = {
# }

# Keep this short for first tests. NN layerwise costs roughly num_layers times raw NN.
# TRACKERS_TO_RUN = [
# ]

print("Available layerwise trackers:", list(TRACKER_CONFIGS))
print("Layerwise trackers requested:", TRACKERS_TO_RUN)
print("Layer config:", LAYER_CONFIG.metadata())

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

if RUN_TRACKERS:
    results = run_requested_layerwise_trackers(
        tracker_configs=TRACKER_CONFIGS,
        trackers_to_run=TRACKERS_TO_RUN,
        images=images,
        valid_masks=valid_masks,
        frame_pressure_hpa=frame_pressure_hpa,
        lon_grid=lon_grid,
        lat_grid=lat_grid,
        frame_times_loaded=frame_times_loaded,
        frame_interval_minutes=FRAME_INTERVAL_MINUTES,
        tracker_root_dir=TRACKER_ROOT_DIR,
        component_pairs=TRACKER_COMPONENT_PAIRS,
        layer_config=LAYER_CONFIG,
        pressure_pair_mode=PRESSURE_PAIR_MODE,
        min_valid_components=MIN_VALID_COMPONENTS,
        save_native_npz=SAVE_NATIVE_NPZ,
        save_component_npz=SAVE_COMPONENT_NPZ,
    )
    print("Finished requested layerwise trackers:", list(results))
else:
    print("RUN_TRACKERS is False. No tracker was run.")

print("Tracker output root:", TRACKER_ROOT_DIR)
for path in sorted(TRACKER_ROOT_DIR.rglob("*_final_native.npz")):
    print(path)

save_case_inputs(globals(), sensor="seviri")
