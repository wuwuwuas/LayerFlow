"""Run raw-image PIV / optical-flow trackers.

This entry point runs traditional optical-flow baselines. It only runs tracking and
saves PIV-native NPZ files. It does not perform OCA pressure assignment, QC,
ERA5 comparison, IGRA comparison, or TTO/refinement.
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
    cv2.setNumThreads(max(1, (os.cpu_count() or 4) - 1))
    print("OpenCV threads:", cv2.getNumThreads())
except Exception as exc:
    print("OpenCV thread setup skipped:", repr(exc))

from tracking import run_deepflow, run_dis, run_dualtvl1, run_farneback, run_ilk, run_pcaflow, run_tvl1
from data_io.case_observations import load_case_observations
from pipeline.tracker_runner import run_requested_trackers


# -----------------------------------------------------------------------------
# Load original SEVIRI observations
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
    load_oca_data=False,
)

TRACKER_ROOT_DIR.mkdir(parents=True, exist_ok=True)
images_stack = observations.images
valid_masks_stack = observations.valid_masks
images = [images_stack[i] for i in range(images_stack.shape[0])]
valid_masks = [valid_masks_stack[i] for i in range(valid_masks_stack.shape[0])]
lon_grid = observations.lon_grid
lat_grid = observations.lat_grid
frame_times_loaded = list(observations.frame_times)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("CASE_NAME:", CASE_NAME)
print("DATA_CASE_DIR:", DATA_CASE_DIR)
print("Image stack:", images_stack.shape, images_stack.dtype)
print("Valid masks:", valid_masks_stack.shape, valid_masks_stack.dtype)
print("Lon/lat:", lon_grid.shape, lat_grid.shape)
print("Frame times:", frame_times_loaded)
print("Original SEVIRI files:")
for index, path in enumerate(observations.seviri_paths):
    print(f"  I{index}: {path}")

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
    "alpha": 1.0,
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


TRACKER_CONFIGS = {'raw_farneback_opencv': (run_farneback, FARNEBACK_PARAMS_OPENCV), 'raw_deepflow_opencv': (run_deepflow, DEEPFLOW_PARAMS_OPENCV), 'raw_dis': (run_dis, DIS_PARAMS), 'raw_tvl1': (run_tvl1, TVL1_PARAMS), 'raw_dualtvl1': (run_dualtvl1, DUALTVL1_PARAMS), 'raw_ilk': (run_ilk, ILK_PARAMS), 'raw_pcaflow': (run_pcaflow, PCAFLOW_PARAMS)}

# Keep this short for first tests. Uncomment more after raw_farneback/raw_deepflow run correctly.
TRACKERS_TO_RUN = ['raw_farneback_opencv', 'raw_deepflow_opencv', 'raw_dis', 'raw_tvl1', 'raw_dualtvl1', 'raw_ilk', 'raw_pcaflow']

print("Available trackers:", list(TRACKER_CONFIGS))
print("Trackers requested:", TRACKERS_TO_RUN)

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

if RUN_TRACKERS:
    results = run_requested_trackers(
        tracker_configs=TRACKER_CONFIGS,
        trackers_to_run=TRACKERS_TO_RUN,
        images=images,
        valid_masks=valid_masks,
        lon_grid=lon_grid,
        lat_grid=lat_grid,
        frame_times_loaded=frame_times_loaded,
        frame_interval_minutes=FRAME_INTERVAL_MINUTES,
        tracker_root_dir=TRACKER_ROOT_DIR,
        component_pairs=TRACKER_COMPONENT_PAIRS,
        # New tracking adapters normalize internally. Keep this None to avoid
        # double normalization and to avoid tracker-name dtype assumptions.
        normalize_image_pair=None,
        min_valid_components=MIN_VALID_COMPONENTS,
        save_native_npz=SAVE_NATIVE_NPZ,
        save_component_npz=SAVE_COMPONENT_NPZ,
    )
    print("Finished requested trackers:", list(results))
else:
    print("RUN_TRACKERS is False. No tracker was run.")

print("Tracker output root:", TRACKER_ROOT_DIR)
for path in sorted(TRACKER_ROOT_DIR.rglob("*_final_native.npz")):
    print(path)

save_case_inputs(globals(), sensor="seviri")
