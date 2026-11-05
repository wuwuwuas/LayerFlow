"""Run FlowDiffuser on original (non-layered) GOES ABI C14 images.

GOES counterpart of ``scripts/run_neural_baselines_seviri.py``.  The model, checkpoint
and tiling settings are identical to the SEVIRI run; only the temporal
definition and wind conversion follow the GOES centre-anchored pipeline used
by all other GOES trackers:

* C1: centre -> previous, signed dt < 0
* C2: centre -> next, signed dt > 0
* final: vector mean of C1/C2, both required

Outputs follow the same naming as the GOES traditional trackers, so
``scripts/evaluate_era5_baselines_goes.py`` can evaluate them directly.
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

DATA_CASE_DIR = PROJECT_ROOT / "data" / CASE_NAME
OUTPUT_DIR = PROJECT_ROOT / "outputs" / CASE_NAME
TRACKER_ROOT_DIR = OUTPUT_DIR / "trackers"
PRESSURE_OUTPUT_DIR = OUTPUT_DIR / "pressure_assignment"

GEOMETRY_MODES = ("cloud_height",)
GEOMETRY_CHUNK_ROWS = 256
INPUT_BACKGROUND_SIGMA_PIXELS = 35.0

SAVE_NATIVE_NPZ = True
SAVE_COMPONENT_NPZ = False
SAVE_GOES_PRESSURE_PRODUCT = True
SAVE_DIAGNOSTIC_FIGURES = False
QUICKLOOK_MAXIMUM_DIMENSION = 1200
SKIP_EXISTING_FINAL = False

REQUESTED_DEVICE = os.environ.get("AMV_DEVICE", "auto")

# Same "nn_" prefix as SEVIRI; the satellite suffix is appended (_g16/_g18).
TRACKERS_TO_RUN = ['nn_raw_flowdiffuser', 'nn_raw_windflow_inv']


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

cv2.setUseOptimized(True)
cv2.setNumThreads(max(1, (os.cpu_count() or 4) - 1))

from data_io.goes import discover_goes_frames, frame_for, load_goes_frame  # noqa: E402
from pipeline.goes_traditional_tracker_runner import (  # noqa: E402
    run_goes_traditional_tracker_satellite,
)
from preprocessing.goes import GoesPcfMaskConfig  # noqa: E402
from tracking import run_flowdiffuser, run_windflow


def resolve_device(requested: str) -> str:
    import torch

    text = str(requested).strip().lower()
    if text == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if text.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested {text}, but CUDA is unavailable; using CPU.")
        return "cpu"
    return text


DEVICE = resolve_device(REQUESTED_DEVICE)


# -----------------------------------------------------------------------------
# FlowDiffuser configuration: identical to SEVIRI run_neural_baselines_seviri.py
# -----------------------------------------------------------------------------

FLOWDIFFUSER_PARAMS = {
    "repository": os.environ.get("AMV_FLOWDIFFUSER_REPO", str(PROJECT_ROOT / "external" / "FlowDiffuser")),
    "checkpoint": os.environ.get("AMV_FLOWDIFFUSER_CHECKPOINT", str(PROJECT_ROOT / "weights" / "FlowDiffuser-things.pth")),
    "tile_size": 512,
    "overlap": 128,
    "batch_size": 4,
    "seed": 0,
    "device": DEVICE,
    "mixed_precision": True,
    "strict": True,
    "min_tile_output_fraction": 0.0005,
    "max_tile_flow_px": 100.0,
}

# WindFlow (Vandal et al., KDD 2022): public RAFT checkpoint trained on GEOS-5
# Nature Run humidity.  Inference recipe follows the WindFlow repository
# (per-image histogram equalization, 512/384 tiles, Gaussian blending, 12 iters).
WINDFLOW_PARAMS = {
    "checkpoint": os.environ.get("AMV_WINDFLOW_CHECKPOINT", str(PROJECT_ROOT / "weights" / "windflow.raft.pth.tar")),
    "tile_size": 512,
    "overlap": 384,
    "batch_size": 8,
    "iterations": 12,
    "device": DEVICE,
    "mixed_precision": True,
    "polarity": "original",  # as in the WindFlow GOES pipeline
    "equalization_mode": "per_image",
    "equalization_bins": 500,
    "min_tile_output_fraction": 0.0005,
}

# Polarity matched to the humidity training images (cold/moist = bright).
WINDFLOW_INV_PARAMS = {**WINDFLOW_PARAMS, "polarity": "inverted"}
# Joint (training-style) equalization: one mapping for both frames.
WINDFLOW_INV_JOINT_PARAMS = {**WINDFLOW_INV_PARAMS, "equalization_mode": "joint_pair"}

TRACKER_CONFIGS = {'nn_raw_flowdiffuser': (run_flowdiffuser, FLOWDIFFUSER_PARAMS), 'nn_raw_windflow_inv': (run_windflow, WINDFLOW_INV_JOINT_PARAMS)}

# Same mask as every other GOES tracker -> identical centre output domain.
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


def _validate_requested_trackers() -> None:
    unknown = [name for name in TRACKERS_TO_RUN if name not in TRACKER_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown NN trackers: {unknown}")


def main() -> None:
    _validate_requested_trackers()
    print("=" * 112)
    print("GOES LEARNED OPTICAL FLOW: FLOWDIFFUSER / WINDFLOW (ORIGINAL C14 IMAGES)")
    print("=" * 112)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("CASE_NAME:", CASE_NAME)
    print("DATA_CASE_DIR:", DATA_CASE_DIR)
    print("SATELLITES:", SATELLITES)
    print("TRACKERS_TO_RUN:", TRACKERS_TO_RUN)
    print("GEOMETRY_MODES:", GEOMETRY_MODES)
    print("Device:", DEVICE)
    print("Temporal components:")
    print("  C1: center -> previous; signed dt < 0")
    print("  C2: center -> next; signed dt > 0")
    print("  both dense components are required")

    frame_paths = discover_goes_frames(DATA_CASE_DIR)
    for satellite in SATELLITES:
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
            print("  pressure:", paths.pressure)

        for base_name in TRACKERS_TO_RUN:
            tracker_func, tracker_params = TRACKER_CONFIGS[base_name]
            tracker_name = f"{base_name}_{satellite.lower()}"
            expected = (
                TRACKER_ROOT_DIR
                / tracker_name
                / "final"
                / f"{tracker_name}_{GEOMETRY_MODES[0]}_final_native.npz"
            )
            if SKIP_EXISTING_FINAL and expected.is_file():
                print(f"\n[{tracker_name}] existing final product found; skip: {expected}")
                continue

            print("\n" + "-" * 112)
            print("RUNNING:", tracker_name)
            print("parameters:", tracker_params)
            result = run_goes_traditional_tracker_satellite(
                frames_by_role=frames_by_role,
                tracker_name=tracker_name,
                tracker_func=tracker_func,
                tracker_params=tracker_params,
                tracker_root_dir=TRACKER_ROOT_DIR,
                pressure_output_dir=PRESSURE_OUTPUT_DIR,
                mask_config=MASK_CONFIG,
                geometry_modes=GEOMETRY_MODES,
                geometry_chunk_rows=GEOMETRY_CHUNK_ROWS,
                background_sigma_pixels=INPUT_BACKGROUND_SIGMA_PIXELS,
                save_native_npz=SAVE_NATIVE_NPZ,
                save_component_npz=SAVE_COMPONENT_NPZ,
                save_pressure_product=SAVE_GOES_PRESSURE_PRODUCT,
                save_diagnostic_figures=SAVE_DIAGNOSTIC_FIGURES,
                quicklook_maximum_dimension=QUICKLOOK_MAXIMUM_DIMENSION,
            )
            print("Done:", tracker_name)
            print("Final native products:", result.get("final_paths"))
            print("GOES-pressure AMV products:", result.get("pressure_paths"))
            print("Point counts:", result.get("pressure_point_counts"))
            print("Runtime summary:", result.get("summary_path"))
            print("Total runtime minutes:", float(result["runtime_seconds"]) / 60.0)
            del result
            gc.collect()

        del frames_by_role
        gc.collect()


if __name__ == "__main__":
    main()
