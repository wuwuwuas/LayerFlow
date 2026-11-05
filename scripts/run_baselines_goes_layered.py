"""Run existing traditional trackers on GOES pressure-layer C14 images.

Pressure boundaries are configured here and passed into the generic runner.
The tracking implementations contain no GOES pressure-layer definitions.
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
PRESSURE_PAIR_MODE = "independent"

SAVE_NATIVE_NPZ = True
SAVE_COMPONENT_NPZ = False
SAVE_GOES_PRESSURE_PRODUCT = True
SAVE_DIAGNOSTIC_FIGURES = False
QUICKLOOK_MAXIMUM_DIMENSION = 1200
SKIP_EXISTING_FINAL = False

# Same traditional method set as the SEVIRI Table I.  Layerwise runs cost
# roughly four tracker calls per temporal component.
TRACKERS_TO_RUN = ['layerwise_farneback_opencv', 'layerwise_deepflow_opencv', 'layerwise_dis', 'layerwise_tvl1', 'layerwise_dualtvl1', 'layerwise_ilk', 'layerwise_pcaflow']


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

cv2.setUseOptimized(True)
cv2.setNumThreads(max(1, (os.cpu_count() or 4) - 1))

from data_io.goes import discover_goes_frames, frame_for, load_goes_frame  # noqa: E402
from pipeline.goes_traditional_tracker_runner_layerwise import (  # noqa: E402
    run_goes_traditional_layerwise_tracker_satellite,
)
from preprocessing.goes import GoesPcfMaskConfig  # noqa: E402
from preprocessing.layerwise_image import (  # noqa: E402
    LayerwiseImageConfig,
    PressureLayer,
)
from tracking import run_deepflow, run_dis, run_dualtvl1, run_farneback, run_ilk, run_pcaflow, run_tvl1


# -----------------------------------------------------------------------------
# Pressure layers belong to this executable configuration
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

TRACKER_CONFIGS = {'layerwise_farneback_opencv': (run_farneback, {'backend': 'opencv', 'pyr_scale': 0.5, 'levels': 3, 'winsize': 25, 'iterations': 3, 'poly_n': 7, 'poly_sigma': 1.1, 'flags': 0}), 'layerwise_dis': (run_dis, {'preset': 'medium'}), 'layerwise_deepflow_opencv': (run_deepflow, {'backend': 'opencv', 'alpha': 1.0, 'delta': 0.5, 'gamma': 5.0, 'sigma': 0.6, 'min_size': 25, 'downscale_factor': 0.95, 'fixed_point_iterations': 5, 'sor_iterations': 25, 'omega': 1.6, 'input_dtype': 'uint8_0_255', 'verbose': False}), 'layerwise_tvl1': (run_tvl1, {'attachment': 15.0, 'tightness': 0.3, 'num_warp': 5, 'num_iter': 10, 'tol': 0.0001, 'prefilter': False}), 'layerwise_ilk': (run_ilk, {'radius': 11, 'num_warp': 5, 'gaussian': True, 'prefilter': True}), 'layerwise_pcaflow': (run_pcaflow, {}), 'layerwise_dualtvl1': (run_dualtvl1, {'lambda': 0.1, 'theta': 0.3, 'nscales': 5, 'warps': 5, 'epsilon': 0.01, 'medianFiltering': 5})}

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
        raise ValueError(f"Unknown layerwise traditional trackers: {unknown}")


def _print_pressure_layers() -> None:
    print("Pressure-conditioned layers:")
    for index, layer in enumerate(LAYER_CONFIG.layers):
        print(
            f"  {index}: {layer.name:22s} "
            f"[{layer.p_min_hpa:g}, {layer.p_max_hpa:g}) hPa"
        )
    print("  mode:", LAYER_CONFIG.mode)
    print("  pressure pair mode:", PRESSURE_PAIR_MODE)
    print("  require pressure in both frames:", LAYER_CONFIG.require_pressure_in_both_frames)


def main() -> None:
    _validate_requested_trackers()
    print("=" * 112)
    print("GOES TRADITIONAL PRESSURE-LAYER OPTICAL-FLOW BASELINES")
    print("=" * 112)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("CASE_NAME:", CASE_NAME)
    print("DATA_CASE_DIR:", DATA_CASE_DIR)
    print("SATELLITES:", SATELLITES)
    print("TRACKERS_TO_RUN:", TRACKERS_TO_RUN)
    print("GEOMETRY_MODES:", GEOMETRY_MODES)
    print("OpenCV:", cv2.__version__)
    print("OpenCV Python threads:", cv2.getNumThreads())
    print("Temporal components:")
    print("  C1: center/Pcenter -> previous/Pprevious; signed dt < 0")
    print("  C2: center/Pcenter -> next/Pnext; signed dt > 0")
    print("  both dense components are required")
    print("Destination-product hard filtering: disabled")
    _print_pressure_layers()

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
            print("  height:", paths.height)
            print("  pressure:", paths.pressure)
            print("  cloud mask:", paths.cloud_mask)

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
            result = run_goes_traditional_layerwise_tracker_satellite(
                frames_by_role=frames_by_role,
                tracker_name=tracker_name,
                tracker_func=tracker_func,
                tracker_params=tracker_params,
                tracker_root_dir=TRACKER_ROOT_DIR,
                pressure_output_dir=PRESSURE_OUTPUT_DIR,
                layer_config=LAYER_CONFIG,
                pressure_pair_mode=PRESSURE_PAIR_MODE,
                mask_config=MASK_CONFIG,
                geometry_modes=GEOMETRY_MODES,
                geometry_chunk_rows=GEOMETRY_CHUNK_ROWS,
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
