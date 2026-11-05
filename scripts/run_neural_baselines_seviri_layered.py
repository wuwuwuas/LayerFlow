"""Run only bundled neural optical-flow trackers on pressure-layer image pairs."""

from __future__ import annotations

import os

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_io.case_observations import load_case_observations
from pipeline.tracker_runner_layerwise import run_requested_layerwise_trackers
from preprocessing.layerwise_image import LayerwiseImageConfig, PressureLayer
from tracking import run_flowdiffuser, run_windflow


# CASE_NAME = "test_2018-01-12T11"
CASE_NAME = os.environ.get("AMV_CASE", "test_2018-01-12T11")
DATA_CASE_DIR = PROJECT_ROOT / "data" / CASE_NAME.removeprefix("test_")
TRACKER_ROOT_DIR = PROJECT_ROOT / "outputs" / CASE_NAME / "trackers"
FRAME_INTERVAL_MINUTES = 15
MIN_VALID_COMPONENTS = 2
COMPONENT_PAIRS = [(0, 1), (1, 2), (2, 3)]
PRESSURE_PAIR_MODE = "independent"
SAVE_NATIVE_NPZ = True
SAVE_COMPONENT_NPZ = False
REQUESTED_DEVICE = os.environ.get("AMV_DEVICE", "auto")

NN_OUTPUT_PREFIX = "nn_"  # keep validation runs separate from existing outputs
NN_MODELS_TO_RUN = ['nn_layerwise_flowdiffuser', 'nn_layerwise_windflow_inv']


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

LAYER_CONFIG = LayerwiseImageConfig(
    mode="hard_soft_boundary",
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

FLOWDIFFUSER_PARAMS = {
    "repository": os.environ.get("AMV_FLOWDIFFUSER_REPO", str(PROJECT_ROOT / "external" / "FlowDiffuser")),
    "checkpoint": os.environ.get("AMV_FLOWDIFFUSER_CHECKPOINT", str(PROJECT_ROOT / "weights" / "FlowDiffuser-things.pth")),
    "tile_size": 512,
    "overlap": 128,
    "batch_size": 8,
    "seed": 0,
    "device": DEVICE,
    "mixed_precision": True,
    "strict": True,
    "min_tile_output_fraction": 0.0005,
    "max_tile_flow_px": 100.0,
    "layer_normalization_dilation_pixels": 16,
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
    "layer_normalization_dilation_pixels": 16,
}
# Polarity matched to the humidity training images (cold/moist = bright).
WINDFLOW_INV_PARAMS = {**WINDFLOW_PARAMS, "polarity": "inverted"}
# Joint (training-style) equalization: one mapping for both frames.
WINDFLOW_INV_JOINT_PARAMS = {**WINDFLOW_INV_PARAMS, "equalization_mode": "joint_pair"}
TRACKER_CONFIGS = {f'{NN_OUTPUT_PREFIX}layerwise_flowdiffuser': (run_flowdiffuser, FLOWDIFFUSER_PARAMS), f'{NN_OUTPUT_PREFIX}layerwise_windflow': (run_windflow, WINDFLOW_PARAMS), f'{NN_OUTPUT_PREFIX}layerwise_windflow_inv': (run_windflow, WINDFLOW_INV_JOINT_PARAMS)}


observations = load_case_observations(
    case_name=CASE_NAME,
    data_case_dir=DATA_CASE_DIR,
    frame_count=4,
    interval_minutes=FRAME_INTERVAL_MINUTES,
    channel="IR10.8",
    load_oca_data=True,
)
if observations.frame_pressure_hpa is None:
    raise RuntimeError("Four OCA pressure frames P0..P3 were not loaded")

images = [observations.images[i] for i in range(observations.images.shape[0])]
valid_masks = [
    observations.valid_masks[i] for i in range(observations.valid_masks.shape[0])
]
pressures = [
    observations.frame_pressure_hpa[i]
    for i in range(observations.frame_pressure_hpa.shape[0])
]

print("PROJECT_ROOT:", PROJECT_ROOT)
print("CASE_NAME:", CASE_NAME)
print("Device:", DEVICE)
print("NN trackers:", NN_MODELS_TO_RUN)
print("Pressure pair mode:", PRESSURE_PAIR_MODE)

run_requested_layerwise_trackers(
    tracker_configs=TRACKER_CONFIGS,
    trackers_to_run=NN_MODELS_TO_RUN,
    images=images,
    valid_masks=valid_masks,
    frame_pressure_hpa=pressures,
    lon_grid=observations.lon_grid,
    lat_grid=observations.lat_grid,
    frame_times_loaded=list(observations.frame_times),
    frame_interval_minutes=FRAME_INTERVAL_MINUTES,
    tracker_root_dir=TRACKER_ROOT_DIR,
    component_pairs=COMPONENT_PAIRS,
    layer_config=LAYER_CONFIG,
    pressure_pair_mode=PRESSURE_PAIR_MODE,
    min_valid_components=MIN_VALID_COMPONENTS,
    save_native_npz=SAVE_NATIVE_NPZ,
    save_component_npz=SAVE_COMPONENT_NPZ,
)
