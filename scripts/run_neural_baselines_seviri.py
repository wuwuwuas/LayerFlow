"""Run only bundled neural optical-flow trackers on raw SEVIRI image pairs."""

from __future__ import annotations

import os

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_io.case_observations import load_case_observations
from pipeline.tracker_runner import run_requested_trackers
from tracking import run_flowdiffuser, run_windflow


CASE_NAME = os.environ.get("AMV_CASE", "test_2018-01-12T11")
CASE_NAME = os.environ.get("AMV_CASE", "test_2018-01-12T11")
DATA_CASE_DIR = PROJECT_ROOT / "data" / CASE_NAME.removeprefix("test_")
TRACKER_ROOT_DIR = PROJECT_ROOT / "outputs" / CASE_NAME / "trackers"
FRAME_INTERVAL_MINUTES = 15
MIN_VALID_COMPONENTS = 2
COMPONENT_PAIRS = [(0, 1), (1, 2), (2, 3)]
SAVE_NATIVE_NPZ = True
SAVE_COMPONENT_NPZ = False
REQUESTED_DEVICE = os.environ.get("AMV_DEVICE", "auto")

# Start with one model. The same runner supports all bundled NN models.
NN_OUTPUT_PREFIX = "nn_"  # keep validation runs separate from existing outputs
NN_MODELS_TO_RUN = ['nn_raw_flowdiffuser', 'nn_raw_windflow_inv']


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
WINDFLOW_INV_JOINT_PARAMS = {**WINDFLOW_INV_PARAMS, "equalization_mode": "joint_pair"}
TRACKER_CONFIGS = {f'{NN_OUTPUT_PREFIX}raw_flowdiffuser': (run_flowdiffuser, FLOWDIFFUSER_PARAMS), f'{NN_OUTPUT_PREFIX}raw_windflow': (run_windflow, WINDFLOW_PARAMS), f'{NN_OUTPUT_PREFIX}raw_windflow_inv': (run_windflow, WINDFLOW_INV_JOINT_PARAMS)}


observations = load_case_observations(
    case_name=CASE_NAME,
    data_case_dir=DATA_CASE_DIR,
    frame_count=4,
    interval_minutes=FRAME_INTERVAL_MINUTES,
    channel="IR10.8",
    load_oca_data=False,
)
images = [observations.images[i] for i in range(observations.images.shape[0])]
valid_masks = [
    observations.valid_masks[i] for i in range(observations.valid_masks.shape[0])
]

print("PROJECT_ROOT:", PROJECT_ROOT)
print("CASE_NAME:", CASE_NAME)
print("Device:", DEVICE)
print("NN trackers:", NN_MODELS_TO_RUN)

run_requested_trackers(
    tracker_configs=TRACKER_CONFIGS,
    trackers_to_run=NN_MODELS_TO_RUN,
    images=images,
    valid_masks=valid_masks,
    lon_grid=observations.lon_grid,
    lat_grid=observations.lat_grid,
    frame_times_loaded=list(observations.frame_times),
    frame_interval_minutes=FRAME_INTERVAL_MINUTES,
    tracker_root_dir=TRACKER_ROOT_DIR,
    component_pairs=COMPONENT_PAIRS,
    normalize_image_pair=None,
    min_valid_components=MIN_VALID_COMPONENTS,
    save_native_npz=SAVE_NATIVE_NPZ,
    save_component_npz=SAVE_COMPONENT_NPZ,
)
