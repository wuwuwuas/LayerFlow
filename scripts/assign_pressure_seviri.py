# -*- coding: utf-8 -*-
"""Pressure assignment for PIV-AMV products.

This script assigns pressure to clean stacked PIV/OF wind products.

It does not run PIV tracking, QC, ERA5 comparison, or CSV export. It only
creates compact pressure-assigned AMV NPZ files.

Supported products
------------------
- PIV-AMV + OCA component-start pressure grids
- PIV-AMV + official AMV sparse pressure points

Path discovery
--------------
The script no longer hard-codes individual OCA / official-AMV NetCDF file
names. You only need to set:

    CASE_NAME
    DATA_CASE_DIR

Then files are discovered from:

    DATA_CASE_DIR/eumetsat_oca/native/**/*.nc
    DATA_CASE_DIR/eumetsat_amv/native/**/*.nc

Requires:
    src/data_io/product_discovery.py
"""

from __future__ import annotations

import os

from pathlib import Path
import sys

try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)


# %%
# ---------------------------------------------------------------------
# Project root discovery
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print("PROJECT_ROOT =", PROJECT_ROOT)
print("SRC_DIR      =", SRC_DIR)


# %%
# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------
# Output case name. This should match the OF/tracker output directory.
CASE_NAME = os.environ.get("AMV_CASE", "test_2018-01-12T11")
from runtime_options import read_case_inputs
CASE_INPUTS = read_case_inputs(PROJECT_ROOT, CASE_NAME)
DATA_CASE_NAME = CASE_NAME.removeprefix("test_")
DATA_CASE_DIR = Path(CASE_INPUTS.get("data_dir", PROJECT_ROOT / "data" / DATA_CASE_NAME))

OUTPUT_DIR = PROJECT_ROOT / "outputs" / CASE_NAME
PRESSURE_OUTPUT_DIR = OUTPUT_DIR / "pressure_assignment"

# Reuse the retrieval's first frame, or fall back to a timestamp directory.
EXPECTED_CASE_TIME = (CASE_INPUTS.get("frame_times") or [DATA_CASE_DIR.name])[0]

# Process one or multiple trackers.
TRACKER_NAMES = ['raw_farneback_opencv', 'raw_deepflow_opencv', 'raw_dis', 'raw_tvl1', 'raw_dualtvl1', 'raw_ilk', 'raw_pcaflow', 'layerwise_farneback_opencv', 'layerwise_deepflow_opencv', 'layerwise_dis', 'layerwise_tvl1', 'layerwise_dualtvl1', 'layerwise_ilk', 'layerwise_pcaflow', 'nn_raw_flowdiffuser', 'nn_layerwise_flowdiffuser', 'nn_raw_windflow_inv', 'nn_layerwise_windflow_inv']

# Clean PIV final product template.
PIV_FINAL_TEMPLATE = (
    OUTPUT_DIR
    / "trackers"
    / "{tracker}"
    / "final"
    / "{tracker}_final_native.npz"
)

# ---------------------------------------------------------------------
# Pressure source settings
# ---------------------------------------------------------------------
RUN_OCA_PRESSURE = True
RUN_OFFICIAL_PRESSURE = False  # Use cloud-product pressure for evaluation.

OCA_NATIVE_DIR = DATA_CASE_DIR / "eumetsat_oca" / "native"
OFFICIAL_AMV_NATIVE_DIR = DATA_CASE_DIR / "eumetsat_amv" / "native"

# ---------------------------------------------------------------------
# OCA pressure assignment settings
# ---------------------------------------------------------------------
# Order matters after auto-discovery:
#   C1 uses OCA_COMPONENT_FILES[0]
#   C2 uses OCA_COMPONENT_FILES[1]
#   C3 uses OCA_COMPONENT_FILES[2]
#
# Assignment mode:
#   auto         : dense one-vector-per-pixel PIV -> pixel direct;
#                  sparse/coarse PIV -> window
#   pixel_direct : force same/nearest-pixel OCA assignment
#   window       : force local OCA window statistic
# For sparse/coarse PIV, the default local statistic is 3x3 median.
MIN_VALID_PRESSURE_COMPONENTS = 2
OCA_ASSIGNMENT_MODE = "auto"
OCA_WINDOW_SIZE = 3
OCA_WINDOW_STATISTIC = "median"
OCA_N_COMPONENTS = 3
OCA_INTERVAL_MINUTES = CASE_INPUTS.get("interval_minutes") or 15
OCA_TOLERANCE_MINUTES = 7.5

# ---------------------------------------------------------------------
# Official AMV pressure source settings
# ---------------------------------------------------------------------
# Official AMV is sparse and anchors the output.
# For each official point, PIV u/v is averaged over nearby valid PIV vectors.
# Output lon/lat and pressure are official AMV lon/lat and official pressure.
OFFICIAL_RADIUS_KM = 50.0
MIN_PIV_POINTS_PER_OFFICIAL = 3
OFFICIAL_PIV_STATISTIC = "mean"
OFFICIAL_TOLERANCE_MINUTES = 90.0

# Optional pressure-derived standard atmosphere height.
# Keep False if the product should only contain pressure_hpa.
INCLUDE_HEIGHT_M = False

PRESSURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("CASE_NAME           =", CASE_NAME)
print("DATA_CASE_DIR       =", DATA_CASE_DIR)
print("EXPECTED_CASE_TIME  =", EXPECTED_CASE_TIME)
print("OUTPUT_DIR          =", OUTPUT_DIR)
print("PRESSURE_OUTPUT_DIR =", PRESSURE_OUTPUT_DIR)


# %%
import numpy as np

from data_io.piv_amv import load_piv_final_native_npz, inspect_npz
from data_io.oca import OcaData, load_oca_component_files
from data_io.official_amv import load_official_amv_file
from data_io.product_discovery import (
    find_oca_component_files,
    find_official_amv_file,
)
from postprocess.height_assignment import (
    assign_piv_pressure_from_oca_components,
    assign_piv_pressure_from_official_points,
)


# %%
# ---------------------------------------------------------------------
# Automatically resolve local pressure source files
# ---------------------------------------------------------------------
OCA_COMPONENT_FILES: list[Path] = []
if RUN_OCA_PRESSURE:
    OCA_COMPONENT_FILES = find_oca_component_files(
        OCA_NATIVE_DIR,
        n_components=OCA_N_COMPONENTS,
        expected_start_time=EXPECTED_CASE_TIME,
        interval_minutes=OCA_INTERVAL_MINUTES,
        tolerance_minutes=OCA_TOLERANCE_MINUTES,
        verbose=True,
    )

OFFICIAL_AMV_FILE: Path | None = None
if RUN_OFFICIAL_PRESSURE:
    OFFICIAL_AMV_FILE = find_official_amv_file(
        OFFICIAL_AMV_NATIVE_DIR,
        expected_time=EXPECTED_CASE_TIME,
        tolerance_minutes=OFFICIAL_TOLERANCE_MINUTES,
        verbose=True,
    )


# %%
# ---------------------------------------------------------------------
# Check input paths before processing
# ---------------------------------------------------------------------
print("PIV inputs:")
for tracker in TRACKER_NAMES:
    piv_path = Path(str(PIV_FINAL_TEMPLATE).format(tracker=tracker))
    print(f"  {tracker:<28} {piv_path}  exists={piv_path.exists()}")

if RUN_OCA_PRESSURE:
    print("\nOCA component files:")
    for i, path in enumerate(OCA_COMPONENT_FILES, start=1):
        print(f"  C{i}: {path}  exists={path.exists()}")

if RUN_OFFICIAL_PRESSURE:
    print("\nOfficial AMV file:")
    print(" ", OFFICIAL_AMV_FILE, "exists=", OFFICIAL_AMV_FILE.exists() if OFFICIAL_AMV_FILE else False)


# %%
# ---------------------------------------------------------------------
# Load pressure sources once
# ---------------------------------------------------------------------
oca = None
if RUN_OCA_PRESSURE:
    if CASE_INPUTS.get("aoi"):
        from data_io.case_observations import load_case_observations
        observations = load_case_observations(
            case_name=CASE_NAME, data_case_dir=DATA_CASE_DIR,
            frame_times=CASE_INPUTS.get("frame_times") or None,
            interval_minutes=OCA_INTERVAL_MINUTES, aoi=CASE_INPUTS["aoi"],
        )
        oca = OcaData(
            ctp_hpa=tuple(observations.frame_pressure_hpa[:3]),
            cloud_probability=tuple(observations.frame_cloud_probability[:3]),
            times=observations.oca_times[:3], paths=observations.oca_paths[:3],
            component_source="start_frame",
        )
    else:
        oca = load_oca_component_files(OCA_COMPONENT_FILES)
    print("Loaded OCA component pressure files:")
    for i, (path, ctp) in enumerate(zip(oca.paths[:3], oca.ctp_hpa[:3]), start=1):
        print(f"  C{i}: {path.name}, shape={ctp.shape}, finite={np.isfinite(ctp).sum():,}")
    if any(item is not None for item in oca.cloud_probability[:3]):
        print("OCA cloud probability is available.")
    else:
        print("OCA cloud probability is not available in these files.")

official = None
if RUN_OFFICIAL_PRESSURE:
    if OFFICIAL_AMV_FILE is None:
        raise RuntimeError("RUN_OFFICIAL_PRESSURE=True but OFFICIAL_AMV_FILE was not resolved.")
    official = load_official_amv_file(OFFICIAL_AMV_FILE)
    print("Loaded official AMV:", official.path)
    print("Official AMV points with pressure:", len(official.frame))
    display(official.frame.head())


# %%
# ---------------------------------------------------------------------
# Run pressure assignment for all requested trackers
# ---------------------------------------------------------------------
outputs = {}

for tracker in TRACKER_NAMES:
    print("=" * 90)
    print("Tracker:", tracker)

    piv_path = Path(str(PIV_FINAL_TEMPLATE).format(tracker=tracker))
    if not piv_path.exists():
        print("Missing PIV file, skip:", piv_path)
        continue

    piv = load_piv_final_native_npz(piv_path)
    print("Loaded PIV:", piv_path)
    print("  shape:", piv["u_ms"].shape)
    print("  valid final wind pixels:", int(np.asarray(piv["valid_mask"], dtype=bool).sum()))

    tracker_outputs = {}

    if RUN_OCA_PRESSURE:
        if oca is None:
            raise RuntimeError("RUN_OCA_PRESSURE=True but OCA data were not loaded.")
        oca_output_path = PRESSURE_OUTPUT_DIR / f"{tracker}_final_amv_oca_pressure.npz"
        oca_result = assign_piv_pressure_from_oca_components(
            piv=piv,
            oca=oca,
            output_path=oca_output_path,
            min_valid_pressure_components=MIN_VALID_PRESSURE_COMPONENTS,
            oca_assignment_mode=OCA_ASSIGNMENT_MODE,
            oca_window_size=OCA_WINDOW_SIZE,
            oca_window_statistic=OCA_WINDOW_STATISTIC,
            include_height_m=INCLUDE_HEIGHT_M,
        )
        tracker_outputs["oca"] = oca_output_path
        print("  saved OCA pressure product:", oca_output_path)
        print("  OCA pressure-assigned points:", oca_result["pressure_hpa"].size)
        print("  OCA spatial assignment:", oca_result["metadata"].get("spatial_assignment_resolved"))

    if RUN_OFFICIAL_PRESSURE:
        if official is None:
            raise RuntimeError("RUN_OFFICIAL_PRESSURE=True but official AMV data were not loaded.")
        official_output_path = PRESSURE_OUTPUT_DIR / f"{tracker}_final_amv_official_pressure.npz"
        official_result = assign_piv_pressure_from_official_points(
            piv=piv,
            official_frame=official.frame,
            output_path=official_output_path,
            radius_km=OFFICIAL_RADIUS_KM,
            min_piv_points=MIN_PIV_POINTS_PER_OFFICIAL,
            statistic=OFFICIAL_PIV_STATISTIC,
            include_height_m=INCLUDE_HEIGHT_M,
        )
        tracker_outputs["official"] = official_output_path
        print("  saved official-pressure product:", official_output_path)
        print("  official pressure-assigned points:", official_result["pressure_hpa"].size)

    outputs[tracker] = tracker_outputs

print("\nDone.")
print(outputs)


# %%
# ---------------------------------------------------------------------
# Inspect generated products
# ---------------------------------------------------------------------
# for tracker, items in outputs.items():
#     print("=" * 90)
#     print("Tracker:", tracker)
#     for source, path in items.items():
#         print("\nSource:", source)
#         inspect_npz(path)


# %% [markdown]
# ## Expected output fields
#
# ### OCA-pressure product
#
# The OCA output is compact 1-D and keeps only PIV points with assigned pressure:
#
# ```text
# row_center_original
# col_center_original
# longitude
# latitude
# u_ms
# v_ms
# valid_component_count
# pressure_hpa
# pressure_component_count
# pressure_c1_hpa
# pressure_c2_hpa
# pressure_c3_hpa
# cloud_probability              # if available
# cloud_probability_c1           # if available
# cloud_probability_c2           # if available
# cloud_probability_c3           # if available
# pressure_source
# metadata_json
# ```
#
# Spatial assignment is `pixel_direct` for dense one-vector-per-pixel PIV and
# `3x3 median` for sparse/coarse PIV when `OCA_ASSIGNMENT_MODE = "auto"`.
# C1/C2/C3 pressure components are combined with a valid-component mean.
#
# ### Official-AMV-pressure product
#
# The official-pressure output is official-point anchored. It keeps official
# AMV longitude/latitude and official pressure, while replacing official u/v
# with local PIV-window mean u/v:
#
# ```text
# official_id
# longitude
# latitude
# u_ms
# v_ms
# pressure_hpa
# piv_neighbor_count
# pressure_source
# metadata_json
# ```
#
# No `diag_*`, no pressure method array, no official quality, no PIV row/col,
# and no PIV distance diagnostic fields are written.
