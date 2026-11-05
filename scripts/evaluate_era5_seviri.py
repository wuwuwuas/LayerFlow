
import os

# Evaluate SEVIRI winds against ERA5



try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)


# %% [markdown]
# # Compare AMV products with ERA5
#


# %%
# 0. Imports and project paths

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_io.official_amv import load_official_amv
from data_io.piv_amv import inspect_npz
from data_io.product_discovery import (
    find_official_amv_file,
    find_era5_file,
    infer_case_start_time_from_path,
    infer_amv_valid_time_from_path,
)
from validation.compare_era5 import (
    inspect_official_time_variables,
    ensure_official_era5_cache,
    update_oca_era5_cache_from_amv_product,
    load_era5_cache_npz,
    cache_coverage_summary,
    build_official_amv_vs_era5_table,
    build_pressure_amv_vs_era5_table,
    compare_to_era5,
)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("SRC_DIR:", SRC_DIR)


# %%
# 1. Settings

# ---------------------------------------------------------------------
# Case settings
# ---------------------------------------------------------------------
# Reuse observation paths and times saved by the retrieval.
CASE_NAME = os.environ.get("AMV_CASE", "test_2018-01-12T11")
from runtime_options import read_case_inputs
CASE_INPUTS = read_case_inputs(PROJECT_ROOT, CASE_NAME)
DATA_CASE_NAME = CASE_NAME.removeprefix("test_")
DATA_CASE_DIR = Path(CASE_INPUTS.get("data_dir", PROJECT_ROOT / "data" / DATA_CASE_NAME))

# For a 4-frame sequence 11:00, 11:15, 11:30, 11:45, the final AMV valid
# time is usually the last frame time, i.e. case start + 45 minutes.
AMV_VALID_OFFSET_MINUTES = 45

CASE_START_TIME = (
    pd.Timestamp(CASE_INPUTS["frame_times"][0]).to_pydatetime()
    if CASE_INPUTS.get("frame_times") else infer_case_start_time_from_path(DATA_CASE_DIR)
)
AMV_VALID_TIME = pd.Timestamp(
    CASE_INPUTS["frame_times"][-1] if CASE_INPUTS.get("frame_times") else infer_amv_valid_time_from_path(
        DATA_CASE_DIR,
        offset_minutes=AMV_VALID_OFFSET_MINUTES,
    )
)

# Standard products follow this naming pattern:
#   {tracker}_final_amv_oca_pressure.npz
#   {tracker}_final_amv_official_pressure.npz
TRACKER_NAMES = ['raw_farneback_opencv', 'raw_deepflow_opencv', 'raw_dis', 'raw_tvl1', 'raw_dualtvl1', 'raw_ilk', 'raw_pcaflow', 'layerwise_farneback_opencv', 'layerwise_deepflow_opencv', 'layerwise_dis', 'layerwise_tvl1', 'layerwise_dualtvl1', 'layerwise_ilk', 'layerwise_pcaflow', 'nn_raw_flowdiffuser', 'nn_layerwise_flowdiffuser', 'nn_raw_windflow_inv', 'nn_layerwise_windflow_inv', 'pressure_conditioned_flow']

OUTPUT_DIR = PROJECT_ROOT / "outputs" / CASE_NAME
PRESSURE_ASSIGNMENT_DIR = OUTPUT_DIR / "pressure_assignment"

COMPARISON_DIR = OUTPUT_DIR / "comparison" / "amv_vs_era5"
ERA5_COLLOCATION_DIR = COMPARISON_DIR / "era5_collocated"
METRICS_DIR = COMPARISON_DIR / "metrics"
FIGURE_DIR = COMPARISON_DIR / "figure"

for path in (ERA5_COLLOCATION_DIR, METRICS_DIR, FIGURE_DIR):
    path.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Auto-discover external reference files from DATA_CASE_DIR
# ---------------------------------------------------------------------
OFFICIAL_AMV_NATIVE_DIR = DATA_CASE_DIR / "eumetsat_amv" / "native"
ERA5_DIR = DATA_CASE_DIR / "era5"

OFFICIAL_AMV_FILE = find_official_amv_file(
    OFFICIAL_AMV_NATIVE_DIR,
    expected_time=AMV_VALID_TIME.to_pydatetime(),
    tolerance_minutes=90.0,
) if any(OFFICIAL_AMV_NATIVE_DIR.rglob("*.nc")) else None

ERA5_FILE = find_era5_file(
    ERA5_DIR,
    expected_time=AMV_VALID_TIME.to_pydatetime(),
    tolerance_minutes=180.0,
)

# ---------------------------------------------------------------------
# Product lists
# ---------------------------------------------------------------------
OCA_PRODUCT_SPECS = [
    {
        "tracker": tracker,
        "comparison": f"{tracker}_oca_pressure",
        "path": PRESSURE_ASSIGNMENT_DIR / f"{tracker}_final_amv_oca_pressure.npz",
    }
    for tracker in TRACKER_NAMES
]

OFFICIAL_PRODUCT_SPECS = [
    {
        "tracker": tracker,
        "comparison": f"{tracker}_official_pressure",
        "path": PRESSURE_ASSIGNMENT_DIR / f"{tracker}_final_amv_official_pressure.npz",
    }
    for tracker in TRACKER_NAMES
]

OFFICIAL_ERA5_CACHE = ERA5_COLLOCATION_DIR / "official_amv_era5_collocated.npz"
OCA_ERA5_CACHE = ERA5_COLLOCATION_DIR / "of_dense_piv_oca_pressure_era5_collocated.npz"

# Cache controls.
REBUILD_OFFICIAL_ERA5_CACHE = False
REBUILD_OCA_ERA5_CACHE = False
SAVE_COMPRESSED_NPZ = False
ERA5_CHUNK_SIZE = 50_000

# Run controls.
RUN_OFFICIAL_AMV = OFFICIAL_AMV_FILE is not None
RUN_OCA_PRESSURE_PRODUCTS = True
RUN_OFFICIAL_PRESSURE_PRODUCTS = False  # Use cloud-product pressure for evaluation.
SAVE_FIGURES = False  # Result visualization uses dedicated plot_* scripts.

print("CASE_NAME:", CASE_NAME)
print("DATA_CASE_DIR:", DATA_CASE_DIR)
print("CASE_START_TIME:", CASE_START_TIME)
print("AMV_VALID_TIME:", AMV_VALID_TIME)
print("OFFICIAL_AMV_NATIVE_DIR:", OFFICIAL_AMV_NATIVE_DIR)
print("OFFICIAL_AMV_FILE:", OFFICIAL_AMV_FILE)
print("ERA5_DIR:", ERA5_DIR)
print("ERA5_FILE:", ERA5_FILE, ERA5_FILE.exists())
print("PRESSURE_ASSIGNMENT_DIR:", PRESSURE_ASSIGNMENT_DIR)
print("FIGURE_DIR:", FIGURE_DIR)

print("\nOCA products:")
for spec in OCA_PRODUCT_SPECS:
    print(" ", spec["comparison"], "->", spec["path"], spec["path"].exists())

print("\nOfficial-pressure products:")
for spec in OFFICIAL_PRODUCT_SPECS:
    print(" ", spec["comparison"], "->", spec["path"], spec["path"].exists())


# %%
# 2. Inspect time-like variables in the raw official AMV file

if RUN_OFFICIAL_AMV:
    time_variables = inspect_official_time_variables(OFFICIAL_AMV_FILE)
    display(time_variables)
else:
    print("Evaluating retrieved AMVs; no operational AMV input selected.")


# %%
# 3. Inspect AMV pressure-assigned products

# if RUN_OCA_PRESSURE_PRODUCTS:
#     print("\n" + "=" * 90)
#     print("OCA-pressure products")
#     for spec in OCA_PRODUCT_SPECS:
#         print("\nComparison:", spec["comparison"])
#         inspect_npz(spec["path"])

# if RUN_OFFICIAL_PRESSURE_PRODUCTS:
#     print("\n" + "=" * 90)
#     print("Official-pressure products")
#     for spec in OFFICIAL_PRODUCT_SPECS:
#         print("\nComparison:", spec["comparison"])
#         inspect_npz(spec["path"])


# %%
# 4. Load ERA5 and official AMV

if not ERA5_FILE.exists():
    raise FileNotFoundError(f"ERA5 file not found: {ERA5_FILE}")
# xarray keeps data lazy until interpolation chunks are requested.
era5_ds = xr.open_dataset(ERA5_FILE)
print("ERA5 dataset:")
print(era5_ds)

official = None
if RUN_OFFICIAL_AMV:
    official_config = {"official_amv_file": OFFICIAL_AMV_FILE}
    official = load_official_amv(official_config)
    print("Official AMV rows:", len(official.frame))
    display(official.frame.head())


# %%
# 5. Build or reuse ERA5 collocation caches

# 5a. Official-AMV geometry cache.
if RUN_OFFICIAL_AMV:
    t0 = time.perf_counter()
    ensure_official_era5_cache(
        official_amv=official,
        era5=era5_ds,
        cache_path=OFFICIAL_ERA5_CACHE,
        valid_time=AMV_VALID_TIME,
        chunk_size=ERA5_CHUNK_SIZE,
        rebuild=REBUILD_OFFICIAL_ERA5_CACHE,
        compress=SAVE_COMPRESSED_NPZ,
    )
    print(f"Official ERA5 cache ready in {time.perf_counter() - t0:.1f} s")

# 5b. Reuse row/column/pressure collocations across OCA-pressure products.
if RUN_OCA_PRESSURE_PRODUCTS:
    first_existing = True
    for spec in OCA_PRODUCT_SPECS:
        oca_path = spec["path"]
        product_name = spec["comparison"]
        if not oca_path.exists():
            print("Missing OCA-P product, skipping:", oca_path)
            continue
        t0 = time.perf_counter()
        update_oca_era5_cache_from_amv_product(
            amv_product_path=oca_path,
            product_name=product_name,
            era5=era5_ds,
            cache_path=OCA_ERA5_CACHE,
            valid_time=AMV_VALID_TIME,
            amv_geometry_source="of_dense_piv",
            pressure_source="oca",
            chunk_size=ERA5_CHUNK_SIZE,
            rebuild=REBUILD_OCA_ERA5_CACHE and first_existing,
            compress=SAVE_COMPRESSED_NPZ,
        )
        print(f"OCA ERA5 cache update for {product_name} finished in {time.perf_counter() - t0:.1f} s")
        first_existing = False
        REBUILD_OCA_ERA5_CACHE = False

print("\nOfficial cache:", OFFICIAL_ERA5_CACHE)
print("OCA cache:", OCA_ERA5_CACHE)
if OCA_ERA5_CACHE.exists():
    display(cache_coverage_summary(OCA_ERA5_CACHE))


# %%
# 6. Compute AMV-vs-ERA5 metrics: all points and QC-filtered points

# Raw metrics are computed for all ERA5-collocated valid AMVs.
# QC metrics additionally require:
#   QI >= QC_QI_MIN, speed_min <= AMV speed <= speed_max, and pressure in range.
# The QC CSV also contains raw_sample_count and retained_fraction for each subset.

QC_QI_MIN = 60.0
QC_SPEED_MIN = 0.0
QC_SPEED_MAX = 150.0
QC_PRESSURE_RANGE = (100.0, 1000.0)

# Set to 50.0 if you also want to require OCA cloud_probability >= 50 for OCA-P products.
# Keep None for the main QI+speed QC experiment.
QC_CLOUD_PROBABILITY_MIN = 50

PRESSURE_LAYER_SUBSETS_FOR_METRICS = [
    "all_valid",
    "pressure_high_lt400",
    "pressure_mid_400_700",
    "pressure_low_gt700",
]


def _numeric_series(table, column, default=np.nan):
    if column in table.columns:
        return pd.to_numeric(table[column], errors="coerce").astype(float)
    return pd.Series(default, index=table.index, dtype=float)


def _bool_series(table, column, default=False):
    if column in table.columns:
        return table[column].fillna(default).astype(bool)
    return pd.Series(default, index=table.index, dtype=bool)


def _first_existing_column(table, candidates):
    for column in candidates:
        if column in table.columns:
            return column
    return None


def _normalise_quality_to_0_100(values):
    q = pd.to_numeric(values, errors="coerce").astype(float)
    finite = np.isfinite(q)
    if finite.any() and float(np.nanmedian(q[finite])) <= 1.5:
        q = q * 100.0
    return q


def _safe_corr(a, b):
    frame = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(frame) < 2:
        return np.nan
    if frame["a"].std() <= 0 or frame["b"].std() <= 0:
        return np.nan
    return float(frame["a"].corr(frame["b"]))


def build_raw_subset_masks_for_metrics(table):
    """All valid + pressure-layer masks without applying QC."""
    u_amv = _numeric_series(table, "u_amv")
    v_amv = _numeric_series(table, "v_amv")
    u_era5 = _numeric_series(table, "u_era5")
    v_era5 = _numeric_series(table, "v_era5")
    p = _numeric_series(table, "pressure_hpa")

    base = (
        np.isfinite(u_amv)
        & np.isfinite(v_amv)
        & np.isfinite(u_era5)
        & np.isfinite(v_era5)
    )
    if "era5_valid_mask" in table.columns:
        base = base & _bool_series(table, "era5_valid_mask", default=False)

    masks = {
        "all_valid": pd.Series(base, index=table.index),
        "pressure_high_lt400": pd.Series(base & p.lt(400.0), index=table.index),
        "pressure_mid_400_700": pd.Series(base & p.ge(400.0) & p.lt(700.0), index=table.index),
        "pressure_low_gt700": pd.Series(base & p.ge(700.0), index=table.index),
    }
    return masks


def build_qc_subset_masks_for_metrics(
    table,
    *,
    qi_min=60.0,
    speed_min=5.0,
    speed_max=150.0,
    pressure_range=(100.0, 1000.0),
    cloud_probability_min=None,
):
    """QC masks with the same subset names as raw masks.

    Because the QC metrics are written to a separate CSV, using the same subset
    names keeps raw and QC tables directly comparable.
    """
    raw = build_raw_subset_masks_for_metrics(table)
    base = raw["all_valid"].copy()

    p = _numeric_series(table, "pressure_hpa")
    speed = _numeric_series(table, "speed_amv")
    if not np.isfinite(speed).any() and {"u_amv", "v_amv"}.issubset(table.columns):
        speed = np.hypot(_numeric_series(table, "u_amv"), _numeric_series(table, "v_amv"))

    pressure_ok = p.ge(float(pressure_range[0])) & p.le(float(pressure_range[1]))
    speed_ok = speed.ge(float(speed_min)) & speed.le(float(speed_max))

    q_col = _first_existing_column(
        table,
        [
            "quality_index",
            "official_quality",
            "quality",
            "qi",
            "quality_score",
            "final_quality_score",
        ],
    )
    if q_col is not None:
        q = _normalise_quality_to_0_100(table[q_col])
        qi_ok = q.ge(float(qi_min))
        finite_q = q[np.isfinite(q)]
        if finite_q.size:
            print(
                f"  quality column used: {q_col}; "
                f"range={float(np.nanmin(finite_q)):.2f}–{float(np.nanmax(finite_q)):.2f}; "
                f"threshold={qi_min:.1f}"
            )
    else:
        qi_ok = pd.Series(True, index=table.index)
        print("  quality column used: None; QI filter not applied for this table")

    cloud_ok = pd.Series(True, index=table.index)
    if cloud_probability_min is not None:
        cp_col = _first_existing_column(
            table,
            [
                "cloud_probability",
                "cloud_probability_mean",
                "cloud_probability_median",
                "cloud_prob",
                "probability_of_cloud",
            ],
        )
        if cp_col is not None:
            cp = _normalise_quality_to_0_100(table[cp_col])
            cloud_ok = cp.ge(float(cloud_probability_min))
            print(f"  cloud probability column used: {cp_col}; threshold={cloud_probability_min:.1f}")
        else:
            print("  cloud probability column used: None; cloud filter not applied for this table")

    qc_base = base & pressure_ok & speed_ok & qi_ok & cloud_ok

    masks = {
        "all_valid": qc_base,
        "pressure_high_lt400": qc_base & p.lt(400.0),
        "pressure_mid_400_700": qc_base & p.ge(400.0) & p.lt(700.0),
        "pressure_low_gt700": qc_base & p.ge(700.0),
    }
    return masks


def add_correlation_metrics(metrics, table, subset_masks):
    """Add speed/u/v/component correlations against ERA5 for each subset."""
    if metrics.empty:
        return metrics

    out = metrics.copy()
    for subset_name, mask in subset_masks.items():
        selected = table.loc[mask.fillna(False)].copy()
        idx = out.index[out["subset"].eq(subset_name)]
        if len(idx) == 0:
            continue
        idx = idx[0]

        speed_amv = _numeric_series(selected, "speed_amv")
        speed_era5 = _numeric_series(selected, "speed_era5")
        if not np.isfinite(speed_amv).any() and {"u_amv", "v_amv"}.issubset(selected.columns):
            speed_amv = np.hypot(_numeric_series(selected, "u_amv"), _numeric_series(selected, "v_amv"))
        if not np.isfinite(speed_era5).any() and {"u_era5", "v_era5"}.issubset(selected.columns):
            speed_era5 = np.hypot(_numeric_series(selected, "u_era5"), _numeric_series(selected, "v_era5"))

        u_amv = _numeric_series(selected, "u_amv")
        u_era5 = _numeric_series(selected, "u_era5")
        v_amv = _numeric_series(selected, "v_amv")
        v_era5 = _numeric_series(selected, "v_era5")

        out.loc[idx, "speed_correlation"] = _safe_corr(speed_amv, speed_era5)
        out.loc[idx, "u_correlation"] = _safe_corr(u_amv, u_era5)
        out.loc[idx, "v_correlation"] = _safe_corr(v_amv, v_era5)
        out.loc[idx, "vector_component_correlation"] = _safe_corr(
            np.concatenate([u_amv.to_numpy(float), v_amv.to_numpy(float)]),
            np.concatenate([u_era5.to_numpy(float), v_era5.to_numpy(float)]),
        )

    return out


def compute_metric_table(table, *, comparison, tracker, comparison_source, metric_set, subset_masks):
    metrics = compare_to_era5(table, comparison=comparison, subset_masks=subset_masks)
    metrics = add_correlation_metrics(metrics, table, subset_masks)
    metrics.insert(0, "metric_set", metric_set)
    metrics.insert(0, "comparison_source", comparison_source)
    metrics.insert(0, "tracker", tracker)
    return metrics


def attach_qc_retention_fraction(raw_metrics, qc_metrics):
    raw = raw_metrics.copy()
    qc = qc_metrics.copy()

    id_cols = ["tracker", "comparison_source", "comparison", "subset"]
    raw_counts = raw[id_cols + ["sample_count"]].rename(columns={"sample_count": "raw_sample_count"})
    qc = qc.merge(raw_counts, on=id_cols, how="left")
    qc["retained_fraction"] = qc["sample_count"] / qc["raw_sample_count"]

    raw["raw_sample_count"] = raw["sample_count"]
    raw["retained_fraction"] = 1.0
    return raw, qc


raw_metrics_list = []
qc_metrics_list = []
comparison_tables_for_debug = {}

# 6a. Official AMV vs ERA5.
if RUN_OFFICIAL_AMV:
    print("\n" + "=" * 90)
    print("Comparing official_amv vs ERA5")
    table = build_official_amv_vs_era5_table(official, OFFICIAL_ERA5_CACHE)
    print("Rows after joining ERA5 cache:", len(table))

    raw_masks = build_raw_subset_masks_for_metrics(table)
    qc_masks = build_qc_subset_masks_for_metrics(
        table,
        qi_min=QC_QI_MIN,
        speed_min=QC_SPEED_MIN,
        speed_max=QC_SPEED_MAX,
        pressure_range=QC_PRESSURE_RANGE,
        cloud_probability_min=None,
    )

    raw_metrics_list.append(
        compute_metric_table(
            table,
            comparison="official_amv",
            tracker="official",
            comparison_source="official_amv",
            metric_set="all_points",
            subset_masks=raw_masks,
        )
    )
    qc_metrics_list.append(
        compute_metric_table(
            table,
            comparison="official_amv",
            tracker="official",
            comparison_source="official_amv",
            metric_set="qc_qi60_speed5",
            subset_masks=qc_masks,
        )
    )
    comparison_tables_for_debug["official_amv"] = table

# 6b. PIV-AMV products vs ERA5.
for tracker in TRACKER_NAMES:
    if RUN_OCA_PRESSURE_PRODUCTS:
        oca_path = PRESSURE_ASSIGNMENT_DIR / f"{tracker}_final_amv_oca_pressure.npz"
        if oca_path.exists() and OCA_ERA5_CACHE.exists():
            print("\n" + "=" * 90)
            print(f"Comparing {tracker}_oca_pressure vs ERA5")
            table = build_pressure_amv_vs_era5_table(
                oca_path,
                OCA_ERA5_CACHE,
                join_mode="row_col_pressure",
                amv_source=f"{tracker}_oca_pressure",
            )
            print("Rows after joining ERA5 cache:", len(table))

            raw_masks = build_raw_subset_masks_for_metrics(table)
            qc_masks = build_qc_subset_masks_for_metrics(
                table,
                qi_min=QC_QI_MIN,
                speed_min=QC_SPEED_MIN,
                speed_max=QC_SPEED_MAX,
                pressure_range=QC_PRESSURE_RANGE,
                cloud_probability_min=QC_CLOUD_PROBABILITY_MIN,
            )

            raw_metrics_list.append(
                compute_metric_table(
                    table,
                    comparison=f"{tracker}_oca_pressure",
                    tracker=tracker,
                    comparison_source="oca_pressure",
                    metric_set="all_points",
                    subset_masks=raw_masks,
                )
            )
            qc_metrics_list.append(
                compute_metric_table(
                    table,
                    comparison=f"{tracker}_oca_pressure",
                    tracker=tracker,
                    comparison_source="oca_pressure",
                    metric_set="qc_qi60_speed5",
                    subset_masks=qc_masks,
                )
            )
            comparison_tables_for_debug[f"{tracker}_oca_pressure"] = table
        else:
            print("Missing OCA-P product or cache, skipping:", oca_path)

    if RUN_OFFICIAL_PRESSURE_PRODUCTS:
        official_p_path = PRESSURE_ASSIGNMENT_DIR / f"{tracker}_final_amv_official_pressure.npz"
        if official_p_path.exists() and OFFICIAL_ERA5_CACHE.exists():
            print("\n" + "=" * 90)
            print(f"Comparing {tracker}_official_pressure vs ERA5")
            table = build_pressure_amv_vs_era5_table(
                official_p_path,
                OFFICIAL_ERA5_CACHE,
                join_mode="official_id",
                amv_source=f"{tracker}_official_pressure",
            )
            print("Rows after joining ERA5 cache:", len(table))

            raw_masks = build_raw_subset_masks_for_metrics(table)
            qc_masks = build_qc_subset_masks_for_metrics(
                table,
                qi_min=QC_QI_MIN,
                speed_min=QC_SPEED_MIN,
                speed_max=QC_SPEED_MAX,
                pressure_range=QC_PRESSURE_RANGE,
                cloud_probability_min=None,
            )

            raw_metrics_list.append(
                compute_metric_table(
                    table,
                    comparison=f"{tracker}_official_pressure",
                    tracker=tracker,
                    comparison_source="official_pressure",
                    metric_set="all_points",
                    subset_masks=raw_masks,
                )
            )
            qc_metrics_list.append(
                compute_metric_table(
                    table,
                    comparison=f"{tracker}_official_pressure",
                    tracker=tracker,
                    comparison_source="official_pressure",
                    metric_set="qc_qi60_speed5",
                    subset_masks=qc_masks,
                )
            )
            comparison_tables_for_debug[f"{tracker}_official_pressure"] = table
        else:
            print("Missing official-P product or cache, skipping:", official_p_path)

if raw_metrics_list:
    metrics_by_subset_raw = pd.concat(raw_metrics_list, ignore_index=True)
else:
    metrics_by_subset_raw = pd.DataFrame()

if qc_metrics_list:
    metrics_by_subset_qc = pd.concat(qc_metrics_list, ignore_index=True)
else:
    metrics_by_subset_qc = pd.DataFrame()

if not metrics_by_subset_raw.empty and not metrics_by_subset_qc.empty:
    metrics_by_subset_raw, metrics_by_subset_qc = attach_qc_retention_fraction(
        metrics_by_subset_raw,
        metrics_by_subset_qc,
    )

metrics_by_subset = pd.concat(
    [metrics_by_subset_raw, metrics_by_subset_qc],
    ignore_index=True,
) if (not metrics_by_subset_raw.empty or not metrics_by_subset_qc.empty) else pd.DataFrame()

# raw_metrics_path = METRICS_DIR / "metrics_by_source_subset_all_points.csv"
# qc_metrics_path = METRICS_DIR / "metrics_by_source_subset_qc_qi60_speed5.csv"
combined_metrics_path = METRICS_DIR / "metrics.csv"

# metrics_by_subset_raw.to_csv(raw_metrics_path, index=False)
# metrics_by_subset_qc.to_csv(qc_metrics_path, index=False)
metrics_by_subset.to_csv(combined_metrics_path, index=False)

# print("Saved raw metrics:", raw_metrics_path)
# print("Saved QC metrics:", qc_metrics_path)
print("Saved combined metrics:", combined_metrics_path)

print("\nRaw metrics:")
display(metrics_by_subset_raw)

print("\nQC metrics with retention fraction:")
display(metrics_by_subset_qc)


# %%
# 7. Plot ERA5 comparison metrics with pressure-layer grouped bars
# Two figures are produced:
#   1. all valid points
#   2. QC points: QI >= 60 and speed >= 5 m/s
#
# Each figure has 4 subplots:
#   mean absolute speed error
#   mean vector difference
#   mean absolute direction difference
#   speed correlation
#
# In each subplot:
#   x-axis = method
#   bar colors = all / high / middle / low pressure subsets
#   number above each bar = sample_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------
# Load metrics if variables are not already in memory
# ---------------------------------------------------------------------
ALL_POINTS_METRICS_CSV = METRICS_DIR / "metrics_by_source_subset_all_points.csv"
QC_METRICS_CSV = METRICS_DIR / "metrics_by_source_subset_qc_qi60_speed5.csv"
QC_RETENTION_CSV = METRICS_DIR / "metrics_qc_retention_summary.csv"

if "metrics_by_subset_raw" in globals() and not metrics_by_subset_raw.empty:
    metrics_all_points_for_plot = metrics_by_subset_raw.copy()
elif "metrics_by_subset_all_points" in globals() and not metrics_by_subset_all_points.empty:
    metrics_all_points_for_plot = metrics_by_subset_all_points.copy()
elif ALL_POINTS_METRICS_CSV.exists():
    metrics_all_points_for_plot = pd.read_csv(ALL_POINTS_METRICS_CSV)
else:
    raise FileNotFoundError(f"Missing all-points metrics CSV: {ALL_POINTS_METRICS_CSV}")

if "metrics_by_subset_qc" in globals() and not metrics_by_subset_qc.empty:
    metrics_qc_for_plot = metrics_by_subset_qc.copy()
elif QC_METRICS_CSV.exists():
    metrics_qc_for_plot = pd.read_csv(QC_METRICS_CSV)
else:
    raise FileNotFoundError(f"Missing QC metrics CSV: {QC_METRICS_CSV}")

print("All-points subsets:", metrics_all_points_for_plot["subset"].unique())
print("QC subsets:", metrics_qc_for_plot["subset"].unique())

# ---------------------------------------------------------------------
# Plot settings
# ---------------------------------------------------------------------
METRIC_FIGURE_DPI = 180
METRIC_FIGURE_FONT_SIZE = 15
METRIC_FIGURE_TITLE_SIZE = 18
METRIC_FIGURE_COUNT_SIZE = 8
METRIC_FIGURE_XTICK_SIZE = 10
METRIC_FIGURE_LEGEND_SIZE = 13

# 4 metrics = 4 subplots
METRICS_4PANEL = [
    ("mean_absolute_speed_error", "Mean abs. speed error\n(m s$^{-1}$)"),
    ("mean_vector_difference", "Mean vector difference\n(m s$^{-1}$)"),
    ("mean_absolute_direction_difference", "Mean abs. direction diff.\n(deg)"),
    ("speed_correlation", "Speed correlation\nwith ERA5"),
]

# Important:
# Both CSVs use the same subset names.
# The QC CSV is already computed after QC filtering, so its subset names are still:
#   all_valid / pressure_high_lt400 / pressure_mid_400_700 / pressure_low_gt700
LAYER_SUBSETS_RAW = [
    ("all_valid", "All valid"),
    ("pressure_high_lt400", "High: p < 400 hPa"),
    ("pressure_mid_400_700", "Middle: 400–700 hPa"),
    ("pressure_low_gt700", "Low: p ≥ 700 hPa"),
]

LAYER_SUBSETS_QC = [
    ("all_valid", "All QC"),
    ("pressure_high_lt400", "QC high: p < 400 hPa"),
    ("pressure_mid_400_700", "QC middle: 400–700 hPa"),
    ("pressure_low_gt700", "QC low: p ≥ 700 hPa"),
]

LAYER_COLORS = {
    "All valid": "0.55",
    "All QC": "0.55",
    "High: p < 400 hPa": "tab:blue",
    "QC high: p < 400 hPa": "tab:blue",
    "Middle: 400–700 hPa": "tab:orange",
    "QC middle: 400–700 hPa": "tab:orange",
    "Low: p ≥ 700 hPa": "tab:green",
    "QC low: p ≥ 700 hPa": "tab:green",
}


def _clean_method_name(row):
    """Build short method labels for plotting."""
    source = str(row.get("comparison_source", ""))
    tracker = str(row.get("tracker", ""))
    comparison = str(row.get("comparison", ""))

    if source == "official_amv" or tracker == "official":
        return "Official AMV"

    if "layerwise_soft" in tracker and source == "oca_pressure":
        return "Layerwise-soft\nOCA-P"

    if "hybrid_midlow_layerwise" in tracker and source == "oca_pressure":
        return "Hybrid-midlow\nOCA-P"

    if tracker == "farneback" and source == "oca_pressure":
        return "Farneback\nOCA-P"

    if "layerwise_soft" in tracker and source == "official_pressure":
        return "Layerwise-soft\nofficial-P"

    if "hybrid_midlow_layerwise" in tracker and source == "official_pressure":
        return "Hybrid-midlow\nofficial-P"

    if tracker == "farneback" and source == "official_pressure":
        return "Farneback\nofficial-P"

    # fallback
    label = comparison or tracker or source
    label = label.replace("_oca_pressure", "\nOCA-P")
    label = label.replace("_official_pressure", "\nofficial-P")
    label = label.replace("farneback", "Farneback")
    label = label.replace("layerwise_soft", "Layerwise-soft")
    label = label.replace("hybrid_midlow_layerwise", "Hybrid-midlow")
    return label


def _build_method_order(metrics):
    """Use all_valid rows to define method order."""
    if "subset" not in metrics.columns:
        raise KeyError("metrics must contain a 'subset' column.")

    if metrics["subset"].eq("all_valid").any():
        base = metrics.loc[metrics["subset"].eq("all_valid")].copy()
    else:
        base = metrics.copy()

    labels = []
    for _, row in base.iterrows():
        label = _clean_method_name(row)
        if label not in labels:
            labels.append(label)

    return labels


def _metric_value(metrics, method_label, subset_name, metric):
    tmp = metrics.loc[
        (metrics["method_label"].eq(method_label)) &
        (metrics["subset"].eq(subset_name))
    ]

    if tmp.empty or metric not in tmp.columns:
        return np.nan, np.nan

    value = pd.to_numeric(tmp.iloc[0][metric], errors="coerce")
    count = pd.to_numeric(tmp.iloc[0].get("sample_count", np.nan), errors="coerce")
    return float(value), float(count)




# ---------------------------------------------------------------------
# Save the two requested figures
# ---------------------------------------------------------------------
figure_paths = []


print("Figure paths:")
for path in figure_paths:
    print(" ", path)

# ---------------------------------------------------------------------
# Optional: display QC retention summary
# ---------------------------------------------------------------------
if QC_RETENTION_CSV.exists():
    qc_retention_summary = pd.read_csv(QC_RETENTION_CSV)
    print("QC retention summary:")
    display(qc_retention_summary)


# %%
# 8. Cache summary

for cache_path in (OFFICIAL_ERA5_CACHE, OCA_ERA5_CACHE):
    print("\n" + "=" * 90)
    print(cache_path)
    if not cache_path.exists():
        print("missing")
        continue
    with np.load(cache_path, allow_pickle=False) as data:
        for key in data.files:
            arr = data[key]
            print(f"{key:<32} shape={arr.shape}, dtype={arr.dtype}")


# %%
# # 9. Overall ERA5 metric table without pressure-layer splitting
# # This table contains:
# #   first block: all valid points
# #   blank row
# #   second block: QC points
# #
# # QC condition is defined in Cell 6:
# #   QI >= 60, speed >= 5 m/s, speed <= 150 m/s, pressure in 100-1000 hPa

# import numpy as np
# import pandas as pd
# from pathlib import Path

# OVERALL_TABLE_COLUMNS = [
#     "tracker",
#     "comparison_source",
#     "sample_count",
#     "mean_absolute_speed_error",
#     "mean_vector_difference",
#     "vector_difference_rmse",
#     "mean_absolute_direction_difference",
#     "speed_correlation",
#     "u_bias_amv_minus_era5",
#     "v_bias_amv_minus_era5",
#     "u_correlation",
#     "v_correlation",
#     "retained_fraction",
# ]

# ALL_POINTS_METRICS_CSV = METRICS_DIR / "metrics_by_source_subset_all_points.csv"
# QC_METRICS_CSV = METRICS_DIR / "metrics_by_source_subset_qc_qi60_speed5.csv"

# if "metrics_by_subset_all_points" not in globals():
#     metrics_by_subset_all_points = pd.read_csv(ALL_POINTS_METRICS_CSV)

# if "metrics_by_subset_qc" not in globals():
#     metrics_by_subset_qc = pd.read_csv(QC_METRICS_CSV)


# def _overall_method_sort_key(row):
#     tracker = str(row.get("tracker", ""))
#     source = str(row.get("comparison_source", ""))

#     # official AMV first
#     if source == "official_amv" or tracker == "official":
#         return (0, 0, tracker)

#     # for each tracker, official-P first, then OCA-P
#     source_order = {
#         "official_pressure": 1,
#         "oca_pressure": 2,
#     }.get(source, 9)

#     tracker_order = {
#         "farneback": 0,
#         "farneback_layerwise_soft": 1,
#         "farneback_hybrid_midlow_layerwise": 2,
#     }.get(tracker, 99)

#     return (tracker_order + 1, source_order, tracker)


# def _make_overall_metric_table(metrics, *, is_qc):
#     """Use only all_valid subset; no pressure-layer splitting."""
#     table = metrics.loc[metrics["subset"].eq("all_valid")].copy()

#     if table.empty:
#         raise ValueError("No all_valid subset found in metric table.")

#     table["_sort_key"] = table.apply(_overall_method_sort_key, axis=1)
#     table = table.sort_values("_sort_key").drop(columns=["_sort_key"])

#     for col in OVERALL_TABLE_COLUMNS:
#         if col not in table.columns:
#             table[col] = np.nan

#     table = table[OVERALL_TABLE_COLUMNS].copy()

#     # retained_fraction only makes sense for QC table
#     if not is_qc:
#         table["retained_fraction"] = np.nan

#     return table


# metrics_overall_all_points = _make_overall_metric_table(
#     metrics_by_subset_all_points,
#     is_qc=False,
# )

# metrics_overall_qc_qi60_speed5 = _make_overall_metric_table(
#     metrics_by_subset_qc,
#     is_qc=True,
# )

# # combined table with one blank separator row
# blank_row = pd.DataFrame([{col: "" for col in OVERALL_TABLE_COLUMNS}])
# metrics_overall_all_and_qc = pd.concat(
#     [
#         metrics_overall_all_points,
#         blank_row,
#         metrics_overall_qc_qi60_speed5,
#     ],
#     ignore_index=True,
# )

# # Save tables
# overall_all_path = METRICS_DIR / "metrics_overall_all_points.csv"
# overall_qc_path = METRICS_DIR / "metrics_overall_qc_qi60_speed5.csv"
# overall_combined_path = METRICS_DIR / "metrics_overall_all_and_qc_table.csv"

# metrics_overall_all_points.to_csv(overall_all_path, index=False)
# metrics_overall_qc_qi60_speed5.to_csv(overall_qc_path, index=False)
# metrics_overall_all_and_qc.to_csv(overall_combined_path, index=False)

# print("Saved:", overall_all_path)
# print("Saved:", overall_qc_path)
# print("Saved:", overall_combined_path)

# # Display a readable table in notebook
# display_columns = OVERALL_TABLE_COLUMNS

# formatters = {
#     "sample_count": lambda x: "" if x == "" or pd.isna(x) else f"{int(float(x))}",
#     "mean_absolute_speed_error": lambda x: "" if x == "" or pd.isna(x) else f"{float(x):.3f}",
#     "mean_vector_difference": lambda x: "" if x == "" or pd.isna(x) else f"{float(x):.3f}",
#     "vector_difference_rmse": lambda x: "" if x == "" or pd.isna(x) else f"{float(x):.3f}",
#     "mean_absolute_direction_difference": lambda x: "" if x == "" or pd.isna(x) else f"{float(x):.3f}",
#     "speed_correlation": lambda x: "" if x == "" or pd.isna(x) else f"{float(x):.3f}",
#     "u_bias_amv_minus_era5": lambda x: "" if x == "" or pd.isna(x) else f"{float(x):.3f}",
#     "v_bias_amv_minus_era5": lambda x: "" if x == "" or pd.isna(x) else f"{float(x):.3f}",
#     "u_correlation": lambda x: "" if x == "" or pd.isna(x) else f"{float(x):.3f}",
#     "v_correlation": lambda x: "" if x == "" or pd.isna(x) else f"{float(x):.3f}",
#     "retained_fraction": lambda x: "" if x == "" or pd.isna(x) else f"{float(x):.3f}",
# }

# display(metrics_overall_all_and_qc[display_columns].style.format(formatters))
