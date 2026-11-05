"""Collocate SEVIRI AMVs with IGRA radiosondes and report wind metrics.

Read station archives from data/igra and reuse point-level collocation
geometry across trackers and QI thresholds."""

from __future__ import annotations

import os

from pathlib import Path
import sys
import time
import json
import hashlib
import pickle

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print("PROJECT_ROOT:", PROJECT_ROOT)
print("SRC_DIR:", SRC_DIR)


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
CASE_NAME = os.environ.get("AMV_CASE", "test_2018-01-12T11")
from runtime_options import read_case_inputs
CASE_INPUTS = read_case_inputs(PROJECT_ROOT, CASE_NAME)

DATA_CASE_NAME = CASE_NAME.removeprefix("test_")
DATA_CASE_DIR = Path(CASE_INPUTS.get("data_dir", PROJECT_ROOT / "data" / DATA_CASE_NAME))

# Process one or multiple trackers.  These names should match the output folders
# created by the tracking/pressure-assignment pipeline.
TRACKER_NAMES = ['raw_farneback_opencv', 'raw_deepflow_opencv', 'raw_dis', 'raw_tvl1', 'raw_dualtvl1', 'raw_ilk', 'raw_pcaflow', 'layerwise_farneback_opencv', 'layerwise_deepflow_opencv', 'layerwise_dis', 'layerwise_tvl1', 'layerwise_dualtvl1', 'layerwise_ilk', 'layerwise_pcaflow', 'nn_raw_flowdiffuser', 'nn_layerwise_flowdiffuser', 'nn_raw_windflow_inv', 'nn_layerwise_windflow_inv', 'pressure_conditioned_flow']

OUTPUT_DIR = PROJECT_ROOT / "outputs" / CASE_NAME
PRESSURE_ASSIGNMENT_DIR = OUTPUT_DIR / "pressure_assignment"
COMPARISON_DIR = OUTPUT_DIR / "comparison" / "amv_vs_igra"
IGRA_PROFILE_CACHE_DIR = COMPARISON_DIR / "profile_cache"
GEOMETRY_CACHE_DIR = COMPARISON_DIR / "geometry_cache"
COLLOCATION_DIR = COMPARISON_DIR / "igra_collocations"
METRICS_DIR = COMPARISON_DIR / "metrics"
FIGURE_DIR = COMPARISON_DIR / "figure"

for directory in [IGRA_PROFILE_CACHE_DIR, GEOMETRY_CACHE_DIR, COLLOCATION_DIR, METRICS_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Global IGRA raw-data cache.  This should contain:
#   data/igra/igra2-station-list.txt
#   data/igra/stations/*-data.txt.zip
IGRA_ROOT = PROJECT_ROOT / "data" / "igra"
AOI = None

RUN_OFFICIAL_AMV = True
RUN_OCA_PRESSURE_PRODUCTS = True
RUN_OFFICIAL_PRESSURE_PRODUCTS = False  # Use cloud-product pressure for evaluation.
SAVE_METRICS = True
SAVE_FIGURES = False  # Result visualization uses dedicated plot_* scripts.

# AMV time convention: for a case starting at HH:00 with frames at HH:00,
# HH:15, HH:30, HH:45, validate the AMV at HH:45.
AMV_VALID_OFFSET_MINUTES = 45

# IGRA collocation settings.
MAX_DISTANCE_KM = 100.0
# Time criterion.  "level": per-level observation time = reported release time
# + elapsed time since launch (Seidel et al., 2011 recommend observed ascent
# times); levels without RELTIME fall back to the nominal sounding time.
# "nominal": the original behaviour (nominal 00/12 UTC sounding time).
IGRA_TIME_REFERENCE = "level"
MAX_TIME_MINUTES = 60.0
# Balloon drift: measure distances from the drifted balloon position at each
# sonde level (layer-mean wind x layer time; Laroche and Sarrazin, 2013;
# Voggenberger et al., 2024).  False keeps the launch-site position.
USE_BALLOON_DRIFT = True
PRESSURE_WINDOW_HPA = 25.0
IGRA_PROFILE_LOAD_TIME_WINDOW_HOURS = 3.0
MIN_IGRA_LEVELS = 5
IGRA_AGGREGATION = "distance_weighted"  # "distance_weighted" or "mean"
IGRA_DISTANCE_WEIGHT_SIGMA_KM = 50.0
MIN_AMV_POINTS_PER_LEVEL_OFFICIAL = 1
MIN_AMV_POINTS_PER_LEVEL_OCA = 3
MIN_AMV_POINTS_PER_LEVEL_OFFICIAL_PRESSURE = 1

# Cache controls.
REUSE_IGRA_PROFILE_CACHE = True
SAVE_IGRA_PROFILE_CACHE = True
REUSE_GEOMETRY_CACHE = True
SAVE_GEOMETRY_CACHE = True
REUSE_COLLOCATION_CACHE = True
SAVE_COLLOCATION_CACHE = True
SAVE_COMPRESSED_NPZ = False

# QC settings; keep consistent with the ERA5 comparison workflow.
QC_QI_MIN = 60.0
QC_SPEED_MIN = 5.0
QC_SPEED_MAX = 150.0
QC_PRESSURE_RANGE = (100.0, 1000.0)
QC_OCA_VALID_COMPONENT_COUNT_MIN = 2
QC_OFFICIAL_NEIGHBOR_COUNT_MIN = 3
QC_METRIC_SET_NAME = "qc_qi60_speed5"


# -----------------------------------------------------------------------------
# Imports that depend on PROJECT_ROOT/SRC_DIR
# -----------------------------------------------------------------------------
from data_io.official_amv import load_official_amv_file
from data_io.igra import PARSER_VERSION as IGRA_PARSER_VERSION, load_igra_profiles, profiles_to_dataframe
from data_io.product_discovery import (
    find_official_amv_file,
    infer_amv_valid_time_from_path,
)
from validation.compare_igra import (
    standardize_official_amv,
    load_pressure_amv_npz,
    inspect_npz_product,
    compare_to_igra_metrics,
    save_collocation_npz,
    load_collocation_npz,
    read_collocation_npz_metadata,
    _cache_metadata_matches,
    amv_geometry_fingerprint,
    geometry_or_load,
    collocate_from_igra_geometry,
)


# -----------------------------------------------------------------------------
# Resolved case time and official AMV file
# -----------------------------------------------------------------------------
AMV_VALID_TIME = pd.Timestamp(
    CASE_INPUTS["frame_times"][-1] if CASE_INPUTS.get("frame_times") else
    infer_amv_valid_time_from_path(DATA_CASE_DIR, offset_minutes=AMV_VALID_OFFSET_MINUTES),
    tz="UTC",
)
OFFICIAL_AMV_NATIVE_DIR = DATA_CASE_DIR / "eumetsat_amv" / "native"
OFFICIAL_AMV_FILE = find_official_amv_file(
    OFFICIAL_AMV_NATIVE_DIR,
    expected_time=AMV_VALID_TIME.to_pydatetime().replace(tzinfo=None),
)

print("CASE_NAME:", CASE_NAME)
print("DATA_CASE_DIR:", DATA_CASE_DIR, "exists=", DATA_CASE_DIR.exists())
print("AMV_VALID_TIME:", AMV_VALID_TIME)
print("IGRA_ROOT:", IGRA_ROOT, "exists=", IGRA_ROOT.exists())
print("OFFICIAL_AMV_FILE:", OFFICIAL_AMV_FILE, "exists=", OFFICIAL_AMV_FILE.exists())
print("PRESSURE_ASSIGNMENT_DIR:", PRESSURE_ASSIGNMENT_DIR, "exists=", PRESSURE_ASSIGNMENT_DIR.exists())


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def _normalise_quality_0_100(values) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce").astype(float)
    finite = np.isfinite(out)
    if finite.any() and float(np.nanmedian(out[finite])) <= 1.5:
        out = out * 100.0
    return out


# Self-supervised QI (scripts/run_quality_control.py): attach it to dense
# products so the QC subset below uses QI >= QC_QI_MIN (top 40 % per case).
USE_SELFSUP_QI = True


def attach_selfsup_qi(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Merge quality_index from qc_selfsup/<name>/<name>_selfsup_qc.npz by (row, col)."""

    if not USE_SELFSUP_QI or frame.empty:
        return frame
    path = OUTPUT_DIR / "qc_selfsup" / name / f"{name}_selfsup_qc.npz"
    if not path.exists():
        print(f"  [selfsup QI] missing {path}; QC subset uses the product's own columns")
        return frame
    with np.load(path, allow_pickle=False) as data:
        qi = pd.Series(
            np.asarray(data["quality_index"], dtype=float),
            index=pd.MultiIndex.from_arrays([
                np.asarray(data["row_center_original"], dtype=np.int64),
                np.asarray(data["col_center_original"], dtype=np.int64),
            ]),
        )
    qi = qi[~qi.index.duplicated()]
    key = pd.MultiIndex.from_arrays([
        np.rint(pd.to_numeric(frame["row_center_original"])).astype(np.int64),
        np.rint(pd.to_numeric(frame["col_center_original"])).astype(np.int64),
    ])
    out = frame.copy()
    out["quality_index"] = qi.reindex(key).to_numpy()
    hit = np.isfinite(out["quality_index"]).mean()
    print(f"  [selfsup QI] {path.name}: matched {hit:.1%} of {len(out):,} vectors")
    return out


def filter_amv_for_igra_qc(
    frame: pd.DataFrame,
    *,
    source: str,
    qi_min: float = QC_QI_MIN,
    speed_min: float = QC_SPEED_MIN,
    speed_max: float = QC_SPEED_MAX,
    pressure_range: tuple[float, float] = QC_PRESSURE_RANGE,
    oca_valid_component_count_min: int = QC_OCA_VALID_COMPONENT_COUNT_MIN,
    official_neighbor_count_min: int = QC_OFFICIAL_NEIGHBOR_COUNT_MIN,
) -> pd.DataFrame:
    """Apply the same broad QC logic used in the ERA5 comparison."""

    table = frame.copy()
    mask = pd.Series(True, index=table.index)

    for column in ["longitude", "latitude", "pressure_hpa", "u_amv", "v_amv"]:
        if column in table.columns:
            mask &= np.isfinite(pd.to_numeric(table[column], errors="coerce"))

    p = pd.to_numeric(table["pressure_hpa"], errors="coerce")
    mask &= p.ge(float(pressure_range[0])) & p.le(float(pressure_range[1]))

    if "speed_amv" in table.columns:
        speed = pd.to_numeric(table["speed_amv"], errors="coerce")
    else:
        speed = np.hypot(pd.to_numeric(table["u_amv"], errors="coerce"), pd.to_numeric(table["v_amv"], errors="coerce"))
    mask &= speed.ge(float(speed_min)) & speed.le(float(speed_max))

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
        q = _normalise_quality_0_100(table[q_col])
        mask &= q.ge(float(qi_min))
        finite_q = q[np.isfinite(q)]
        if finite_q.size:
            print(
                f"  QC quality column used for {source}: {q_col}; "
                f"range={float(np.nanmin(finite_q)):.2f}–{float(np.nanmax(finite_q)):.2f}; "
                f"threshold={qi_min:.1f}"
            )
    else:
        print(f"  QC quality column used for {source}: None; QI filter not applied")

    if source == "oca_pressure":
        comp_col = _first_existing_column(
            table,
            [
                "valid_component_count",
                "pressure_component_count",
                "valid_pressure_component_count",
            ],
        )
        if comp_col is not None:
            mask &= pd.to_numeric(table[comp_col], errors="coerce").ge(int(oca_valid_component_count_min))
            print(f"  QC OCA component column used: {comp_col}; threshold={oca_valid_component_count_min}")
    elif source == "official_pressure":
        count_col = _first_existing_column(table, ["piv_neighbor_count", "neighbor_count"])
        if count_col is not None:
            mask &= pd.to_numeric(table[count_col], errors="coerce").ge(int(official_neighbor_count_min))
            print(f"  QC official-pressure neighbor column used: {count_col}; threshold={official_neighbor_count_min}")

    return table.loc[mask.fillna(False)].reset_index(drop=True)


def _igra_profile_cache_key() -> str:
    return json.dumps(
        {
            "igra_root": str(IGRA_ROOT),
            "target_time": AMV_VALID_TIME.isoformat(),
            "time_window_hours": IGRA_PROFILE_LOAD_TIME_WINDOW_HOURS,
            "minimum_levels": MIN_IGRA_LEVELS,
            "aoi": AOI,
            # Profiles now carry per-level time/drift; rebuild caches from older parsers.
            "parser_version": IGRA_PARSER_VERSION,
        },
        sort_keys=True,
        default=str,
    )


PROFILE_SET_ID = hashlib.md5(_igra_profile_cache_key().encode("utf-8")).hexdigest()[:12]


def _diagnose_no_profiles() -> None:
    """Explain why no radiosonde profile was found, then stop (no empty metrics)."""

    station_list = IGRA_ROOT / "igra2-station-list.txt"
    archives = sorted((IGRA_ROOT / "stations").glob("*-data.txt.zip"))
    print("\n" + "!" * 90)
    print("No IGRA profile found -- stopping before writing empty metrics.")
    print("  IGRA_ROOT:", IGRA_ROOT, "| station list exists:", station_list.exists(),
          "| station archives:", len(archives))
    print("  target time:", AMV_VALID_TIME, "+/-", IGRA_PROFILE_LOAD_TIME_WINDOW_HOURS, "h;",
          "minimum wind levels:", MIN_IGRA_LEVELS)
    day = AMV_VALID_TIME.strftime("%Y %m %d")
    from zipfile import ZipFile
    for archive in archives[:5]:
        try:
            with ZipFile(archive) as z, z.open(z.namelist()[0]) as h:
                headers = [line.decode("ascii", "replace")[:40] for line in h if line.startswith(b"#")]
            hits = [x for x in headers if x[13:23] == day]
            print(f"  {archive.name}: last header {headers[-1] if headers else '-'}; soundings on {day}: {hits}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {archive.name}: unreadable ({type(exc).__name__}: {exc})")
    print("!" * 90)
    raise SystemExit(1)


def load_profiles_with_cache():
    key = _igra_profile_cache_key()
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    cache_path = IGRA_PROFILE_CACHE_DIR / f"igra_profiles_{digest}.pkl"

    if REUSE_IGRA_PROFILE_CACHE and cache_path.exists():
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("profile_cache_key") != key:
            raise RuntimeError(f"IGRA profile cache key mismatch: {cache_path}")
        if payload["profiles"]:
            print("Loaded IGRA profiles from cache:", len(payload["profiles"]), cache_path)
            return payload["profiles"], cache_path
        print("Ignoring EMPTY IGRA profile cache (left by an earlier run):", cache_path)

    config = {
        "paths": {"igra": str(IGRA_ROOT)},
        "target_time": AMV_VALID_TIME.isoformat(),
        "aoi": AOI,
    }
    profiles = load_igra_profiles(
        config,
        aoi=AOI,
        target_time=AMV_VALID_TIME,
        time_window_hours=IGRA_PROFILE_LOAD_TIME_WINDOW_HOURS,
        minimum_levels=MIN_IGRA_LEVELS,
        root=IGRA_ROOT,
    )
    print("IGRA profiles loaded from source:", len(profiles))
    if not profiles:
        _diagnose_no_profiles()
    if SAVE_IGRA_PROFILE_CACHE and profiles:
        with cache_path.open("wb") as handle:
            pickle.dump({"profile_cache_key": key, "profiles": profiles}, handle)
        print("Saved IGRA profile cache:", cache_path)
    return profiles, cache_path


def _safe_name(text: str) -> str:
    import re

    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return value or "product"


def aggregate_or_load_from_geometry(
    *,
    cache_path: Path,
    frame: pd.DataFrame,
    geometry: pd.DataFrame,
    source: str,
    min_amv_points_per_level: int,
    metric_set: str,
    geometry_fingerprint: str,
) -> pd.DataFrame:
    product_fingerprint = amv_geometry_fingerprint(frame)
    metadata = {
        "cache_type": "igra_station_level_collocation_from_geometry",
        "source": source,
        "metric_set": metric_set,
        "product_time": str(AMV_VALID_TIME),
        "product_geometry_fingerprint": product_fingerprint,
        "geometry_fingerprint": geometry_fingerprint,
        "min_amv_points_per_level": int(min_amv_points_per_level),
        "max_distance_km": float(MAX_DISTANCE_KM),
        "max_time_minutes": float(MAX_TIME_MINUTES),
        "use_balloon_drift": bool(USE_BALLOON_DRIFT),
        "time_reference": str(IGRA_TIME_REFERENCE),
        "pressure_window_hpa": float(PRESSURE_WINDOW_HPA),
        "aggregation": IGRA_AGGREGATION,
        "distance_weight_sigma_km": float(IGRA_DISTANCE_WEIGHT_SIGMA_KM),
        "profile_set": PROFILE_SET_ID,
    }
    if REUSE_COLLOCATION_CACHE and cache_path.exists():
        cached_metadata = read_collocation_npz_metadata(cache_path)
        if _cache_metadata_matches(cached_metadata, metadata):
            return load_collocation_npz(cache_path)
        print("Ignoring stale IGRA collocation cache:", cache_path)

    table = collocate_from_igra_geometry(
        amv=frame,
        geometry=geometry,
        source=source,
        min_amv_points_per_level=min_amv_points_per_level,
    )
    if SAVE_COLLOCATION_CACHE and not table.empty:
        save_collocation_npz(table, cache_path, metadata=metadata, compressed=SAVE_COMPRESSED_NPZ)
    return table


def compute_metric_table(
    table: pd.DataFrame,
    *,
    comparison: str,
    tracker: str,
    comparison_source: str,
    metric_set: str,
) -> pd.DataFrame:
    metrics = compare_to_igra_metrics(table, comparison=comparison)
    if metrics.empty:
        return metrics
    metrics.insert(0, "metric_set", metric_set)
    metrics.insert(0, "comparison_source", comparison_source)
    metrics.insert(0, "tracker", tracker)

    # ERA5-like aliases while keeping original sonde-specific names.
    if "u_bias_amv_minus_sonde" in metrics.columns:
        metrics["u_bias_amv_minus_igra"] = metrics["u_bias_amv_minus_sonde"]
    if "v_bias_amv_minus_sonde" in metrics.columns:
        metrics["v_bias_amv_minus_igra"] = metrics["v_bias_amv_minus_sonde"]
    if "speed_bias_amv_minus_sonde" in metrics.columns:
        metrics["speed_bias_amv_minus_igra"] = metrics["speed_bias_amv_minus_sonde"]
    return metrics


def attach_qc_retention_fraction(raw_metrics: pd.DataFrame, qc_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = raw_metrics.copy()
    qc = qc_metrics.copy()
    if raw.empty or qc.empty:
        return raw, qc
    id_cols = ["tracker", "comparison_source", "comparison", "subset"]
    raw_counts = raw[id_cols + ["sample_count"]].rename(columns={"sample_count": "raw_sample_count"})
    qc = qc.merge(raw_counts, on=id_cols, how="left")
    qc["retained_fraction"] = qc["sample_count"] / qc["raw_sample_count"]
    raw["raw_sample_count"] = raw["sample_count"]
    raw["retained_fraction"] = 1.0
    return raw, qc


def load_official_frame() -> pd.DataFrame:
    official = load_official_amv_file(OFFICIAL_AMV_FILE, aoi=AOI)
    frame = standardize_official_amv(official)
    print("Official AMV vectors:", len(frame))
    return frame


def process_product(
    *,
    raw_frame: pd.DataFrame,
    qc_frame: pd.DataFrame,
    geometry_frame: pd.DataFrame | None = None,
    source_name: str,
    comparison: str,
    tracker: str,
    comparison_source: str,
    min_points: int,
    geometry_cache_path: Path,
    geometry_source: str,
    profiles: list,
    raw_metrics_list: list[pd.DataFrame],
    qc_metrics_list: list[pd.DataFrame],
    collocation_tables: dict[str, pd.DataFrame],
) -> None:
    if raw_frame.empty:
        print("Empty AMV frame, skip:", comparison)
        return

    if geometry_frame is None:
        geometry_frame = raw_frame

    t0 = time.perf_counter()
    geometry = geometry_or_load(
        cache_path=geometry_cache_path,
        amv=geometry_frame,
        igra_profiles=profiles,
        source=geometry_source,
        product_time=AMV_VALID_TIME,
        reuse_cache=REUSE_GEOMETRY_CACHE,
        save_cache=SAVE_GEOMETRY_CACHE,
        compressed=SAVE_COMPRESSED_NPZ,
        max_distance_km=MAX_DISTANCE_KM,
        max_time_minutes=MAX_TIME_MINUTES,
        use_balloon_drift=USE_BALLOON_DRIFT,
        time_reference=IGRA_TIME_REFERENCE,
        pressure_window_hpa=PRESSURE_WINDOW_HPA,
        aggregation=IGRA_AGGREGATION,
        distance_weight_sigma_km=IGRA_DISTANCE_WEIGHT_SIGMA_KM,
    )
    geometry_fp = amv_geometry_fingerprint(geometry_frame)
    print(
        f"Geometry ready for {comparison}: rows={len(geometry):,}, "
        f"time={time.perf_counter() - t0:.1f} s"
    )

    raw_cache = COLLOCATION_DIR / f"{_safe_name(comparison)}_all_points_collocated.npz"
    raw_table = aggregate_or_load_from_geometry(
        cache_path=raw_cache,
        frame=raw_frame,
        geometry=geometry,
        source=source_name,
        min_amv_points_per_level=min_points,
        metric_set="all_points",
        geometry_fingerprint=geometry_fp,
    )
    collocation_tables[f"{comparison}__all_points"] = raw_table
    raw_metrics_list.append(
        compute_metric_table(
            raw_table,
            comparison=comparison,
            tracker=tracker,
            comparison_source=comparison_source,
            metric_set="all_points",
        )
    )
    print(f"  all_points collocation rows: {len(raw_table):,}")

    qc_cache = COLLOCATION_DIR / f"{_safe_name(comparison)}_{QC_METRIC_SET_NAME}_collocated.npz"
    qc_table = aggregate_or_load_from_geometry(
        cache_path=qc_cache,
        frame=qc_frame,
        geometry=geometry,
        source=source_name,
        min_amv_points_per_level=min_points,
        metric_set=QC_METRIC_SET_NAME,
        geometry_fingerprint=geometry_fp,
    )
    collocation_tables[f"{comparison}__{QC_METRIC_SET_NAME}"] = qc_table
    qc_metrics_list.append(
        compute_metric_table(
            qc_table,
            comparison=comparison,
            tracker=tracker,
            comparison_source=comparison_source,
            metric_set=QC_METRIC_SET_NAME,
        )
    )
    print(f"  {QC_METRIC_SET_NAME} collocation rows: {len(qc_table):,}")


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
print("\nLoading IGRA profiles...")
profiles, profile_cache_path = load_profiles_with_cache()
print("IGRA profiles available:", len(profiles))

if profiles:
    profile_df = profiles_to_dataframe(profiles)
    station_summary_path = METRICS_DIR / "igra_profile_station_summary.csv"
    if not profile_df.empty:
        station_summary = (
            profile_df[["station_id", "station_name", "station_lat", "station_lon", "sounding_time"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        station_summary.to_csv(station_summary_path, index=False)
        print("Saved IGRA station/profile summary:", station_summary_path)

        # Level time / balloon drift diagnostics (how many soundings have
        # RELTIME/ETIME, typical drift by pressure layer).
        if {"time_source", "drift_km"}.issubset(profile_df.columns):
            per_profile = profile_df.drop_duplicates(["station_id", "sounding_time"])
            print("Level-time source per sounding:")
            print(per_profile["time_source"].value_counts().to_string())
            bins = pd.cut(profile_df["pressure_hpa"], [0, 200, 300, 400, 500, 700, 850, 1100])
            drift_table = profile_df.groupby(bins, observed=True)["drift_km"].describe(percentiles=[0.5, 0.9])
            print("Balloon drift by pressure band [km]:")
            print(drift_table[["count", "50%", "90%", "max"]].round(1).to_string())
            drift_table.to_csv(METRICS_DIR / "igra_balloon_drift_by_pressure.csv")
else:
    print("No IGRA profiles loaded. Metrics will be empty.")

raw_metrics_list: list[pd.DataFrame] = []
qc_metrics_list: list[pd.DataFrame] = []
collocation_tables: dict[str, pd.DataFrame] = {}

# Official AMV baseline.
if RUN_OFFICIAL_AMV:
    print("\n" + "=" * 90)
    print("Official AMV vs IGRA")
    official_raw = load_official_frame()
    official_qc = filter_amv_for_igra_qc(official_raw, source="official_amv")
    official_geometry_fp = amv_geometry_fingerprint(official_raw)
    official_geometry_path = GEOMETRY_CACHE_DIR / f"official_amv_geometry_{official_geometry_fp}_{PROFILE_SET_ID}.npz"
    process_product(
        raw_frame=official_raw,
        qc_frame=official_qc,
        source_name="official_amv",
        comparison="official_amv",
        tracker="official",
        comparison_source="official_amv",
        min_points=MIN_AMV_POINTS_PER_LEVEL_OFFICIAL,
        geometry_cache_path=official_geometry_path,
        geometry_source="official_amv_geometry",
        profiles=profiles,
        raw_metrics_list=raw_metrics_list,
        qc_metrics_list=qc_metrics_list,
        collocation_tables=collocation_tables,
    )
else:
    official_raw = pd.DataFrame()
    official_geometry_path = None

# Tracker products.
for tracker in TRACKER_NAMES:
    if RUN_OCA_PRESSURE_PRODUCTS:
        path = PRESSURE_ASSIGNMENT_DIR / f"{tracker}_final_amv_oca_pressure.npz"
        if path.exists():
            print("\n" + "=" * 90)
            print(f"{tracker} OCA-pressure vs IGRA")
            frame = load_pressure_amv_npz(path, tracker=tracker, amv_source="oca_pressure")
            frame = attach_selfsup_qi(frame, tracker)
            qc = filter_amv_for_igra_qc(frame, source="oca_pressure")
            geometry_fp = amv_geometry_fingerprint(frame)
            geometry_path = GEOMETRY_CACHE_DIR / f"oca_pressure_geometry_{geometry_fp}_{PROFILE_SET_ID}.npz"
            process_product(
                raw_frame=frame,
                qc_frame=qc,
                source_name=f"{tracker}_oca_pressure",
                comparison=f"{tracker}_oca_pressure",
                tracker=tracker,
                comparison_source="oca_pressure",
                min_points=MIN_AMV_POINTS_PER_LEVEL_OCA,
                geometry_cache_path=geometry_path,
                geometry_source=f"oca_pressure_geometry_{geometry_fp}",
                profiles=profiles,
                raw_metrics_list=raw_metrics_list,
                qc_metrics_list=qc_metrics_list,
                collocation_tables=collocation_tables,
            )
        else:
            print("Missing OCA-pressure product, skip:", path)

    if RUN_OFFICIAL_PRESSURE_PRODUCTS:
        path = PRESSURE_ASSIGNMENT_DIR / f"{tracker}_final_amv_official_pressure.npz"
        if path.exists():
            print("\n" + "=" * 90)
            print(f"{tracker} official-pressure vs IGRA")
            frame = load_pressure_amv_npz(path, tracker=tracker, amv_source="official_pressure")
            qc = filter_amv_for_igra_qc(frame, source="official_pressure")
            # Official-pressure products share the official_id geometry cache when possible.
            if RUN_OFFICIAL_AMV and not official_raw.empty:
                geometry_fp = amv_geometry_fingerprint(official_raw)
                geometry_path = GEOMETRY_CACHE_DIR / f"official_amv_geometry_{geometry_fp}_{PROFILE_SET_ID}.npz"
                geometry_source = "official_amv_geometry"
            else:
                geometry_fp = amv_geometry_fingerprint(frame)
                geometry_path = GEOMETRY_CACHE_DIR / f"official_pressure_geometry_{geometry_fp}_{PROFILE_SET_ID}.npz"
                geometry_source = f"official_pressure_geometry_{geometry_fp}"
            process_product(
                raw_frame=frame,
                qc_frame=qc,
                geometry_frame=official_raw if RUN_OFFICIAL_AMV and not official_raw.empty else None,
                source_name=f"{tracker}_official_pressure",
                comparison=f"{tracker}_official_pressure",
                tracker=tracker,
                comparison_source="official_pressure",
                min_points=MIN_AMV_POINTS_PER_LEVEL_OFFICIAL_PRESSURE,
                geometry_cache_path=geometry_path,
                geometry_source=geometry_source,
                profiles=profiles,
                raw_metrics_list=raw_metrics_list,
                qc_metrics_list=qc_metrics_list,
                collocation_tables=collocation_tables,
            )
        else:
            print("Missing official-pressure product, skip:", path)

metrics_by_subset_raw = pd.concat(raw_metrics_list, ignore_index=True) if raw_metrics_list else pd.DataFrame()
metrics_by_subset_qc = pd.concat(qc_metrics_list, ignore_index=True) if qc_metrics_list else pd.DataFrame()
metrics_by_subset_raw, metrics_by_subset_qc = attach_qc_retention_fraction(
    metrics_by_subset_raw,
    metrics_by_subset_qc,
)
metrics_by_subset = pd.concat(
    [metrics_by_subset_raw, metrics_by_subset_qc],
    ignore_index=True,
) if (not metrics_by_subset_raw.empty or not metrics_by_subset_qc.empty) else pd.DataFrame()

if SAVE_METRICS:
    raw_metrics_path = METRICS_DIR / "metrics_all_points.csv"
    qc_metrics_path = METRICS_DIR / "metrics_qc_qi60_speed5.csv"
    combined_metrics_path = METRICS_DIR / "metrics.csv"
    metrics_by_subset_raw.to_csv(raw_metrics_path, index=False)
    metrics_by_subset_qc.to_csv(qc_metrics_path, index=False)
    metrics_by_subset.to_csv(combined_metrics_path, index=False)
    print("\nSaved raw metrics:", raw_metrics_path)
    print("Saved QC metrics:", qc_metrics_path)
    print("Saved combined metrics:", combined_metrics_path)

print("\nRaw metrics:")
display(metrics_by_subset_raw)
print("\nQC metrics:")
display(metrics_by_subset_qc)


# Compact collocation summary.
summary_rows = []
for key, table in collocation_tables.items():
    summary_rows.append(
        {
            "source": key,
            "rows": len(table),
            "stations": table["station_id"].nunique() if "station_id" in table.columns else 0,
            "profiles": table[["station_id", "sounding_time"]].drop_duplicates().shape[0]
            if {"station_id", "sounding_time"}.issubset(table.columns) else 0,
            "mean_neighbors": float(pd.to_numeric(table["amv_neighbor_count"], errors="coerce").mean())
            if "amv_neighbor_count" in table.columns and not table.empty else np.nan,
            "mean_distance_km": float(pd.to_numeric(table["distance_mean_km"], errors="coerce").mean())
            if "distance_mean_km" in table.columns and not table.empty else np.nan,
        }
    )
summary = pd.DataFrame(summary_rows)
summary_path = METRICS_DIR / "collocation_summary.csv"
summary.to_csv(summary_path, index=False)
print("Saved collocation summary:", summary_path)
display(summary)

print("\nDone.")
