"""IGRA radiosonde validation for AMV products.

This module implements the station/sounding/pressure-level aggregation agreed for
AMV-vs-IGRA comparison:

* no vertical interpolation of IGRA winds;
* every AMV point is assigned to its nearest IGRA pressure level;
* keep only points within 150 km, 30 minutes, and 25 hPa by default;
* aggregate AMV points within each station/sounding/sonde-level before metrics.

Optionally (``use_balloon_drift=True``) distances are measured from the drifted
balloon position at each sonde level, and (``time_reference="level"``) the time
criterion uses the per-level observation time (release time + elapsed time)
instead of the nominal sounding time.  Both are computed in ``data_io.igra``.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping
import json
import re

import numpy as np
import pandas as pd
from pyproj import Geod

from .metrics import direction_difference

GEOD = Geod(ellps="WGS84")


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _as_utc_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _safe_name(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return value or "product"


def _speed(u, v) -> np.ndarray:
    return np.hypot(np.asarray(u, dtype=float), np.asarray(v, dtype=float))


def _weighted_mean(values: Any, weights: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    valid = np.isfinite(array) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(array[valid], weights=weights[valid]))


def _pressure_layer(pressure_hpa: float) -> str | None:
    if not np.isfinite(pressure_hpa):
        return None
    if pressure_hpa < 400.0:
        return "high"
    if pressure_hpa < 700.0:
        return "middle"
    return "low"


TIME_REFERENCES = ("level", "nominal")


def _level_table_from_profile(
    profile: Any,
    *,
    use_balloon_drift: bool = False,
    time_reference: str = "nominal",
) -> pd.DataFrame:
    """Sonde levels with wind, observation position and observation time.

    ``obs_lat``/``obs_lon`` are the drifted balloon positions when requested and
    available, otherwise the station position.  ``obs_time`` is the per-level
    time when ``time_reference="level"`` and available, otherwise the nominal
    sounding time.
    """

    columns = [
        "level_index", "sonde_pressure_hpa", "u_sonde", "v_sonde",
        "obs_lat", "obs_lon", "obs_time", "drift_km",
    ]
    reference = str(time_reference).strip().lower()
    if reference not in TIME_REFERENCES:
        raise ValueError(f"time_reference must be one of {TIME_REFERENCES}, got {time_reference!r}")

    pressure = np.asarray(getattr(profile, "pressure_hpa"), dtype=float)
    u = np.asarray(getattr(profile, "u_ms"), dtype=float)
    v = np.asarray(getattr(profile, "v_ms"), dtype=float)
    n = pressure.size
    station_lat = float(getattr(profile, "station_lat"))
    station_lon = float(getattr(profile, "station_lon"))
    nominal = _as_utc_timestamp(getattr(profile, "sounding_time")).tz_localize(None).to_datetime64()

    level_lat = getattr(profile, "level_lat", None)
    level_lon = getattr(profile, "level_lon", None)
    drift = getattr(profile, "drift_km", None)
    level_time = getattr(profile, "level_time", None)
    if use_balloon_drift and level_lat is not None and level_lon is not None and len(level_lat) == n:
        obs_lat = np.asarray(level_lat, dtype=float)
        obs_lon = np.asarray(level_lon, dtype=float)
        drift_km = np.asarray(drift, dtype=float) if drift is not None else np.full(n, np.nan)
        bad = ~(np.isfinite(obs_lat) & np.isfinite(obs_lon))
        obs_lat = np.where(bad, station_lat, obs_lat)
        obs_lon = np.where(bad, station_lon, obs_lon)
    else:
        obs_lat = np.full(n, station_lat)
        obs_lon = np.full(n, station_lon)
        drift_km = np.zeros(n)
    if reference == "level" and level_time is not None and len(level_time) == n:
        obs_time = np.asarray(level_time, dtype="datetime64[ns]")
        obs_time = np.where(np.isnat(obs_time), nominal, obs_time).astype("datetime64[ns]")
    else:
        obs_time = np.full(n, nominal, dtype="datetime64[ns]")

    valid = np.isfinite(pressure) & np.isfinite(u) & np.isfinite(v) & (pressure > 0)
    if valid.sum() == 0:
        return pd.DataFrame(columns=columns)
    levels = pd.DataFrame(
        {
            "sonde_pressure_hpa": pressure[valid].astype(np.float32),
            "u_sonde": u[valid].astype(np.float32),
            "v_sonde": v[valid].astype(np.float32),
            "obs_lat": obs_lat[valid],
            "obs_lon": obs_lon[valid],
            "obs_time": obs_time[valid],
            "drift_km": drift_km[valid].astype(np.float32),
        }
    )
    # If duplicate pressure levels exist, keep the first after sorting by pressure.
    levels = levels.sort_values("sonde_pressure_hpa").drop_duplicates("sonde_pressure_hpa", keep="first").reset_index(drop=True)
    levels["level_index"] = np.arange(len(levels), dtype=np.int32)
    return levels[columns]


def _level_time_difference_minutes(levels: pd.DataFrame, target_time: pd.Timestamp) -> np.ndarray:
    target = target_time.tz_convert("UTC").tz_localize(None).to_datetime64()
    delta = (levels["obs_time"].to_numpy(dtype="datetime64[ns]") - target) / np.timedelta64(1, "s")
    return np.abs(delta.astype(float)) / 60.0


def _profile_candidates(
    lon: np.ndarray,
    lat: np.ndarray,
    pressure: np.ndarray,
    levels: pd.DataFrame,
    *,
    station_lon: float,
    station_lat: float,
    time_ok: np.ndarray,
    max_distance_km: float,
    pressure_window_hpa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """AMV indices matched to one sounding: (idx, level_index, delta_p, distance_km).

    Each AMV point is assigned to its nearest sonde pressure level; the point is
    kept only if that level satisfies the time criterion, the pressure window,
    and lies within ``max_distance_km`` of the level's observation position.
    """

    empty = (np.array([], dtype=np.int64), np.array([], dtype=np.int32), np.array([], dtype=np.float32), np.array([], dtype=float))
    if not np.any(time_ok):
        return empty
    level_lat = levels["obs_lat"].to_numpy(float)
    level_lon = levels["obs_lon"].to_numpy(float)
    offset = GEOD.inv(
        np.full(level_lon.size, station_lon), np.full(level_lat.size, station_lat), level_lon, level_lat
    )[2]
    max_offset_km = float(np.nanmax(np.asarray(offset, dtype=float)[time_ok])) / 1000.0 if time_ok.any() else 0.0
    rough = _rough_station_mask(lon, lat, station_lon, station_lat, max_distance_km + max(max_offset_km, 0.0))
    if not rough.any():
        return empty
    rough_indices = np.flatnonzero(rough)
    nearest_level, delta_p = _nearest_level_indices(pressure[rough_indices], levels["sonde_pressure_hpa"].to_numpy(float))
    keep = (nearest_level >= 0) & np.isfinite(delta_p) & (np.abs(delta_p) <= float(pressure_window_hpa))
    if keep.any():
        keep[keep] &= time_ok[nearest_level[keep]]
    if not keep.any():
        return empty
    idx = rough_indices[keep]
    nearest_level = nearest_level[keep]
    delta_p = delta_p[keep]
    _, _, distance_m = GEOD.inv(lon[idx], lat[idx], level_lon[nearest_level], level_lat[nearest_level])
    distance_km = np.asarray(distance_m, dtype=float) / 1000.0
    within = np.isfinite(distance_km) & (distance_km <= float(max_distance_km))
    return idx[within], nearest_level[within], delta_p[within], distance_km[within]


def _nearest_level_indices(amv_pressure: np.ndarray, sonde_pressure: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest sonde-level indices and signed AMV-minus-sonde pressure differences."""

    p = np.asarray(amv_pressure, dtype=float)
    levels = np.asarray(sonde_pressure, dtype=float)
    if levels.size == 0:
        return np.full(p.shape, -1, dtype=np.int32), np.full(p.shape, np.nan, dtype=np.float32)
    order = np.argsort(levels)
    levels_sorted = levels[order]
    insert = np.searchsorted(levels_sorted, p)
    left = np.clip(insert - 1, 0, levels_sorted.size - 1)
    right = np.clip(insert, 0, levels_sorted.size - 1)
    left_diff = np.abs(p - levels_sorted[left])
    right_diff = np.abs(p - levels_sorted[right])
    choose_right = right_diff < left_diff
    chosen_sorted = np.where(choose_right, right, left)
    chosen_original = order[chosen_sorted]
    signed_delta = p - levels[chosen_original]
    invalid = ~np.isfinite(p)
    chosen_original = chosen_original.astype(np.int32)
    chosen_original[invalid] = -1
    signed_delta = signed_delta.astype(np.float32)
    signed_delta[invalid] = np.nan
    return chosen_original, signed_delta


def _distance_weights(distance_km: np.ndarray, aggregation: str, sigma_km: float) -> np.ndarray:
    distance = np.asarray(distance_km, dtype=float)
    if aggregation == "mean":
        return np.ones(distance.shape, dtype=np.float32)
    if aggregation == "distance_weighted":
        sigma = float(sigma_km)
        if sigma <= 0:
            raise ValueError("distance_weight_sigma_km must be positive")
        return np.exp(-0.5 * (distance / sigma) ** 2).astype(np.float32)
    raise ValueError("aggregation must be 'mean' or 'distance_weighted'")


def _rough_station_mask(lon: np.ndarray, lat: np.ndarray, station_lon: float, station_lat: float, radius_km: float) -> np.ndarray:
    """Cheap lon/lat bounding-box prefilter before geodesic distance."""

    lat_delta = float(radius_km) / 111.2
    cos_lat = max(np.cos(np.deg2rad(float(station_lat))), 0.1)
    lon_delta = float(radius_km) / (111.2 * cos_lat)
    # Wrap longitude differences so stations near the dateline (GOES-West) work.
    dlon = (np.asarray(lon, dtype=float) - float(station_lon) + 180.0) % 360.0 - 180.0
    return (
        np.isfinite(lon)
        & np.isfinite(lat)
        & (lat >= station_lat - lat_delta)
        & (lat <= station_lat + lat_delta)
        & (np.abs(dlon) <= lon_delta)
    )


# -----------------------------------------------------------------------------
# AMV standardisation
# -----------------------------------------------------------------------------


def load_pressure_amv_npz(path: str | Path, *, tracker: str | None = None, amv_source: str | None = None) -> pd.DataFrame:
    """Load a pressure-assigned PIV-AMV NPZ into a standard AMV table.

    Returned columns include ``longitude``, ``latitude``, ``pressure_hpa``,
    ``u_amv``, ``v_amv``, and ``speed_amv``. Available keys such as
    ``row_center_original``, ``col_center_original``, and ``official_id`` are
    preserved for diagnostics.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"AMV NPZ not found: {source}")
    with np.load(source, allow_pickle=False) as data:
        files = list(data.files)
        required = ["longitude", "latitude", "u_ms", "v_ms", "pressure_hpa"]
        missing = [name for name in required if name not in files]
        if missing:
            raise KeyError(f"{source} is missing required arrays: {missing}")
        n_points = np.asarray(data["longitude"]).size
        columns: dict[str, Any] = {}
        for key in files:
            if key == "metadata_json":
                continue
            arr = np.asarray(data[key])
            if arr.ndim == 1 and arr.size == n_points:
                columns[key] = arr.copy()
            elif arr.size == 1 and key == "pressure_source":
                columns[key] = np.repeat(str(arr.ravel()[0]), n_points)
    frame = pd.DataFrame(columns)
    frame = frame.rename(columns={"u_ms": "u_amv", "v_ms": "v_amv"})
    frame["speed_amv"] = _speed(frame["u_amv"], frame["v_amv"])
    frame["tracker"] = tracker or ""
    if amv_source is None:
        if "pressure_source" in frame.columns:
            src = str(frame["pressure_source"].iloc[0])
            amv_source = "oca_pressure" if src == "oca" else "official_pressure" if "official" in src else src
        else:
            amv_source = "amv"
    frame["amv_source"] = amv_source
    finite = (
        np.isfinite(frame["longitude"])
        & np.isfinite(frame["latitude"])
        & np.isfinite(frame["pressure_hpa"])
        & np.isfinite(frame["u_amv"])
        & np.isfinite(frame["v_amv"])
    )
    return frame.loc[finite].reset_index(drop=True)


def standardize_official_amv(official_amv: Any) -> pd.DataFrame:
    """Return official AMVs using the same columns as PIV-AMV products."""

    if hasattr(official_amv, "frame"):
        frame = official_amv.frame.copy()
    elif isinstance(official_amv, pd.DataFrame):
        frame = official_amv.copy()
    else:
        raise TypeError("official_amv must be a DataFrame or an object with a .frame attribute")

    rename = {
        "lon": "longitude",
        "lat": "latitude",
        "p_official": "pressure_hpa",
        "u_official": "u_amv",
        "v_official": "v_amv",
        "speed_official": "speed_amv",
    }
    frame = frame.rename(columns={old: new for old, new in rename.items() if old in frame.columns})
    required = ["longitude", "latitude", "pressure_hpa", "u_amv", "v_amv"]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise KeyError(f"official AMV is missing required columns after standardisation: {missing}")
    if "speed_amv" not in frame.columns:
        frame["speed_amv"] = _speed(frame["u_amv"], frame["v_amv"])
    if "official_id" not in frame.columns:
        frame["official_id"] = np.arange(len(frame), dtype=np.int32)
    frame["tracker"] = ""
    frame["amv_source"] = "official_amv"
    frame["pressure_source"] = "official_amv"
    finite = (
        np.isfinite(frame["longitude"])
        & np.isfinite(frame["latitude"])
        & np.isfinite(frame["pressure_hpa"])
        & np.isfinite(frame["u_amv"])
        & np.isfinite(frame["v_amv"])
    )
    return frame.loc[finite].reset_index(drop=True)


def inspect_npz_product(path: str | Path) -> pd.DataFrame:
    """Return a compact table of NPZ arrays for notebook inspection."""

    rows = []
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        for key in data.files:
            arr = np.asarray(data[key])
            rows.append({"key": key, "shape": str(arr.shape), "dtype": str(arr.dtype)})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Collocation
# -----------------------------------------------------------------------------


def collocate_amv_with_igra_levels(
    amv: pd.DataFrame,
    igra_profiles: list[Any],
    *,
    source: str,
    product_time,
    max_distance_km: float = 150.0,
    max_time_minutes: float = 30.0,
    pressure_window_hpa: float = 25.0,
    min_amv_points_per_level: int = 1,
    aggregation: str = "distance_weighted",
    distance_weight_sigma_km: float = 75.0,
    use_balloon_drift: bool = False,
    time_reference: str = "nominal",
) -> pd.DataFrame:
    """Aggregate AMVs around IGRA stations and pressure levels.

    One output row represents one AMV source collocated with one radiosonde
    sounding at one radiosonde pressure level. AMV points are matched to their
    nearest radiosonde pressure level and retained only when the pressure
    difference is within ``pressure_window_hpa``.
    """

    if int(min_amv_points_per_level) < 1:
        raise ValueError("min_amv_points_per_level must be at least 1")

    target_time = _as_utc_timestamp(product_time)
    frame = amv.copy().reset_index(drop=True)
    required = ["longitude", "latitude", "pressure_hpa", "u_amv", "v_amv"]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise KeyError(f"AMV table is missing required columns: {missing}")

    lon = frame["longitude"].to_numpy(float)
    lat = frame["latitude"].to_numpy(float)
    pressure = frame["pressure_hpa"].to_numpy(float)
    u_amv = frame["u_amv"].to_numpy(float)
    v_amv = frame["v_amv"].to_numpy(float)

    rows: list[dict[str, Any]] = []
    for profile in igra_profiles:
        sounding_time = _as_utc_timestamp(getattr(profile, "sounding_time"))
        levels = _level_table_from_profile(
            profile, use_balloon_drift=use_balloon_drift, time_reference=time_reference
        )
        if levels.empty:
            continue
        level_dt = _level_time_difference_minutes(levels, target_time)
        time_ok = np.isfinite(level_dt) & (level_dt <= float(max_time_minutes))
        if not time_ok.any():
            continue

        station_lon = float(getattr(profile, "station_lon"))
        station_lat = float(getattr(profile, "station_lat"))
        idx, nearest_level, delta_p, dist = _profile_candidates(
            lon, lat, pressure, levels,
            station_lon=station_lon, station_lat=station_lat, time_ok=time_ok,
            max_distance_km=max_distance_km, pressure_window_hpa=pressure_window_hpa,
        )
        if idx.size == 0:
            continue

        weights = _distance_weights(dist, aggregation, distance_weight_sigma_km)
        for level_index in np.unique(nearest_level):
            member = nearest_level == level_index
            count = int(member.sum())
            if count < int(min_amv_points_per_level):
                continue
            level = levels.iloc[int(level_index)]
            w = weights[member]
            amv_idx = idx[member]

            u_agg = _weighted_mean(u_amv[amv_idx], w)
            v_agg = _weighted_mean(v_amv[amv_idx], w)
            u_sonde = float(level["u_sonde"])
            v_sonde = float(level["v_sonde"])
            signed_direction = float(direction_difference([u_agg], [v_agg], [u_sonde], [v_sonde])[0])
            speed_amv = float(np.hypot(u_agg, v_agg))
            speed_sonde = float(np.hypot(u_sonde, v_sonde))
            sonde_pressure = float(level["sonde_pressure_hpa"])
            amv_pressure = _weighted_mean(pressure[amv_idx], w)
            pressure_difference = amv_pressure - sonde_pressure if np.isfinite(amv_pressure) else np.nan

            tracker = str(frame["tracker"].iloc[0]) if "tracker" in frame.columns and len(frame) else ""
            amv_source = str(frame["amv_source"].iloc[0]) if "amv_source" in frame.columns and len(frame) else "amv"
            pressure_source = str(frame["pressure_source"].iloc[0]) if "pressure_source" in frame.columns and len(frame) else ""
            rows.append(
                {
                    "source": source,
                    "tracker": tracker,
                    "amv_source": amv_source,
                    "pressure_source": pressure_source,
                    "station_id": str(getattr(profile, "station_id")),
                    "station_name": str(getattr(profile, "station_name", "")),
                    "station_lat": station_lat,
                    "station_lon": station_lon,
                    "sounding_time": sounding_time.isoformat(),
                    "time_difference_minutes": float(level_dt[int(level_index)]),
                    "obs_lat": float(level["obs_lat"]),
                    "obs_lon": float(level["obs_lon"]),
                    "drift_km": float(level["drift_km"]),
                    "sonde_pressure_hpa": sonde_pressure,
                    "amv_pressure_hpa": float(amv_pressure),
                    "pressure_difference_hpa": float(pressure_difference),
                    "pressure_layer": _pressure_layer(sonde_pressure),
                    "amv_neighbor_count": count,
                    "distance_mean_km": _weighted_mean(dist[member], w),
                    "longitude_mean": _weighted_mean(lon[amv_idx], w),
                    "latitude_mean": _weighted_mean(lat[amv_idx], w),
                    "u_amv": u_agg,
                    "v_amv": v_agg,
                    "speed_amv": speed_amv,
                    "u_sonde": u_sonde,
                    "v_sonde": v_sonde,
                    "speed_sonde": speed_sonde,
                    "speed_bias_amv_minus_sonde": speed_amv - speed_sonde,
                    "vector_difference_amv_vs_sonde": float(np.hypot(u_agg - u_sonde, v_agg - v_sonde)),
                    "signed_direction_difference_deg": signed_direction,
                    "abs_direction_difference_deg": abs(signed_direction),
                    "aggregation_method": aggregation,
                    "max_distance_km": float(max_distance_km),
                    "pressure_window_hpa": float(pressure_window_hpa),
                    "use_balloon_drift": bool(use_balloon_drift),
                    "time_reference": str(time_reference),
                }
            )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Diagnostics and metrics
# -----------------------------------------------------------------------------


def build_igra_comparison_masks(table: pd.DataFrame) -> dict[str, pd.Series]:
    """Return the standard subset masks for AMV-vs-IGRA comparisons."""

    index = table.index
    finite = pd.Series(True, index=index)
    for column in ["u_amv", "v_amv", "u_sonde", "v_sonde"]:
        finite &= np.isfinite(pd.to_numeric(table.get(column, np.nan), errors="coerce"))
    masks: dict[str, pd.Series] = {"all_valid": finite}
    if "speed_amv" in table.columns:
        masks["amv_speed_ge5"] = finite & pd.to_numeric(table["speed_amv"], errors="coerce").ge(5.0)
    if "pressure_layer" in table.columns:
        layer = table["pressure_layer"].astype(str)
        masks["pressure_high_lt400"] = finite & layer.eq("high")
        masks["pressure_mid_400_700"] = finite & layer.eq("middle")
        masks["pressure_low_gt700"] = finite & layer.eq("low")
    elif "sonde_pressure_hpa" in table.columns:
        p = pd.to_numeric(table["sonde_pressure_hpa"], errors="coerce")
        masks["pressure_high_lt400"] = finite & p.lt(400.0)
        masks["pressure_mid_400_700"] = finite & p.ge(400.0) & p.lt(700.0)
        masks["pressure_low_gt700"] = finite & p.ge(700.0)
    return masks


def _metric_row(selected: pd.DataFrame, *, subset: str, comparison: str) -> dict[str, Any]:
    row: dict[str, Any] = {"subset": subset, "comparison": comparison, "sample_count": int(len(selected))}
    names = [
        "u_bias_amv_minus_sonde",
        "v_bias_amv_minus_sonde",
        "speed_bias_amv_minus_sonde",
        "mean_absolute_speed_error",
        "speed_rmse",
        "mean_vector_difference",
        "median_vector_difference",
        "vector_difference_rmse",
        "mean_absolute_direction_difference",
        "median_absolute_direction_difference",
        "mean_signed_direction_difference",
        "speed_correlation",
        "u_correlation",
        "v_correlation",
        "vector_component_correlation",
        "station_count",
        "profile_count",
        "mean_amv_neighbor_count",
        "mean_distance_km",
        "mean_abs_pressure_difference_hpa",
    ]
    if selected.empty:
        row.update({name: np.nan for name in names})
        row["station_count"] = 0
        row["profile_count"] = 0
        return row

    u1 = pd.to_numeric(selected["u_amv"], errors="coerce")
    v1 = pd.to_numeric(selected["v_amv"], errors="coerce")
    u2 = pd.to_numeric(selected["u_sonde"], errors="coerce")
    v2 = pd.to_numeric(selected["v_sonde"], errors="coerce")
    valid = np.isfinite(u1) & np.isfinite(v1) & np.isfinite(u2) & np.isfinite(v2)
    u1, v1, u2, v2 = u1[valid], v1[valid], u2[valid], v2[valid]
    if len(u1) == 0:
        row.update({name: np.nan for name in names})
        row["station_count"] = 0
        row["profile_count"] = 0
        return row
    du = u1 - u2
    dv = v1 - v2
    speed1 = np.hypot(u1, v1)
    speed2 = np.hypot(u2, v2)
    speed_error = speed1 - speed2
    vd = np.hypot(du, dv)
    angle = direction_difference(u1, v1, u2, v2)
    row.update(
        {
            "sample_count": int(len(u1)),
            "u_bias_amv_minus_sonde": float(np.nanmean(du)),
            "v_bias_amv_minus_sonde": float(np.nanmean(dv)),
            "speed_bias_amv_minus_sonde": float(np.nanmean(speed_error)),
            "mean_absolute_speed_error": float(np.nanmean(np.abs(speed_error))),
            "speed_rmse": float(np.sqrt(np.nanmean(speed_error**2))),
            "mean_vector_difference": float(np.nanmean(vd)),
            "median_vector_difference": float(np.nanmedian(vd)),
            "vector_difference_rmse": float(np.sqrt(np.nanmean(vd**2))),
            "mean_absolute_direction_difference": float(np.nanmean(np.abs(angle))),
            "median_absolute_direction_difference": float(np.nanmedian(np.abs(angle))),
            "mean_signed_direction_difference": float(np.nanmean(angle)),
            "speed_correlation": float(pd.Series(speed1).corr(pd.Series(speed2))) if len(speed1) > 1 and np.nanstd(speed1) > 0 and np.nanstd(speed2) > 0 else np.nan,
            "u_correlation": float(pd.Series(u1).corr(pd.Series(u2))) if len(u1) > 1 and np.nanstd(u1) > 0 and np.nanstd(u2) > 0 else np.nan,
            "v_correlation": float(pd.Series(v1).corr(pd.Series(v2))) if len(v1) > 1 and np.nanstd(v1) > 0 and np.nanstd(v2) > 0 else np.nan,
            "vector_component_correlation": float(pd.Series(np.concatenate([np.asarray(u1), np.asarray(v1)])).corr(pd.Series(np.concatenate([np.asarray(u2), np.asarray(v2)])))) if len(u1) > 1 and np.nanstd(np.concatenate([np.asarray(u1), np.asarray(v1)])) > 0 and np.nanstd(np.concatenate([np.asarray(u2), np.asarray(v2)])) > 0 else np.nan,
            "station_count": int(selected.loc[valid, "station_id"].nunique()) if "station_id" in selected.columns else np.nan,
            "profile_count": int(selected.loc[valid, ["station_id", "sounding_time"]].drop_duplicates().shape[0]) if {"station_id", "sounding_time"}.issubset(selected.columns) else np.nan,
            "mean_amv_neighbor_count": float(pd.to_numeric(selected.loc[valid, "amv_neighbor_count"], errors="coerce").mean()) if "amv_neighbor_count" in selected.columns else np.nan,
            "mean_distance_km": float(pd.to_numeric(selected.loc[valid, "distance_mean_km"], errors="coerce").mean()) if "distance_mean_km" in selected.columns else np.nan,
            "mean_abs_pressure_difference_hpa": float(pd.to_numeric(selected.loc[valid, "pressure_difference_hpa"], errors="coerce").abs().mean()) if "pressure_difference_hpa" in selected.columns else np.nan,
        }
    )
    return row


def compare_to_igra_metrics(
    collocation_table: pd.DataFrame,
    *,
    comparison: str = "amv_vs_igra",
    subset_masks: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Compute standard metrics from station-level AMV-IGRA collocations."""

    if collocation_table.empty:
        return pd.DataFrame()
    masks = subset_masks or build_igra_comparison_masks(collocation_table)
    rows = []
    for subset, mask in masks.items():
        selected = collocation_table.loc[mask.fillna(False)].copy()
        rows.append(_metric_row(selected, subset=subset, comparison=comparison))
    return pd.DataFrame(rows)


def compare_many_to_igra(collocation_tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute metrics for several named AMV-IGRA collocation tables."""

    rows = []
    for name, table in collocation_tables.items():
        metrics = compare_to_igra_metrics(table, comparison=name)
        if metrics.empty:
            continue
        if "tracker" in table.columns and table["tracker"].notna().any():
            tracker_values = table["tracker"].dropna().astype(str)
            tracker = tracker_values.iloc[0] if not tracker_values.empty else ""
            metrics.insert(0, "tracker", tracker)
        if "amv_source" in table.columns and table["amv_source"].notna().any():
            src_values = table["amv_source"].dropna().astype(str)
            src = src_values.iloc[0] if not src_values.empty else ""
            metrics.insert(1 if "tracker" in metrics.columns else 0, "comparison_source", src)
        metrics.insert(0, "model", name)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# -----------------------------------------------------------------------------
# NPZ persistence for station-level collocation cache
# -----------------------------------------------------------------------------


def save_collocation_npz(
    table: pd.DataFrame,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    compressed: bool = False,
) -> Path:
    """Save a station-level collocation table as NPZ."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {}
    for column in table.columns:
        series = table[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            arrays[column] = series.astype(str).to_numpy(dtype="U")
        elif pd.api.types.is_bool_dtype(series):
            arrays[column] = series.to_numpy(dtype=bool)
        elif pd.api.types.is_numeric_dtype(series):
            arrays[column] = series.to_numpy()
        else:
            arrays[column] = series.fillna("").astype(str).to_numpy(dtype="U")
    meta = dict(metadata or {})
    meta.setdefault("n_rows", int(len(table)))
    arrays["metadata_json"] = np.asarray([json.dumps(meta, default=str)], dtype="U")
    if compressed:
        np.savez_compressed(destination, **arrays)
    else:
        np.savez(destination, **arrays)
    return destination


def load_collocation_npz(path: str | Path) -> pd.DataFrame:
    """Load a station-level collocation NPZ created by ``save_collocation_npz``."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as data:
        columns = {key: np.asarray(data[key]).copy() for key in data.files if key != "metadata_json"}
    frame = pd.DataFrame(columns)
    if "sounding_time" in frame.columns:
        frame["sounding_time"] = pd.to_datetime(frame["sounding_time"], utc=True, errors="coerce")
    return frame


def read_collocation_npz_metadata(path: str | Path) -> dict[str, Any]:
    """Read metadata saved in a collocation NPZ, if present.

    This is used to avoid accidentally reusing a stale collocation cache after
    changing settings such as ``min_amv_points_per_level``.
    """

    source = Path(path)
    if not source.exists():
        return {}
    with np.load(source, allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            return {}
        raw = np.asarray(data["metadata_json"]).ravel()
        if raw.size == 0:
            return {}
        try:
            return json.loads(str(raw[0]))
        except Exception:
            return {}


def _cache_metadata_matches(cached: Mapping[str, Any], requested: Mapping[str, Any]) -> bool:
    """Return True when a collocation cache was made with these settings."""

    if not cached:
        return False
    for key, value in requested.items():
        if key not in cached:
            return False
        cached_value = cached[key]
        if isinstance(value, float):
            try:
                if not np.isclose(float(cached_value), float(value), rtol=0.0, atol=1e-9):
                    return False
            except Exception:
                return False
        else:
            if str(cached_value) != str(value):
                return False
    return True


def collocate_or_load(
    *,
    cache_path: str | Path,
    amv: pd.DataFrame,
    igra_profiles: list[Any],
    source: str,
    product_time,
    reuse_cache: bool = True,
    save_cache: bool = True,
    compressed: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Load a collocation cache if available, otherwise compute and optionally save it."""

    cache = Path(cache_path)
    metadata = {"source": source, "product_time": str(product_time), **kwargs}
    if reuse_cache and cache.exists():
        cached_metadata = read_collocation_npz_metadata(cache)
        if _cache_metadata_matches(cached_metadata, metadata):
            return load_collocation_npz(cache)
        print(f"Ignoring stale collocation cache with mismatched metadata: {cache}")
    table = collocate_amv_with_igra_levels(
        amv,
        igra_profiles,
        source=source,
        product_time=product_time,
        **kwargs,
    )
    if save_cache:
        save_collocation_npz(
            table,
            cache,
            metadata=metadata,
            compressed=compressed,
        )
    return table


# -----------------------------------------------------------------------------
# Accelerated point-level IGRA geometry cache
# -----------------------------------------------------------------------------


def _rounded_int_series(values: Any, scale: float) -> np.ndarray:
    """Return stable int64 keys from numeric values, with a sentinel for NaN."""

    array = np.asarray(values, dtype=float)
    out = np.full(array.shape, np.iinfo(np.int64).min, dtype=np.int64)
    finite = np.isfinite(array)
    out[finite] = np.rint(array[finite] * float(scale)).astype(np.int64)
    return out


def add_amv_point_keys(amv: pd.DataFrame, *, include_pressure: bool = True) -> pd.DataFrame:
    """Return an AMV table with stable numeric point-key columns.

    The key is used to reuse an IGRA geometry cache across products that share
    the same AMV geometry.  Official-pressure products use ``official_id`` when
    available.  OCA-pressure products use rounded row/column/pressure keys.
    """

    frame = amv.copy()
    n = len(frame)

    if "official_id" in frame.columns:
        key1 = pd.to_numeric(frame["official_id"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        frame["_amv_key_kind"] = "official_id"
        frame["_amv_key1"] = key1
        frame["_amv_key2"] = np.zeros(n, dtype=np.int64)
        frame["_amv_key3"] = np.zeros(n, dtype=np.int64)
        return frame

    if {"row_center_original", "col_center_original"}.issubset(frame.columns):
        frame["_amv_key_kind"] = "row_col_pressure" if include_pressure else "row_col"
        frame["_amv_key1"] = _rounded_int_series(frame["row_center_original"], 1000.0)
        frame["_amv_key2"] = _rounded_int_series(frame["col_center_original"], 1000.0)
        if include_pressure and "pressure_hpa" in frame.columns:
            frame["_amv_key3"] = _rounded_int_series(frame["pressure_hpa"], 100.0)
        else:
            frame["_amv_key3"] = np.zeros(n, dtype=np.int64)
        return frame

    frame["_amv_key_kind"] = "row_index"
    frame["_amv_key1"] = np.arange(n, dtype=np.int64)
    frame["_amv_key2"] = np.zeros(n, dtype=np.int64)
    frame["_amv_key3"] = np.zeros(n, dtype=np.int64)
    return frame


def amv_geometry_fingerprint(amv: pd.DataFrame, *, include_pressure: bool = True) -> str:
    """Return a compact fingerprint for an AMV product's matching geometry."""

    import hashlib

    keyed = add_amv_point_keys(amv, include_pressure=include_pressure)
    kind = str(keyed["_amv_key_kind"].iloc[0]) if len(keyed) else "empty"
    digest = hashlib.sha1()
    digest.update(kind.encode("utf-8"))
    digest.update(str(len(keyed)).encode("ascii"))
    for column in ["_amv_key1", "_amv_key2", "_amv_key3"]:
        arr = np.ascontiguousarray(keyed[column].to_numpy(dtype=np.int64))
        digest.update(arr.shape[0].to_bytes(8, byteorder="little", signed=False))
        digest.update(arr.tobytes())
    return f"{kind}_{len(keyed)}_{digest.hexdigest()[:16]}"


def _pressure_layer_array(pressure_hpa: np.ndarray) -> np.ndarray:
    pressure = np.asarray(pressure_hpa, dtype=float)
    out = np.full(pressure.shape, "", dtype="U8")
    out[np.isfinite(pressure) & (pressure < 400.0)] = "high"
    out[np.isfinite(pressure) & (pressure >= 400.0) & (pressure < 700.0)] = "middle"
    out[np.isfinite(pressure) & (pressure >= 700.0)] = "low"
    return out


def build_igra_point_geometry(
    amv: pd.DataFrame,
    igra_profiles: list[Any],
    *,
    source: str,
    product_time,
    max_distance_km: float = 150.0,
    max_time_minutes: float = 30.0,
    pressure_window_hpa: float = 25.0,
    aggregation: str = "distance_weighted",
    distance_weight_sigma_km: float = 75.0,
    include_pressure_in_key: bool = True,
    use_balloon_drift: bool = False,
    time_reference: str = "nominal",
) -> pd.DataFrame:
    """Build a point-level AMV-to-IGRA geometry/membership table.

    This is the expensive part: it searches stations, computes geodesic
    distance, and assigns each AMV point to the nearest radiosonde pressure
    level.  It stores only geometry and sonde fields, not AMV wind values, so
    the same cache can be reused by multiple trackers sharing the same geometry.
    """

    target_time = _as_utc_timestamp(product_time)
    frame = add_amv_point_keys(amv, include_pressure=include_pressure_in_key).reset_index(drop=True)
    required = ["longitude", "latitude", "pressure_hpa"]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise KeyError(f"AMV table is missing required geometry columns: {missing}")

    lon = pd.to_numeric(frame["longitude"], errors="coerce").to_numpy(float)
    lat = pd.to_numeric(frame["latitude"], errors="coerce").to_numpy(float)
    pressure = pd.to_numeric(frame["pressure_hpa"], errors="coerce").to_numpy(float)
    key_kind = frame["_amv_key_kind"].astype(str).to_numpy()
    key1 = frame["_amv_key1"].to_numpy(np.int64)
    key2 = frame["_amv_key2"].to_numpy(np.int64)
    key3 = frame["_amv_key3"].to_numpy(np.int64)

    chunks: list[pd.DataFrame] = []
    for profile in igra_profiles:
        sounding_time = _as_utc_timestamp(getattr(profile, "sounding_time"))
        levels = _level_table_from_profile(
            profile, use_balloon_drift=use_balloon_drift, time_reference=time_reference
        )
        if levels.empty:
            continue
        level_dt = _level_time_difference_minutes(levels, target_time)
        time_ok = np.isfinite(level_dt) & (level_dt <= float(max_time_minutes))
        if not time_ok.any():
            continue

        station_lon = float(getattr(profile, "station_lon"))
        station_lat = float(getattr(profile, "station_lat"))
        idx, nearest_level, delta_p, dist = _profile_candidates(
            lon, lat, pressure, levels,
            station_lon=station_lon, station_lat=station_lat, time_ok=time_ok,
            max_distance_km=max_distance_km, pressure_window_hpa=pressure_window_hpa,
        )
        if idx.size == 0:
            continue

        weights = _distance_weights(dist, aggregation, distance_weight_sigma_km)

        level_take = levels.iloc[nearest_level].reset_index(drop=True)
        sonde_pressure = level_take["sonde_pressure_hpa"].to_numpy(float)
        chunks.append(
            pd.DataFrame(
                {
                    "source": source,
                    "amv_index": idx.astype(np.int64),
                    "_amv_key_kind": key_kind[idx],
                    "_amv_key1": key1[idx],
                    "_amv_key2": key2[idx],
                    "_amv_key3": key3[idx],
                    "station_id": str(getattr(profile, "station_id")),
                    "station_name": str(getattr(profile, "station_name", "")),
                    "station_lat": np.full(idx.size, station_lat, dtype=np.float32),
                    "station_lon": np.full(idx.size, station_lon, dtype=np.float32),
                    "sounding_time": sounding_time.isoformat(),
                    "time_difference_minutes": level_dt[nearest_level].astype(np.float32),
                    "obs_lat": level_take["obs_lat"].to_numpy(np.float32),
                    "obs_lon": level_take["obs_lon"].to_numpy(np.float32),
                    "drift_km": level_take["drift_km"].to_numpy(np.float32),
                    "level_index": level_take["level_index"].to_numpy(np.int32),
                    "sonde_pressure_hpa": sonde_pressure.astype(np.float32),
                    "u_sonde": level_take["u_sonde"].to_numpy(np.float32),
                    "v_sonde": level_take["v_sonde"].to_numpy(np.float32),
                    "pressure_layer": _pressure_layer_array(sonde_pressure),
                    "distance_km": dist.astype(np.float32),
                    "pressure_difference_hpa": delta_p.astype(np.float32),
                    "weight": weights.astype(np.float32),
                }
            )
        )

    if not chunks:
        return pd.DataFrame(
            columns=[
                "source", "amv_index", "_amv_key_kind", "_amv_key1", "_amv_key2", "_amv_key3",
                "station_id", "station_name", "station_lat", "station_lon", "sounding_time",
                "time_difference_minutes", "obs_lat", "obs_lon", "drift_km",
                "level_index", "sonde_pressure_hpa", "u_sonde", "v_sonde",
                "pressure_layer", "distance_km", "pressure_difference_hpa", "weight",
            ]
        )

    return pd.concat(chunks, ignore_index=True)


def geometry_or_load(
    *,
    cache_path: str | Path,
    amv: pd.DataFrame,
    igra_profiles: list[Any],
    source: str,
    product_time,
    reuse_cache: bool = True,
    save_cache: bool = True,
    compressed: bool = False,
    include_pressure_in_key: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Load or build a point-level IGRA geometry cache."""

    cache = Path(cache_path)
    fingerprint = amv_geometry_fingerprint(amv, include_pressure=include_pressure_in_key)
    metadata = {
        "cache_type": "igra_point_geometry",
        "source": source,
        "product_time": str(product_time),
        "amv_geometry_fingerprint": fingerprint,
        "include_pressure_in_key": bool(include_pressure_in_key),
        **kwargs,
    }
    if reuse_cache and cache.exists():
        cached_metadata = read_collocation_npz_metadata(cache)
        if _cache_metadata_matches(cached_metadata, metadata):
            return load_collocation_npz(cache)
        print(f"Ignoring stale IGRA geometry cache with mismatched metadata: {cache}")

    table = build_igra_point_geometry(
        amv,
        igra_profiles,
        source=source,
        product_time=product_time,
        include_pressure_in_key=include_pressure_in_key,
        **kwargs,
    )
    if save_cache:
        save_collocation_npz(table, cache, metadata=metadata, compressed=compressed)
    return table


def collocate_from_igra_geometry(
    *,
    amv: pd.DataFrame,
    geometry: pd.DataFrame,
    source: str,
    min_amv_points_per_level: int = 1,
    include_pressure_in_key: bool = True,
) -> pd.DataFrame:
    """Aggregate AMV winds using a point-level IGRA geometry cache."""

    if int(min_amv_points_per_level) < 1:
        raise ValueError("min_amv_points_per_level must be at least 1")
    if geometry.empty:
        return pd.DataFrame()

    frame = add_amv_point_keys(amv, include_pressure=include_pressure_in_key).copy()
    required = ["u_amv", "v_amv", "pressure_hpa"]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise KeyError(f"AMV table is missing required wind columns: {missing}")

    # Keep only required AMV fields before merging to limit memory use.
    amv_cols = ["_amv_key_kind", "_amv_key1", "_amv_key2", "_amv_key3", "u_amv", "v_amv", "pressure_hpa"]
    optional_cols = [
        "longitude", "latitude", "speed_amv", "tracker", "amv_source", "pressure_source",
        "official_id", "row_center_original", "col_center_original",
    ]
    for col in optional_cols:
        if col in frame.columns and col not in amv_cols:
            amv_cols.append(col)
    subset = frame[amv_cols].copy()

    joined = geometry.merge(
        subset,
        on=["_amv_key_kind", "_amv_key1", "_amv_key2", "_amv_key3"],
        how="inner",
        suffixes=("", "_amv_product"),
    )
    if joined.empty:
        return pd.DataFrame()

    for column in ["u_amv", "v_amv", "pressure_hpa", "weight", "distance_km"]:
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
    valid = (
        np.isfinite(joined["u_amv"])
        & np.isfinite(joined["v_amv"])
        & np.isfinite(joined["pressure_hpa"])
        & np.isfinite(joined["u_sonde"])
        & np.isfinite(joined["v_sonde"])
        & np.isfinite(joined["weight"])
        & (joined["weight"] > 0)
    )
    joined = joined.loc[valid].copy()
    if joined.empty:
        return pd.DataFrame()

    joined["uw"] = joined["u_amv"] * joined["weight"]
    joined["vw"] = joined["v_amv"] * joined["weight"]
    joined["pw"] = joined["pressure_hpa"] * joined["weight"]
    joined["dw"] = joined["distance_km"] * joined["weight"]

    group_cols = ["station_id", "sounding_time", "level_index"]
    grouped = joined.groupby(group_cols, sort=False, dropna=False)

    agg = grouped.agg(
        station_name=("station_name", "first"),
        station_lat=("station_lat", "first"),
        station_lon=("station_lon", "first"),
        time_difference_minutes=("time_difference_minutes", "first"),
        **({
            "obs_lat": ("obs_lat", "first"),
            "obs_lon": ("obs_lon", "first"),
            "drift_km": ("drift_km", "first"),
        } if {"obs_lat", "obs_lon", "drift_km"}.issubset(joined.columns) else {}),
        sonde_pressure_hpa=("sonde_pressure_hpa", "first"),
        u_sonde=("u_sonde", "first"),
        v_sonde=("v_sonde", "first"),
        pressure_layer=("pressure_layer", "first"),
        amv_neighbor_count=("u_amv", "size"),
        weight_sum=("weight", "sum"),
        uw_sum=("uw", "sum"),
        vw_sum=("vw", "sum"),
        pw_sum=("pw", "sum"),
        dw_sum=("dw", "sum"),
        pressure_difference_abs_mean=("pressure_difference_hpa", lambda x: float(np.nanmean(np.abs(pd.to_numeric(x, errors="coerce"))))),
    ).reset_index()

    agg = agg.loc[(agg["amv_neighbor_count"] >= int(min_amv_points_per_level)) & (agg["weight_sum"] > 0)].copy()
    if agg.empty:
        return pd.DataFrame()

    agg["u_amv"] = agg["uw_sum"] / agg["weight_sum"]
    agg["v_amv"] = agg["vw_sum"] / agg["weight_sum"]
    agg["amv_pressure_hpa"] = agg["pw_sum"] / agg["weight_sum"]
    agg["distance_mean_km"] = agg["dw_sum"] / agg["weight_sum"]
    agg["pressure_difference_hpa"] = agg["amv_pressure_hpa"] - agg["sonde_pressure_hpa"]

    agg["speed_amv"] = np.hypot(agg["u_amv"], agg["v_amv"])
    agg["speed_sonde"] = np.hypot(agg["u_sonde"], agg["v_sonde"])
    agg["speed_bias_amv_minus_sonde"] = agg["speed_amv"] - agg["speed_sonde"]
    agg["vector_difference_amv_vs_sonde"] = np.hypot(agg["u_amv"] - agg["u_sonde"], agg["v_amv"] - agg["v_sonde"])
    agg["signed_direction_difference_deg"] = direction_difference(
        agg["u_amv"], agg["v_amv"], agg["u_sonde"], agg["v_sonde"]
    )
    agg["abs_direction_difference_deg"] = np.abs(agg["signed_direction_difference_deg"])

    tracker = str(frame["tracker"].iloc[0]) if "tracker" in frame.columns and len(frame) else ""
    amv_source = str(frame["amv_source"].iloc[0]) if "amv_source" in frame.columns and len(frame) else "amv"
    pressure_source = str(frame["pressure_source"].iloc[0]) if "pressure_source" in frame.columns and len(frame) else ""

    agg["source"] = source
    agg["tracker"] = tracker
    agg["amv_source"] = amv_source
    agg["pressure_source"] = pressure_source
    agg["aggregation_method"] = str(geometry.get("aggregation_method", "geometry_cache")) if "aggregation_method" in geometry.columns else "geometry_cache"

    # Keep output close to collocate_amv_with_igra_levels.
    out_cols = [
        "source", "tracker", "amv_source", "pressure_source",
        "station_id", "station_name", "station_lat", "station_lon", "sounding_time",
        "time_difference_minutes", "obs_lat", "obs_lon", "drift_km",
        "sonde_pressure_hpa", "amv_pressure_hpa",
        "pressure_difference_hpa", "pressure_layer", "amv_neighbor_count",
        "distance_mean_km", "u_amv", "v_amv", "speed_amv",
        "u_sonde", "v_sonde", "speed_sonde", "speed_bias_amv_minus_sonde",
        "vector_difference_amv_vs_sonde", "signed_direction_difference_deg",
        "abs_direction_difference_deg", "aggregation_method",
    ]
    for column in out_cols:
        if column not in agg.columns:
            agg[column] = np.nan
    return agg[out_cols].reset_index(drop=True)


def collocate_with_geometry_or_load(
    *,
    geometry_cache_path: str | Path,
    collocation_cache_path: str | Path | None,
    amv: pd.DataFrame,
    igra_profiles: list[Any],
    source: str,
    product_time,
    min_amv_points_per_level: int = 1,
    reuse_cache: bool = True,
    save_cache: bool = True,
    compressed: bool = False,
    include_pressure_in_key: bool = True,
    **geometry_kwargs,
) -> pd.DataFrame:
    """Build/load point geometry, then aggregate one AMV product.

    ``geometry_cache_path`` can be shared across trackers with the same geometry;
    ``collocation_cache_path`` is optional and stores the final aggregated table
    for one tracker/mask.
    """

    collocation_cache = Path(collocation_cache_path) if collocation_cache_path is not None else None
    geometry_fingerprint = amv_geometry_fingerprint(amv, include_pressure=include_pressure_in_key)
    collocation_metadata = {
        "cache_type": "igra_station_level_collocation_from_geometry",
        "source": source,
        "product_time": str(product_time),
        "amv_geometry_fingerprint": geometry_fingerprint,
        "min_amv_points_per_level": int(min_amv_points_per_level),
        "include_pressure_in_key": bool(include_pressure_in_key),
        **geometry_kwargs,
    }
    if reuse_cache and collocation_cache is not None and collocation_cache.exists():
        cached_metadata = read_collocation_npz_metadata(collocation_cache)
        if _cache_metadata_matches(cached_metadata, collocation_metadata):
            return load_collocation_npz(collocation_cache)
        print(f"Ignoring stale IGRA collocation cache with mismatched metadata: {collocation_cache}")

    geometry = geometry_or_load(
        cache_path=geometry_cache_path,
        amv=amv,
        igra_profiles=igra_profiles,
        source=source,
        product_time=product_time,
        reuse_cache=reuse_cache,
        save_cache=save_cache,
        compressed=compressed,
        include_pressure_in_key=include_pressure_in_key,
        **geometry_kwargs,
    )
    table = collocate_from_igra_geometry(
        amv=amv,
        geometry=geometry,
        source=source,
        min_amv_points_per_level=min_amv_points_per_level,
        include_pressure_in_key=include_pressure_in_key,
    )
    if save_cache and collocation_cache is not None:
        save_collocation_npz(table, collocation_cache, metadata=collocation_metadata, compressed=compressed)
    return table
