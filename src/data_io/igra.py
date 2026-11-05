"""IGRA independent-reference wind profile loading and QC.

Each wind profile also carries per-level observation times and balloon
positions:

* level time = reported release time (header RELTIME) + elapsed time since
  launch (record ETIME); missing ETIME is interpolated in log-pressure from
  other levels, or estimated from height with a constant 5 m/s ascent rate;
  if RELTIME is missing the nominal sounding time is used and flagged.
* balloon drift is integrated upward from the launch site using the layer-mean
  wind multiplied by the layer time interval (Laroche and Sarrazin, 2013;
  Voggenberger et al., 2024).  Heights missing at wind levels are interpolated
  in log-pressure from reported geopotential heights, falling back to the
  standard atmosphere.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from config import repo_path, resolve_case_config
from .common import normalize_aoi


MISSING = {-9999, -8888}

# Constant balloon ascent rate used when elapsed times are unavailable
# (Laroche and Sarrazin, 2013; Voggenberger et al., 2024).
DEFAULT_ASCENT_RATE_MS = 5.0
PARSER_VERSION = "igra2_levels_reltime_etime_drift_v1"


@dataclass(frozen=True)
class WindProfile:
    """One quality-controlled radiosonde wind profile."""

    station_id: str
    station_name: str
    station_lat: float
    station_lon: float
    sounding_time: pd.Timestamp
    pressure_hpa: np.ndarray
    u_ms: np.ndarray
    v_ms: np.ndarray
    height_m: np.ndarray | None = None
    source_file: str = ""
    # Per-level observation geometry (same length/order as pressure_hpa).
    release_time: pd.Timestamp | None = None
    level_time: np.ndarray | None = None  # datetime64[ns] (UTC)
    level_lat: np.ndarray | None = None
    level_lon: np.ndarray | None = None
    drift_km: np.ndarray | None = None
    elapsed_seconds: np.ndarray | None = None
    time_source: str = "nominal"  # "release+etime", "release+etime_interp", "release+ascent", "nominal"
    drift_source: str = "none"  # "etime", "ascent_rate", "none"


def _as_config(case_config) -> dict[str, Any]:
    if isinstance(case_config, Mapping):
        return dict(case_config)
    return resolve_case_config(case_config)


def _resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return repo_path(value)


def igra_root(case_config=None, root: str | Path | None = None) -> Path:
    """Return local IGRA root directory.

    Preferred project layout:
        <paths["igra"]>/igra2-station-list.txt
        <paths["igra"]>/stations/*-data.txt.zip
    """

    if root is not None:
        return _resolve_path(root)
    if case_config is not None:
        config = _as_config(case_config)
        paths = config.get("paths", {}) or {}
        if paths.get("igra") is not None:
            return _resolve_path(paths["igra"])
        if config.get("igra_root") is not None:
            return _resolve_path(config["igra_root"])
    return repo_path("cache/reference_data/igra")


def igra_station_list_path(case_config=None, root: str | Path | None = None) -> Path:
    config = _as_config(case_config) if case_config is not None else {}
    if config.get("igra_station_list") is not None:
        return _resolve_path(config["igra_station_list"])
    return igra_root(config, root) / "igra2-station-list.txt"


def igra_station_dir(case_config=None, root: str | Path | None = None) -> Path:
    config = _as_config(case_config) if case_config is not None else {}
    if config.get("igra_station_dir") is not None:
        return _resolve_path(config["igra_station_dir"])
    return igra_root(config, root) / "stations"


def parse_station_list(path: str | Path) -> pd.DataFrame:
    """Parse the public fixed-width IGRA station inventory."""

    rows = []
    for line in Path(path).read_text(encoding="ascii", errors="replace").splitlines():
        try:
            rows.append(
                {
                    "station_id": line[0:11].strip(),
                    "station_lat": float(line[12:20]),
                    "station_lon": float(line[21:30]),
                    "station_name": line[41:71].strip(),
                    "first_year": int(line[72:76]),
                    "last_year": int(line[77:81]),
                }
            )
        except (ValueError, IndexError):
            continue
    return pd.DataFrame(rows)


def _wind(direction_deg: np.ndarray, speed_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert meteorological wind direction/speed to u/v components."""

    direction = np.deg2rad(direction_deg)
    return -speed_ms * np.sin(direction), -speed_ms * np.cos(direction)


def _parse_int(text: str) -> int | None:
    try:
        return int(text)
    except ValueError:
        return None


def _parse_etime_seconds(text: str) -> float:
    """IGRA2 ETIME (MMMSS, not zero padded) to seconds; NaN when missing."""

    value = _parse_int(text)
    if value is None or value in MISSING or value < 0:
        return np.nan
    minutes, seconds = divmod(value, 100)
    if seconds >= 60:
        return np.nan
    return float(minutes * 60 + seconds)


def _release_time(nominal: pd.Timestamp | None, header: str) -> pd.Timestamp | None:
    """Release time from header RELTIME (HHMM); None when unavailable.

    RELTIME is given in UTC without a date.  The release is attached to the
    calendar day that places it closest to the nominal time (e.g. 23:15 on the
    previous day for a nominal 00 UTC sounding).  ``HH99`` (hour only) uses HH:00.
    """

    if nominal is None:
        return None
    value = _parse_int(header[27:31])
    if value is None or value == 9999 or value < 0:
        return None
    hour, minute = divmod(value, 100)
    if minute == 99:
        minute = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    candidate = nominal.normalize() + pd.Timedelta(hours=hour, minutes=minute)
    delta = candidate - nominal
    if delta > pd.Timedelta(hours=12):
        candidate -= pd.Timedelta(days=1)
    elif delta < -pd.Timedelta(hours=12):
        candidate += pd.Timedelta(days=1)
    return candidate


def _standard_atmosphere_height(pressure_hpa: np.ndarray) -> np.ndarray:
    p = np.asarray(pressure_hpa, dtype=float)
    return 44330.8 * (1.0 - (p / 1013.25) ** 0.190263)


def _interp_logp(target_p: np.ndarray, known_p: np.ndarray, known_v: np.ndarray) -> np.ndarray:
    """Linear interpolation in log-pressure; NaN outside the known range."""

    target = np.asarray(target_p, dtype=float)
    p = np.asarray(known_p, dtype=float)
    v = np.asarray(known_v, dtype=float)
    ok = np.isfinite(p) & np.isfinite(v) & (p > 0)
    out = np.full(target.shape, np.nan)
    if ok.sum() < 2:
        return out
    x = -np.log(p[ok])
    order = np.argsort(x)
    x, y = x[order], v[ok][order]
    x, unique_index = np.unique(x, return_index=True)
    y = y[unique_index]
    tx = -np.log(np.clip(target, 1e-6, None))
    inside = (tx >= x[0]) & (tx <= x[-1])
    out[inside] = np.interp(tx[inside], x, y)
    return out


def _level_heights(pressure: np.ndarray, gph: np.ndarray, all_p: np.ndarray, all_gph: np.ndarray) -> np.ndarray:
    """Heights at wind levels: reported GPH, else log-p interpolation, else std atm."""

    height = np.asarray(gph, dtype=float).copy()
    missing = ~np.isfinite(height)
    if missing.any():
        height[missing] = _interp_logp(pressure[missing], all_p, all_gph)
    missing = ~np.isfinite(height)
    if missing.any():
        # Standard-atmosphere shape anchored to the reported heights when possible.
        std = _standard_atmosphere_height(pressure)
        known = np.isfinite(all_gph) & np.isfinite(all_p)
        offset = 0.0
        if known.any():
            offset = float(np.nanmedian(all_gph[known] - _standard_atmosphere_height(all_p[known])))
        height[missing] = std[missing] + offset
    return height


def balloon_level_geometry(
    *,
    pressure_hpa: np.ndarray,
    u_ms: np.ndarray,
    v_ms: np.ndarray,
    height_m: np.ndarray,
    elapsed_seconds: np.ndarray,
    launch_lat: float,
    launch_lon: float,
    surface_height_m: float | None = None,
    ascent_rate_ms: float = DEFAULT_ASCENT_RATE_MS,
) -> dict[str, Any]:
    """Elapsed time and drifted position at each wind level.

    Levels must be ordered from the surface upward (decreasing pressure).
    The displacement over each layer is the layer-mean wind multiplied by the
    layer time interval; below the lowest wind level the lowest wind is used.
    """

    p = np.asarray(pressure_hpa, dtype=float)
    u = np.asarray(u_ms, dtype=float)
    v = np.asarray(v_ms, dtype=float)
    z = np.asarray(height_m, dtype=float)
    t = np.asarray(elapsed_seconds, dtype=float).copy()
    n = p.size
    if n == 0:
        empty = np.array([], dtype=float)
        return {"elapsed_seconds": empty, "lat": empty, "lon": empty, "drift_km": empty, "drift_source": "none", "time_kind": "none"}

    known_t = np.isfinite(t)
    if known_t.sum() >= 2:
        t[~known_t] = _interp_logp(p[~known_t], p[known_t], t[known_t])
        time_kind = "etime" if known_t.all() else "etime_interp"
    else:
        time_kind = "none"
    still = ~np.isfinite(t)
    if still.any():
        z0 = surface_height_m if surface_height_m is not None and np.isfinite(surface_height_m) else np.nanmin(z)
        if not np.isfinite(z0):
            z0 = 0.0
        t[still] = np.clip(z[still] - z0, 0.0, None) / float(ascent_rate_ms)
        if time_kind == "none":
            time_kind = "ascent_rate"
    # Elapsed time must not decrease with height.
    t = np.maximum.accumulate(np.where(np.isfinite(t), t, 0.0))

    dx = np.zeros(n)
    dy = np.zeros(n)
    dx[0] = u[0] * t[0]
    dy[0] = v[0] * t[0]
    for k in range(1, n):
        dt = max(t[k] - t[k - 1], 0.0)
        dx[k] = dx[k - 1] + 0.5 * (u[k - 1] + u[k]) * dt
        dy[k] = dy[k - 1] + 0.5 * (v[k - 1] + v[k]) * dt

    distance = np.hypot(dx, dy)
    azimuth = np.degrees(np.arctan2(dx, dy))
    try:
        from pyproj import Geod

        lon, lat, _ = Geod(ellps="WGS84").fwd(
            np.full(n, float(launch_lon)), np.full(n, float(launch_lat)), azimuth, distance
        )
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
    except Exception:  # pragma: no cover - pyproj is a project dependency
        radius = 6371008.8
        lat = float(launch_lat) + np.degrees(dy / radius)
        lon = float(launch_lon) + np.degrees(dx / (radius * np.cos(np.deg2rad(float(launch_lat)))))
    drift_source = "etime" if time_kind in {"etime", "etime_interp"} else "ascent_rate"
    return {
        "elapsed_seconds": t,
        "lat": lat,
        "lon": lon,
        "drift_km": distance / 1000.0,
        "drift_source": drift_source,
        "time_kind": time_kind,
    }


def _as_utc_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _case_target_time(config: Mapping[str, Any]) -> pd.Timestamp:
    """Choose the reference time for IGRA profile filtering."""

    for key in ("validation_time", "target_time", "end_time", "product_time"):
        if config.get(key) is not None:
            return _as_utc_timestamp(config[key])
    frame_times = config.get("frame_times")
    if frame_times:
        return _as_utc_timestamp(frame_times[-1])
    raise KeyError("config must contain end_time, product_time, validation_time, or frame_times")


def _profile_time_selected(
    time: pd.Timestamp,
    *,
    start_time: pd.Timestamp | None,
    end_time: pd.Timestamp | None,
    target_time: pd.Timestamp | None,
    time_window_hours: float | None,
    target_dates: set[pd.Timestamp] | None,
) -> bool:
    if start_time is not None and end_time is not None:
        return start_time <= time <= end_time
    if target_time is not None and time_window_hours is not None:
        delta = abs((time - target_time).total_seconds()) / 3600.0
        return delta <= time_window_hours
    if target_dates is not None:
        return time.normalize() in target_dates
    return True


def parse_station_archive(
    path: str | Path,
    station: pd.Series | dict,
    target_dates: set[pd.Timestamp] | None = None,
    minimum_levels: int = 5,
    *,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    target_time: pd.Timestamp | None = None,
    time_window_hours: float | None = None,
) -> list[WindProfile]:
    """Parse wind-bearing IGRA levels without vertical extrapolation."""

    source = Path(path)
    metadata = dict(station)

    # If station metadata came from ``stations.set_index("station_id").loc[...]``,
    # the index value is not guaranteed to remain as a normal column. Preserve it
    # here so WindProfile construction is robust.
    if "station_id" not in metadata or pd.isna(metadata.get("station_id")):
        station_name_from_index = getattr(station, "name", None)
        if station_name_from_index is not None:
            metadata["station_id"] = str(station_name_from_index).strip()
        else:
            metadata["station_id"] = source.name.split("-data", 1)[0].strip()

    profiles: list[WindProfile] = []
    try:
        archive = ZipFile(source)
    except BadZipFile:
        return profiles

    with archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if not members:
            return profiles
        member = members[0]
        with archive.open(member) as handle:
            while True:
                raw = handle.readline()
                if not raw:
                    break
                header = raw.decode("ascii", errors="replace").rstrip()
                if not header.startswith("#"):
                    continue
                try:
                    count = int(header[32:36])
                except ValueError:
                    continue
                try:
                    time = pd.Timestamp(
                        year=int(header[13:17]),
                        month=int(header[18:20]),
                        day=int(header[21:23]),
                        hour=int(header[24:26]),
                        tz="UTC",
                    )
                except ValueError:
                    time = None
                selected = time is not None and _profile_time_selected(
                    time,
                    start_time=start_time,
                    end_time=end_time,
                    target_time=target_time,
                    time_window_hours=time_window_hours,
                    target_dates=target_dates,
                )
                levels = []
                all_levels = []
                surface_height = np.nan
                for _ in range(count):
                    record = handle.readline().decode("ascii", errors="replace")
                    if not selected:
                        continue
                    pressure_pa = _parse_int(record[9:15])
                    if pressure_pa is None or pressure_pa in MISSING or pressure_pa <= 0:
                        continue
                    height = _parse_int(record[16:21])
                    height_m = np.nan if height is None or height in MISSING else float(height)
                    elapsed = _parse_etime_seconds(record[3:8])
                    all_levels.append((pressure_pa / 100.0, height_m, elapsed))
                    if record[1:2] == "1" and np.isfinite(height_m):
                        surface_height = height_m
                    direction = _parse_int(record[40:45])
                    speed_tenths = _parse_int(record[46:51])
                    if (
                        direction is None
                        or speed_tenths is None
                        or direction in MISSING
                        or speed_tenths in MISSING
                        or not 0 <= direction <= 360
                        or not 0 <= speed_tenths <= 1500
                    ):
                        continue
                    levels.append(
                        (
                            pressure_pa / 100.0,
                            height_m,
                            float(direction),
                            speed_tenths / 10.0,
                            elapsed,
                        )
                    )
                if len(levels) < minimum_levels:
                    continue
                values = np.asarray(levels, dtype=float)
                # Use descending pressure order: near-surface to upper levels.
                order = np.argsort(values[:, 0])[::-1]
                values = values[order]
                unique_pressure, unique_index = np.unique(values[:, 0], return_index=True)
                values = values[np.sort(unique_index)]
                u, v = _wind(values[:, 2], values[:, 3])

                all_values = np.asarray(all_levels, dtype=float).reshape(-1, 3)
                level_p = values[:, 0]
                heights = _level_heights(level_p, values[:, 1], all_values[:, 0], all_values[:, 1])
                elapsed_s = values[:, 4].copy()
                missing_elapsed = ~np.isfinite(elapsed_s)
                if missing_elapsed.any():
                    elapsed_s[missing_elapsed] = _interp_logp(
                        level_p[missing_elapsed], all_values[:, 0], all_values[:, 2]
                    )

                header_lat = _parse_int(header[55:62])
                header_lon = _parse_int(header[63:71])
                launch_lat = float(metadata["station_lat"])
                launch_lon = float(metadata["station_lon"])
                if header_lat is not None and header_lat not in MISSING and abs(header_lat) <= 900000:
                    launch_lat = header_lat / 10000.0
                if header_lon is not None and header_lon not in MISSING and abs(header_lon) <= 1800000:
                    launch_lon = header_lon / 10000.0

                geometry = balloon_level_geometry(
                    pressure_hpa=level_p,
                    u_ms=u,
                    v_ms=v,
                    height_m=heights,
                    elapsed_seconds=elapsed_s,
                    launch_lat=launch_lat,
                    launch_lon=launch_lon,
                    surface_height_m=surface_height,
                )
                release = _release_time(time, header)
                if release is not None:
                    level_time = (
                        release.tz_convert("UTC").tz_localize(None).to_datetime64()
                        + (geometry["elapsed_seconds"] * 1e9).astype("timedelta64[ns]")
                    )
                    time_source = {
                        "etime": "release+etime",
                        "etime_interp": "release+etime_interp",
                    }.get(geometry["time_kind"], "release+ascent")
                else:
                    level_time = np.full(
                        level_p.size, time.tz_convert("UTC").tz_localize(None).to_datetime64()
                    )
                    time_source = "nominal"

                profiles.append(
                    WindProfile(
                        str(metadata["station_id"]).strip(),
                        str(metadata.get("station_name", "")),
                        float(metadata["station_lat"]),
                        float(metadata["station_lon"]),
                        time,
                        level_p,
                        u,
                        v,
                        heights,
                        source.name,
                        release_time=release,
                        level_time=level_time.astype("datetime64[ns]"),
                        level_lat=np.asarray(geometry["lat"], dtype=float),
                        level_lon=np.asarray(geometry["lon"], dtype=float),
                        drift_km=np.asarray(geometry["drift_km"], dtype=float),
                        elapsed_seconds=np.asarray(geometry["elapsed_seconds"], dtype=float),
                        time_source=time_source,
                        drift_source=str(geometry["drift_source"]),
                    )
                )
    return profiles


def _filter_stations_by_aoi(stations: pd.DataFrame, aoi=None) -> pd.DataFrame:
    bounds = normalize_aoi(aoi)
    if bounds is None:
        return stations
    west, east, south, north = bounds
    if west <= east:
        lon_mask = stations.station_lon.between(west, east)
    else:
        lon_mask = (stations.station_lon >= west) | (stations.station_lon <= east)
    return stations.loc[lon_mask & stations.station_lat.between(south, north)]


def _profile_window_dates(
    target_time: pd.Timestamp,
    time_window_hours: float,
) -> set[pd.Timestamp]:
    start = target_time - pd.Timedelta(hours=time_window_hours)
    end = target_time + pd.Timedelta(hours=time_window_hours)
    days = pd.date_range(start.normalize(), end.normalize(), freq="D", tz="UTC")
    return {day.normalize() for day in days}


def load_igra_profiles(
    case_config,
    aoi=None,
    *,
    target_time=None,
    time_window_hours: float = 6.0,
    start_time=None,
    end_time=None,
    minimum_levels: int = 5,
    root: str | Path | None = None,
    max_archives: int | None = None,
) -> list[WindProfile]:
    """Load local IGRA profiles near the case time and optional AOI.

    For an AMV ending at 11:45 UTC, use the default target time of
    ``config['end_time']`` and a window such as 3--6 hours to include the nearby
    12 UTC radiosonde profiles.
    """

    config = _as_config(case_config)
    station_list = igra_station_list_path(config, root)
    station_dir = igra_station_dir(config, root)
    if not station_list.exists() or not station_dir.exists():
        return []

    stations = parse_station_list(station_list)
    if stations.empty:
        return []
    requested_aoi = aoi if aoi is not None else config.get("aoi")
    stations = _filter_stations_by_aoi(stations, requested_aoi)
    if stations.empty:
        return []

    if start_time is not None and end_time is not None:
        start_ts = _as_utc_timestamp(start_time)
        end_ts = _as_utc_timestamp(end_time)
        target_ts = None
        target_dates = {
            day.normalize()
            for day in pd.date_range(start_ts.normalize(), end_ts.normalize(), freq="D", tz="UTC")
        }
    else:
        target_ts = _as_utc_timestamp(target_time) if target_time is not None else _case_target_time(config)
        start_ts = None
        end_ts = None
        target_dates = _profile_window_dates(target_ts, time_window_hours)

    # Keep station_id both as an index and as a normal column. Otherwise
    # station_lookup.loc[station_id] may return a Series without a station_id
    # field, which later causes KeyError("station_id").
    station_lookup = (
        stations.drop_duplicates("station_id")
        .set_index("station_id", drop=False)
    )
    profiles: list[WindProfile] = []
    archives = sorted(station_dir.glob("*-data.txt.zip"))
    if max_archives is not None:
        archives = archives[:max_archives]
    for archive in archives:
        station_id = archive.name.split("-data", 1)[0]
        if station_id not in station_lookup.index:
            continue
        profiles.extend(
            parse_station_archive(
                archive,
                station_lookup.loc[station_id],
                target_dates=target_dates,
                minimum_levels=minimum_levels,
                start_time=start_ts,
                end_time=end_ts,
                target_time=target_ts,
                time_window_hours=time_window_hours,
            )
        )
    return profiles


def profiles_to_dataframe(profiles: Sequence[WindProfile]) -> pd.DataFrame:
    """Flatten profiles to a level-by-level DataFrame for diagnostics."""

    rows: list[dict[str, Any]] = []
    for profile in profiles:
        height = profile.height_m if profile.height_m is not None else np.full_like(profile.pressure_hpa, np.nan)
        n_levels = len(profile.pressure_hpa)
        level_lat = getattr(profile, "level_lat", None)
        level_lon = getattr(profile, "level_lon", None)
        level_time = getattr(profile, "level_time", None)
        drift = getattr(profile, "drift_km", None)
        level_lat = np.full(n_levels, profile.station_lat) if level_lat is None else level_lat
        level_lon = np.full(n_levels, profile.station_lon) if level_lon is None else level_lon
        level_time = np.full(n_levels, np.datetime64("NaT", "ns")) if level_time is None else level_time
        drift = np.full(n_levels, np.nan) if drift is None else drift
        for index, (pressure, u, v, z) in enumerate(zip(profile.pressure_hpa, profile.u_ms, profile.v_ms, height)):
            rows.append(
                {
                    "station_id": profile.station_id,
                    "station_name": profile.station_name,
                    "station_lat": profile.station_lat,
                    "station_lon": profile.station_lon,
                    "sounding_time": profile.sounding_time,
                    "pressure_hpa": float(pressure),
                    "u_ms": float(u),
                    "v_ms": float(v),
                    "height_m": float(z) if np.isfinite(z) else np.nan,
                    "level_lat": float(level_lat[index]),
                    "level_lon": float(level_lon[index]),
                    "level_time": pd.Timestamp(level_time[index]),
                    "drift_km": float(drift[index]),
                    "time_source": getattr(profile, "time_source", "nominal"),
                    "source_file": profile.source_file,
                }
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Notebook quicklook helpers
# -----------------------------------------------------------------------------


def igra_selected_station_csv_path(case_config=None, root: str | Path | None = None) -> Path:
    """Return the station-selection CSV written by download_igra(..., mode='search')."""

    return igra_root(case_config, root) / "index" / "igra_selected_stations.csv"


def load_igra_selected_stations(
    case_config=None,
    aoi=None,
    *,
    root: str | Path | None = None,
    require_selected_csv: bool = True,
) -> pd.DataFrame:
    """Load the selected IGRA station table without parsing station archives.

    This is the fast path for station-location quicklooks. It reads
    ``<paths['igra']>/index/igra_selected_stations.csv`` when available. If the
    CSV is absent and ``require_selected_csv`` is False, it falls back to the
    public station inventory and applies the optional AOI filter.
    """

    selected_csv = igra_selected_station_csv_path(case_config, root)
    if selected_csv.exists():
        stations = pd.read_csv(selected_csv)
    else:
        if require_selected_csv:
            raise FileNotFoundError(
                f"Selected IGRA station CSV not found: {selected_csv}. "
                "Run download_igra(..., mode='search') first."
            )
        station_list = igra_station_list_path(case_config, root)
        if not station_list.exists():
            raise FileNotFoundError(f"IGRA station list not found: {station_list}")
        stations = parse_station_list(station_list)

    requested_aoi = aoi
    if requested_aoi is None and case_config is not None:
        config = _as_config(case_config)
        requested_aoi = config.get("aoi")

    # Normalize common downloaded CSV column names to the names used by data_io.
    rename = {}
    if "lat" in stations.columns and "station_lat" not in stations.columns:
        rename["lat"] = "station_lat"
    if "lon" in stations.columns and "station_lon" not in stations.columns:
        rename["lon"] = "station_lon"
    if rename:
        stations = stations.rename(columns=rename)

    if requested_aoi is not None and {"station_lat", "station_lon"}.issubset(stations.columns):
        stations = _filter_stations_by_aoi(stations, requested_aoi)

    return stations.reset_index(drop=True)


def load_igra_quicklook_profiles(
    case_config,
    aoi=None,
    *,
    target_time=None,
    time_window_hours: float = 6.0,
    minimum_levels: int = 5,
    max_archives: int | None = 5,
    root: str | Path | None = None,
) -> tuple[list[WindProfile], pd.DataFrame]:
    """Load a small IGRA profile subset and return both profiles and a DataFrame.

    This keeps notebook quicklook cells short and avoids accidentally parsing all
    downloaded period-of-record station archives. Set ``max_archives=None`` only
    when a full local scan is intended.
    """

    profiles = load_igra_profiles(
        case_config,
        aoi=aoi,
        target_time=target_time,
        time_window_hours=time_window_hours,
        minimum_levels=minimum_levels,
        root=root,
        max_archives=max_archives,
    )
    return profiles, profiles_to_dataframe(profiles)
