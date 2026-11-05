"""Official operational-reference AMV loading.

The recommended notebook workflow is explicit-file based:
``load_official_amv_file(path)``.

The older config-driven ``load_official_amv(case_config)`` wrapper is kept for
compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
import re

from netCDF4 import Dataset
import numpy as np
import pandas as pd

from config import REPO_ROOT, repo_path, resolve_case_config
from .common import normalize_aoi


@dataclass(frozen=True)
class OfficialAmv:
    """Official AMV table, source path, and product metadata."""

    frame: pd.DataFrame
    path: Path
    metadata: dict[str, Any]


def _as_config(case_config) -> dict[str, Any]:
    if isinstance(case_config, Mapping):
        return dict(case_config)
    return resolve_case_config(case_config)


def _resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return repo_path(value)


def _parse_datetime(value) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)


def _case_target_time(config: Mapping[str, Any]) -> datetime | None:
    for key in ("validation_time", "target_time", "end_time", "product_time"):
        if config.get(key) is not None:
            return _parse_datetime(config[key])
    frame_times = config.get("frame_times")
    if frame_times:
        return _parse_datetime(frame_times[-1])
    return None


def _parse_interval_from_name(path: Path) -> tuple[datetime, datetime] | None:
    text = path.name
    match = re.search(r"((?:19|20)\d{12})Z?_((?:19|20)\d{12})Z?", text)
    if not match:
        return None
    try:
        return (
            datetime.strptime(match.group(1), "%Y%m%d%H%M%S"),
            datetime.strptime(match.group(2), "%Y%m%d%H%M%S"),
        )
    except ValueError:
        return None


def _read(dataset: Dataset, names: tuple[str, ...] | list[str], size: int) -> np.ndarray:
    for name in names:
        if name in dataset.variables:
            values = np.ma.filled(dataset.variables[name][:], np.nan).astype(float).ravel()
            return values
    return np.full(size, np.nan)


def _windows_path(path: Path) -> str:
    value = str(path.resolve())
    return "\\\\?\\" + value if len(value) >= 248 and not value.startswith("\\\\?\\") else value


def _pressure_to_hpa(values: np.ndarray, units: str = "") -> np.ndarray:
    out = values.astype(float).copy()
    finite = np.isfinite(out)
    unit_text = units.lower()
    if "pa" in unit_text and "hpa" not in unit_text:
        out = out / 100.0
    elif finite.any() and np.nanmedian(out[finite]) > 2000:
        out = out / 100.0
    out[(out < 50) | (out > 1100)] = np.nan
    return out


def _uv_from_speed_direction(speed: np.ndarray, direction_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = np.deg2rad(direction_deg)
    return -speed * np.sin(direction), -speed * np.cos(direction)


def load_official_amv_file(path: str | Path, aoi=None) -> OfficialAmv:
    """Load one official AMV NetCDF file and optionally filter by lon/lat AOI."""

    source = _resolve_path(path)
    if not source.exists():
        raise FileNotFoundError(f"Official AMV file not found: {source}")

    with Dataset(_windows_path(source), "r") as dataset:
        dimension = "numAMV" if "numAMV" in dataset.dimensions else next(iter(dataset.dimensions))
        size = len(dataset.dimensions[dimension])

        lat = _read(dataset, ("latitude", "lat"), size)
        lon = _read(dataset, ("longitude", "lon"), size)
        u = _read(dataset, ("u", "u_wind", "wind_u"), size)
        v = _read(dataset, ("v", "v_wind", "wind_v"), size)
        speed = _read(dataset, ("speed", "wind_speed"), size)
        direction = _read(dataset, ("dir", "direction", "wind_direction"), size)
        pressure_raw = _read(dataset, ("height", "pressure", "air_pressure", "p"), size)
        qi = _read(dataset, ("qi", "quality", "quality_index"), size)

        if np.isnan(u).all() and not np.isnan(speed).all() and not np.isnan(direction).all():
            u, v = _uv_from_speed_direction(speed, direction)
        if np.isnan(speed).all() and not np.isnan(u).all() and not np.isnan(v).all():
            speed = np.sqrt(u * u + v * v)

        pressure_units = ""
        for name in ("height", "pressure", "air_pressure", "p"):
            if name in dataset.variables:
                pressure_units = str(getattr(dataset.variables[name], "units", ""))
                break
        pressure = _pressure_to_hpa(pressure_raw, pressure_units)

        metadata = {name: getattr(dataset, name) for name in dataset.ncattrs()}
        metadata["selected_file"] = str(source)
        metadata["interval_from_filename"] = _parse_interval_from_name(source)
        metadata["pressure_units_original"] = pressure_units

    frame = pd.DataFrame(
        {
            "official_id": np.arange(size),
            "lat": lat,
            "lon": lon,
            "u_official": u,
            "v_official": v,
            "speed_official": speed,
            "direction_official": direction,
            "p_official": pressure,
            "official_quality": qi,
        }
    ).dropna(subset=["lat", "lon", "u_official", "v_official", "p_official"])

    bounds = normalize_aoi(aoi)
    if bounds is not None:
        west, east, south, north = bounds
        if west <= east:
            lon_mask = frame.lon.between(west, east)
        else:
            lon_mask = (frame.lon >= west) | (frame.lon <= east)
        frame = frame.loc[lon_mask & frame.lat.between(south, north)]

    return OfficialAmv(frame.reset_index(drop=True), source, metadata)


# -----------------------------------------------------------------------------
# Backward-compatible config-driven loader
# -----------------------------------------------------------------------------


def _candidate_files(config: dict) -> list[Path]:
    candidates: list[Path] = []

    for key in ("official_amv_file", "amv_file"):
        if config.get(key):
            candidates.append(_resolve_path(config[key]))
    paths = config.get("paths", {}) or {}
    for key in ("official_amv_file", "amv_file"):
        if paths.get(key):
            candidates.append(_resolve_path(paths[key]))
    for pattern in (config.get("official_amv_glob"), paths.get("official_amv_glob"), paths.get("amv_glob")):
        if pattern:
            candidates.extend(path for path in REPO_ROOT.glob(str(pattern)) if path.is_file())

    amv_root = paths.get("amv")
    if amv_root is not None:
        root = _resolve_path(amv_root)
        candidates.extend(path for path in root.rglob("*.nc") if path.is_file())

    output = []
    seen = set()
    for path in sorted(candidates):
        key = str(path)
        if key not in seen and path.exists():
            seen.add(key)
            output.append(path)
    return output


def _select_file(config: dict) -> Path:
    matches = _candidate_files(config)
    if not matches:
        raise FileNotFoundError("No official AMV NetCDF found. Check official_amv_file, paths['amv'], or glob settings.")
    if len(matches) == 1:
        return matches[0]

    target_time = _case_target_time(config)
    if target_time is None:
        return sorted(matches)[0]

    intervals = [(path, _parse_interval_from_name(path)) for path in matches]
    containing = [path for path, interval in intervals if interval is not None and interval[0] <= target_time < interval[1]]
    if containing:
        return sorted(containing)[0]

    def distance(path_interval):
        path, interval = path_interval
        if interval is None:
            return float("inf")
        start, end = interval
        if target_time < start:
            return abs((start - target_time).total_seconds())
        if target_time >= end:
            return abs((target_time - end).total_seconds())
        return 0.0

    best_path, _ = min(intervals, key=distance)
    return best_path


def load_official_amv(case_config, aoi=None) -> OfficialAmv:
    """Config-driven wrapper. New notebooks can use load_official_amv_file."""

    config = _as_config(case_config)
    return load_official_amv_file(_select_file(config), aoi=aoi if aoi is not None else config.get("aoi"))
