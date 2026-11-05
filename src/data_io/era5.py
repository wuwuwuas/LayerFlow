"""Local ERA5 reference-wind loading."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
import re

import xarray as xr

from config import REPO_ROOT, repo_path, resolve_case_config
from .common import normalize_aoi


@dataclass(frozen=True)
class Era5Data:
    """A local ERA5 file and optional geographic subset."""

    path: Path
    dataset: xr.Dataset


def _as_config(case_config) -> dict[str, Any]:
    if isinstance(case_config, Mapping):
        return dict(case_config)
    return resolve_case_config(case_config)


def _resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return repo_path(value)


def _parse_time(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _case_time_bounds(config: Mapping[str, Any]) -> tuple[datetime | None, datetime | None]:
    return _parse_time(config.get("start_time")), _parse_time(config.get("end_time"))


def _candidate_files(config: dict) -> list[Path]:
    candidates: list[Path] = []
    paths = config.get("paths", {}) or {}

    for key in ("era5_file",):
        if config.get(key):
            candidates.append(_resolve_path(config[key]))
        if paths.get(key):
            candidates.append(_resolve_path(paths[key]))

    for pattern in (config.get("era5_glob"), paths.get("era5_glob")):
        if pattern:
            candidates.extend(path for path in REPO_ROOT.glob(str(pattern)) if path.is_file())

    era5_root = paths.get("era5")
    if era5_root is not None:
        root = _resolve_path(era5_root)
        candidates.extend(path for path in root.rglob("*.nc") if path.is_file())

    candidates.extend(path for path in REPO_ROOT.glob("data/**/*era5*.nc") if path.is_file())

    output = []
    seen = set()
    for path in sorted(candidates):
        key = str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        if any(part in {"tmp", "requests", "background_jobs"} for part in path.parts):
            continue
        output.append(path)
    return output


def _interval_from_name(path: Path) -> tuple[datetime, datetime] | None:
    match = re.search(r"era5_(\d{12})_(\d{12})", path.name)
    if not match:
        return None
    try:
        return (
            datetime.strptime(match.group(1), "%Y%m%d%H%M"),
            datetime.strptime(match.group(2), "%Y%m%d%H%M"),
        )
    except ValueError:
        return None


def _select_file(config: dict) -> Path | None:
    candidates = _candidate_files(config)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    start, end = _case_time_bounds(config)
    if start is None or end is None:
        return candidates[0]

    intervals = [(path, _interval_from_name(path)) for path in candidates]
    containing = [path for path, interval in intervals if interval is not None and interval[0] <= start and interval[1] >= end]
    if containing:
        return sorted(containing)[0]
    return candidates[0]


def _subset_aoi(dataset: xr.Dataset, aoi) -> xr.Dataset:
    bounds = normalize_aoi(aoi)
    if bounds is None:
        return dataset
    west, east, south, north = bounds
    lat_name = "latitude" if "latitude" in dataset.coords else "lat" if "lat" in dataset.coords else None
    lon_name = "longitude" if "longitude" in dataset.coords else "lon" if "lon" in dataset.coords else None
    if lat_name is None or lon_name is None:
        return dataset

    latitude = dataset[lat_name]
    lat_slice = slice(north, south) if float(latitude[0]) > float(latitude[-1]) else slice(south, north)
    if west <= east:
        return dataset.sel({lat_name: lat_slice, lon_name: slice(west, east)})
    # Dateline-crossing AOI; rare for MSG, but keep safe.
    left = dataset.sel({lat_name: lat_slice, lon_name: slice(west, 180)})
    right = dataset.sel({lat_name: lat_slice, lon_name: slice(-180, east)})
    return xr.concat([left, right], dim=lon_name)


def load_era5(case_config, aoi=None) -> Era5Data | None:
    """Open existing local ERA5 data without submitting a CDS request."""

    config = _as_config(case_config)
    path = _select_file(config)
    if path is None:
        return None
    dataset = xr.open_dataset(path)
    requested_aoi = aoi if aoi is not None else config.get("aoi")
    dataset = _subset_aoi(dataset, requested_aoi)
    return Era5Data(path, dataset)
