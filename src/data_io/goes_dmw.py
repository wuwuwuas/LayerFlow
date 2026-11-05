"""Read and standardise GOES ABI Level-2 Derived Motion Winds.

The operational ABI DMW files are point products.  Band-14 files contain
cloud-tracked winds with wind speed and meteorological wind-from direction,
not Cartesian u/v components.  This module converts them to the repository's
standard eastward/northward convention and keeps the original quality fields
for later filtering and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


REQUIRED_VARIABLES = (
    "lat",
    "lon",
    "time",
    "wind_speed",
    "wind_direction",
    "pressure",
    "temperature",
    "local_zenith_angle",
    "solar_zenith_angle",
    "DQF",
)


@dataclass(frozen=True)
class GoesDmwFile:
    satellite: str
    path: Path


def _require_xarray() -> Any:
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("GOES DMW reading requires xarray") from exc
    return xr


def _satellite_from_path(path: Path) -> str:
    name = path.name.upper()
    for satellite in ("G16", "G17", "G18", "G19"):
        if f"_{satellite}_" in name:
            return satellite
    for part in path.parts:
        lower = part.lower()
        if lower.startswith("goes") and lower[4:].isdigit():
            return f"G{int(lower[4:]):02d}"
    raise ValueError(f"Could not infer GOES satellite from path: {path}")


def discover_goes_dmw_files(
    case_dir: str | Path,
    satellites: Sequence[str] | None = None,
) -> tuple[GoesDmwFile, ...]:
    """Find one downloaded Band-14 Full Disk DMW file per satellite."""

    root = Path(case_dir).resolve()
    requested = None if satellites is None else {value.upper() for value in satellites}
    candidates = sorted(
        path
        for path in (root / "official_amv").glob("goes*/ABI-L2-DMWF-C14/*.nc")
        if path.is_file()
    )
    files: list[GoesDmwFile] = []
    for path in candidates:
        satellite = _satellite_from_path(path)
        if requested is not None and satellite not in requested:
            continue
        files.append(GoesDmwFile(satellite=satellite, path=path))

    grouped: dict[str, list[GoesDmwFile]] = {}
    for item in files:
        grouped.setdefault(item.satellite, []).append(item)
    duplicates = {key: values for key, values in grouped.items() if len(values) != 1}
    if duplicates:
        detail = ", ".join(f"{key}={len(value)}" for key, value in duplicates.items())
        raise RuntimeError(f"Expected exactly one DMW file per satellite; found {detail}")
    if requested is not None:
        missing = sorted(requested - set(grouped))
        if missing:
            raise FileNotFoundError(f"Missing official Band-14 DMW files for: {missing}")
    return tuple(sorted(files, key=lambda item: item.satellite))


def meteorological_speed_direction_to_uv(
    speed_ms: np.ndarray,
    direction_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert wind-from direction clockwise from north to eastward/northward u/v."""

    speed = np.asarray(speed_ms, dtype=np.float64)
    direction = np.deg2rad(np.asarray(direction_deg, dtype=np.float64))
    u = -speed * np.sin(direction)
    v = -speed * np.cos(direction)
    return u.astype(np.float32), v.astype(np.float32)


def uv_to_meteorological_direction(
    u_ms: np.ndarray,
    v_ms: np.ndarray,
) -> np.ndarray:
    """Return meteorological wind-from direction in [0, 360) degrees."""

    u = np.asarray(u_ms, dtype=np.float64)
    v = np.asarray(v_ms, dtype=np.float64)
    return (np.degrees(np.arctan2(-u, -v)) % 360.0).astype(np.float32)


def _normalise_longitudes(values: np.ndarray) -> np.ndarray:
    lon = np.asarray(values, dtype=np.float64)
    normalised = ((lon + 180.0) % 360.0) - 180.0
    # Preserve positive 180 when it is explicitly present; this can be useful for
    # selecting the west-Pacific ERA5 segment instead of the -180 endpoint.
    normalised[np.isclose(lon, 180.0)] = 180.0
    return normalised.astype(np.float32)


def load_goes_dmw(path: str | Path, satellite: str | None = None) -> pd.DataFrame:
    """Load one ABI DMW file into a standard point table.

    No quality filter is applied here.  The returned table retains every decoded
    measure so that the evaluation script can report the exact DQF attrition.
    """

    xr = _require_xarray()
    source = Path(path).resolve()
    resolved_satellite = satellite.upper() if satellite else _satellite_from_path(source)

    with xr.open_dataset(source, decode_cf=True, mask_and_scale=True) as ds:
        missing = [name for name in REQUIRED_VARIABLES if name not in ds.variables]
        if missing:
            raise KeyError(
                f"{source.name} is missing required DMW variables {missing}; "
                f"available={list(ds.variables)}"
            )
        values = {name: np.asarray(ds[name].values) for name in REQUIRED_VARIABLES}
        global_attrs = dict(ds.attrs)
        dqf_attrs = dict(ds["DQF"].attrs)

    speed = values["wind_speed"].astype(np.float32)
    direction = values["wind_direction"].astype(np.float32)
    u, v = meteorological_speed_direction_to_uv(speed, direction)

    time_values = pd.to_datetime(values["time"].reshape(-1), errors="coerce")
    frame = pd.DataFrame(
        {
            "satellite": resolved_satellite,
            "longitude": _normalise_longitudes(values["lon"].reshape(-1)),
            "latitude": values["lat"].reshape(-1).astype(np.float32),
            "observation_time": time_values,
            "speed_ms": speed.reshape(-1),
            "direction_deg": direction.reshape(-1),
            "u_ms": u.reshape(-1),
            "v_ms": v.reshape(-1),
            "pressure_hpa": values["pressure"].reshape(-1).astype(np.float32),
            "temperature_k": values["temperature"].reshape(-1).astype(np.float32),
            "local_zenith_angle_deg": values["local_zenith_angle"].reshape(-1).astype(np.float32),
            "solar_zenith_angle_deg": values["solar_zenith_angle"].reshape(-1).astype(np.float32),
            "dqf": values["DQF"].reshape(-1).astype(np.int16),
        }
    )
    frame["source_file"] = source.name
    frame.attrs = {
        "source_path": str(source),
        "global_attrs": global_attrs,
        "dqf_attrs": dqf_attrs,
    }
    return frame


def dmw_basic_valid_mask(frame: pd.DataFrame) -> np.ndarray:
    """Return finite/range validity independent of DQF and ERA5 availability."""

    required = (
        "longitude",
        "latitude",
        "observation_time",
        "speed_ms",
        "direction_deg",
        "u_ms",
        "v_ms",
        "pressure_hpa",
        "local_zenith_angle_deg",
    )
    mask = np.ones(len(frame), dtype=bool)
    for name in required:
        if name == "observation_time":
            mask &= frame[name].notna().to_numpy()
        else:
            mask &= np.isfinite(frame[name].to_numpy(dtype=np.float64))
    mask &= frame["latitude"].between(-90.0, 90.0).to_numpy()
    mask &= frame["longitude"].between(-180.0, 180.0).to_numpy()
    mask &= frame["speed_ms"].between(0.0, 200.0).to_numpy()
    mask &= frame["direction_deg"].between(0.0, 360.0).to_numpy()
    return mask
