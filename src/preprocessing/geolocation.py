"""Geolocation sampling and local pixel-basis calculations."""

from __future__ import annotations

import numpy as np
from pyproj import Geod
from scipy import ndimage


GEOD = Geod(ellps="WGS84")


def build_coarse_grid_centers(
    original_shape: tuple[int, int],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map reduced-grid cell centers to original-image pixel coordinates."""

    if not 0 < scale <= 1:
        raise ValueError("scale must satisfy 0 < scale <= 1")
    low_shape = tuple(max(1, int(np.floor(value * scale))) for value in original_shape)
    rows = (np.arange(low_shape[0], dtype=np.float32) + 0.5) / scale - 0.5
    cols = (np.arange(low_shape[1], dtype=np.float32) + 0.5) / scale - 0.5
    return np.meshgrid(rows, cols, indexing="ij")


def sample_at_centers(
    field: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """Sample a two-dimensional field while preserving missing values."""

    values = np.asarray(field, dtype=np.float64)
    finite = np.isfinite(values)
    numerator = ndimage.map_coordinates(
        np.where(finite, values, 0.0),
        [rows, cols],
        order=order,
        mode="nearest",
        prefilter=False,
    )
    denominator = ndimage.map_coordinates(
        finite.astype(float),
        [rows, cols],
        order=order,
        mode="nearest",
        prefilter=False,
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(rows.shape, np.nan),
        where=denominator > 1e-6,
    ).astype(np.float32)


def _step_vectors(
    longitude: np.ndarray, latitude: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    if axis == 1:
        lon0, lon1 = longitude[:, :-1], longitude[:, 1:]
        lat0, lat1 = latitude[:, :-1], latitude[:, 1:]
    else:
        lon0, lon1 = longitude[:-1, :], longitude[1:, :]
        lat0, lat1 = latitude[:-1, :], latitude[1:, :]
    azimuth, _, distance = GEOD.inv(lon0, lat0, lon1, lat1)
    east = distance * np.sin(np.deg2rad(azimuth))
    north = distance * np.cos(np.deg2rad(azimuth))
    out_e = np.full(longitude.shape, np.nan)
    out_n = np.full(longitude.shape, np.nan)
    if axis == 1:
        out_e[:, :-1], out_n[:, :-1] = east, north
        out_e[:, -1], out_n[:, -1] = east[:, -1], north[:, -1]
    else:
        out_e[:-1, :], out_n[:-1, :] = east, north
        out_e[-1, :], out_n[-1, :] = east[-1, :], north[-1, :]
    return out_e.astype(np.float32), out_n.astype(np.float32)


def compute_local_geolocation_basis(
    longitude: np.ndarray,
    latitude: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute east/north metres for one positive row and column step."""

    lon = np.asarray(longitude, dtype=float)
    lat = np.asarray(latitude, dtype=float)
    if lon.shape != lat.shape or lon.ndim != 2:
        raise ValueError("longitude and latitude must be equally shaped 2-D arrays")
    col_east, col_north = _step_vectors(lon, lat, axis=1)
    row_east, row_north = _step_vectors(lon, lat, axis=0)
    return {
        "col_east_m": col_east,
        "col_north_m": col_north,
        "row_east_m": row_east,
        "row_north_m": row_north,
    }
