"""Shared geographic crop helpers."""

from __future__ import annotations

import numpy as np


def normalize_aoi(aoi) -> tuple[float, float, float, float] | None:
    """Normalize AOI to ``(west, east, south, north)``."""

    if aoi is None:
        return None
    if isinstance(aoi, dict):
        return (
            float(aoi["lon_min"]),
            float(aoi["lon_max"]),
            float(aoi["lat_min"]),
            float(aoi["lat_max"]),
        )
    if len(aoi) != 4:
        raise ValueError("AOI must contain west, east, south, north")
    return tuple(float(value) for value in aoi)


def geographic_crop(
    longitude: np.ndarray,
    latitude: np.ndarray,
    aoi,
) -> tuple[slice, slice] | None:
    """Return the smallest rectangular slice covering finite AOI pixels."""

    bounds = normalize_aoi(aoi)
    if bounds is None:
        return None
    west, east, south, north = bounds
    valid = (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & (longitude >= west)
        & (longitude <= east)
        & (latitude >= south)
        & (latitude <= north)
    )
    rows, cols = np.where(valid)
    if not rows.size:
        raise ValueError("AOI contains no geolocated pixels")
    return (
        slice(int(rows.min()), int(rows.max()) + 1),
        slice(int(cols.min()), int(cols.max()) + 1),
    )
