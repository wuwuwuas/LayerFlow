"""Convert pixel displacement to geolocated eastward/northward wind."""

from __future__ import annotations

import numpy as np

from models import FlowResult, WindResult
from preprocessing import compute_local_geolocation_basis, sample_at_centers


def _speed_direction(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.hypot(u, v)
    direction_from = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return speed.astype(np.float32), direction_from.astype(np.float32)


def flow_to_wind(
    flow_result: FlowResult,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    dt_seconds: float,
) -> WindResult:
    """Convert displacement in original pixels to wind in m/s."""

    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be positive")
    longitude = np.asarray(lon_grid, dtype=float)
    latitude = np.asarray(lat_grid, dtype=float)
    basis = compute_local_geolocation_basis(longitude, latitude)
    sampled = {
        name: sample_at_centers(
            field,
            flow_result.row_center_original,
            flow_result.col_center_original,
        )
        for name, field in basis.items()
    }
    u = (
        flow_result.dx_pixels * sampled["col_east_m"]
        + flow_result.dy_pixels * sampled["row_east_m"]
    ) / float(dt_seconds)
    v = (
        flow_result.dx_pixels * sampled["col_north_m"]
        + flow_result.dy_pixels * sampled["row_north_m"]
    ) / float(dt_seconds)
    lon = sample_at_centers(
        longitude, flow_result.row_center_original, flow_result.col_center_original
    )
    lat = sample_at_centers(
        latitude, flow_result.row_center_original, flow_result.col_center_original
    )
    valid = (
        flow_result.valid_mask
        & np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(lon)
        & np.isfinite(lat)
    )
    u = np.where(valid, u, np.nan).astype(np.float32)
    v = np.where(valid, v, np.nan).astype(np.float32)
    speed, direction = _speed_direction(u, v)
    return WindResult(
        u,
        v,
        speed,
        direction,
        flow_result.row_center_original,
        flow_result.col_center_original,
        lon,
        lat,
        valid,
        metadata={**flow_result.metadata, "dt_seconds": float(dt_seconds)},
    )
