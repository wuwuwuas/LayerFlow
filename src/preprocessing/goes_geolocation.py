"""GOES-R ABI fixed-grid ray geometry at an assigned cloud-top height.

The ABI fixed grid stores one-dimensional x/y scan angles.  A pixel therefore
represents a satellite line of sight rather than a fixed horizontal distance.
This module converts scan angles to Earth-centred Earth-fixed (ECEF) positions
by intersecting the line of sight with an ellipsoid expanded by the requested
cloud-top height.

The implementation follows the GOES-R geostationary projection convention with
``sweep_angle_axis='x'``.  Height zero reproduces the standard geostationary
projection inverse; non-zero height supplies the parallax-corrected cloud
location needed to convert optical-flow displacement to horizontal wind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class GoesProjection:
    perspective_point_height_m: float
    semi_major_axis_m: float
    semi_minor_axis_m: float
    longitude_of_projection_origin_deg: float
    latitude_of_projection_origin_deg: float = 0.0
    sweep_angle_axis: str = "x"

    @property
    def satellite_distance_from_earth_center_m(self) -> float:
        return self.perspective_point_height_m + self.semi_major_axis_m

    @property
    def origin_longitude_rad(self) -> float:
        return float(np.deg2rad(self.longitude_of_projection_origin_deg))


def projection_from_attrs(attrs: Mapping[str, Any]) -> GoesProjection:
    """Build a validated projection object from ``goes_imager_projection`` attrs."""

    required = (
        "perspective_point_height",
        "semi_major_axis",
        "semi_minor_axis",
        "longitude_of_projection_origin",
    )
    missing = [name for name in required if name not in attrs]
    if missing:
        raise KeyError(f"GOES projection attributes are missing: {missing}")
    sweep = str(attrs.get("sweep_angle_axis", "x")).lower()
    if sweep != "x":
        raise NotImplementedError(
            f"Only GOES sweep_angle_axis='x' is supported; received {sweep!r}"
        )
    latitude_origin = float(attrs.get("latitude_of_projection_origin", 0.0))
    if abs(latitude_origin) > 1.0e-8:
        raise NotImplementedError(
            "Only an equatorial geostationary projection origin is supported"
        )
    return GoesProjection(
        perspective_point_height_m=float(attrs["perspective_point_height"]),
        semi_major_axis_m=float(attrs["semi_major_axis"]),
        semi_minor_axis_m=float(attrs["semi_minor_axis"]),
        longitude_of_projection_origin_deg=float(
            attrs["longitude_of_projection_origin"]
        ),
        latitude_of_projection_origin_deg=latitude_origin,
        sweep_angle_axis=sweep,
    )


def interpolate_axis_at_indices(axis: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Linearly sample one fixed-grid coordinate axis at fractional pixel indices."""

    values = np.asarray(axis, dtype=np.float64)
    query = np.asarray(indices, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("axis must be a one-dimensional array with at least 2 values")
    output = np.full(query.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(query) & (query >= 0.0) & (query <= values.size - 1)
    if not np.any(valid):
        return output
    q = query[valid]
    lower = np.floor(q).astype(np.int64)
    upper = np.minimum(lower + 1, values.size - 1)
    fraction = q - lower
    output[valid] = (
        values[lower] * (1.0 - fraction) + values[upper] * fraction
    )
    return output


def scan_angles_at_pixels(
    x_scan_axis: np.ndarray,
    y_scan_axis: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y scan angles for fractional row/column coordinates."""

    x = interpolate_axis_at_indices(x_scan_axis, cols)
    y = interpolate_axis_at_indices(y_scan_axis, rows)
    return x, y


def satellite_position_ecef(projection: GoesProjection) -> np.ndarray:
    """Return the geostationary satellite ECEF position in metres."""

    lon0 = projection.origin_longitude_rad
    radius = projection.satellite_distance_from_earth_center_m
    return np.asarray(
        [radius * np.cos(lon0), radius * np.sin(lon0), 0.0],
        dtype=np.float64,
    )


def _ray_unit_vectors_ecef(
    x_scan_rad: np.ndarray,
    y_scan_rad: np.ndarray,
    projection: GoesProjection,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ECEF unit vectors from the satellite through fixed-grid pixels."""

    x = np.asarray(x_scan_rad, dtype=np.float64)
    y = np.asarray(y_scan_rad, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x_scan_rad and y_scan_rad must have the same shape")

    lon0 = projection.origin_longitude_rad
    radial = np.asarray([np.cos(lon0), np.sin(lon0), 0.0])
    east = np.asarray([-np.sin(lon0), np.cos(lon0), 0.0])
    north = np.asarray([0.0, 0.0, 1.0])

    radial_weight = -np.cos(x) * np.cos(y)
    east_weight = np.sin(x)
    north_weight = np.cos(x) * np.sin(y)

    dx = (
        radial_weight * radial[0]
        + east_weight * east[0]
        + north_weight * north[0]
    )
    dy = (
        radial_weight * radial[1]
        + east_weight * east[1]
        + north_weight * north[1]
    )
    dz = (
        radial_weight * radial[2]
        + east_weight * east[2]
        + north_weight * north[2]
    )
    return dx, dy, dz


def fixed_grid_to_ecef(
    x_scan_rad: np.ndarray,
    y_scan_rad: np.ndarray,
    height_m: np.ndarray | float,
    projection: GoesProjection | Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Intersect GOES lines of sight with a cloud-height ellipsoid.

    Returns ``x_ecef, y_ecef, z_ecef, valid``.  The height surface is represented
    by an ellipsoid with semi-axes ``a + h`` and ``b + h``.  This is the standard
    practical approximation for parallax correction at a spatially varying
    geodetic height.
    """

    proj = (
        projection
        if isinstance(projection, GoesProjection)
        else projection_from_attrs(projection)
    )
    x = np.asarray(x_scan_rad, dtype=np.float64)
    y = np.asarray(y_scan_rad, dtype=np.float64)
    h = np.broadcast_to(np.asarray(height_m, dtype=np.float64), x.shape)
    if x.shape != y.shape:
        raise ValueError("x_scan_rad and y_scan_rad must have the same shape")

    dx, dy, dz = _ray_unit_vectors_ecef(x, y, proj)
    satellite = satellite_position_ecef(proj)
    a = proj.semi_major_axis_m + h
    b = proj.semi_minor_axis_m + h

    finite = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(h)
        & (a > 0.0)
        & (b > 0.0)
    )
    coefficient_a = (dx * dx + dy * dy) / (a * a) + (dz * dz) / (b * b)
    coefficient_b = 2.0 * (
        (satellite[0] * dx + satellite[1] * dy) / (a * a)
        + satellite[2] * dz / (b * b)
    )
    coefficient_c = (
        (satellite[0] ** 2 + satellite[1] ** 2) / (a * a)
        + satellite[2] ** 2 / (b * b)
        - 1.0
    )
    discriminant = coefficient_b * coefficient_b - 4.0 * coefficient_a * coefficient_c
    valid = finite & np.isfinite(discriminant) & (discriminant >= 0.0)

    distance = np.full(x.shape, np.nan, dtype=np.float64)
    if np.any(valid):
        sqrt_discriminant = np.sqrt(discriminant[valid])
        distance[valid] = (
            -coefficient_b[valid] - sqrt_discriminant
        ) / (2.0 * coefficient_a[valid])
    valid &= np.isfinite(distance) & (distance > 0.0)

    x_ecef = np.full(x.shape, np.nan, dtype=np.float64)
    y_ecef = np.full(x.shape, np.nan, dtype=np.float64)
    z_ecef = np.full(x.shape, np.nan, dtype=np.float64)
    x_ecef[valid] = satellite[0] + distance[valid] * dx[valid]
    y_ecef[valid] = satellite[1] + distance[valid] * dy[valid]
    z_ecef[valid] = satellite[2] + distance[valid] * dz[valid]
    return x_ecef, y_ecef, z_ecef, valid


def ecef_to_geodetic_lonlat_on_height_ellipsoid(
    x_ecef_m: np.ndarray,
    y_ecef_m: np.ndarray,
    z_ecef_m: np.ndarray,
    height_m: np.ndarray | float,
    projection: GoesProjection | Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert points on the expanded height ellipsoid to lon/lat degrees."""

    proj = (
        projection
        if isinstance(projection, GoesProjection)
        else projection_from_attrs(projection)
    )
    x = np.asarray(x_ecef_m, dtype=np.float64)
    y = np.asarray(y_ecef_m, dtype=np.float64)
    z = np.asarray(z_ecef_m, dtype=np.float64)
    h = np.broadcast_to(np.asarray(height_m, dtype=np.float64), x.shape)
    a = proj.semi_major_axis_m + h
    b = proj.semi_minor_axis_m + h
    rho = np.hypot(x, y)
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arctan2(z * a * a, rho * b * b))
    invalid = ~(
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & np.isfinite(h)
    )
    lon[invalid] = np.nan
    lat[invalid] = np.nan
    return lon.astype(np.float32), lat.astype(np.float32)


def local_east_north_up_basis(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return local ECEF east/north/up unit-vector components."""

    lon = np.deg2rad(np.asarray(longitude_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    return {
        "east": (-sin_lon, cos_lon, np.zeros(lon.shape, dtype=np.float64)),
        "north": (
            -sin_lat * cos_lon,
            -sin_lat * sin_lon,
            cos_lat,
        ),
        "up": (
            cos_lat * cos_lon,
            cos_lat * sin_lon,
            sin_lat,
        ),
    }


def local_zenith_angle_deg(
    x_ecef_m: np.ndarray,
    y_ecef_m: np.ndarray,
    z_ecef_m: np.ndarray,
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    projection: GoesProjection | Mapping[str, Any],
) -> np.ndarray:
    """Return satellite local zenith angle at the cloud point."""

    proj = (
        projection
        if isinstance(projection, GoesProjection)
        else projection_from_attrs(projection)
    )
    satellite = satellite_position_ecef(proj)
    x = np.asarray(x_ecef_m, dtype=np.float64)
    y = np.asarray(y_ecef_m, dtype=np.float64)
    z = np.asarray(z_ecef_m, dtype=np.float64)
    to_sat_x = satellite[0] - x
    to_sat_y = satellite[1] - y
    to_sat_z = satellite[2] - z
    norm = np.sqrt(to_sat_x**2 + to_sat_y**2 + to_sat_z**2)
    basis = local_east_north_up_basis(longitude_deg, latitude_deg)
    up_x, up_y, up_z = basis["up"]
    cosine = np.divide(
        to_sat_x * up_x + to_sat_y * up_y + to_sat_z * up_z,
        norm,
        out=np.full(norm.shape, np.nan, dtype=np.float64),
        where=norm > 0.0,
    )
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle.astype(np.float32)
