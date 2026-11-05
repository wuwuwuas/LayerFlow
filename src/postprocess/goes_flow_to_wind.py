"""Convert GOES fixed-grid dense optical flow to east/north wind.

Two geometries are supported without rerunning PCF:

* ``surface``: intersect both source and displaced target rays with the nominal
  reference ellipsoid (SEVIRI-equivalent geolocation baseline).
* ``cloud_height``: intersect both rays with the source/centre cloud-top height
  surface, providing a centre-CTH parallax correction.

No previous/next cloud, pressure, height, or destination-validity mask is used
as a post-hoc filter.  Target-frame masks have already constrained the PCF data
term.  The only target-side checks here are mathematical: the displaced pixel
must remain on the fixed grid and the requested ray intersection must exist.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from models import FlowResult, WindResult
from preprocessing import sample_at_centers
from preprocessing.goes_geolocation import (
    GoesProjection,
    ecef_to_geodetic_lonlat_on_height_ellipsoid,
    fixed_grid_to_ecef,
    local_east_north_up_basis,
    local_zenith_angle_deg,
    projection_from_attrs,
    scan_angles_at_pixels,
)


def _speed_direction(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.hypot(u, v)
    direction_from = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return speed.astype(np.float32), direction_from.astype(np.float32)


def _sample_field(
    field: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    order: int = 1,
) -> np.ndarray:
    return sample_at_centers(field, rows, cols, order=order)


def goes_flow_to_wind(
    flow_result: FlowResult,
    *,
    x_scan_angle: np.ndarray,
    y_scan_angle: np.ndarray,
    projection_attrs: Mapping[str, Any] | GoesProjection,
    dt_seconds: float,
    geometry_mode: str = "surface",
    center_height_m: np.ndarray | None = None,
    output_valid_mask: np.ndarray | None = None,
    chunk_rows: int = 256,
) -> WindResult:
    """Convert one centre-anchored GOES optical-flow component to wind.

    ``dt_seconds`` is signed ``target_time - centre_time``.  Therefore the
    centre-to-previous component uses a negative interval and the
    centre-to-next component a positive interval.
    """

    if not np.isfinite(dt_seconds) or float(dt_seconds) == 0.0:
        raise ValueError("dt_seconds must be finite and non-zero")
    mode = str(geometry_mode).strip().lower()
    if mode not in {"surface", "cloud_height"}:
        raise ValueError("geometry_mode must be 'surface' or 'cloud_height'")

    projection = (
        projection_attrs
        if isinstance(projection_attrs, GoesProjection)
        else projection_from_attrs(projection_attrs)
    )
    x_axis = np.asarray(x_scan_angle, dtype=np.float64)
    y_axis = np.asarray(y_scan_angle, dtype=np.float64)
    shape = np.asarray(flow_result.dx_pixels).shape
    if len(y_axis) != shape[0] or len(x_axis) != shape[1]:
        raise ValueError(
            "GOES x/y scan-angle lengths must match the optical-flow HxW shape"
        )

    if mode == "surface":
        geometry_height = np.zeros(shape, dtype=np.float32)
    else:
        if center_height_m is None:
            raise ValueError("center_height_m is required for cloud_height mode")
        geometry_height = np.asarray(center_height_m, dtype=np.float32)
        if geometry_height.shape != shape:
            raise ValueError(
                f"center_height_m shape {geometry_height.shape} does not match {shape}"
            )

    rows = np.asarray(flow_result.row_center_original, dtype=np.float32)
    cols = np.asarray(flow_result.col_center_original, dtype=np.float32)
    dx_pixels = np.asarray(flow_result.dx_pixels, dtype=np.float32)
    dy_pixels = np.asarray(flow_result.dy_pixels, dtype=np.float32)
    base_valid = np.asarray(flow_result.valid_mask, dtype=bool)
    if output_valid_mask is not None:
        requested_output = np.asarray(output_valid_mask, dtype=bool)
        if requested_output.shape != shape:
            raise ValueError("output_valid_mask must match the flow HxW shape")
        base_valid &= requested_output
    if not (
        rows.shape
        == cols.shape
        == dx_pixels.shape
        == dy_pixels.shape
        == base_valid.shape
        == shape
    ):
        raise ValueError("All FlowResult fields must share one shape")

    u = np.full(shape, np.nan, dtype=np.float32)
    v = np.full(shape, np.nan, dtype=np.float32)
    lon = np.full(shape, np.nan, dtype=np.float32)
    lat = np.full(shape, np.nan, dtype=np.float32)
    zenith = np.full(shape, np.nan, dtype=np.float32)
    geometry_valid = np.zeros(shape, dtype=bool)

    chunk = max(1, int(chunk_rows))
    for row_start in range(0, shape[0], chunk):
        row_stop = min(row_start + chunk, shape[0])
        sl = slice(row_start, row_stop)
        source_rows = rows[sl]
        source_cols = cols[sl]
        destination_rows = source_rows + dy_pixels[sl]
        destination_cols = source_cols + dx_pixels[sl]

        destination_in_grid = (
            np.isfinite(destination_rows)
            & np.isfinite(destination_cols)
            & (destination_rows >= 0.0)
            & (destination_rows <= shape[0] - 1)
            & (destination_cols >= 0.0)
            & (destination_cols <= shape[1] - 1)
        )

        source_x_scan, source_y_scan = scan_angles_at_pixels(
            x_axis,
            y_axis,
            source_rows,
            source_cols,
        )
        target_x_scan, target_y_scan = scan_angles_at_pixels(
            x_axis,
            y_axis,
            destination_rows,
            destination_cols,
        )
        source_h = _sample_field(
            geometry_height,
            source_rows,
            source_cols,
            order=1,
        ).astype(np.float64)
        # One centre-assigned height is used for both endpoints.  This prevents
        # temporal CTH retrieval noise from appearing as horizontal velocity.
        target_h = source_h

        sx, sy, sz, source_ray_valid = fixed_grid_to_ecef(
            source_x_scan,
            source_y_scan,
            source_h,
            projection,
        )
        tx, ty, tz, target_ray_valid = fixed_grid_to_ecef(
            target_x_scan,
            target_y_scan,
            target_h,
            projection,
        )
        source_lon, source_lat = ecef_to_geodetic_lonlat_on_height_ellipsoid(
            sx,
            sy,
            sz,
            source_h,
            projection,
        )
        basis = local_east_north_up_basis(source_lon, source_lat)
        east_x, east_y, east_z = basis["east"]
        north_x, north_y, north_z = basis["north"]
        delta_x = tx - sx
        delta_y = ty - sy
        delta_z = tz - sz
        chunk_u = (
            delta_x * east_x + delta_y * east_y + delta_z * east_z
        ) / float(dt_seconds)
        chunk_v = (
            delta_x * north_x + delta_y * north_y + delta_z * north_z
        ) / float(dt_seconds)
        chunk_zenith = local_zenith_angle_deg(
            sx,
            sy,
            sz,
            source_lon,
            source_lat,
            projection,
        )

        # Source coordinates describe the common centre-frame vector grid.
        # They must not depend on whether this particular temporal component
        # has a valid displaced endpoint.  Otherwise C1 and C2 obtain different
        # NaN patterns in lon/lat and cannot be combined even though both are
        # centre-anchored on the same source pixels.
        coordinate_valid = (
            source_ray_valid
            & np.isfinite(source_h)
            & np.isfinite(source_lon)
            & np.isfinite(source_lat)
        )
        valid = (
            base_valid[sl]
            & coordinate_valid
            & destination_in_grid
            & target_ray_valid
            & np.isfinite(chunk_u)
            & np.isfinite(chunk_v)
        )
        u[sl] = np.where(valid, chunk_u, np.nan).astype(np.float32)
        v[sl] = np.where(valid, chunk_v, np.nan).astype(np.float32)
        lon[sl] = np.where(coordinate_valid, source_lon, np.nan).astype(np.float32)
        lat[sl] = np.where(coordinate_valid, source_lat, np.nan).astype(np.float32)
        zenith[sl] = np.where(
            coordinate_valid, chunk_zenith, np.nan
        ).astype(np.float32)
        geometry_valid[sl] = valid

    speed, direction = _speed_direction(u, v)
    geometry_description = (
        "goes_fixed_grid_nominal_surface"
        if mode == "surface"
        else "goes_fixed_grid_center_cloud_height_parallax_corrected"
    )
    return WindResult(
        u_ms=u,
        v_ms=v,
        speed_ms=speed,
        direction_deg=direction,
        row_center_original=rows,
        col_center_original=cols,
        longitude=lon,
        latitude=lat,
        valid_mask=geometry_valid,
        diagnostics={"local_zenith_angle_deg": zenith},
        metadata={
            **dict(getattr(flow_result, "metadata", {}) or {}),
            "geometry": geometry_description,
            "geometry_mode": mode,
            "height_endpoint_rule": "same_center_height_for_source_and_target",
            "destination_product_validity_filter": False,
            "dt_seconds_signed": float(dt_seconds),
            "longitude_of_projection_origin_deg": (
                projection.longitude_of_projection_origin_deg
            ),
        },
    )
