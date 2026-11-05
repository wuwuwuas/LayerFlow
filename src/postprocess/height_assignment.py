"""Pressure assignment for PIV-AMV products.

Pressure assignment is treated as a post-processing experiment:
- OCA pressure: dense component-start pressure grids assigned to C1/C2/C3.
  Dense pixel-level PIV uses same-pixel indexing; sparse/coarse PIV uses a
  small local window statistic, median by default.  Component pressures are
  then averaged where at least two components are valid.
- Official AMV pressure: sparse official pressure points anchor the output.
  Around each official point, nearby valid PIV winds are aggregated in u/v
  space, mean by default.  The output keeps official lon/lat and official
  pressure, but replaces official wind with the local PIV wind aggregate.

Outputs are compact 1-D products: points without assigned pressure are not
written.  No QC score, ERA5 comparison, or downstream validation is created
here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from data_io.piv_amv import (
    base_valid_mask,
    compact_piv_arrays,
    load_piv_final_native_npz,
    save_pressure_assigned_piv_amv_npz,
)


EARTH_RADIUS_KM = 6371.0088


def clean_pressure_hpa(pressure: Any) -> np.ndarray:
    """Return pressure in hPa with nonphysical values set to NaN."""

    out = np.asarray(pressure, dtype=np.float32).copy()
    finite = np.isfinite(out)
    if finite.any() and np.nanmedian(out[finite]) > 2000.0:
        out = out / 100.0  # likely Pa -> hPa
    out[(out < 50.0) | (out > 1100.0)] = np.nan
    return out.astype(np.float32)


def pressure_to_standard_height_m(pressure_hpa: Any) -> np.ndarray:
    """Approximate standard-atmosphere height from pressure.

    This is not a geometric cloud height. Use only when a pressure-derived
    height is explicitly needed.
    """

    pressure = clean_pressure_hpa(pressure_hpa).astype(np.float32)
    height = 44330.0 * (1.0 - np.power(pressure / 1013.25, 0.1903))
    return np.where(np.isfinite(pressure), height, np.nan).astype(np.float32)


def _as_piv_dict(piv: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(piv, (str, Path)):
        return load_piv_final_native_npz(piv)
    return dict(piv)


def _as_2d_grid(field: Any, *, name: str = "grid") -> np.ndarray:
    values = np.squeeze(np.asarray(field, dtype=np.float32))
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2-D after squeeze, got shape {values.shape}")
    return values.astype(np.float32)


def _direct_sample_grid(field: Any, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Same/nearest-pixel sampling without interpolation."""

    values = _as_2d_grid(field)
    r = np.rint(np.asarray(rows)).astype(np.int64)
    c = np.rint(np.asarray(cols)).astype(np.int64)
    inside = (r >= 0) & (r < values.shape[0]) & (c >= 0) & (c < values.shape[1])
    sampled = np.full(np.shape(rows), np.nan, dtype=np.float32)
    sampled[inside] = values[r[inside], c[inside]]
    return sampled


def _aggregate(values: np.ndarray, statistic: str) -> np.float32:
    statistic = str(statistic).lower()
    values = np.asarray(values, dtype=np.float32)
    if statistic == "mean":
        return np.float32(np.nanmean(values))
    if statistic == "median":
        return np.float32(np.nanmedian(values))
    raise ValueError("statistic must be 'mean' or 'median'")


def _window_sample_grid(
    field: Any,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    window_size: int = 3,
    statistic: str = "median",
) -> np.ndarray:
    """Sample a grid using a local window statistic around each row/col.

    This is intended for sparse/coarse PIV coordinates. Dense pixel-level PIV
    should use direct indexing instead.
    """

    values = _as_2d_grid(field)
    if int(window_size) < 1 or int(window_size) % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    statistic = str(statistic).lower()
    if statistic not in {"median", "mean"}:
        raise ValueError("window_statistic must be 'median' or 'mean'")

    rows_arr = np.asarray(rows, dtype=np.float32)
    cols_arr = np.asarray(cols, dtype=np.float32)
    sampled = np.full(rows_arr.shape, np.nan, dtype=np.float32)
    half = int(window_size) // 2

    flat_rows = rows_arr.ravel()
    flat_cols = cols_arr.ravel()
    flat_out = sampled.ravel()

    for k, (row, col) in enumerate(zip(flat_rows, flat_cols)):
        if not (np.isfinite(row) and np.isfinite(col)):
            continue
        r = int(np.rint(row))
        c = int(np.rint(col))
        r0 = max(0, r - half)
        r1 = min(values.shape[0], r + half + 1)
        c0 = max(0, c - half)
        c1 = min(values.shape[1], c + half + 1)
        block = values[r0:r1, c0:c1]
        selected = block[np.isfinite(block)]
        if selected.size:
            flat_out[k] = _aggregate(selected, statistic)

    return sampled


def _is_pixel_dense_coordinate_grid(rows: np.ndarray, cols: np.ndarray, *, atol: float = 1e-3) -> bool:
    """Return True if coordinates look like one PIV vector per image pixel.

    This also works for cropped dense domains because it checks row/col step
    size, not the absolute starting row/col.
    """

    rows = np.asarray(rows, dtype=np.float32)
    cols = np.asarray(cols, dtype=np.float32)
    if rows.ndim != 2 or cols.ndim != 2 or rows.shape != cols.shape:
        return False
    if rows.shape[0] < 2 or rows.shape[1] < 2:
        return False

    # Dense pixel grids have constant row values along columns and constant
    # column values along rows.
    if not np.allclose(rows, rows[:, :1], equal_nan=True, atol=atol):
        return False
    if not np.allclose(cols, cols[:1, :], equal_nan=True, atol=atol):
        return False

    row_steps = np.diff(rows[:, 0])
    col_steps = np.diff(cols[0, :])
    if not (np.allclose(row_steps, 1.0, equal_nan=True, atol=atol) and np.allclose(col_steps, 1.0, equal_nan=True, atol=atol)):
        return False

    # Pixel-direct products should be located at integer-ish pixel centers.
    return bool(
        np.nanmax(np.abs(rows - np.rint(rows))) <= atol
        and np.nanmax(np.abs(cols - np.rint(cols))) <= atol
    )


def _resolve_oca_assignment_mode(rows: np.ndarray, cols: np.ndarray, mode: str) -> str:
    mode = str(mode).lower()
    aliases = {
        "direct": "pixel_direct",
        "point": "pixel_direct",
        "nearest": "pixel_direct",
        "pixel": "pixel_direct",
    }
    mode = aliases.get(mode, mode)
    if mode == "auto":
        return "pixel_direct" if _is_pixel_dense_coordinate_grid(rows, cols) else "window"
    if mode not in {"pixel_direct", "window"}:
        raise ValueError("oca_assignment_mode must be 'auto', 'pixel_direct', or 'window'")
    return mode


def _sample_pressure_or_cloud_grid(
    field: Any,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    resolved_mode: str,
    window_size: int,
    window_statistic: str,
) -> np.ndarray:
    if resolved_mode == "pixel_direct":
        return _direct_sample_grid(field, rows, cols)
    return _window_sample_grid(field, rows, cols, window_size=window_size, statistic=window_statistic)


def _component_grid_triplet(source: Any, attr_name: str) -> tuple[Any, Any, Any]:
    values = getattr(source, attr_name)
    if len(values) < 3:
        raise ValueError(f"{attr_name} must contain at least three component fields")
    return values[0], values[1], values[2]


def _mean_valid_components(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(stack)
    count = valid.sum(axis=0).astype(np.int16)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(np.where(valid, stack, np.nan), axis=0).astype(np.float32)
    mean = np.where(count > 0, mean, np.nan).astype(np.float32)
    return mean, count


def assign_piv_pressure_from_component_grids(
    piv: Mapping[str, Any] | str | Path,
    pressure_components: tuple[Any, Any, Any] | list[Any],
    output_path: str | Path | None = None,
    *,
    cloud_components: tuple[Any | None, Any | None, Any | None] | list[Any | None] | None = None,
    pressure_source: str = "oca",
    min_valid_pressure_components: int = 2,
    oca_assignment_mode: str = "auto",
    oca_window_size: int = 3,
    oca_window_statistic: str = "median",
    include_height_m: bool = False,
    compress: bool = True,
    source_paths: list[str] | tuple[Any, ...] | None = None,
    component_source: str | None = None,
) -> dict[str, Any]:
    """Assign dense component pressure grids to a stacked PIV final product.

    This is the generic dense-grid route for OCA-like or future NN pressure
    products. The pressure components must be ordered as C1/C2/C3 and be on the
    same SEVIRI row/column grid as the PIV product.

    ``oca_assignment_mode='auto'`` uses same-pixel indexing for dense one-vector-
    per-pixel PIV products and a local window statistic for sparse/coarse PIV
    products.  Spatial window aggregation defaults to median, while the temporal
    C1/C2/C3 pressure combination is a mean of valid component pressures.
    """

    if len(pressure_components) < 3:
        raise ValueError("pressure_components must contain at least three grids: C1, C2, C3")
    if not 1 <= int(min_valid_pressure_components) <= 3:
        raise ValueError("min_valid_pressure_components must be 1, 2, or 3")

    piv_data = _as_piv_dict(piv)
    rows = np.asarray(piv_data["row_center_original"], dtype=np.float32)
    cols = np.asarray(piv_data["col_center_original"], dtype=np.float32)
    resolved_mode = _resolve_oca_assignment_mode(rows, cols, oca_assignment_mode)

    ctp1, ctp2, ctp3 = pressure_components[:3]
    pressure_c1 = clean_pressure_hpa(
        _sample_pressure_or_cloud_grid(ctp1, rows, cols, resolved_mode=resolved_mode, window_size=oca_window_size, window_statistic=oca_window_statistic)
    )
    pressure_c2 = clean_pressure_hpa(
        _sample_pressure_or_cloud_grid(ctp2, rows, cols, resolved_mode=resolved_mode, window_size=oca_window_size, window_statistic=oca_window_statistic)
    )
    pressure_c3 = clean_pressure_hpa(
        _sample_pressure_or_cloud_grid(ctp3, rows, cols, resolved_mode=resolved_mode, window_size=oca_window_size, window_statistic=oca_window_statistic)
    )

    component_valid_masks = [
        np.asarray(piv_data["valid_c1_mask"], dtype=bool),
        np.asarray(piv_data["valid_c2_mask"], dtype=bool),
        np.asarray(piv_data["valid_c3_mask"], dtype=bool),
    ]
    pressure_components_masked = []
    for pressure, valid in zip((pressure_c1, pressure_c2, pressure_c3), component_valid_masks):
        pressure_components_masked.append(
            np.where(valid & np.isfinite(pressure), pressure, np.nan).astype(np.float32)
        )
    pressure_stack = np.stack(pressure_components_masked, axis=0)
    pressure_hpa, pressure_component_count = _mean_valid_components(pressure_stack)

    base_mask = base_valid_mask(piv_data)
    keep = (
        base_mask
        & np.isfinite(pressure_hpa)
        & (pressure_component_count >= int(min_valid_pressure_components))
    )

    output = compact_piv_arrays(piv_data, keep)
    output.update(
        {
            "pressure_hpa": pressure_hpa[keep].astype(np.float32),
            "pressure_component_count": pressure_component_count[keep].astype(np.int16),
            "pressure_c1_hpa": pressure_components_masked[0][keep].astype(np.float32),
            "pressure_c2_hpa": pressure_components_masked[1][keep].astype(np.float32),
            "pressure_c3_hpa": pressure_components_masked[2][keep].astype(np.float32),
        }
    )

    if cloud_components is not None and len(cloud_components) >= 3 and any(item is not None for item in cloud_components[:3]):
        cloud_masked_components = []
        for cloud_grid, valid in zip(cloud_components[:3], component_valid_masks):
            if cloud_grid is None:
                cloud = np.full(rows.shape, np.nan, dtype=np.float32)
            else:
                cloud = _sample_pressure_or_cloud_grid(
                    np.asarray(cloud_grid, dtype=np.float32),
                    rows,
                    cols,
                    resolved_mode=resolved_mode,
                    window_size=oca_window_size,
                    window_statistic=oca_window_statistic,
                )
            cloud_masked_components.append(
                np.where(valid & np.isfinite(cloud), cloud, np.nan).astype(np.float32)
            )
        cloud_stack = np.stack(cloud_masked_components, axis=0)
        cloud_probability, _ = _mean_valid_components(cloud_stack)
        output.update(
            {
                "cloud_probability": cloud_probability[keep].astype(np.float32),
                "cloud_probability_c1": cloud_masked_components[0][keep].astype(np.float32),
                "cloud_probability_c2": cloud_masked_components[1][keep].astype(np.float32),
                "cloud_probability_c3": cloud_masked_components[2][keep].astype(np.float32),
            }
        )

    if include_height_m:
        output["height_m"] = pressure_to_standard_height_m(output["pressure_hpa"])

    metadata = {
        "product": "piv_amv_pressure_assigned",
        "input_product": "final_piv_wind_stacked_native",
        "pressure_source": str(pressure_source),
        "min_valid_pressure_components": int(min_valid_pressure_components),
        "spatial_assignment_requested": str(oca_assignment_mode),
        "spatial_assignment_resolved": resolved_mode,
        "spatial_window_size": int(oca_window_size) if resolved_mode == "window" else None,
        "spatial_window_statistic": str(oca_window_statistic) if resolved_mode == "window" else None,
        "component_pressure_combination": "mean_valid_components",
        "n_input_pixels": int(np.size(piv_data["u_ms"])),
        "n_output_points": int(output["pressure_hpa"].size),
    }
    if component_source is not None:
        metadata["component_source"] = str(component_source)
    if source_paths is not None:
        metadata["pressure_source_paths"] = [str(path) for path in source_paths[:3]]
    if "source_path" in piv_data:
        metadata["piv_source_path"] = str(piv_data["source_path"])

    if output_path is not None:
        saved = save_pressure_assigned_piv_amv_npz(
            output,
            output_path,
            pressure_source=str(pressure_source),
            metadata=metadata,
            compress=compress,
        )
        output["output_path"] = saved
    output["metadata"] = metadata
    return output


def assign_piv_pressure_from_oca_components(
    piv: Mapping[str, Any] | str | Path,
    oca: Any,
    output_path: str | Path | None = None,
    *,
    min_valid_pressure_components: int = 2,
    oca_assignment_mode: str = "auto",
    oca_window_size: int = 3,
    oca_window_statistic: str = "median",
    include_height_m: bool = False,
    compress: bool = True,
) -> dict[str, Any]:
    """Assign OCA component-start pressure grids to a stacked PIV final product."""

    pressure_components = _component_grid_triplet(oca, "ctp_hpa")
    cloud_values = getattr(oca, "cloud_probability", None)
    cloud_components = tuple(cloud_values[:3]) if cloud_values is not None and len(cloud_values) >= 3 else None

    return assign_piv_pressure_from_component_grids(
        piv=piv,
        pressure_components=pressure_components,
        output_path=output_path,
        cloud_components=cloud_components,
        pressure_source="oca",
        min_valid_pressure_components=min_valid_pressure_components,
        oca_assignment_mode=oca_assignment_mode,
        oca_window_size=oca_window_size,
        oca_window_statistic=oca_window_statistic,
        include_height_m=include_height_m,
        compress=compress,
        source_paths=list(getattr(oca, "paths", ())[:3]),
        component_source=getattr(oca, "component_source", "explicit_component_files"),
    )


def _lonlat_to_unit_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    cos_lat = np.cos(lat)
    return np.column_stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)))


def assign_piv_pressure_from_official_points(
    piv: Mapping[str, Any] | str | Path,
    official_frame: Any,
    output_path: str | Path | None = None,
    *,
    radius_km: float = 25.0,
    min_piv_points: int = 3,
    statistic: str = "mean",
    pressure_column: str = "p_official",
    lon_column: str = "lon",
    lat_column: str = "lat",
    id_column: str = "official_id",
    include_height_m: bool = False,
    compress: bool = True,
) -> dict[str, Any]:
    """Assign official-AMV pressure using official-anchored PIV wind averages.

    For every valid official AMV pressure point, find valid PIV vectors within
    ``radius_km``. If at least ``min_piv_points`` PIV neighbours exist, output
    one record at the official AMV lon/lat, with official pressure and local
    PIV-mean u/v. Points with too few PIV neighbours are not written.

    The output intentionally does not store PIV row/col, PIV component counts,
    official quality, or distance diagnostics.  It stores only the fields needed
    to represent this official-pressure PIV-AMV product plus provenance.
    """

    try:
        from scipy.spatial import cKDTree
    except Exception as exc:  # pragma: no cover - scipy is expected in this project
        raise ImportError("assign_piv_pressure_from_official_points requires scipy") from exc

    if radius_km <= 0:
        raise ValueError("radius_km must be positive")
    if int(min_piv_points) < 1:
        raise ValueError("min_piv_points must be at least 1")
    statistic = str(statistic).lower()
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")

    piv_data = _as_piv_dict(piv)
    base = base_valid_mask(piv_data)
    base_indices = np.flatnonzero(base.ravel())
    if base_indices.size == 0:
        raise ValueError("PIV product contains no valid wind points")

    lon_all = np.asarray(piv_data["longitude"], dtype=np.float32).ravel()
    lat_all = np.asarray(piv_data["latitude"], dtype=np.float32).ravel()
    u_all = np.asarray(piv_data["u_ms"], dtype=np.float32).ravel()
    v_all = np.asarray(piv_data["v_ms"], dtype=np.float32).ravel()

    piv_lon = lon_all[base_indices]
    piv_lat = lat_all[base_indices]
    piv_u = u_all[base_indices]
    piv_v = v_all[base_indices]

    tree = cKDTree(_lonlat_to_unit_xyz(piv_lon, piv_lat))
    radius_chord = 2.0 * np.sin(float(radius_km) / (2.0 * EARTH_RADIUS_KM))

    official = official_frame.copy()
    for col in (lon_column, lat_column, pressure_column):
        if col not in official.columns:
            raise KeyError(f"official_frame is missing column {col!r}")

    off_lon_all = official[lon_column].to_numpy(dtype=float)
    off_lat_all = official[lat_column].to_numpy(dtype=float)
    off_pressure_all = clean_pressure_hpa(official[pressure_column].to_numpy())
    official_valid = (
        np.isfinite(off_lon_all)
        & np.isfinite(off_lat_all)
        & np.isfinite(off_pressure_all)
    )
    if not np.any(official_valid):
        raise ValueError("No valid official pressure points are available")

    off_indices = np.flatnonzero(official_valid)
    off_lon = off_lon_all[off_indices]
    off_lat = off_lat_all[off_indices]
    off_pressure = off_pressure_all[off_indices]

    if id_column in official.columns:
        off_id_all = official[id_column].to_numpy()
        off_id = off_id_all[off_indices]
    else:
        off_id = off_indices

    outputs: dict[str, list[Any]] = {
        "official_id": [],
        "longitude": [],
        "latitude": [],
        "u_ms": [],
        "v_ms": [],
        "pressure_hpa": [],
        "piv_neighbor_count": [],
    }

    matches = tree.query_ball_point(_lonlat_to_unit_xyz(off_lon, off_lat), r=radius_chord)
    for j, hit_indices in enumerate(matches):
        if len(hit_indices) < int(min_piv_points):
            continue
        h = np.asarray(hit_indices, dtype=int)
        finite = np.isfinite(piv_u[h]) & np.isfinite(piv_v[h])
        if finite.sum() < int(min_piv_points):
            continue
        h = h[finite]

        if statistic == "mean":
            u_value = np.float32(np.nanmean(piv_u[h]))
            v_value = np.float32(np.nanmean(piv_v[h]))
        else:
            u_value = np.float32(np.nanmedian(piv_u[h]))
            v_value = np.float32(np.nanmedian(piv_v[h]))

        outputs["official_id"].append(off_id[j])
        outputs["longitude"].append(np.float32(off_lon[j]))
        outputs["latitude"].append(np.float32(off_lat[j]))
        outputs["u_ms"].append(u_value)
        outputs["v_ms"].append(v_value)
        outputs["pressure_hpa"].append(np.float32(off_pressure[j]))
        outputs["piv_neighbor_count"].append(np.int32(h.size))

    if not outputs["pressure_hpa"]:
        raise ValueError(
            "No official AMV pressure points had enough nearby valid PIV vectors. "
            "Try increasing radius_km or lowering min_piv_points."
        )

    output: dict[str, Any] = {}
    for key, values in outputs.items():
        if key == "official_id":
            output[key] = np.asarray(values)
        elif key == "piv_neighbor_count":
            output[key] = np.asarray(values, dtype=np.int32)
        else:
            output[key] = np.asarray(values, dtype=np.float32)

    if include_height_m:
        output["height_m"] = pressure_to_standard_height_m(output["pressure_hpa"])

    metadata = {
        "product": "piv_amv_pressure_assigned",
        "input_product": "final_piv_wind_stacked_native",
        "pressure_source": "official_amv",
        "assignment_definition": "official_lonlat_pressure_with_local_piv_uv",
        "wind_source": "piv_window_mean_u_v" if statistic == "mean" else "piv_window_median_u_v",
        "longitude_latitude_source": "official_amv",
        "radius_km": float(radius_km),
        "min_piv_points": int(min_piv_points),
        "piv_statistic": statistic,
        "n_official_pressure_points": int(off_pressure.size),
        "n_input_valid_piv_points": int(base_indices.size),
        "n_output_points": int(output["pressure_hpa"].size),
    }
    if "source_path" in piv_data:
        metadata["piv_source_path"] = str(piv_data["source_path"])

    if output_path is not None:
        saved = save_pressure_assigned_piv_amv_npz(
            output,
            output_path,
            pressure_source="official_amv",
            metadata=metadata,
            compress=compress,
        )
        output["output_path"] = saved
    output["metadata"] = metadata
    return output
