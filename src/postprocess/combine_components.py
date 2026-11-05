"""Average valid temporal wind components into a dense native product.

Retain component winds, validity masks and pixel displacements in the output."""

from __future__ import annotations

import warnings
from typing import Any, Mapping

import numpy as np


def _first_attr(obj: Any, names: tuple[str, ...] | list[str], default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _as_float32(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _as_bool(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=bool)


def validate_same_centers_and_geolocation(
    winds: Mapping[str, Any],
    *,
    check_lonlat: bool = True,
    atol: float = 1e-6,
) -> None:
    """Verify that all component winds share vector centers and optionally lon/lat.

    This lets the final native file store one common row/col/lon/lat grid instead
    of repeating these arrays for C1/C2/C3.
    """

    if not winds:
        raise ValueError("No component winds were provided.")

    names = list(winds)
    first = winds[names[0]]
    row0 = _as_float32(_first_attr(first, ["row_center_original"]))
    col0 = _as_float32(_first_attr(first, ["col_center_original"]))
    lon0 = _as_float32(_first_attr(first, ["longitude"]))
    lat0 = _as_float32(_first_attr(first, ["latitude"]))

    for name in names[1:]:
        wind = winds[name]
        row = _as_float32(_first_attr(wind, ["row_center_original"]))
        col = _as_float32(_first_attr(wind, ["col_center_original"]))
        if row.shape != row0.shape or col.shape != col0.shape:
            raise ValueError(f"{name} center shape differs from {names[0]}")
        if not (np.allclose(row, row0, equal_nan=True, atol=atol) and np.allclose(col, col0, equal_nan=True, atol=atol)):
            raise ValueError(f"{name} vector centers differ from {names[0]}")

        if check_lonlat:
            lon = _as_float32(_first_attr(wind, ["longitude"]))
            lat = _as_float32(_first_attr(wind, ["latitude"]))
            if lon.shape != lon0.shape or lat.shape != lat0.shape:
                raise ValueError(f"{name} lon/lat shape differs from {names[0]}")
            if not (np.allclose(lon, lon0, equal_nan=True, atol=atol) and np.allclose(lat, lat0, equal_nan=True, atol=atol)):
                raise ValueError(f"{name} lon/lat differs from {names[0]}")


def combine_components_to_stacked_arrays(
    flows: Mapping[str, Any],
    winds: Mapping[str, Any],
    *,
    min_valid_components: int = 2,
    component_order: tuple[str, ...] = ("C1", "C2", "C3"),
    check_lonlat: bool = True,
) -> dict[str, Any]:
    """Create one clean PIV-native output dictionary.

    The output contains one common coordinate grid, final mean u/v, and C1/C2/C3
    component u/v, dx/dy, and valid masks. Speed and wind direction are not
    saved because they are exactly reproducible from u/v.
    """

    if not 1 <= int(min_valid_components) <= len(component_order):
        raise ValueError("min_valid_components must be between 1 and number of components")

    missing = [name for name in component_order if name not in winds]
    if missing:
        raise ValueError(f"Missing component winds: {missing}")

    wind_subset = {name: winds[name] for name in component_order}
    validate_same_centers_and_geolocation(wind_subset, check_lonlat=check_lonlat)

    first = winds[component_order[0]]
    arrays: dict[str, Any] = {
        "row_center_original": _as_float32(_first_attr(first, ["row_center_original"])),
        "col_center_original": _as_float32(_first_attr(first, ["col_center_original"])),
        "longitude": _as_float32(_first_attr(first, ["longitude"])),
        "latitude": _as_float32(_first_attr(first, ["latitude"])),
    }

    u_items = []
    v_items = []
    valid_items = []

    for name in component_order:
        suffix = name.lower()
        wind = winds[name]
        flow = flows.get(name)

        u = _as_float32(_first_attr(wind, ["u_ms"]))
        v = _as_float32(_first_attr(wind, ["v_ms"]))
        valid = _as_bool(_first_attr(wind, ["valid_mask"])) & np.isfinite(u) & np.isfinite(v)

        arrays[f"u_{suffix}_ms"] = np.where(valid, u, np.nan).astype(np.float32)
        arrays[f"v_{suffix}_ms"] = np.where(valid, v, np.nan).astype(np.float32)
        arrays[f"valid_{suffix}_mask"] = valid.astype(bool)

        if flow is not None:
            dx = _first_attr(flow, ["dx_pixels", "dx", "flow_x"], default=None)
            dy = _first_attr(flow, ["dy_pixels", "dy", "flow_y"], default=None)
            if dx is not None:
                arrays[f"dx_{suffix}_pixels"] = _as_float32(dx)
            if dy is not None:
                arrays[f"dy_{suffix}_pixels"] = _as_float32(dy)

        u_items.append(np.where(valid, u, np.nan).astype(np.float32))
        v_items.append(np.where(valid, v, np.nan).astype(np.float32))
        valid_items.append(valid)

    u_stack = np.stack(u_items, axis=0)
    v_stack = np.stack(v_items, axis=0)
    valid_stack = np.stack(valid_items, axis=0)
    valid_component_count = valid_stack.sum(axis=0).astype(np.int16)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        u_final = np.nanmean(u_stack, axis=0).astype(np.float32)
        v_final = np.nanmean(v_stack, axis=0).astype(np.float32)

    valid_final = (
        (valid_component_count >= int(min_valid_components))
        & np.isfinite(u_final)
        & np.isfinite(v_final)
    )

    arrays["u_ms"] = np.where(valid_final, u_final, np.nan).astype(np.float32)
    arrays["v_ms"] = np.where(valid_final, v_final, np.nan).astype(np.float32)
    arrays["valid_mask"] = valid_final.astype(bool)
    arrays["valid_component_count"] = valid_component_count

    arrays["metadata"] = {
        "product": "final_piv_wind_stacked_native",
        "component_method": "mean_valid_u_v",
        "minimum_valid_components": int(min_valid_components),
        "component_order": list(component_order),
        "contains_pressure": False,
        "contains_height": False,
        "contains_qc_diagnostics": False,
        "speed_direction_saved": False,
    }
    return arrays
