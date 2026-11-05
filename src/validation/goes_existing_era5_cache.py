"""Reuse an existing GOES ERA5 collocation cache for AMV products.

The cache geometry is defined by satellite, geometry mode, centre-frame
longitude/latitude, centre-frame cloud-top pressure, and valid time. AMV winds
are deliberately ignored when reusing the cache. No ERA5 interpolation is
performed in this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from validation.compare_era5 import (
    add_era5_comparison_diagnostics,
    load_era5_cache_npz,
    load_pressure_amv_npz,
)

_REQUIRED_GEOMETRY = (
    "row_center_original",
    "col_center_original",
    "longitude",
    "latitude",
    "pressure_hpa",
)
_REQUIRED_ERA5 = ("u_era5", "v_era5", "speed_era5", "era5_valid_mask")


def pcf_cache_candidates(
    project_root: str | Path,
    case_name: str,
    satellite: str,
    geometry_mode: str,
) -> tuple[Path, ...]:
    """Return known locations of the cache previously written by PCF evaluation."""

    root = Path(project_root).expanduser().resolve()
    sat = str(satellite).strip().lower()
    tracker = f"pressure_conditioned_flow_{sat}"
    filename = f"{tracker}_{geometry_mode}_era5_collocated.npz"
    output_root = root / "outputs" / case_name
    return (
        output_root / "goes_pcf_vs_era5" / filename,
        output_root / "comparison" / "amv_vs_era5" / "era5_collocated" / filename,
    )


def resolve_existing_pcf_cache(
    project_root: str | Path,
    case_name: str,
    satellite: str,
    geometry_mode: str,
) -> Path:
    """Resolve an already-existing PCF cache; never build a replacement."""

    candidates = pcf_cache_candidates(project_root, case_name, satellite, geometry_mode)
    for path in candidates:
        if path.is_file():
            return path
    listed = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Existing PCF ERA5 cache was not found. The traditional evaluator is "
        "configured to reuse it and will not rerun ERA5 interpolation. Checked:\n"
        f"{listed}"
    )


def _require_columns(frame: pd.DataFrame, names: Iterable[str], label: str) -> None:
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def _integer_pixel_keys(frame: pd.DataFrame, label: str) -> np.ndarray:
    rows = frame["row_center_original"].to_numpy(dtype=np.float64)
    cols = frame["col_center_original"].to_numpy(dtype=np.float64)
    if not np.isfinite(rows).all() or not np.isfinite(cols).all():
        raise ValueError(f"{label} contains non-finite row/column coordinates")
    row_i = np.rint(rows).astype(np.int64)
    col_i = np.rint(cols).astype(np.int64)
    if np.max(np.abs(rows - row_i), initial=0.0) > 1.0e-4:
        raise ValueError(f"{label} row coordinates are not integer pixel indices")
    if np.max(np.abs(cols - col_i), initial=0.0) > 1.0e-4:
        raise ValueError(f"{label} column coordinates are not integer pixel indices")
    if (row_i < 0).any() or (col_i < 0).any():
        raise ValueError(f"{label} contains negative row/column coordinates")
    if (row_i >= 2**31).any() or (col_i >= 2**31).any():
        raise ValueError(f"{label} row/column coordinates exceed key capacity")
    return (row_i << np.int64(32)) | col_i


def _direct_geometry_match(amv: pd.DataFrame, cache: pd.DataFrame) -> bool:
    if len(amv) != len(cache):
        return False
    return all(
        np.array_equal(amv[name].to_numpy(), cache[name].to_numpy())
        for name in ("row_center_original", "col_center_original")
    )


def _subset_index(amv: pd.DataFrame, cache: pd.DataFrame) -> np.ndarray:
    """Map AMV rows to cache rows without constructing a giant DataFrame join."""

    amv_keys = _integer_pixel_keys(amv, "AMV product")
    cache_keys = _integer_pixel_keys(cache, "PCF ERA5 cache")
    order = np.argsort(cache_keys, kind="mergesort")
    sorted_keys = cache_keys[order]
    if len(sorted_keys) > 1 and np.any(sorted_keys[1:] == sorted_keys[:-1]):
        raise ValueError("PCF ERA5 cache contains duplicate centre-pixel keys")
    positions = np.searchsorted(sorted_keys, amv_keys)
    in_range = positions < len(sorted_keys)
    matched = np.zeros(len(amv_keys), dtype=bool)
    matched[in_range] = sorted_keys[positions[in_range]] == amv_keys[in_range]
    if not matched.all():
        missing = int((~matched).sum())
        raise ValueError(
            f"AMV product contains {missing:,} centre pixels absent from the existing "
            "PCF ERA5 cache. ERA5 interpolation was not rerun."
        )
    return order[positions]


def _geometry_differences(
    amv: pd.DataFrame,
    cache: pd.DataFrame,
    cache_index: np.ndarray,
) -> dict[str, float]:
    output: dict[str, float] = {}
    for name in ("longitude", "latitude", "pressure_hpa"):
        left = amv[name].to_numpy(dtype=np.float64)
        right = cache[name].to_numpy(dtype=np.float64)[cache_index]
        finite = np.isfinite(left) & np.isfinite(right)
        mismatch_finite = np.isfinite(left) ^ np.isfinite(right)
        if mismatch_finite.any():
            output[name] = np.inf
        elif finite.any():
            output[name] = float(np.max(np.abs(left[finite] - right[finite])))
        else:
            output[name] = 0.0
    return output


def build_amv_vs_existing_goes_era5_cache(
    amv_product_path: str | Path,
    cache_path: str | Path,
    *,
    longitude_latitude_tolerance_deg: float = 2.0e-5,
    pressure_tolerance_hpa: float = 2.0e-3,
    amv_source: str = "goes_traditional_optical_flow",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach existing ERA5 values to one GOES AMV product.

    The fast path reuses arrays in their existing order. If an algorithm has a
    strict subset of the common centre pixels, a pixel-key lookup extracts the
    corresponding cache rows. Geometry is always validated before ERA5 values
    are attached.
    """

    product_path = Path(amv_product_path).expanduser().resolve()
    reference_path = Path(cache_path).expanduser().resolve()
    amv = load_pressure_amv_npz(product_path)
    cache = load_era5_cache_npz(reference_path)
    _require_columns(amv, _REQUIRED_GEOMETRY, "AMV product")
    _require_columns(cache, (*_REQUIRED_GEOMETRY, *_REQUIRED_ERA5), "PCF ERA5 cache")

    direct = _direct_geometry_match(amv, cache)
    if direct:
        cache_index = np.arange(len(amv), dtype=np.int64)
        alignment = "direct_same_order"
    else:
        cache_index = _subset_index(amv, cache)
        alignment = "centre_pixel_subset_lookup"

    differences = _geometry_differences(amv, cache, cache_index)
    if differences["longitude"] > float(longitude_latitude_tolerance_deg):
        raise ValueError(
            "Longitude geometry differs from the PCF cache: "
            f"max_abs_difference={differences['longitude']:.9g} deg"
        )
    if differences["latitude"] > float(longitude_latitude_tolerance_deg):
        raise ValueError(
            "Latitude geometry differs from the PCF cache: "
            f"max_abs_difference={differences['latitude']:.9g} deg"
        )
    if differences["pressure_hpa"] > float(pressure_tolerance_hpa):
        raise ValueError(
            "Pressure geometry differs from the PCF cache: "
            f"max_abs_difference={differences['pressure_hpa']:.9g} hPa"
        )

    table = amv.copy()
    for name in _REQUIRED_ERA5:
        table[name] = cache[name].to_numpy()[cache_index]
    if "era5_region" in cache.columns:
        table["era5_region"] = cache["era5_region"].to_numpy()[cache_index]
    table["amv_source"] = str(amv_source)
    table = add_era5_comparison_diagnostics(table).reset_index(drop=True)

    info = {
        "cache_path": str(reference_path),
        "cache_points": int(len(cache)),
        "product_points": int(len(amv)),
        "alignment": alignment,
        "max_abs_longitude_difference_deg": differences["longitude"],
        "max_abs_latitude_difference_deg": differences["latitude"],
        "max_abs_pressure_difference_hpa": differences["pressure_hpa"],
        "era5_interpolation_rerun": False,
    }
    return table, info
