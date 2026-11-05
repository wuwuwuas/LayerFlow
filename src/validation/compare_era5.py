"""AMV versus ERA5 pressure-level comparison utilities.

This module treats ERA5 collocation as a reusable reference product:
ERA5 winds are sampled onto an AMV geometry defined by longitude, latitude,
assigned pressure, and a fixed valid time.  The collocated ERA5 product does
not store AMV winds or metrics, so the same cache can be reused by many PIV
tracking algorithms that share the same geometry/pressure assignment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Iterable
import json
import re

import numpy as np
import pandas as pd
import xarray as xr

from .metrics import direction_difference, summary_metrics


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _metadata_array(metadata: Mapping[str, Any] | None) -> np.ndarray:
    return np.array([json.dumps(dict(metadata or {}), default=str)])


def _decode_metadata(value: Any) -> dict[str, Any]:
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return {}
        text = str(arr.ravel()[0])
        return json.loads(text) if text else {}
    except Exception:
        return {}


def _scalar_text(value: Any, default: str = "") -> str:
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return default
        return str(arr.ravel()[0])
    except Exception:
        return default


def _safe_token(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", str(text).strip()).strip("_").lower()
    return value or "product"


def coverage_column(product_name: str) -> str:
    """Return the dynamic cache coverage-mask column for one AMV product."""

    return f"valid_{_safe_token(product_name)}"


def _coord_name(dataset: xr.Dataset, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in dataset.coords or name in dataset.dims or name in dataset.variables:
            return name
    raise KeyError(f"None of these coordinates/variables were found: {tuple(candidates)}")


def _var_name(dataset: xr.Dataset, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in dataset.data_vars or name in dataset.variables:
            return name
    raise KeyError(f"None of these ERA5 variables were found: {tuple(candidates)}")


def _prepare_era5_dataset(dataset: xr.Dataset) -> tuple[xr.Dataset, dict[str, str]]:
    """Return an ERA5 dataset sorted for interpolation and its standard names."""

    names = {
        "time": _coord_name(dataset, ("valid_time", "time")),
        "level": _coord_name(dataset, ("pressure_level", "level", "isobaricInhPa")),
        "lat": _coord_name(dataset, ("latitude", "lat")),
        "lon": _coord_name(dataset, ("longitude", "lon")),
        "u": _var_name(dataset, ("u", "u_component_of_wind")),
        "v": _var_name(dataset, ("v", "v_component_of_wind")),
    }

    prepared = dataset
    for coord in (names["time"], names["level"], names["lat"], names["lon"]):
        values = np.asarray(prepared[coord].values)
        if values.size > 1:
            try:
                if values[0] > values[-1]:
                    prepared = prepared.sortby(coord)
            except Exception:
                pass
    return prepared, names


def _target_longitudes_for_dataset(lon: np.ndarray, dataset_lon: np.ndarray) -> np.ndarray:
    """Map target longitudes onto ERA5 longitude convention."""

    target = np.asarray(lon, dtype=np.float64).copy()
    ds_lon = np.asarray(dataset_lon, dtype=np.float64)
    finite = np.isfinite(ds_lon)
    if not finite.any() or target.size == 0:
        return target
    lon_min = float(np.nanmin(ds_lon[finite]))
    lon_max = float(np.nanmax(ds_lon[finite]))
    if lon_min >= 0.0 and np.nanmin(target) < 0.0:
        target = np.mod(target, 360.0)
    elif lon_max <= 180.0 and np.nanmax(target) > 180.0:
        target = ((target + 180.0) % 360.0) - 180.0
    return target


def _vertical_log_pressure_interpolation(
    levels_hpa: np.ndarray,
    profiles: np.ndarray,
    targets_hpa: np.ndarray,
) -> np.ndarray:
    """Interpolate profiles to target pressure using log-pressure coordinates.

    ERA5 pressure-level profiles are normally complete at all levels after
    horizontal interpolation, so this uses a vectorised two-level bracket for
    speed.  Points outside the ERA5 pressure range or with non-finite bracket
    values remain NaN.
    """

    levels = np.asarray(levels_hpa, dtype=np.float64)
    values = np.asarray(profiles, dtype=np.float64)
    targets = np.asarray(targets_hpa, dtype=np.float64)
    output = np.full(targets.shape[0], np.nan, dtype=np.float32)

    finite_levels = np.isfinite(levels) & (levels > 0)
    if finite_levels.sum() < 2 or values.ndim != 2 or values.shape[0] != targets.shape[0]:
        return output

    order = np.argsort(levels[finite_levels])
    level_indices = np.where(finite_levels)[0][order]
    levels_sorted = levels[level_indices]
    values_sorted = values[:, level_indices]

    valid_target = (
        np.isfinite(targets)
        & (targets > 0)
        & (targets >= levels_sorted[0])
        & (targets <= levels_sorted[-1])
    )
    if not valid_target.any():
        return output

    upper = np.searchsorted(levels_sorted, targets[valid_target], side="left")
    upper = np.clip(upper, 1, len(levels_sorted) - 1)
    lower = upper - 1

    row_idx = np.where(valid_target)[0]
    p0 = levels_sorted[lower]
    p1 = levels_sorted[upper]
    y0 = values_sorted[row_idx, lower]
    y1 = values_sorted[row_idx, upper]

    exact_lower = np.isclose(targets[row_idx], p0, rtol=0, atol=1e-7)
    exact_upper = np.isclose(targets[row_idx], p1, rtol=0, atol=1e-7)
    finite_pair = np.isfinite(y0) & np.isfinite(y1)

    interp_rows = row_idx[finite_pair]
    if interp_rows.size:
        w = (np.log(targets[interp_rows]) - np.log(p0[finite_pair])) / (np.log(p1[finite_pair]) - np.log(p0[finite_pair]))
        output[interp_rows] = (y0[finite_pair] + w * (y1[finite_pair] - y0[finite_pair])).astype(np.float32)

    # If target lies exactly on a level, accept the finite single-level value.
    rows_lower = row_idx[exact_lower & np.isfinite(y0)]
    if rows_lower.size:
        output[rows_lower] = y0[exact_lower & np.isfinite(y0)].astype(np.float32)
    rows_upper = row_idx[exact_upper & np.isfinite(y1)]
    if rows_upper.size:
        output[rows_upper] = y1[exact_upper & np.isfinite(y1)].astype(np.float32)

    return output


# -----------------------------------------------------------------------------
# Input standardisation
# -----------------------------------------------------------------------------


def load_pressure_amv_npz(path: str | Path) -> pd.DataFrame:
    """Load a pressure-assigned AMV NPZ as a compact point table.

    Standard columns are ``longitude``, ``latitude``, ``pressure_hpa``,
    ``u_amv``, ``v_amv``, and ``speed_amv``.  Geometry identifiers such as
    ``row_center_original``, ``col_center_original``, and ``official_id`` are
    preserved when present.
    """

    source = _as_path(path)
    if not source.exists():
        raise FileNotFoundError(f"AMV NPZ not found: {source}")

    with np.load(source, allow_pickle=False) as data:
        files = list(data.files)
        if "longitude" not in files or "latitude" not in files or "pressure_hpa" not in files:
            raise KeyError(f"{source} must contain longitude, latitude, and pressure_hpa")
        n_points = int(np.asarray(data["longitude"]).size)
        columns: dict[str, Any] = {}
        for key in files:
            if key in {"metadata_json", "pressure_source"}:
                continue
            arr = np.asarray(data[key])
            if arr.ndim == 1 and arr.size == n_points:
                columns[key] = arr.copy()
        pressure_source = _scalar_text(data["pressure_source"], "") if "pressure_source" in files else ""
        metadata = _decode_metadata(data["metadata_json"]) if "metadata_json" in files else {}

    frame = pd.DataFrame(columns)
    if "u_ms" in frame.columns:
        frame = frame.rename(columns={"u_ms": "u_amv"})
    if "v_ms" in frame.columns:
        frame = frame.rename(columns={"v_ms": "v_amv"})
    if "speed_amv" not in frame.columns and {"u_amv", "v_amv"}.issubset(frame.columns):
        frame["speed_amv"] = np.hypot(frame["u_amv"], frame["v_amv"])
    frame["pressure_source"] = pressure_source
    frame["source_path"] = str(source)
    frame.attrs["metadata"] = metadata
    return frame


def standardize_official_amv(official_amv: Any) -> pd.DataFrame:
    """Return official AMV data using the AMV-vs-ERA5 schema."""

    if hasattr(official_amv, "frame"):
        frame = official_amv.frame.copy()
    elif isinstance(official_amv, pd.DataFrame):
        frame = official_amv.copy()
    else:
        raise TypeError("official_amv must be a DataFrame or an object with a .frame attribute")

    rename = {
        "lon": "longitude",
        "lat": "latitude",
        "p_official": "pressure_hpa",
        "u_official": "u_amv",
        "v_official": "v_amv",
        "speed_official": "speed_amv",
    }
    frame = frame.rename(columns={key: value for key, value in rename.items() if key in frame.columns})
    required = ("longitude", "latitude", "pressure_hpa", "u_amv", "v_amv")
    missing = [key for key in required if key not in frame.columns]
    if missing:
        raise KeyError(f"official AMV table is missing required columns after standardisation: {missing}")
    if "speed_amv" not in frame.columns:
        frame["speed_amv"] = np.hypot(frame["u_amv"], frame["v_amv"])
    finite = (
        np.isfinite(frame["longitude"])
        & np.isfinite(frame["latitude"])
        & np.isfinite(frame["pressure_hpa"])
        & np.isfinite(frame["u_amv"])
        & np.isfinite(frame["v_amv"])
    )
    keep = [col for col in frame.columns if col in {
        "official_id", "longitude", "latitude", "pressure_hpa", "u_amv", "v_amv", "speed_amv",
        "official_quality", "direction_official",
    }]
    return frame.loc[finite, keep].reset_index(drop=True)


def official_geometry_from_amv(official_amv: Any) -> pd.DataFrame:
    """Return official AMV geometry/pressure for ERA5 collocation."""

    frame = standardize_official_amv(official_amv)
    keep = [col for col in ("official_id", "longitude", "latitude", "pressure_hpa") if col in frame.columns]
    return frame[keep].copy()


def oca_geometry_from_pressure_amv(path: str | Path) -> pd.DataFrame:
    """Return OCA-P PIV-AMV geometry/pressure for ERA5 collocation."""

    frame = load_pressure_amv_npz(path)
    required = ("row_center_original", "col_center_original", "longitude", "latitude", "pressure_hpa")
    missing = [key for key in required if key not in frame.columns]
    if missing:
        raise KeyError(f"OCA-P AMV product is missing geometry keys: {missing}")
    keep = list(required)
    return frame[keep].copy()


# -----------------------------------------------------------------------------
# ERA5 collocation
# -----------------------------------------------------------------------------


def collocate_era5_to_points(
    points: pd.DataFrame,
    era5: str | Path | xr.Dataset,
    *,
    valid_time=None,
    time_column: str | None = None,
    chunk_size: int = 50_000,
    progress: bool = True,
) -> pd.DataFrame:
    """Sample ERA5 u/v to AMV lon/lat/pressure/time.

    ``points`` must contain ``longitude``, ``latitude``, and ``pressure_hpa``.
    Supply either one shared ``valid_time`` or ``time_column`` for pairwise
    point-specific observation times.  Existing fixed-time callers are
    unchanged.  The returned DataFrame contains the same index and ERA5
    columns only.
    """

    required = ("longitude", "latitude", "pressure_hpa")
    missing = [key for key in required if key not in points.columns]
    if missing:
        raise KeyError(f"points is missing required columns: {missing}")
    if (valid_time is None) == (time_column is None):
        raise ValueError(
            "Supply exactly one of valid_time or time_column"
        )
    if time_column is not None and time_column not in points.columns:
        raise KeyError(f"points is missing time column: {time_column!r}")

    dataset = xr.open_dataset(era5) if not isinstance(era5, xr.Dataset) else era5
    dataset, names = _prepare_era5_dataset(dataset)
    levels = np.asarray(dataset[names["level"]].values, dtype=np.float64)
    lon_values = np.asarray(dataset[names["lon"]].values, dtype=np.float64)

    n_points = len(points)
    u_out = np.full(n_points, np.nan, dtype=np.float32)
    v_out = np.full(n_points, np.nan, dtype=np.float32)

    shared_time_value = (
        None
        if valid_time is None
        else np.datetime64(pd.Timestamp(valid_time).to_datetime64())
    )
    all_time_values = (
        None
        if time_column is None
        else pd.to_datetime(points[time_column], errors="coerce")
        .to_numpy(dtype="datetime64[ns]")
    )
    all_lon = _target_longitudes_for_dataset(points["longitude"].to_numpy(float), lon_values)
    all_lat = points["latitude"].to_numpy(float)
    all_pressure = points["pressure_hpa"].to_numpy(float)

    if int(chunk_size) <= 0:
        chunk_size = max(n_points, 1)

    for start in range(0, n_points, int(chunk_size)):
        stop = min(start + int(chunk_size), n_points)
        if progress:
            print(f"  ERA5 interpolation points {start:,}–{stop:,} / {n_points:,}")
        chunk_points = xr.DataArray(np.arange(stop - start), dims="points")
        time_coord = (
            shared_time_value
            if all_time_values is None
            else xr.DataArray(all_time_values[start:stop], dims="points")
        )
        interp_coords = {
            names["time"]: time_coord,
            names["lat"]: xr.DataArray(all_lat[start:stop], dims="points"),
            names["lon"]: xr.DataArray(all_lon[start:stop], dims="points"),
        }
        sampled_profiles: dict[str, np.ndarray] = {}
        for out_name, var_name in (("u", names["u"]), ("v", names["v"])):
            sampled = dataset[var_name].interp(interp_coords)
            # Ensure a 2-D (points, level) profile array even if xarray returns
            # dimensions in a different order.
            if names["level"] not in sampled.dims:
                sampled_profiles[out_name] = np.full((stop - start, len(levels)), np.nan, dtype=np.float64)
                continue
            sampled = sampled.transpose("points", names["level"])
            sampled_profiles[out_name] = np.asarray(sampled.values, dtype=np.float64)
        u_out[start:stop] = _vertical_log_pressure_interpolation(levels, sampled_profiles["u"], all_pressure[start:stop])
        v_out[start:stop] = _vertical_log_pressure_interpolation(levels, sampled_profiles["v"], all_pressure[start:stop])

    out = pd.DataFrame(index=points.index)
    out["u_era5"] = u_out
    out["v_era5"] = v_out
    out["speed_era5"] = np.hypot(u_out, v_out).astype(np.float32)
    out["era5_valid_mask"] = np.isfinite(out["u_era5"]) & np.isfinite(out["v_era5"])
    return out


# -----------------------------------------------------------------------------
# Cache I/O
# -----------------------------------------------------------------------------


def save_era5_cache_npz(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    pressure_source: str,
    amv_geometry_source: str,
    metadata: Mapping[str, Any] | None = None,
    compress: bool = False,
) -> Path:
    """Save an ERA5-collocated AMV reference product."""

    destination = _as_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    output: dict[str, Any] = {}
    n_points = len(frame)
    for column in frame.columns:
        values = frame[column]
        if column.startswith("valid_") or column == "era5_valid_mask":
            output[column] = values.to_numpy(dtype=bool)
        elif column in {"official_id"}:
            output[column] = values.to_numpy(dtype=np.int32)
        elif column in {"row_center_original", "col_center_original", "longitude", "latitude", "pressure_hpa", "u_era5", "v_era5", "speed_era5"}:
            output[column] = values.to_numpy(dtype=np.float32)
        else:
            # Keep only 1-D numeric/boolean columns needed by comparison.
            arr = np.asarray(values)
            if arr.ndim == 1 and arr.shape[0] == n_points and arr.dtype.kind in {"i", "u", "f", "b"}:
                output[column] = arr

    output["pressure_source"] = np.array([str(pressure_source)])
    output["amv_geometry_source"] = np.array([str(amv_geometry_source)])
    merged_metadata = dict(metadata or {})
    merged_metadata.update(
        {
            "pressure_source": str(pressure_source),
            "amv_geometry_source": str(amv_geometry_source),
            "n_points": int(n_points),
        }
    )
    output["metadata_json"] = _metadata_array(merged_metadata)

    if compress:
        np.savez_compressed(destination, **output)
    else:
        np.savez(destination, **output)
    return destination


def load_era5_cache_npz(path: str | Path) -> pd.DataFrame:
    """Load an ERA5-collocated AMV reference product as a DataFrame."""

    source = _as_path(path)
    if not source.exists():
        raise FileNotFoundError(f"ERA5 collocation cache not found: {source}")
    with np.load(source, allow_pickle=False) as data:
        columns: dict[str, Any] = {}
        for key in data.files:
            if key in {"metadata_json", "pressure_source", "amv_geometry_source"}:
                continue
            arr = np.asarray(data[key])
            if arr.ndim == 1:
                columns[key] = arr.copy()
        metadata = _decode_metadata(data["metadata_json"]) if "metadata_json" in data.files else {}
        pressure_source = _scalar_text(data["pressure_source"], "") if "pressure_source" in data.files else ""
        geometry_source = _scalar_text(data["amv_geometry_source"], "") if "amv_geometry_source" in data.files else ""
    frame = pd.DataFrame(columns)
    frame.attrs["metadata"] = metadata
    frame.attrs["pressure_source"] = pressure_source
    frame.attrs["amv_geometry_source"] = geometry_source
    frame.attrs["source_path"] = str(source)
    return frame


# -----------------------------------------------------------------------------
# Cache generation/update
# -----------------------------------------------------------------------------


def ensure_official_era5_cache(
    *,
    official_amv: Any,
    era5: str | Path | xr.Dataset,
    cache_path: str | Path,
    valid_time,
    chunk_size: int = 50_000,
    rebuild: bool = False,
    compress: bool = False,
    progress: bool = True,
) -> Path:
    """Create or reuse the official-AMV-geometry ERA5 cache."""

    destination = _as_path(cache_path)
    if destination.exists() and not rebuild:
        if progress:
            print(f"Using existing official ERA5 cache: {destination}")
        return destination

    geometry = official_geometry_from_amv(official_amv)
    if progress:
        print(f"Building official ERA5 cache with {len(geometry):,} points")
    era5_values = collocate_era5_to_points(geometry, era5, valid_time=valid_time, chunk_size=chunk_size, progress=progress)
    cache = pd.concat([geometry.reset_index(drop=True), era5_values.reset_index(drop=True)], axis=1)
    metadata = {
        "era5_file": str(era5) if not isinstance(era5, xr.Dataset) else "xarray.Dataset",
        "amv_valid_time": str(pd.Timestamp(valid_time)),
        "interpolation_method": "xarray horizontal linear + log-pressure vertical linear",
    }
    return save_era5_cache_npz(
        cache,
        destination,
        pressure_source="official_amv",
        amv_geometry_source="official_amv",
        metadata=metadata,
        compress=compress,
    )


def _oca_key_frame(frame: pd.DataFrame, *, pressure_decimals: int = 2) -> pd.DataFrame:
    required = ("row_center_original", "col_center_original", "pressure_hpa")
    missing = [key for key in required if key not in frame.columns]
    if missing:
        raise KeyError(f"OCA geometry is missing key columns: {missing}")
    return pd.DataFrame(
        {
            "row_key": np.rint(pd.to_numeric(frame["row_center_original"], errors="coerce").to_numpy(float) * 100.0).astype(np.int64),
            "col_key": np.rint(pd.to_numeric(frame["col_center_original"], errors="coerce").to_numpy(float) * 100.0).astype(np.int64),
            "p_key": np.rint(pd.to_numeric(frame["pressure_hpa"], errors="coerce").to_numpy(float) * (10**pressure_decimals)).astype(np.int64),
        }
    )


def _official_key_frame(frame: pd.DataFrame, *, lonlat_decimals: int = 5, pressure_decimals: int = 2) -> pd.DataFrame:
    required = ("longitude", "latitude", "pressure_hpa")
    missing = [key for key in required if key not in frame.columns]
    if missing:
        raise KeyError(f"official geometry is missing key columns: {missing}")
    return pd.DataFrame(
        {
            "lon_key": np.rint(pd.to_numeric(frame["longitude"], errors="coerce").to_numpy(float) * (10**lonlat_decimals)).astype(np.int64),
            "lat_key": np.rint(pd.to_numeric(frame["latitude"], errors="coerce").to_numpy(float) * (10**lonlat_decimals)).astype(np.int64),
            "p_key": np.rint(pd.to_numeric(frame["pressure_hpa"], errors="coerce").to_numpy(float) * (10**pressure_decimals)).astype(np.int64),
        }
    )


def update_oca_era5_cache_from_amv_product(
    *,
    amv_product_path: str | Path,
    product_name: str,
    era5: str | Path | xr.Dataset,
    cache_path: str | Path,
    valid_time,
    amv_geometry_source: str = "of_dense_piv",
    pressure_source: str = "oca",
    chunk_size: int = 50_000,
    rebuild: bool = False,
    compress: bool = False,
    progress: bool = True,
) -> Path:
    """Create/update a union OCA-P PIV ERA5 cache from one AMV product.

    A dynamic boolean column ``valid_<product_name>`` is added or updated.
    Existing cache points are reused; only new row/col/pressure geometry points
    are interpolated against ERA5 and appended.
    """

    destination = _as_path(cache_path)
    coverage = coverage_column(product_name)
    geometry = oca_geometry_from_pressure_amv(amv_product_path).reset_index(drop=True)
    geometry_keys = _oca_key_frame(geometry)

    if rebuild and destination.exists():
        destination.unlink()

    if not destination.exists():
        if progress:
            print(f"Building OCA ERA5 cache {destination.name} from {product_name}: {len(geometry):,} points")
        era5_values = collocate_era5_to_points(geometry, era5, valid_time=valid_time, chunk_size=chunk_size, progress=progress)
        cache = pd.concat([geometry, era5_values.reset_index(drop=True)], axis=1)
        cache[coverage] = True
        metadata = {
            "era5_file": str(era5) if not isinstance(era5, xr.Dataset) else "xarray.Dataset",
            "amv_valid_time": str(pd.Timestamp(valid_time)),
            "interpolation_method": "xarray horizontal linear + log-pressure vertical linear",
            "coverage_products": [str(product_name)],
        }
        return save_era5_cache_npz(
            cache,
            destination,
            pressure_source=pressure_source,
            amv_geometry_source=amv_geometry_source,
            metadata=metadata,
            compress=compress,
        )

    cache = load_era5_cache_npz(destination).reset_index(drop=True)
    cache_keys = _oca_key_frame(cache)
    cache_index = pd.MultiIndex.from_frame(cache_keys)
    product_index = pd.MultiIndex.from_frame(geometry_keys)
    exists = product_index.isin(cache_index)

    if coverage not in cache.columns:
        cache[coverage] = False

    # Update coverage for existing points using a key-index merge.
    cache_lookup = cache_keys.copy()
    cache_lookup["_cache_index"] = np.arange(len(cache_lookup), dtype=np.int64)
    product_existing = geometry_keys.loc[exists].copy()
    if not product_existing.empty:
        merged = product_existing.merge(cache_lookup, on=["row_key", "col_key", "p_key"], how="left")
        matched_indices = merged["_cache_index"].dropna().astype(np.int64).to_numpy()
        cache.loc[matched_indices, coverage] = True

    new_geometry = geometry.loc[~exists].reset_index(drop=True)
    if progress:
        print(
            f"Updating OCA ERA5 cache with {product_name}: "
            f"existing={int(exists.sum()):,}, new={len(new_geometry):,}, total_before={len(cache):,}"
        )
    if not new_geometry.empty:
        era5_values = collocate_era5_to_points(new_geometry, era5, valid_time=valid_time, chunk_size=chunk_size, progress=progress)
        new_cache = pd.concat([new_geometry, era5_values.reset_index(drop=True)], axis=1)
        for column in [c for c in cache.columns if c.startswith("valid_")]:
            new_cache[column] = False
        new_cache[coverage] = True
        # Ensure both frames have the same columns before concatenation.
        for column in new_cache.columns:
            if column not in cache.columns:
                cache[column] = False if column.startswith("valid_") else np.nan
        for column in cache.columns:
            if column not in new_cache.columns:
                new_cache[column] = False if column.startswith("valid_") else np.nan
        cache = pd.concat([cache, new_cache[cache.columns]], ignore_index=True)

    metadata = dict(cache.attrs.get("metadata", {}))
    products = list(metadata.get("coverage_products", []))
    if str(product_name) not in products:
        products.append(str(product_name))
    metadata.update(
        {
            "era5_file": str(era5) if not isinstance(era5, xr.Dataset) else metadata.get("era5_file", "xarray.Dataset"),
            "amv_valid_time": str(pd.Timestamp(valid_time)),
            "interpolation_method": "xarray horizontal linear + log-pressure vertical linear",
            "coverage_products": products,
        }
    )
    if progress:
        print(f"Saving OCA ERA5 cache: {len(cache):,} union points")
    return save_era5_cache_npz(
        cache,
        destination,
        pressure_source=pressure_source,
        amv_geometry_source=amv_geometry_source,
        metadata=metadata,
        compress=compress,
    )


# -----------------------------------------------------------------------------
# Join AMV products with ERA5 caches and compute diagnostics/metrics
# -----------------------------------------------------------------------------


def _merge_by_official_id(amv: pd.DataFrame, cache: pd.DataFrame) -> pd.DataFrame:
    if "official_id" in amv.columns and "official_id" in cache.columns:
        return amv.merge(
            cache[["official_id", "u_era5", "v_era5", "speed_era5", "era5_valid_mask"]],
            on="official_id",
            how="inner",
            validate="many_to_one",
        )
    # Fallback for older products without official_id.
    amv_keys = _official_key_frame(amv)
    cache_keys = _official_key_frame(cache)
    amv_tmp = pd.concat([amv.reset_index(drop=True), amv_keys], axis=1)
    cache_tmp = pd.concat([
        cache[["u_era5", "v_era5", "speed_era5", "era5_valid_mask"]].reset_index(drop=True),
        cache_keys,
    ], axis=1)
    return amv_tmp.merge(cache_tmp, on=["lon_key", "lat_key", "p_key"], how="inner", validate="many_to_one")


def _merge_by_oca_geometry(amv: pd.DataFrame, cache: pd.DataFrame) -> pd.DataFrame:
    amv_keys = _oca_key_frame(amv)
    cache_keys = _oca_key_frame(cache)
    amv_tmp = pd.concat([amv.reset_index(drop=True), amv_keys], axis=1)
    cache_tmp = pd.concat([
        cache[["u_era5", "v_era5", "speed_era5", "era5_valid_mask"]].reset_index(drop=True),
        cache_keys,
    ], axis=1)
    return amv_tmp.merge(cache_tmp, on=["row_key", "col_key", "p_key"], how="inner", validate="many_to_one")


def add_era5_comparison_diagnostics(table: pd.DataFrame) -> pd.DataFrame:
    """Add AMV-minus-ERA5 point-level diagnostics."""

    result = table.copy()
    if "speed_amv" not in result.columns:
        result["speed_amv"] = np.hypot(result["u_amv"], result["v_amv"])
    if "speed_era5" not in result.columns:
        result["speed_era5"] = np.hypot(result["u_era5"], result["v_era5"])
    result["speed_bias_amv_minus_era5"] = result["speed_amv"] - result["speed_era5"]
    result["vector_difference_amv_vs_era5"] = np.hypot(
        result["u_amv"] - result["u_era5"],
        result["v_amv"] - result["v_era5"],
    )
    signed = direction_difference(result["u_amv"], result["v_amv"], result["u_era5"], result["v_era5"])
    result["signed_direction_difference_deg"] = signed
    result["abs_direction_difference_deg"] = np.abs(signed)
    result["pressure_layer"] = pd.cut(
        result["pressure_hpa"],
        bins=[-np.inf, 400.0, 700.0, np.inf],
        labels=["high", "middle", "low"],
        right=False,
    ).astype("object")
    return result


def build_official_amv_vs_era5_table(official_amv: Any, era5_cache: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Return official AMV wind joined to the official-geometry ERA5 cache."""

    amv = standardize_official_amv(official_amv)
    cache = load_era5_cache_npz(era5_cache) if not isinstance(era5_cache, pd.DataFrame) else era5_cache.copy()
    joined = _merge_by_official_id(amv, cache)
    joined["amv_source"] = "official_amv"
    return add_era5_comparison_diagnostics(joined).reset_index(drop=True)


def build_pressure_amv_vs_era5_table(
    amv_product_path: str | Path,
    era5_cache: str | Path | pd.DataFrame,
    *,
    join_mode: str,
    amv_source: str,
) -> pd.DataFrame:
    """Join one pressure-assigned PIV-AMV product with an ERA5 cache."""

    amv = load_pressure_amv_npz(amv_product_path)
    cache = load_era5_cache_npz(era5_cache) if not isinstance(era5_cache, pd.DataFrame) else era5_cache.copy()
    join_mode = str(join_mode).lower()
    if join_mode in {"official", "official_id", "official_pressure"}:
        joined = _merge_by_official_id(amv, cache)
    elif join_mode in {"oca", "row_col_pressure", "piv_oca"}:
        joined = _merge_by_oca_geometry(amv, cache)
    else:
        raise ValueError("join_mode must be 'official_id' or 'row_col_pressure'")
    joined["amv_source"] = str(amv_source)
    return add_era5_comparison_diagnostics(joined).reset_index(drop=True)


def build_era5_comparison_masks(table: pd.DataFrame) -> dict[str, pd.Series]:
    """Return only generic default masks for AMV-vs-ERA5 comparisons.

    QC thresholds, top-percentile masks, and other experiment-specific
    subset definitions should be created in the notebook and passed to
    ``compare_to_era5(..., subset_masks=...)``.
    """

    base = (
        np.isfinite(table.get("u_amv", np.nan))
        & np.isfinite(table.get("v_amv", np.nan))
        & np.isfinite(table.get("u_era5", np.nan))
        & np.isfinite(table.get("v_era5", np.nan))
    )
    if "era5_valid_mask" in table.columns:
        base &= table["era5_valid_mask"].astype(bool)
    masks: dict[str, pd.Series] = {
        "all_valid": pd.Series(base, index=table.index),
    }
    if "speed_amv" in table.columns:
        masks["amv_speed_ge5"] = masks["all_valid"] & table["speed_amv"].ge(5.0)
    if "pressure_hpa" in table.columns:
        p = table["pressure_hpa"]
        masks["pressure_high_lt400"] = masks["all_valid"] & p.lt(400.0)
        masks["pressure_mid_400_700"] = masks["all_valid"] & p.ge(400.0) & p.lt(700.0)
        masks["pressure_low_gt700"] = masks["all_valid"] & p.ge(700.0)
    return masks


def _rename_metric_keys(row: dict[str, Any]) -> dict[str, Any]:
    renamed = dict(row)
    mapping = {
        "u_bias": "u_bias_amv_minus_era5",
        "v_bias": "v_bias_amv_minus_era5",
        "signed_speed_bias": "speed_bias_amv_minus_era5",
    }
    for old, new in mapping.items():
        if old in renamed:
            renamed[new] = renamed.pop(old)
    return renamed


def compare_to_era5(
    comparison_table: pd.DataFrame,
    *,
    comparison: str = "amv_vs_era5",
    subset_masks: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Compute AMV-vs-ERA5 metrics for the standard subsets."""

    if comparison_table.empty:
        return pd.DataFrame()
    masks = subset_masks or build_era5_comparison_masks(comparison_table)
    rows: list[dict[str, Any]] = []
    for subset, mask in masks.items():
        selected = comparison_table.loc[mask.fillna(False)].copy()
        row = summary_metrics(
            selected.get("u_amv", pd.Series(dtype=float)),
            selected.get("v_amv", pd.Series(dtype=float)),
            selected.get("u_era5", pd.Series(dtype=float)),
            selected.get("v_era5", pd.Series(dtype=float)),
            subset=subset,
            comparison=comparison,
        )
        row = _rename_metric_keys(row)
        row["mean_pressure_hpa"] = float(selected["pressure_hpa"].mean()) if "pressure_hpa" in selected and not selected.empty else np.nan
        row["era5_valid_fraction"] = float(selected["era5_valid_mask"].mean()) if "era5_valid_mask" in selected and not selected.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Inspection helpers for notebooks
# -----------------------------------------------------------------------------


def inspect_official_time_variables(path: str | Path, max_values: int = 12) -> pd.DataFrame:
    """Return a compact summary of time/date-like variables in a NetCDF file."""

    source = _as_path(path)
    ds = xr.open_dataset(source)
    names: list[str] = []
    for name in list(ds.coords) + list(ds.data_vars):
        lower = name.lower()
        if any(token in lower for token in ("time", "date", "start", "end")):
            names.append(name)
    rows: list[dict[str, Any]] = []
    for name in names:
        arr = ds[name]
        values = np.asarray(arr.values)
        flat = values.ravel()
        preview = flat[:max_values]
        row: dict[str, Any] = {
            "name": name,
            "dims": str(arr.dims),
            "shape": str(arr.shape),
            "dtype": str(values.dtype),
            "first_values": np.array2string(preview, threshold=max_values),
        }
        try:
            if np.issubdtype(values.dtype, np.datetime64) and values.size:
                row["min"] = str(np.nanmin(values))
                row["max"] = str(np.nanmax(values))
            elif values.size:
                numeric = pd.to_numeric(flat, errors="coerce")
                numeric = numeric[np.isfinite(numeric)]
                if numeric.size:
                    row["min"] = float(numeric.min())
                    row["max"] = float(numeric.max())
        except Exception as exc:  # pragma: no cover - inspection only
            row["summary_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    ds.close()
    return pd.DataFrame(rows)


def cache_coverage_summary(cache_path: str | Path) -> pd.DataFrame:
    """Summarise dynamic valid_* coverage columns in an ERA5 cache."""

    cache = load_era5_cache_npz(cache_path)
    rows = []
    n = len(cache)
    for column in cache.columns:
        if column.startswith("valid_") and column != "era5_valid_mask":
            count = int(cache[column].astype(bool).sum())
            rows.append({"coverage_column": column, "count": count, "fraction": count / n if n else np.nan})
    return pd.DataFrame(rows)


# =============================================================================
# Fast fixed-geometry path
# =============================================================================
# The functions below intentionally override the slower union/coverage cache logic
# above.  For the current AMV workflow, official-P and OCA-P geometries are fixed:
# ERA5 only needs to be interpolated once to those fixed P points.  Later tracker
# products reuse the same ERA5 cache and only join u/v before computing metrics.

_ERA5_CACHE_FRAME_CACHE: dict[str, pd.DataFrame] = {}


def clear_era5_cache_memory() -> None:
    """Clear in-process ERA5 cache DataFrames loaded by fast comparison helpers."""

    _ERA5_CACHE_FRAME_CACHE.clear()


def _get_era5_cache_frame(era5_cache: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load one ERA5 cache once per Python process.

    Notebook loops often compare many trackers against the same official/OCA ERA5
    cache.  Re-reading a multi-million-point NPZ for every tracker is unnecessary,
    so path inputs are memoized.  Passing an explicit DataFrame still works.
    """

    if isinstance(era5_cache, pd.DataFrame):
        return era5_cache
    key = str(_as_path(era5_cache))
    if key not in _ERA5_CACHE_FRAME_CACHE:
        _ERA5_CACHE_FRAME_CACHE[key] = load_era5_cache_npz(key).reset_index(drop=True)
    return _ERA5_CACHE_FRAME_CACHE[key]


def _append_era5_columns_direct(amv: pd.DataFrame, cache: pd.DataFrame) -> pd.DataFrame:
    """Append ERA5 columns by row order without a Pandas key merge."""

    era5_cols = ["u_era5", "v_era5", "speed_era5", "era5_valid_mask"]
    missing = [col for col in era5_cols if col not in cache.columns]
    if missing:
        raise KeyError(f"ERA5 cache is missing columns: {missing}")
    return pd.concat(
        [amv.reset_index(drop=True), cache.loc[:, era5_cols].reset_index(drop=True)],
        axis=1,
    )


def _official_geometry_same_order(amv: pd.DataFrame, cache: pd.DataFrame) -> bool:
    """True if official AMV product and ERA5 cache are row-aligned."""

    if len(amv) != len(cache):
        return False
    if "official_id" in amv.columns and "official_id" in cache.columns:
        return np.array_equal(
            pd.to_numeric(amv["official_id"], errors="coerce").to_numpy(np.int64),
            pd.to_numeric(cache["official_id"], errors="coerce").to_numpy(np.int64),
        )
    required = ("longitude", "latitude", "pressure_hpa")
    if any(col not in amv.columns or col not in cache.columns for col in required):
        return False
    return bool(
        np.allclose(amv["longitude"].to_numpy(float), cache["longitude"].to_numpy(float), rtol=0, atol=1e-5, equal_nan=True)
        and np.allclose(amv["latitude"].to_numpy(float), cache["latitude"].to_numpy(float), rtol=0, atol=1e-5, equal_nan=True)
        and np.allclose(amv["pressure_hpa"].to_numpy(float), cache["pressure_hpa"].to_numpy(float), rtol=0, atol=1e-2, equal_nan=True)
    )


def _oca_geometry_same_order(amv: pd.DataFrame, cache: pd.DataFrame, *, pressure_decimals: int = 2) -> bool:
    """True if OCA-P AMV product and ERA5 cache are row/col/pressure aligned."""

    if len(amv) != len(cache):
        return False
    required = ("row_center_original", "col_center_original", "pressure_hpa")
    if any(col not in amv.columns or col not in cache.columns for col in required):
        return False
    scale = 10 ** pressure_decimals
    return bool(
        np.array_equal(
            np.rint(amv["row_center_original"].to_numpy(float) * 100.0).astype(np.int64),
            np.rint(cache["row_center_original"].to_numpy(float) * 100.0).astype(np.int64),
        )
        and np.array_equal(
            np.rint(amv["col_center_original"].to_numpy(float) * 100.0).astype(np.int64),
            np.rint(cache["col_center_original"].to_numpy(float) * 100.0).astype(np.int64),
        )
        and np.array_equal(
            np.rint(amv["pressure_hpa"].to_numpy(float) * scale).astype(np.int64),
            np.rint(cache["pressure_hpa"].to_numpy(float) * scale).astype(np.int64),
        )
    )


def ensure_oca_era5_cache_from_reference(
    *,
    amv_product_path: str | Path,
    era5: str | Path | xr.Dataset,
    cache_path: str | Path,
    valid_time,
    amv_geometry_source: str = "of_dense_piv",
    pressure_source: str = "oca",
    chunk_size: int = 50_000,
    rebuild: bool = False,
    compress: bool = False,
    progress: bool = True,
) -> Path:
    """Create or reuse a fixed-geometry OCA-P ERA5 cache.

    This is the fast replacement for the older union/coverage cache.  It assumes
    that OCA pressure assignment gives the same row/col/pressure geometry for all
    trackers.  ERA5 is interpolated once to the reference product geometry; all
    later products are compared against this cache by row order when possible.
    """

    destination = _as_path(cache_path)
    if destination.exists() and not rebuild:
        if progress:
            print(f"Using existing fixed OCA ERA5 cache: {destination}")
        return destination

    if rebuild and destination.exists():
        destination.unlink()

    geometry = oca_geometry_from_pressure_amv(amv_product_path).reset_index(drop=True)
    if progress:
        print(f"Building fixed OCA ERA5 cache {destination.name}: {len(geometry):,} points")
    era5_values = collocate_era5_to_points(
        geometry,
        era5,
        valid_time=valid_time,
        chunk_size=chunk_size,
        progress=progress,
    )
    cache = pd.concat([geometry, era5_values.reset_index(drop=True)], axis=1)
    metadata = {
        "era5_file": str(era5) if not isinstance(era5, xr.Dataset) else "xarray.Dataset",
        "amv_valid_time": str(pd.Timestamp(valid_time)),
        "interpolation_method": "xarray horizontal linear + log-pressure vertical linear",
        "cache_mode": "fixed_oca_geometry_no_union_coverage",
        "reference_product": str(amv_product_path),
    }
    saved = save_era5_cache_npz(
        cache,
        destination,
        pressure_source=pressure_source,
        amv_geometry_source=amv_geometry_source,
        metadata=metadata,
        compress=compress,
    )
    _ERA5_CACHE_FRAME_CACHE.pop(str(destination), None)
    return saved


def update_oca_era5_cache_from_amv_product(
    *,
    amv_product_path: str | Path,
    product_name: str,
    era5: str | Path | xr.Dataset,
    cache_path: str | Path,
    valid_time,
    amv_geometry_source: str = "of_dense_piv",
    pressure_source: str = "oca",
    chunk_size: int = 50_000,
    rebuild: bool = False,
    compress: bool = False,
    progress: bool = True,
) -> Path:
    """Backward-compatible fast OCA cache creator.

    The old implementation updated a union cache and a dynamic valid_<product>
    coverage column for every tracker.  That is unnecessary when OCA-P geometry
    is fixed.  This function now builds the cache from the first/reference product
    and simply reuses it for later products.
    """

    if progress:
        print(f"OCA ERA5 fixed-geometry cache request from {product_name}")
    return ensure_oca_era5_cache_from_reference(
        amv_product_path=amv_product_path,
        era5=era5,
        cache_path=cache_path,
        valid_time=valid_time,
        amv_geometry_source=amv_geometry_source,
        pressure_source=pressure_source,
        chunk_size=chunk_size,
        rebuild=rebuild,
        compress=compress,
        progress=progress,
    )


def _merge_by_official_id(amv: pd.DataFrame, cache: pd.DataFrame) -> pd.DataFrame:
    """Join official-P products with ERA5 cache, using direct row alignment first."""

    cache = cache.reset_index(drop=True)
    amv = amv.reset_index(drop=True)
    if _official_geometry_same_order(amv, cache):
        return _append_era5_columns_direct(amv, cache)

    if "official_id" in amv.columns and "official_id" in cache.columns:
        return amv.merge(
            cache[["official_id", "u_era5", "v_era5", "speed_era5", "era5_valid_mask"]],
            on="official_id",
            how="inner",
            validate="many_to_one",
        )

    amv_keys = _official_key_frame(amv)
    cache_keys = _official_key_frame(cache)
    amv_tmp = pd.concat([amv.reset_index(drop=True), amv_keys], axis=1)
    cache_tmp = pd.concat([
        cache[["u_era5", "v_era5", "speed_era5", "era5_valid_mask"]].reset_index(drop=True),
        cache_keys,
    ], axis=1)
    return amv_tmp.merge(cache_tmp, on=["lon_key", "lat_key", "p_key"], how="inner", validate="many_to_one")


def _merge_by_oca_geometry(amv: pd.DataFrame, cache: pd.DataFrame) -> pd.DataFrame:
    """Join OCA-P products with ERA5 cache, using direct row alignment first."""

    cache = cache.reset_index(drop=True)
    amv = amv.reset_index(drop=True)
    if _oca_geometry_same_order(amv, cache):
        return _append_era5_columns_direct(amv, cache)

    # Fallback only for non-identical geometry/order.  This should rarely run in
    # the fixed-OCA workflow, but it preserves compatibility with older products.
    amv_keys = _oca_key_frame(amv)
    cache_keys = _oca_key_frame(cache)
    amv_tmp = pd.concat([amv.reset_index(drop=True), amv_keys], axis=1)
    cache_tmp = pd.concat([
        cache[["u_era5", "v_era5", "speed_era5", "era5_valid_mask"]].reset_index(drop=True),
        cache_keys,
    ], axis=1)
    return amv_tmp.merge(cache_tmp, on=["row_key", "col_key", "p_key"], how="inner", validate="many_to_one")


def build_official_amv_vs_era5_table(official_amv: Any, era5_cache: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Return official AMV wind joined to the official-geometry ERA5 cache."""

    amv = standardize_official_amv(official_amv)
    cache = _get_era5_cache_frame(era5_cache)
    joined = _merge_by_official_id(amv, cache)
    joined["amv_source"] = "official_amv"
    return add_era5_comparison_diagnostics(joined).reset_index(drop=True)


def build_pressure_amv_vs_era5_table(
    amv_product_path: str | Path,
    era5_cache: str | Path | pd.DataFrame,
    *,
    join_mode: str,
    amv_source: str,
) -> pd.DataFrame:
    """Join one pressure-assigned PIV-AMV product with an ERA5 cache.

    Fast path: if the product and cache have the same fixed geometry/order,
    ERA5 columns are appended directly by row index, avoiding a multi-million-row
    Pandas merge.
    """

    amv = load_pressure_amv_npz(amv_product_path)
    cache = _get_era5_cache_frame(era5_cache)
    join_mode = str(join_mode).lower()
    if join_mode in {"official", "official_id", "official_pressure"}:
        joined = _merge_by_official_id(amv, cache)
    elif join_mode in {"oca", "row_col_pressure", "piv_oca"}:
        joined = _merge_by_oca_geometry(amv, cache)
    else:
        raise ValueError("join_mode must be 'official_id' or 'row_col_pressure'")
    joined["amv_source"] = str(amv_source)
    return add_era5_comparison_diagnostics(joined).reset_index(drop=True)
