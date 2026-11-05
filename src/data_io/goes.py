"""GOES ABI case discovery, standardised reading, and inspection helpers.

The downloader stores a resolved paired case under ``data/<case_name>`` and
writes ``manifests/goes_selected_products.csv``.  This module treats that
manifest as the authoritative mapping between temporal roles and local files.
It does not contact NOAA S3 and is therefore suitable for the offline server.

The first-stage reader standardises the four core fields used by the PCF
experiment:

* ABI Band-14 Cloud and Moisture Imagery brightness temperature -> kelvin
* ABI Cloud Top Height -> metres
* ABI Cloud Top Pressure -> hPa
* ABI Clear Sky Mask -> categorical values as stored by NOAA

Files are opened one at a time during inspection to avoid retaining all six
Full Disk frames in memory.  ``load_goes_frame`` is the later tracking-facing
API and loads one satellite/time role at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CORE_PRODUCTS = ("image", "height", "pressure", "cloud_mask")
ROLE_ORDER = {"previous": 0, "center": 1, "next": 2}
PRODUCT_ORDER = {name: index for index, name in enumerate(CORE_PRODUCTS)}

PRIMARY_VARIABLE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "image": ("CMI", "brightness_temperature", "Brightness_Temperature"),
    "height": ("HT", "ACHA", "cloud_top_height", "Cloud_Top_Height"),
    "pressure": ("P", "CTP", "cloud_top_pressure", "Cloud_Top_Pressure"),
    # ACM is the four-level mask when available; BCM is the binary mask.
    "cloud_mask": ("ACM", "BCM", "cloud_mask", "Clear_Sky_Mask"),
}

QUALITY_VARIABLE_CANDIDATES = ("DQF", "quality_flag", "Quality_Flag")
PROJECTION_VARIABLE_CANDIDATES = (
    "goes_imager_projection",
    "goes_projection",
)


@dataclass(frozen=True)
class GoesFramePaths:
    """Resolved local paths for one satellite and one temporal role."""

    satellite: str
    role: str
    scan_start: datetime
    scan_end: datetime
    scan_midpoint: datetime
    image: Path
    height: Path
    pressure: Path
    cloud_mask: Path

    def path_for(self, logical_product: str) -> Path:
        if logical_product not in CORE_PRODUCTS:
            raise KeyError(f"Unknown GOES logical product: {logical_product!r}")
        return Path(getattr(self, logical_product))


@dataclass
class GoesFrameData:
    """One standardised GOES frame ready for tracking/postprocessing."""

    satellite: str
    role: str
    scan_start: datetime
    scan_end: datetime
    scan_midpoint: datetime
    brightness_temperature_k: np.ndarray
    cloud_top_height_m: np.ndarray
    cloud_top_pressure_hpa: np.ndarray
    cloud_mask: np.ndarray
    image_dqf: np.ndarray | None
    height_dqf: np.ndarray | None
    pressure_dqf: np.ndarray | None
    cloud_mask_dqf: np.ndarray | None
    x_scan_angle: np.ndarray
    y_scan_angle: np.ndarray
    projection_attrs: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PrimaryField:
    """Decoded primary data variable and its metadata."""

    variable_name: str
    values: np.ndarray
    units_original: str
    units_standard: str
    quality_name: str | None
    quality: np.ndarray | None
    attrs: dict[str, Any]


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(
        tzinfo=None
    )


def _read_manifest(path: Path) -> tuple[dict[str, str], ...]:
    if not path.is_file():
        raise FileNotFoundError(f"GOES selection manifest was not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def discover_goes_frames(case_dir: str | Path) -> tuple[GoesFramePaths, ...]:
    """Discover the six satellite/time frames from the downloader manifest."""

    root = Path(case_dir).resolve()
    manifest = root / "manifests" / "goes_selected_products.csv"
    rows = _read_manifest(manifest)

    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        satellite = str(row.get("satellite", "")).upper()
        role = str(row.get("role", "")).lower()
        logical = str(row.get("logical_product", "")).lower()
        if not satellite or role not in ROLE_ORDER or logical not in CORE_PRODUCTS:
            continue
        grouped.setdefault((satellite, role), {})[logical] = row

    frames: list[GoesFramePaths] = []
    for (satellite, role), products in grouped.items():
        missing = [name for name in CORE_PRODUCTS if name not in products]
        if missing:
            raise RuntimeError(
                f"{satellite} {role} is missing manifest products: {missing}"
            )
        reference = products["image"]
        paths = {
            name: root / products[name]["local_path"] for name in CORE_PRODUCTS
        }
        absent = [str(path) for path in paths.values() if not path.is_file()]
        if absent:
            raise FileNotFoundError(
                f"{satellite} {role} has missing local GOES files:\n  "
                + "\n  ".join(absent)
            )
        frames.append(
            GoesFramePaths(
                satellite=satellite,
                role=role,
                scan_start=_parse_time(reference["scan_start"]),
                scan_end=_parse_time(reference["scan_end"]),
                scan_midpoint=_parse_time(reference["scan_midpoint"]),
                image=paths["image"],
                height=paths["height"],
                pressure=paths["pressure"],
                cloud_mask=paths["cloud_mask"],
            )
        )

    frames.sort(key=lambda item: (item.satellite, ROLE_ORDER[item.role]))
    satellites = sorted({item.satellite for item in frames})
    for satellite in satellites:
        roles = [item.role for item in frames if item.satellite == satellite]
        if roles != ["previous", "center", "next"]:
            raise RuntimeError(
                f"Expected previous/center/next for {satellite}; found {roles}"
            )
    if len(satellites) != 2 or len(frames) != 6:
        raise RuntimeError(
            f"Expected two satellites and six frames; found {satellites}, "
            f"frames={len(frames)}"
        )
    return tuple(frames)


def frame_for(
    frames: Sequence[GoesFramePaths], satellite: str, role: str
) -> GoesFramePaths:
    """Return one unique frame from a discovered case."""

    matches = [
        item
        for item in frames
        if item.satellite == satellite.upper() and item.role == role.lower()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one frame for {satellite} {role}; found {len(matches)}"
        )
    return matches[0]


def _require_xarray() -> Any:
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("GOES reading requires xarray") from exc
    return xr


def _find_variable(ds: Any, candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in ds.variables:
            return candidate
    lower = {str(name).lower(): str(name) for name in ds.variables}
    for candidate in candidates:
        match = lower.get(candidate.lower())
        if match is not None:
            return match
    return None


def _find_primary_variable(ds: Any, logical_product: str) -> str:
    explicit = _find_variable(ds, PRIMARY_VARIABLE_CANDIDATES[logical_product])
    if explicit is not None:
        return explicit

    excluded = {
        *QUALITY_VARIABLE_CANDIDATES,
        *PROJECTION_VARIABLE_CANDIDATES,
        "x",
        "y",
        "t",
        "time",
        "time_bounds",
        "x_image_bounds",
        "y_image_bounds",
    }
    candidates: list[tuple[int, str]] = []
    for name, da in ds.data_vars.items():
        if str(name) in excluded or da.ndim < 2:
            continue
        candidates.append((int(np.prod(da.shape)), str(name)))
    if not candidates:
        raise KeyError(
            f"Could not identify the primary {logical_product} variable. "
            f"Available data variables: {list(ds.data_vars)}"
        )
    candidates.sort(reverse=True)
    return candidates[0][1]


def _to_float_array(value: Any) -> np.ndarray:
    array = np.asanyarray(value)
    if np.ma.isMaskedArray(array):
        array = array.filled(np.nan)
    return np.asarray(array, dtype=np.float32)


def _normalise_units(
    logical_product: str, values: np.ndarray, units: str
) -> tuple[np.ndarray, str]:
    text = str(units or "").strip().lower().replace(" ", "")
    output = np.asarray(values, dtype=np.float32)
    if logical_product == "image":
        # CMIP infrared channels are normally decoded in kelvin.  Leave unknown
        # units untouched so the inventory exposes an unexpected product.
        return output, "K" if text in {"k", "kelvin", "degrees_k"} else units
    if logical_product == "height":
        if text in {"km", "kilometer", "kilometers", "kilometre", "kilometres"}:
            output = output * np.float32(1000.0)
        return output, "m"
    if logical_product == "pressure":
        if text in {"pa", "pascal", "pascals"}:
            output = output / np.float32(100.0)
        return output, "hPa"
    return output, units


def read_primary_field(
    path: str | Path,
    logical_product: str,
    *,
    apply_quality_mask: bool = False,
) -> PrimaryField:
    """Read and standardise one primary GOES product field.

    CF scale/offset and fill values are decoded by xarray.  When
    ``apply_quality_mask`` is true and a same-shape DQF exists, values with
    DQF != 0 are set to NaN.  The unmodified DQF is still returned.
    """

    if logical_product not in CORE_PRODUCTS:
        raise KeyError(logical_product)
    xr = _require_xarray()
    with xr.open_dataset(path, decode_cf=True, mask_and_scale=True) as ds:
        variable_name = _find_primary_variable(ds, logical_product)
        da = ds[variable_name]
        values = _to_float_array(da.values)
        units_original = str(da.attrs.get("units", ""))
        values, units_standard = _normalise_units(
            logical_product, values, units_original
        )
        quality_name = _find_variable(ds, QUALITY_VARIABLE_CANDIDATES)
        quality: np.ndarray | None = None
        if quality_name is not None:
            q = np.asanyarray(ds[quality_name].values)
            if np.ma.isMaskedArray(q):
                q = q.filled(-1)
            quality = np.asarray(q)
            if quality.shape != values.shape:
                quality = None
                quality_name = None
        if apply_quality_mask and quality is not None:
            values = values.copy()
            values[quality != 0] = np.nan
        attrs = {str(key): _jsonable(value) for key, value in da.attrs.items()}
    return PrimaryField(
        variable_name=variable_name,
        values=values,
        units_original=units_original,
        units_standard=str(units_standard or units_original),
        quality_name=quality_name,
        quality=quality,
        attrs=attrs,
    )


def _read_grid_and_projection(image_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    xr = _require_xarray()
    with xr.open_dataset(image_path, decode_cf=True, mask_and_scale=True) as ds:
        if "x" not in ds.variables or "y" not in ds.variables:
            raise KeyError(f"GOES fixed-grid x/y coordinates are absent: {image_path}")
        x = np.asarray(ds["x"].values, dtype=np.float64)
        y = np.asarray(ds["y"].values, dtype=np.float64)
        projection_name = _find_variable(ds, PROJECTION_VARIABLE_CANDIDATES)
        if projection_name is None:
            raise KeyError(f"GOES projection variable is absent: {image_path}")
        projection = {
            str(key): _jsonable(value)
            for key, value in ds[projection_name].attrs.items()
        }
        projection["variable_name"] = projection_name
    return x, y, projection


def load_goes_frame(
    frame: GoesFramePaths,
    *,
    apply_quality_mask: bool = False,
) -> GoesFrameData:
    """Load one satellite/time frame into standardised NumPy arrays."""

    image = read_primary_field(
        frame.image, "image", apply_quality_mask=apply_quality_mask
    )
    height = read_primary_field(
        frame.height, "height", apply_quality_mask=apply_quality_mask
    )
    pressure = read_primary_field(
        frame.pressure, "pressure", apply_quality_mask=apply_quality_mask
    )
    cloud_mask = read_primary_field(
        frame.cloud_mask, "cloud_mask", apply_quality_mask=False
    )

    shapes = {
        image.values.shape,
        height.values.shape,
        pressure.values.shape,
        cloud_mask.values.shape,
    }
    if len(shapes) != 1:
        raise ValueError(
            f"GOES products do not share one grid for {frame.satellite} "
            f"{frame.role}: {sorted(shapes)}"
        )
    x, y, projection = _read_grid_and_projection(frame.image)
    shape = image.values.shape
    if shape != (len(y), len(x)):
        raise ValueError(
            f"Image shape {shape} does not match y/x lengths {(len(y), len(x))}"
        )
    return GoesFrameData(
        satellite=frame.satellite,
        role=frame.role,
        scan_start=frame.scan_start,
        scan_end=frame.scan_end,
        scan_midpoint=frame.scan_midpoint,
        brightness_temperature_k=image.values,
        cloud_top_height_m=height.values,
        cloud_top_pressure_hpa=pressure.values,
        cloud_mask=cloud_mask.values,
        image_dqf=image.quality,
        height_dqf=height.quality,
        pressure_dqf=pressure.quality,
        cloud_mask_dqf=cloud_mask.quality,
        x_scan_angle=x,
        y_scan_angle=y,
        projection_attrs=projection,
        metadata={
            "variables": {
                "image": image.variable_name,
                "height": height.variable_name,
                "pressure": pressure.variable_name,
                "cloud_mask": cloud_mask.variable_name,
            },
            "units_original": {
                "image": image.units_original,
                "height": height.units_original,
                "pressure": pressure.units_original,
                "cloud_mask": cloud_mask.units_original,
            },
            "units_standard": {
                "image": image.units_standard,
                "height": height.units_standard,
                "pressure": pressure.units_standard,
                "cloud_mask": cloud_mask.units_standard,
            },
            "paths": {
                name: str(frame.path_for(name)) for name in CORE_PRODUCTS
            },
        },
    )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def dataset_inventory(path: str | Path) -> dict[str, Any]:
    """Return dimensions, fields, encodings, and global attributes for a NetCDF."""

    xr = _require_xarray()
    source = Path(path)
    with xr.open_dataset(source, decode_cf=False, mask_and_scale=False) as ds:
        fields: list[dict[str, Any]] = []
        coordinate_names = set(str(name) for name in ds.coords)
        for name, da in ds.variables.items():
            kind = "coordinate" if str(name) in coordinate_names else "data_variable"
            scalar_value: Any = None
            if da.ndim == 0:
                try:
                    scalar_value = _jsonable(da.values.item())
                except Exception:
                    scalar_value = None
            fields.append(
                {
                    "kind": kind,
                    "name": str(name),
                    "dims": list(da.dims),
                    "shape": list(da.shape),
                    "dtype": str(da.dtype),
                    "scalar_value": scalar_value,
                    "attrs": {
                        str(key): _jsonable(value)
                        for key, value in da.attrs.items()
                    },
                    "encoding": {
                        str(key): _jsonable(value)
                        for key, value in da.encoding.items()
                        if key in {"dtype", "_FillValue", "scale_factor", "add_offset"}
                    },
                }
            )
        return {
            "path": str(source),
            "file_size_bytes": source.stat().st_size if source.is_file() else None,
            "dimensions": {str(key): int(value) for key, value in ds.sizes.items()},
            "global_attrs": {
                str(key): _jsonable(value) for key, value in ds.attrs.items()
            },
            "fields": fields,
        }


def print_dataset_inventory(
    path: str | Path,
    *,
    title: str | None = None,
    include_global_attrs: bool = True,
) -> None:
    """Print a readable field inventory without loading large 2-D arrays."""

    inventory = dataset_inventory(path)
    print("\n" + "-" * 118)
    print(title or Path(path).name)
    print("-" * 118)
    print("file:", inventory["path"])
    print("dimensions:", inventory["dimensions"])
    if include_global_attrs:
        print("global attributes:")
        for key, value in inventory["global_attrs"].items():
            text = repr(value)
            if len(text) > 240:
                text = text[:237] + "..."
            print(f"  {key:38s}: {text}")
    print("fields:")
    for field in inventory["fields"]:
        attrs = field["attrs"]
        units = attrs.get("units", "")
        long_name = str(attrs.get("long_name", attrs.get("standard_name", "")))
        if len(long_name) > 72:
            long_name = long_name[:69] + "..."
        print(
            f"  {field['name']:34s} {field['kind']:13s} "
            f"dims={tuple(field['dims'])!s:24s} shape={tuple(field['shape'])!s:18s} "
            f"dtype={field['dtype']:10s} units={str(units)!r} long_name={long_name!r}"
        )
        useful = {
            key: attrs[key]
            for key in (
                "_FillValue",
                "missing_value",
                "scale_factor",
                "add_offset",
                "valid_range",
                "flag_values",
                "flag_meanings",
                "grid_mapping",
            )
            if key in attrs
        }
        if field["scalar_value"] is not None:
            useful["scalar_value"] = field["scalar_value"]
        if useful:
            print("    metadata:", useful)


def _finite_statistics(values: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float32).ravel()
    valid = np.isfinite(flat)
    valid_count = int(valid.sum())
    total_count = int(flat.size)
    if valid_count == 0:
        return {
            "total_count": total_count,
            "valid_count": 0,
            "valid_fraction": 0.0,
            "minimum": None,
            "p01": None,
            "p05": None,
            "median": None,
            "mean": None,
            "std": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }
    sample = flat[valid].astype(np.float64, copy=False)
    quantiles = np.nanpercentile(sample, [1, 5, 50, 95, 99])
    return {
        "total_count": total_count,
        "valid_count": valid_count,
        "valid_fraction": valid_count / max(total_count, 1),
        "minimum": float(np.nanmin(sample)),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "median": float(quantiles[2]),
        "mean": float(np.nanmean(sample)),
        "std": float(np.nanstd(sample)),
        "p95": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "maximum": float(np.nanmax(sample)),
    }


def summarise_primary_file(
    frame: GoesFramePaths,
    logical_product: str,
    *,
    apply_quality_mask: bool = False,
) -> dict[str, Any]:
    """Compute decoded field statistics for one core GOES file."""

    path = frame.path_for(logical_product)
    field = read_primary_field(
        path,
        logical_product,
        apply_quality_mask=apply_quality_mask,
    )
    stats = _finite_statistics(field.values)
    quality_valid_fraction: float | None = None
    quality_counts: dict[str, int] | None = None
    if field.quality is not None:
        q_values, q_counts = np.unique(field.quality, return_counts=True)
        quality_counts = {
            str(_jsonable(value)): int(count)
            for value, count in zip(q_values, q_counts)
        }
        quality_valid_fraction = float(np.mean(field.quality == 0))

    categorical_counts: dict[str, int] | None = None
    if logical_product == "cloud_mask":
        finite = field.values[np.isfinite(field.values)]
        if finite.size:
            values, counts = np.unique(finite, return_counts=True)
            categorical_counts = {
                str(float(value)): int(count)
                for value, count in zip(values, counts)
            }

    return {
        "satellite": frame.satellite,
        "role": frame.role,
        "scan_start": frame.scan_start.isoformat(),
        "scan_end": frame.scan_end.isoformat(),
        "scan_midpoint": frame.scan_midpoint.isoformat(),
        "logical_product": logical_product,
        "path": str(path),
        "variable_name": field.variable_name,
        "shape": list(field.values.shape),
        "dtype_decoded": str(field.values.dtype),
        "units_original": field.units_original,
        "units_standard": field.units_standard,
        "quality_variable": field.quality_name,
        "quality_good_fraction": quality_valid_fraction,
        "quality_counts": quality_counts,
        "categorical_counts": categorical_counts,
        **stats,
    }


def validate_core_grid_alignment(
    frames: Sequence[GoesFramePaths],
    *,
    coordinate_tolerance: float = 1e-12,
) -> list[dict[str, Any]]:
    """Check shape, x/y coordinates, and projection agreement within each frame."""

    xr = _require_xarray()
    rows: list[dict[str, Any]] = []
    for frame in frames:
        reference_x: np.ndarray | None = None
        reference_y: np.ndarray | None = None
        reference_projection: dict[str, Any] | None = None
        reference_shape: tuple[int, ...] | None = None
        for logical_product in CORE_PRODUCTS:
            path = frame.path_for(logical_product)
            with xr.open_dataset(path, decode_cf=True, mask_and_scale=True) as ds:
                variable = _find_primary_variable(ds, logical_product)
                shape = tuple(int(value) for value in ds[variable].shape)
                x = np.asarray(ds["x"].values, dtype=np.float64)
                y = np.asarray(ds["y"].values, dtype=np.float64)
                projection_name = _find_variable(ds, PROJECTION_VARIABLE_CANDIDATES)
                projection = (
                    {
                        str(key): _jsonable(value)
                        for key, value in ds[projection_name].attrs.items()
                    }
                    if projection_name is not None
                    else {}
                )
            if reference_x is None:
                reference_x = x
                reference_y = y
                reference_projection = projection
                reference_shape = shape
            max_x_difference = (
                float(np.max(np.abs(x - reference_x)))
                if x.shape == reference_x.shape
                else math.inf
            )
            max_y_difference = (
                float(np.max(np.abs(y - reference_y)))
                if y.shape == reference_y.shape
                else math.inf
            )
            row = {
                "satellite": frame.satellite,
                "role": frame.role,
                "logical_product": logical_product,
                "path": str(path),
                "primary_variable": variable,
                "shape": list(shape),
                "reference_shape": list(reference_shape or ()),
                "shape_matches": shape == reference_shape,
                "x_shape_matches": x.shape == reference_x.shape,
                "y_shape_matches": y.shape == reference_y.shape,
                "max_abs_x_difference": max_x_difference,
                "max_abs_y_difference": max_y_difference,
                "coordinates_match": (
                    max_x_difference <= coordinate_tolerance
                    and max_y_difference <= coordinate_tolerance
                ),
                "projection_matches": projection == reference_projection,
            }
            row["all_aligned"] = bool(
                row["shape_matches"]
                and row["coordinates_match"]
                and row["projection_matches"]
            )
            rows.append(row)
    return rows


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(_jsonable(value), ensure_ascii=True, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )
    return path


def write_json(document: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_jsonable(document), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return target


def write_field_inventories(
    labeled_files: Sequence[tuple[str, Path]],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write complete per-file inventory JSON and flattened field CSV."""

    output = Path(output_dir)
    inventories: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for label, path in labeled_files:
        inventory = dataset_inventory(path)
        inventory["label"] = label
        inventories.append(inventory)
        for field in inventory["fields"]:
            attrs = field["attrs"]
            flat_rows.append(
                {
                    "label": label,
                    "path": inventory["path"],
                    "kind": field["kind"],
                    "name": field["name"],
                    "dims": field["dims"],
                    "shape": field["shape"],
                    "dtype": field["dtype"],
                    "units": attrs.get("units", ""),
                    "long_name": attrs.get("long_name", ""),
                    "standard_name": attrs.get("standard_name", ""),
                    "fill_value": attrs.get("_FillValue", field["encoding"].get("_FillValue")),
                    "missing_value": attrs.get("missing_value"),
                    "scale_factor": attrs.get("scale_factor", field["encoding"].get("scale_factor")),
                    "add_offset": attrs.get("add_offset", field["encoding"].get("add_offset")),
                    "valid_range": attrs.get("valid_range"),
                    "flag_values": attrs.get("flag_values"),
                    "flag_meanings": attrs.get("flag_meanings"),
                    "grid_mapping": attrs.get("grid_mapping"),
                    "scalar_value": field["scalar_value"],
                }
            )
    json_path = write_json(inventories, output / "netcdf_field_inventory.json")
    csv_path = _write_csv(flat_rows, output / "netcdf_field_inventory.csv")
    return csv_path, json_path


def representative_files(
    case_dir: str | Path,
    frames: Sequence[GoesFramePaths],
    *,
    include_dmw: bool = True,
    include_era5: bool = True,
) -> tuple[tuple[str, Path], ...]:
    """Choose one representative file for each satellite/product/reference type."""

    root = Path(case_dir)
    output: list[tuple[str, Path]] = []
    for satellite in sorted({frame.satellite for frame in frames}):
        center = frame_for(frames, satellite, "center")
        for logical in CORE_PRODUCTS:
            output.append((f"{satellite} center {logical}", center.path_for(logical)))
    if include_dmw:
        for path in sorted((root / "official_amv").glob("**/*.nc")):
            output.append((f"official DMW {path.parent.parent.name}", path))
    if include_era5:
        era5_root = root / "era5"
        if era5_root.is_dir():
            for region in sorted(path for path in era5_root.iterdir() if path.is_dir()):
                files = sorted(region.glob("*.nc"))
                if files:
                    output.append((f"ERA5 {region.name}", files[0]))
    return tuple(output)


def _plot_limits(logical_product: str, values: np.ndarray) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0 or logical_product == "cloud_mask":
        return None, None
    lower, upper = np.nanpercentile(finite, [1.0, 99.0])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        return None, None
    return float(lower), float(upper)


def create_field_quicklook(
    frame: GoesFramePaths,
    logical_product: str,
    output_path: str | Path,
    *,
    max_plot_dimension: int = 1800,
    apply_quality_mask: bool = False,
) -> Path:
    """Create one fixed-grid quicklook for a centre GOES product."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Quicklook creation requires matplotlib") from exc

    field = read_primary_field(
        frame.path_for(logical_product),
        logical_product,
        apply_quality_mask=apply_quality_mask,
    )
    x, y, _ = _read_grid_and_projection(frame.image)
    values = field.values
    stride = max(1, int(math.ceil(max(values.shape) / max_plot_dimension)))
    display = values[::stride, ::stride]
    display_x = x[::stride]
    display_y = y[::stride]
    extent = [
        float(display_x.min()),
        float(display_x.max()),
        float(display_y.min()),
        float(display_y.max()),
    ]
    vmin, vmax = _plot_limits(logical_product, display)
    colormap = {
        "image": "gray_r",
        "height": "viridis",
        "pressure": "plasma_r",
        "cloud_mask": "tab10",
    }[logical_product]
    labels = {
        "image": "Band-14 brightness temperature",
        "height": "Cloud top height",
        "pressure": "Cloud top pressure",
        "cloud_mask": "Clear sky/cloud mask",
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    image = ax.imshow(
        display,
        origin="upper",
        extent=extent,
        interpolation="nearest",
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(
        f"{frame.satellite} {frame.role} — {labels[logical_product]}\n"
        f"scan midpoint {frame.scan_midpoint.isoformat()} UTC"
    )
    ax.set_xlabel("ABI fixed-grid x scan angle [rad]")
    ax.set_ylabel("ABI fixed-grid y scan angle [rad]")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    units = field.units_standard or field.units_original
    colorbar.set_label(units if units else "category")
    ax.text(
        0.01,
        0.01,
        f"variable={field.variable_name}; display stride={stride}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def inspect_goes_case(
    case_dir: str | Path,
    output_dir: str | Path,
    *,
    print_fields: bool = True,
    include_dmw: bool = True,
    include_era5: bool = True,
    create_figures: bool = True,
    max_plot_dimension: int = 1800,
    apply_quality_mask_for_statistics: bool = False,
) -> dict[str, Any]:
    """Run the complete first-stage GOES data inspection."""

    root = Path(case_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames = discover_goes_frames(root)

    alignment_rows = validate_core_grid_alignment(frames)
    alignment_csv = _write_csv(
        alignment_rows, output / "goes_core_grid_alignment.csv"
    )
    alignment_json = write_json(
        alignment_rows, output / "goes_core_grid_alignment.json"
    )
    if not all(bool(row["all_aligned"]) for row in alignment_rows):
        failed = [
            f"{row['satellite']} {row['role']} {row['logical_product']}"
            for row in alignment_rows
            if not row["all_aligned"]
        ]
        raise RuntimeError("GOES core grid alignment failed for: " + ", ".join(failed))

    summary_rows: list[dict[str, Any]] = []
    for frame in frames:
        for logical_product in CORE_PRODUCTS:
            summary_rows.append(
                summarise_primary_file(
                    frame,
                    logical_product,
                    apply_quality_mask=apply_quality_mask_for_statistics,
                )
            )
    summary_csv = _write_csv(summary_rows, output / "goes_primary_field_summary.csv")
    summary_json = write_json(summary_rows, output / "goes_primary_field_summary.json")

    labeled_files = representative_files(
        root,
        frames,
        include_dmw=include_dmw,
        include_era5=include_era5,
    )
    inventory_csv, inventory_json = write_field_inventories(labeled_files, output)
    if print_fields:
        for label, path in labeled_files:
            print_dataset_inventory(path, title=label)

    figures: list[Path] = []
    if create_figures:
        figure_dir = output / "figures"
        for satellite in sorted({frame.satellite for frame in frames}):
            center = frame_for(frames, satellite, "center")
            for logical_product in CORE_PRODUCTS:
                figures.append(
                    create_field_quicklook(
                        center,
                        logical_product,
                        figure_dir
                        / f"{satellite}_center_{logical_product}.png",
                        max_plot_dimension=max_plot_dimension,
                    )
                )

    report = {
        "case_directory": str(root),
        "output_directory": str(output),
        "satellites": sorted({frame.satellite for frame in frames}),
        "frames": [
            {
                "satellite": frame.satellite,
                "role": frame.role,
                "scan_start": frame.scan_start.isoformat(),
                "scan_end": frame.scan_end.isoformat(),
                "scan_midpoint": frame.scan_midpoint.isoformat(),
            }
            for frame in frames
        ],
        "core_file_count": len(frames) * len(CORE_PRODUCTS),
        "representative_inventory_file_count": len(labeled_files),
        "alignment_ok": True,
        "outputs": {
            "alignment_csv": str(alignment_csv),
            "alignment_json": str(alignment_json),
            "field_summary_csv": str(summary_csv),
            "field_summary_json": str(summary_json),
            "field_inventory_csv": str(inventory_csv),
            "field_inventory_json": str(inventory_json),
            "figures": [str(path) for path in figures],
        },
    }
    report_path = write_json(report, output / "inspection_report.json")
    report["outputs"]["report"] = str(report_path)
    return report
