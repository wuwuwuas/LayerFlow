"""OCA cloud-property loading.

This module supports two workflows:

1. Explicit-file workflow, recommended for notebooks without YAML configs:
   ``load_oca_component_files([C1_file, C2_file, C3_file])``.

2. Config-driven workflow, kept for backward compatibility:
   ``load_oca(case_config, seviri_data=...)``.

OCA longitude/latitude is assumed to be on the SEVIRI grid.  The direct
component loader therefore returns cloud-property fields and does not try to
load SEVIRI geolocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
import re

import numpy as np

from config import REPO_ROOT, repo_path, resolve_case_config
from .seviri import SeviriImages, load_seviri_images


@dataclass(frozen=True)
class OcaData:
    """OCA cloud properties loaded for AMV pressure assignment and TTO.

    ``ctp_hpa`` / ``cloud_probability`` / ``times`` / ``paths`` are the
    component-start fields used by the AMV workflow.  For C1/C2/C3, these are
    normally the OCA products at the start frame of each motion component.

    ``scene_classification`` follows the OCA Release 1 definition:
    0 clear, 1 single-layer liquid water, 2 single-layer ice water,
    3 two-layer, 10 failed retrieval.  It is optional so older NPZ products
    remain readable.
    """

    ctp_hpa: tuple[np.ndarray, ...]
    cloud_probability: tuple[np.ndarray | None, ...]
    times: tuple[datetime | None, ...]
    paths: tuple[Path, ...]

    scene_classification: tuple[np.ndarray | None, ...] = ()
    ctp_error_log: tuple[np.ndarray | None, ...] = ()
    measurement_cost_log: tuple[np.ndarray | None, ...] = ()
    ctp_lower_hpa: tuple[np.ndarray | None, ...] = ()
    ctp_error_log_lower: tuple[np.ndarray | None, ...] = ()
    cot_log: tuple[np.ndarray | None, ...] = ()
    cot_log_lower: tuple[np.ndarray | None, ...] = ()

    frame_ctp_hpa: tuple[np.ndarray, ...] = ()
    frame_cloud_probability: tuple[np.ndarray | None, ...] = ()
    frame_scene_classification: tuple[np.ndarray | None, ...] = ()
    frame_ctp_error_log: tuple[np.ndarray | None, ...] = ()
    frame_measurement_cost_log: tuple[np.ndarray | None, ...] = ()
    frame_ctp_lower_hpa: tuple[np.ndarray | None, ...] = ()
    frame_ctp_error_log_lower: tuple[np.ndarray | None, ...] = ()
    frame_cot_log: tuple[np.ndarray | None, ...] = ()
    frame_cot_log_lower: tuple[np.ndarray | None, ...] = ()
    frame_times: tuple[datetime | None, ...] = ()
    frame_paths: tuple[Path, ...] = ()
    component_source: str = "start_frame"


def _as_config(case_config) -> dict[str, Any]:
    if isinstance(case_config, Mapping):
        return dict(case_config)
    return resolve_case_config(case_config)


def _resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return repo_path(value)


def _parse_time(value) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)


def _time_from_text(text: str) -> datetime | None:
    matches = re.findall(r"(?:19|20)\d{12}(?:\.\d+)?", text)
    if not matches:
        return None
    token = matches[0]
    main, _, frac = token.partition(".")
    try:
        value = datetime.strptime(main, "%Y%m%d%H%M%S")
    except ValueError:
        return None
    if frac:
        value = value.replace(microsecond=int((frac + "000000")[:6]))
    return value


def _time_from_path(path: Path) -> datetime | None:
    return _time_from_text(str(path))


def _clean_pressure_hpa(pressure: np.ndarray | None) -> np.ndarray | None:
    if pressure is None:
        return None
    out = np.asarray(pressure, dtype=np.float32).copy()
    finite = np.isfinite(out)
    if finite.any() and np.nanmedian(out[finite]) > 2000:
        out = out / 100.0  # likely Pa -> hPa
    out[(out < 50) | (out > 1100)] = np.nan
    return out.astype(np.float32)


def _maybe_array(value) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.squeeze(np.asarray(value))
    if arr.ndim == 0:
        return None
    return arr


def _first_npz_key(data, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in data.files), None)


def _read_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        time_key = _first_npz_key(data, ("name_time", "time", "start_time", "end_time"))
        if time_key is not None:
            text = str(np.asarray(data[time_key]).item())
            timestamp = _time_from_text(text) or _parse_time(text)
        else:
            timestamp = _time_from_path(path)

        pressure_key = _first_npz_key(
            data,
            (
                "ctp_hpa", "ctp", "cloud_top_pressure_hpa", "cloud_top_pressure",
                "cth_pressure", "ctth_pressure", "pressure_hpa", "pressure",
            ),
        )
        if pressure_key is None:
            raise KeyError(f"No OCA cloud-top pressure key found in {path}. Keys: {list(data.files)}")
        pressure = _clean_pressure_hpa(np.squeeze(data[pressure_key]))

        lower_key = _first_npz_key(data, ("cloud_top_pressure_lower_layer_hpa", "cloud_top_pressure_lower_layer", "ctp_lower_hpa", "ctp2_hpa"))
        lower = _clean_pressure_hpa(np.squeeze(data[lower_key])) if lower_key is not None else None

        cloud_key = _first_npz_key(data, ("cloud_probability", "cloud_prob", "probability_of_cloud", "cma"))
        scene_key = _first_npz_key(data, ("scene_classification", "cloud_phase", "phase"))
        ctp_error_key = _first_npz_key(data, ("cloud_top_pressure_error_log", "ctp_error_log", "err_ctp_log"))
        cost_key = _first_npz_key(data, ("measurement_cost_function_log", "measurement_cost_log", "cost_log"))
        lower_err_key = _first_npz_key(data, ("cloud_top_pressure_error_log_lower_layer", "ctp_error_log_lower", "err_ctp2_log"))
        cot_key = _first_npz_key(data, ("cloud_optical_depth_log", "cot_log"))
        cot_lower_key = _first_npz_key(data, ("cloud_optical_depth_log_lower_layer", "cot_log_lower"))

        return {
            "timestamp": timestamp,
            "ctp_hpa": pressure,
            "cloud_probability": _maybe_array(data[cloud_key]) if cloud_key is not None else None,
            "scene_classification": _maybe_array(data[scene_key]) if scene_key is not None else None,
            "ctp_error_log": _maybe_array(data[ctp_error_key]) if ctp_error_key is not None else None,
            "measurement_cost_log": _maybe_array(data[cost_key]) if cost_key is not None else None,
            "ctp_lower_hpa": lower,
            "ctp_error_log_lower": _maybe_array(data[lower_err_key]) if lower_err_key is not None else None,
            "cot_log": _maybe_array(data[cot_key]) if cot_key is not None else None,
            "cot_log_lower": _maybe_array(data[cot_lower_key]) if cot_lower_key is not None else None,
        }


def _read_netcdf(path: Path) -> dict[str, Any]:
    try:
        import xarray as xr
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Reading native OCA NetCDF requires xarray") from exc

    with xr.open_dataset(path, mask_and_scale=True, decode_times=False) as ds:
        timestamp = None
        for key in ("time", "start_time", "end_time", "nominal_time"):
            if key in ds.coords:
                values = np.asarray(ds[key].values).ravel()
                if values.size:
                    timestamp = _parse_time(str(values[0]))
                    break
            if key in ds.variables:
                values = np.asarray(ds[key].values).ravel()
                if values.size:
                    timestamp = _parse_time(str(values[0]))
                    break
            if key in ds.attrs:
                timestamp = _time_from_text(str(ds.attrs[key])) or _parse_time(ds.attrs[key])
                break
        if timestamp is None:
            timestamp = _time_from_path(path)

        def var(*names: str):
            key = next((name for name in names if name in ds.variables), None)
            return _maybe_array(ds[key].values) if key is not None else None

        pressure = var("ctp_hpa", "ctp", "cloud_top_pressure_hpa", "cloud_top_pressure", "cth_pressure", "ctth_pressure", "pressure_hpa", "pressure", "CTP")
        if pressure is None:
            pressure_name = next((name for name in ds.variables if "pressure" in name.lower() or "ctp" in name.lower()), None)
            pressure = _maybe_array(ds[pressure_name].values) if pressure_name is not None else None
        if pressure is None:
            raise KeyError(f"No OCA cloud-top pressure variable found in {path}. Variables: {list(ds.variables)}")

        return {
            "timestamp": timestamp,
            "ctp_hpa": _clean_pressure_hpa(pressure),
            "cloud_probability": var("cloud_probability", "cloud_prob", "probability_of_cloud", "cma"),
            "scene_classification": var("scene_classification", "cloud_phase", "phase"),
            "ctp_error_log": var("cloud_top_pressure_error_log", "ctp_error_log", "err_ctp_log"),
            "measurement_cost_log": var("measurement_cost_function_log", "measurement_cost_log", "cost_log"),
            "ctp_lower_hpa": _clean_pressure_hpa(var("cloud_top_pressure_lower_layer", "cloud_top_pressure_lower_layer_hpa", "ctp_lower_hpa", "ctp2_hpa")),
            "ctp_error_log_lower": var("cloud_top_pressure_error_log_lower_layer", "ctp_error_log_lower", "err_ctp2_log"),
            "cot_log": var("cloud_optical_depth_log", "cot_log"),
            "cot_log_lower": var("cloud_optical_depth_log_lower_layer", "cot_log_lower"),
        }


def _read_fields(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".npz":
        return _read_npz(path)
    return _read_netcdf(path)


def _read(path: Path) -> tuple[datetime | None, np.ndarray, np.ndarray | None]:
    """Backward-compatible reader returning only timestamp, CTP and cloud probability."""

    fields = _read_fields(path)
    return fields["timestamp"], fields["ctp_hpa"], fields["cloud_probability"]


def _field_tuple(records: list[dict[str, Any]], key: str) -> tuple[Any, ...]:
    return tuple(record.get(key) for record in records)


def load_oca_component_files(
    component_files: Iterable[str | Path],
    *,
    component_source: str = "explicit_component_files",
    require_three: bool = True,
) -> OcaData:
    """Load explicit OCA component-start files.

    The input order is preserved.  For a C1/C2/C3 PIV product, pass files as:
    ``[OCA_for_C1_start, OCA_for_C2_start, OCA_for_C3_start]``.
    """

    paths = tuple(_resolve_path(path) for path in component_files)
    if require_three and len(paths) < 3:
        raise ValueError("At least three OCA files are required for C1/C2/C3 assignment")
    if not paths:
        raise ValueError("No OCA component files were provided")

    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"OCA file not found: {path}")
        records.append(_read_fields(path))

    return OcaData(
        ctp_hpa=_field_tuple(records, "ctp_hpa"),
        cloud_probability=_field_tuple(records, "cloud_probability"),
        times=_field_tuple(records, "timestamp"),
        paths=paths,
        scene_classification=_field_tuple(records, "scene_classification"),
        ctp_error_log=_field_tuple(records, "ctp_error_log"),
        measurement_cost_log=_field_tuple(records, "measurement_cost_log"),
        ctp_lower_hpa=_field_tuple(records, "ctp_lower_hpa"),
        ctp_error_log_lower=_field_tuple(records, "ctp_error_log_lower"),
        cot_log=_field_tuple(records, "cot_log"),
        cot_log_lower=_field_tuple(records, "cot_log_lower"),
        frame_ctp_hpa=_field_tuple(records, "ctp_hpa"),
        frame_cloud_probability=_field_tuple(records, "cloud_probability"),
        frame_scene_classification=_field_tuple(records, "scene_classification"),
        frame_ctp_error_log=_field_tuple(records, "ctp_error_log"),
        frame_measurement_cost_log=_field_tuple(records, "measurement_cost_log"),
        frame_ctp_lower_hpa=_field_tuple(records, "ctp_lower_hpa"),
        frame_ctp_error_log_lower=_field_tuple(records, "ctp_error_log_lower"),
        frame_cot_log=_field_tuple(records, "cot_log"),
        frame_cot_log_lower=_field_tuple(records, "cot_log_lower"),
        frame_times=_field_tuple(records, "timestamp"),
        frame_paths=paths,
        component_source=component_source,
    )


def find_oca_files(root: str | Path, pattern: str = "*.npz") -> list[Path]:
    """Return sorted OCA files under a root directory for explicit workflows."""

    base = _resolve_path(root)
    if base.is_file():
        return [base]
    if not base.exists():
        raise FileNotFoundError(f"OCA root does not exist: {base}")
    files = sorted(path for path in base.rglob(pattern) if path.is_file())
    files = [path for path in files if not any(part in {"index", "metadata", "requests"} for part in path.parts)]
    return files


# -----------------------------------------------------------------------------
# Backward-compatible config-driven loader
# -----------------------------------------------------------------------------


def _candidate_files(config: dict) -> list[Path]:
    paths_cfg = config.get("paths", {}) or {}
    candidates: list[Path] = []
    for pattern in (config.get("oca_glob"), paths_cfg.get("oca_glob"), paths_cfg.get("oca_npz_glob")):
        if pattern:
            candidates.extend(path for path in REPO_ROOT.glob(str(pattern)) if path.is_file())
    oca_root = paths_cfg.get("oca")
    if oca_root is not None:
        root = _resolve_path(oca_root)
        candidates.extend(path for path in root.rglob("*.npz") if path.is_file())
        candidates.extend(path for path in root.rglob("*.nc") if path.is_file())
    output = []
    seen = set()
    for path in sorted(candidates):
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if any(part in {"index", "metadata", "requests"} for part in path.parts):
            continue
        output.append(path)
    return output


def _select_record(records, target_time: datetime, tolerance_minutes: float):
    record = min(records, key=lambda item: abs((item[1]["timestamp"] - target_time).total_seconds()))
    path, fields = record
    timestamp = fields["timestamp"]
    diff_min = abs((timestamp - target_time).total_seconds()) / 60.0
    if diff_min > tolerance_minutes:
        raise FileNotFoundError(
            f"No OCA file within {tolerance_minutes} minutes of {target_time}. Closest: {path} at {timestamp}, dt={diff_min:.2f} min"
        )
    return path, fields


def _apply_seviri_crop(array: np.ndarray | None, seviri: SeviriImages):
    if array is None:
        return None
    if seviri.crop is not None and array.shape != seviri.images[0].shape:
        return array[seviri.crop].copy()
    return array.copy()


def _crop_fields(fields: dict[str, Any], seviri: SeviriImages) -> dict[str, Any]:
    out = dict(fields)
    for key in (
        "ctp_hpa", "cloud_probability", "scene_classification", "ctp_error_log",
        "measurement_cost_log", "ctp_lower_hpa", "ctp_error_log_lower", "cot_log", "cot_log_lower",
    ):
        out[key] = _apply_seviri_crop(out.get(key), seviri)
    return out


def load_oca(
    case_config,
    aoi=None,
    *,
    seviri_data: SeviriImages | None = None,
    time_tolerance_minutes: float = 30.0,
) -> OcaData:
    """Load OCA cloud properties for SEVIRI frames and components.

    This config-driven function is kept for compatibility.  New notebooks can
    avoid YAML by using ``load_oca_component_files`` instead.
    """

    config = _as_config(case_config)
    seviri = seviri_data or load_seviri_images(config, aoi=aoi)
    candidates = _candidate_files(config)
    if not candidates:
        raise FileNotFoundError("No OCA NPZ/NetCDF files found. Check paths['oca'] or oca_glob.")

    records = []
    errors = []
    for path in candidates:
        try:
            fields = _read_fields(path)
            if fields["timestamp"] is None:
                raise ValueError("missing timestamp")
            records.append((path, fields))
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    if not records:
        raise RuntimeError("No readable OCA files found:\n" + "\n".join(errors[:20]))

    frame_records: list[dict[str, Any]] = []
    frame_times: list[datetime | None] = []
    frame_paths: list[Path] = []
    for target_time in seviri.frame_times:
        path, fields = _select_record(records, target_time, time_tolerance_minutes)
        fields = _crop_fields(fields, seviri)
        frame_records.append(fields)
        frame_times.append(fields["timestamp"])
        frame_paths.append(path)

    n_components = max(1, len(seviri.frame_times) - 1)
    comp_records = frame_records[:n_components]

    return OcaData(
        ctp_hpa=_field_tuple(comp_records, "ctp_hpa"),
        cloud_probability=_field_tuple(comp_records, "cloud_probability"),
        times=tuple(frame_times[:n_components]),
        paths=tuple(frame_paths[:n_components]),
        scene_classification=_field_tuple(comp_records, "scene_classification"),
        ctp_error_log=_field_tuple(comp_records, "ctp_error_log"),
        measurement_cost_log=_field_tuple(comp_records, "measurement_cost_log"),
        ctp_lower_hpa=_field_tuple(comp_records, "ctp_lower_hpa"),
        ctp_error_log_lower=_field_tuple(comp_records, "ctp_error_log_lower"),
        cot_log=_field_tuple(comp_records, "cot_log"),
        cot_log_lower=_field_tuple(comp_records, "cot_log_lower"),
        frame_ctp_hpa=_field_tuple(frame_records, "ctp_hpa"),
        frame_cloud_probability=_field_tuple(frame_records, "cloud_probability"),
        frame_scene_classification=_field_tuple(frame_records, "scene_classification"),
        frame_ctp_error_log=_field_tuple(frame_records, "ctp_error_log"),
        frame_measurement_cost_log=_field_tuple(frame_records, "measurement_cost_log"),
        frame_ctp_lower_hpa=_field_tuple(frame_records, "ctp_lower_hpa"),
        frame_ctp_error_log_lower=_field_tuple(frame_records, "ctp_error_log_lower"),
        frame_cot_log=_field_tuple(frame_records, "cot_log"),
        frame_cot_log_lower=_field_tuple(frame_records, "cot_log_lower"),
        frame_times=tuple(frame_times),
        frame_paths=tuple(frame_paths),
        component_source="start_frame",
    )
