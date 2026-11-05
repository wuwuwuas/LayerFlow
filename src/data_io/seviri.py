"""SEVIRI IR10.8 image and geolocation loading.

This module can load either converted NPZ frames or downloaded native SEVIRI
files.  Native loading uses Satpy.  The geolocation cache is stored beside the
raw SEVIRI download directory, usually::

    <paths['seviri']>/lon_lat/seviri_lonlat_cache.npz
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Mapping
import csv
import re

import numpy as np

from config import REPO_ROOT, repo_path, resolve_case_config
from .common import geographic_crop


@dataclass(frozen=True)
class SeviriFrame:
    """One SEVIRI frame."""

    path: Path
    image: np.ndarray
    start_time: datetime
    end_time: datetime
    units: str = "K"
    x0: int = 0
    y0: int = 0


@dataclass(frozen=True)
class SeviriGeolocation:
    """SEVIRI longitude/latitude grid, optionally cropped by AOI."""

    longitude: np.ndarray
    latitude: np.ndarray
    path: Path
    crop: tuple[slice, slice] | None = None


@dataclass(frozen=True)
class SeviriImages:
    """Aligned case frames and their common geolocation."""

    images: tuple[np.ndarray, ...]
    valid_masks: tuple[np.ndarray, ...]
    frame_times: tuple[datetime, ...]
    paths: tuple[Path, ...]
    longitude: np.ndarray
    latitude: np.ndarray
    crop: tuple[slice, slice] | None


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _as_config(case_config) -> dict:
    """Return a config dictionary while accepting notebook-created dict configs."""

    if isinstance(case_config, Mapping):
        return dict(case_config)
    return resolve_case_config(case_config)


def _resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root unless it is absolute."""

    value = Path(path)
    if value.is_absolute():
        return value
    return repo_path(value)


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _scalar(data: np.lib.npyio.NpzFile, name: str, default=None):
    return np.asarray(data[name]).item() if name in data.files else default


def _parse_datetime(value) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)


# -----------------------------------------------------------------------------
# Lon/lat cache helpers
# -----------------------------------------------------------------------------


def seviri_lonlat_cache_path(seviri_path: str | Path) -> Path:
    """Return the standard lon/lat cache path for a SEVIRI download root.

    If ``seviri_path`` is already an NPZ file, it is returned unchanged.
    Otherwise the returned path is::

        <seviri_path>/lon_lat/seviri_lonlat_cache.npz
    """

    path = _resolve_path(seviri_path)
    if path.suffix.lower() == ".npz":
        return path
    return path / "lon_lat" / "seviri_lonlat_cache.npz"


def _read_lonlat_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read lon/lat arrays from an NPZ using common key names."""

    source = _resolve_path(path)
    with np.load(source, allow_pickle=False) as data:
        lon_key = next(
            (name for name in ("lons", "lon", "longitude", "longitudes") if name in data.files),
            None,
        )
        lat_key = next(
            (name for name in ("lats", "lat", "latitude", "latitudes") if name in data.files),
            None,
        )
        if lon_key is None or lat_key is None:
            raise KeyError(
                f"No lon/lat arrays found in {source}. Available keys: {list(data.files)}"
            )
        lon = np.asarray(data[lon_key], dtype=np.float32)
        lat = np.asarray(data[lat_key], dtype=np.float32)
    if lon.shape != lat.shape:
        raise ValueError(f"lon/lat shapes differ in {source}: {lon.shape} vs {lat.shape}")
    return lon, lat


def save_seviri_lonlat(
    longitude: np.ndarray,
    latitude: np.ndarray,
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """Save SEVIRI lon/lat arrays to an NPZ file."""

    output = _resolve_path(output_path)
    if output.exists() and not overwrite:
        return output

    lon = np.asarray(longitude, dtype=np.float32)
    lat = np.asarray(latitude, dtype=np.float32)
    if lon.shape != lat.shape:
        raise ValueError(f"longitude/latitude shapes differ: {lon.shape} vs {lat.shape}")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, lons=lon, lats=lat, longitude=lon, latitude=lat)
    return output


def load_seviri_lonlat(path: str | Path, aoi=None) -> SeviriGeolocation:
    """Load only the SEVIRI lon/lat grid from an NPZ file."""

    source = _resolve_path(path)
    lon, lat = _read_lonlat_npz(source)
    crop = geographic_crop(lon, lat, aoi)
    if crop:
        lon = lon[crop].copy()
        lat = lat[crop].copy()
    return SeviriGeolocation(lon, lat, source, crop)


def _is_probable_native_seviri_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() in {".xml", ".json", ".txt", ".csv", ".zip", ".npz", ".tmp"}:
        return False
    try:
        if path.stat().st_size < 1024 * 1024:
            return False
    except OSError:
        return False
    name = path.name.upper()
    return "SEVI" in name or "MSG" in name or "HRSEVIRI" in name


def find_native_seviri_file(path: str | Path) -> Path:
    """Find one native SEVIRI file under a downloaded SEVIRI root."""

    root = _resolve_path(path)
    if root.is_file() and _is_probable_native_seviri_file(root):
        return root
    if not root.exists():
        raise FileNotFoundError(f"SEVIRI path does not exist: {root}")

    search_roots = []
    if (root / "native").exists():
        search_roots.append(root / "native")
    search_roots.append(root)

    candidates: list[Path] = []
    for item in search_roots:
        candidates.extend(sorted(path for path in item.rglob("*") if _is_probable_native_seviri_file(path)))
    candidates = _unique_paths(candidates)
    if not candidates:
        raise FileNotFoundError(
            f"No native SEVIRI data file found under {root}. Run download_seviri(..., extract=True) first."
        )
    return candidates[0]


def _channel_to_satpy_dataset(channel: str | None) -> str:
    if channel is None:
        return "IR_108"
    # Satpy SEVIRI dataset names use forms such as ``WV_062`` and ``IR_108``.
    # Notebook/user-facing names are often written as ``WV6.2`` or ``IR8.7``.
    # Normalise both forms here before calling Satpy.
    raw = str(channel).strip()
    upper = raw.upper().replace(" ", "")
    text = upper.replace(".", "").replace("_", "").replace("-", "")
    mapping = {
        "IR108": "IR_108",
        "IR10_8": "IR_108",
        "IR1080": "IR_108",
        "IR87": "IR_087",
        "IR087": "IR_087",
        "IR08_7": "IR_087",
        "IR097": "IR_097",
        "IR97": "IR_097",
        "IR09_7": "IR_097",
        "IR120": "IR_120",
        "IR12_0": "IR_120",
        "IR134": "IR_134",
        "IR13_4": "IR_134",
        "IR039": "IR_039",
        "IR39": "IR_039",
        "IR03_9": "IR_039",
        "WV62": "WV_062",
        "WV062": "WV_062",
        "WV06_2": "WV_062",
        "WV73": "WV_073",
        "WV073": "WV_073",
        "WV07_3": "WV_073",
        "VIS006": "VIS006",
        "VIS06": "VIS006",
        "VIS008": "VIS008",
        "VIS08": "VIS008",
        "IR016": "IR_016",
        "NIR016": "IR_016",
        "NIR16": "IR_016",
        "HRV": "HRV",
    }
    if text in mapping:
        return mapping[text]
    # If the caller already passes an exact Satpy dataset name, keep it.
    if upper in {"WV_062", "WV_073", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134", "IR_039", "IR_016", "VIS006", "VIS008", "HRV"}:
        return upper
    return raw


def _satpy_scene(native_file: Path, dataset_name: str):
    try:
        from satpy import Scene  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Native SEVIRI loading requires Satpy. Install/configure satpy, or convert native files to NPZ first."
        ) from exc

    errors: list[str] = []
    for kwargs in (
        {"reader": "seviri_l1b_native", "filenames": [str(native_file)]},
        {"filenames": [str(native_file)]},
    ):
        try:
            scene = Scene(**kwargs)
            scene.load([dataset_name])
            return scene
        except Exception as exc:  # pragma: no cover - depends on Satpy/readers
            errors.append(f"{kwargs}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"Satpy could not open native SEVIRI file or load {dataset_name!r}: {native_file}\n"
        + "\n".join(errors)
    )


def build_seviri_lonlat_cache_from_native(
    source_path: str | Path,
    output_path: str | Path | None = None,
    channel: str | None = "IR10.8",
    overwrite: bool = False,
) -> Path:
    """Build and save a SEVIRI lon/lat cache from one downloaded native file."""

    source = _resolve_path(source_path)
    output = _resolve_path(output_path) if output_path is not None else seviri_lonlat_cache_path(source if source.is_dir() else source.parent)
    if output.exists() and not overwrite:
        return output

    native_file = find_native_seviri_file(source)
    dataset_name = _channel_to_satpy_dataset(channel)
    scene = _satpy_scene(native_file, dataset_name)
    area = scene[dataset_name].attrs.get("area")
    if area is None or not hasattr(area, "get_lonlats"):
        raise RuntimeError(f"No Satpy area definition found for dataset {dataset_name!r}")
    lon, lat = area.get_lonlats()
    if hasattr(lon, "compute"):
        lon = lon.compute()
    if hasattr(lat, "compute"):
        lat = lat.compute()
    return save_seviri_lonlat(lon, lat, output, overwrite=True)


def ensure_seviri_lonlat_cache(
    seviri_path: str | Path,
    output_path: str | Path | None = None,
    channel: str | None = "IR10.8",
    overwrite: bool = False,
) -> Path:
    """Return an existing lon/lat cache, or build it from downloaded native data."""

    source = _resolve_path(seviri_path)
    output = _resolve_path(output_path) if output_path is not None else seviri_lonlat_cache_path(source)
    if output.exists() and not overwrite:
        return output
    return build_seviri_lonlat_cache_from_native(source, output_path=output, channel=channel, overwrite=overwrite)


def infer_aoi_from_lonlat(
    path: str | Path,
    buffer_degrees: float = 0.0,
    round_to: float | None = 0.25,
) -> dict[str, float]:
    """Infer an AOI dict from finite lon/lat pixels in a cache file."""

    geo = load_seviri_lonlat(path, aoi=None)
    lon = np.asarray(geo.longitude, dtype=float)
    lat = np.asarray(geo.latitude, dtype=float)
    valid = np.isfinite(lon) & np.isfinite(lat)
    if valid.sum() == 0:
        raise RuntimeError(f"No finite SEVIRI lon/lat pixels found in {geo.path}")

    lon_min = float(np.nanmin(lon[valid])) - buffer_degrees
    lon_max = float(np.nanmax(lon[valid])) + buffer_degrees
    lat_min = float(np.nanmin(lat[valid])) - buffer_degrees
    lat_max = float(np.nanmax(lat[valid])) + buffer_degrees
    lon_min, lon_max = max(-180.0, lon_min), min(180.0, lon_max)
    lat_min, lat_max = max(-90.0, lat_min), min(90.0, lat_max)

    if round_to is not None and round_to > 0:
        import math

        lon_min = math.floor(lon_min / round_to) * round_to
        lon_max = math.ceil(lon_max / round_to) * round_to
        lat_min = math.floor(lat_min / round_to) * round_to
        lat_max = math.ceil(lat_max / round_to) * round_to

    return {"lon_min": lon_min, "lon_max": lon_max, "lat_min": lat_min, "lat_max": lat_max}


# -----------------------------------------------------------------------------
# Frame loading
# -----------------------------------------------------------------------------


def _time_from_path(path: Path) -> datetime | None:
    text = str(path)
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


def _time_from_npz(path: Path, data: np.lib.npyio.NpzFile) -> datetime:
    for key in ("start_time", "name_time", "time"):
        if key in data.files:
            text = str(_scalar(data, key))
            return (
                datetime.strptime(text[:14], "%Y%m%d%H%M%S")
                if re.fullmatch(r"\d{14}.*", text)
                else datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
            )
    parsed = _time_from_path(path)
    if parsed is None:
        raise ValueError(f"Cannot identify frame time in {path.name}")
    return parsed


def _candidate_time(path: Path) -> datetime | None:
    if path.suffix.lower() == ".npz":
        try:
            with np.load(path, allow_pickle=False) as data:
                return _time_from_npz(path, data)
        except Exception:
            return _time_from_path(path)
    return _time_from_path(path)


def load_frame(path: str | Path, channel: str | None = "IR10.8") -> SeviriFrame:
    """Load one SEVIRI frame from converted NPZ or native file."""

    source = _resolve_path(path)
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as data:
            image_key = next(
                (name for name in ("image", "IR108", "IR_108", "bt108", "brightness_temperature") if name in data.files),
                None,
            )
            if image_key is None:
                raise KeyError(f"No IR10.8 image key found in {source.name}")
            start = _time_from_npz(source, data)
            end_text = _scalar(data, "end_time")
            end = _parse_datetime(end_text) if end_text is not None else start
            return SeviriFrame(
                source,
                np.asarray(data[image_key], dtype=np.float32),
                start,
                end,
                str(_scalar(data, "units", "K")),
                int(_scalar(data, "x0", 0)),
                int(_scalar(data, "y0", 0)),
            )

    dataset_name = _channel_to_satpy_dataset(channel)
    scene = _satpy_scene(source, dataset_name)
    array = scene[dataset_name]
    values = array.values
    if hasattr(values, "compute"):
        values = values.compute()
    image = np.asarray(values, dtype=np.float32)
    units = str(array.attrs.get("units", "K"))
    start = _time_from_path(source) or datetime.fromtimestamp(source.stat().st_mtime)
    return SeviriFrame(source, image, start, start, units, 0, 0)


def _requested_frame_times(config: dict) -> tuple[datetime, ...]:
    time_texts = config.get("frame_times")
    if time_texts is None:
        pairs = config.get("component_pairs")
        if not pairs:
            raise KeyError("config must contain frame_times or component_pairs")
        time_texts = [pairs[0]["start_time"], *[item["end_time"] for item in pairs]]
    return tuple(_parse_datetime(value) for value in time_texts)


def _selected_index_files(seviri_root: Path) -> list[Path]:
    index = seviri_root / "index" / "seviri_selected_frames.csv"
    if not index.exists():
        return []
    paths: list[Path] = []
    with index.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get("native_files") or row.get("native_file") or ""
            for text in value.split(";"):
                text = text.strip()
                if not text:
                    continue
                path = Path(text)
                if not path.is_absolute():
                    path = _resolve_path(path)
                if path.exists():
                    paths.append(path)
    return _unique_paths(paths)


def _candidate_frame_files(config: dict, include_native: bool = True) -> list[Path]:
    """Find converted NPZ candidates first, then downloaded native files."""

    paths_cfg = config.get("paths", {}) or {}
    candidates: list[Path] = []

    for pattern in (
        config.get("seviri_glob"),
        paths_cfg.get("seviri_aoi_npz_glob"),
        paths_cfg.get("seviri_npz_glob"),
    ):
        if pattern:
            candidates.extend(path for path in REPO_ROOT.glob(str(pattern)) if path.is_file())

    seviri_root_value = paths_cfg.get("seviri")
    if seviri_root_value is not None:
        seviri_root = _resolve_path(seviri_root_value)
        candidates.extend(_selected_index_files(seviri_root))
        if include_native and (seviri_root / "native").exists():
            candidates.extend(path for path in (seviri_root / "native").rglob("*") if _is_probable_native_seviri_file(path))
        candidates.extend(
            path for path in seviri_root.rglob("*.npz")
            if "lon_lat" not in path.parts and "lonlat" not in path.name.lower()
        )

    return _unique_paths(candidates)


def _select_frames_by_time(
    candidates: list[Path],
    times: tuple[datetime, ...],
    tolerance_minutes: float,
) -> tuple[Path, ...]:
    indexed = [(time, path) for path in candidates if (time := _candidate_time(path)) is not None]
    if not indexed:
        raise FileNotFoundError("No timestamped SEVIRI frame candidates were found")

    selected: list[Path] = []
    missing: list[datetime] = []
    tolerance = timedelta(minutes=tolerance_minutes)
    used: set[Path] = set()
    for target_time in times:
        available = [(time, path) for time, path in indexed if path not in used]
        if not available:
            missing.append(target_time)
            continue
        best_time, best_path = min(available, key=lambda item: abs(item[0] - target_time))
        if abs(best_time - target_time) <= tolerance:
            selected.append(best_path)
            used.add(best_path)
        else:
            missing.append(target_time)
    if missing:
        examples = ", ".join(f"{time.isoformat()}={path.name}" for time, path in indexed[:20])
        raise FileNotFoundError(
            f"Missing SEVIRI frames within {tolerance_minutes} minutes of {missing}. "
            f"Available examples: {examples}"
        )
    return tuple(selected)


def _load_lonlat_for_frame(config: dict, first: SeviriFrame, channel: str | None) -> tuple[np.ndarray, np.ndarray]:
    explicit = config.get("latlon_file") or config.get("paths", {}).get("latlon_file")
    if explicit is not None:
        lon, lat = _read_lonlat_npz(_resolve_path(explicit))
    else:
        seviri_root = config.get("paths", {}).get("seviri")
        if seviri_root is None:
            # Last resort for converted NPZ frames that include x0/y0 and use legacy cache.
            legacy = repo_path("data/intermediate/seviri_ir108/seviri_lonlat_cache.npz")
            if not legacy.exists():
                raise FileNotFoundError(
                    "No latlon_file and no paths['seviri'] found in config. "
                    "Create lon/lat cache with ensure_seviri_lonlat_cache(paths['seviri'])."
                )
            lon, lat = _read_lonlat_npz(legacy)
        else:
            cache = ensure_seviri_lonlat_cache(seviri_root, channel=channel)
            lon, lat = _read_lonlat_npz(cache)

    if lon.shape != first.image.shape:
        y1 = first.y0 + first.image.shape[0]
        x1 = first.x0 + first.image.shape[1]
        lon, lat = lon[first.y0:y1, first.x0:x1], lat[first.y0:y1, first.x0:x1]
    return lon, lat


def load_seviri_images(
    case_config,
    aoi=None,
    channel: str | None = "IR10.8",
    time_tolerance_minutes: float = 8.0,
) -> SeviriImages:
    """Load case frames and optionally crop them by lon/lat AOI.

    The loader first searches converted NPZ products and then native SEVIRI
    files downloaded by ``download_seviri``. Native loading requires Satpy.
    """

    config = _as_config(case_config)
    candidates = _candidate_frame_files(config, include_native=True)
    if not candidates:
        raise FileNotFoundError(
            "No SEVIRI frame files were found. Expected converted NPZ files or native files under paths['seviri']."
        )

    times = _requested_frame_times(config)
    selected_paths = _select_frames_by_time(candidates, times, time_tolerance_minutes)
    frames = tuple(load_frame(path, channel=channel) for path in selected_paths)
    if len({frame.image.shape for frame in frames}) != 1:
        raise ValueError("SEVIRI frame shapes differ")

    longitude, latitude = _load_lonlat_for_frame(config, frames[0], channel)
    requested_aoi = aoi if aoi is not None else config.get("aoi")
    crop = geographic_crop(longitude, latitude, requested_aoi)
    images = tuple(frame.image[crop].copy() if crop else frame.image.copy() for frame in frames)
    if crop:
        longitude, latitude = longitude[crop].copy(), latitude[crop].copy()

    geo_valid = np.isfinite(longitude) & np.isfinite(latitude)
    # 150–350 K follows the common SEVIRI IR BT conversion/LUT domain.
    valid_min = float(config.get("preprocessing", {}).get("valid_bt_min", 150))
    valid_max = float(config.get("preprocessing", {}).get("valid_bt_max", 350))
    masks = tuple(geo_valid & np.isfinite(image) & (image >= valid_min) & (image <= valid_max) for image in images)

    return SeviriImages(
        images=images,
        valid_masks=masks,
        frame_times=times,
        paths=tuple(frame.path for frame in frames),
        longitude=longitude,
        latitude=latitude,
        crop=crop,
    )
