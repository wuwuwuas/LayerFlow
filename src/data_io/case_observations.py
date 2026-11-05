"""Load one AMV case directly from original SEVIRI and OCA products.

This module is the single file-system boundary for tracker examples.  It keeps
raw-product discovery/loading out of the example scripts and returns the common
array contracts expected by the existing tracker runners.

The formal runtime path is:

    original SEVIRI native products -> I0/I1/I2/I3
    original OCA NetCDF products    -> P0/P1/P2/P3

No tracking-input bundle is read, padded, or repaired here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from config import REPO_ROOT
from .oca import OcaData, load_oca
from .seviri import SeviriImages, load_seviri_images


@dataclass(frozen=True)
class CaseObservations:
    """Four aligned SEVIRI/OCA observation frames for one AMV case."""

    case_name: str
    data_case_dir: Path
    images: np.ndarray
    valid_masks: np.ndarray
    lon_grid: np.ndarray
    lat_grid: np.ndarray
    frame_times: tuple[datetime, ...]
    seviri_paths: tuple[Path, ...]
    frame_pressure_hpa: np.ndarray | None = None
    frame_cloud_probability: np.ndarray | None = None
    oca_times: tuple[datetime | None, ...] = ()
    oca_paths: tuple[Path, ...] = ()
    raw_config: Mapping[str, Any] | None = None

    @property
    def pressure_hpa(self) -> np.ndarray | None:
        """Alias for ``frame_pressure_hpa``."""

        return self.frame_pressure_hpa

    @property
    def cloud_probability(self) -> np.ndarray | None:
        """Alias for ``frame_cloud_probability``."""

        return self.frame_cloud_probability

    @property
    def longitude(self) -> np.ndarray:
        return self.lon_grid

    @property
    def latitude(self) -> np.ndarray:
        return self.lat_grid


def _parse_case_start(value: str | Path) -> datetime:
    """Parse ``YYYY-MM-DDTHH`` from a case directory or case name."""

    text = Path(value).name
    if text.startswith("test_"):
        text = text[len("test_") :]
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            "Could not infer the case start time from "
            f"{value!r}. Pass explicit frame_times instead."
        ) from exc
    return parsed.replace(tzinfo=None)


def _normalize_frame_times(
    frame_times: Sequence[Any] | None,
    *,
    case_start_source: str | Path,
    frame_count: int,
    interval_minutes: float,
) -> tuple[datetime, ...]:
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")

    if frame_times is None:
        start = _parse_case_start(case_start_source)
        return tuple(
            start + timedelta(minutes=float(interval_minutes) * index)
            for index in range(frame_count)
        )

    values: list[datetime] = []
    for value in frame_times:
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(
                str(value).replace("Z", "+00:00")
            )
        values.append(parsed.replace(tzinfo=None))

    if len(values) != frame_count:
        raise ValueError(
            f"Expected {frame_count} frame_times, got {len(values)}"
        )
    if any(second <= first for first, second in zip(values, values[1:])):
        raise ValueError("frame_times must be strictly increasing")
    return tuple(values)


def build_raw_case_config(
    *,
    case_name: str,
    data_case_dir: str | Path,
    frame_times: Sequence[Any] | None = None,
    frame_count: int = 4,
    interval_minutes: float = 15.0,
    valid_bt_min: float = 150.0,
    valid_bt_max: float = 350.0,
    aoi: Any = None,
) -> dict[str, Any]:
    """Build the minimal config consumed by existing SEVIRI/OCA loaders.

    Absolute raw-data paths are used deliberately, so this runtime does not
    depend on a pre-exported bundle or converted image products.
    """

    data_dir = Path(data_case_dir).expanduser().resolve()
    seviri_root = data_dir / "eumetsat_seviri_l15"
    oca_root = data_dir / "eumetsat_oca"

    if not data_dir.exists():
        raise FileNotFoundError(f"Case data directory not found: {data_dir}")
    if not seviri_root.exists():
        raise FileNotFoundError(
            f"Original SEVIRI directory not found: {seviri_root}"
        )

    times = _normalize_frame_times(
        frame_times,
        case_start_source=data_dir.name,
        frame_count=frame_count,
        interval_minutes=interval_minutes,
    )

    config: dict[str, Any] = {
        "case_name": str(case_name),
        "frame_times": [value.isoformat(sep=" ") for value in times],
        "paths": {
            "seviri": str(seviri_root),
            "oca": str(oca_root),
        },
        "preprocessing": {
            "valid_bt_min": float(valid_bt_min),
            "valid_bt_max": float(valid_bt_max),
        },
    }
    if aoi is not None:
        config["aoi"] = aoi
    return config


def _stack_required_frames(
    values: Sequence[np.ndarray],
    *,
    name: str,
    frame_count: int,
    expected_shape: tuple[int, int],
    dtype: Any,
) -> np.ndarray:
    if len(values) != frame_count:
        raise ValueError(
            f"{name} must contain exactly {frame_count} frames, got {len(values)}"
        )
    arrays = [np.asarray(value, dtype=dtype) for value in values]
    for index, array in enumerate(arrays):
        if array.shape != expected_shape:
            raise ValueError(
                f"{name}[{index}] shape {array.shape} does not match "
                f"SEVIRI shape {expected_shape}"
            )
    return np.stack(arrays, axis=0)


def _stack_optional_frames(
    values: Sequence[np.ndarray | None],
    *,
    name: str,
    frame_count: int,
    expected_shape: tuple[int, int],
    dtype: Any,
) -> np.ndarray | None:
    if len(values) != frame_count:
        raise ValueError(
            f"{name} must contain exactly {frame_count} entries, got {len(values)}"
        )
    if any(value is None for value in values):
        return None
    return _stack_required_frames(
        values,  # type: ignore[arg-type]
        name=name,
        frame_count=frame_count,
        expected_shape=expected_shape,
        dtype=dtype,
    )


def _validate_unique_paths(paths: Sequence[Path], *, name: str) -> None:
    resolved = [Path(path).resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError(
            f"{name} contains repeated files; every observation frame must "
            "come from an independent original product"
        )


def _validate_oca_times(
    requested: Sequence[datetime],
    actual: Sequence[datetime | None],
    *,
    tolerance_minutes: float,
) -> None:
    if len(actual) != len(requested):
        raise ValueError(
            f"OCA loader returned {len(actual)} frame times for "
            f"{len(requested)} requested frames"
        )
    for index, (target, observed) in enumerate(zip(requested, actual)):
        if observed is None:
            raise ValueError(f"OCA frame P{index} has no timestamp")
        difference = abs((observed - target).total_seconds()) / 60.0
        if difference > float(tolerance_minutes):
            raise ValueError(
                f"OCA frame P{index} time mismatch: requested={target}, "
                f"actual={observed}, dt={difference:.3f} min"
            )


def load_case_observations(
    *,
    case_name: str,
    data_case_dir: str | Path | None = None,
    frame_times: Sequence[Any] | None = None,
    frame_count: int = 4,
    interval_minutes: float = 15.0,
    channel: str | None = "IR10.8",
    aoi: Any = None,
    load_oca_data: bool = True,
    seviri_time_tolerance_minutes: float = 8.0,
    oca_time_tolerance_minutes: float = 7.5,
    valid_bt_min: float = 150.0,
    valid_bt_max: float = 350.0,
) -> CaseObservations:
    """Load aligned original products for one tracker case.

    Parameters
    ----------
    case_name:
        Output/experiment name, for example ``test_2015-06-12T11``.
    data_case_dir:
        Raw case directory.  When omitted it is resolved as
        ``REPO_ROOT/data/<case_name without test_>``.
    frame_times:
        Optional explicit times.  Otherwise four times are generated from the
        raw case directory name at ``interval_minutes`` spacing.
    load_oca_data:
        ``False`` for raw-image PIV; ``True`` for layerwise and PCF runs.
    """

    if data_case_dir is None:
        data_name = str(case_name)
        if data_name.startswith("test_"):
            data_name = data_name[len("test_") :]
        data_case_dir = REPO_ROOT / "data" / data_name

    config = build_raw_case_config(
        case_name=case_name,
        data_case_dir=data_case_dir,
        frame_times=frame_times,
        frame_count=frame_count,
        interval_minutes=interval_minutes,
        valid_bt_min=valid_bt_min,
        valid_bt_max=valid_bt_max,
        aoi=aoi,
    )

    seviri: SeviriImages = load_seviri_images(
        config,
        aoi=aoi,
        channel=channel,
        time_tolerance_minutes=seviri_time_tolerance_minutes,
    )

    if len(seviri.images) != frame_count:
        raise ValueError(
            f"SEVIRI loader returned {len(seviri.images)} frames; "
            f"expected {frame_count}"
        )
    _validate_unique_paths(seviri.paths, name="SEVIRI frame paths")

    images = _stack_required_frames(
        seviri.images,
        name="SEVIRI images",
        frame_count=frame_count,
        expected_shape=np.asarray(seviri.images[0]).shape,
        dtype=np.float32,
    )
    valid_masks = _stack_required_frames(
        seviri.valid_masks,
        name="SEVIRI valid masks",
        frame_count=frame_count,
        expected_shape=images.shape[1:],
        dtype=bool,
    )
    lon_grid = np.asarray(seviri.longitude, dtype=np.float32)
    lat_grid = np.asarray(seviri.latitude, dtype=np.float32)
    if lon_grid.shape != images.shape[1:] or lat_grid.shape != images.shape[1:]:
        raise ValueError(
            "SEVIRI longitude/latitude grids do not match image shape: "
            f"images={images.shape[1:]}, lon={lon_grid.shape}, lat={lat_grid.shape}"
        )

    pressure: np.ndarray | None = None
    cloud: np.ndarray | None = None
    oca_times: tuple[datetime | None, ...] = ()
    oca_paths: tuple[Path, ...] = ()

    if load_oca_data:
        oca_root = Path(config["paths"]["oca"])
        if not oca_root.exists():
            raise FileNotFoundError(
                f"Original OCA directory not found: {oca_root}"
            )

        oca: OcaData = load_oca(
            config,
            aoi=aoi,
            seviri_data=seviri,
            time_tolerance_minutes=oca_time_tolerance_minutes,
        )
        oca_paths = tuple(Path(path).resolve() for path in oca.frame_paths)
        oca_times = tuple(oca.frame_times)

        if len(oca_paths) != frame_count:
            raise ValueError(
                f"OCA loader returned {len(oca_paths)} files; "
                f"expected {frame_count} independent P0..P{frame_count - 1} products"
            )
        _validate_unique_paths(oca_paths, name="OCA frame paths")
        _validate_oca_times(
            seviri.frame_times,
            oca_times,
            tolerance_minutes=oca_time_tolerance_minutes,
        )

        pressure = _stack_required_frames(
            oca.frame_ctp_hpa,
            name="OCA frame pressure",
            frame_count=frame_count,
            expected_shape=images.shape[1:],
            dtype=np.float32,
        )
        cloud = _stack_optional_frames(
            oca.frame_cloud_probability,
            name="OCA frame cloud probability",
            frame_count=frame_count,
            expected_shape=images.shape[1:],
            dtype=np.float32,
        )

    return CaseObservations(
        case_name=str(case_name),
        data_case_dir=Path(data_case_dir).expanduser().resolve(),
        images=images,
        valid_masks=valid_masks,
        lon_grid=lon_grid,
        lat_grid=lat_grid,
        frame_times=tuple(seviri.frame_times),
        seviri_paths=tuple(Path(path).resolve() for path in seviri.paths),
        frame_pressure_hpa=pressure,
        frame_cloud_probability=cloud,
        oca_times=oca_times,
        oca_paths=oca_paths,
        raw_config=config,
    )
