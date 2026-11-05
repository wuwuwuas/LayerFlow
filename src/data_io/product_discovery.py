"""Product discovery helpers for AMV case directories.

Resolve local product paths for data_io.oca and data_io.official_amv.

Typical case directory layout
-----------------------------
data/YYYY-MM-DDTHH/
    eumetsat_oca/native/**/*.nc
    eumetsat_amv/native/**/*.nc

Examples
--------
from data_io.product_discovery import (
    find_oca_component_files,
    find_official_amv_file,
)

DATA_CASE_DIR = PROJECT_ROOT / "data/2018-01-12T11"

OCA_COMPONENT_FILES = find_oca_component_files(
    DATA_CASE_DIR / "eumetsat_oca" / "native",
    n_components=3,
)

OFFICIAL_AMV_FILE = find_official_amv_file(
    DATA_CASE_DIR / "eumetsat_amv" / "native",
)
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable


def _parse_yyyymmddhhmmss_from_text(text: str) -> list[datetime]:
    """Parse all 14-digit YYYYMMDDHHMMSS timestamps from a string."""
    times: list[datetime] = []
    for stamp in re.findall(r"(?<!\d)(\d{14})(?!\d)", text):
        try:
            times.append(datetime.strptime(stamp, "%Y%m%d%H%M%S"))
        except ValueError:
            continue
    return times


def _first_time_in_path(path: Path) -> datetime | None:
    times = _parse_yyyymmddhhmmss_from_text(str(path))
    return times[0] if times else None


def _last_time_in_path(path: Path) -> datetime | None:
    times = _parse_yyyymmddhhmmss_from_text(str(path))
    return times[-1] if times else None


def _all_nc_files(root: str | Path, *, product_name: str) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{product_name} native directory does not exist: {root}")

    files = sorted(root.rglob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No {product_name} .nc files found under: {root}")

    return files


def _format_paths(paths: Iterable[Path]) -> str:
    return "\n".join(f"  {path}" for path in paths)


def find_oca_component_files(
    oca_native_dir: str | Path,
    n_components: int = 3,
    expected_start_time: str | datetime | None = None,
    interval_minutes: int = 15,
    tolerance_minutes: float = 7.5,
    verbose: bool = True,
) -> list[Path]:
    """Find OCA component files in a local case directory.

    Parameters
    ----------
    oca_native_dir:
        Directory such as ``data/YYYY-MM-DDTHH/eumetsat_oca/native``.
    n_components:
        Number of component-start OCA files required. For C1/C2/C3 use 3.
    expected_start_time:
        Optional case start time. If provided, files are selected closest to
        ``start + k * interval_minutes``. If omitted, timestamped files are
        sorted by time and the earliest ``n_components`` files are returned.
    interval_minutes:
        Expected spacing between OCA component files.
    tolerance_minutes:
        Maximum allowed time difference when ``expected_start_time`` is given.
    verbose:
        Print selected files and warnings.

    Returns
    -------
    list[Path]
        Ordered OCA files. Order matters: C1, C2, C3.

    Notes
    -----
    This function resolves local paths only. Actual loading is still done by
    ``data_io.oca.load_oca_component_files``.
    """
    oca_native_dir = Path(oca_native_dir)
    files = _all_nc_files(oca_native_dir, product_name="OCA")

    parsed: list[tuple[datetime, Path]] = []
    unparsed: list[Path] = []
    for path in files:
        # OCA paths often contain the same timestamp in folder and filename.
        # Using the last timestamp is usually closest to the actual .nc file.
        timestamp = _last_time_in_path(path)
        if timestamp is None:
            unparsed.append(path)
        else:
            parsed.append((timestamp, path))

    if len(parsed) < n_components:
        raise RuntimeError(
            f"Found only {len(parsed)} timestamped OCA files under {oca_native_dir}, "
            f"but n_components={n_components} is required.\n"
            f"All .nc files:\n{_format_paths(files)}"
        )

    selected_items: list[tuple[datetime, Path]]

    if expected_start_time is not None:
        if isinstance(expected_start_time, str):
            start_time = datetime.fromisoformat(expected_start_time.replace("Z", "+00:00"))
            if start_time.tzinfo is not None:
                start_time = start_time.replace(tzinfo=None)
        else:
            start_time = expected_start_time

        unused = set(path for _, path in parsed)
        selected_items = []

        for i in range(n_components):
            expected_time = start_time + timedelta(minutes=interval_minutes * i)
            candidates = [
                (abs((actual_time - expected_time).total_seconds()) / 60.0, actual_time, path)
                for actual_time, path in parsed
                if path in unused
            ]

            if not candidates:
                raise RuntimeError(f"Could not find OCA file for expected time {expected_time}")

            dt_min, actual_time, path = min(candidates, key=lambda item: item[0])
            if dt_min > tolerance_minutes:
                raise RuntimeError(
                    "Nearest OCA file is too far from expected component time. "
                    f"expected={expected_time}, actual={actual_time}, "
                    f"dt={dt_min:.2f} min, tolerance={tolerance_minutes:.2f} min, "
                    f"path={path}"
                )

            selected_items.append((actual_time, path))
            unused.remove(path)
    else:
        parsed = sorted(parsed, key=lambda item: (item[0], str(item[1])))
        selected_items = parsed[:n_components]

    selected = [path for _, path in selected_items]

    if verbose:
        print("[OCA auto-select]")
        print("  oca_native_dir:", oca_native_dir)
        print("  found .nc files:", len(files))
        for i, (timestamp, path) in enumerate(selected_items, start=1):
            print(f"  C{i}: time={timestamp} | {path}")

        extra_count = len(parsed) - n_components
        if expected_start_time is None and extra_count > 0:
            print(f"  warning: {extra_count} extra timestamped OCA files were not used.")
            for timestamp, path in sorted(parsed[n_components:], key=lambda item: item[0]):
                print(f"    unused: time={timestamp} | {path}")

        if unparsed:
            print("  warning: some OCA .nc files had no parseable timestamp:")
            for path in unparsed:
                print(f"    unparsed: {path}")

    return selected


def parse_amv_time_window_from_path(path: str | Path) -> tuple[datetime, datetime] | None:
    """Parse an AMV time window from a file/path string.

    Example
    -------
    ``amv_20150612110000_20150612120000_xxxx.nc``
    returns ``(2015-06-12 11:00:00, 2015-06-12 12:00:00)``.
    """
    path = Path(path)
    times = _parse_yyyymmddhhmmss_from_text(str(path))

    if len(times) < 2:
        return None

    for i in range(len(times) - 1):
        t0 = times[i]
        t1 = times[i + 1]
        if t0 <= t1:
            return t0, t1

    return None


def find_official_amv_file(
    official_amv_native_dir: str | Path,
    expected_time: str | datetime | None = None,
    tolerance_minutes: float = 90.0,
    verbose: bool = True,
) -> Path:
    """Find official AMV NetCDF file in a local case directory.

    Parameters
    ----------
    official_amv_native_dir:
        Directory such as ``data/YYYY-MM-DDTHH/eumetsat_amv/native``.
    expected_time:
        Optional case time. If provided, a file whose time window contains this
        time is preferred. If none contains it, the nearest time-window center is
        used within ``tolerance_minutes``.
    tolerance_minutes:
        Maximum allowed distance to time-window center when ``expected_time`` is
        provided and no file contains that time.
    verbose:
        Print selected file and warnings.

    Returns
    -------
    Path
        Selected official AMV .nc file.
    """
    official_amv_native_dir = Path(official_amv_native_dir)
    files = _all_nc_files(official_amv_native_dir, product_name="official AMV")

    if len(files) == 1 and expected_time is None:
        selected = files[0]
        if verbose:
            print("[Official AMV auto-select]")
            print("  official_amv_native_dir:", official_amv_native_dir)
            print("  found .nc files: 1")
            print("  selected:", selected)
        return selected

    # Prefer files that look like AMV products.
    amv_like = [
        path for path in files
        if "amv" in path.name.lower() or "amv" in path.parent.name.lower()
    ]
    candidates = amv_like if amv_like else files

    parsed: list[dict[str, object]] = []
    unparsed: list[Path] = []

    for path in candidates:
        window = parse_amv_time_window_from_path(path)
        if window is None:
            unparsed.append(path)
            continue

        start_time, end_time = window
        center_time = start_time + (end_time - start_time) / 2
        parsed.append(
            {
                "path": path,
                "start_time": start_time,
                "end_time": end_time,
                "center_time": center_time,
            }
        )

    selected: Path

    if expected_time is not None and parsed:
        if isinstance(expected_time, str):
            t_expected = datetime.fromisoformat(expected_time.replace("Z", "+00:00"))
            if t_expected.tzinfo is not None:
                t_expected = t_expected.replace(tzinfo=None)
        else:
            t_expected = expected_time

        containing = [
            item for item in parsed
            if item["start_time"] <= t_expected <= item["end_time"]  # type: ignore[operator]
        ]

        if containing:
            best = sorted(
                containing,
                key=lambda item: (
                    item["end_time"] - item["start_time"],  # type: ignore[operator]
                    abs((item["center_time"] - t_expected).total_seconds()),  # type: ignore[union-attr]
                    str(item["path"]),
                ),
            )[0]
        else:
            def center_dt(item: dict[str, object]) -> float:
                return abs((item["center_time"] - t_expected).total_seconds()) / 60.0  # type: ignore[union-attr]

            best = min(parsed, key=center_dt)
            best_dt = center_dt(best)
            if best_dt > tolerance_minutes:
                raise RuntimeError(
                    "Nearest official AMV file is too far from expected time. "
                    f"expected_time={t_expected}, "
                    f"amv_window={best['start_time']} -> {best['end_time']}, "
                    f"center_dt={best_dt:.2f} min, "
                    f"tolerance={tolerance_minutes:.2f} min, "
                    f"path={best['path']}"
                )

        selected = Path(best["path"])  # type: ignore[arg-type]
    elif parsed:
        # No expected time supplied: choose the earliest AMV time window.
        best = sorted(parsed, key=lambda item: (item["start_time"], str(item["path"])))[0]
        selected = Path(best["path"])  # type: ignore[arg-type]
    else:
        if len(candidates) == 1:
            selected = candidates[0]
        else:
            raise RuntimeError(
                "Could not parse official AMV time window and multiple candidates exist. "
                "Please inspect manually:\n"
                f"{_format_paths(candidates)}"
            )

    if verbose:
        print("[Official AMV auto-select]")
        print("  official_amv_native_dir:", official_amv_native_dir)
        print("  found .nc files:", len(files))
        print("  AMV-like candidates:", len(amv_like))
        print("  selected:", selected)

        selected_window = parse_amv_time_window_from_path(selected)
        if selected_window is not None:
            print("  selected window:", selected_window[0], "->", selected_window[1])

        if unparsed:
            print("  warning: some official AMV candidates had no parseable time window:")
            for path in unparsed:
                print(f"    unparsed: {path}")

    return selected

def _parse_yyyymmddhhmm_or_yyyymmddhhmmss_from_text(text: str) -> list[datetime]:
    """Parse YYYYMMDDHHMMSS or YYYYMMDDHHMM timestamps from a string."""
    times: list[datetime] = []
    for stamp in re.findall(r"(?<!\d)(\d{14}|\d{12})(?!\d)", text):
        fmt = "%Y%m%d%H%M%S" if len(stamp) == 14 else "%Y%m%d%H%M"
        try:
            times.append(datetime.strptime(stamp, fmt))
        except ValueError:
            continue
    return times


def infer_case_start_time_from_path(value: str | Path) -> datetime:
    """Infer case start time from strings/paths containing YYYY-MM-DDTHH."""
    text = str(value)
    match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2})", text)
    if match is None:
        raise ValueError(
            "Could not infer case start time. Expected a string/path containing "
            f"YYYY-MM-DDTHH, got: {value!r}"
        )
    return datetime.strptime(match.group(1), "%Y-%m-%dT%H")


def infer_amv_valid_time_from_path(
    value: str | Path,
    offset_minutes: int = 45,
) -> datetime:
    """Infer AMV valid time as case start time plus an offset in minutes."""
    return infer_case_start_time_from_path(value) + timedelta(minutes=int(offset_minutes))


def _era5_time_span_from_path(path: Path) -> tuple[datetime, datetime] | None:
    """Parse ERA5 time span from a path, supporting 12- or 14-digit timestamps."""
    times = _parse_yyyymmddhhmm_or_yyyymmddhhmmss_from_text(str(path))
    if len(times) < 2:
        return None
    for i in range(len(times) - 1):
        t0 = times[i]
        t1 = times[i + 1]
        if t0 <= t1:
            return t0, t1
    return None


def find_era5_file(
    era5_dir: str | Path,
    expected_time: str | datetime | None = None,
    tolerance_minutes: float = 180.0,
    verbose: bool = True,
) -> Path:
    """Find ERA5 NetCDF file under a local case ERA5 directory.

    The function recursively searches ``*.nc`` files. If ``expected_time`` is
    provided, a file whose parsed time span contains that time is preferred.
    ERA5 filenames with either ``YYYYMMDDHHMM`` or ``YYYYMMDDHHMMSS`` time stamps
    are supported, for example::

        era5_201506121100_201506121145_pressure_levels.nc
    """
    era5_dir = Path(era5_dir)
    files = _all_nc_files(era5_dir, product_name="ERA5")

    if len(files) == 1 and expected_time is None:
        selected = files[0]
        if verbose:
            print("[ERA5 auto-select]")
            print("  era5_dir:", era5_dir)
            print("  found .nc files: 1")
            print("  selected:", selected)
        return selected

    parsed: list[dict[str, object]] = []
    unparsed: list[Path] = []

    for path in files:
        span = _era5_time_span_from_path(path)
        times = _parse_yyyymmddhhmm_or_yyyymmddhhmmss_from_text(str(path))
        if span is not None:
            start_time, end_time = span
            center_time = start_time + (end_time - start_time) / 2
            parsed.append(
                {
                    "path": path,
                    "start_time": start_time,
                    "end_time": end_time,
                    "center_time": center_time,
                }
            )
        elif times:
            parsed.append(
                {
                    "path": path,
                    "start_time": times[0],
                    "end_time": times[0],
                    "center_time": times[0],
                }
            )
        else:
            unparsed.append(path)

    if not parsed:
        if len(files) == 1:
            selected = files[0]
            if verbose:
                print("[ERA5 auto-select]")
                print("  warning: could not parse time, but only one file exists.")
                print("  selected:", selected)
            return selected
        raise RuntimeError(
            "Could not parse ERA5 times and multiple candidates exist. "
            "Please inspect manually:\n"
            f"{_format_paths(files)}"
        )

    if expected_time is not None:
        if isinstance(expected_time, str):
            t_expected = datetime.fromisoformat(expected_time.replace("Z", "+00:00"))
            if t_expected.tzinfo is not None:
                t_expected = t_expected.replace(tzinfo=None)
        else:
            t_expected = expected_time

        containing = [
            item for item in parsed
            if item["start_time"] <= t_expected <= item["end_time"]  # type: ignore[operator]
        ]

        if containing:
            best = sorted(
                containing,
                key=lambda item: (
                    item["end_time"] - item["start_time"],  # type: ignore[operator]
                    abs((item["center_time"] - t_expected).total_seconds()),  # type: ignore[union-attr]
                    str(item["path"]),
                ),
            )[0]
        else:
            def dt_min(item: dict[str, object]) -> float:
                return abs((item["center_time"] - t_expected).total_seconds()) / 60.0  # type: ignore[union-attr]

            best = min(parsed, key=dt_min)
            best_dt = dt_min(best)
            if best_dt > tolerance_minutes:
                raise RuntimeError(
                    "Nearest ERA5 file is too far from expected time. "
                    f"expected_time={t_expected}, "
                    f"era5_window={best['start_time']} -> {best['end_time']}, "
                    f"center_dt={best_dt:.2f} min, "
                    f"tolerance={tolerance_minutes:.2f} min, "
                    f"path={best['path']}"
                )
    else:
        best = sorted(parsed, key=lambda item: (item["start_time"], str(item["path"])))[0]

    selected = Path(best["path"])  # type: ignore[arg-type]

    if verbose:
        print("[ERA5 auto-select]")
        print("  era5_dir:", era5_dir)
        print("  found .nc files:", len(files))
        print("  selected:", selected)
        print("  selected time:", best["start_time"], "->", best["end_time"])
        if unparsed:
            print("  warning: some ERA5 .nc files had no parseable time:")
            for path in unparsed:
                print(f"    unparsed: {path}")

    return selected
