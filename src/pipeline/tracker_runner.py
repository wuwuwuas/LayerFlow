"""Core PIV / optical-flow tracker workflow for notebooks and scripts.

One-file PIV-native workflow:
tracker -> flow_to_wind -> combine C1/C2/C3 u/v -> save final native NPZ.

This runner is intentionally PIV-only:
- no OCA / cloud-top-pressure assignment
- no height assignment
- no QC or diagnostic score calculation
- no CSV preview files
- component NPZ files are optional; by default all component u/v and dx/dy are
  stored in the final native NPZ.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from postprocess.flow_to_wind import flow_to_wind
from postprocess.combine_components import combine_components_to_stacked_arrays


def _first_attr(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _metadata_json(obj: Any) -> np.ndarray:
    return np.array([json.dumps(obj or {}, default=str)])


def maybe_normalize_for_tracker(
    image0: np.ndarray,
    image1: np.ndarray,
    valid0: np.ndarray,
    valid1: np.ndarray,
    tracker_name: str,
    normalize_image_pair: Callable | None = None,
):
    """Normalize a pair only when the notebook passes a normalizer."""

    if normalize_image_pair is None:
        return image0, image1

    output_dtype = "uint8" if str(tracker_name).lower().startswith("farneback") else "float32"

    try:
        return normalize_image_pair(
            image0,
            image1,
            valid0=valid0,
            valid1=valid1,
            output_dtype=output_dtype,
        )
    except TypeError:
        return normalize_image_pair(image0, image1)


def tracker_output_dirs(tracker_root_dir: str | Path, tracker_name: str) -> dict[str, Path]:
    root = Path(tracker_root_dir) / tracker_name
    dirs = {
        "root": root,
        "components": root / "components",
        "final": root / "final",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def component_name_from_pair(i0: int, i1: int) -> str:
    if (i0, i1) == (0, 1):
        return "C1"
    if (i0, i1) == (1, 2):
        return "C2"
    if (i0, i1) == (2, 3):
        return "C3"
    return f"F{i0}_F{i1}"


def save_stacked_piv_npz(arrays: Mapping[str, Any], path: str | Path) -> Path:
    """Save final PIV-native arrays.

    The saved file contains final u/v plus C1/C2/C3 component u/v and dx/dy.
    Speed and direction are intentionally omitted because they are exactly
    derived from u/v.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {}
    for key, value in arrays.items():
        if key == "metadata":
            continue
        if value is not None:
            output[key] = np.asarray(value)
    output["metadata_json"] = _metadata_json(arrays.get("metadata", {}))
    np.savez(path, **output)
    return path


def _validate_inputs(
    images: list[np.ndarray],
    valid_masks: list[np.ndarray] | None,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    component_pairs: list[tuple[int, int]],
) -> None:
    if len(images) < 2:
        raise ValueError("Need at least two SEVIRI frames.")
    if np.shape(lon_grid) != np.shape(lat_grid):
        raise ValueError(f"lon/lat shape mismatch: {np.shape(lon_grid)} vs {np.shape(lat_grid)}")
    for k, image in enumerate(images):
        if np.shape(image) != np.shape(lon_grid):
            raise ValueError(
                f"Image {k} shape {np.shape(image)} does not match lon/lat shape {np.shape(lon_grid)}"
            )
    if valid_masks is not None:
        if len(valid_masks) != len(images):
            raise ValueError("valid_masks length must match images length")
        for k, mask in enumerate(valid_masks):
            if np.shape(mask) != np.shape(images[k]):
                raise ValueError(
                    f"valid_mask {k} shape {np.shape(mask)} does not match image shape {np.shape(images[k])}"
                )
    n_frames = len(images)
    for i0, i1 in component_pairs:
        if not (0 <= i0 < n_frames and 0 <= i1 < n_frames):
            raise ValueError(f"Invalid component pair {(i0, i1)} for {n_frames} frames")
        if i1 <= i0:
            raise ValueError(f"Component pair must be forward in time, got {(i0, i1)}")


def run_tracker_components(
    *,
    tracker_name: str,
    tracker_func: Callable,
    tracker_params: Mapping[str, Any],
    images: list[np.ndarray],
    valid_masks: list[np.ndarray] | None,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    frame_times_loaded: list[Any],
    frame_interval_minutes: float,
    tracker_root_dir: str | Path,
    component_pairs: list[tuple[int, int]],
    normalize_image_pair: Callable | None = None,
    min_valid_components: int = 2,
    save_native_npz: bool = True,
    save_component_npz: bool = False,
) -> dict[str, Any]:
    """Run one tracker through selected PIV components."""

    _validate_inputs(images, valid_masks, lon_grid, lat_grid, component_pairs)

    dirs = tracker_output_dirs(tracker_root_dir, tracker_name)
    component_winds: dict[str, Any] = {}
    component_flows: dict[str, Any] = {}

    print(f"\n========== Running tracker: {tracker_name} ==========")
    print("Output root:", dirs["root"])
    print("Image shapes:", [np.shape(img) for img in images])
    print("Lon/lat shape:", np.shape(lon_grid), np.shape(lat_grid))
    print("Params:", dict(tracker_params))

    total_t0 = time.perf_counter()

    for i0, i1 in component_pairs:
        cname = component_name_from_pair(i0, i1)
        print(f"\n[{tracker_name}] {cname}: frame {i0} -> {i1}")

        img0, img1 = images[i0], images[i1]
        valid0 = valid_masks[i0] if valid_masks else np.isfinite(img0)
        valid1 = valid_masks[i1] if valid_masks else np.isfinite(img1)
        valid_pair = valid0 & valid1
        print(f"[{tracker_name}] {cname}: valid pixels = {int(np.sum(valid_pair)):,}")

        t0 = time.perf_counter()
        img0_in, img1_in = maybe_normalize_for_tracker(
            img0,
            img1,
            valid0,
            valid1,
            tracker_name,
            normalize_image_pair=normalize_image_pair,
        )
        print(f"[{tracker_name}] {cname}: normalized in {time.perf_counter() - t0:.1f} s")

        params = dict(tracker_params)
        params["valid_mask"] = valid_pair
        if bool(getattr(tracker_func, "requires_nn_context", False)):
            params["_nn_context"] = {
                "image_mode": "original",
                "component_name": cname,
                "component_index": int(i0),
                "component_seed": int(params.get("seed", 0)) + int(i0) * 10000,
            }

        t0 = time.perf_counter()
        flow = tracker_func(img0_in, img1_in, params)
        print(f"[{tracker_name}] {cname}: tracker finished in {time.perf_counter() - t0:.1f} s")

        t0_frame = pd.Timestamp(frame_times_loaded[i0])
        t1_frame = pd.Timestamp(frame_times_loaded[i1])
        dt_seconds = float((t1_frame - t0_frame).total_seconds())
        if not np.isfinite(dt_seconds) or dt_seconds <= 0:
            dt_seconds = float(frame_interval_minutes) * 60.0

        t0 = time.perf_counter()
        wind = flow_to_wind(flow, lon_grid, lat_grid, dt_seconds)
        print(f"[{tracker_name}] {cname}: flow_to_wind finished in {time.perf_counter() - t0:.1f} s")

        component_flows[cname] = flow
        component_winds[cname] = wind

        if save_component_npz and save_native_npz:
            component_path = dirs["components"] / f"{tracker_name}_{cname}_native.npz"
            arrays = {
                "row_center_original": wind.row_center_original,
                "col_center_original": wind.col_center_original,
                "dx_pixels": flow.dx_pixels,
                "dy_pixels": flow.dy_pixels,
                "longitude": wind.longitude,
                "latitude": wind.latitude,
                "u_ms": wind.u_ms,
                "v_ms": wind.v_ms,
                "valid_mask": wind.valid_mask,
                "flow_metadata_json": _metadata_json(getattr(flow, "metadata", {})),
                "wind_metadata_json": _metadata_json(getattr(wind, "metadata", {})),
            }
            np.savez(component_path, **arrays)
            print(f"[{tracker_name}] {cname}: saved component native PIV result {component_path}")

    required = ["C1", "C2", "C3"]
    if not all(name in component_winds for name in required):
        print(f"[{tracker_name}] Not all C1/C2/C3 are available. Skipping final combine.")
        print(f"[{tracker_name}] total runtime: {(time.perf_counter() - total_t0) / 60:.1f} min")
        return {"components": component_winds, "flows": component_flows, "final": None, "output_dirs": dirs}

    t0 = time.perf_counter()
    final_arrays = combine_components_to_stacked_arrays(
        component_flows,
        component_winds,
        min_valid_components=min_valid_components,
        component_order=("C1", "C2", "C3"),
        check_lonlat=True,
    )
    print(
        f"[{tracker_name}] combine_components_to_stacked_arrays mean_valid_u_v finished "
        f"in {time.perf_counter() - t0:.1f} s"
    )

    final_path = None
    if save_native_npz:
        final_path = dirs["final"] / f"{tracker_name}_final_native.npz"
        save_stacked_piv_npz(final_arrays, final_path)
        print(f"[{tracker_name}] saved final native PIV result {final_path}")

    print(f"[{tracker_name}] total runtime: {(time.perf_counter() - total_t0) / 60:.1f} min")
    return {
        "components": component_winds,
        "flows": component_flows,
        "final": final_arrays,
        "final_path": final_path,
        "output_dirs": dirs,
    }


def run_requested_trackers(
    *,
    tracker_configs: Mapping[str, tuple[Callable, Mapping[str, Any]]],
    trackers_to_run: list[str],
    **kwargs,
) -> dict[str, Any]:
    """Run requested trackers one by one."""

    results = {}
    for tracker_name in trackers_to_run:
        if tracker_name not in tracker_configs:
            raise ValueError(f"Unknown tracker: {tracker_name}")
        tracker_func, tracker_params = tracker_configs[tracker_name]
        results[tracker_name] = run_tracker_components(
            tracker_name=tracker_name,
            tracker_func=tracker_func,
            tracker_params=tracker_params,
            **kwargs,
        )
        print(f"\nFinished tracker: {tracker_name}")
    return results
