"""Layerwise optical-flow tracker workflow.

This runner keeps the same final PIV-native output schema as
``pipeline.tracker_runner``.  Difference from raw mode:

    raw image pair -> tracker

becomes

    raw image pair + pressure grids
    -> build pressure-layer image pairs
    -> run the same tracker on each layer image pair
    -> merge layer flows into one dense flow
    -> flow_to_wind -> combine C1/C2/C3 -> save final native NPZ

No refinement, no TTO, no pressure assignment, no ERA5/IGRA comparison.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from models import FlowResult
from postprocess.flow_to_wind import flow_to_wind
from postprocess.combine_components import combine_components_to_stacked_arrays
from preprocessing.layerwise_image import (
    LayerwiseImageConfig,
    build_layer_image_pair,
    gaussian_background,
    merge_layer_flows,
    pressure_layer_id,
)
from pipeline.tracker_runner import component_name_from_pair, save_stacked_piv_npz


PRESSURE_PAIR_MODES = ("independent", "source_shared", "target_shared")


def _resolve_pressure_pair(
    frame_pressure_hpa: Sequence[np.ndarray],
    first_index: int,
    second_index: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Resolve pressure grids used to construct and merge one image pair.

    ``independent`` uses the pressure observed with each image.
    ``source_shared`` uses the source-frame pressure for both images.
    ``target_shared`` uses the target-frame pressure for both images.

    The returned third grid is used by ``merge_layer_flows`` so that layer
    selection is consistent with the pressure weights used for image0.
    """

    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in PRESSURE_PAIR_MODES:
        raise ValueError(
            f"pressure_pair_mode must be one of {PRESSURE_PAIR_MODES}, "
            f"got {mode!r}"
        )

    source_pressure = np.asarray(
        frame_pressure_hpa[first_index], dtype=np.float32
    )
    target_pressure = np.asarray(
        frame_pressure_hpa[second_index], dtype=np.float32
    )

    if resolved_mode == "independent":
        pressure0 = source_pressure
        pressure1 = target_pressure
        merge_pressure = source_pressure
        pressure0_index = first_index
        pressure1_index = second_index
        merge_pressure_index = first_index
    elif resolved_mode == "source_shared":
        pressure0 = source_pressure
        pressure1 = source_pressure
        merge_pressure = source_pressure
        pressure0_index = first_index
        pressure1_index = first_index
        merge_pressure_index = first_index
    else:  # target_shared
        pressure0 = target_pressure
        pressure1 = target_pressure
        merge_pressure = target_pressure
        pressure0_index = second_index
        pressure1_index = second_index
        merge_pressure_index = second_index

    mapping = {
        "mode": resolved_mode,
        "source_image_index": int(first_index),
        "target_image_index": int(second_index),
        "image0_pressure_index": int(pressure0_index),
        "image1_pressure_index": int(pressure1_index),
        "merge_pressure_index": int(merge_pressure_index),
        "mapping": (
            f"I{first_index}/P{pressure0_index} -> "
            f"I{second_index}/P{pressure1_index}"
        ),
    }
    return pressure0, pressure1, merge_pressure, mapping


def _metadata_json(obj: Any) -> np.ndarray:
    return np.array([json.dumps(obj or {}, default=str)])


def _validate_inputs(
    images: Sequence[np.ndarray],
    valid_masks: Sequence[np.ndarray] | None,
    frame_pressure_hpa: Sequence[np.ndarray],
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    component_pairs: Sequence[tuple[int, int]],
) -> None:
    if len(images) < 2:
        raise ValueError("Need at least two image frames")
    if len(frame_pressure_hpa) < len(images):
        raise ValueError("Need one pressure grid per image frame")
    shape = np.shape(images[0])
    if np.shape(lon_grid) != shape or np.shape(lat_grid) != shape:
        raise ValueError("lon_grid and lat_grid must match image shape")
    for k, image in enumerate(images):
        if np.shape(image) != shape:
            raise ValueError(f"Image {k} shape {np.shape(image)} does not match {shape}")
    for k, pressure in enumerate(frame_pressure_hpa[: len(images)]):
        if np.shape(pressure) != shape:
            raise ValueError(f"Pressure grid {k} shape {np.shape(pressure)} does not match image shape {shape}")
    if valid_masks is not None:
        if len(valid_masks) != len(images):
            raise ValueError("valid_masks length must match images length")
        for k, mask in enumerate(valid_masks):
            if np.shape(mask) != shape:
                raise ValueError(f"valid_mask {k} shape mismatch")
    for i0, i1 in component_pairs:
        if not (0 <= i0 < len(images) and 0 <= i1 < len(images) and i1 > i0):
            raise ValueError(f"Invalid component pair {(i0, i1)}")


def _dt_seconds(t0: Any, t1: Any, fallback_minutes: float) -> float:
    a = pd.Timestamp(t0)
    b = pd.Timestamp(t1)
    dt = float((b - a).total_seconds())
    if not np.isfinite(dt) or dt <= 0:
        dt = float(fallback_minutes) * 60.0
    return dt


def tracker_output_dirs(tracker_root_dir: str | Path, tracker_name: str) -> dict[str, Path]:
    root = Path(tracker_root_dir) / tracker_name
    dirs = {"root": root, "components": root / "components", "final": root / "final"}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def run_layerwise_tracker_components(
    *,
    tracker_name: str,
    tracker_func: Callable,
    tracker_params: Mapping[str, Any],
    images: Sequence[np.ndarray],
    valid_masks: Sequence[np.ndarray] | None,
    frame_pressure_hpa: Sequence[np.ndarray],
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    frame_times_loaded: Sequence[Any],
    frame_interval_minutes: float,
    tracker_root_dir: str | Path,
    component_pairs: Sequence[tuple[int, int]],
    layer_config: LayerwiseImageConfig = LayerwiseImageConfig(),
    pressure_pair_mode: str = "independent",
    min_valid_components: int = 2,
    save_native_npz: bool = True,
    save_component_npz: bool = False,
) -> dict[str, Any]:
    """Run one tracker through selected layerwise PIV components."""

    pressure_pair_mode = str(pressure_pair_mode).strip().lower()
    if pressure_pair_mode not in PRESSURE_PAIR_MODES:
        raise ValueError(
            f"pressure_pair_mode must be one of {PRESSURE_PAIR_MODES}, "
            f"got {pressure_pair_mode!r}"
        )

    _validate_inputs(images, valid_masks, frame_pressure_hpa, lon_grid, lat_grid, component_pairs)
    dirs = tracker_output_dirs(tracker_root_dir, tracker_name)
    component_flows: dict[str, FlowResult] = {}
    component_winds: dict[str, Any] = {}
    layer_id_by_component: dict[str, np.ndarray] = {}
    pressure_mapping_by_component: dict[str, dict[str, Any]] = {}
    layer_diagnostics: list[dict[str, Any]] = []

    print(f"\n========== Running layerwise tracker: {tracker_name} ==========")
    print("Output root:", dirs["root"])
    print("Image shapes:", [np.shape(img) for img in images])
    print("Layer mode:", layer_config.mode)
    print("Pressure pair mode:", pressure_pair_mode)
    print("Layers:", [(x.name, x.p_min_hpa, x.p_max_hpa) for x in layer_config.layers])
    print("Tracker params:", dict(tracker_params))

    total_t0 = time.perf_counter()
    for i0, i1 in component_pairs:
        cname = component_name_from_pair(i0, i1)
        print(f"\n[{tracker_name}] {cname}: frame {i0} -> {i1}")
        img0 = np.asarray(images[i0], dtype=np.float32)
        img1 = np.asarray(images[i1], dtype=np.float32)
        pressure0, pressure1, merge_pressure, pressure_mapping = _resolve_pressure_pair(
            frame_pressure_hpa,
            i0,
            i1,
            pressure_pair_mode,
        )
        pressure_mapping_by_component[cname] = pressure_mapping
        valid0 = np.isfinite(img0) if valid_masks is None else (np.asarray(valid_masks[i0], dtype=bool) & np.isfinite(img0))
        valid1 = np.isfinite(img1) if valid_masks is None else (np.asarray(valid_masks[i1], dtype=bool) & np.isfinite(img1))
        print(f"[{tracker_name}] {cname}: {pressure_mapping['mapping']}")
        print(
            f"[{tracker_name}] {cname}: finite pressure pixels "
            f"image0={int(np.isfinite(pressure0).sum()):,}, "
            f"image1={int(np.isfinite(pressure1).sum()):,}"
        )

        bg0 = gaussian_background(img0, layer_config.background_sigma_pixels)
        bg1 = gaussian_background(img1, layer_config.background_sigma_pixels)

        layer_flows: list[FlowResult] = []
        layer_weights0: list[np.ndarray] = []
        for layer_index, layer in enumerate(layer_config.layers):
            layer_pair = build_layer_image_pair(
                img0,
                img1,
                pressure0,
                pressure1,
                layer,
                layer_config,
                valid0=valid0,
                valid1=valid1,
                background0=bg0,
                background1=bg1,
            )
            params = dict(tracker_params)
            params["valid_mask"] = layer_pair["valid_mask"]
            if bool(getattr(tracker_func, "requires_nn_context", False)):
                params["_nn_context"] = {
                    "image_mode": "layerwise",
                    "component_name": cname,
                    "component_index": int(i0),
                    "component_seed": int(params.get("seed", 0)) + int(i0) * 10000,
                    "layer_index": int(layer_index),
                    "layer_name": layer.name,
                    "layer_weight0": layer_pair["weight0"],
                    "layer_weight1": layer_pair["weight1"],
                    "source_valid_mask": valid0,
                    "target_valid_mask": valid1,
                    "layer_normalization_dilation_pixels": int(
                        params.get("layer_normalization_dilation_pixels", 16)
                    ),
                }
            t0 = time.perf_counter()
            flow = tracker_func(layer_pair["image0"], layer_pair["image1"], params)
            runtime = time.perf_counter() - t0
            # Force layer-specific validity; dx/dy remain dense arrays.
            flow.valid_mask[:] = flow.valid_mask & layer_pair["valid_mask"]
            flow.dx_pixels[~flow.valid_mask] = np.nan
            flow.dy_pixels[~flow.valid_mask] = np.nan
            flow.metadata.update(
                {
                    "image_mode": "layerwise",
                    "pressure_pair_mode": pressure_pair_mode,
                    "pressure_pair_mapping": dict(pressure_mapping),
                    "layer": layer_pair["metadata"],
                }
            )
            layer_flows.append(flow)
            layer_weights0.append(layer_pair["weight0"])
            layer_diagnostics.append(
                {
                    "component": cname,
                    "layer": layer.name,
                    "pressure_pair_mode": pressure_pair_mode,
                    "pressure_pair_mapping": dict(pressure_mapping),
                    "valid_pixels": int(flow.valid_mask.sum()),
                    "runtime_tracker_sec": float(runtime),
                    "tracker_metadata": dict(flow.metadata),
                }
            )
            print(f"[{tracker_name}] {cname} {layer.name}: tracker finished in {runtime:.1f} s; valid={int(flow.valid_mask.sum()):,}")

        flow, lid = merge_layer_flows(
            layer_flows,
            layer_weights0,
            merge_pressure,
            layer_config,
            base_valid0=valid0,
            tracker_name=tracker_name,
            metadata={
                "tracker_parameters": dict(tracker_params),
                "pressure_pair_mode": pressure_pair_mode,
                "pressure_pair_mapping": dict(pressure_mapping),
            },
        )
        print(f"[{tracker_name}] {cname}: merged valid flow pixels = {int(flow.valid_mask.sum()):,}")

        dt = _dt_seconds(frame_times_loaded[i0], frame_times_loaded[i1], frame_interval_minutes)
        t0 = time.perf_counter()
        wind = flow_to_wind(flow, lon_grid, lat_grid, dt)
        print(f"[{tracker_name}] {cname}: flow_to_wind finished in {time.perf_counter() - t0:.1f} s")
        component_flows[cname] = flow
        component_winds[cname] = wind
        layer_id_by_component[cname] = lid

        if save_component_npz and save_native_npz:
            component_path = dirs["components"] / f"{tracker_name}_{cname}_native.npz"
            np.savez_compressed(
                component_path,
                row_center_original=wind.row_center_original,
                col_center_original=wind.col_center_original,
                dx_pixels=flow.dx_pixels,
                dy_pixels=flow.dy_pixels,
                longitude=wind.longitude,
                latitude=wind.latitude,
                u_ms=wind.u_ms,
                v_ms=wind.v_ms,
                valid_mask=wind.valid_mask,
                pressure_layer_id=lid,
                flow_metadata_json=_metadata_json(flow.metadata),
                wind_metadata_json=_metadata_json(getattr(wind, "metadata", {})),
            )
            print(f"[{tracker_name}] {cname}: saved component native PIV result {component_path}")

    required = ["C1", "C2", "C3"]
    if not all(name in component_winds for name in required):
        print(f"[{tracker_name}] Not all C1/C2/C3 are available. Skipping final combine.")
        return {
            "components": component_winds,
            "flows": component_flows,
            "final": None,
            "output_dirs": dirs,
            "layer_diagnostics": layer_diagnostics,
        }

    t0 = time.perf_counter()
    final_arrays = combine_components_to_stacked_arrays(
        component_flows,
        component_winds,
        min_valid_components=min_valid_components,
        component_order=("C1", "C2", "C3"),
        check_lonlat=True,
    )
    print(f"[{tracker_name}] combined components in {time.perf_counter() - t0:.1f} s")

    for cname, lid in layer_id_by_component.items():
        final_arrays[f"pressure_layer_id_{cname.lower()}"] = lid.astype(np.int16)
    final_arrays["metadata"] = {
        **final_arrays.get("metadata", {}),
        "tracker": tracker_name,
        "image_mode": "layerwise",
        "tracker_parameters": dict(tracker_params),
        "layerwise_config": layer_config.metadata(),
        "pressure_pair_mode": pressure_pair_mode,
        "pressure_pair_mapping": pressure_mapping_by_component,
        "component_start_pressure_assignment": {
            component_name_from_pair(i0, i1): f"P{i0}"
            for i0, i1 in component_pairs
        },
        "layer_diagnostics": layer_diagnostics,
    }

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
        "layer_diagnostics": layer_diagnostics,
    }


def run_requested_layerwise_trackers(
    *,
    tracker_configs: Mapping[str, tuple[Callable, Mapping[str, Any]]],
    trackers_to_run: list[str],
    **kwargs,
) -> dict[str, Any]:
    """Run requested layerwise trackers one by one."""

    results = {}
    for tracker_name in trackers_to_run:
        if tracker_name not in tracker_configs:
            raise ValueError(f"Unknown tracker: {tracker_name}")
        tracker_func, tracker_params = tracker_configs[tracker_name]
        results[tracker_name] = run_layerwise_tracker_components(
            tracker_name=tracker_name,
            tracker_func=tracker_func,
            tracker_params=tracker_params,
            **kwargs,
        )
        print(f"\nFinished layerwise tracker: {tracker_name}")
    return results
