"""Retrieve SEVIRI pressure-conditioned AMVs.

Estimate C1/C2/C3, convert pixel displacements to wind, combine valid
components, and save dense native and compact OCA-pressure-assigned products."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from pipeline.tracker_runner import component_name_from_pair, save_stacked_piv_npz
from postprocess.combine_components import combine_components_to_stacked_arrays
from postprocess.flow_to_wind import flow_to_wind
from postprocess.height_assignment import assign_piv_pressure_from_component_grids
from preprocessing.layerwise_image import LayerwiseImageConfig
from tracking.pressure_conditioned_flow import (
    PressureConditionedFlowConfig,
    default_pressure_layer_config,
    run_pressure_conditioned_component,
)


DEFAULT_TRACKER_NAME = "pressure_conditioned_flow"
DEFAULT_COMPONENT_PAIRS = ((0, 1), (1, 2), (2, 3))


def pressure_to_hpa(pressure: np.ndarray) -> np.ndarray:
    """Convert a pressure grid to hPa and preserve NaNs."""

    output = np.asarray(pressure, dtype=np.float32)
    finite = output[np.isfinite(output)]
    if finite.size and float(np.nanmedian(finite)) > 2000.0:
        output = output / 100.0
    return output.astype(np.float32, copy=False)


def _dt_seconds(first: Any, second: Any, fallback_minutes: float) -> float:
    try:
        value = float(
            (pd.Timestamp(second) - pd.Timestamp(first)).total_seconds()
        )
    except Exception:
        value = float("nan")
    if not np.isfinite(value) or value <= 0.0:
        value = float(fallback_minutes) * 60.0
    return value


def _compact_json(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def _validate_inputs(
    images: np.ndarray,
    valid_masks: np.ndarray,
    pressure_hpa: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    frame_times: Sequence[Any],
) -> None:
    if images.ndim != 3 or images.shape[0] < 4:
        raise ValueError(f"images must be [T,H,W] with T>=4, got {images.shape}")
    if valid_masks.shape != images.shape:
        raise ValueError(
            f"valid_masks shape {valid_masks.shape} does not match images {images.shape}"
        )
    if pressure_hpa.shape != images.shape:
        raise ValueError(
            f"pressure_hpa shape {pressure_hpa.shape} does not match images {images.shape}"
        )
    if lon_grid.shape != images.shape[1:] or lat_grid.shape != images.shape[1:]:
        raise ValueError("lon_grid/lat_grid must match the image HxW shape")
    if len(frame_times) < 4:
        raise ValueError("At least four frame times are required")


def run_pressure_conditioned_flow_case(
    *,
    images: np.ndarray | Sequence[np.ndarray],
    valid_masks: np.ndarray | Sequence[np.ndarray],
    pressure_hpa: np.ndarray | Sequence[np.ndarray],
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    frame_times: Sequence[Any],
    tracker_root_dir: str | Path,
    pressure_output_dir: str | Path,
    cloud_probability: np.ndarray | Sequence[np.ndarray] | None = None,
    tracker_name: str = DEFAULT_TRACKER_NAME,
    component_pairs: Sequence[tuple[int, int]] = DEFAULT_COMPONENT_PAIRS,
    layer_config: LayerwiseImageConfig | None = None,
    solver_config: PressureConditionedFlowConfig | Mapping[str, Any] | None = None,
    frame_interval_minutes: float = 15.0,
    min_valid_components: int = 2,
    min_valid_pressure_components: int = 2,
    include_height_m: bool = False,
    save_native_npz: bool = True,
    save_component_npz: bool = False,
    save_pressure_product: bool = True,
    pressure_component_source: str = "provided_pressure_frame_start_grids",
) -> dict[str, Any]:
    """Run the validated pressure-conditioned tracker and pressure assignment.

    Parameters use project-standard dense arrays. The first four frames are used
    as I0..I3 and P0..P3. Final pressure assignment uses P0/P1/P2 for C1/C2/C3,
    exactly matching ``postprocess.height_assignment``.
    """

    images_array = np.stack(
        [np.asarray(value, dtype=np.float32) for value in images], axis=0
    )
    valid_array = np.stack(
        [np.asarray(value, dtype=bool) for value in valid_masks], axis=0
    )
    pressure_array = np.stack(
        [pressure_to_hpa(value) for value in pressure_hpa], axis=0
    )
    lon = np.asarray(lon_grid)
    lat = np.asarray(lat_grid)
    times = list(frame_times)

    _validate_inputs(
        images_array,
        valid_array,
        pressure_array,
        lon,
        lat,
        times,
    )

    cloud_array: np.ndarray | None = None
    if cloud_probability is not None:
        cloud_array = np.stack(
            [np.asarray(value, dtype=np.float32) for value in cloud_probability],
            axis=0,
        )
        if cloud_array.shape != images_array.shape:
            raise ValueError(
                "cloud_probability must have the same [T,H,W] shape as images"
            )

    layer_config = layer_config or default_pressure_layer_config()
    if solver_config is None:
        solver_config = PressureConditionedFlowConfig()
    elif isinstance(solver_config, Mapping):
        valid_options = {
            key: value
            for key, value in solver_config.items()
            if key in PressureConditionedFlowConfig.__dataclass_fields__
        }
        solver_config = PressureConditionedFlowConfig(**valid_options)

    tracker_parent = Path(tracker_root_dir)
    tracker_dir = tracker_parent / tracker_name
    component_dir = tracker_dir / "components"
    final_dir = tracker_dir / "final"
    pressure_dir = Path(pressure_output_dir)
    component_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    pressure_dir.mkdir(parents=True, exist_ok=True)

    total_started = time.perf_counter()
    component_flows: dict[str, Any] = {}
    component_winds: dict[str, Any] = {}
    layer_id_by_component: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []

    for first_index, second_index in component_pairs:
        component_name = component_name_from_pair(first_index, second_index)
        component_started = time.perf_counter()
        print(
            f"\n[{tracker_name}] {component_name}: "
            f"I{first_index}/P{first_index} -> I{second_index}/P{second_index}"
        )

        image0 = images_array[first_index]
        image1 = images_array[second_index]
        valid0 = valid_array[first_index] & np.isfinite(image0)
        valid1 = valid_array[second_index] & np.isfinite(image1)

        flow, layer_id, diagnostics = run_pressure_conditioned_component(
            image0,
            image1,
            pressure_array[first_index],
            pressure_array[second_index],
            valid0,
            valid1,
            layer_config=layer_config,
            solver_config=solver_config,
            tracker_name=tracker_name,
        )

        dt_seconds = _dt_seconds(
            times[first_index],
            times[second_index],
            frame_interval_minutes,
        )
        wind_started = time.perf_counter()
        wind = flow_to_wind(flow, lon, lat, dt_seconds)
        wind_seconds = time.perf_counter() - wind_started

        component_flows[component_name] = flow
        component_winds[component_name] = wind
        layer_id_by_component[component_name] = np.asarray(
            layer_id, dtype=np.int16
        )

        if save_component_npz:
            component_path = (
                component_dir / f"{tracker_name}_{component_name}_native.npz"
            )
            np.savez(
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
                pressure_layer_id=layer_id_by_component[component_name],
                flow_metadata_json=np.asarray(
                    [_compact_json(getattr(flow, "metadata", {}))]
                ),
                wind_metadata_json=np.asarray(
                    [_compact_json(getattr(wind, "metadata", {}))]
                ),
            )
            print(f"[{tracker_name}] saved component: {component_path}")

        solver_diagnostics = diagnostics["solver"]
        component_seconds = time.perf_counter() - component_started
        summary_rows.append(
            {
                "record_type": "component",
                "component": component_name,
                "runtime_seconds": component_seconds,
                "solver_runtime_seconds": float(
                    solver_diagnostics["runtime_seconds"]
                ),
                "pyramid_build_seconds": float(
                    solver_diagnostics["pyramid_build_seconds"]
                ),
                "flow_to_wind_seconds": wind_seconds,
                "pyramid_level_count": int(
                    solver_diagnostics["pyramid_level_count"]
                ),
                "valid_count": int(flow.valid_mask.sum()),
                "state_runtime_json": _compact_json(
                    solver_diagnostics["state_runtime_seconds"]
                ),
                "private_opencv_threads": int(
                    solver_diagnostics["native_threads"]
                ),
                "private_opencv_version": solver_diagnostics[
                    "native_opencv_version"
                ],
            }
        )
        print(
            f"[{tracker_name}] {component_name}: "
            f"valid={int(flow.valid_mask.sum()):,}; "
            f"solver={solver_diagnostics['runtime_seconds']/60.0:.2f} min; "
            f"total={component_seconds/60.0:.2f} min"
        )
        gc.collect()

    required_components = ("C1", "C2", "C3")
    if not all(name in component_winds for name in required_components):
        total_seconds = time.perf_counter() - total_started
        print(
            "Not all C1/C2/C3 components were run; final and pressure products "
            "were not created."
        )
        return {
            "components": component_winds,
            "flows": component_flows,
            "final": None,
            "final_path": None,
            "pressure_product": None,
            "pressure_path": None,
            "runtime_seconds": total_seconds,
        }

    final_arrays = combine_components_to_stacked_arrays(
        component_flows,
        component_winds,
        min_valid_components=min_valid_components,
        component_order=required_components,
        check_lonlat=True,
    )
    for name, layer_id in layer_id_by_component.items():
        final_arrays[f"pressure_layer_id_{name.lower()}"] = layer_id

    final_arrays["metadata"] = {
        **final_arrays.get("metadata", {}),
        "tracker": tracker_name,
        "method": (
            "four-state pressure-conditioned dense optical flow with "
            "source/target membership and normalized masked pyramids"
        ),
        "legacy_implementation": "pressure_state_opencv_masked_v1",
        "reference_case": "test_2015-06-12T11",
        "reference_mvd_ms": 4.580368,
        "solver_parameters": solver_config.metadata(),
        "layerwise_config": layer_config.metadata(),
        "pressure_mapping": {
            "C1": "I0/P0 -> I1/P1",
            "C2": "I1/P1 -> I2/P2",
            "C3": "I2/P2 -> I3/P3",
        },
        "pressure_guided_diffusion": False,
        "explicit_pressure_state_images": True,
        "final_pressure_product_written_separately": bool(
            save_pressure_product
        ),
        "pressure_component_source": str(pressure_component_source),
    }

    final_path: Path | None = None
    if save_native_npz:
        final_path = final_dir / f"{tracker_name}_final_native.npz"
        save_stacked_piv_npz(final_arrays, final_path)
        print(f"[{tracker_name}] saved final native product: {final_path}")

    pressure_result: dict[str, Any] | None = None
    pressure_path: Path | None = None
    if save_pressure_product:
        pressure_path = (
            pressure_dir / f"{tracker_name}_final_amv_oca_pressure.npz"
        )
        pressure_result = assign_piv_pressure_from_component_grids(
            piv=final_arrays,
            pressure_components=(
                pressure_array[0],
                pressure_array[1],
                pressure_array[2],
            ),
            cloud_components=(
                None
                if cloud_array is None
                else cloud_array[0],
                None
                if cloud_array is None
                else cloud_array[1],
                None
                if cloud_array is None
                else cloud_array[2],
            ),
            output_path=pressure_path,
            pressure_source="oca",
            min_valid_pressure_components=min_valid_pressure_components,
            oca_assignment_mode="pixel_direct",
            oca_window_size=3,
            oca_window_statistic="median",
            include_height_m=include_height_m,
            compress=True,
            component_source=str(pressure_component_source),
        )
        print(
            f"[{tracker_name}] saved OCA-pressure product: {pressure_path}"
        )
        print(
            f"[{tracker_name}] pressure-assigned points: "
            f"{pressure_result['pressure_hpa'].size:,}"
        )

    total_seconds = time.perf_counter() - total_started
    summary_rows.append(
        {
            "record_type": "total",
            "component": "ALL",
            "runtime_seconds": total_seconds,
            "solver_runtime_seconds": np.nan,
            "pyramid_build_seconds": np.nan,
            "flow_to_wind_seconds": np.nan,
            "pyramid_level_count": np.nan,
            "valid_count": int(np.asarray(final_arrays["valid_mask"]).sum()),
            "state_runtime_json": "",
            "private_opencv_threads": summary_rows[0][
                "private_opencv_threads"
            ],
            "private_opencv_version": summary_rows[0][
                "private_opencv_version"
            ],
        }
    )
    summary_path = tracker_dir / "run_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    return {
        "components": component_winds,
        "flows": component_flows,
        "final": final_arrays,
        "final_path": final_path,
        "pressure_product": pressure_result,
        "pressure_path": pressure_path,
        "summary_path": summary_path,
        "runtime_seconds": total_seconds,
    }
