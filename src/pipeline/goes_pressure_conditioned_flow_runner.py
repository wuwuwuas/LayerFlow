"""Retrieve centre-anchored GOES pressure-conditioned AMVs.

Estimate centre-to-previous and centre-to-next flow, convert pixel
displacements using surface and cloud-height geometry, and combine both
temporal components into pressure-assigned AMV products."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from pipeline.tracker_runner import save_stacked_piv_npz
from postprocess.combine_components import combine_components_to_stacked_arrays
from postprocess.goes_flow_to_wind import goes_flow_to_wind
from preprocessing import sample_at_centers
from preprocessing.goes import (
    GoesPcfMaskConfig,
    build_goes_center_height_mask,
    build_goes_center_output_mask,
    build_goes_solver_valid_mask,
)
from preprocessing.layerwise_image import LayerwiseImageConfig
from tracking.pressure_conditioned_flow import (
    PressureConditionedFlowConfig,
    run_pressure_conditioned_shared_source_components,
)


COMPONENT_DEFINITIONS = (
    ("C1", "backward", "center", "previous"),
    ("C2", "forward", "center", "next"),
)
GEOMETRY_MODES = ("surface", "cloud_height")


def _compact_json(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def _frame_mapping(frames_by_role: Mapping[str, Any]) -> dict[str, Any]:
    required = {"previous", "center", "next"}
    missing = sorted(required - set(frames_by_role))
    if missing:
        raise KeyError(f"GOES frame mapping is missing roles: {missing}")
    return {role: frames_by_role[role] for role in ("previous", "center", "next")}


def _validate_frames(frames: Mapping[str, Any]) -> tuple[int, int]:
    center = frames["center"]
    shape = np.asarray(center.brightness_temperature_k).shape
    if len(shape) != 2:
        raise ValueError(f"GOES images must be 2-D; got {shape}")
    satellite = str(center.satellite).upper()
    x0 = np.asarray(center.x_scan_angle)
    y0 = np.asarray(center.y_scan_angle)
    projection0 = dict(center.projection_attrs)
    for role, frame in frames.items():
        if str(frame.satellite).upper() != satellite:
            raise ValueError("All GOES frames must belong to one satellite")
        fields = (
            frame.brightness_temperature_k,
            frame.cloud_top_pressure_hpa,
            frame.cloud_top_height_m,
            frame.cloud_mask,
        )
        if any(np.asarray(value).shape != shape for value in fields):
            raise ValueError(f"GOES {role} fields do not share shape {shape}")
        if not np.array_equal(np.asarray(frame.x_scan_angle), x0):
            raise ValueError(f"GOES {role} x scan-angle grid differs from centre")
        if not np.array_equal(np.asarray(frame.y_scan_angle), y0):
            raise ValueError(f"GOES {role} y scan-angle grid differs from centre")
        if dict(frame.projection_attrs) != projection0:
            raise ValueError(f"GOES {role} projection differs from centre")
    if shape != (len(y0), len(x0)):
        raise ValueError("GOES image shape does not match y/x scan-angle lengths")
    return int(shape[0]), int(shape[1])


def _signed_dt_seconds(source: Any, target: Any) -> float:
    value = float(
        (pd.Timestamp(target.scan_midpoint) - pd.Timestamp(source.scan_midpoint))
        .total_seconds()
    )
    if not np.isfinite(value) or value == 0.0:
        raise ValueError(
            f"Invalid GOES signed time interval: {source.scan_midpoint} -> "
            f"{target.scan_midpoint}"
        )
    return value


def _sample_center_field(
    field: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    order: int,
) -> np.ndarray:
    return sample_at_centers(field, rows, cols, order=order)


def _sample_quality(
    value: np.ndarray | None,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    if value is None:
        return np.zeros(rows.shape, dtype=np.int16)
    sampled = _sample_center_field(value, rows, cols, order=0)
    return np.where(np.isfinite(sampled), sampled, -1).astype(np.int16)


def _attach_center_products(
    arrays: dict[str, Any],
    *,
    center: Any,
    center_mask: np.ndarray,
    geometry_mode: str,
) -> dict[str, Any]:
    rows = np.asarray(arrays["row_center_original"], dtype=np.float32)
    cols = np.asarray(arrays["col_center_original"], dtype=np.float32)
    pressure = _sample_center_field(
        center.cloud_top_pressure_hpa, rows, cols, order=1
    ).astype(np.float32)
    height = _sample_center_field(
        center.cloud_top_height_m, rows, cols, order=1
    ).astype(np.float32)
    cloud_sample = _sample_center_field(center.cloud_mask, rows, cols, order=0)
    cloud_mask = np.where(np.isfinite(cloud_sample), cloud_sample, -1).astype(
        np.int16
    )
    mask_sample = _sample_center_field(
        center_mask.astype(np.float32), rows, cols, order=0
    ) >= 0.5

    final_valid = (
        np.asarray(arrays["valid_mask"], dtype=bool)
        & mask_sample
        & np.isfinite(pressure)
    )
    if geometry_mode == "cloud_height":
        final_valid &= np.isfinite(height)

    arrays["valid_mask"] = final_valid
    arrays["u_ms"] = np.where(final_valid, arrays["u_ms"], np.nan).astype(
        np.float32
    )
    arrays["v_ms"] = np.where(final_valid, arrays["v_ms"], np.nan).astype(
        np.float32
    )
    arrays["pressure_hpa"] = np.where(
        final_valid, pressure, np.nan
    ).astype(np.float32)
    # Preserve centre CTH as a diagnostic even for the surface baseline.  It is
    # only a validity requirement for the cloud-height geometry.
    arrays["height_m"] = height
    arrays["cloud_mask"] = cloud_mask
    arrays["image_dqf"] = _sample_quality(center.image_dqf, rows, cols)
    arrays["height_dqf"] = _sample_quality(center.height_dqf, rows, cols)
    arrays["pressure_dqf"] = _sample_quality(center.pressure_dqf, rows, cols)
    arrays["cloud_mask_dqf"] = _sample_quality(
        center.cloud_mask_dqf, rows, cols
    )
    return arrays


def _save_component_npz(
    *,
    path: Path,
    flow: Any,
    winds: Mapping[str, Any],
    layer_id: np.ndarray,
    component_role: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {
        "row_center_original": flow.row_center_original,
        "col_center_original": flow.col_center_original,
        "dx_pixels": flow.dx_pixels,
        "dy_pixels": flow.dy_pixels,
        "flow_valid_mask": flow.valid_mask,
        "pressure_layer_id": np.asarray(layer_id, dtype=np.int16),
        "component_role": np.asarray([component_role]),
        "flow_metadata_json": np.asarray([_compact_json(flow.metadata)]),
    }
    for mode, wind in winds.items():
        output[f"longitude_{mode}"] = wind.longitude
        output[f"latitude_{mode}"] = wind.latitude
        output[f"u_{mode}_ms"] = wind.u_ms
        output[f"v_{mode}_ms"] = wind.v_ms
        output[f"valid_{mode}_mask"] = wind.valid_mask
        output[f"local_zenith_angle_{mode}_deg"] = wind.diagnostics[
            "local_zenith_angle_deg"
        ]
        output[f"wind_{mode}_metadata_json"] = np.asarray(
            [_compact_json(wind.metadata)]
        )
    np.savez(path, **output)
    return path


def _save_compact_goes_amv(
    *,
    arrays: Mapping[str, Any],
    center_frame: Any,
    path: Path,
    tracker_name: str,
    satellite: str,
    geometry_mode: str,
    metadata: Mapping[str, Any],
) -> tuple[Path, int]:
    valid = np.asarray(arrays["valid_mask"], dtype=bool).copy()
    pressure = np.asarray(arrays["pressure_hpa"], dtype=np.float32)
    height = np.asarray(arrays["height_m"], dtype=np.float32)
    longitude = np.asarray(arrays["longitude"], dtype=np.float32)
    latitude = np.asarray(arrays["latitude"], dtype=np.float32)
    u = np.asarray(arrays["u_ms"], dtype=np.float32)
    v = np.asarray(arrays["v_ms"], dtype=np.float32)
    valid &= (
        np.isfinite(pressure)
        & np.isfinite(longitude)
        & np.isfinite(latitude)
        & np.isfinite(u)
        & np.isfinite(v)
    )
    if geometry_mode == "cloud_height":
        valid &= np.isfinite(height)

    speed = np.hypot(u, v).astype(np.float32)
    direction = ((270.0 - np.degrees(np.arctan2(v, u))) % 360.0).astype(
        np.float32
    )
    output = {
        "row_center_original": np.asarray(arrays["row_center_original"])[valid],
        "col_center_original": np.asarray(arrays["col_center_original"])[valid],
        "longitude": longitude[valid],
        "latitude": latitude[valid],
        "u_ms": u[valid],
        "v_ms": v[valid],
        "speed_ms": speed[valid],
        "direction_deg": direction[valid],
        "pressure_hpa": pressure[valid],
        "height_m": height[valid],
        "valid_component_count": np.asarray(
            arrays["valid_component_count"], dtype=np.int16
        )[valid],
        "local_zenith_angle_deg": np.asarray(
            arrays["local_zenith_angle_deg"], dtype=np.float32
        )[valid],
        "cloud_mask": np.asarray(arrays["cloud_mask"])[valid],
        "image_dqf": np.asarray(arrays["image_dqf"])[valid],
        "height_dqf": np.asarray(arrays["height_dqf"])[valid],
        "pressure_dqf": np.asarray(arrays["pressure_dqf"])[valid],
        "cloud_mask_dqf": np.asarray(arrays["cloud_mask_dqf"])[valid],
        "pressure_source": np.asarray(
            ["goes_abi_cloud_top_pressure_center_frame"]
        ),
        "height_source": np.asarray(
            ["goes_abi_cloud_top_height_center_frame_diagnostic"]
        ),
        "geometry_mode": np.asarray([geometry_mode]),
        "satellite": np.asarray([satellite]),
        "observation_time": np.asarray(
            [pd.Timestamp(center_frame.scan_midpoint).isoformat()]
        ),
        "metadata_json": np.asarray(
            [
                _compact_json(
                    {
                        **dict(metadata),
                        "tracker": tracker_name,
                        "satellite": satellite,
                        "geometry_mode": geometry_mode,
                        "centre_scan_midpoint": str(center_frame.scan_midpoint),
                        "point_count": int(np.count_nonzero(valid)),
                    }
                )
            ]
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **output)
    return path, int(np.count_nonzero(valid))


def _save_diagnostic_figures(
    *,
    flows: Mapping[str, Any],
    final_by_geometry: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; GOES PCF diagnostic figures skipped")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for component in ("C1", "C2"):
        flow = flows[component]
        valid = np.asarray(flow.valid_mask, dtype=bool)
        pixel_speed = np.hypot(flow.dx_pixels, flow.dy_pixels)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for ax, values, title in (
            (axes[0, 0], np.where(valid, flow.dx_pixels, np.nan), "dx pixels"),
            (axes[0, 1], np.where(valid, flow.dy_pixels, np.nan), "dy pixels"),
            (
                axes[1, 0],
                np.where(valid, pixel_speed, np.nan),
                "pixel displacement magnitude",
            ),
            (axes[1, 1], valid.astype(np.float32), "centre-output flow mask"),
        ):
            image = ax.imshow(values, origin="upper")
            ax.set_title(title)
            ax.set_axis_off()
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{component.lower()}_flow_quicklook.png", dpi=140
        )
        plt.close(fig)

    for mode, arrays in final_by_geometry.items():
        final_valid = np.asarray(arrays["valid_mask"], dtype=bool)
        final_speed = np.hypot(arrays["u_ms"], arrays["v_ms"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        image = axes[0].imshow(
            np.where(final_valid, final_speed, np.nan), origin="upper"
        )
        axes[0].set_title(f"{mode} wind speed (m/s)")
        axes[0].set_axis_off()
        fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
        image = axes[1].imshow(
            np.asarray(arrays["valid_component_count"]), origin="upper"
        )
        axes[1].set_title("valid component count")
        axes[1].set_axis_off()
        fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / f"final_{mode}_wind_quicklook.png", dpi=140)
        plt.close(fig)


def run_goes_pressure_conditioned_flow_satellite(
    *,
    frames_by_role: Mapping[str, Any],
    tracker_root_dir: str | Path,
    pressure_output_dir: str | Path,
    tracker_name: str,
    layer_config: LayerwiseImageConfig | None = None,
    solver_config: PressureConditionedFlowConfig | Mapping[str, Any] | None = None,
    mask_config: GoesPcfMaskConfig | None = None,
    geometry_modes: tuple[str, ...] = GEOMETRY_MODES,
    geometry_chunk_rows: int = 256,
    save_native_npz: bool = True,
    save_component_npz: bool = False,
    save_pressure_product: bool = True,
    save_diagnostic_figures: bool = True,
    source_reuse_policy: str = "exact",
) -> dict[str, Any]:
    """Run two centre-anchored GOES PCF components for one satellite."""

    frames = _frame_mapping(frames_by_role)
    _validate_frames(frames)
    center = frames["center"]
    satellite = str(center.satellite).upper()
    cfg = mask_config or GoesPcfMaskConfig()
    modes = tuple(str(value).lower() for value in geometry_modes)
    invalid_modes = sorted(set(modes) - set(GEOMETRY_MODES))
    if invalid_modes:
        raise ValueError(f"Unsupported GOES geometry modes: {invalid_modes}")

    solver_masks = {
        role: build_goes_solver_valid_mask(frame, cfg)
        for role, frame in frames.items()
    }
    center_output_mask = build_goes_center_output_mask(center, cfg)
    center_height_mask = build_goes_center_height_mask(center, cfg)
    for role in ("previous", "center", "next"):
        print(
            f"[{satellite}] {role:8s}: "
            f"solver_support={int(np.count_nonzero(solver_masks[role])):,}; "
            f"time={frames[role].scan_midpoint}"
        )
    print(
        f"[{satellite}] centre output: "
        f"surface={int(np.count_nonzero(center_output_mask)):,}; "
        f"cloud_height={int(np.count_nonzero(center_height_mask)):,}"
    )

    if solver_config is None:
        solver = PressureConditionedFlowConfig()
    elif isinstance(solver_config, Mapping):
        solver = PressureConditionedFlowConfig(
            **{
                key: value
                for key, value in solver_config.items()
                if key in PressureConditionedFlowConfig.__dataclass_fields__
            }
        )
    else:
        solver = solver_config
    if layer_config is None:
        raise ValueError(
            "layer_config is required for GOES. Define the pressure layers in "
            "examples/run_layerflow_goes.py and pass them "
            "explicitly to the runner."
        )
    layers = layer_config

    tracker_dir = Path(tracker_root_dir) / tracker_name
    component_dir = tracker_dir / "components"
    final_dir = tracker_dir / "final"
    pressure_dir = Path(pressure_output_dir)
    component_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    pressure_dir.mkdir(parents=True, exist_ok=True)

    total_started = time.perf_counter()
    flows: dict[str, Any] = {}
    winds_by_geometry: dict[str, dict[str, Any]] = {
        mode: {} for mode in modes
    }
    layer_ids: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []

    print(
        f"\n[{tracker_name}] running shared-source sequential PCF: "
        f"policy={source_reuse_policy}"
    )
    shared_results, shared_diagnostics = (
        run_pressure_conditioned_shared_source_components(
            center.brightness_temperature_k,
            center.cloud_top_pressure_hpa,
            solver_masks["center"],
            target_images={
                "C1": frames["previous"].brightness_temperature_k,
                "C2": frames["next"].brightness_temperature_k,
            },
            target_pressures_hpa={
                "C1": frames["previous"].cloud_top_pressure_hpa,
                "C2": frames["next"].cloud_top_pressure_hpa,
            },
            target_valid_masks={
                "C1": solver_masks["previous"],
                "C2": solver_masks["next"],
            },
            output_valid_mask=center_output_mask,
            layer_config=layers,
            solver_config=solver,
            tracker_name=tracker_name,
            source_reuse_policy=source_reuse_policy,
        )
    )
    print(
        f"[{tracker_name}] shared PCF finished in "
        f"{float(shared_diagnostics['runtime_seconds'])/60.0:.2f} min; "
        f"source pyramids reused for "
        f"{shared_diagnostics['source_reused_state_count']}/"
        f"{shared_diagnostics['state_count']} states"
    )

    for component, temporal_role, source_role, target_role in COMPONENT_DEFINITIONS:
        source = frames[source_role]
        target = frames[target_role]
        dt_seconds = _signed_dt_seconds(source, target)
        print(
            f"\n[{tracker_name}] {component} {temporal_role}: "
            f"{source_role} -> {target_role}; dt={dt_seconds:+.3f} s"
        )
        started = time.perf_counter()
        flow, layer_id, diagnostics = shared_results[component]
        flow.metadata.update(
            {
                "component": component,
                "temporal_role": temporal_role,
                "source_role": source_role,
                "target_role": target_role,
                "source_time": str(source.scan_midpoint),
                "target_time": str(target.scan_midpoint),
                "dt_seconds_signed": dt_seconds,
                "output_validity": "centre_frame_only",
                "destination_product_validity_filter": False,
                "source_reuse_policy": source_reuse_policy,
            }
        )
        flows[component] = flow
        layer_ids[component] = np.asarray(layer_id, dtype=np.int16)

        geometry_seconds: dict[str, float] = {}
        component_winds: dict[str, Any] = {}
        for mode in modes:
            geometry_started = time.perf_counter()
            mode_mask = (
                center_output_mask if mode == "surface" else center_height_mask
            )
            wind = goes_flow_to_wind(
                flow,
                x_scan_angle=center.x_scan_angle,
                y_scan_angle=center.y_scan_angle,
                projection_attrs=center.projection_attrs,
                dt_seconds=dt_seconds,
                geometry_mode=mode,
                center_height_m=(
                    None if mode == "surface" else center.cloud_top_height_m
                ),
                output_valid_mask=mode_mask,
                chunk_rows=geometry_chunk_rows,
            )
            wind.metadata.update(flow.metadata)
            geometry_seconds[mode] = time.perf_counter() - geometry_started
            winds_by_geometry[mode][component] = wind
            component_winds[mode] = wind

        if save_component_npz:
            path = component_dir / f"{tracker_name}_{component}_native.npz"
            _save_component_npz(
                path=path,
                flow=flow,
                winds=component_winds,
                layer_id=layer_id,
                component_role=temporal_role,
            )
            print(f"[{tracker_name}] saved component: {path}")

        solver_diagnostics = diagnostics["solver"]
        postprocess_elapsed = time.perf_counter() - started
        summary_rows.append(
            {
                "record_type": "component",
                "component": component,
                "temporal_role": temporal_role,
                "source_role": source_role,
                "target_role": target_role,
                "dt_seconds_signed": dt_seconds,
                "runtime_seconds": float(solver_diagnostics["runtime_seconds"])
                + postprocess_elapsed,
                "component_preparation_seconds": diagnostics.get(
                    "component_preparation_seconds", np.nan
                ),
                "pyramid_build_seconds": float(
                    solver_diagnostics["pyramid_build_seconds"]
                ),
                "target_pyramid_build_seconds": float(
                    solver_diagnostics.get(
                        "target_pyramid_build_seconds", np.nan
                    )
                ),
                "solver_runtime_seconds": float(
                    solver_diagnostics["runtime_seconds"]
                ),
                "surface_geometry_seconds": geometry_seconds.get(
                    "surface", np.nan
                ),
                "cloud_height_geometry_seconds": geometry_seconds.get(
                    "cloud_height", np.nan
                ),
                "pyramid_level_count": int(
                    solver_diagnostics["pyramid_level_count"]
                ),
                "flow_valid_count": int(np.count_nonzero(flow.valid_mask)),
                "source_reuse_policy": diagnostics["source_reuse_policy"],
                "source_reused_state_count": int(
                    diagnostics["source_reused_state_count"]
                ),
                "private_opencv_threads": int(
                    solver_diagnostics["native_threads"]
                ),
                "private_opencv_version": solver_diagnostics[
                    "native_opencv_version"
                ],
                "state_preparation_json": _compact_json(
                    diagnostics.get("sequential_state_rows", [])
                ),
                "pyramid_state_json": _compact_json(
                    diagnostics.get("sequential_state_rows", [])
                ),
                "state_runtime_json": _compact_json(
                    solver_diagnostics["state_runtime_seconds"]
                ),
            }
        )
        print(
            f"[{tracker_name}] {component}: "
            f"flow_valid={int(np.count_nonzero(flow.valid_mask)):,}; "
            f"solver={float(solver_diagnostics['runtime_seconds'])/60.0:.2f} min; "
            f"postprocess={postprocess_elapsed/60.0:.2f} min"
        )
        gc.collect()

    final_by_geometry: dict[str, dict[str, Any]] = {}
    final_paths: dict[str, Path | None] = {}
    pressure_paths: dict[str, Path | None] = {}
    pressure_point_counts: dict[str, int] = {}

    for mode in modes:
        arrays = combine_components_to_stacked_arrays(
            flows,
            winds_by_geometry[mode],
            min_valid_components=2,
            component_order=("C1", "C2"),
            check_lonlat=True,
        )
        mode_mask = center_output_mask if mode == "surface" else center_height_mask
        arrays = _attach_center_products(
            arrays,
            center=center,
            center_mask=mode_mask,
            geometry_mode=mode,
        )
        arrays["local_zenith_angle_deg"] = np.asarray(
            winds_by_geometry[mode]["C1"].diagnostics[
                "local_zenith_angle_deg"
            ],
            dtype=np.float32,
        )
        for component, layer_id in layer_ids.items():
            arrays[f"pressure_layer_id_{component.lower()}"] = layer_id

        metadata = {
            **dict(arrays.get("metadata", {})),
            "tracker": tracker_name,
            "satellite": satellite,
            "method": "GOES centre-anchored two-component pressure-conditioned flow",
            "component_mapping": {
                "C1": "center -> previous (backward; signed dt < 0)",
                "C2": "center -> next (forward; signed dt > 0)",
            },
            "component_combination": "mean C1/C2; both components required",
            "minimum_valid_components": 2,
            "pressure_state_execution": "sequential_build_solve_release",
            "source_reuse_policy": source_reuse_policy,
            "source_reused_state_count": int(
                shared_diagnostics["source_reused_state_count"]
            ),
            "geometry_mode": mode,
            "destination_product_validity_filter": False,
            "final_pressure_source": "centre-frame GOES ABI CTP",
            "centre_scan_start": str(center.scan_start),
            "centre_scan_end": str(center.scan_end),
            "centre_scan_midpoint": str(center.scan_midpoint),
            "solver_parameters": solver.metadata(),
            "layerwise_config": layers.metadata(),
            "mask_config": cfg.__dict__,
        }
        arrays["metadata"] = metadata
        final_by_geometry[mode] = arrays

        final_path: Path | None = None
        if save_native_npz:
            final_path = (
                final_dir / f"{tracker_name}_{mode}_final_native.npz"
            )
            save_stacked_piv_npz(arrays, final_path)
            print(f"[{tracker_name}] saved {mode} native product: {final_path}")
        final_paths[mode] = final_path

        pressure_path: Path | None = None
        point_count = 0
        if save_pressure_product:
            pressure_path = (
                pressure_dir
                / f"{tracker_name}_{mode}_final_amv_goes_pressure.npz"
            )
            pressure_path, point_count = _save_compact_goes_amv(
                arrays=arrays,
                center_frame=center,
                path=pressure_path,
                tracker_name=tracker_name,
                satellite=satellite,
                geometry_mode=mode,
                metadata=metadata,
            )
            print(
                f"[{tracker_name}] saved {mode} GOES-pressure AMV: "
                f"{pressure_path}; points={point_count:,}"
            )
        pressure_paths[mode] = pressure_path
        pressure_point_counts[mode] = point_count

    if save_diagnostic_figures:
        _save_diagnostic_figures(
            flows=flows,
            final_by_geometry=final_by_geometry,
            output_dir=tracker_dir / "figures",
        )

    total_seconds = time.perf_counter() - total_started
    for mode in modes:
        summary_rows.append(
            {
                "record_type": "final",
                "component": "C1+C2",
                "temporal_role": mode,
                "source_role": "center",
                "target_role": "previous+next",
                "dt_seconds_signed": np.nan,
                "runtime_seconds": total_seconds,
                "component_preparation_seconds": np.nan,
                "pyramid_build_seconds": np.nan,
                "solver_runtime_seconds": np.nan,
                "surface_geometry_seconds": np.nan,
                "cloud_height_geometry_seconds": np.nan,
                "pyramid_level_count": np.nan,
                "flow_valid_count": int(
                    np.count_nonzero(final_by_geometry[mode]["valid_mask"])
                ),
                "private_opencv_threads": summary_rows[0][
                    "private_opencv_threads"
                ],
                "private_opencv_version": summary_rows[0][
                    "private_opencv_version"
                ],
                "state_preparation_json": "",
                "pyramid_state_json": "",
                "state_runtime_json": "",
            }
        )
    summary_path = tracker_dir / "run_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    return {
        "satellite": satellite,
        "flows": flows,
        "winds_by_geometry": winds_by_geometry,
        "final_by_geometry": final_by_geometry,
        "final_paths": final_paths,
        "pressure_paths": pressure_paths,
        "pressure_point_counts": pressure_point_counts,
        "summary_path": summary_path,
        "runtime_seconds": total_seconds,
    }
