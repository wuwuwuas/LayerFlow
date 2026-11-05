"""GOES pressure-layer runner for existing traditional dense trackers."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from postprocess.goes_flow_to_wind import goes_flow_to_wind
from preprocessing.goes import GoesPcfMaskConfig
from preprocessing.layerwise_image import (
    LayerwiseImageConfig,
    build_layer_image_pair,
    gaussian_background,
    merge_layer_flows,
)
from pipeline.goes_traditional_common import (
    COMPONENT_DEFINITIONS,
    GEOMETRY_MODES,
    compact_json,
    finalize_components,
    frame_mapping,
    prepare_goes_masks,
    save_component_npz as save_component_file,
    signed_dt_seconds,
    validate_frames,
)


PRESSURE_PAIR_MODES = ("independent", "source_shared", "target_shared")


def _resolve_pressure_pair(
    frames: Mapping[str, Any],
    *,
    source_role: str,
    target_role: str,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    resolved = str(mode).strip().lower()
    if resolved not in PRESSURE_PAIR_MODES:
        raise ValueError(
            f"pressure_pair_mode must be one of {PRESSURE_PAIR_MODES}, got {mode!r}"
        )
    source_pressure = np.asarray(
        frames[source_role].cloud_top_pressure_hpa, dtype=np.float32
    )
    target_pressure = np.asarray(
        frames[target_role].cloud_top_pressure_hpa, dtype=np.float32
    )
    if resolved == "independent":
        pressure0 = source_pressure
        pressure1 = target_pressure
        merge_pressure = source_pressure
        mapping = f"{source_role}/P{source_role} -> {target_role}/P{target_role}"
    elif resolved == "source_shared":
        pressure0 = source_pressure
        pressure1 = source_pressure
        merge_pressure = source_pressure
        mapping = f"{source_role}/P{source_role} -> {target_role}/P{source_role}"
    else:
        pressure0 = target_pressure
        pressure1 = target_pressure
        merge_pressure = target_pressure
        mapping = f"{source_role}/P{target_role} -> {target_role}/P{target_role}"
    return pressure0, pressure1, merge_pressure, {
        "mode": resolved,
        "source_role": source_role,
        "target_role": target_role,
        "mapping": mapping,
    }


def run_goes_traditional_layerwise_tracker_satellite(
    *,
    frames_by_role: Mapping[str, Any],
    tracker_name: str,
    tracker_func: Callable,
    tracker_params: Mapping[str, Any] | None,
    tracker_root_dir: str | Path,
    pressure_output_dir: str | Path,
    layer_config: LayerwiseImageConfig,
    pressure_pair_mode: str = "independent",
    mask_config: GoesPcfMaskConfig | None = None,
    geometry_modes: tuple[str, ...] = ("cloud_height",),
    geometry_chunk_rows: int = 256,
    save_native_npz: bool = True,
    save_component_npz: bool = False,
    save_pressure_product: bool = True,
    save_diagnostic_figures: bool = False,
    quicklook_maximum_dimension: int = 1200,
) -> dict[str, Any]:
    """Run one traditional tracker on four pressure-conditioned GOES layers."""

    frames = frame_mapping(frames_by_role)
    validate_frames(frames)
    center = frames["center"]
    satellite = str(center.satellite).upper()
    cfg = mask_config or GoesPcfMaskConfig()
    modes = tuple(str(value).strip().lower() for value in geometry_modes)
    invalid_modes = sorted(set(modes) - set(GEOMETRY_MODES))
    if invalid_modes:
        raise ValueError(f"Unsupported GOES geometry modes: {invalid_modes}")
    pressure_pair_mode = str(pressure_pair_mode).strip().lower()
    if pressure_pair_mode not in PRESSURE_PAIR_MODES:
        raise ValueError(
            f"pressure_pair_mode must be one of {PRESSURE_PAIR_MODES}, "
            f"got {pressure_pair_mode!r}"
        )

    masks = prepare_goes_masks(frames, cfg)
    center_output_mask = np.asarray(masks["center_output"], dtype=bool)
    center_height_mask = np.asarray(masks["center_height"], dtype=bool)
    image_valid_masks = masks["image_valid"]

    images = {
        role: np.asarray(frame.brightness_temperature_k, dtype=np.float32)
        for role, frame in frames.items()
    }
    backgrounds = {
        role: gaussian_background(
            np.where(image_valid_masks[role], images[role], np.nan),
            layer_config.background_sigma_pixels,
        )
        for role in ("previous", "center", "next")
    }
    for role in ("previous", "center", "next"):
        print(
            f"[{tracker_name}] {role:8s}: image_support="
            f"{int(np.count_nonzero(image_valid_masks[role])):,}; "
            f"finite_pressure={int(np.isfinite(frames[role].cloud_top_pressure_hpa).sum()):,}; "
            f"time={frames[role].scan_midpoint}"
        )
    print(
        f"[{tracker_name}] centre output: "
        f"surface={int(np.count_nonzero(center_output_mask)):,}; "
        f"cloud_height={int(np.count_nonzero(center_height_mask)):,}"
    )
    print(f"[{tracker_name}] pressure pair mode: {pressure_pair_mode}")
    print(
        f"[{tracker_name}] layers: "
        f"{[(layer.name, layer.p_min_hpa, layer.p_max_hpa) for layer in layer_config.layers]}"
    )

    tracker_dir = Path(tracker_root_dir) / tracker_name
    component_dir = tracker_dir / "components"
    component_dir.mkdir(parents=True, exist_ok=True)

    total_started = time.perf_counter()
    flows: dict[str, Any] = {}
    winds_by_geometry: dict[str, dict[str, Any]] = {mode: {} for mode in modes}
    layer_ids: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []
    pressure_mappings: dict[str, dict[str, Any]] = {}

    for component, temporal_role, source_role, target_role in COMPONENT_DEFINITIONS:
        source = frames[source_role]
        target = frames[target_role]
        dt_seconds = signed_dt_seconds(source, target)
        pressure0, pressure1, merge_pressure, pressure_mapping = _resolve_pressure_pair(
            frames,
            source_role=source_role,
            target_role=target_role,
            mode=pressure_pair_mode,
        )
        pressure_mappings[component] = pressure_mapping
        print(
            f"\n[{tracker_name}] {component} {temporal_role}: "
            f"{source_role} -> {target_role}; dt={dt_seconds:+.3f} s"
        )
        print(f"[{tracker_name}] {component}: {pressure_mapping['mapping']}")

        layer_flows: list[Any] = []
        layer_weights0: list[np.ndarray] = []
        layer_runtime_rows: list[dict[str, Any]] = []

        for layer_index, layer in enumerate(layer_config.layers):
            pair_started = time.perf_counter()
            layer_pair = build_layer_image_pair(
                images[source_role],
                images[target_role],
                pressure0,
                pressure1,
                layer,
                layer_config,
                valid0=image_valid_masks[source_role],
                valid1=image_valid_masks[target_role],
                background0=backgrounds[source_role],
                background1=backgrounds[target_role],
            )
            preparation_seconds = time.perf_counter() - pair_started

            # The source/centre pressure defines which state may contribute at
            # an output pixel.  Target pressure/image support modifies the
            # conditioned target image but is not a post-hoc hard output mask.
            source_layer_output = (
                center_output_mask
                & (np.asarray(layer_pair["weight0"], dtype=np.float32) > 0.0)
                & np.isfinite(layer_pair["image0"])
                & np.isfinite(layer_pair["image1"])
            )
            params = dict(tracker_params or {})
            params["valid_mask"] = source_layer_output
            if bool(getattr(tracker_func, "requires_nn_context", False)):
                # Same in-memory context as the SEVIRI layerwise NN runner, so
                # FlowDiffuser uses layer-aware uint8 normalization support.
                component_index = int(str(component).lstrip("Cc") or 1) - 1
                params["_nn_context"] = {
                    "image_mode": "layerwise",
                    "component_name": component,
                    "component_index": component_index,
                    "component_seed": int(params.get("seed", 0)) + component_index * 10000,
                    "layer_index": int(layer_index),
                    "layer_name": layer.name,
                    "layer_weight0": layer_pair["weight0"],
                    "layer_weight1": layer_pair["weight1"],
                    "source_valid_mask": image_valid_masks[source_role],
                    "target_valid_mask": image_valid_masks[target_role],
                    "layer_normalization_dilation_pixels": int(
                        params.get("layer_normalization_dilation_pixels", 16)
                    ),
                }

            tracker_started = time.perf_counter()
            layer_flow = tracker_func(
                layer_pair["image0"],
                layer_pair["image1"],
                params,
            )
            tracker_seconds = time.perf_counter() - tracker_started
            layer_flow.valid_mask[:] = (
                np.asarray(layer_flow.valid_mask, dtype=bool)
                & source_layer_output
                & np.isfinite(layer_flow.dx_pixels)
                & np.isfinite(layer_flow.dy_pixels)
            )
            layer_flow.dx_pixels[~layer_flow.valid_mask] = np.nan
            layer_flow.dy_pixels[~layer_flow.valid_mask] = np.nan
            layer_flow.metadata.update(
                {
                    "component": component,
                    "temporal_role": temporal_role,
                    "source_role": source_role,
                    "target_role": target_role,
                    "dt_seconds_signed": float(dt_seconds),
                    "image_mode": "layerwise_goes_c14",
                    "layer_index": int(layer_index),
                    "layer_name": layer.name,
                    "pressure_pair_mode": pressure_pair_mode,
                    "pressure_pair_mapping": dict(pressure_mapping),
                    "output_validity": "centre_frame_source_layer_only",
                    "destination_product_validity_filter": False,
                }
            )
            layer_flows.append(layer_flow)
            layer_weights0.append(np.asarray(layer_pair["weight0"], dtype=np.float32))
            layer_runtime_rows.append(
                {
                    "layer_index": int(layer_index),
                    "layer_name": layer.name,
                    "preparation_seconds": float(preparation_seconds),
                    "tracker_seconds": float(tracker_seconds),
                    "valid_count": int(np.count_nonzero(layer_flow.valid_mask)),
                }
            )
            print(
                f"[{tracker_name}] {component} {layer.name}: "
                f"prep={preparation_seconds:.2f}s; tracker={tracker_seconds:.2f}s; "
                f"valid={int(np.count_nonzero(layer_flow.valid_mask)):,}"
            )

        merge_started = time.perf_counter()
        flow, layer_id = merge_layer_flows(
            layer_flows,
            layer_weights0,
            merge_pressure,
            layer_config,
            base_valid0=center_output_mask,
            tracker_name=tracker_name,
            metadata={
                "pressure_pair_mode": pressure_pair_mode,
                "pressure_pair_mapping": dict(pressure_mapping),
                "output_validity": "centre_frame_only",
                "destination_product_validity_filter": False,
            },
        )
        merge_seconds = time.perf_counter() - merge_started
        flow.valid_mask[:] = (
            np.asarray(flow.valid_mask, dtype=bool)
            & center_output_mask
            & np.isfinite(flow.dx_pixels)
            & np.isfinite(flow.dy_pixels)
        )
        flow.dx_pixels[~flow.valid_mask] = np.nan
        flow.dy_pixels[~flow.valid_mask] = np.nan
        flow.metadata.update(
            {
                "component": component,
                "temporal_role": temporal_role,
                "source_role": source_role,
                "target_role": target_role,
                "source_time": str(source.scan_midpoint),
                "target_time": str(target.scan_midpoint),
                "dt_seconds_signed": float(dt_seconds),
                "image_mode": "layerwise_goes_c14",
                "pressure_pair_mode": pressure_pair_mode,
                "pressure_pair_mapping": dict(pressure_mapping),
                "output_validity": "centre_frame_only",
                "destination_product_validity_filter": False,
                "layer_runtime_rows": layer_runtime_rows,
            }
        )
        flows[component] = flow
        layer_ids[component] = np.asarray(layer_id, dtype=np.int16)
        print(
            f"[{tracker_name}] {component}: merged in {merge_seconds:.2f}s; "
            f"valid={int(np.count_nonzero(flow.valid_mask)):,}"
        )

        component_winds: dict[str, Any] = {}
        geometry_seconds: dict[str, float] = {}
        for mode in modes:
            geometry_started = time.perf_counter()
            mode_mask = center_output_mask if mode == "surface" else center_height_mask
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
            save_component_file(
                path=path,
                flow=flow,
                winds=component_winds,
                component_role=temporal_role,
                layer_id=layer_id,
            )
            print(f"[{tracker_name}] saved component: {path}")

        summary_rows.append(
            {
                "record_type": "component",
                "tracker": tracker_name,
                "satellite": satellite,
                "component": component,
                "temporal_role": temporal_role,
                "source_role": source_role,
                "target_role": target_role,
                "dt_seconds_signed": float(dt_seconds),
                "pressure_pair_mode": pressure_pair_mode,
                "layer_count": len(layer_config.layers),
                "layer_tracker_runtime_seconds": float(
                    sum(row["tracker_seconds"] for row in layer_runtime_rows)
                ),
                "layer_preparation_seconds": float(
                    sum(row["preparation_seconds"] for row in layer_runtime_rows)
                ),
                "layer_merge_seconds": float(merge_seconds),
                "surface_geometry_seconds": geometry_seconds.get("surface", np.nan),
                "cloud_height_geometry_seconds": geometry_seconds.get("cloud_height", np.nan),
                "flow_valid_count": int(np.count_nonzero(flow.valid_mask)),
                "layer_runtime_json": compact_json(layer_runtime_rows),
                "tracker_metadata_json": compact_json(flow.metadata),
            }
        )
        del layer_flows, layer_weights0
        gc.collect()

    base_metadata = {
        "method": "GOES centre-anchored pressure-layer traditional dense optical flow",
        "image_mode": "layerwise_goes_c14",
        "tracker_parameters": dict(tracker_params or {}),
        "layerwise_config": layer_config.metadata(),
        "pressure_pair_mode": pressure_pair_mode,
        "pressure_pair_mapping": pressure_mappings,
        "mask_config": cfg.__dict__,
    }
    final_result = finalize_components(
        tracker_name=tracker_name,
        satellite=satellite,
        center=center,
        flows=flows,
        winds_by_geometry=winds_by_geometry,
        center_output_mask=center_output_mask,
        center_height_mask=center_height_mask,
        tracker_root_dir=tracker_root_dir,
        pressure_output_dir=pressure_output_dir,
        metadata=base_metadata,
        layer_ids=layer_ids,
        save_native_npz=save_native_npz,
        save_pressure_product=save_pressure_product,
        save_diagnostic_figures_flag=save_diagnostic_figures,
        quicklook_maximum_dimension=quicklook_maximum_dimension,
    )

    total_seconds = time.perf_counter() - total_started
    summary_rows.append(
        {
            "record_type": "final",
            "tracker": tracker_name,
            "satellite": satellite,
            "component": "C1+C2",
            "temporal_role": "+".join(modes),
            "source_role": "center",
            "target_role": "previous+next",
            "dt_seconds_signed": np.nan,
            "pressure_pair_mode": pressure_pair_mode,
            "layer_count": len(layer_config.layers),
            "layer_tracker_runtime_seconds": total_seconds,
            "layer_preparation_seconds": np.nan,
            "layer_merge_seconds": np.nan,
            "surface_geometry_seconds": np.nan,
            "cloud_height_geometry_seconds": np.nan,
            "flow_valid_count": int(
                np.count_nonzero(
                    final_result["final_by_geometry"][modes[0]]["valid_mask"]
                )
            ),
            "layer_runtime_json": "",
            "tracker_metadata_json": compact_json(base_metadata),
        }
    )
    summary_path = tracker_dir / "run_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    return {
        "satellite": satellite,
        "flows": flows,
        "winds_by_geometry": winds_by_geometry,
        "layer_ids": layer_ids,
        **final_result,
        "summary_path": summary_path,
        "runtime_seconds": total_seconds,
    }
