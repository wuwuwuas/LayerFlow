"""GOES centre-anchored runner for existing traditional dense trackers."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from postprocess.goes_flow_to_wind import goes_flow_to_wind
from preprocessing.goes import GoesPcfMaskConfig
from pipeline.goes_traditional_common import (
    COMPONENT_DEFINITIONS,
    GEOMETRY_MODES,
    compact_json,
    fill_image_for_dense_tracker,
    finalize_components,
    frame_mapping,
    prepare_goes_masks,
    save_component_npz as save_component_file,
    signed_dt_seconds,
    validate_frames,
)


def run_goes_traditional_tracker_satellite(
    *,
    frames_by_role: Mapping[str, Any],
    tracker_name: str,
    tracker_func: Callable,
    tracker_params: Mapping[str, Any] | None,
    tracker_root_dir: str | Path,
    pressure_output_dir: str | Path,
    mask_config: GoesPcfMaskConfig | None = None,
    geometry_modes: tuple[str, ...] = ("cloud_height",),
    geometry_chunk_rows: int = 256,
    background_sigma_pixels: float = 35.0,
    save_native_npz: bool = True,
    save_component_npz: bool = False,
    save_pressure_product: bool = True,
    save_diagnostic_figures: bool = False,
    quicklook_maximum_dimension: int = 1200,
) -> dict[str, Any]:
    """Run one existing traditional tracker on centre->previous/next C14 pairs."""

    frames = frame_mapping(frames_by_role)
    validate_frames(frames)
    center = frames["center"]
    satellite = str(center.satellite).upper()
    cfg = mask_config or GoesPcfMaskConfig()
    modes = tuple(str(value).strip().lower() for value in geometry_modes)
    invalid_modes = sorted(set(modes) - set(GEOMETRY_MODES))
    if invalid_modes:
        raise ValueError(f"Unsupported GOES geometry modes: {invalid_modes}")

    masks = prepare_goes_masks(frames, cfg)
    center_output_mask = np.asarray(masks["center_output"], dtype=bool)
    center_height_mask = np.asarray(masks["center_height"], dtype=bool)
    image_valid_masks = masks["image_valid"]

    prepared_images: dict[str, np.ndarray] = {}
    for role in ("previous", "center", "next"):
        prepared, _ = fill_image_for_dense_tracker(
            frames[role].brightness_temperature_k,
            image_valid_masks[role],
            background_sigma_pixels=background_sigma_pixels,
        )
        prepared_images[role] = prepared
        print(
            f"[{tracker_name}] {role:8s}: image_support="
            f"{int(np.count_nonzero(image_valid_masks[role])):,}; "
            f"time={frames[role].scan_midpoint}"
        )
    print(
        f"[{tracker_name}] centre output: "
        f"surface={int(np.count_nonzero(center_output_mask)):,}; "
        f"cloud_height={int(np.count_nonzero(center_height_mask)):,}"
    )

    tracker_dir = Path(tracker_root_dir) / tracker_name
    component_dir = tracker_dir / "components"
    component_dir.mkdir(parents=True, exist_ok=True)

    total_started = time.perf_counter()
    flows: dict[str, Any] = {}
    winds_by_geometry: dict[str, dict[str, Any]] = {mode: {} for mode in modes}
    summary_rows: list[dict[str, Any]] = []

    for component, temporal_role, source_role, target_role in COMPONENT_DEFINITIONS:
        source = frames[source_role]
        target = frames[target_role]
        dt_seconds = signed_dt_seconds(source, target)
        print(
            f"\n[{tracker_name}] {component} {temporal_role}: "
            f"{source_role} -> {target_role}; dt={dt_seconds:+.3f} s"
        )

        params = dict(tracker_params or {})
        # Existing adapters use valid_mask to define the returned dense product.
        # Inputs have already been finitely background-filled, so this remains a
        # centre/source validity rule and does not impose target-frame hard QC.
        params["valid_mask"] = center_output_mask
        if bool(getattr(tracker_func, "requires_nn_context", False)):
            # Mirror the SEVIRI raw NN runner: the NN valid_mask is the joint
            # image support of the pair (it sets the uint8 normalization support
            # and the processed tiles).  The final product is still restricted
            # to the centre output mask immediately after the tracker call.
            params["valid_mask"] = (
                np.asarray(image_valid_masks[source_role], dtype=bool)
                & np.asarray(image_valid_masks[target_role], dtype=bool)
            )
            component_index = int(str(component).lstrip("Cc") or 1) - 1
            params["_nn_context"] = {
                "image_mode": "original",
                "component_name": component,
                "component_index": component_index,
                "component_seed": int(params.get("seed", 0)) + component_index * 10000,
            }

        tracker_started = time.perf_counter()
        flow = tracker_func(
            prepared_images[source_role],
            prepared_images[target_role],
            params,
        )
        tracker_seconds = time.perf_counter() - tracker_started
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
                "image_mode": "raw_goes_c14",
                "output_validity": "centre_frame_only",
                "destination_product_validity_filter": False,
                "input_invalid_fill": "finite_aware_gaussian_background",
                "input_background_sigma_pixels": float(background_sigma_pixels),
            }
        )
        flows[component] = flow
        print(
            f"[{tracker_name}] {component}: tracker={tracker_seconds:.2f}s; "
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
                "tracker_runtime_seconds": float(tracker_seconds),
                "surface_geometry_seconds": geometry_seconds.get("surface", np.nan),
                "cloud_height_geometry_seconds": geometry_seconds.get("cloud_height", np.nan),
                "flow_valid_count": int(np.count_nonzero(flow.valid_mask)),
                "tracker_metadata_json": compact_json(flow.metadata),
            }
        )
        gc.collect()

    base_metadata = {
        "method": "GOES centre-anchored traditional dense optical flow",
        "image_mode": "raw_goes_c14",
        "tracker_parameters": dict(tracker_params or {}),
        "mask_config": cfg.__dict__,
        "input_invalid_fill": "finite_aware_gaussian_background",
        "input_background_sigma_pixels": float(background_sigma_pixels),
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
            "tracker_runtime_seconds": total_seconds,
            "surface_geometry_seconds": np.nan,
            "cloud_height_geometry_seconds": np.nan,
            "flow_valid_count": int(
                np.count_nonzero(
                    final_result["final_by_geometry"][modes[0]]["valid_mask"]
                )
            ),
            "tracker_metadata_json": compact_json(base_metadata),
        }
    )
    summary_path = tracker_dir / "run_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    return {
        "satellite": satellite,
        "flows": flows,
        "winds_by_geometry": winds_by_geometry,
        **final_result,
        "summary_path": summary_path,
        "runtime_seconds": total_seconds,
    }
