"""Shared GOES helpers for migrated traditional optical-flow trackers.

This module contains no optical-flow algorithm.  It standardises the existing
traditional trackers to the same GOES product semantics used by the PCF
pipeline:

* C1 is centre -> previous and keeps a negative time interval;
* C2 is centre -> next and keeps a positive time interval;
* both flows are defined on the common centre/source grid;
* final output validity is determined by the centre GOES products;
* previous/next product validity is not applied as a post-hoc destination
  filter;
* the same pixel flow can be converted with surface or centre-CTH geometry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from pipeline.tracker_runner import save_stacked_piv_npz
from postprocess.combine_components import combine_components_to_stacked_arrays
from preprocessing import sample_at_centers
from preprocessing.goes import (
    GoesPcfMaskConfig,
    build_goes_center_height_mask,
    build_goes_center_output_mask,
)
from preprocessing.layerwise_image import gaussian_background


COMPONENT_DEFINITIONS = (
    ("C1", "backward", "center", "previous"),
    ("C2", "forward", "center", "next"),
)
GEOMETRY_MODES = ("surface", "cloud_height")


def compact_json(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def frame_mapping(frames_by_role: Mapping[str, Any]) -> dict[str, Any]:
    required = {"previous", "center", "next"}
    missing = sorted(required - set(frames_by_role))
    if missing:
        raise KeyError(f"GOES frame mapping is missing roles: {missing}")
    return {role: frames_by_role[role] for role in ("previous", "center", "next")}


def validate_frames(frames: Mapping[str, Any]) -> tuple[int, int]:
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


def signed_dt_seconds(source: Any, target: Any) -> float:
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


def build_image_valid_mask(
    frame: Any,
    *,
    require_image_dqf_zero: bool = True,
) -> np.ndarray:
    """Return image-only solver support for a traditional raw tracker.

    Unlike the PCF pressure-state solver mask, this does not require CTP or a
    cloudy ACM category.  Pressure/cloud products define the final AMV output
    mask downstream; the raw traditional tracker receives the C14 image field.
    """

    image = np.asarray(frame.brightness_temperature_k, dtype=np.float32)
    valid = np.isfinite(image)
    dqf = getattr(frame, "image_dqf", None)
    if require_image_dqf_zero and dqf is not None:
        values = np.asarray(dqf)
        if values.shape != image.shape:
            raise ValueError("image_dqf must match brightness-temperature shape")
        valid &= np.isfinite(values) & (values == 0)
    return np.asarray(valid, dtype=bool)


def fill_image_for_dense_tracker(
    image: np.ndarray,
    valid_mask: np.ndarray,
    *,
    background_sigma_pixels: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill unsupported pixels with a finite smooth background.

    Existing traditional tracker adapters do not consume solver masks in their
    numerical kernels; their ``valid_mask`` controls the returned FlowResult.
    Finite background filling therefore prevents NaNs/outside-disk values from
    entering OpenCV while preserving centre-only output validity.
    """

    values = np.asarray(image, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    if values.shape != valid.shape:
        raise ValueError("image and valid_mask must share a shape")
    masked = np.where(valid, values, np.nan).astype(np.float32)
    background = gaussian_background(masked, float(background_sigma_pixels))
    filled = np.where(valid, values, background).astype(np.float32)
    if not np.isfinite(filled).all():
        finite = filled[np.isfinite(filled)]
        fallback = float(np.nanmedian(finite)) if finite.size else 0.0
        filled = np.where(np.isfinite(filled), filled, fallback).astype(np.float32)
    return filled, background


def prepare_goes_masks(
    frames: Mapping[str, Any],
    mask_config: GoesPcfMaskConfig,
) -> dict[str, Any]:
    image_valid = {
        role: build_image_valid_mask(
            frame,
            require_image_dqf_zero=mask_config.require_image_dqf_zero,
        )
        for role, frame in frames.items()
    }
    center = frames["center"]
    return {
        "image_valid": image_valid,
        "center_output": build_goes_center_output_mask(center, mask_config),
        "center_height": build_goes_center_height_mask(center, mask_config),
    }


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


def attach_center_products(
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
    cloud_mask = np.where(np.isfinite(cloud_sample), cloud_sample, -1).astype(np.int16)
    mask_sample = _sample_center_field(
        np.asarray(center_mask, dtype=np.float32), rows, cols, order=0
    ) >= 0.5

    final_valid = (
        np.asarray(arrays["valid_mask"], dtype=bool)
        & mask_sample
        & np.isfinite(pressure)
    )
    if str(geometry_mode).lower() == "cloud_height":
        final_valid &= np.isfinite(height)

    arrays["valid_mask"] = final_valid
    arrays["u_ms"] = np.where(final_valid, arrays["u_ms"], np.nan).astype(np.float32)
    arrays["v_ms"] = np.where(final_valid, arrays["v_ms"], np.nan).astype(np.float32)
    arrays["pressure_hpa"] = np.where(final_valid, pressure, np.nan).astype(np.float32)
    arrays["height_m"] = height
    arrays["cloud_mask"] = cloud_mask
    arrays["image_dqf"] = _sample_quality(center.image_dqf, rows, cols)
    arrays["height_dqf"] = _sample_quality(center.height_dqf, rows, cols)
    arrays["pressure_dqf"] = _sample_quality(center.pressure_dqf, rows, cols)
    arrays["cloud_mask_dqf"] = _sample_quality(center.cloud_mask_dqf, rows, cols)
    return arrays


def save_component_npz(
    *,
    path: Path,
    flow: Any,
    winds: Mapping[str, Any],
    component_role: str,
    layer_id: np.ndarray | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {
        "row_center_original": flow.row_center_original,
        "col_center_original": flow.col_center_original,
        "dx_pixels": flow.dx_pixels,
        "dy_pixels": flow.dy_pixels,
        "flow_valid_mask": flow.valid_mask,
        "component_role": np.asarray([component_role]),
        "flow_metadata_json": np.asarray([compact_json(flow.metadata)]),
    }
    if layer_id is not None:
        output["pressure_layer_id"] = np.asarray(layer_id, dtype=np.int16)
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
            [compact_json(wind.metadata)]
        )
    np.savez(path, **output)
    return path


def save_compact_goes_amv(
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
    direction = ((270.0 - np.degrees(np.arctan2(v, u))) % 360.0).astype(np.float32)
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
                compact_json(
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


def save_diagnostic_figures(
    *,
    flows: Mapping[str, Any],
    final_by_geometry: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    maximum_dimension: int = 1200,
) -> None:
    """Save downsampled quicklooks without full-resolution Matplotlib stalls."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; GOES traditional quicklooks skipped")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    shape = np.asarray(next(iter(flows.values())).dx_pixels).shape
    stride = max(1, int(np.ceil(max(shape) / max(1, int(maximum_dimension)))))
    sl = (slice(None, None, stride), slice(None, None, stride))

    for component in ("C1", "C2"):
        flow = flows[component]
        valid = np.asarray(flow.valid_mask, dtype=bool)
        pixel_speed = np.hypot(flow.dx_pixels, flow.dy_pixels)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for ax, values, title in (
            (axes[0, 0], np.where(valid, flow.dx_pixels, np.nan)[sl], "dx pixels"),
            (axes[0, 1], np.where(valid, flow.dy_pixels, np.nan)[sl], "dy pixels"),
            (axes[1, 0], np.where(valid, pixel_speed, np.nan)[sl], "pixel displacement magnitude"),
            (axes[1, 1], valid.astype(np.float32)[sl], "centre-output flow mask"),
        ):
            image = ax.imshow(values, origin="upper")
            ax.set_title(title)
            ax.set_axis_off()
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / f"{component.lower()}_flow_quicklook.png", dpi=140)
        plt.close(fig)

    for mode, arrays in final_by_geometry.items():
        final_valid = np.asarray(arrays["valid_mask"], dtype=bool)
        final_speed = np.hypot(arrays["u_ms"], arrays["v_ms"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        image = axes[0].imshow(
            np.where(final_valid, final_speed, np.nan)[sl], origin="upper"
        )
        axes[0].set_title(f"{mode} wind speed (m/s)")
        axes[0].set_axis_off()
        fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
        image = axes[1].imshow(
            np.asarray(arrays["valid_component_count"])[sl], origin="upper"
        )
        axes[1].set_title("valid component count")
        axes[1].set_axis_off()
        fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / f"final_{mode}_wind_quicklook.png", dpi=140)
        plt.close(fig)


def finalize_components(
    *,
    tracker_name: str,
    satellite: str,
    center: Any,
    flows: Mapping[str, Any],
    winds_by_geometry: Mapping[str, Mapping[str, Any]],
    center_output_mask: np.ndarray,
    center_height_mask: np.ndarray,
    tracker_root_dir: str | Path,
    pressure_output_dir: str | Path,
    metadata: Mapping[str, Any],
    layer_ids: Mapping[str, np.ndarray] | None = None,
    save_native_npz: bool = True,
    save_pressure_product: bool = True,
    save_diagnostic_figures_flag: bool = False,
    quicklook_maximum_dimension: int = 1200,
) -> dict[str, Any]:
    tracker_dir = Path(tracker_root_dir) / tracker_name
    final_dir = tracker_dir / "final"
    pressure_dir = Path(pressure_output_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    pressure_dir.mkdir(parents=True, exist_ok=True)

    final_by_geometry: dict[str, dict[str, Any]] = {}
    final_paths: dict[str, Path | None] = {}
    pressure_paths: dict[str, Path | None] = {}
    pressure_point_counts: dict[str, int] = {}

    for mode, component_winds in winds_by_geometry.items():
        arrays = combine_components_to_stacked_arrays(
            flows,
            component_winds,
            min_valid_components=2,
            component_order=("C1", "C2"),
            check_lonlat=True,
        )
        mode_mask = center_output_mask if mode == "surface" else center_height_mask
        arrays = attach_center_products(
            arrays,
            center=center,
            center_mask=mode_mask,
            geometry_mode=mode,
        )
        arrays["local_zenith_angle_deg"] = np.asarray(
            component_winds["C1"].diagnostics["local_zenith_angle_deg"],
            dtype=np.float32,
        )
        if layer_ids is not None:
            for component, layer_id in layer_ids.items():
                arrays[f"pressure_layer_id_{component.lower()}"] = np.asarray(
                    layer_id, dtype=np.int16
                )

        final_metadata = {
            **dict(arrays.get("metadata", {})),
            **dict(metadata),
            "tracker": tracker_name,
            "satellite": satellite,
            "component_mapping": {
                "C1": "center -> previous (backward; signed dt < 0)",
                "C2": "center -> next (forward; signed dt > 0)",
            },
            "component_combination": "mean C1/C2; both components required",
            "minimum_valid_components": 2,
            "geometry_mode": mode,
            "destination_product_validity_filter": False,
            "final_pressure_source": "centre-frame GOES ABI CTP",
            "centre_scan_start": str(center.scan_start),
            "centre_scan_end": str(center.scan_end),
            "centre_scan_midpoint": str(center.scan_midpoint),
        }
        arrays["metadata"] = final_metadata
        final_by_geometry[mode] = arrays

        final_path: Path | None = None
        if save_native_npz:
            final_path = final_dir / f"{tracker_name}_{mode}_final_native.npz"
            save_stacked_piv_npz(arrays, final_path)
            print(f"[{tracker_name}] saved {mode} native product: {final_path}")
        final_paths[mode] = final_path

        pressure_path: Path | None = None
        point_count = 0
        if save_pressure_product:
            pressure_path = (
                pressure_dir / f"{tracker_name}_{mode}_final_amv_goes_pressure.npz"
            )
            pressure_path, point_count = save_compact_goes_amv(
                arrays=arrays,
                center_frame=center,
                path=pressure_path,
                tracker_name=tracker_name,
                satellite=satellite,
                geometry_mode=mode,
                metadata=final_metadata,
            )
            print(
                f"[{tracker_name}] saved {mode} GOES-pressure AMV: "
                f"{pressure_path}; points={point_count:,}"
            )
        pressure_paths[mode] = pressure_path
        pressure_point_counts[mode] = point_count

    if save_diagnostic_figures_flag:
        save_diagnostic_figures(
            flows=flows,
            final_by_geometry=final_by_geometry,
            output_dir=tracker_dir / "figures",
            maximum_dimension=quicklook_maximum_dimension,
        )

    return {
        "final_by_geometry": final_by_geometry,
        "final_paths": final_paths,
        "pressure_paths": pressure_paths,
        "pressure_point_counts": pressure_point_counts,
    }
