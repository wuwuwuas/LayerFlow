"""GOES-specific masks for centre-anchored pressure-conditioned flow.

The masks are deliberately separated by responsibility:

* ``solver_valid_mask`` controls where each input frame contributes image and
  pressure support to the PCF data term.
* ``center_output_mask`` controls which centre-frame pixels are eligible for a
  dense AMV output.  It does not require the previous/next frame to be valid at
  the same array coordinate.
* ``center_height_mask`` is only needed by the optional cloud-height geometry.

The actual pressure-state membership remains defined by
``default_pressure_layer_config`` (50--1100 hPa in four states).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GoesPcfMaskConfig:
    minimum_tracking_pressure_hpa: float = 50.0
    maximum_tracking_pressure_hpa: float = 1100.0
    minimum_cloud_mask_value: int = 2
    require_image_dqf_zero: bool = True
    require_pressure_dqf_zero: bool = True
    require_cloud_mask_dqf_zero: bool = False
    minimum_height_m: float = -500.0
    maximum_height_m: float = 25000.0
    require_height_dqf_zero: bool = True


# Backward-compatible name used by the first GOES patch.
GoesTrackingMaskConfig = GoesPcfMaskConfig


def _require_same_shape(*arrays: np.ndarray) -> tuple[int, int]:
    shapes = {np.asarray(value).shape for value in arrays}
    if len(shapes) != 1:
        raise ValueError(f"GOES fields must share one HxW shape; got {sorted(shapes)}")
    shape = next(iter(shapes))
    if len(shape) != 2:
        raise ValueError(f"GOES fields must be two-dimensional; got {shape}")
    return int(shape[0]), int(shape[1])


def _apply_dqf(valid: np.ndarray, dqf: np.ndarray | None, required: bool) -> np.ndarray:
    if not required or dqf is None:
        return valid
    quality = np.asarray(dqf)
    if quality.shape != valid.shape:
        raise ValueError(
            f"DQF shape {quality.shape} does not match data shape {valid.shape}"
        )
    return valid & (quality == 0)


def build_goes_solver_valid_mask(
    frame: Any,
    config: GoesPcfMaskConfig | None = None,
) -> np.ndarray:
    """Return the per-frame PCF support mask.

    This mask is supplied to the existing pressure-conditioned solver for both
    source and target frames.  Target-frame invalid areas therefore contribute
    no target data term (``target_invalid_weight=0``), while the dense solver can
    still propagate a flow estimate through spatial regularisation.
    """

    cfg = config or GoesPcfMaskConfig()
    image = np.asarray(frame.brightness_temperature_k, dtype=np.float32)
    pressure = np.asarray(frame.cloud_top_pressure_hpa, dtype=np.float32)
    cloud_mask = np.asarray(frame.cloud_mask)
    _require_same_shape(image, pressure, cloud_mask)

    valid = (
        np.isfinite(image)
        & np.isfinite(pressure)
        & np.isfinite(cloud_mask)
        & (pressure >= float(cfg.minimum_tracking_pressure_hpa))
        & (pressure < float(cfg.maximum_tracking_pressure_hpa))
        & (cloud_mask >= int(cfg.minimum_cloud_mask_value))
    )
    valid = _apply_dqf(valid, frame.image_dqf, cfg.require_image_dqf_zero)
    valid = _apply_dqf(valid, frame.pressure_dqf, cfg.require_pressure_dqf_zero)
    valid = _apply_dqf(
        valid,
        frame.cloud_mask_dqf,
        cfg.require_cloud_mask_dqf_zero,
    )
    return np.asarray(valid, dtype=bool)


def build_goes_center_output_mask(
    center_frame: Any,
    config: GoesPcfMaskConfig | None = None,
) -> np.ndarray:
    """Return centre-frame pixels eligible for a dense AMV output.

    It intentionally depends on I0 only.  Previous/next validity is already used
    inside the PCF data term and is not re-applied as a same-pixel or displaced-
    destination hard filter.
    """

    return build_goes_solver_valid_mask(center_frame, config)


def build_goes_center_height_mask(
    center_frame: Any,
    config: GoesPcfMaskConfig | None = None,
) -> np.ndarray:
    """Return the additional centre CTH validity for parallax-corrected geometry."""

    cfg = config or GoesPcfMaskConfig()
    height = np.asarray(center_frame.cloud_top_height_m, dtype=np.float32)
    output = build_goes_center_output_mask(center_frame, cfg)
    if height.shape != output.shape:
        raise ValueError("GOES centre height shape does not match the output mask")
    output &= (
        np.isfinite(height)
        & (height >= float(cfg.minimum_height_m))
        & (height <= float(cfg.maximum_height_m))
    )
    output = _apply_dqf(
        output,
        center_frame.height_dqf,
        cfg.require_height_dqf_zero,
    )
    return np.asarray(output, dtype=bool)


def build_goes_tracking_valid_mask(
    frame: Any,
    config: GoesPcfMaskConfig | None = None,
) -> np.ndarray:
    """Backward-compatible alias for the per-frame solver support mask."""

    return build_goes_solver_valid_mask(frame, config)
