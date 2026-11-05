"""Small helpers for dense optical-flow tracker adapters.

The real algorithms either live in OpenCV/scikit-image or under ``piv/``.
The ``tracking`` package only converts their outputs into the common
``FlowResult`` interface used by ``pipeline.tracker_runner``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from models import FlowResult


def input_valid_mask(image0: np.ndarray, image1: np.ndarray, params: Mapping[str, Any] | None = None) -> np.ndarray:
    """Finite input pixels intersected with an optional caller-provided valid mask."""

    settings = dict(params or {})
    valid = np.isfinite(image0) & np.isfinite(image1)
    if settings.get("valid_mask") is not None:
        supplied = np.asarray(settings["valid_mask"], dtype=bool)
        if supplied.shape != image0.shape:
            raise ValueError(f"valid_mask shape {supplied.shape} does not match image shape {image0.shape}")
        valid &= supplied
    return valid


def normalize_pair_robust(image0: np.ndarray, image1: np.ndarray) -> dict[str, Any]:
    """Jointly normalize a 2-D image pair to float32 [0,1] and uint8 [0,255]."""

    first = np.asarray(image0, dtype=np.float32)
    second = np.asarray(image1, dtype=np.float32)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Dense OF inputs must be equally shaped 2-D images")

    finite0 = np.isfinite(first)
    finite1 = np.isfinite(second)
    values = np.concatenate([first[finite0].ravel(), second[finite1].ravel()])
    info: dict[str, Any] = {"normalization": "joint_p1_p99", "finite_sample_count": int(values.size)}
    if values.size >= 16:
        p1, p99 = np.nanpercentile(values, [1.0, 99.0])
        if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
            p1 = float(np.nanmin(values)) if values.size else 0.0
            p99 = float(np.nanmax(values)) if values.size else 1.0
            info["normalization"] = "joint_min_max_fallback"
    elif values.size > 0:
        p1 = float(np.nanmin(values))
        p99 = float(np.nanmax(values))
        info["normalization"] = "joint_min_max_small_sample"
    else:
        p1, p99 = 0.0, 1.0
        info["normalization"] = "all_zero_no_finite_pixels"

    denom = float(p99 - p1)
    if not np.isfinite(denom) or denom <= 1e-6:
        denom = 1.0
        info["normalization"] = f"{info['normalization']}_flat"
    info["p1"] = float(p1)
    info["p99"] = float(p99)

    def scale(img: np.ndarray) -> np.ndarray:
        out = (np.where(np.isfinite(img), img, p1) - p1) / (denom + 1e-6)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    f0 = scale(first)
    f1 = scale(second)
    u0 = np.clip(np.rint(f0 * 255.0), 0, 255).astype(np.uint8)
    u1 = np.clip(np.rint(f1 * 255.0), 0, 255).astype(np.uint8)
    return {"float32": (f0, f1), "uint8": (u0, u1), "diagnostics": info}


def make_dense_flow_result(
    image0: np.ndarray,
    image1: np.ndarray,
    flow: np.ndarray,
    params: Mapping[str, Any] | None,
    *,
    tracker: str,
    parameters: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FlowResult:
    """Convert an HxWx2 dense flow array to the common FlowResult object."""

    first = np.asarray(image0)
    arr = np.asarray(flow, dtype=np.float32)
    if arr.shape != first.shape + (2,):
        raise ValueError(f"{tracker} flow must have shape {first.shape + (2,)}, got {arr.shape}")
    rows, cols = np.indices(first.shape, dtype=np.float32)
    valid = input_valid_mask(np.asarray(image0), np.asarray(image1), params)
    valid &= np.isfinite(arr[..., 0]) & np.isfinite(arr[..., 1])
    dx = np.where(valid, arr[..., 0], np.nan).astype(np.float32)
    dy = np.where(valid, arr[..., 1], np.nan).astype(np.float32)
    return FlowResult(
        dx,
        dy,
        rows,
        cols,
        valid,
        {
            "tracker": tracker,
            "scale": 1.0,
            "parameters": dict(parameters or {}),
            **dict(metadata or {}),
        },
    )


def flow_diagnostics(flow: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(flow, dtype=np.float32)
    speed = np.sqrt(arr[..., 0] ** 2 + arr[..., 1] ** 2)
    finite = np.isfinite(speed)
    if not finite.any():
        return {"nan_count": int(np.isnan(arr).sum()), "inf_count": int(np.isinf(arr).sum())}
    return {
        "mean_abs_dx_px": float(np.nanmean(np.abs(arr[..., 0]))),
        "mean_abs_dy_px": float(np.nanmean(np.abs(arr[..., 1]))),
        "median_speed_px": float(np.nanmedian(speed)),
        "p95_speed_px": float(np.nanpercentile(speed, 95)),
        "p99_speed_px": float(np.nanpercentile(speed, 99)),
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
    }
