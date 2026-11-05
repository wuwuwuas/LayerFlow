"""Image-only preprocessing shared by all trackers."""

from __future__ import annotations

import cv2
import numpy as np


def normalize_image_pair(
    image0: np.ndarray,
    image1: np.ndarray,
    *,
    valid0: np.ndarray | None = None,
    valid1: np.ndarray | None = None,
    mode: str = "percentile_normalize",
    percentiles: tuple[float, float] = (1.0, 99.0),
    output_dtype: str = "float32",
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize two images jointly without changing their geometry."""

    first = np.asarray(image0, dtype=np.float32)
    second = np.asarray(image1, dtype=np.float32)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Images must be equally shaped two-dimensional arrays")
    valid = np.isfinite(first) & np.isfinite(second)
    if valid0 is not None:
        valid &= np.asarray(valid0, dtype=bool)
    if valid1 is not None:
        valid &= np.asarray(valid1, dtype=bool)
    if not np.any(valid):
        raise ValueError("Image pair contains no jointly valid pixels")
    if mode not in {"percentile_normalize", "joint_percentile", "none"}:
        raise ValueError(f"Unsupported normalization mode: {mode}")
    if mode == "none":
        out0, out1 = first.copy(), second.copy()
    else:
        values = np.concatenate((first[valid], second[valid]))
        low, high = np.nanpercentile(values, percentiles)
        if not np.isfinite(low + high) or high <= low:
            high = low + 1.0
        out0 = np.clip((first - low) / (high - low), 0.0, 1.0)
        out1 = np.clip((second - low) / (high - low), 0.0, 1.0)
    fill0 = float(np.nanmedian(out0[valid]))
    fill1 = float(np.nanmedian(out1[valid]))
    out0 = np.where(np.isfinite(out0), out0, fill0)
    out1 = np.where(np.isfinite(out1), out1, fill1)
    if output_dtype == "uint8":
        return (
            np.round(out0 * 255.0).astype(np.uint8),
            np.round(out1 * 255.0).astype(np.uint8),
        )
    if output_dtype != "float32":
        raise ValueError("output_dtype must be 'float32' or 'uint8'")
    return out0.astype(np.float32), out1.astype(np.float32)


def downsample_image_pair(
    image0: np.ndarray,
    image1: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample equally shaped images using area averaging."""

    first = np.asarray(image0)
    second = np.asarray(image1)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Images must be equally shaped two-dimensional arrays")
    if not 0 < scale <= 1:
        raise ValueError("scale must satisfy 0 < scale <= 1")
    if scale == 1:
        return first.copy(), second.copy()
    size = (
        max(1, int(np.floor(first.shape[1] * scale))),
        max(1, int(np.floor(first.shape[0] * scale))),
    )
    return (
        cv2.resize(first, size, interpolation=cv2.INTER_AREA),
        cv2.resize(second, size, interpolation=cv2.INTER_AREA),
    )
