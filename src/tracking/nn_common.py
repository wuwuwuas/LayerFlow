"""Shared NN tracker wrapper utilities.

Concrete model entry points live in lraft.py, lraft_st.py, raft.py, and
flowdiffuser.py. This module converts model outputs into the common FlowResult
contract used by the raw and layerwise pipelines.
"""

from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np

from models import FlowResult
from preprocessing import build_coarse_grid_centers, downsample_image_pair
from .dense_common import input_valid_mask


def _overlap_pixels(overlap: Any, tile_size: int) -> int:
    if overlap is None:
        return int(round(tile_size * 0.5))
    value = float(overlap)
    if 0 <= value < 1:
        return int(round(tile_size * value))
    return int(round(value))


def _resize_mask_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    if source.shape == shape:
        return source.copy()
    resized = cv2.resize(
        source.astype(np.uint8),
        (int(shape[1]), int(shape[0])),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def _predict_nn_pair(
    image0: np.ndarray,
    image1: np.ndarray,
    params: dict[str, Any],
    *,
    model_name: str,
    inference_valid_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    predictor = params.get("predictor")
    if predictor is None:
        from piv.nn_tracker_adapter import get_cached_nn_predictor

        predictor = get_cached_nn_predictor(model_name, params)

    context = params.get("_nn_context")
    if hasattr(predictor, "predict_pair"):
        output = predictor.predict_pair(
            image0,
            image1,
            valid_mask=inference_valid_mask,
            context=context,
        )
    else:
        output = predictor(image0, image1)
    flow, metadata = output if isinstance(output, tuple) else (output, {})
    flow = np.asarray(flow, dtype=np.float32)
    if flow.shape != (*image0.shape, 2):
        raise ValueError(
            f"{model_name} output must have shape "
            f"{(image0.shape[0], image0.shape[1], 2)}, got {flow.shape}"
        )
    return flow, dict(metadata)


def run_nn_flow(
    image0: np.ndarray,
    image1: np.ndarray,
    params: dict[str, Any] | None,
    *,
    model_name: str,
) -> FlowResult:
    """Run one NN model and return FlowResult.

    ``scale == 1`` returns dense output. ``scale < 1`` returns a coarse/vector
    grid whose centers and displacements are mapped to original image pixels.

    Validity is defined only by the caller-provided mask, finite inputs, and
    finite model output. Invalid displacement values are stored as NaN so that
    finite-value counts and ``valid_mask`` cannot disagree downstream.
    """

    settings = dict(params or {})
    first = np.asarray(image0, dtype=np.float32)
    second = np.asarray(image1, dtype=np.float32)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("NN tracker inputs must be equally shaped 2-D images")

    scale = float(settings.get("scale", 1.0))
    if not 0 < scale <= 1:
        raise ValueError("scale must satisfy 0 < scale <= 1")

    original_valid = input_valid_mask(first, second, settings)
    output_mode = "dense" if np.isclose(scale, 1.0) else "coarse"
    low0, low1 = downsample_image_pair(first, second, scale)
    low_valid = _resize_mask_nearest(original_valid, low0.shape)
    low_flow, metadata = _predict_nn_pair(
        low0,
        low1,
        settings,
        model_name=model_name,
        inference_valid_mask=low_valid,
    )

    if output_mode == "dense":
        if scale < 1:
            flow = cv2.resize(
                low_flow,
                (first.shape[1], first.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            ) / scale
        else:
            flow = low_flow
        rows, cols = np.indices(first.shape, dtype=np.float32)
        valid = original_valid.copy()
    else:
        flow = low_flow / scale
        rows, cols = build_coarse_grid_centers(first.shape, scale)
        rr = np.clip(np.rint(rows).astype(int), 0, first.shape[0] - 1)
        cc = np.clip(np.rint(cols).astype(int), 0, first.shape[1] - 1)
        valid = original_valid[rr, cc]

    finite_flow = np.isfinite(flow[..., 0]) & np.isfinite(flow[..., 1])
    valid &= finite_flow
    dx = np.where(valid, flow[..., 0], np.nan).astype(np.float32)
    dy = np.where(valid, flow[..., 1], np.nan).astype(np.float32)

    public_parameters = {
        key: value
        for key, value in settings.items()
        if key not in {"predictor", "valid_mask", "_nn_context"}
        and np.isscalar(value)
    }
    return FlowResult(
        dx,
        dy,
        rows,
        cols,
        valid,
        {
            "tracker": model_name.lower().replace("-", "_"),
            "model_name": model_name,
            "mode": output_mode,
            "scale": scale,
            "checkpoint": str(settings.get("checkpoint", "")),
            "tile_size": int(settings.get("tile_size", 600)),
            "overlap": settings.get("overlap", 0.5),
            "batch_size": int(settings.get("batch_size", 1)),
            "device": str(settings.get("device", "cuda:0")),
            "amp": bool(settings.get("amp", False)),
            "input_valid_count": int(original_valid.sum()),
            "inference_valid_count": int(low_valid.sum()),
            "finite_model_output_count": int(finite_flow.sum()),
            "output_valid_count": int(valid.sum()),
            "invalid_values_are_nan": True,
            "parameters": public_parameters,
            **metadata,
        },
    )
