"""OpenCV Farneback adapter returning the shared dense-flow contract."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from models import FlowResult
from preprocessing import normalize_image_pair
from .dense_common import input_valid_mask, make_dense_flow_result


DEFAULTS = {
    "backend": "opencv",
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 25,
    "iterations": 3,
    "poly_n": 7,
    "poly_sigma": 1.1,
    "flags": 0,
}

FARNEBACK_PARAM_KEYS = (
    "pyr_scale",
    "levels",
    "winsize",
    "iterations",
    "poly_n",
    "poly_sigma",
    "flags",
)


def _algorithm_params(settings: dict[str, Any]) -> dict[str, Any]:
    return {name: settings[name] for name in FARNEBACK_PARAM_KEYS if name in settings}


def _normalize_for_farneback(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = normalize_image_pair(
        first,
        second,
        output_dtype="uint8",
        valid0=np.isfinite(first),
        valid1=np.isfinite(second),
    )
    return normalized[0], normalized[1]


def _run_opencv_farneback(first: np.ndarray, second: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    img0, img1 = _normalize_for_farneback(first, second)
    return cv2.calcOpticalFlowFarneback(
        img0,
        img1,
        None,
        float(settings["pyr_scale"]),
        int(settings["levels"]),
        int(settings["winsize"]),
        int(settings["iterations"]),
        int(settings["poly_n"]),
        float(settings["poly_sigma"]),
        int(settings["flags"]),
    ).astype(np.float32)




def run_farneback(
    image0: np.ndarray, image1: np.ndarray, params: dict[str, Any] | None = None
) -> FlowResult:
    """Run Farneback and return displacement in original image pixel units."""

    settings = {**DEFAULTS, **(params or {})}
    first = np.asarray(image0)
    second = np.asarray(image1)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Farneback inputs must be equally shaped 2-D images")

    backend = str(settings.get("backend", "opencv")).lower()
    if backend in {"opencv", "cv2"}:
        flow = _run_opencv_farneback(first, second, settings)
        backend_name = "opencv"
    else:
        raise ValueError("Farneback uses the OpenCV backend")

    result = make_dense_flow_result(
        first,
        second,
        flow,
        settings,
        tracker="farneback",
        parameters={**_algorithm_params(settings), "backend": backend_name},
        metadata={"backend": backend_name},
    )
    # Keep exact legacy metadata keys where practical.
    result.metadata["scale"] = 1.0
    return result
