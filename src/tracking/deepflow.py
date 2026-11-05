"""OpenCV DeepFlow adapter returning the shared dense-flow contract."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from models import FlowResult
from .dense_common import flow_diagnostics, make_dense_flow_result, normalize_pair_robust


DEFAULTS = {
    "backend": "opencv",
    "alpha": 1.0,
    "delta": 0.5,
    "gamma": 5.0,
    "sigma": 0.6,
    "min_size": 25,
    "downscale_factor": 0.95,
    "fixed_point_iterations": 5,
    "sor_iterations": 25,
    "omega": 1.6,
    "input_dtype": "uint8_0_255",
    "verbose": False,
}

_OPENCV_ATTRS = {
    "min_size": "minSize",
    "downscale_factor": "downscaleFactor",
    "fixed_point_iterations": "fixedPointIterations",
    "sor_iterations": "sorIterations",
}
_DIRECT_ATTRS = ("alpha", "delta", "gamma", "sigma", "omega")


def _params(settings: dict[str, Any]) -> dict[str, Any]:
    return {k: settings[k] for k in DEFAULTS if k in settings}


def _set_opencv_deepflow_params(obj: Any, settings: dict[str, Any]) -> dict[str, Any]:
    applied: dict[str, Any] = {}
    for key in _DIRECT_ATTRS:
        if key in settings and hasattr(obj, key):
            setattr(obj, key, settings[key])
            applied[key] = settings[key]
        elif key in settings and not np.isclose(settings[key], DEFAULTS[key], rtol=1e-7, atol=0):
            raise ValueError(f"OpenCV's Python DeepFlow API does not expose {key}; use its default {DEFAULTS[key]}")
    for key, attr in _OPENCV_ATTRS.items():
        if key in settings and hasattr(obj, attr):
            setattr(obj, attr, settings[key])
            applied[key] = settings[key]
        elif key in settings and not np.isclose(settings[key], DEFAULTS[key], rtol=1e-7, atol=0):
            raise ValueError(f"OpenCV's Python DeepFlow API does not expose {key}; use its default {DEFAULTS[key]}")
    return applied


def _run_opencv(pair: dict[str, Any], settings: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if not hasattr(cv2, "optflow") or not hasattr(cv2.optflow, "createOptFlow_DeepFlow"):
        raise RuntimeError("cv2.optflow.createOptFlow_DeepFlow is not available. Install opencv-contrib-python.")
    obj = cv2.optflow.createOptFlow_DeepFlow()
    applied = _set_opencv_deepflow_params(obj, settings)
    mode = str(settings.get("input_dtype", "uint8_0_255")).lower()
    if mode in {"uint8_0_255", "uint8"}:
        img0, img1 = pair["uint8"]
        input_dtype_used = "uint8_0_255"
    else:
        img0, img1 = pair["float32"]
        input_dtype_used = "float32_0_1"
    flow = obj.calc(img0, img1, None).astype(np.float32)
    return flow, {
        **pair["diagnostics"],
        **flow_diagnostics(flow),
        "backend": "opencv",
        "factory": "cv2.optflow.createOptFlow_DeepFlow",
        "input_dtype_used": input_dtype_used,
        "opencv_params_applied": applied,
    }






def run_deepflow(image0: np.ndarray, image1: np.ndarray, params: dict[str, Any] | None = None) -> FlowResult:
    settings = {**DEFAULTS, **(params or {})}
    first = np.asarray(image0, dtype=np.float32)
    second = np.asarray(image1, dtype=np.float32)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("DeepFlow inputs must be equally shaped 2-D images")
    pair = normalize_pair_robust(first, second)
    backend = str(settings.get("backend", "opencv")).lower()
    if backend in {"opencv", "cv2"}:
        flow, diag = _run_opencv(pair, settings)
    else:
        raise ValueError("DeepFlow uses the OpenCV backend")
    return make_dense_flow_result(
        first,
        second,
        flow,
        settings,
        tracker="deepflow",
        parameters=_params(settings),
        metadata={"diagnostics": diag, "backend": diag.get("backend", backend)},
    )
