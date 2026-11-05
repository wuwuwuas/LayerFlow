"""OpenCV DIS dense optical-flow tracker adapter."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from models import FlowResult
from .dense_common import flow_diagnostics, make_dense_flow_result, normalize_pair_robust

DEFAULTS = {"preset": "medium"}


def _preset_value(name: str) -> int:
    key = str(name).lower()
    mapping = {
        "ultrafast": "DISOpticalFlow_PRESET_ULTRAFAST",
        "fast": "DISOpticalFlow_PRESET_FAST",
        "medium": "DISOpticalFlow_PRESET_MEDIUM",
    }
    attr = mapping.get(key)
    if attr is None or not hasattr(cv2, attr):
        raise RuntimeError(f"DIS preset is not available: {name!r}")
    return int(getattr(cv2, attr))


def run_dis(image0: np.ndarray, image1: np.ndarray, params: dict[str, Any] | None = None) -> FlowResult:
    settings = {**DEFAULTS, **(params or {})}
    first = np.asarray(image0, dtype=np.float32)
    second = np.asarray(image1, dtype=np.float32)
    pair = normalize_pair_robust(first, second)
    if not hasattr(cv2, "DISOpticalFlow_create"):
        raise RuntimeError("cv2.DISOpticalFlow_create is not available")
    obj = cv2.DISOpticalFlow_create(_preset_value(settings["preset"]))
    flow = obj.calc(pair["uint8"][0], pair["uint8"][1], None).astype(np.float32)
    return make_dense_flow_result(
        first,
        second,
        flow,
        settings,
        tracker="dis",
        parameters={"preset": settings["preset"]},
        metadata={"backend": "opencv", "diagnostics": {**pair["diagnostics"], **flow_diagnostics(flow)}},
    )
