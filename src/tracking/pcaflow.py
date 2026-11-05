"""OpenCV PCAFlow tracker adapter."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from models import FlowResult
from .dense_common import flow_diagnostics, make_dense_flow_result, normalize_pair_robust

DEFAULTS = {}


def run_pcaflow(image0: np.ndarray, image1: np.ndarray, params: dict[str, Any] | None = None) -> FlowResult:
    settings = {**DEFAULTS, **(params or {})}
    first = np.asarray(image0, dtype=np.float32)
    second = np.asarray(image1, dtype=np.float32)
    pair = normalize_pair_robust(first, second)
    if not hasattr(cv2, "optflow") or not hasattr(cv2.optflow, "createOptFlow_PCAFlow"):
        raise RuntimeError("cv2.optflow.createOptFlow_PCAFlow is not available. Install opencv-contrib-python.")
    obj = cv2.optflow.createOptFlow_PCAFlow()
    flow = obj.calc(pair["uint8"][0], pair["uint8"][1], None).astype(np.float32)
    return make_dense_flow_result(
        first,
        second,
        flow,
        settings,
        tracker="pcaflow",
        parameters={},
        metadata={"backend": "opencv", "diagnostics": {**pair["diagnostics"], **flow_diagnostics(flow)}},
    )
