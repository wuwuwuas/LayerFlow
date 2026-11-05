"""OpenCV Dual TV-L1 optical-flow tracker adapter."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from models import FlowResult
from .dense_common import flow_diagnostics, make_dense_flow_result, normalize_pair_robust

DEFAULTS = {
    # Best kept from the benchmark: dualtvl1_smooth.
    "lambda": 0.10,
    "theta": 0.30,
    "nscales": 5,
    "warps": 5,
    "epsilon": 0.01,
    "medianFiltering": 5,
}


def _create_dualtvl1():
    factories = (
        lambda: cv2.optflow.DualTVL1OpticalFlow_create(),
        lambda: cv2.optflow.createOptFlow_DualTVL1(),
        lambda: cv2.DualTVL1OpticalFlow_create(),
    )
    for factory in factories:
        try:
            return factory()
        except Exception:
            pass
    raise RuntimeError("DualTVL1 OpenCV factory is not available. Install opencv-contrib-python if needed.")


def _set_if_exists(obj, names, value):
    for name in names:
        method = getattr(obj, name, None)
        if method is None:
            continue
        try:
            method(value)
            return name
        except Exception:
            continue
    return None


def run_dualtvl1(image0: np.ndarray, image1: np.ndarray, params: dict[str, Any] | None = None) -> FlowResult:
    settings = {**DEFAULTS, **(params or {})}
    first = np.asarray(image0, dtype=np.float32)
    second = np.asarray(image1, dtype=np.float32)
    pair = normalize_pair_robust(first, second)
    obj = _create_dualtvl1()
    applied = {
        "lambda": _set_if_exists(obj, ("setLambda", "setLambda_"), float(settings["lambda"])),
        "theta": _set_if_exists(obj, ("setTheta",), float(settings["theta"])),
        "nscales": _set_if_exists(obj, ("setScalesNumber", "setNScales"), int(settings["nscales"])),
        "warps": _set_if_exists(obj, ("setWarpingsNumber", "setWarps"), int(settings["warps"])),
        "epsilon": _set_if_exists(obj, ("setEpsilon",), float(settings["epsilon"])),
        "medianFiltering": _set_if_exists(obj, ("setMedianFiltering",), int(settings["medianFiltering"])),
    }
    flow = obj.calc(pair["float32"][0], pair["float32"][1], None).astype(np.float32)
    return make_dense_flow_result(
        first,
        second,
        flow,
        settings,
        tracker="dualtvl1",
        parameters={k: settings[k] for k in DEFAULTS},
        metadata={"backend": "opencv", "setters_applied": applied, "diagnostics": {**pair["diagnostics"], **flow_diagnostics(flow)}},
    )
