"""scikit-image iterative Lucas-Kanade optical-flow tracker adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from models import FlowResult
from .dense_common import flow_diagnostics, make_dense_flow_result, normalize_pair_robust

DEFAULTS = {
    # Best kept from benchmark: skimage_ilk_large_radius.
    "radius": 11,
    "num_warp": 5,
    "gaussian": True,
    "prefilter": True,
}


def run_ilk(image0: np.ndarray, image1: np.ndarray, params: dict[str, Any] | None = None) -> FlowResult:
    from skimage.registration import optical_flow_ilk

    settings = {**DEFAULTS, **(params or {})}
    first = np.asarray(image0, dtype=np.float32)
    second = np.asarray(image1, dtype=np.float32)
    pair = normalize_pair_robust(first, second)
    v, u = optical_flow_ilk(
        pair["float32"][0],
        pair["float32"][1],
        radius=int(settings["radius"]),
        num_warp=int(settings["num_warp"]),
        gaussian=bool(settings["gaussian"]),
        prefilter=bool(settings["prefilter"]),
    )
    flow = np.stack([u, v], axis=-1).astype(np.float32)
    return make_dense_flow_result(
        first,
        second,
        flow,
        settings,
        tracker="ilk",
        parameters={k: settings[k] for k in DEFAULTS},
        metadata={"backend": "skimage.registration.optical_flow_ilk", "diagnostics": {**pair["diagnostics"], **flow_diagnostics(flow)}},
    )
