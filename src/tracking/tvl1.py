"""scikit-image TV-L1 optical-flow tracker adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from models import FlowResult
from .dense_common import flow_diagnostics, make_dense_flow_result, normalize_pair_robust

DEFAULTS = {
    # Best kept from the benchmark: skimage_tvl1_default.
    # These are scikit-image defaults unless the installed version differs.
    "attachment": 15.0,
    "tightness": 0.3,
    "num_warp": 5,
    "num_iter": 10,
    "tol": 1e-4,
    "prefilter": False,
}


def run_tvl1(image0: np.ndarray, image1: np.ndarray, params: dict[str, Any] | None = None) -> FlowResult:
    from skimage.registration import optical_flow_tvl1

    settings = {**DEFAULTS, **(params or {})}
    first = np.asarray(image0, dtype=np.float32)
    second = np.asarray(image1, dtype=np.float32)
    pair = normalize_pair_robust(first, second)
    v, u = optical_flow_tvl1(
        pair["float32"][0],
        pair["float32"][1],
        attachment=float(settings["attachment"]),
        tightness=float(settings["tightness"]),
        num_warp=int(settings["num_warp"]),
        num_iter=int(settings["num_iter"]),
        tol=float(settings["tol"]),
        prefilter=bool(settings["prefilter"]),
    )
    flow = np.stack([u, v], axis=-1).astype(np.float32)
    return make_dense_flow_result(
        first,
        second,
        flow,
        settings,
        tracker="tvl1",
        parameters={k: settings[k] for k in DEFAULTS},
        metadata={"backend": "skimage.registration.optical_flow_tvl1", "diagnostics": {**pair["diagnostics"], **flow_diagnostics(flow)}},
    )
