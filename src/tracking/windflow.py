"""WindFlow tracker entry point (Vandal et al., KDD 2022)."""

from __future__ import annotations

from typing import Any

import numpy as np

from models import FlowResult
from .nn_common import run_nn_flow


def run_windflow(
    image0: np.ndarray,
    image1: np.ndarray,
    params: dict[str, Any] | None = None,
) -> FlowResult:
    """Run the public WindFlow RAFT checkpoint through FlowResult."""

    return run_nn_flow(image0, image1, params, model_name="WindFlow")


# The generic pipelines use this marker to pass non-serializable in-memory context.
run_windflow.requires_nn_context = True  # type: ignore[attr-defined]
