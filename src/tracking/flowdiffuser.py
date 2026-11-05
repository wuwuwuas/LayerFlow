"""FlowDiffuser tracker entry point."""

from __future__ import annotations

from typing import Any

import numpy as np

from models import FlowResult
from .nn_common import run_nn_flow


def run_flowdiffuser(
    image0: np.ndarray,
    image1: np.ndarray,
    params: dict[str, Any] | None = None,
) -> FlowResult:
    """Run the retained single-seed FlowDiffuser through FlowResult."""

    return run_nn_flow(image0, image1, params, model_name="FlowDiffuser")


# The generic pipelines use this marker to pass non-serializable in-memory context.
run_flowdiffuser.requires_nn_context = True  # type: ignore[attr-defined]
