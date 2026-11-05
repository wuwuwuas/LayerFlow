"""Configuration for pressure-conditioned dense optical flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from preprocessing.layerwise_image import LayerwiseImageConfig, PressureLayer


@dataclass
class PressureConditionedFlowConfig:
    """Parameters of the validated pressure-conditioned solver.

    The numerical defaults reproduce the legacy
    ``pressure_state_opencv_masked_v1`` implementation. ``native_install_dir``
    may point either to the private OpenCV installation root or directly to its
    ``lib``/``lib64`` directory. When it is ``None``, the native loader searches
    the environment and repository-local default locations.
    """

    # DeepFlow / VariationalRefinement outer parameters.
    alpha: float = 1.0
    delta: float = 0.5
    gamma: float = 5.0
    sigma: float = 0.6
    min_size: int = 25
    downscale_factor: float = 0.95
    fixed_point_iterations: int = 5
    sor_iterations: int = 25
    omega: float = 1.6
    max_layers: int = 200
    max_pyramid_levels: int | None = None
    interpolation: int = cv2.INTER_LINEAR

    # Exact validated masked-pyramid construction.
    input_dtype: str = "uint8_0_255"
    solver_active_threshold: float = 0.01
    normalized_blur_epsilon: float = 1.0e-4
    target_invalid_weight: float = 0.0
    pyramid_storage_dtype: str = "float32"

    # Native implementation settings.
    opencv_num_threads: int | None = None
    native_install_dir: str | Path | None = None
    use_native_uv_interface: bool = True
    collect_native_profile: bool = True

    # Diagnostics.
    verbose: bool = True
    print_every_levels: int = 10
    collect_level_diagnostics: bool = False

    def metadata(self) -> dict[str, Any]:
        output = asdict(self)
        if output["native_install_dir"] is not None:
            output["native_install_dir"] = str(output["native_install_dir"])
        return output


def default_pressure_layer_config() -> LayerwiseImageConfig:
    """Return the four pressure states used by the validated baseline."""

    return LayerwiseImageConfig(
        mode="hard_soft_boundary",
        soft_sigma_hpa=50.0,
        soft_min_weight=0.01,
        background_sigma_pixels=35.0,
        output_erode_pixels=0,
        min_weight_sum=0.01,
        require_pressure_in_both_frames=False,
        hard_fallback_mode="nearest_valid_layer",
        layers=(
            PressureLayer("high_050_300", 50.0, 300.0),
            PressureLayer("upper_mid_300_500", 300.0, 500.0),
            PressureLayer("lower_mid_500_700", 500.0, 700.0),
            PressureLayer("low_700_1100", 700.0, 1100.0),
        ),
    )
