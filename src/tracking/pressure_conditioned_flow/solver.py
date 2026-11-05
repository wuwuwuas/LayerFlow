from __future__ import annotations

from typing import Any, Sequence
import time

import cv2
import numpy as np

from .config import PressureConditionedFlowConfig
from .native import NativePressureConditionedVR, NativeVRParams


def _storage_dtype(name: str) -> np.dtype:
    value = str(name).lower()
    if value in {"float32", "fp32"}:
        return np.dtype(np.float32)
    if value in {"float16", "fp16", "half"}:
        return np.dtype(np.float16)
    raise ValueError(f"Unsupported pyramid_storage_dtype={name!r}")


def _prepare_image(image: np.ndarray, input_dtype: str) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(array)
    if not np.all(finite):
        fill = float(np.nanmedian(array[finite])) if np.any(finite) else 0.0
        array = np.asarray(np.where(finite, array, fill), dtype=np.float32)

    mode = str(input_dtype).lower()
    if mode in {"uint8_0_255", "uint8"}:
        return np.ascontiguousarray(
            np.clip(np.rint(array * 255.0), 0.0, 255.0),
            dtype=np.float32,
        )
    if mode in {"float32_0_255", "float_0_255"}:
        return np.ascontiguousarray(array * 255.0, dtype=np.float32)
    if mode in {"float32_0_1", "float_0_1"}:
        return np.ascontiguousarray(array, dtype=np.float32)
    raise ValueError(f"Unsupported input_dtype={input_dtype!r}")


def _normalized_gaussian(
    image: np.ndarray,
    membership: np.ndarray,
    sigma: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image, dtype=np.float32)
    membership = np.clip(np.asarray(membership, dtype=np.float32), 0.0, 1.0)
    kernel_len = max(1, int(np.floor(3.0 * float(sigma))) * 2 + 1)

    if kernel_len <= 1 or sigma <= 0:
        output = image.copy()
        output[membership <= 0.0] = 0.0
        return (
            np.ascontiguousarray(output, dtype=np.float32),
            np.ascontiguousarray(membership, dtype=np.float32),
        )

    numerator = cv2.GaussianBlur(
        image * membership,
        (kernel_len, kernel_len),
        float(sigma),
        borderType=cv2.BORDER_DEFAULT,
    )
    denominator = cv2.GaussianBlur(
        membership,
        (kernel_len, kernel_len),
        float(sigma),
        borderType=cv2.BORDER_DEFAULT,
    )
    output = numerator / np.maximum(denominator, float(epsilon))
    output[denominator <= float(epsilon)] = 0.0
    return (
        np.ascontiguousarray(output, dtype=np.float32),
        np.ascontiguousarray(denominator, dtype=np.float32),
    )


def _resize_normalized(
    image: np.ndarray,
    membership: np.ndarray,
    size_wh: tuple[int, int],
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    numerator = cv2.resize(
        image * membership,
        size_wh,
        interpolation=cv2.INTER_LINEAR,
    )
    denominator = cv2.resize(
        membership,
        size_wh,
        interpolation=cv2.INTER_AREA,
    )
    output = numerator / np.maximum(denominator, float(epsilon))
    output[denominator <= float(epsilon)] = 0.0
    np.clip(denominator, 0.0, 1.0, out=denominator)
    return (
        np.ascontiguousarray(output, dtype=np.float32),
        np.ascontiguousarray(denominator, dtype=np.float32),
    )


def _build_masked_pyramid(
    image: np.ndarray,
    membership: np.ndarray,
    config: PressureConditionedFlowConfig,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    image0, membership0 = _normalized_gaussian(
        image,
        membership,
        config.sigma,
        config.normalized_blur_epsilon,
    )
    image_pyramid = [image0]
    mask_pyramid = [membership0]

    previous_image = image0
    previous_mask = membership0
    maximum = int(config.max_layers)
    if config.max_pyramid_levels is not None:
        maximum = min(
            maximum,
            max(0, int(config.max_pyramid_levels) - 1),
        )

    for _ in range(maximum):
        height, width = previous_image.shape
        next_width = int(width * float(config.downscale_factor) + 0.5)
        next_height = int(height * float(config.downscale_factor) + 0.5)
        if (
            next_height <= int(config.min_size)
            or next_width <= int(config.min_size)
        ):
            break

        next_image, next_mask = _resize_normalized(
            previous_image,
            previous_mask,
            (next_width, next_height),
            config.normalized_blur_epsilon,
        )
        image_pyramid.append(next_image)
        mask_pyramid.append(next_mask)
        previous_image = next_image
        previous_mask = next_mask

    storage = _storage_dtype(config.pyramid_storage_dtype)
    if storage != np.dtype(np.float32):
        image_pyramid = [
            np.ascontiguousarray(item.astype(storage))
            for item in image_pyramid
        ]
        mask_pyramid = [
            np.ascontiguousarray(item.astype(storage))
            for item in mask_pyramid
        ]

    return image_pyramid, mask_pyramid


def _accumulate_native_profile(
    totals: dict[str, float],
    profile: dict[str, float | int | bool],
) -> None:
    for key, value in profile.items():
        if key.endswith("_seconds"):
            totals[key] = totals.get(key, 0.0) + float(value)


class PressureConditionedFlowSolver:
    """Level-major pressure-conditioned solver using the masked OpenCV 4.13 fork.

    This implementation intentionally retains the exact state-image and
    normalized masked-pyramid construction that produced the validated reference.
    Only the inner variational implementation changes from the Torch reference
    to the editable OpenCV 4.13 masked fork.
    """

    def __init__(
        self,
        config: PressureConditionedFlowConfig | dict[str, Any] | None = None,
    ) -> None:
        if config is None:
            config = PressureConditionedFlowConfig()
        elif isinstance(config, dict):
            valid = {
                key: value
                for key, value in config.items()
                if key in PressureConditionedFlowConfig.__dataclass_fields__
            }
            config = PressureConditionedFlowConfig(**valid)
        self.config = config

    def calc(
        self,
        state_image0: np.ndarray,
        state_image1: np.ndarray,
        *,
        source_membership: np.ndarray,
        target_membership: np.ndarray,
        output_masks: np.ndarray,
        state_names: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        config = self.config
        started = time.perf_counter()

        state_image0 = np.asarray(state_image0, dtype=np.float32)
        state_image1 = np.asarray(state_image1, dtype=np.float32)
        source_membership = np.clip(
            np.asarray(source_membership, dtype=np.float32),
            0.0,
            1.0,
        )
        target_membership = np.clip(
            np.asarray(target_membership, dtype=np.float32),
            0.0,
            1.0,
        )
        output_masks = np.asarray(output_masks, dtype=bool)

        if state_image0.ndim != 3:
            raise ValueError(
                f"state_image0 must be [K,H,W], got {state_image0.shape}"
            )
        if not (
            state_image0.shape
            == state_image1.shape
            == source_membership.shape
            == target_membership.shape
            == output_masks.shape
        ):
            raise ValueError(
                "All state images, memberships, and masks must have [K,H,W] shape"
            )

        state_count, height0, width0 = state_image0.shape
        if state_names is None:
            state_names = [
                f"state_{index}"
                for index in range(state_count)
            ]
        state_names = list(state_names)
        if len(state_names) != state_count:
            raise ValueError("state_names length must equal K")

        image_pyramids0: list[list[np.ndarray]] = []
        image_pyramids1: list[list[np.ndarray]] = []
        mask_pyramids0: list[list[np.ndarray]] = []
        mask_pyramids1: list[list[np.ndarray]] = []

        pyramid_started = time.perf_counter()
        if config.verbose:
            print(
                "[pressure-conditioned-flow] building masked pyramids: "
                f"K={state_count}, finest={(height0, width0)}, "
                f"downscale_factor={config.downscale_factor}, "
                f"storage={config.pyramid_storage_dtype}",
                flush=True,
            )
        pyramid_state_seconds: dict[str, float] = {}
        for state in range(state_count):
            state_pyramid_started = time.perf_counter()
            prepared0 = _prepare_image(
                state_image0[state],
                config.input_dtype,
            )
            prepared1 = _prepare_image(
                state_image1[state],
                config.input_dtype,
            )
            pyr0, mask0 = _build_masked_pyramid(
                prepared0,
                source_membership[state],
                config,
            )
            pyr1, mask1 = _build_masked_pyramid(
                prepared1,
                target_membership[state],
                config,
            )
            if len(pyr0) != len(pyr1):
                raise RuntimeError(
                    f"State {state_names[state]} source/target pyramid lengths differ"
                )
            image_pyramids0.append(pyr0)
            image_pyramids1.append(pyr1)
            mask_pyramids0.append(mask0)
            mask_pyramids1.append(mask1)
            elapsed_state = time.perf_counter() - state_pyramid_started
            pyramid_state_seconds[state_names[state]] = float(elapsed_state)
            if config.verbose:
                estimated_gib = (
                    sum(item.nbytes for item in pyr0)
                    + sum(item.nbytes for item in pyr1)
                    + sum(item.nbytes for item in mask0)
                    + sum(item.nbytes for item in mask1)
                ) / (1024.0 ** 3)
                print(
                    "[pressure-conditioned-flow] pyramid state "
                    f"{state + 1}/{state_count} {state_names[state]}: "
                    f"levels={len(pyr0)}, coarsest={pyr0[-1].shape}, "
                    f"{elapsed_state:.2f}s, stored={estimated_gib:.2f} GiB",
                    flush=True,
                )

        pyramid_build_seconds = (
            time.perf_counter() - pyramid_started
        )
        if config.verbose:
            print(
                "[pressure-conditioned-flow] all masked pyramids ready: "
                f"{pyramid_build_seconds:.2f}s",
                flush=True,
            )
        level_count = len(image_pyramids0[0])
        reference_shapes = [
            item.shape
            for item in image_pyramids0[0]
        ]

        for state in range(state_count):
            if len(image_pyramids0[state]) != level_count:
                raise RuntimeError(
                    "Pressure states have different pyramid lengths"
                )
            if (
                [item.shape for item in image_pyramids0[state]]
                != reference_shapes
            ):
                raise RuntimeError(
                    "Pressure-conditioned states have different pyramid shapes"
                )

        smallest_height, smallest_width = reference_shapes[-1]
        state_runtime = np.zeros(
            state_count,
            dtype=np.float64,
        )
        level_records: list[dict[str, Any]] = []

        vr_params = NativeVRParams(
            alpha=4.0 * float(config.alpha),
            delta=float(config.delta) / 3.0,
            gamma=float(config.gamma) / 3.0,
            fixed_point_iterations=int(
                config.fixed_point_iterations
            ),
            sor_iterations=int(config.sor_iterations),
            omega=float(config.omega),
        )

        native_profile_totals: dict[str, float] = {}
        native_profile_call_count = 0

        with NativePressureConditionedVR(
            vr_params,
            install_dir=config.native_install_dir,
            num_threads=config.opencv_num_threads,
        ) as vr:
            use_uv_interface = bool(
                config.use_native_uv_interface
                and vr.supports_uv_interface
            )
            if config.use_native_uv_interface and not use_uv_interface:
                raise RuntimeError(
                    "use_native_uv_interface=True but the native library does "
                    "not expose the optimized u/v API; rebuild the module"
                )
            if not vr.supports_sparse_sor_control:
                raise RuntimeError(
                    "The native library does not expose sparse masked SOR; "
                    "rebuild the pressure-conditioned native module"
                )
            if not vr.get_sparse_sor_enabled():
                raise RuntimeError(
                    "Sparse masked SOR is unexpectedly disabled in the native solver"
                )

            if use_uv_interface:
                flows_u = [
                    np.zeros(
                        (smallest_height, smallest_width),
                        dtype=np.float32,
                    )
                    for _ in range(state_count)
                ]
                flows_v = [
                    np.zeros(
                        (smallest_height, smallest_width),
                        dtype=np.float32,
                    )
                    for _ in range(state_count)
                ]
                flows = None
            else:
                flows = [
                    np.zeros(
                        (smallest_height, smallest_width, 2),
                        dtype=np.float32,
                    )
                    for _ in range(state_count)
                ]
                flows_u = None
                flows_v = None

            if config.verbose:
                print(
                    "[pressure-conditioned-flow] "
                    f"K={state_count}, levels={level_count}, "
                    f"coarsest={reference_shapes[-1]}, "
                    f"finest={reference_shapes[0]}, "
                    f"OpenCV={vr.opencv_version}, "
                    f"threads={vr.get_num_threads()}, "
                    f"flow_interface={'uv' if use_uv_interface else 'xy'}, "
                    "sparse_sor=True, pressure_diffusion=False",
                    flush=True,
                )

            for level in range(
                level_count - 1,
                -1,
                -1,
            ):
                level_started = time.perf_counter()
                state_level_records = []

                for state in range(state_count):
                    state_started = time.perf_counter()

                    image0_level = np.ascontiguousarray(
                        image_pyramids0[state][level],
                        dtype=np.float32,
                    )
                    image1_level = np.ascontiguousarray(
                        image_pyramids1[state][level],
                        dtype=np.float32,
                    )
                    source_level = np.ascontiguousarray(
                        mask_pyramids0[state][level],
                        dtype=np.float32,
                    )
                    target_level = np.ascontiguousarray(
                        mask_pyramids1[state][level],
                        dtype=np.float32,
                    )

                    if use_uv_interface:
                        assert flows_u is not None and flows_v is not None
                        vr.refine_masked_uv_inplace(
                            image0_level,
                            image1_level,
                            source_level,
                            target_level,
                            flows_u[state],
                            flows_v[state],
                            active_threshold=float(
                                config.solver_active_threshold
                            ),
                            target_invalid_weight=float(
                                config.target_invalid_weight
                            ),
                        )
                    else:
                        assert flows is not None
                        vr.refine_masked_inplace(
                            image0_level,
                            image1_level,
                            source_level,
                            target_level,
                            flows[state],
                            active_threshold=float(
                                config.solver_active_threshold
                            ),
                            target_invalid_weight=float(
                                config.target_invalid_weight
                            ),
                        )

                    native_profile = (
                        vr.last_profile()
                        if config.collect_native_profile
                        else {}
                    )
                    if native_profile:
                        _accumulate_native_profile(
                            native_profile_totals,
                            native_profile,
                        )
                        native_profile_call_count += 1

                    elapsed = (
                        time.perf_counter()
                        - state_started
                    )
                    state_runtime[state] += elapsed

                    if config.collect_level_diagnostics:
                        state_level_records.append(
                            {
                                "state": state_names[state],
                                "runtime_seconds": elapsed,
                                "active_fraction": float(
                                    np.mean(
                                        source_level
                                        >= float(
                                            config.solver_active_threshold
                                        )
                                    )
                                ),
                                "native_profile": native_profile,
                            }
                        )

                level_elapsed = (
                    time.perf_counter() - level_started
                )
                coarse_to_fine_number = (
                    level_count - level
                )
                should_print = (
                    config.verbose
                    and (
                        coarse_to_fine_number == 1
                        or coarse_to_fine_number
                        == level_count
                        or coarse_to_fine_number
                        % max(
                            1,
                            int(config.print_every_levels),
                        )
                        == 0
                    )
                )
                if should_print:
                    print(
                        "[pressure-conditioned-flow] "
                        f"level "
                        f"{coarse_to_fine_number}/"
                        f"{level_count}, "
                        f"shape={reference_shapes[level]}, "
                        f"all_states={level_elapsed:.2f}s",
                        flush=True,
                    )

                if config.collect_level_diagnostics:
                    level_records.append(
                        {
                            "fine_to_coarse_index": int(level),
                            "shape": list(
                                reference_shapes[level]
                            ),
                            "runtime_seconds": (
                                level_elapsed
                            ),
                            "states": state_level_records,
                        }
                    )

                for state in range(state_count):
                    image_pyramids0[state][level] = None
                    image_pyramids1[state][level] = None
                    mask_pyramids0[state][level] = None
                    mask_pyramids1[state][level] = None

                if level > 0:
                    next_height, next_width = (
                        reference_shapes[level - 1]
                    )
                    for state in range(state_count):
                        next_active = (
                            np.asarray(
                                mask_pyramids0[state][level - 1],
                                dtype=np.float32,
                            )
                            >= float(
                                config.solver_active_threshold
                            )
                        )

                        if use_uv_interface:
                            assert flows_u is not None and flows_v is not None
                            old_height, old_width = flows_u[state].shape
                            resized_u = cv2.resize(
                                flows_u[state],
                                (next_width, next_height),
                                interpolation=int(config.interpolation),
                            )
                            resized_v = cv2.resize(
                                flows_v[state],
                                (next_width, next_height),
                                interpolation=int(config.interpolation),
                            )
                            resized_u *= next_width / float(old_width)
                            resized_v *= next_height / float(old_height)
                            resized_u[~next_active] = 0.0
                            resized_v[~next_active] = 0.0
                            flows_u[state] = np.ascontiguousarray(resized_u)
                            flows_v[state] = np.ascontiguousarray(resized_v)
                        else:
                            assert flows is not None
                            old_height, old_width = flows[state].shape[:2]
                            resized = cv2.resize(
                                flows[state],
                                (next_width, next_height),
                                interpolation=int(config.interpolation),
                            )
                            resized[..., 0] *= next_width / float(old_width)
                            resized[..., 1] *= next_height / float(old_height)
                            resized[~next_active] = 0.0
                            flows[state] = np.ascontiguousarray(resized)

            vr.collect_garbage()
            native_identity = vr.identity
            native_library = str(vr.library_path)
            native_threads = int(vr.get_num_threads())
            native_version = vr.opencv_version

        if config.verbose and native_profile_totals:
            native_total = native_profile_totals.get(
                "total_seconds", 0.0
            )
            print(
                "[pressure-conditioned-flow] native profile totals: "
                f"calls={native_profile_call_count}, "
                f"total={native_total:.3f}s, "
                f"split={native_profile_totals.get('wrapper_split_seconds', 0.0):.3f}s, "
                f"merge={native_profile_totals.get('wrapper_merge_seconds', 0.0):.3f}s, "
                f"prepare={native_profile_totals.get('prepare_buffers_seconds', 0.0):.3f}s, "
                f"masked_prepare={native_profile_totals.get('prepare_masked_buffers_seconds', 0.0):.3f}s, "
                f"data_gate={native_profile_totals.get('data_gate_seconds', 0.0):.3f}s, "
                f"data={native_profile_totals.get('data_term_seconds', 0.0):.3f}s, "
                f"smooth_h={native_profile_totals.get('smoothness_horizontal_seconds', 0.0):.3f}s, "
                f"smooth_v={native_profile_totals.get('smoothness_vertical_seconds', 0.0):.3f}s, "
                f"SOR={native_profile_totals.get('sor_seconds', 0.0):.3f}s",
                flush=True,
            )

        if use_uv_interface:
            assert flows_u is not None and flows_v is not None
            flow_stack = np.stack(
                [
                    np.stack((flows_u[state], flows_v[state]), axis=-1)
                    for state in range(state_count)
                ],
                axis=0,
            )
        else:
            assert flows is not None
            flow_stack = np.stack(flows, axis=0)
        flow_stack[~output_masks] = 0.0

        diagnostics = {
            "backend": (
                "private_opencv_4.13_masked_vr_"
                "level_major_uv_optimized"
            ),
            "config": config.metadata(),
            "native_identity": native_identity,
            "native_library": native_library,
            "native_opencv_version": native_version,
            "native_threads": native_threads,
            "native_flow_interface": (
                "uv_inplace" if use_uv_interface else "xy_interleaved"
            ),
            "native_profile_call_count": int(native_profile_call_count),
            "native_profile_totals_seconds": native_profile_totals,
            "state_names": state_names,
            "state_count": int(state_count),
            "input_shape": [
                int(height0),
                int(width0),
            ],
            "pyramid_level_count": int(level_count),
            "pyramid_shapes_fine_to_coarse": [
                list(item)
                for item in reference_shapes
            ],
            "pyramid_build_seconds": float(
                pyramid_build_seconds
            ),
            "pyramid_state_seconds": pyramid_state_seconds,
            "state_runtime_seconds": {
                state_names[index]: float(
                    state_runtime[index]
                )
                for index in range(state_count)
            },
            "levels": level_records,
            "runtime_seconds": float(
                time.perf_counter() - started
            ),
            "pressure_guided_diffusion": False,
            "explicit_pressure_state_images": True,
            "legacy_diagnostic_key": {
                "explicit_state_images_retained_for_gate": True
            },
            "note": (
                "Validated four-state pressure-conditioned construction with "
                "normalized masked pyramids and the editable exact OpenCV "
                "4.13 masked VariationalRefinement fork with direct u/v "
                "in-place flow storage and native phase profiling."
            ),
        }
        return flow_stack, diagnostics


def build_reusable_masked_pyramid(
    image: np.ndarray,
    membership: np.ndarray,
    config: PressureConditionedFlowConfig,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    """Build one prepared masked pyramid for sequential-state execution.

    This is the exact ``_prepare_image`` + ``_build_masked_pyramid`` path used by
    :meth:`PressureConditionedFlowSolver.calc`.  It is exposed so a caller can
    keep one source pyramid while solving more than one target component.
    """

    started = time.perf_counter()
    prepared = _prepare_image(image, config.input_dtype)
    image_pyramid, mask_pyramid = _build_masked_pyramid(
        prepared,
        membership,
        config,
    )
    seconds = time.perf_counter() - started
    stored_bytes = int(
        sum(np.asarray(item).nbytes for item in image_pyramid)
        + sum(np.asarray(item).nbytes for item in mask_pyramid)
    )
    return image_pyramid, mask_pyramid, {
        "seconds": float(seconds),
        "stored_bytes": stored_bytes,
        "stored_gib": float(stored_bytes / (1024.0 ** 3)),
        "level_count": int(len(image_pyramid)),
        "shapes_fine_to_coarse": [list(np.asarray(item).shape) for item in image_pyramid],
    }


def solve_one_state_from_pyramids(
    source_image_pyramid: list[np.ndarray],
    target_image_pyramid: list[np.ndarray],
    source_mask_pyramid: list[np.ndarray],
    target_mask_pyramid: list[np.ndarray],
    *,
    config: PressureConditionedFlowConfig,
    state_name: str,
    target_name: str,
    release_target_levels: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve one independent pressure state from prebuilt pyramids.

    The source pyramid is treated as read-only and may be reused for another
    temporal target.  Target levels may be released as soon as they are used.
    The numerical refinement path is the same coarse-to-fine native masked VR
    path as the production level-major solver, but with ``K=1``.
    """

    started = time.perf_counter()
    level_count = len(source_image_pyramid)
    if level_count == 0:
        raise ValueError("Source pyramid is empty")
    collections = (
        target_image_pyramid,
        source_mask_pyramid,
        target_mask_pyramid,
    )
    if any(len(value) != level_count for value in collections):
        raise ValueError("Source/target image and mask pyramids must share one length")

    reference_shapes = [tuple(np.asarray(item).shape) for item in source_image_pyramid]
    for name, values in (
        ("target_image", target_image_pyramid),
        ("source_mask", source_mask_pyramid),
        ("target_mask", target_mask_pyramid),
    ):
        shapes = [tuple(np.asarray(item).shape) for item in values]
        if shapes != reference_shapes:
            raise ValueError(f"{name} pyramid shapes differ from the source pyramid")

    smallest_height, smallest_width = reference_shapes[-1]
    vr_params = NativeVRParams(
        alpha=4.0 * float(config.alpha),
        delta=float(config.delta) / 3.0,
        gamma=float(config.gamma) / 3.0,
        fixed_point_iterations=int(config.fixed_point_iterations),
        sor_iterations=int(config.sor_iterations),
        omega=float(config.omega),
    )
    native_profile_totals: dict[str, float] = {}
    native_profile_call_count = 0
    level_records: list[dict[str, Any]] = []

    with NativePressureConditionedVR(
        vr_params,
        install_dir=config.native_install_dir,
        num_threads=config.opencv_num_threads,
    ) as vr:
        use_uv_interface = bool(
            config.use_native_uv_interface and vr.supports_uv_interface
        )
        if config.use_native_uv_interface and not use_uv_interface:
            raise RuntimeError(
                "use_native_uv_interface=True but the native library does not "
                "expose the optimized u/v API; rebuild the module"
            )
        if not vr.supports_sparse_sor_control:
            raise RuntimeError(
                "The native library does not expose sparse masked SOR; rebuild "
                "the pressure-conditioned native module"
            )
        if not vr.get_sparse_sor_enabled():
            raise RuntimeError(
                "Sparse masked SOR is unexpectedly disabled in the native solver"
            )

        if use_uv_interface:
            flow_u = np.zeros((smallest_height, smallest_width), dtype=np.float32)
            flow_v = np.zeros((smallest_height, smallest_width), dtype=np.float32)
            flow_xy = None
        else:
            flow_xy = np.zeros(
                (smallest_height, smallest_width, 2), dtype=np.float32
            )
            flow_u = None
            flow_v = None

        if config.verbose:
            print(
                "[pressure-conditioned-flow] sequential solve: "
                f"state={state_name}, target={target_name}, levels={level_count}, "
                f"coarsest={reference_shapes[-1]}, finest={reference_shapes[0]}, "
                f"OpenCV={vr.opencv_version}, threads={vr.get_num_threads()}, "
                f"flow_interface={'uv' if use_uv_interface else 'xy'}, "
                "sparse_sor=True",
                flush=True,
            )

        for level in range(level_count - 1, -1, -1):
            level_started = time.perf_counter()
            image0_level = np.ascontiguousarray(
                source_image_pyramid[level], dtype=np.float32
            )
            image1_level = np.ascontiguousarray(
                target_image_pyramid[level], dtype=np.float32
            )
            source_level = np.ascontiguousarray(
                source_mask_pyramid[level], dtype=np.float32
            )
            target_level = np.ascontiguousarray(
                target_mask_pyramid[level], dtype=np.float32
            )

            if use_uv_interface:
                assert flow_u is not None and flow_v is not None
                vr.refine_masked_uv_inplace(
                    image0_level,
                    image1_level,
                    source_level,
                    target_level,
                    flow_u,
                    flow_v,
                    active_threshold=float(config.solver_active_threshold),
                    target_invalid_weight=float(config.target_invalid_weight),
                )
            else:
                assert flow_xy is not None
                vr.refine_masked_inplace(
                    image0_level,
                    image1_level,
                    source_level,
                    target_level,
                    flow_xy,
                    active_threshold=float(config.solver_active_threshold),
                    target_invalid_weight=float(config.target_invalid_weight),
                )

            native_profile = vr.last_profile() if config.collect_native_profile else {}
            if native_profile:
                _accumulate_native_profile(native_profile_totals, native_profile)
                native_profile_call_count += 1

            level_elapsed = time.perf_counter() - level_started
            coarse_to_fine_number = level_count - level
            should_print = (
                config.verbose
                and (
                    coarse_to_fine_number == 1
                    or coarse_to_fine_number == level_count
                    or coarse_to_fine_number
                    % max(1, int(config.print_every_levels))
                    == 0
                )
            )
            if should_print:
                print(
                    "[pressure-conditioned-flow] "
                    f"state={state_name} target={target_name} level "
                    f"{coarse_to_fine_number}/{level_count}, "
                    f"shape={reference_shapes[level]}, {level_elapsed:.2f}s",
                    flush=True,
                )
            if config.collect_level_diagnostics:
                level_records.append(
                    {
                        "fine_to_coarse_index": int(level),
                        "shape": list(reference_shapes[level]),
                        "runtime_seconds": float(level_elapsed),
                        "active_fraction": float(
                            np.mean(
                                source_level
                                >= float(config.solver_active_threshold)
                            )
                        ),
                        "native_profile": native_profile,
                    }
                )

            if release_target_levels:
                target_image_pyramid[level] = None  # type: ignore[assignment]
                target_mask_pyramid[level] = None  # type: ignore[assignment]

            if level > 0:
                next_height, next_width = reference_shapes[level - 1]
                next_active = (
                    np.asarray(source_mask_pyramid[level - 1], dtype=np.float32)
                    >= float(config.solver_active_threshold)
                )
                if use_uv_interface:
                    assert flow_u is not None and flow_v is not None
                    old_height, old_width = flow_u.shape
                    resized_u = cv2.resize(
                        flow_u,
                        (next_width, next_height),
                        interpolation=int(config.interpolation),
                    )
                    resized_v = cv2.resize(
                        flow_v,
                        (next_width, next_height),
                        interpolation=int(config.interpolation),
                    )
                    resized_u *= next_width / float(old_width)
                    resized_v *= next_height / float(old_height)
                    resized_u[~next_active] = 0.0
                    resized_v[~next_active] = 0.0
                    flow_u = np.ascontiguousarray(resized_u)
                    flow_v = np.ascontiguousarray(resized_v)
                else:
                    assert flow_xy is not None
                    old_height, old_width = flow_xy.shape[:2]
                    resized = cv2.resize(
                        flow_xy,
                        (next_width, next_height),
                        interpolation=int(config.interpolation),
                    )
                    resized[..., 0] *= next_width / float(old_width)
                    resized[..., 1] *= next_height / float(old_height)
                    resized[~next_active] = 0.0
                    flow_xy = np.ascontiguousarray(resized)

        vr.collect_garbage()
        native_identity = vr.identity
        native_library = str(vr.library_path)
        native_threads = int(vr.get_num_threads())
        native_version = vr.opencv_version

    if use_uv_interface:
        assert flow_u is not None and flow_v is not None
        output = np.stack((flow_u, flow_v), axis=-1).astype(np.float32, copy=False)
    else:
        assert flow_xy is not None
        output = np.asarray(flow_xy, dtype=np.float32)

    if config.verbose and native_profile_totals:
        print(
            "[pressure-conditioned-flow] sequential native profile: "
            f"state={state_name}, target={target_name}, "
            f"calls={native_profile_call_count}, "
            f"total={native_profile_totals.get('total_seconds', 0.0):.3f}s, "
            f"SOR={native_profile_totals.get('sor_seconds', 0.0):.3f}s",
            flush=True,
        )

    return output, {
        "backend": "private_opencv_4.13_masked_vr_sequential_state",
        "config": config.metadata(),
        "state_name": state_name,
        "target_name": target_name,
        "native_identity": native_identity,
        "native_library": native_library,
        "native_opencv_version": native_version,
        "native_threads": native_threads,
        "native_flow_interface": (
            "uv_inplace" if use_uv_interface else "xy_interleaved"
        ),
        "native_profile_call_count": int(native_profile_call_count),
        "native_profile_totals_seconds": native_profile_totals,
        "pyramid_level_count": int(level_count),
        "pyramid_shapes_fine_to_coarse": [list(value) for value in reference_shapes],
        "levels": level_records,
        "runtime_seconds": float(time.perf_counter() - started),
        "pressure_guided_diffusion": False,
        "explicit_pressure_state_images": True,
    }
