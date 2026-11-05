from __future__ import annotations

from typing import Any
import gc
import time

import numpy as np

from models import FlowResult
from preprocessing.layerwise_image import (
    LayerwiseImageConfig,
    build_layer_image_pair,
    gaussian_background,
    merge_layer_flows,
)
from tracking.dense_common import (
    make_dense_flow_result,
    normalize_pair_robust,
)

from .config import PressureConditionedFlowConfig
from .solver import PressureConditionedFlowSolver


def run_pressure_conditioned_component(
    image0: np.ndarray,
    image1: np.ndarray,
    pressure0_hpa: np.ndarray,
    pressure1_hpa: np.ndarray,
    valid0: np.ndarray,
    valid1: np.ndarray,
    *,
    layer_config: LayerwiseImageConfig,
    output_valid_mask: np.ndarray | None = None,
    solver_config: (
        PressureConditionedFlowConfig
        | dict[str, Any]
    ),
    tracker_name: str = (
        "pressure_conditioned_flow"
    ),
) -> tuple[FlowResult, np.ndarray, dict[str, Any]]:
    """Run one component through the pressure-conditioned masked OpenCV VR kernel.

    This implementation deliberately retains the exact layer-image formula,
    normalization, membership generation, output mask, and merge logic used by
    the validated reference. It isolates only the inner-kernel replacement.
    """

    image0 = np.asarray(image0, dtype=np.float32)
    image1 = np.asarray(image1, dtype=np.float32)
    pressure0_hpa = np.asarray(
        pressure0_hpa,
        dtype=np.float32,
    )
    pressure1_hpa = np.asarray(
        pressure1_hpa,
        dtype=np.float32,
    )
    valid0 = (
        np.asarray(valid0, dtype=bool)
        & np.isfinite(image0)
    )
    valid1 = (
        np.asarray(valid1, dtype=bool)
        & np.isfinite(image1)
    )

    if output_valid_mask is not None:
        output_valid_mask = np.asarray(output_valid_mask, dtype=bool)

    if not (
        image0.shape
        == image1.shape
        == pressure0_hpa.shape
        == pressure1_hpa.shape
        == valid0.shape
        == valid1.shape
    ):
        raise ValueError(
            "Images, pressures, and valid masks "
            "must have identical HxW shapes"
        )
    if output_valid_mask is not None and output_valid_mask.shape != image0.shape:
        raise ValueError(
            "output_valid_mask must have the same HxW shape as the images"
        )

    if isinstance(solver_config, dict):
        config = PressureConditionedFlowConfig(
            **{
                key: value
                for key, value in solver_config.items()
                if key
                in PressureConditionedFlowConfig
                .__dataclass_fields__
            }
        )
    else:
        config = solver_config

    preparation_started = time.perf_counter()
    if config.verbose:
        print(
            "[pressure-conditioned-flow] preparing pressure-conditioned "
            f"states for shape={image0.shape}, K={len(layer_config.layers)}",
            flush=True,
        )

    background_started = time.perf_counter()
    background0 = gaussian_background(
        image0,
        layer_config.background_sigma_pixels,
    )
    background1 = gaussian_background(
        image1,
        layer_config.background_sigma_pixels,
    )
    background_seconds = time.perf_counter() - background_started
    if config.verbose:
        print(
            "[pressure-conditioned-flow] Gaussian backgrounds ready: "
            f"{background_seconds:.2f}s",
            flush=True,
        )

    normalized_state_image0 = []
    normalized_state_image1 = []
    source_memberships = []
    target_memberships = []
    output_masks = []
    source_weights_for_merge = []
    state_names = []
    state_metadata = []
    normalization_diagnostics = []
    state_preparation_seconds: dict[str, float] = {}

    for state_index, layer in enumerate(layer_config.layers, start=1):
        state_started = time.perf_counter()
        pair = build_layer_image_pair(
            image0,
            image1,
            pressure0_hpa,
            pressure1_hpa,
            layer,
            layer_config,
            valid0=valid0,
            valid1=valid1,
            background0=background0,
            background1=background1,
        )
        if (
            "weight0" not in pair
            or "weight1" not in pair
        ):
            raise KeyError(
                "build_layer_image_pair must return "
                "weight0 and weight1"
            )

        first = np.asarray(
            pair["image0"],
            dtype=np.float32,
        )
        second = np.asarray(
            pair["image1"],
            dtype=np.float32,
        )
        normalized = normalize_pair_robust(
            first,
            second,
        )

        source = np.clip(
            np.asarray(
                pair["weight0"],
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
        target = np.clip(
            np.asarray(
                pair["weight1"],
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
        source *= valid0.astype(np.float32)
        target *= valid1.astype(np.float32)

        if output_valid_mask is None:
            output_mask = (
                np.asarray(
                    pair["valid_mask"],
                    dtype=bool,
                )
                & valid0
            )
        else:
            # GOES centre-anchored dense output: target-frame validity still
            # controls the target data term through ``target`` membership, but
            # it does not hard-mask the source-grid output at the same pixel.
            output_mask = (
                np.asarray(output_valid_mask, dtype=bool)
                & valid0
                & (source > 0.0)
                & np.isfinite(first)
            )

        normalized_state_image0.append(
            np.asarray(
                normalized["float32"][0],
                dtype=np.float32,
            )
        )
        normalized_state_image1.append(
            np.asarray(
                normalized["float32"][1],
                dtype=np.float32,
            )
        )
        source_memberships.append(source)
        target_memberships.append(target)
        output_masks.append(output_mask)
        source_weights_for_merge.append(
            np.asarray(
                pair["weight0"],
                dtype=np.float32,
            )
        )
        state_names.append(layer.name)
        state_metadata.append(
            dict(pair.get("metadata", {}))
        )
        normalization_diagnostics.append(
            dict(
                normalized.get(
                    "diagnostics",
                    {},
                )
            )
        )
        state_elapsed = time.perf_counter() - state_started
        state_preparation_seconds[layer.name] = float(state_elapsed)
        if config.verbose:
            print(
                "[pressure-conditioned-flow] prepared state "
                f"{state_index}/{len(layer_config.layers)} "
                f"{layer.name}: {state_elapsed:.2f}s; "
                f"source_active={int(np.count_nonzero(source > 0.0)):,}; "
                f"target_active={int(np.count_nonzero(target > 0.0)):,}; "
                f"output={int(np.count_nonzero(output_mask)):,}",
                flush=True,
            )

    stack_started = time.perf_counter()
    state_image0_stack = np.stack(
        normalized_state_image0,
        axis=0,
    ).astype(np.float32, copy=False)
    state_image1_stack = np.stack(
        normalized_state_image1,
        axis=0,
    ).astype(np.float32, copy=False)
    source_membership_stack = np.stack(
        source_memberships,
        axis=0,
    ).astype(np.float32, copy=False)
    target_membership_stack = np.stack(
        target_memberships,
        axis=0,
    ).astype(np.float32, copy=False)
    output_mask_stack = np.stack(
        output_masks,
        axis=0,
    ).astype(bool, copy=False)
    stack_seconds = time.perf_counter() - stack_started
    preparation_seconds = time.perf_counter() - preparation_started
    if config.verbose:
        stack_gib = (
            state_image0_stack.nbytes
            + state_image1_stack.nbytes
            + source_membership_stack.nbytes
            + target_membership_stack.nbytes
            + output_mask_stack.nbytes
        ) / (1024.0 ** 3)
        print(
            "[pressure-conditioned-flow] state stacks ready: "
            f"{stack_seconds:.2f}s; preparation_total={preparation_seconds:.2f}s; "
            f"stack_memory={stack_gib:.2f} GiB",
            flush=True,
        )

    solver = PressureConditionedFlowSolver(config)
    flow_stack, solver_diagnostics = solver.calc(
        state_image0_stack,
        state_image1_stack,
        source_membership=source_membership_stack,
        target_membership=target_membership_stack,
        output_masks=output_mask_stack,
        state_names=state_names,
    )

    state_flows: list[FlowResult] = []
    for index, state_name in enumerate(state_names):
        flow = make_dense_flow_result(
            image0,
            image1,
            flow_stack[index],
            config.metadata(),
            tracker=tracker_name,
            parameters=config.metadata(),
            metadata={
                "backend": (
                    "private_opencv_4.13_"
                    "masked_vr_level_major"
                ),
                "pressure_state": state_name,
                "state_metadata": (
                    state_metadata[index]
                ),
                "normalization": (
                    normalization_diagnostics[index]
                ),
            },
        )
        flow.valid_mask[:] = (
            np.asarray(
                flow.valid_mask,
                dtype=bool,
            )
            & np.asarray(
                output_mask_stack[index],
                dtype=bool,
            )
            & np.isfinite(
                flow_stack[index, ..., 0]
            )
            & np.isfinite(
                flow_stack[index, ..., 1]
            )
        )
        flow.dx_pixels[~flow.valid_mask] = 0.0
        flow.dy_pixels[~flow.valid_mask] = 0.0
        state_flows.append(flow)

    merged_flow, layer_id = merge_layer_flows(
        state_flows,
        source_weights_for_merge,
        pressure0_hpa,
        layer_config,
        base_valid0=valid0,
        tracker_name=tracker_name,
        metadata={
            "implementation": (
                "one component call; outer pyramid "
                "loop; inner independent pressure-conditioned state "
                "loop; editable OpenCV masked VR"
            ),
            "solver_parameters": config.metadata(),
            "state_names": state_names,
            "pressure_guided_diffusion": False,
        },
    )

    diagnostics = {
        "state_names": state_names,
        "component_preparation_seconds": float(preparation_seconds),
        "background_seconds": float(background_seconds),
        "state_preparation_seconds": state_preparation_seconds,
        "stack_seconds": float(stack_seconds),
        "output_mask_mode": (
            "source_override" if output_valid_mask is not None else "legacy_pair_valid"
        ),
        "state_output_valid_counts": {
            state_names[index]: int(
                state_flows[index].valid_mask.sum()
            )
            for index in range(len(state_names))
        },
        "merged_valid_count": int(
            merged_flow.valid_mask.sum()
        ),
        "solver": solver_diagnostics,
    }
    return merged_flow, layer_id, diagnostics


# -----------------------------------------------------------------------------
# Shared-source, sequential-state execution used by centre-anchored GOES PCF.
# The legacy single-pair function above remains unchanged for SEVIRI.
# -----------------------------------------------------------------------------


def _array_equal_exact(first: np.ndarray, second: np.ndarray) -> bool:
    a = np.asarray(first)
    b = np.asarray(second)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    try:
        return bool(np.array_equal(a, b, equal_nan=True))
    except TypeError:  # NumPy versions before equal_nan support.
        finite_equal = np.array_equal(
            np.where(np.isnan(a), 0, a),
            np.where(np.isnan(b), 0, b),
        )
        return bool(finite_equal and np.array_equal(np.isnan(a), np.isnan(b)))


def _source_difference(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(second, dtype=np.float32)
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return float("nan")
    return float(np.max(np.abs(a[finite] - b[finite])))


def _new_merge_accumulator(
    shape: tuple[int, int],
    *,
    hard_output: bool,
) -> dict[str, np.ndarray]:
    if hard_output:
        return {
            "dx": np.full(shape, np.nan, dtype=np.float32),
            "dy": np.full(shape, np.nan, dtype=np.float32),
            "valid": np.zeros(shape, dtype=bool),
            "best_dist": np.full(shape, np.inf, dtype=np.float32),
            "best_dx": np.full(shape, np.nan, dtype=np.float32),
            "best_dy": np.full(shape, np.nan, dtype=np.float32),
        }
    return {
        "dx_num": np.zeros(shape, dtype=np.float32),
        "dy_num": np.zeros(shape, dtype=np.float32),
        "weight_sum": np.zeros(shape, dtype=np.float32),
    }


def _update_merge_accumulator(
    accumulator: dict[str, np.ndarray],
    *,
    flow_xy: np.ndarray,
    state_valid: np.ndarray,
    source_weight: np.ndarray,
    pressure0_hpa: np.ndarray,
    layer: Any,
    layer_index: int,
    layer_id: np.ndarray,
    base_valid0: np.ndarray,
    layer_config: LayerwiseImageConfig,
) -> None:
    dx = np.asarray(flow_xy[..., 0], dtype=np.float32)
    dy = np.asarray(flow_xy[..., 1], dtype=np.float32)
    valid = (
        np.asarray(state_valid, dtype=bool)
        & np.isfinite(dx)
        & np.isfinite(dy)
    )
    hard_output = layer_config.mode in ("hard_select", "hard_soft_boundary")
    if hard_output:
        direct = (layer_id == int(layer_index)) & valid
        accumulator["dx"][direct] = dx[direct]
        accumulator["dy"][direct] = dy[direct]
        accumulator["valid"][direct] = True

        # Retain only the nearest valid state candidate for the legacy hard
        # fallback.  This replaces storage of all full-resolution state flows.
        from preprocessing.layerwise_image import layer_distance_hpa

        distance = layer_distance_hpa(pressure0_hpa, layer)
        candidate = base_valid0 & valid & (distance < accumulator["best_dist"])
        accumulator["best_dist"][candidate] = distance[candidate]
        accumulator["best_dx"][candidate] = dx[candidate]
        accumulator["best_dy"][candidate] = dy[candidate]
        return

    weight = np.where(
        valid,
        np.asarray(source_weight, dtype=np.float32),
        0.0,
    )
    accumulator["dx_num"] += weight * np.where(valid, dx, 0.0)
    accumulator["dy_num"] += weight * np.where(valid, dy, 0.0)
    accumulator["weight_sum"] += weight


def _finalize_merge_accumulator(
    accumulator: dict[str, np.ndarray],
    *,
    image_shape: tuple[int, int],
    pressure0_hpa: np.ndarray,
    valid0: np.ndarray,
    output_valid_mask: np.ndarray | None,
    layer_config: LayerwiseImageConfig,
    tracker_name: str,
    metadata: dict[str, Any],
) -> FlowResult:
    pressure0 = np.asarray(pressure0_hpa, dtype=np.float32)
    base_valid = np.asarray(valid0, dtype=bool) & np.isfinite(pressure0)
    if output_valid_mask is not None:
        base_valid &= np.asarray(output_valid_mask, dtype=bool)
    hard_output = layer_config.mode in ("hard_select", "hard_soft_boundary")

    if hard_output:
        dx = accumulator["dx"]
        dy = accumulator["dy"]
        valid = accumulator["valid"]
        fallback_mode = str(layer_config.hard_fallback_mode).lower()
        if fallback_mode not in ("none", "nearest_valid_layer"):
            raise ValueError(
                "hard_fallback_mode must be 'none' or 'nearest_valid_layer'"
            )
        if fallback_mode == "nearest_valid_layer":
            need = (~valid) & base_valid
            take = (
                need
                & np.isfinite(accumulator["best_dx"])
                & np.isfinite(accumulator["best_dy"])
            )
            dx[take] = accumulator["best_dx"][take]
            dy[take] = accumulator["best_dy"][take]
            valid[take] = True
        valid &= base_valid & np.isfinite(dx) & np.isfinite(dy)
        weight_sum = np.where(np.isfinite(pressure0), 1.0, 0.0).astype(
            np.float32
        )
    else:
        weight_sum = accumulator["weight_sum"]
        valid = (
            base_valid
            & (weight_sum >= float(layer_config.min_weight_sum))
        )
        dx = np.where(
            valid,
            accumulator["dx_num"] / np.maximum(weight_sum, 1.0e-6),
            np.nan,
        ).astype(np.float32)
        dy = np.where(
            valid,
            accumulator["dy_num"] / np.maximum(weight_sum, 1.0e-6),
            np.nan,
        ).astype(np.float32)
        valid &= np.isfinite(dx) & np.isfinite(dy)

    dx = np.where(valid, dx, np.nan).astype(np.float32)
    dy = np.where(valid, dy, np.nan).astype(np.float32)
    rows, cols = np.indices(image_shape, dtype=np.float32)
    return FlowResult(
        dx_pixels=dx,
        dy_pixels=dy,
        row_center_original=rows,
        col_center_original=cols,
        valid_mask=valid,
        metadata={
            "tracker": tracker_name,
            "scale": 1.0,
            "image_mode": "pressure_conditioned_shared_source_sequential",
            "layerwise_config": layer_config.metadata(),
            "valid_pixels": int(np.count_nonzero(valid)),
            "weight_sum_finite_fraction": float(np.isfinite(weight_sum).mean()),
            **metadata,
        },
    )


def run_pressure_conditioned_shared_source_components(
    image0: np.ndarray,
    pressure0_hpa: np.ndarray,
    valid0: np.ndarray,
    *,
    target_images: dict[str, np.ndarray],
    target_pressures_hpa: dict[str, np.ndarray],
    target_valid_masks: dict[str, np.ndarray],
    layer_config: LayerwiseImageConfig,
    solver_config: PressureConditionedFlowConfig | dict[str, Any],
    output_valid_mask: np.ndarray | None = None,
    tracker_name: str = "pressure_conditioned_flow",
    source_reuse_policy: str = "exact",
) -> tuple[
    dict[str, tuple[FlowResult, np.ndarray, dict[str, Any]]],
    dict[str, Any],
]:
    """Run several centre-anchored components one pressure state at a time.

    The function is generic: one source image is paired with any number of
    named targets.  It keeps at most one pressure state's source/target
    pyramids in memory, merges each solved state immediately, and releases it.

    ``source_reuse_policy='exact'`` reuses a source pyramid only when the
    pair-normalized source image and source membership are bit-for-bit equal for
    every target.  This preserves the established numerical definition.  When
    pair normalization makes them differ, the state remains sequential but the
    source pyramid is rebuilt for each target and the fallback is logged.
    """

    from .solver import (
        build_reusable_masked_pyramid,
        solve_one_state_from_pyramids,
    )
    from preprocessing.layerwise_image import pressure_layer_id

    total_started = time.perf_counter()
    source_image = np.asarray(image0, dtype=np.float32)
    source_pressure = np.asarray(pressure0_hpa, dtype=np.float32)
    source_valid = np.asarray(valid0, dtype=bool) & np.isfinite(source_image)
    shape = source_image.shape
    if source_image.ndim != 2 or source_pressure.shape != shape:
        raise ValueError("Source image and pressure must be equally shaped 2-D arrays")
    if source_valid.shape != shape:
        raise ValueError("Source valid mask must match the source image")
    if output_valid_mask is not None:
        output_valid_mask = np.asarray(output_valid_mask, dtype=bool)
        if output_valid_mask.shape != shape:
            raise ValueError("output_valid_mask must match the source image")

    names = tuple(target_images)
    if not names:
        raise ValueError("At least one target component is required")
    if set(names) != set(target_pressures_hpa) or set(names) != set(
        target_valid_masks
    ):
        raise ValueError("Target image, pressure, and valid-mask keys must match")

    targets: dict[str, dict[str, np.ndarray]] = {}
    for name in names:
        image = np.asarray(target_images[name], dtype=np.float32)
        pressure = np.asarray(target_pressures_hpa[name], dtype=np.float32)
        valid = np.asarray(target_valid_masks[name], dtype=bool) & np.isfinite(image)
        if image.shape != shape or pressure.shape != shape or valid.shape != shape:
            raise ValueError(f"Target {name!r} fields must match source shape {shape}")
        targets[name] = {"image": image, "pressure": pressure, "valid": valid}

    if isinstance(solver_config, dict):
        config = PressureConditionedFlowConfig(
            **{
                key: value
                for key, value in solver_config.items()
                if key in PressureConditionedFlowConfig.__dataclass_fields__
            }
        )
    else:
        config = solver_config
    if config is None:
        raise ValueError("solver_config is required")
    policy = str(source_reuse_policy).strip().lower()
    if policy not in {"exact", "never"}:
        raise ValueError("source_reuse_policy must be 'exact' or 'never'")

    if config.verbose:
        print(
            "[pressure-conditioned-flow] shared-source sequential execution: "
            f"targets={list(names)}, K={len(layer_config.layers)}, shape={shape}, "
            f"source_reuse_policy={policy}",
            flush=True,
        )

    background_started = time.perf_counter()
    source_background = gaussian_background(
        source_image, layer_config.background_sigma_pixels
    )
    target_backgrounds = {
        name: gaussian_background(
            targets[name]["image"], layer_config.background_sigma_pixels
        )
        for name in names
    }
    background_seconds = time.perf_counter() - background_started
    if config.verbose:
        print(
            "[pressure-conditioned-flow] triplet Gaussian backgrounds ready: "
            f"{background_seconds:.2f}s",
            flush=True,
        )

    hard_output = layer_config.mode in ("hard_select", "hard_soft_boundary")
    layer_id = pressure_layer_id(source_pressure, layer_config.layers).astype(
        np.int16
    )
    accumulators = {
        name: _new_merge_accumulator(shape, hard_output=hard_output)
        for name in names
    }
    component_state_counts: dict[str, dict[str, int]] = {
        name: {} for name in names
    }
    component_solver_seconds: dict[str, float] = {name: 0.0 for name in names}
    component_target_pyramid_seconds: dict[str, float] = {
        name: 0.0 for name in names
    }
    component_native: dict[str, dict[str, Any]] = {name: {} for name in names}
    state_rows: list[dict[str, Any]] = []
    source_reuse_count = 0

    for layer_index, layer in enumerate(layer_config.layers):
        state_started = time.perf_counter()
        prepared_by_target: dict[str, dict[str, Any]] = {}
        for name in names:
            pair = build_layer_image_pair(
                source_image,
                targets[name]["image"],
                source_pressure,
                targets[name]["pressure"],
                layer,
                layer_config,
                valid0=source_valid,
                valid1=targets[name]["valid"],
                background0=source_background,
                background1=target_backgrounds[name],
            )
            normalized = normalize_pair_robust(
                np.asarray(pair["image0"], dtype=np.float32),
                np.asarray(pair["image1"], dtype=np.float32),
            )
            source_membership = np.clip(
                np.asarray(pair["weight0"], dtype=np.float32), 0.0, 1.0
            )
            target_membership = np.clip(
                np.asarray(pair["weight1"], dtype=np.float32), 0.0, 1.0
            )
            source_membership *= source_valid.astype(np.float32)
            target_membership *= targets[name]["valid"].astype(np.float32)
            if output_valid_mask is None:
                state_output = (
                    np.asarray(pair["valid_mask"], dtype=bool) & source_valid
                )
            else:
                state_output = (
                    output_valid_mask
                    & source_valid
                    & (source_membership > 0.0)
                    & np.isfinite(pair["image0"])
                )
            prepared_by_target[name] = {
                "source_image": np.asarray(
                    normalized["float32"][0], dtype=np.float32
                ),
                "target_image": np.asarray(
                    normalized["float32"][1], dtype=np.float32
                ),
                "source_membership": source_membership,
                "target_membership": target_membership,
                "output_mask": np.asarray(state_output, dtype=bool),
                "source_weight": np.asarray(pair["weight0"], dtype=np.float32),
                "normalization": dict(normalized.get("diagnostics", {})),
            }

        reference = prepared_by_target[names[0]]
        source_equal = policy == "exact"
        maximum_source_difference = 0.0
        for name in names[1:]:
            current = prepared_by_target[name]
            maximum_source_difference = max(
                maximum_source_difference,
                _source_difference(
                    reference["source_image"], current["source_image"]
                ),
            )
            source_equal &= _array_equal_exact(
                reference["source_image"], current["source_image"]
            ) and _array_equal_exact(
                reference["source_membership"], current["source_membership"]
            )
        reuse_source = bool(source_equal and policy == "exact")
        if reuse_source:
            source_reuse_count += 1

        if config.verbose:
            print(
                "[pressure-conditioned-flow] state "
                f"{layer_index + 1}/{len(layer_config.layers)} {layer.name}: "
                f"source_pyramid_reuse={'yes' if reuse_source else 'no'}; "
                f"max_pair_normalized_source_difference="
                f"{maximum_source_difference:.8g}",
                flush=True,
            )

        shared_source_image_pyramid = None
        shared_source_mask_pyramid = None
        shared_source_diag: dict[str, Any] | None = None
        if reuse_source:
            (
                shared_source_image_pyramid,
                shared_source_mask_pyramid,
                shared_source_diag,
            ) = build_reusable_masked_pyramid(
                reference["source_image"],
                reference["source_membership"],
                config,
            )
            if config.verbose:
                print(
                    "[pressure-conditioned-flow] shared source pyramid ready: "
                    f"state={layer.name}, {shared_source_diag['seconds']:.2f}s, "
                    f"stored={shared_source_diag['stored_gib']:.2f} GiB",
                    flush=True,
                )

        state_target_rows: dict[str, Any] = {}
        for name in names:
            prepared = prepared_by_target[name]
            source_image_pyramid = shared_source_image_pyramid
            source_mask_pyramid = shared_source_mask_pyramid
            source_diag = shared_source_diag
            if not reuse_source:
                (
                    source_image_pyramid,
                    source_mask_pyramid,
                    source_diag,
                ) = build_reusable_masked_pyramid(
                    prepared["source_image"],
                    prepared["source_membership"],
                    config,
                )
            assert source_image_pyramid is not None
            assert source_mask_pyramid is not None
            assert source_diag is not None

            (
                target_image_pyramid,
                target_mask_pyramid,
                target_diag,
            ) = build_reusable_masked_pyramid(
                prepared["target_image"],
                prepared["target_membership"],
                config,
            )
            component_target_pyramid_seconds[name] += float(
                target_diag["seconds"]
            )
            if config.verbose:
                print(
                    "[pressure-conditioned-flow] target pyramid ready: "
                    f"state={layer.name}, target={name}, "
                    f"{target_diag['seconds']:.2f}s, "
                    f"stored={target_diag['stored_gib']:.2f} GiB",
                    flush=True,
                )

            flow_xy, solve_diag = solve_one_state_from_pyramids(
                source_image_pyramid,
                target_image_pyramid,
                source_mask_pyramid,
                target_mask_pyramid,
                config=config,
                state_name=layer.name,
                target_name=name,
                release_target_levels=True,
            )
            state_valid = (
                prepared["output_mask"]
                & np.isfinite(flow_xy[..., 0])
                & np.isfinite(flow_xy[..., 1])
            )
            _update_merge_accumulator(
                accumulators[name],
                flow_xy=flow_xy,
                state_valid=state_valid,
                source_weight=prepared["source_weight"],
                pressure0_hpa=source_pressure,
                layer=layer,
                layer_index=layer_index,
                layer_id=layer_id,
                base_valid0=source_valid,
                layer_config=layer_config,
            )
            component_state_counts[name][layer.name] = int(
                np.count_nonzero(state_valid)
            )
            component_solver_seconds[name] += float(solve_diag["runtime_seconds"])
            component_native[name][layer.name] = solve_diag
            state_target_rows[name] = {
                "source_pyramid_seconds": float(source_diag["seconds"]),
                "target_pyramid_seconds": float(target_diag["seconds"]),
                "solve_seconds": float(solve_diag["runtime_seconds"]),
                "valid_count": int(np.count_nonzero(state_valid)),
            }

            del flow_xy, target_image_pyramid, target_mask_pyramid
            if not reuse_source:
                del source_image_pyramid, source_mask_pyramid

        del shared_source_image_pyramid, shared_source_mask_pyramid
        del prepared_by_target
        gc.collect()
        state_rows.append(
            {
                "state": layer.name,
                "source_reused": reuse_source,
                "max_pair_normalized_source_difference": float(
                    maximum_source_difference
                ),
                "targets": state_target_rows,
                "runtime_seconds": float(time.perf_counter() - state_started),
            }
        )
        if config.verbose:
            print(
                "[pressure-conditioned-flow] released state "
                f"{layer_index + 1}/{len(layer_config.layers)} {layer.name}; "
                f"elapsed={(time.perf_counter() - state_started)/60.0:.2f} min",
                flush=True,
            )

    results: dict[str, tuple[FlowResult, np.ndarray, dict[str, Any]]] = {}
    total_seconds = time.perf_counter() - total_started
    for name in names:
        merged = _finalize_merge_accumulator(
            accumulators[name],
            image_shape=shape,
            pressure0_hpa=source_pressure,
            valid0=source_valid,
            output_valid_mask=output_valid_mask,
            layer_config=layer_config,
            tracker_name=tracker_name,
            metadata={
                "implementation": (
                    "shared-source centre-anchored components; sequential "
                    "pressure-state pyramid build/solve/release"
                ),
                "solver_parameters": config.metadata(),
                "state_names": [layer.name for layer in layer_config.layers],
                "source_reuse_policy": policy,
                "source_reused_state_count": int(source_reuse_count),
                "target_component": name,
                "pressure_guided_diffusion": False,
            },
        )
        diagnostics = {
            "state_names": [layer.name for layer in layer_config.layers],
            "state_output_valid_counts": component_state_counts[name],
            "merged_valid_count": int(np.count_nonzero(merged.valid_mask)),
            "component_preparation_seconds": float(background_seconds),
            "source_reuse_policy": policy,
            "source_reused_state_count": int(source_reuse_count),
            "sequential_state_rows": state_rows,
            "solver": {
                "backend": "sequential_state_shared_source",
                "runtime_seconds": float(component_solver_seconds[name]),
                "target_pyramid_build_seconds": float(
                    component_target_pyramid_seconds[name]
                ),
                "pyramid_build_seconds": float(
                    sum(
                        row["targets"][name]["source_pyramid_seconds"]
                        + row["targets"][name]["target_pyramid_seconds"]
                        for row in state_rows
                    )
                ),
                "pyramid_level_count": int(
                    next(iter(component_native[name].values()))[
                        "pyramid_level_count"
                    ]
                ),
                "state_runtime_seconds": {
                    state: float(value["runtime_seconds"])
                    for state, value in component_native[name].items()
                },
                "native_threads": int(
                    next(iter(component_native[name].values()))[
                        "native_threads"
                    ]
                ),
                "native_opencv_version": next(
                    iter(component_native[name].values())
                )["native_opencv_version"],
                "per_state": component_native[name],
            },
        }
        results[name] = (merged, layer_id.copy(), diagnostics)

    shared_diagnostics = {
        "runtime_seconds": float(total_seconds),
        "background_seconds": float(background_seconds),
        "source_reuse_policy": policy,
        "source_reused_state_count": int(source_reuse_count),
        "state_count": int(len(layer_config.layers)),
        "state_rows": state_rows,
    }
    return results, shared_diagnostics
