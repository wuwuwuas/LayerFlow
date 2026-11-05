"""Pressure-layer image construction and layer-flow composition.

This module is preprocessing for the layerwise tracker pipeline.  It does not
run an optical-flow algorithm itself.  The layerwise runner calls a normal
``tracking.run_xxx`` function on each conditioned layer image, then calls
``merge_layer_flows`` to compose the full dense flow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from models import FlowResult


@dataclass(frozen=True)
class PressureLayer:
    """One pressure interval in hPa, using [p_min_hpa, p_max_hpa)."""

    name: str
    p_min_hpa: float
    p_max_hpa: float

    @property
    def center_hpa(self) -> float:
        return 0.5 * (float(self.p_min_hpa) + float(self.p_max_hpa))


DEFAULT_PRESSURE_LAYERS: tuple[PressureLayer, ...] = (
    PressureLayer("high_050_300", 50.0, 300.0),
    PressureLayer("upper_mid_300_500", 300.0, 500.0),
    PressureLayer("lower_mid_500_700", 500.0, 700.0),
    PressureLayer("low_700_1100", 700.0, 1100.0),
)


@dataclass(frozen=True)
class LayerwiseImageConfig:
    """Settings for pressure-conditioned images and layer-flow merging.

    mode:
        hard_select: hard 0/1 layer masks and hard final assignment.
        soft_weighted: legacy center-distance Gaussian weights and soft final average.
        soft_interval: interval weight=1 inside each layer, Gaussian taper outside;
            final output is weighted average of all valid layer flows.
        hard_soft_boundary: interval-soft images, but hard final assignment by
            source-frame pressure layer id. This is usually the most useful AMV mode.
    """

    layers: tuple[PressureLayer, ...] = DEFAULT_PRESSURE_LAYERS
    mode: str = "hard_soft_boundary"
    soft_sigma_hpa: float = 50.0
    soft_min_weight: float = 0.01
    background_sigma_pixels: float = 35.0
    output_erode_pixels: int = 0
    min_weight_sum: float = 0.01
    require_pressure_in_both_frames: bool = False
    hard_fallback_mode: str = "nearest_valid_layer"

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "soft_sigma_hpa": float(self.soft_sigma_hpa),
            "soft_min_weight": float(self.soft_min_weight),
            "background_sigma_pixels": float(self.background_sigma_pixels),
            "output_erode_pixels": int(self.output_erode_pixels),
            "min_weight_sum": float(self.min_weight_sum),
            "require_pressure_in_both_frames": bool(self.require_pressure_in_both_frames),
            "hard_fallback_mode": str(self.hard_fallback_mode),
            "layers": [asdict(layer) for layer in self.layers],
        }


def pressure_layer_id(
    pressure_hpa: np.ndarray,
    layers: Sequence[PressureLayer] = DEFAULT_PRESSURE_LAYERS,
) -> np.ndarray:
    """Return 0-based pressure layer id; -1 means no layer."""

    p = np.asarray(pressure_hpa, dtype=np.float32)
    out = np.full(p.shape, -1, dtype=np.int16)
    for i, layer in enumerate(layers):
        mask = np.isfinite(p) & (p >= layer.p_min_hpa) & (p < layer.p_max_hpa)
        out[mask] = i
    return out


def gaussian_background(image: np.ndarray, sigma_pixels: float) -> np.ndarray:
    """Finite-aware Gaussian smooth background for non-layer pixels."""

    img = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros_like(img, dtype=np.float32)
    fallback = float(np.nanmedian(img[finite]))
    if float(sigma_pixels) <= 0:
        return np.full_like(img, fallback, dtype=np.float32)
    filled = np.where(finite, img, 0.0).astype(np.float32)
    weight = finite.astype(np.float32)
    num = cv2.GaussianBlur(filled, (0, 0), float(sigma_pixels), borderType=cv2.BORDER_REPLICATE)
    den = cv2.GaussianBlur(weight, (0, 0), float(sigma_pixels), borderType=cv2.BORDER_REPLICATE)
    bg = num / np.maximum(den, 1e-6)
    bg[den <= 1e-6] = fallback
    return bg.astype(np.float32)


def erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if int(pixels) <= 0:
        return np.asarray(mask, dtype=bool)
    kernel = np.ones((2 * int(pixels) + 1, 2 * int(pixels) + 1), dtype=np.uint8)
    return cv2.erode(np.asarray(mask, dtype=np.uint8), kernel, iterations=1).astype(bool)


def layer_distance_hpa(pressure_hpa: np.ndarray, layer: PressureLayer) -> np.ndarray:
    """Distance in hPa to a pressure interval; zero inside the interval."""

    p = np.asarray(pressure_hpa, dtype=np.float32)
    d_low = np.maximum(float(layer.p_min_hpa) - p, 0.0)
    d_high = np.maximum(p - float(layer.p_max_hpa), 0.0)
    dist = np.maximum(d_low, d_high).astype(np.float32)
    dist[~np.isfinite(p)] = np.inf
    return dist


def layer_weight(
    pressure_hpa: np.ndarray,
    layer: PressureLayer,
    *,
    mode: str,
    sigma_hpa: float,
    min_weight: float,
) -> np.ndarray:
    """Compute one layer's pressure weight map."""

    p = np.asarray(pressure_hpa, dtype=np.float32)
    if mode == "hard_select":
        return (np.isfinite(p) & (p >= layer.p_min_hpa) & (p < layer.p_max_hpa)).astype(np.float32)
    if mode == "soft_weighted":
        sigma = max(float(sigma_hpa), 1e-3)
        w = np.exp(-0.5 * ((p - layer.center_hpa) / sigma) ** 2).astype(np.float32)
        w[~np.isfinite(p)] = 0.0
        w[w < float(min_weight)] = 0.0
        return w
    if mode in ("soft_interval", "hard_soft_boundary"):
        sigma = max(float(sigma_hpa), 1e-3)
        d = layer_distance_hpa(p, layer)
        w = np.exp(-0.5 * (d / sigma) ** 2).astype(np.float32)
        w[~np.isfinite(d)] = 0.0
        w[w < float(min_weight)] = 0.0
        return w
    raise ValueError("mode must be hard_select, soft_weighted, soft_interval, or hard_soft_boundary")


def condition_image(image: np.ndarray, weight: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Blend original image with smooth background according to a 0--1 layer weight."""

    img = np.asarray(image, dtype=np.float32)
    w = np.clip(np.asarray(weight, dtype=np.float32), 0.0, 1.0)
    bg = np.asarray(background, dtype=np.float32)
    fill_img = np.where(np.isfinite(img), img, bg)
    out = w * fill_img + (1.0 - w) * bg
    out[~np.isfinite(out)] = np.nan
    return out.astype(np.float32)


def build_layer_image_pair(
    image0: np.ndarray,
    image1: np.ndarray,
    pressure0_hpa: np.ndarray,
    pressure1_hpa: np.ndarray,
    layer: PressureLayer,
    config: LayerwiseImageConfig,
    *,
    valid0: np.ndarray | None = None,
    valid1: np.ndarray | None = None,
    background0: np.ndarray | None = None,
    background1: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build one pressure layer's conditioned image pair."""

    first = np.asarray(image0, dtype=np.float32)
    second = np.asarray(image1, dtype=np.float32)
    p0 = np.asarray(pressure0_hpa, dtype=np.float32)
    p1 = np.asarray(pressure1_hpa, dtype=np.float32)
    if first.shape != second.shape or first.shape != p0.shape or first.shape != p1.shape:
        raise ValueError("image and pressure grids must have the same shape")
    base_valid0 = np.isfinite(first) if valid0 is None else (np.asarray(valid0, dtype=bool) & np.isfinite(first))
    base_valid1 = np.isfinite(second) if valid1 is None else (np.asarray(valid1, dtype=bool) & np.isfinite(second))
    bg0 = gaussian_background(first, config.background_sigma_pixels) if background0 is None else np.asarray(background0, dtype=np.float32)
    bg1 = gaussian_background(second, config.background_sigma_pixels) if background1 is None else np.asarray(background1, dtype=np.float32)

    w0 = layer_weight(
        p0,
        layer,
        mode=config.mode,
        sigma_hpa=config.soft_sigma_hpa,
        min_weight=config.soft_min_weight,
    )
    w1 = layer_weight(
        p1,
        layer,
        mode=config.mode,
        sigma_hpa=config.soft_sigma_hpa,
        min_weight=config.soft_min_weight,
    )
    if config.require_pressure_in_both_frames:
        keep0 = w1 > 0
        keep1 = w0 > 0
        w0 = np.where(keep0, w0, 0.0).astype(np.float32)
        w1 = np.where(keep1, w1, 0.0).astype(np.float32)

    layer_img0 = condition_image(first, w0, bg0)
    layer_img1 = condition_image(second, w1, bg1)
    out_mask = (w0 > 0) & base_valid0 & base_valid1 & np.isfinite(layer_img0) & np.isfinite(layer_img1)
    if config.mode in ("hard_select", "hard_soft_boundary"):
        out_mask = erode_mask(out_mask, config.output_erode_pixels)
    return {
        "image0": layer_img0,
        "image1": layer_img1,
        "valid_mask": out_mask,
        "weight0": w0.astype(np.float32),
        "weight1": w1.astype(np.float32),
        "layer": layer,
        "metadata": {"layer": asdict(layer), "valid_pixels": int(out_mask.sum())},
    }


def merge_layer_flows(
    layer_flows: Sequence[FlowResult],
    layer_weights0: Sequence[np.ndarray],
    pressure0_hpa: np.ndarray,
    config: LayerwiseImageConfig,
    *,
    base_valid0: np.ndarray | None = None,
    tracker_name: str = "layerwise",
    metadata: Mapping[str, Any] | None = None,
) -> tuple[FlowResult, np.ndarray]:
    """Compose per-layer FlowResult objects into one full dense FlowResult."""

    if len(layer_flows) != len(config.layers):
        raise ValueError("layer_flows length must match config.layers length")
    first_shape = np.asarray(pressure0_hpa).shape
    p0 = np.asarray(pressure0_hpa, dtype=np.float32)
    base_valid = np.isfinite(p0) if base_valid0 is None else (np.asarray(base_valid0, dtype=bool) & np.isfinite(p0))
    rows, cols = np.indices(first_shape, dtype=np.float32)
    hard_output = config.mode in ("hard_select", "hard_soft_boundary")

    if hard_output:
        lid = pressure_layer_id(p0, config.layers)
        dx = np.full(first_shape, np.nan, dtype=np.float32)
        dy = np.full(first_shape, np.nan, dtype=np.float32)
        valid = np.zeros(first_shape, dtype=bool)
        for i, flow in enumerate(layer_flows):
            take = (lid == i) & flow.valid_mask & np.isfinite(flow.dx_pixels) & np.isfinite(flow.dy_pixels)
            dx[take] = flow.dx_pixels[take]
            dy[take] = flow.dy_pixels[take]
            valid[take] = True

        fallback_mode = str(config.hard_fallback_mode).lower()
        if fallback_mode not in ("none", "nearest_valid_layer"):
            raise ValueError("hard_fallback_mode must be 'none' or 'nearest_valid_layer'")
        if fallback_mode == "nearest_valid_layer":
            need = (~valid) & base_valid
            best_dist = np.full(first_shape, np.inf, dtype=np.float32)
            best_dx = np.full(first_shape, np.nan, dtype=np.float32)
            best_dy = np.full(first_shape, np.nan, dtype=np.float32)
            for layer, flow in zip(config.layers, layer_flows):
                valid_layer = flow.valid_mask & np.isfinite(flow.dx_pixels) & np.isfinite(flow.dy_pixels)
                cand = need & valid_layer
                if not np.any(cand):
                    continue
                dist = layer_distance_hpa(p0, layer)
                take = cand & (dist < best_dist)
                best_dist[take] = dist[take]
                best_dx[take] = flow.dx_pixels[take]
                best_dy[take] = flow.dy_pixels[take]
            take = need & np.isfinite(best_dx) & np.isfinite(best_dy)
            dx[take] = best_dx[take]
            dy[take] = best_dy[take]
            valid[take] = True
        weight_sum = np.where(np.isfinite(p0), 1.0, 0.0).astype(np.float32)
    else:
        lid = pressure_layer_id(p0, config.layers)
        dx_num = np.zeros(first_shape, dtype=np.float32)
        dy_num = np.zeros(first_shape, dtype=np.float32)
        weight_sum = np.zeros(first_shape, dtype=np.float32)
        for weight, flow in zip(layer_weights0, layer_flows):
            valid_layer = flow.valid_mask & np.isfinite(flow.dx_pixels) & np.isfinite(flow.dy_pixels)
            ww = np.where(valid_layer, np.asarray(weight, dtype=np.float32), 0.0)
            dx_num += ww * np.where(valid_layer, flow.dx_pixels, 0.0).astype(np.float32)
            dy_num += ww * np.where(valid_layer, flow.dy_pixels, 0.0).astype(np.float32)
            weight_sum += ww
        valid = weight_sum >= float(config.min_weight_sum)
        dx = np.where(valid, dx_num / np.maximum(weight_sum, 1e-6), np.nan).astype(np.float32)
        dy = np.where(valid, dy_num / np.maximum(weight_sum, 1e-6), np.nan).astype(np.float32)

    valid &= np.isfinite(dx) & np.isfinite(dy)
    flow = FlowResult(
        dx,
        dy,
        rows,
        cols,
        valid,
        {
            "tracker": tracker_name,
            "scale": 1.0,
            "image_mode": "layerwise",
            "layerwise_config": config.metadata(),
            "valid_pixels": int(valid.sum()),
            "weight_sum_finite_fraction": float(np.isfinite(weight_sum).mean()),
            **dict(metadata or {}),
        },
    )
    return flow, lid.astype(np.int16)
