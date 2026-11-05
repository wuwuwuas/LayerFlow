from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TiledSettings:
    tile_size: int = 512
    overlap: int = 128
    batch_size: int = 4
    seed: int = 0
    min_tile_output_fraction: float = 0.0005
    max_tile_flow_px: float = 100.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tile_positions(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0 or tile_size > length:
        raise ValueError(f"Invalid tile_size={tile_size} for length={length}")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    if length == tile_size:
        return [0]
    stride = tile_size - overlap
    count = int(math.ceil((length - tile_size) / stride)) + 1
    return np.rint(np.linspace(0, length - tile_size, count)).astype(int).tolist()


def _axis_weight(size: int, overlap: int, at_start: bool, at_end: bool) -> np.ndarray:
    weight = np.ones(size, dtype=np.float32)
    n = min(int(overlap), size // 2)
    if n <= 0:
        return weight
    ramp = 1.0e-3 + (1.0 - 1.0e-3) * 0.5 * (
        1.0 - np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32))
    )
    if not at_start:
        weight[:n] = ramp
    if not at_end:
        weight[-n:] = ramp[::-1]
    return weight


def tile_weight(size: int, overlap: int, y: int, x: int, h: int, w: int) -> np.ndarray:
    return _axis_weight(size, overlap, y == 0, y + size == h)[:, None] * _axis_weight(
        size, overlap, x == 0, x + size == w
    )[None, :]


def normalization_parameters(
    image0: np.ndarray,
    image1: np.ndarray,
    valid0: np.ndarray,
    valid1: np.ndarray,
) -> dict[str, float]:
    first = np.asarray(image0, dtype=np.float32)
    second = np.asarray(image1, dtype=np.float32)
    if first.ndim != 2 or second.ndim != 2 or first.shape != second.shape:
        raise ValueError("Normalization reference images must be equally shaped 2-D arrays")
    v0 = np.asarray(valid0, dtype=bool) & np.isfinite(first)
    v1 = np.asarray(valid1, dtype=bool) & np.isfinite(second)
    if v0.shape != first.shape or v1.shape != second.shape:
        raise ValueError("Normalization masks must match their reference images")
    values = np.concatenate([first[v0], second[v1]])
    if values.size == 0:
        raise ValueError("Pair has no normalization-support pixels")
    lo, hi = np.nanpercentile(values, [1.0, 99.0])
    lo, hi = float(lo), float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if hi <= lo:
        hi = lo + 1.0
    fill = float(np.nanmedian(values))
    return {"low": lo, "high": hi, "fill": fill}


def apply_scene_normalization_uint8(
    image: np.ndarray,
    normalization: Mapping[str, float],
) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {arr.shape}")
    lo = float(normalization["low"])
    hi = float(normalization["high"])
    fill = float(normalization["fill"])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo or not np.isfinite(fill):
        raise ValueError(f"Invalid normalization parameters: {dict(normalization)}")
    arr = np.where(np.isfinite(arr), arr, fill)
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(arr * 255.0).astype(np.uint8)


def normalize_scene_pair_uint8(
    image0: np.ndarray,
    image1: np.ndarray,
    valid0: np.ndarray,
    valid1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    normalization = normalization_parameters(image0, image1, valid0, valid1)
    return (
        apply_scene_normalization_uint8(image0, normalization),
        apply_scene_normalization_uint8(image1, normalization),
        normalization,
    )


def _as_chw_uint8(tile: np.ndarray) -> np.ndarray:
    array = np.asarray(tile, dtype=np.uint8)
    if array.ndim == 2:
        return np.repeat(array[None, ...], 3, axis=0)
    if array.ndim == 3 and array.shape[-1] == 3:
        return np.ascontiguousarray(np.moveaxis(array, -1, 0))
    raise ValueError(
        "Each model input must be either HxW grayscale or HxWx3 channel-last uint8; "
        f"got shape {array.shape}"
    )


def _image_batch(tiles: list[np.ndarray], device: torch.device) -> torch.Tensor:
    if not tiles:
        raise ValueError("Cannot build an empty tile batch")
    array = np.stack([_as_chw_uint8(tile) for tile in tiles], axis=0)
    return torch.from_numpy(array).float().to(device, non_blocking=True)


class _LocalPadder:
    def __init__(self, shape: torch.Size) -> None:
        h, w = int(shape[-2]), int(shape[-1])
        pad_h = (((h // 8) + 1) * 8 - h) % 8
        pad_w = (((w // 8) + 1) * 8 - w) % 8
        self.pad_values = [
            pad_w // 2,
            pad_w - pad_w // 2,
            pad_h // 2,
            pad_h - pad_h // 2,
        ]

    def pad_images(self, *inputs: torch.Tensor) -> list[torch.Tensor]:
        return [F.pad(value, self.pad_values, mode="replicate") for value in inputs]

    def unpad(self, value: torch.Tensor) -> torch.Tensor:
        left, right, top, bottom = self.pad_values
        h, w = value.shape[-2:]
        return value[
            ...,
            top : h - bottom if bottom else h,
            left : w - right if right else w,
        ]


def flow_statistics(flow: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    mag = np.linalg.norm(np.asarray(flow, dtype=np.float32), axis=-1)
    mask = np.isfinite(mag) & np.asarray(valid, dtype=bool)
    values = mag[mask]
    if values.size == 0:
        return {
            "mean": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "max": float(np.max(values)),
    }


@torch.inference_mode()
def infer_tiles_batched(
    *,
    predict: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    image0_u8: np.ndarray,
    image1_u8: np.ndarray,
    output_mask: np.ndarray,
    device: torch.device,
    settings: TiledSettings,
    component_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    first = np.asarray(image0_u8, dtype=np.uint8)
    second = np.asarray(image1_u8, dtype=np.uint8)
    output = np.asarray(output_mask, dtype=bool)
    if first.shape != second.shape or first.ndim != 2 or output.shape != first.shape:
        raise ValueError(
            "FlowDiffuser grayscale image pairs and output_mask must share one HxW shape"
        )
    h, w = first.shape
    ys = tile_positions(h, settings.tile_size, settings.overlap)
    xs = tile_positions(w, settings.tile_size, settings.overlap)
    records: list[tuple[int, int]] = []
    skipped = 0
    for y in ys:
        for x in xs:
            sl = np.s_[y : y + settings.tile_size, x : x + settings.tile_size]
            if float(np.mean(output[sl])) < settings.min_tile_output_fraction:
                skipped += 1
                continue
            records.append((y, x))

    sum_u = np.zeros((h, w), dtype=np.float32)
    sum_v = np.zeros((h, w), dtype=np.float32)
    sum_w = np.zeros((h, w), dtype=np.float32)
    started = time.perf_counter()
    max_seen = 0.0
    n_batches = math.ceil(len(records) / settings.batch_size) if records else 0
    print(
        f"  useful tiles={len(records)}, skipped={skipped}, "
        f"batch_size={settings.batch_size}, input_channels=1"
    )

    for batch_start in range(0, len(records), settings.batch_size):
        batch_records = records[batch_start : batch_start + settings.batch_size]
        tiles0, tiles1 = [], []
        for y, x in batch_records:
            sl = np.s_[y : y + settings.tile_size, x : x + settings.tile_size]
            tiles0.append(first[sl])
            tiles1.append(second[sl])

        tensor0 = _image_batch(tiles0, device)
        tensor1 = _image_batch(tiles1, device)
        padder = _LocalPadder(tensor0.shape)
        tensor0, tensor1 = padder.pad_images(tensor0, tensor1)

        batch_index = batch_start // settings.batch_size
        set_seed(component_seed + batch_index)
        t0 = time.perf_counter()
        flow = predict(tensor0, tensor1)
        flow = padder.unpad(flow)
        if flow.ndim != 4 or flow.shape[1] != 2 or flow.shape[0] != len(batch_records):
            raise ValueError(f"Unexpected model flow shape: {tuple(flow.shape)}")
        flow_np = flow.permute(0, 2, 3, 1).float().cpu().numpy().astype(np.float32)

        for local_index, (y, x) in enumerate(batch_records):
            tile_flow = flow_np[local_index]
            sl = np.s_[y : y + settings.tile_size, x : x + settings.tile_size]
            tile_output = output[sl]
            if not np.isfinite(tile_flow).all():
                raise FloatingPointError(f"Non-finite flow at tile y={y}, x={x}")
            stats = flow_statistics(tile_flow, tile_output)
            limit = settings.max_tile_flow_px
            hard_limit = 10.0 * limit
            if (np.isfinite(stats["p99"]) and stats["p99"] > limit) or (
                np.isfinite(stats["max"]) and stats["max"] > hard_limit
            ):
                raise FloatingPointError(
                    f"Implausible tile flow y={y} x={x}: "
                    f"p99={stats['p99']:.3f}, max={stats['max']:.3f} px"
                )
            max_seen = max(max_seen, stats["max"] if np.isfinite(stats["max"]) else 0.0)
            weight = tile_weight(settings.tile_size, settings.overlap, y, x, h, w)
            weight *= tile_output.astype(np.float32)
            sum_u[sl] += tile_flow[..., 0] * weight
            sum_v[sl] += tile_flow[..., 1] * weight
            sum_w[sl] += weight

        print(
            f"  batch {batch_index + 1:3d}/{n_batches:3d}: "
            f"tiles={len(batch_records)} time={time.perf_counter() - t0:.2f}s"
        )
        del tensor0, tensor1, flow, flow_np

    full = np.full((h, w, 2), np.nan, dtype=np.float32)
    valid_out = output & (sum_w > 0.0)
    full[..., 0][valid_out] = sum_u[valid_out] / sum_w[valid_out]
    full[..., 1][valid_out] = sum_v[valid_out] / sum_w[valid_out]
    diagnostics = {
        "height": h,
        "width": w,
        "tile_size": settings.tile_size,
        "overlap": settings.overlap,
        "batch_size": settings.batch_size,
        "tile_y_positions": ys,
        "tile_x_positions": xs,
        "useful_tile_count": len(records),
        "skipped_tile_count": skipped,
        "valid_output_count": int(valid_out.sum()),
        "max_tile_flow_px": float(max_seen),
        "merged_flow_stats_px": flow_statistics(full, valid_out),
        "runtime_seconds": float(time.perf_counter() - started),
        "persistent_tile_cache": False,
    }
    return full, diagnostics
