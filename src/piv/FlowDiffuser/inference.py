from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch

from .model import build_standard_flowdiffuser
from .tiled import TiledSettings, infer_tiles_batched, normalize_scene_pair_uint8


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_REPOSITORY = Path(os.environ.get("AMV_FLOWDIFFUSER_REPO", str(PACKAGE_DIR.parents[2] / "external" / "FlowDiffuser")))
DEFAULT_CHECKPOINT = Path(os.environ.get("AMV_FLOWDIFFUSER_CHECKPOINT", str(PACKAGE_DIR.parents[2] / "weights" / "FlowDiffuser-things.pth")))


def _dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    base = np.asarray(mask, dtype=bool)
    if pixels <= 0:
        return base.copy()
    size = 2 * int(pixels) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(base.astype(np.uint8), kernel, iterations=1).astype(bool)


def _layerwise_normalization_support(
    *,
    weight0: np.ndarray,
    weight1: np.ndarray,
    valid0: np.ndarray,
    valid1: np.ndarray,
    core: np.ndarray,
    dilation_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    source = _dilate(
        (np.asarray(weight0) > 0.0) & np.asarray(valid0, dtype=bool),
        dilation_pixels,
    )
    target = _dilate(
        (np.asarray(weight1) > 0.0) & np.asarray(valid1, dtype=bool),
        dilation_pixels,
    )
    source |= np.asarray(core, dtype=bool)
    return source, target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FlowDiffuserTracker:
    """Exact single-seed FlowDiffuser inference used by the retained AMV baseline."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        repository_path: str | Path | None = None,
        tile_size: int = 512,
        tile_overlap: int = 128,
        batch_size: int = 4,
        seed: int = 0,
        device: str = "cuda:0",
        mixed_precision: bool = True,
        strict: bool = True,
        min_tile_output_fraction: float = 0.0005,
        max_tile_flow_px: float = 100.0,
        layer_normalization_dilation_pixels: int = 16,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT).expanduser().resolve()
        self.repository_path = Path(repository_path or DEFAULT_REPOSITORY).expanduser().resolve()
        self.device = self._resolve_device(device)
        self.mixed_precision = bool(mixed_precision)
        self.strict = bool(strict)
        self.seed = int(seed)
        self.layer_normalization_dilation_pixels = int(layer_normalization_dilation_pixels)
        self.tiled = TiledSettings(
            tile_size=int(tile_size),
            overlap=int(tile_overlap),
            batch_size=max(1, int(batch_size)),
            seed=int(seed),
            min_tile_output_fraction=float(min_tile_output_fraction),
            max_tile_flow_px=float(max_tile_flow_px),
        )
        if self.tiled.tile_size <= 0 or self.tiled.tile_size % 8 != 0:
            raise ValueError("FlowDiffuser tile_size must be a positive multiple of 8")
        if not 0 <= self.tiled.overlap < self.tiled.tile_size:
            raise ValueError("FlowDiffuser overlap must satisfy 0 <= overlap < tile_size")
        self.checkpoint_sha256 = _sha256(self.checkpoint_path)
        self.model, self.checkpoint_metadata = build_standard_flowdiffuser(
            repo=self.repository_path,
            checkpoint=self.checkpoint_path,
            device=self.device,
            mixed_precision=self.mixed_precision,
            strict=self.strict,
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        requested = str(device)
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print("[FlowDiffuser] CUDA requested but unavailable; using CPU.")
            return torch.device("cpu")
        return torch.device(requested)

    @torch.inference_mode()
    def _predict(self, image0: torch.Tensor, image1: torch.Tensor) -> torch.Tensor:
        output = self.model(image0, image1, test_mode=True)
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise TypeError("FlowDiffuser must return (_, flow_up) in test mode")
        flow = output[1]
        if flow.ndim != 4 or flow.shape[1] != 2:
            raise ValueError(f"Unexpected FlowDiffuser output shape: {tuple(flow.shape)}")
        return flow

    def predict_pair(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        *,
        valid_mask: np.ndarray | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        first = np.asarray(image0, dtype=np.float32)
        second = np.asarray(image1, dtype=np.float32)
        if first.shape != second.shape or first.ndim != 2:
            raise ValueError("FlowDiffuser inputs must be equally shaped 2-D images")
        external_valid = (
            np.ones(first.shape, dtype=bool)
            if valid_mask is None
            else np.asarray(valid_mask, dtype=bool)
        )
        if external_valid.shape != first.shape:
            raise ValueError(
                f"FlowDiffuser valid_mask shape {external_valid.shape} does not match {first.shape}"
            )
        output_mask = external_valid & np.isfinite(first) & np.isfinite(second)
        ctx = dict(context or {})
        image_mode = str(ctx.get("image_mode", "original")).strip().lower()

        if image_mode == "layerwise":
            required = (
                "layer_weight0",
                "layer_weight1",
                "source_valid_mask",
                "target_valid_mask",
            )
            missing = [name for name in required if ctx.get(name) is None]
            if missing:
                raise ValueError(
                    "FlowDiffuser layerwise inference is missing context fields: "
                    + ", ".join(missing)
                )
            support0, support1 = _layerwise_normalization_support(
                weight0=np.asarray(ctx["layer_weight0"], dtype=np.float32),
                weight1=np.asarray(ctx["layer_weight1"], dtype=np.float32),
                valid0=np.asarray(ctx["source_valid_mask"], dtype=bool),
                valid1=np.asarray(ctx["target_valid_mask"], dtype=bool),
                core=output_mask,
                dilation_pixels=int(
                    ctx.get(
                        "layer_normalization_dilation_pixels",
                        self.layer_normalization_dilation_pixels,
                    )
                ),
            )
            image0_u8, image1_u8, normalization = normalize_scene_pair_uint8(
                first,
                second,
                support0,
                support1,
            )
        else:
            support0 = output_mask
            support1 = output_mask
            image0_u8, image1_u8, normalization = normalize_scene_pair_uint8(
                first,
                second,
                output_mask,
                output_mask,
            )

        component_seed = int(ctx.get("component_seed", self.seed))
        if image_mode == "layerwise":
            component_seed += int(ctx.get("layer_index", 0)) * 1000
        flow, tiled_diagnostics = infer_tiles_batched(
            predict=self._predict,
            image0_u8=image0_u8,
            image1_u8=image1_u8,
            output_mask=output_mask,
            device=self.device,
            settings=self.tiled,
            component_seed=component_seed,
        )
        valid = output_mask & np.isfinite(flow[..., 0]) & np.isfinite(flow[..., 1])
        flow[~valid] = np.nan
        metadata = {
            "model_name": "FlowDiffuser",
            "architecture": "flowdiffuser_standard",
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "repository": str(self.repository_path),
            "device": str(self.device),
            "mixed_precision": self.mixed_precision,
            "strict_checkpoint": self.strict,
            "seed": self.seed,
            "component_seed": component_seed,
            "image_mode": image_mode,
            "normalization": normalization,
            "normalization_support0_count": int(np.asarray(support0, dtype=bool).sum()),
            "normalization_support1_count": int(np.asarray(support1, dtype=bool).sum()),
            "output_valid_count": int(valid.sum()),
            "tiled": tiled_diagnostics,
            "checkpoint_metadata": self.checkpoint_metadata,
        }
        return flow.astype(np.float32, copy=False), metadata
