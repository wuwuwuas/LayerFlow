"""WindFlow (Vandal et al., KDD 2022) inference adapter for the AMV project.

WindFlow is a single-channel RAFT trained on GEOS-5 Nature Run specific
humidity with Nature-Run winds as labels.  This adapter reproduces the public
WindFlow inference recipe (``windflow/inference/inference_flows.py``) and
exposes it through the same ``predict_pair`` interface as ``FlowDiffuserTracker``
so it can be used as a parallel learned optical-flow baseline:

* input preprocessing: histogram equalization with 500 bins to [0, 1]
  (WindFlow ``image_histogram_equalization``), applied per image as in the
  WindFlow GOES inference (``equalization_mode="per_image"``) or jointly over
  the pair as in WindFlow training (``"joint_pair"``);
* model: full-size RAFT, one input channel, ``iters=12``, ``test_mode=True``;
* tiling: 512 px tiles, 384 px overlap, 2-D Gaussian blending (nsig=3);
* output: pixel displacement (dx along columns, dy along rows).

Deliberate differences from the public GOES pipeline:

* equalization statistics are computed only on the caller's support pixels
  (off-disk / invalid pixels are not counted as zeros);
* no empirical 0.8 speed factor and no 2-km/10-min speed conversion: the AMV
  project converts pixel flow to wind with its own satellite geometry;
* ``polarity="inverted"`` optionally feeds ``-image`` so that cold/moist
  structures are bright, matching the Nature-Run humidity training images.
"""

from __future__ import annotations

import hashlib
import os
import math
import time
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .external import load_raft_class


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = Path(os.environ.get("AMV_WINDFLOW_CHECKPOINT", str(PACKAGE_DIR.parents[2] / "weights" / "windflow.raft.pth.tar")))

POLARITY_MODES = ("original", "inverted")
EQUALIZATION_MODES = ("per_image", "joint_pair")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    base = np.asarray(mask, dtype=bool)
    if pixels <= 0:
        return base.copy()
    size = 2 * int(pixels) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(base.astype(np.uint8), kernel, iterations=1).astype(bool)


def equalization_mapping(values: np.ndarray, number_bins: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Return the WindFlow histogram-equalization lookup (bin edges, CDF)."""

    data = np.asarray(values, dtype=np.float64).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("Histogram equalization has no finite support values")
    histogram, bins = np.histogram(data, int(number_bins), density=True)
    cdf = histogram.cumsum()
    if not np.isfinite(cdf[-1]) or cdf[-1] <= 0:
        cdf = np.linspace(0.0, 1.0, histogram.size)
    else:
        cdf = cdf / cdf[-1]
    return bins[:-1], cdf


def apply_equalization(image: np.ndarray, mapping: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    edges, cdf = mapping
    arr = np.asarray(image, dtype=np.float64)
    out = np.interp(arr.ravel(), edges, cdf).reshape(arr.shape)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def tile_starts_windflow(length: int, tile_size: int, overlap: int) -> list[int]:
    """WindFlow ``split_array`` start positions (stride, last tile clamped)."""

    stride = int(tile_size) - int(overlap)
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")
    starts: list[int] = []
    for start in range(0, int(length), stride):
        value = min(start, int(length) - int(tile_size))
        if not starts or value != starts[-1]:
            starts.append(value)
    return starts


def gaussian_tile_weight(size: int, nsig: float = 3.0) -> np.ndarray:
    """WindFlow ``reassemble_with_2d_gaussian`` kernel (normalised 2-D pdf)."""

    edges = np.linspace(-nsig, nsig, int(size) + 1)
    cdf = np.array([0.5 * (1.0 + math.erf(value / math.sqrt(2.0))) for value in edges])
    kern1d = np.diff(cdf)
    kern2d = np.outer(kern1d, kern1d)
    return (kern2d / kern2d.sum()).astype(np.float64)


class WindFlowTracker:
    """Tiled WindFlow-RAFT inference with the project ``predict_pair`` contract."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        tile_size: int = 512,
        tile_overlap: int = 384,
        batch_size: int = 8,
        iterations: int = 12,
        device: str = "cuda:0",
        mixed_precision: bool = True,
        polarity: str = "original",
        equalization_mode: str = "per_image",
        equalization_bins: int = 500,
        min_tile_output_fraction: float = 0.0005,
        layer_normalization_dilation_pixels: int = 16,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"WindFlow checkpoint not found: {self.checkpoint_path}. Download "
                "model_weights/windflow.raft.pth.tar from github.com/tjvandal/windflow."
            )
        self.tile_size = int(tile_size)
        self.tile_overlap = int(tile_overlap)
        if self.tile_size <= 0 or self.tile_size % 8 != 0:
            raise ValueError("WindFlow tile_size must be a positive multiple of 8")
        if not 0 <= self.tile_overlap < self.tile_size:
            raise ValueError("WindFlow overlap must satisfy 0 <= overlap < tile_size")
        self.batch_size = max(1, int(batch_size))
        self.iterations = int(iterations)
        self.device = self._resolve_device(device)
        self.mixed_precision = bool(mixed_precision) and self.device.type == "cuda"
        self.polarity = str(polarity).strip().lower()
        if self.polarity not in POLARITY_MODES:
            raise ValueError(f"polarity must be one of {POLARITY_MODES}, got {polarity!r}")
        self.equalization_mode = str(equalization_mode).strip().lower()
        if self.equalization_mode not in EQUALIZATION_MODES:
            raise ValueError(
                f"equalization_mode must be one of {EQUALIZATION_MODES}, got {equalization_mode!r}"
            )
        self.equalization_bins = int(equalization_bins)
        self.min_tile_output_fraction = float(min_tile_output_fraction)
        self.layer_normalization_dilation_pixels = int(layer_normalization_dilation_pixels)
        self.checkpoint_sha256 = _sha256(self.checkpoint_path)
        self.model, self.global_step = self._load_model()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        requested = str(device)
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print("[WindFlow] CUDA requested but unavailable; using CPU.")
            return torch.device("cpu")
        return torch.device(requested)

    def _load_model(self) -> tuple[torch.nn.Module, int | None]:
        # Same construction as windflow.networks.models.get_flow_model("raft", small=False).
        args = {
            "small": False,
            "lr": 1e-5,
            "mixed_precision": self.mixed_precision,
            "dropout": 0.0,
            "corr_levels": 4,
            "corr_radius": 4,
        }
        model = load_raft_class()(args)
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch < 1.13
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        if state and next(iter(state)).startswith("module."):
            state = {key[len("module."):]: value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        model.to(self.device)
        model.eval()
        step = checkpoint.get("global_step") if isinstance(checkpoint, dict) else None
        print(f"[WindFlow] loaded {self.checkpoint_path} (global_step={step}) on {self.device}")
        return model, (None if step is None else int(step))

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _equalize_pair(
        self,
        first: np.ndarray,
        second: np.ndarray,
        support0: np.ndarray,
        support1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if self.polarity == "inverted":
            first = -first
            second = -second
        values0 = first[support0 & np.isfinite(first)]
        values1 = second[support1 & np.isfinite(second)]
        if values0.size == 0 or values1.size == 0:
            raise ValueError("WindFlow pair has no equalization support pixels")
        if self.equalization_mode == "joint_pair":
            mapping = equalization_mapping(
                np.concatenate([values0, values1]), self.equalization_bins
            )
            mapping0 = mapping1 = mapping
        else:
            mapping0 = equalization_mapping(values0, self.equalization_bins)
            mapping1 = equalization_mapping(values1, self.equalization_bins)
        eq0 = apply_equalization(first, mapping0)
        eq1 = apply_equalization(second, mapping1)
        info = {
            "method": "windflow_histogram_equalization",
            "bins": self.equalization_bins,
            "mode": self.equalization_mode,
            "polarity": self.polarity,
            "support0_count": int(values0.size),
            "support1_count": int(values1.size),
        }
        return eq0, eq1, info

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _predict_batch(self, tiles0: list[np.ndarray], tiles1: list[np.ndarray]) -> np.ndarray:
        x0 = torch.from_numpy(np.stack(tiles0)[:, None]).float().to(self.device, non_blocking=True)
        x1 = torch.from_numpy(np.stack(tiles1)[:, None]).float().to(self.device, non_blocking=True)
        h, w = x0.shape[-2:]
        pad_h = (-h) % 8
        pad_w = (-w) % 8
        if pad_h or pad_w:
            x0 = F.pad(x0, (0, pad_w, 0, pad_h), mode="replicate")
            x1 = F.pad(x1, (0, pad_w, 0, pad_h), mode="replicate")
        flow = self.model(x0, x1, iters=self.iterations, test_mode=True)[0]
        flow = flow[..., :h, :w]
        return flow.float().cpu().numpy()

    def _infer_tiled(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        output_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        h, w = image0.shape
        tile = min(self.tile_size, h, w)
        tile -= tile % 8 if tile >= 8 else 0
        if tile < 64:
            raise ValueError(f"Image {image0.shape} is too small for WindFlow tiling")
        # Keep the configured overlap ratio (384/512 by default) if the tile
        # had to shrink for a small image.
        overlap = int(round(tile * self.tile_overlap / float(self.tile_size)))
        overlap = max(0, min(overlap, tile - 8))
        overlap -= overlap % 8
        ys = tile_starts_windflow(h, tile, overlap)
        xs = tile_starts_windflow(w, tile, overlap)
        records: list[tuple[int, int]] = []
        skipped = 0
        for y in ys:
            for x in xs:
                if float(np.mean(output_mask[y : y + tile, x : x + tile])) < self.min_tile_output_fraction:
                    skipped += 1
                    continue
                records.append((y, x))

        weight = gaussian_tile_weight(tile)
        out_sum = np.zeros((2, h, w), dtype=np.float64)
        counter = np.zeros((h, w), dtype=np.float64)
        started = time.perf_counter()
        n_batches = math.ceil(len(records) / self.batch_size) if records else 0
        print(
            f"  WindFlow tiles: useful={len(records)}, skipped={skipped}, "
            f"tile={tile}, overlap={overlap}, batch={self.batch_size}"
        )
        for batch_index, start in enumerate(range(0, len(records), self.batch_size)):
            batch = records[start : start + self.batch_size]
            t0 = time.perf_counter()
            pred = self._predict_batch(
                [image0[y : y + tile, x : x + tile] for y, x in batch],
                [image1[y : y + tile, x : x + tile] for y, x in batch],
            )
            if not np.isfinite(pred).all():
                raise FloatingPointError("WindFlow returned non-finite flow")
            for local, (y, x) in enumerate(batch):
                out_sum[:, y : y + tile, x : x + tile] += pred[local] * weight
                counter[y : y + tile, x : x + tile] += weight
            if (batch_index + 1) % 20 == 0 or batch_index + 1 == n_batches:
                print(
                    f"  batch {batch_index + 1:4d}/{n_batches:4d}: "
                    f"tiles={len(batch)} time={time.perf_counter() - t0:.2f}s"
                )

        flow = np.full((h, w, 2), np.nan, dtype=np.float32)
        covered = counter > 0
        flow[..., 0][covered] = (out_sum[0][covered] / counter[covered]).astype(np.float32)
        flow[..., 1][covered] = (out_sum[1][covered] / counter[covered]).astype(np.float32)
        diagnostics = {
            "tile_size": int(tile),
            "overlap": int(overlap),
            "batch_size": int(self.batch_size),
            "useful_tile_count": len(records),
            "skipped_tile_count": skipped,
            "blending": "windflow_2d_gaussian_nsig3",
            "runtime_seconds": float(time.perf_counter() - started),
        }
        return flow, diagnostics

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
            raise ValueError("WindFlow inputs must be equally shaped 2-D images")
        external_valid = (
            np.ones(first.shape, dtype=bool)
            if valid_mask is None
            else np.asarray(valid_mask, dtype=bool)
        )
        if external_valid.shape != first.shape:
            raise ValueError(
                f"WindFlow valid_mask shape {external_valid.shape} does not match {first.shape}"
            )
        output_mask = external_valid & np.isfinite(first) & np.isfinite(second)
        ctx = dict(context or {})
        image_mode = str(ctx.get("image_mode", "original")).strip().lower()

        if image_mode == "layerwise":
            required = ("layer_weight0", "layer_weight1", "source_valid_mask", "target_valid_mask")
            missing = [name for name in required if ctx.get(name) is None]
            if missing:
                raise ValueError(
                    "WindFlow layerwise inference is missing context fields: " + ", ".join(missing)
                )
            dilation = int(
                ctx.get("layer_normalization_dilation_pixels", self.layer_normalization_dilation_pixels)
            )
            support0 = _dilate(
                (np.asarray(ctx["layer_weight0"]) > 0.0) & np.asarray(ctx["source_valid_mask"], dtype=bool),
                dilation,
            ) | output_mask
            support1 = _dilate(
                (np.asarray(ctx["layer_weight1"]) > 0.0) & np.asarray(ctx["target_valid_mask"], dtype=bool),
                dilation,
            )
        else:
            support0 = output_mask
            support1 = output_mask

        eq0, eq1, equalization = self._equalize_pair(first, second, support0, support1)
        flow, tiled = self._infer_tiled(eq0, eq1, output_mask)
        valid = output_mask & np.isfinite(flow[..., 0]) & np.isfinite(flow[..., 1])
        flow[~valid] = np.nan
        metadata = {
            "model_name": "WindFlow",
            "architecture": "windflow_raft_full_1ch",
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_global_step": self.global_step,
            "device": str(self.device),
            "mixed_precision": self.mixed_precision,
            "iterations": self.iterations,
            "image_mode": image_mode,
            "equalization": equalization,
            "output_valid_count": int(valid.sum()),
            "speed_bias_factor_0p8_applied": False,
            "tiled": tiled,
        }
        return flow.astype(np.float32, copy=False), metadata
