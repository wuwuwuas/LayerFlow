"""Shared data contracts for tracking, postprocessing, and downloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _array(value: Any, dtype: Any = None) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


@dataclass
class FlowResult:
    """Tracker output in original-image pixel coordinates and pixel units."""

    dx_pixels: np.ndarray
    dy_pixels: np.ndarray
    row_center_original: np.ndarray
    col_center_original: np.ndarray
    valid_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dx_pixels = _array(self.dx_pixels, np.float32)
        self.dy_pixels = _array(self.dy_pixels, np.float32)
        self.row_center_original = _array(self.row_center_original, np.float32)
        self.col_center_original = _array(self.col_center_original, np.float32)
        self.valid_mask = _array(self.valid_mask, bool)
        shapes = {
            self.dx_pixels.shape,
            self.dy_pixels.shape,
            self.row_center_original.shape,
            self.col_center_original.shape,
            self.valid_mask.shape,
        }
        if len(shapes) != 1:
            raise ValueError("All FlowResult arrays must have the same shape")


@dataclass
class WindResult:
    """Geolocated wind vectors produced after tracker-independent processing."""

    u_ms: np.ndarray
    v_ms: np.ndarray
    speed_ms: np.ndarray
    direction_deg: np.ndarray
    row_center_original: np.ndarray
    col_center_original: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    valid_mask: np.ndarray
    pressure_hpa: np.ndarray | None = None
    height_m: np.ndarray | None = None
    valid_component_count: np.ndarray | None = None
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = (
            "u_ms",
            "v_ms",
            "speed_ms",
            "direction_deg",
            "row_center_original",
            "col_center_original",
            "longitude",
            "latitude",
        )
        for name in names:
            setattr(self, name, _array(getattr(self, name), np.float32))
        self.valid_mask = _array(self.valid_mask, bool)
        shape = self.u_ms.shape
        if any(getattr(self, name).shape != shape for name in names):
            raise ValueError("All required WindResult arrays must have the same shape")
        if self.valid_mask.shape != shape:
            raise ValueError("valid_mask must match wind arrays")
        for name in ("pressure_hpa", "height_m", "valid_component_count"):
            value = getattr(self, name)
            if value is not None:
                value = _array(value)
                if value.shape != shape:
                    raise ValueError(f"{name} must have shape {shape}")
                setattr(self, name, value)
        self.diagnostics = {
            name: _array(value) for name, value in self.diagnostics.items()
        }
        for name, value in self.diagnostics.items():
            if value.shape != shape:
                raise ValueError(f"Diagnostic {name!r} must have shape {shape}")

    @property
    def p_oca_hpa(self) -> np.ndarray | None:
        """Backward-readable alias for OCA pressure."""

        return self.pressure_hpa

    @property
    def h_oca_m(self) -> np.ndarray | None:
        """Backward-readable alias for OCA height."""

        return self.height_m

    def to_frame(self, valid_only: bool = True) -> pd.DataFrame:
        """Convert vectors to the standard validation table convention."""

        data: dict[str, np.ndarray] = {
            "row_center_original": self.row_center_original.ravel(),
            "col_center_original": self.col_center_original.ravel(),
            "lon": self.longitude.ravel(),
            "lat": self.latitude.ravel(),
            "u_final": self.u_ms.ravel(),
            "v_final": self.v_ms.ravel(),
            "speed_final": self.speed_ms.ravel(),
            "direction_final": self.direction_deg.ravel(),
            "valid_mask": self.valid_mask.ravel(),
        }
        if self.pressure_hpa is not None:
            data["p_oca"] = self.pressure_hpa.ravel()
        if self.height_m is not None:
            data["h_oca"] = self.height_m.ravel()
        if self.valid_component_count is not None:
            data["valid_component_count"] = self.valid_component_count.ravel()
        for name, value in self.diagnostics.items():
            if name not in data:
                data[name] = value.ravel()
        frame = pd.DataFrame(data)
        if valid_only:
            required = ["lon", "lat", "u_final", "v_final"]
            mask = frame["valid_mask"] & frame[required].notna().all(axis=1)
            frame = frame.loc[mask]
        return frame.reset_index(drop=True)


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of one source-specific download or local availability check."""

    product: str
    status: str
    files: tuple[Path, ...] = ()
    message: str = ""
