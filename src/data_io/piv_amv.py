"""PIV-AMV NPZ input/output helpers.

This module keeps notebooks simple:
- read the clean stacked PIV native NPZ produced by ``pipeline.tracker_runner``;
- save compact pressure-assigned PIV-AMV NPZ files.

The pressure-assigned files are compact 1-D products: points without assigned
pressure are not written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PIV_FINAL_REQUIRED_KEYS = (
    "row_center_original",
    "col_center_original",
    "longitude",
    "latitude",
    "u_ms",
    "v_ms",
    "valid_mask",
    "valid_component_count",
    "u_c1_ms",
    "v_c1_ms",
    "valid_c1_mask",
    "dx_c1_pixels",
    "dy_c1_pixels",
    "u_c2_ms",
    "v_c2_ms",
    "valid_c2_mask",
    "dx_c2_pixels",
    "dy_c2_pixels",
    "u_c3_ms",
    "v_c3_ms",
    "valid_c3_mask",
    "dx_c3_pixels",
    "dy_c3_pixels",
)


def _metadata_json(metadata: Mapping[str, Any] | None) -> np.ndarray:
    return np.array([json.dumps(dict(metadata or {}), default=str)])


def _decode_metadata_json(value: Any) -> dict[str, Any]:
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return {}
        text = str(arr.ravel()[0])
        return json.loads(text) if text else {}
    except Exception:
        return {}


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def load_piv_final_native_npz(path: str | Path, *, required: bool = True) -> dict[str, Any]:
    """Load the stacked final PIV native NPZ.

    Parameters
    ----------
    path:
        File produced by the PIV-only tracker runner, e.g.
        ``trackers/farneback/final/farneback_final_native.npz``.
    required:
        If True, verify the full expected stacked-PIV schema.
    """

    source = _as_path(path)
    if not source.exists():
        raise FileNotFoundError(f"PIV final native NPZ not found: {source}")

    with np.load(source, allow_pickle=False) as data:
        keys = set(data.files)
        if required:
            missing = [key for key in PIV_FINAL_REQUIRED_KEYS if key not in keys]
            if missing:
                raise KeyError(
                    f"PIV final NPZ is missing required keys: {missing}. "
                    f"Available keys: {list(data.files)}"
                )
        arrays = {key: np.asarray(data[key]).copy() for key in data.files if key != "metadata_json"}
        metadata = _decode_metadata_json(data["metadata_json"]) if "metadata_json" in data.files else {}

    arrays["metadata"] = metadata
    arrays["source_path"] = source
    return arrays


def base_valid_mask(piv: Mapping[str, Any]) -> np.ndarray:
    """Return the finite wind/geolocation mask from a loaded PIV product."""

    return (
        np.asarray(piv["valid_mask"], dtype=bool)
        & np.isfinite(piv["longitude"])
        & np.isfinite(piv["latitude"])
        & np.isfinite(piv["u_ms"])
        & np.isfinite(piv["v_ms"])
    )


def compact_piv_arrays(
    piv: Mapping[str, Any],
    mask: np.ndarray,
    *,
    include_component_winds: bool = False,
    include_component_displacements: bool = False,
) -> dict[str, np.ndarray]:
    """Flatten a dense PIV product to compact 1-D arrays using ``mask``."""

    selection = np.asarray(mask, dtype=bool)
    output: dict[str, np.ndarray] = {}

    base_keys = (
        "row_center_original",
        "col_center_original",
        "longitude",
        "latitude",
        "u_ms",
        "v_ms",
        "valid_component_count",
    )
    for key in base_keys:
        if key in piv:
            output[key] = np.asarray(piv[key])[selection]

    if include_component_winds:
        for cname in ("c1", "c2", "c3"):
            for key in (f"u_{cname}_ms", f"v_{cname}_ms", f"valid_{cname}_mask"):
                if key in piv:
                    output[key] = np.asarray(piv[key])[selection]

    if include_component_displacements:
        for cname in ("c1", "c2", "c3"):
            for key in (f"dx_{cname}_pixels", f"dy_{cname}_pixels"):
                if key in piv:
                    output[key] = np.asarray(piv[key])[selection]

    return output


def save_pressure_assigned_piv_amv_npz(
    arrays: Mapping[str, Any],
    output_path: str | Path,
    *,
    pressure_source: str,
    metadata: Mapping[str, Any] | None = None,
    compress: bool = True,
) -> Path:
    """Save a compact pressure-assigned PIV-AMV product.

    All vector arrays are expected to be 1-D and already filtered so every saved
    point has a valid pressure.
    """

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    output: dict[str, Any] = {}
    n_points = None
    for key, value in arrays.items():
        if value is None or key in {"metadata", "source_path"}:
            continue
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape != (1,) and arr.dtype.kind not in {"U", "S", "O"}:
            if n_points is None and arr.ndim == 1:
                n_points = arr.shape[0]
            elif arr.ndim == 1 and n_points is not None and arr.shape[0] != n_points:
                raise ValueError(f"Array {key!r} has length {arr.shape[0]}, expected {n_points}")
        output[key] = arr

    output["pressure_source"] = np.array([str(pressure_source)])
    merged_metadata = dict(metadata or {})
    merged_metadata.setdefault("pressure_source", str(pressure_source))
    if n_points is not None:
        merged_metadata.setdefault("n_points", int(n_points))
    output["metadata_json"] = _metadata_json(merged_metadata)

    if compress:
        np.savez_compressed(destination, **output)
    else:
        np.savez(destination, **output)
    return destination


def inspect_npz(path: str | Path) -> None:
    """Print keys, shapes and dtypes for quick notebook inspection."""

    source = _as_path(path)
    print("NPZ path:", source)
    print("Exists:", source.exists())
    if not source.exists():
        return
    with np.load(source, allow_pickle=False) as data:
        print("\nNPZ keys:")
        for key in data.files:
            arr = data[key]
            print(f"  {key:<32} shape={arr.shape}, dtype={arr.dtype}")
