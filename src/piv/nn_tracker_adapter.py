"""Project-owned adapters; external model implementations are not vendored."""
from __future__ import annotations
from typing import Any, Mapping
import torch

_PREDICTOR_CACHE: dict[tuple[Any, ...], Any] = {}

def _bool(value: Any) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"} if isinstance(value, str) else bool(value)

def normalize_model_name(name: str) -> str:
    key = str(name).upper().replace("_", "").replace("-", "")
    try:
        return {"FLOWDIFFUSER": "FlowDiffuser", "FD": "FlowDiffuser", "WINDFLOW": "WindFlow"}[key]
    except KeyError as exc:
        raise ValueError("Only FlowDiffuser and WindFlow are included in the paper adapters") from exc

def _cache_key(name: str, params: Mapping[str, Any]) -> tuple[Any, ...]:
    return (normalize_model_name(name), tuple(sorted((str(k), repr(v)) for k, v in params.items() if not str(k).startswith("_"))))

def clear_nn_predictor_cache() -> None:
    _PREDICTOR_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_cached_nn_predictor(
    model_name: str,
    settings: Mapping[str, Any] | None = None,
):
    params = dict(settings or {})
    key = _cache_key(model_name, params)
    if key in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[key]

    normalized = normalize_model_name(model_name)
    if normalized == "FlowDiffuser":
        from piv.FlowDiffuser import FlowDiffuserTracker

        predictor = FlowDiffuserTracker(
            checkpoint_path=params.get("checkpoint"),
            repository_path=params.get("repository"),
            tile_size=int(params.get("tile_size", 512)),
            tile_overlap=int(params.get("overlap", 128)),
            batch_size=int(params.get("batch_size", 4)),
            seed=int(params.get("seed", 0)),
            device=str(params.get("device", "cuda:0")),
            mixed_precision=_bool(
                params.get("mixed_precision", params.get("amp", True))
            ),
            strict=_bool(params.get("strict", True)),
            min_tile_output_fraction=float(
                params.get("min_tile_output_fraction", 0.0005)
            ),
            max_tile_flow_px=float(params.get("max_tile_flow_px", 100.0)),
            layer_normalization_dilation_pixels=int(
                params.get("layer_normalization_dilation_pixels", 16)
            ),
        )
    elif normalized == "WindFlow":
        from piv.WindFlow import WindFlowTracker

        predictor = WindFlowTracker(
            checkpoint_path=params.get("checkpoint"),
            tile_size=int(params.get("tile_size", 512)),
            tile_overlap=int(params.get("overlap", 384)),
            batch_size=int(params.get("batch_size", 8)),
            iterations=int(params.get("iterations", 12)),
            device=str(params.get("device", "cuda:0")),
            mixed_precision=_bool(
                params.get("mixed_precision", params.get("amp", True))
            ),
            polarity=str(params.get("polarity", "original")),
            equalization_mode=str(params.get("equalization_mode", "per_image")),
            equalization_bins=int(params.get("equalization_bins", 500)),
            min_tile_output_fraction=float(
                params.get("min_tile_output_fraction", 0.0005)
            ),
            layer_normalization_dilation_pixels=int(
                params.get("layer_normalization_dilation_pixels", 16)
            ),
        )
    else:
        raise ValueError(f"Unsupported external baseline: {normalized}")
    _PREDICTOR_CACHE[key] = predictor
    return predictor
