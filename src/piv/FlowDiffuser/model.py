from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch


def _build_without_backbone_download(model_class, args):
    """A complete FlowDiffuser checkpoint supplies both backbone state dicts."""
    import timm

    create_model = timm.create_model

    def local_model(*model_args, **kwargs):
        kwargs["pretrained"] = False
        return create_model(*model_args, **kwargs)

    # Upstream enables ImageNet downloads during construction even though the
    # full checkpoint loaded immediately afterwards replaces those parameters.
    # Do not edit the external repository or require network access at runtime.
    timm.create_model = local_model
    try:
        return model_class(args)
    finally:
        timm.create_model = create_model


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint_payload(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"FlowDiffuser checkpoint not found: {source}")
    payload = _torch_load(source)
    metadata: dict[str, Any] = {}
    if isinstance(payload, Mapping):
        for key in ("epoch", "global_step", "validation", "best_epe", "base_checkpoint", "config"):
            if key in payload:
                metadata[key] = payload[key]
        state: Any = payload
        for key in ("model_state_dict", "state_dict", "model"):
            if key in payload and isinstance(payload[key], Mapping):
                state = payload[key]
                break
    else:
        state = payload
    if not isinstance(state, Mapping):
        raise TypeError(f"Unsupported FlowDiffuser checkpoint payload: {type(payload)}")
    clean = {
        (str(key)[7:] if str(key).startswith("module.") else str(key)): value
        for key, value in state.items()
        if torch.is_tensor(value)
    }
    if not clean:
        raise ValueError(f"No tensor state_dict found in checkpoint: {source}")
    return clean, metadata


def build_standard_flowdiffuser(
    *,
    repo: str | Path,
    checkpoint: str | Path,
    device: torch.device,
    mixed_precision: bool,
    strict: bool = True,
):
    repo_path = Path(repo).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    core = repo_path / "core"
    if not core.is_dir():
        raise FileNotFoundError(f"FlowDiffuser core directory not found: {core}")
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
    importlib.invalidate_caches()
    from flowdiffuser import FlowDiffuser  # type: ignore

    args = SimpleNamespace(
        small=False,
        mixed_precision=bool(mixed_precision),
        alternate_corr=False,
    )
    model = _build_without_backbone_download(FlowDiffuser, args)
    state, metadata = load_checkpoint_payload(checkpoint_path)
    incompatible = model.load_state_dict(state, strict=bool(strict))
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        print("[FlowDiffuser] checkpoint warnings:")
        print("  missing:", list(incompatible.missing_keys)[:30])
        print("  unexpected:", list(incompatible.unexpected_keys)[:30])
    model.to(device)
    model.eval()
    return model, metadata
