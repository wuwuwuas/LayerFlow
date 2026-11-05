"""Load upstream WindFlow RAFT without copying its implementation."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import types


def load_raft_class():
    root = Path(os.environ.get(
        "AMV_WINDFLOW_REPO",
        str(Path(__file__).resolve().parents[3] / "external" / "windflow"),
    )).expanduser().resolve()
    raft_dir = root / "windflow" / "networks" / "raft"
    if not (raft_dir / "raft.py").is_file():
        raise FileNotFoundError(
            f"Upstream WindFlow RAFT was not found under {root}. "
            "Clone https://github.com/tjvandal/windflow into external/windflow "
            "or set AMV_WINDFLOW_REPO to the upstream repository."
        )
    # Avoid importing upstream's package-wide training/data dependencies.
    name = "_amv_external_windflow_raft"
    if name in sys.modules:
        loaded = Path(next(iter(sys.modules[name].__path__))).resolve()
        if loaded != raft_dir:
            raise RuntimeError("Restart Python before selecting a different WindFlow checkout")
    else:
        package = types.ModuleType(name)
        package.__path__ = [str(raft_dir)]
        package.__package__ = name
        sys.modules[name] = package
    return importlib.import_module(f"{name}.raft").RAFT
