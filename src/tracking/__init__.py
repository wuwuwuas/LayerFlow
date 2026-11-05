"""Lazy baseline adapters keep optional dependencies out of LayerFlow imports."""
from importlib import import_module

__all__ = [f"run_{name}" for name in (
    "farneback", "deepflow", "dis", "tvl1", "dualtvl1", "ilk", "pcaflow",
    "flowdiffuser", "windflow",
)]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(import_module(f".{name.removeprefix('run_')}", __name__), name)
    globals()[name] = value
    return value
