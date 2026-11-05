"""Configuration helpers and repository-local default paths."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
PIV_ROOT = REPO_ROOT / "src" / "piv"


def repo_path(value: str | Path) -> Path:
    """Return an absolute path rooted in this repository."""

    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file from an absolute or repository path."""

    config_path = repo_path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    config["_config_path"] = str(config_path.relative_to(REPO_ROOT))
    return config


def resolve_case_config(case: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Resolve a mapping, YAML path, baseline case, or named large-domain case."""

    if isinstance(case, dict):
        return deepcopy(case)
    candidate = repo_path(case)
    if candidate.is_file():
        document = load_config(candidate)
        if "cases" in document:
            raise ValueError("Multi-case YAML files require a case name")
        return document

    name = str(case)
    cases_path = CONFIG_DIR / "large_domain_cases.yaml"
    if cases_path.exists():
        document = load_config(cases_path)
        if name in document.get("cases", {}):
            resolved = deepcopy(document.get("shared", {}))
            resolved.update(deepcopy(document["cases"][name]))
            resolved["case_name"] = name
            resolved["_config_path"] = str(cases_path.relative_to(REPO_ROOT))
            return resolved

    baseline = load_config(CONFIG_DIR / "case_seviri_ofape.yaml")
    if baseline.get("case_name") == name:
        return baseline
    raise KeyError(f"Unknown case: {name}")


def ensure_output_dirs(config: dict[str, Any], output_dir: str | Path | None = None) -> Path:
    """Create repository-local output directories and return the case directory."""

    if output_dir is None:
        output_dir = config.get("paths", {}).get(
            "output_dir", Path("outputs") / config["case_name"]
        )
    destination = repo_path(output_dir)
    if REPO_ROOT not in destination.parents:
        raise ValueError("Output directory must be inside the repository")
    for path in (
        destination,
        destination / "figures",
        destination / "matched",
        destination / "metrics",
        REPO_ROOT / "reports",
        REPO_ROOT / "cache",
        REPO_ROOT / "tmp",
    ):
        path.mkdir(parents=True, exist_ok=True)
    return destination


def write_resolved_config(config: dict[str, Any], destination: str | Path) -> None:
    """Write the effective configuration without private loader metadata."""

    output = deepcopy(config)
    output.pop("_config_path", None)
    path = repo_path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(output, stream, sort_keys=False)
