"""Common command-line options for the main retrieval entry points."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, MutableMapping, Sequence


def configure_retrieval(settings: MutableMapping[str, Any], argv: Sequence[str], *, sensor: str) -> None:
    parser = argparse.ArgumentParser(description="Retrieve dense atmospheric winds from local satellite observations.")
    parser.add_argument("--case", default=settings["CASE_NAME"], help="Name used for this run's output folder")
    parser.add_argument("--data-dir", type=Path, help="Folder containing the observation products")
    parser.add_argument("--output-dir", type=Path, help="Output folder; defaults to outputs/<case>")
    parser.add_argument("--threads", type=int, help="Number of native solver threads")
    if sensor == "seviri":
        parser.add_argument("--start", help="First frame time in UTC (ISO 8601); independent of folder name")
        parser.add_argument("--interval-minutes", type=float, default=settings["FRAME_INTERVAL_MINUTES"])
        parser.add_argument("--aoi", nargs=4, type=float, metavar=("WEST", "EAST", "SOUTH", "NORTH"), help="Optional geographic subset in degrees")
    else:
        parser.add_argument("--satellites", nargs="+", choices=["G16", "G18"], default=settings["SATELLITES"])
    args = parser.parse_args(argv)
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive")
    case = args.case.strip()
    if not case or Path(case).name != case or any(c in case for c in '/\\') or case in {'.', '..'}:
        parser.error("--case must be a single folder name")
    root = settings["PROJECT_ROOT"]
    data = args.data_dir or root / "data" / (case.removeprefix("test_") if sensor == "seviri" else case)
    out = args.output_dir or root / "outputs" / case
    settings.update(CASE_NAME=case, DATA_CASE_DIR=data.expanduser().resolve(),
                    OUTPUT_DIR=out.expanduser().resolve())
    settings["TRACKER_ROOT_DIR"] = settings["OUTPUT_DIR"] / "trackers"
    if "PRESSURE_OUTPUT_DIR" in settings:
        settings["PRESSURE_OUTPUT_DIR"] = settings["OUTPUT_DIR"] / "pressure_assignment"
    if args.threads is not None:
        import cv2
        cv2.setNumThreads(args.threads)
        if "SOLVER_CONFIG" in settings:
            settings["SOLVER_CONFIG"] = replace(settings["SOLVER_CONFIG"], opencv_num_threads=args.threads)
    if sensor == "seviri":
        if args.interval_minutes <= 0:
            parser.error("--interval-minutes must be positive")
        settings["FRAME_INTERVAL_MINUTES"] = args.interval_minutes
        settings["AOI"] = args.aoi
        if args.aoi:
            west,east,south,north = args.aoi
            if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
                parser.error("--aoi must satisfy -180 <= WEST < EAST <= 180 and -90 <= SOUTH < NORTH <= 90")
        if args.start:
            try:
                start = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
            except ValueError:
                parser.error("--start must be an ISO 8601 date/time")
            if start.tzinfo:
                start = start.astimezone(timezone.utc).replace(tzinfo=None)
            settings["FRAME_TIMES"] = tuple(start + timedelta(minutes=args.interval_minutes*i) for i in range(4))
    else:
        settings["SATELLITES"] = tuple(args.satellites)


def save_case_inputs(settings: MutableMapping[str, Any], *, sensor: str) -> None:
    """Keep observation selection with generated outputs for subsequent QC."""
    record = {
        "case": settings["CASE_NAME"], "sensor": sensor,
        "data_dir": str(settings["DATA_CASE_DIR"]),
        "frame_times": [x.isoformat() for x in settings.get("FRAME_TIMES") or ()],
        "interval_minutes": settings.get("FRAME_INTERVAL_MINUTES"),
        "aoi": settings.get("AOI"),
    }
    if sensor == "goes":
        record["satellites"] = list(settings["SATELLITES"])
        record["geometry_modes"] = list(settings["GEOMETRY_MODES"])
    out = settings["OUTPUT_DIR"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_inputs.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def read_case_inputs(root: Path, case: str) -> dict[str, Any]:
    record = root / "outputs" / case / "run_inputs.json"
    return json.loads(record.read_text(encoding="utf-8")) if record.is_file() else {}
