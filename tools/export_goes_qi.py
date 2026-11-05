"""Export forecast-independent GOES C14 QI from an NNJA GDAS BUFR dump.

See README for the NVIDIA BUFR helper and decoding dependencies."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from pathlib import Path

FIELDS = {
    1007: "satellite_id", 5001: "latitude", 6001: "longitude",
    4001: "year", 4002: "month", 4003: "day", 4004: "hour",
    4005: "minute", 4006: "second", 7004: "pressure_pa",
    11001: "wind_direction_deg", 11002: "wind_speed_ms",
    5042: "channel", 7024: "satellite_zenith_deg",
}
COLUMNS = ["satellite", "observation_time_utc", "channel", "latitude",
           "longitude", "pressure_pa", "wind_speed_ms", "wind_direction_deg",
           "satellite_zenith_deg", "qi_without_forecast"] + [
               f"confidence_application_{i}" for i in (4, 5, 6, 7)]


def load_helper(path: Path) -> dict:
    source = path.read_text(encoding="utf-8-sig")
    dependency = "from earth2studio.utils.imports import (\n    OptionalDependencyFailure,\n)"
    if dependency not in source:
        raise RuntimeError("Unsupported upstream BUFR helper revision; see the verified SHA-256 in README")
    # The downloaded helper is left untouched. Only its optional-dependency
    # notifier is replaced in memory so the full Earth2Studio stack is unneeded.
    source = source.replace(dependency, "class OptionalDependencyFailure(Exception): pass", 1)
    scope = {"__name__": "_amv_external_bufr_helper"}
    exec(compile(source, str(path), "exec"), scope)
    return scope


def observation_rows(message, start: datetime, end: datetime):
    td = message.template_data.value
    for descriptors, values in zip(td.decoded_descriptors_all_subsets, td.decoded_values_all_subsets):
        fields, confidence = {}, {}
        application = None
        for descriptor, value in zip(descriptors, values):
            key = descriptor.id
            if key in FIELDS:
                fields.setdefault(FIELDS[key], value)
            if key == 1044:
                application = value
            elif key == 33007 and application is not None:
                confidence[int(application)] = value
                application = None
        if fields.get("satellite_id") not in (270, 272) or fields.get("channel") != 14:
            continue
        stamp = datetime(*[int(fields[k]) for k in ["year", "month", "day", "hour", "minute", "second"]])
        if not start <= stamp <= end:
            continue
        row = {k: fields.get(k) for k in COLUMNS}
        row.update(satellite="G16" if fields["satellite_id"] == 270 else "G18",
                   observation_time_utc=stamp.isoformat() + "Z",
                   qi_without_forecast=confidence.get(5))
        row.update({f"confidence_application_{i}": confidence.get(i) for i in (4, 5, 6, 7)})
        yield row


def utc(value: str) -> datetime:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return stamp.astimezone(timezone.utc).replace(tzinfo=None) if stamp.tzinfo else stamp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bufr", type=Path)
    parser.add_argument("--helper", type=Path, required=True, help="Downloaded upstream utils_bufr.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2024-08-04T11:55:00")
    parser.add_argument("--end", default="2024-08-04T12:15:00")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    start, end = utc(args.start), utc(args.end)
    if end < start:
        parser.error("--end precedes --start")
    if args.output.exists() and not args.overwrite:
        parser.error("Output exists; pass --overwrite to replace it")
    helpers = load_helper(args.helper)
    b, d, messages = helpers["parse_prepbufr_messages"](args.bufr.read_bytes(), False)
    decoder = helpers["create_decoder"](b, d)
    counts = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for raw, category in messages:
            if category != 5 or len(raw) < 18 or raw[17] != 30:
                continue
            for row in observation_rows(decoder.process(raw), start, end):
                writer.writerow(row)
                counts[row["satellite"]] += 1
    if not counts:
        raise RuntimeError(f"No matching GOES C14 records found; output has only a header: {args.output}")
    print(dict(counts), args.output)


if __name__ == "__main__":
    main()
