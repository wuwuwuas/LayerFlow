"""Build the reader manifest from locally downloaded NOAA ABI files."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

PRODUCTS = {
    "image": "ABI-L2-CMIPF-C14",
    "height": "ABI-L2-ACHA2KMF",
    "pressure": "ABI-L2-ACHP2KMF",
    "cloud_mask": "ABI-L2-ACMF",
}


def filename_times(path: Path) -> tuple[datetime, datetime, datetime]:
    def decode(marker: str) -> datetime:
        m = re.search(rf"_{marker}(\d{{13}})(\d*)_", path.name)
        if m is None:
            raise ValueError(f"Cannot read NOAA {marker} timestamp: {path.name}")
        stamp = datetime.strptime(m[1], "%Y%j%H%M%S")
        return stamp + timedelta(seconds=float("0." + m[2]) if m[2] else 0)
    start, end = decode("s"), decode("e")
    if end <= start:
        raise ValueError(f"Invalid scan interval: {path.name}")
    return start, end, start + (end - start) / 2


def build_rows(root: Path, center: datetime, satellites: list[str]) -> list[dict]:
    rows = []
    for satellite in satellites:
        satdir = root / f"goes{int(satellite[1:])}"
        inventory = {
            kind: [(p, filename_times(p)) for p in sorted((satdir / folder).glob("*.nc"))]
            for kind, folder in PRODUCTS.items()
        }
        images = inventory["image"]
        if len(images) < 3:
            raise ValueError(f"Need previous/center/next C14 images for {satellite}")
        nearest = min(images, key=lambda item: abs((item[1][2] - center).total_seconds()))
        if abs((nearest[1][2] - center).total_seconds()) > 120:
            raise ValueError(f"No {satellite} scan midpoint within 120 s of {center}")
        selected = []
        for role, offset in [("previous", -600), ("center", 0), ("next", 600)]:
            wanted = nearest[1][2] + timedelta(seconds=offset)
            image = min(images, key=lambda item: abs((item[1][2] - wanted).total_seconds()))
            if abs((image[1][2] - wanted).total_seconds()) > 120:
                raise ValueError(f"Missing 10-minute {satellite} {role} image")
            selected.append(image[0])
            for kind in PRODUCTS:
                matches = [item for item in inventory[kind] if abs((item[1][0] - image[1][0]).total_seconds()) < 1]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {satellite} {role} {kind} file; found {len(matches)}")
                path, _ = matches[0]
                rows.append({
                    "satellite": satellite, "role": role, "logical_product": kind,
                    "scan_start": image[1][0].isoformat(),
                    "scan_end": image[1][1].isoformat(),
                    "scan_midpoint": image[1][2].isoformat(),
                    "local_path": path.relative_to(root).as_posix(),
                })
        if len(set(selected)) != 3:
            raise ValueError(f"Temporal roles reuse a file for {satellite}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--center", required=True, help="Target scan midpoint in UTC, e.g. 2024-08-04T12:05:06")
    parser.add_argument("--satellites", nargs="+", choices=["G16", "G18"], default=["G16", "G18"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.case_dir.expanduser().resolve()
    center = datetime.fromisoformat(args.center.replace("Z", "+00:00"))
    if center.tzinfo is not None:
        center = center.astimezone(timezone.utc).replace(tzinfo=None)
    rows = build_rows(root, center, args.satellites)
    out = root / "manifests" / "goes_selected_products.csv"
    if out.exists() and not args.overwrite:
        parser.error(f"Manifest exists: {out}; use --overwrite to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
