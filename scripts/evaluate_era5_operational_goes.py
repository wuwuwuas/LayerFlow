"""Evaluate official G16/G18 Band-14 DMW against split-region ERA5.

The two viewpoints are reported separately.  The script uses the common
``validation.compare_era5`` interpolation/metric implementation through the
GOES region-routing adapter.  No pooled G16+G18 row is produced.
"""

from __future__ import annotations

import os

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASE_NAME = os.environ.get("AMV_CASE", "goes_g16_g18_E20240804T120506_W20240804T120508")
SATELLITES = ("G16", "G18")
REQUIRE_DQF_ZERO = True
MIN_PRESSURE_HPA = 100.0
MAX_PRESSURE_HPA = 1000.0
CHUNK_SIZE = 50_000
SAVE_COLLOCATED_CSV_GZ = True

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_io.goes_dmw import (  # noqa: E402
    discover_goes_dmw_files,
    dmw_basic_valid_mask,
    load_goes_dmw,
)
from validation.compare_era5_goes import (  # noqa: E402
    build_goes_official_dmw_vs_era5_table,
    compare_goes_amv_to_era5,
    discover_goes_era5_regions,
)


def main() -> int:
    case_dir = PROJECT_ROOT / "data" / CASE_NAME
    output_dir = PROJECT_ROOT / "outputs" / CASE_NAME / "official_dmw_vs_era5"
    output_dir.mkdir(parents=True, exist_ok=True)

    dmw_files = discover_goes_dmw_files(case_dir, satellites=SATELLITES)
    regions = discover_goes_era5_regions(case_dir)
    metric_tables: list[pd.DataFrame] = []
    collocated_tables: list[pd.DataFrame] = []
    attrition_rows: list[dict[str, Any]] = []
    dqf_rows: list[dict[str, Any]] = []

    print("=" * 112)
    print("OFFICIAL GOES C14 DMW VS ERA5")
    print("=" * 112)
    print("case:", CASE_NAME)
    print("satellites:", SATELLITES)
    print("quality rule:", "DQF == 0" if REQUIRE_DQF_ZERO else "all DQF")
    print("ERA5 pressure range:", MIN_PRESSURE_HPA, "to", MAX_PRESSURE_HPA)

    for source in dmw_files:
        raw = load_goes_dmw(source.path, satellite=source.satellite)
        for dqf, count in raw["dqf"].value_counts(dropna=False).sort_index().items():
            dqf_rows.append(
                {
                    "satellite": source.satellite,
                    "dqf": int(dqf),
                    "count": int(count),
                }
            )
        basic = dmw_basic_valid_mask(raw)
        quality = (
            raw["dqf"].to_numpy() == 0
            if REQUIRE_DQF_ZERO
            else np.ones(len(raw), dtype=bool)
        )
        pressure = raw["pressure_hpa"].between(
            MIN_PRESSURE_HPA, MAX_PRESSURE_HPA
        ).to_numpy()
        selected = raw.loc[basic & quality & pressure].copy().reset_index(drop=True)
        table = build_goes_official_dmw_vs_era5_table(
            selected,
            regions,
            time_column="observation_time",
            chunk_size=CHUNK_SIZE,
            progress=True,
        )
        collocated = table.loc[table["era5_valid_mask"]].copy().reset_index(
            drop=True
        )
        collocated["satellite"] = source.satellite
        collocated_tables.append(collocated)

        metrics = compare_goes_amv_to_era5(
            table,
            comparison="official_goes_c14_dmw_vs_era5",
        )
        metrics.insert(0, "satellite", source.satellite)
        metrics.insert(0, "tracker", "official_goes_c14_dmw")
        metric_tables.append(metrics)
        attrition_rows.append(
            {
                "satellite": source.satellite,
                "raw_measures": int(len(raw)),
                "basic_valid": int(np.count_nonzero(basic)),
                "dqf_zero": int(np.count_nonzero(basic & quality)),
                "pressure_in_era5_range": int(
                    np.count_nonzero(basic & quality & pressure)
                ),
                "era5_collocated": int(len(collocated)),
                "era5_collocation_fraction_of_selected": (
                    len(collocated) / len(selected) if len(selected) else np.nan
                ),
            }
        )
        print(
            f"[{source.satellite}] raw={len(raw):,}; selected={len(selected):,}; "
            f"collocated={len(collocated):,}"
        )

    metrics_all = pd.concat(metric_tables, ignore_index=True)
    attrition = pd.DataFrame(attrition_rows)
    dqf_counts = pd.DataFrame(dqf_rows)
    collocated_all = pd.concat(collocated_tables, ignore_index=True)

    metrics_path = output_dir / "official_dmw_vs_era5_metrics.csv"
    attrition_path = output_dir / "official_dmw_vs_era5_attrition.csv"
    dqf_path = output_dir / "official_dmw_dqf_counts.csv"
    metrics_all.to_csv(metrics_path, index=False)
    attrition.to_csv(attrition_path, index=False)
    dqf_counts.to_csv(dqf_path, index=False)
    if SAVE_COLLOCATED_CSV_GZ:
        collocated_all.to_csv(
            output_dir / "official_dmw_vs_era5_collocated.csv.gz",
            index=False,
            compression="gzip",
        )

    columns = [
        "tracker",
        "satellite",
        "subset",
        "sample_count",
        "mean_vector_difference",
        "mean_absolute_speed_error",
        "vector_difference_rmse",
        "mean_absolute_direction_difference",
        "speed_correlation",
        "u_bias_amv_minus_era5",
        "v_bias_amv_minus_era5",
        "speed_bias_amv_minus_era5",
    ]
    available = [column for column in columns if column in metrics_all]
    print("\n" + "=" * 112)
    print("METRICS (G16 AND G18 REPORTED SEPARATELY)")
    print("=" * 112)
    print(metrics_all[available].to_string(index=False))
    print("\nAttrition:")
    print(attrition.to_string(index=False))
    print("\nmetrics:", metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
