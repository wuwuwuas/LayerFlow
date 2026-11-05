"""GOES adapters for split-region ERA5 validation.

All interpolation and metrics are delegated to the existing validation stack:

* ``validation.compare_era5.collocate_era5_to_points``
* ``validation.compare_era5.compare_to_era5``
* ``validation.metrics.summary_metrics``

This module only discovers/routs the two downloaded ERA5 regions and adds GOES
subsets.  It supports either one shared centre time (PCF) or one observation
time per point (official DMW).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from validation.compare_era5 import (
    add_era5_comparison_diagnostics,
    build_era5_comparison_masks,
    collocate_era5_to_points,
    compare_to_era5,
    load_pressure_amv_npz,
)


@dataclass(frozen=True)
class GoesEra5Region:
    name: str
    path: Path


def discover_goes_era5_regions(
    case_dir: str | Path,
) -> tuple[GoesEra5Region, ...]:
    root = Path(case_dir).resolve() / "era5"
    if not root.is_dir():
        raise FileNotFoundError(f"GOES ERA5 directory was not found: {root}")
    output: list[GoesEra5Region] = []
    for region_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        files = sorted(
            path
            for path in region_dir.glob("era5_*_pressure_levels.nc")
            if path.is_file()
        )
        if len(files) != 1:
            raise RuntimeError(
                f"Expected one completed ERA5 file in {region_dir}; found {len(files)}"
            )
        output.append(GoesEra5Region(region_dir.name, files[0]))
    if not output:
        raise FileNotFoundError(f"No completed ERA5 files found below {root}")
    return tuple(output)


def _coord_name(dataset: xr.Dataset, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in dataset.coords or name in dataset.variables:
            return name
    raise KeyError(f"None of these ERA5 coordinates were found: {tuple(candidates)}")


def _region_membership(points: pd.DataFrame, dataset: xr.Dataset) -> np.ndarray:
    lon_name = _coord_name(dataset, ("longitude", "lon"))
    lat_name = _coord_name(dataset, ("latitude", "lat"))
    ds_lon = np.asarray(dataset[lon_name].values, dtype=np.float64)
    ds_lat = np.asarray(dataset[lat_name].values, dtype=np.float64)
    lon = points["longitude"].to_numpy(dtype=np.float64)
    lat = points["latitude"].to_numpy(dtype=np.float64)
    if np.nanmin(ds_lon) >= 0.0:
        query_lon = np.mod(lon, 360.0)
    else:
        query_lon = ((lon + 180.0) % 360.0) - 180.0
        query_lon[np.isclose(lon, 180.0)] = 180.0
    return (
        np.isfinite(query_lon)
        & np.isfinite(lat)
        & (query_lon >= float(np.nanmin(ds_lon)))
        & (query_lon <= float(np.nanmax(ds_lon)))
        & (lat >= float(np.nanmin(ds_lat)))
        & (lat <= float(np.nanmax(ds_lat)))
    )


def collocate_goes_era5_to_points(
    points: pd.DataFrame,
    regions: Sequence[GoesEra5Region],
    *,
    valid_time: Any | None = None,
    time_column: str | None = None,
    chunk_size: int = 50_000,
    progress: bool = True,
) -> pd.DataFrame:
    """Route GOES points to split ERA5 files and reuse standard collocation."""

    required = {"longitude", "latitude", "pressure_hpa"}
    missing = sorted(required - set(points.columns))
    if missing:
        raise KeyError(f"GOES point table is missing columns: {missing}")
    if (valid_time is None) == (time_column is None):
        raise ValueError("Supply exactly one of valid_time or time_column")

    output = pd.DataFrame(index=points.index)
    output["u_era5"] = np.nan
    output["v_era5"] = np.nan
    output["speed_era5"] = np.nan
    output["era5_valid_mask"] = False
    output["era5_region"] = ""
    remaining = np.ones(len(points), dtype=bool)

    for region in regions:
        with xr.open_dataset(region.path) as dataset:
            member = remaining & _region_membership(points, dataset)
            index = points.index[member]
            if len(index) == 0:
                continue
            if progress:
                print(f"  ERA5 region {region.name}: {len(index):,} GOES points")
            columns = ["longitude", "latitude", "pressure_hpa"]
            if time_column is not None:
                columns.append(time_column)
            collocated = collocate_era5_to_points(
                points.loc[index, columns],
                dataset,
                valid_time=valid_time,
                time_column=time_column,
                chunk_size=chunk_size,
                progress=progress,
            )
            output.loc[index, "u_era5"] = collocated["u_era5"].to_numpy()
            output.loc[index, "v_era5"] = collocated["v_era5"].to_numpy()
            output.loc[index, "speed_era5"] = collocated[
                "speed_era5"
            ].to_numpy()
            valid = collocated["era5_valid_mask"].to_numpy(dtype=bool)
            output.loc[index, "era5_valid_mask"] = valid
            output.loc[index[valid], "era5_region"] = region.name
            position = np.flatnonzero(member)
            remaining[position[valid]] = False

    output["u_era5"] = output["u_era5"].astype(np.float32)
    output["v_era5"] = output["v_era5"].astype(np.float32)
    output["speed_era5"] = output["speed_era5"].astype(np.float32)
    output["era5_valid_mask"] = output["era5_valid_mask"].astype(bool)
    return output


def build_goes_amv_vs_era5_table(
    amv_product_path: str | Path,
    regions: Sequence[GoesEra5Region],
    *,
    valid_time: Any,
    chunk_size: int = 50_000,
    progress: bool = True,
) -> pd.DataFrame:
    amv = load_pressure_amv_npz(amv_product_path)
    collocated = collocate_goes_era5_to_points(
        amv,
        regions,
        valid_time=valid_time,
        chunk_size=chunk_size,
        progress=progress,
    )
    table = amv.copy()
    for column in collocated.columns:
        table[column] = collocated[column]
    table["amv_source"] = "goes_pressure_conditioned_flow"
    return add_era5_comparison_diagnostics(table).reset_index(drop=True)


def build_goes_official_dmw_vs_era5_table(
    dmw: pd.DataFrame,
    regions: Sequence[GoesEra5Region],
    *,
    time_column: str = "observation_time",
    chunk_size: int = 50_000,
    progress: bool = True,
) -> pd.DataFrame:
    """Collocate a standardized official GOES DMW table with point times."""

    table = dmw.copy()
    rename = {}
    if "u_ms" in table and "u_amv" not in table:
        rename["u_ms"] = "u_amv"
    if "v_ms" in table and "v_amv" not in table:
        rename["v_ms"] = "v_amv"
    if "speed_ms" in table and "speed_amv" not in table:
        rename["speed_ms"] = "speed_amv"
    table = table.rename(columns=rename)
    required = {
        "longitude",
        "latitude",
        "pressure_hpa",
        "u_amv",
        "v_amv",
        time_column,
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise KeyError(f"Official GOES DMW table is missing columns: {missing}")
    if "speed_amv" not in table:
        table["speed_amv"] = np.hypot(table["u_amv"], table["v_amv"])

    collocated = collocate_goes_era5_to_points(
        table,
        regions,
        time_column=time_column,
        chunk_size=chunk_size,
        progress=progress,
    )
    for column in collocated.columns:
        table[column] = collocated[column]
    table["amv_source"] = "official_goes_c14_dmw"
    return add_era5_comparison_diagnostics(table).reset_index(drop=True)


def build_goes_era5_comparison_masks(
    table: pd.DataFrame,
) -> dict[str, pd.Series]:
    default = build_era5_comparison_masks(table)
    base = default["all_valid"]
    masks: dict[str, pd.Series] = {"all_valid": base}
    if "pressure_hpa" in table:
        pressure = table["pressure_hpa"]
        masks["pressure_high_lt400"] = base & pressure.lt(400.0)
        masks["pressure_mid_400_700"] = (
            base & pressure.ge(400.0) & pressure.lt(700.0)
        )
        masks["pressure_low_ge700"] = base & pressure.ge(700.0)
    if "local_zenith_angle_deg" in table:
        zenith = table["local_zenith_angle_deg"]
        masks["local_zenith_lt40"] = base & zenith.lt(40.0)
        masks["local_zenith_40_60"] = (
            base & zenith.ge(40.0) & zenith.lt(60.0)
        )
        masks["local_zenith_ge60"] = base & zenith.ge(60.0)
    return masks


def compare_goes_amv_to_era5(
    table: pd.DataFrame,
    *,
    comparison: str,
) -> pd.DataFrame:
    return compare_to_era5(
        table,
        comparison=comparison,
        subset_masks=build_goes_era5_comparison_masks(table),
    )
