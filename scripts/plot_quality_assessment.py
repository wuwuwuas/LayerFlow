"""Quality assessment: QI maps, local winds, ERA5 and IGRA curves.

Run: python scripts/plot_quality_assessment.py figure
Inputs are produced by run_quality_control.py, match_operational_quality_goes.py and
evaluate_quality_igra.py after the ERA5 and IGRA evaluation stages.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_wind_fields as pwd  # noqa: E402  (sets PROJECT_ROOT / src path, matplotlib Agg)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
import matplotlib.colorbar as mcolorbar  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

PROJECT_ROOT = pwd.PROJECT_ROOT
from runtime_options import read_case_inputs

# =============================================================================
# Settings
# =============================================================================
CASE = "test_2018-01-12T11"
VALID_OFFSET_MINUTES = 45
GEOS = pwd.SEVIRI_GEOS
# GOES cases (names starting with "goes"): satellite (CLI --sat), geometry and tracker names
SAT: str | None = None
GOES_SAT_DEFAULT = "G16"
GOES_GEOMETRY = "cloud_height"
GOES_ALIASES = {}
GOES_DMW_PRESSURE_RANGE = (100.0, 1000.0)
_GOES_CENTER = None                                # centre frame of the current GOES case (navigation)


def is_goes() -> bool:
    return read_case_inputs(PROJECT_ROOT, CASE).get("sensor", "goes" if CASE.lower().startswith("goes") else "seviri") == "goes"


def case_data_dir() -> Path:
    default = CASE if is_goes() else CASE.removeprefix("test_")
    return Path(read_case_inputs(PROJECT_ROOT, CASE).get("data_dir", PROJECT_ROOT / "data" / default))


def resolve(tracker: str) -> str:
    return GOES_ALIASES.get(tracker, tracker) if is_goes() else tracker


def qc_name(tracker: str) -> str:
    return f"{resolve(tracker)}_{SAT.lower()}" if is_goes() else tracker

# (tracker, label, colour, dash, line width); "operational" = EUMETSAT AMV with its own QI
METHODS = (
    ("operational", "Operational", "#000000", "-", 1.0),
    ("raw_deepflow_opencv", "DeepFlow", "#2a78d6", "-", 1.0),
    ("layerwise_deepflow_opencv", "DeepFlow (P)", "#2a78d6", (0, (3.2, 1.6)), 1.0),
    ("nn_raw_flowdiffuser", "FlowDiffuser", "#1baf7a", "-", 1.0),
    ("nn_layerwise_flowdiffuser", "FlowDiffuser (P)", "#1baf7a", (0, (3.2, 1.6)), 1.0),
    ("pressure_conditioned_flow", "LayerFlow", "#e34948", "-", 1.5),
)
ABLATION_TRACKER = "pressure_conditioned_flow"
ABLATION = (  # (score in validation_era5_retention.csv, label, colour, dash)
    ("qi", "QI (tracking + height)", "#e34948", "-"),
    ("ee_post", "Tracking term only", "#e34948", (0, (3.2, 1.6))),
    ("ee_nn", "NN prior only", "#e34948", (0, (1.0, 1.2))),
    ("ee_spread", "Component spread only", "#7a7a7a", (0, (3.2, 1.6))),
    ("random", "No QC (random)", "#7a7a7a", "-"),
)
MIN_RETENTION_PCT = 10.0

# Coverage figure
QI_MIN = 60.0                                      # same threshold for every product
COVERAGE_METHODS = ("operational", "layerwise_deepflow_opencv", "nn_layerwise_flowdiffuser", "pressure_conditioned_flow")
BEFORE_QC_TRACKER = "pressure_conditioned_flow"    # grey contour = dense coverage before QC
# "presence": the same rule for every product (incl. "before QC"): a block of
#             COVERAGE_BLOCK_PX pixels is covered if it holds >= 1 QI >= QI_MIN vector
#             (smoothed presence >= PRESENCE_LEVEL).  Default; comparable across products.
# "fraction": dense = retained-pixel fraction per BLOCK_PX block >= CONTOUR_LEVEL,
#             operational = presence (first version; not comparable, fragmented).
# "majority": each product judged against its own vectors.  Before QC: a cell is
#             covered if >= MAJORITY_LEVEL of its earth pixels carry a dense vector.
#             After QC: >= MAJORITY_LEVEL of the cell's vectors pass QI >= QI_MIN
#             (dense: only inside the product's own before-QC coverage;
#             operational: cells with >= MIN_OPERATIONAL_VECTORS vectors).  Default.
COVERAGE_MODE = "majority"
COVERAGE_BLOCK_PX = 32                             # ~100 km at the sub-satellite point
MAJORITY_LEVEL = 0.50
MIN_OPERATIONAL_VECTORS = 2
QI_MIN_OPERATIONAL = 60.0                          # 80 gives the same strictness as our QI >= 60
PRESENCE_LEVEL = 0.50
DISK_CONTOUR_METHODS = COVERAGE_METHODS            # e.g. ("operational", "pressure_conditioned_flow")
# Left panel: "qi_map" = per-pixel QI of QI_MAP_TRACKER drawn as an image over the
# IR (no contours); "contours" = coverage contours (COVERAGE_MODE).
DISK_MODE = "qi_map"
QI_MAP_TRACKER = "pressure_conditioned_flow"
QI_MAP_CMAP = "viridis"
QI_MAP_BOUNDS = (0, 20, 40, 60, 80, 100)           # discrete classes; 60 / 80 = thresholds used in the text
BLOCK_PX = 16                                      # dense: retained-pixel fraction per block
CONTOUR_LEVEL = 0.30                               # dense contour: >= 30 % of the block's pixels kept
OPERATIONAL_BLOCK_PX = 48                          # operational: presence of >= 1 vector per block
OPERATIONAL_LEVEL = 0.50
SMOOTH_BLOCKS = 1.0                                # Gaussian smoothing of the block fields (blocks)
CONTOUR_PT = 0.8
BEFORE_QC_COLOR = "#9a9a9a"
IR_ALPHA = 0.55                                    # IR under contours / vectors (on white)
# Local region (lon_min, lon_max, lat_min, lat_max); None = automatic choice
# North-Atlantic jet streak (clear in both the operational and the dense fields).
# The panels show the square (in the satellite view) enclosing this box.
# None = automatic: the box (all corners within LOCAL_MAX_CENTRAL_ANGLE of the
# sub-satellite point, i.e. away from the limb) with the strongest QC'd winds.
# South-East Atlantic off Angola (stratocumulus deck; curved low-level flow).
# None = automatic (see choose_box); the top candidates are printed so a box can be
# fixed here afterwards.  (-4, 13, -20, -7) = Angola stratocumulus / multilayer case.
LOCAL_BOX: tuple[float, float, float, float] | None = None
LOCAL_BOXES: dict = {   # per case ("case" or "case:SAT"); overrides LOCAL_BOX
    "test_2018-01-12T11": (-12.0, 3.0, -30.0, -15.0),
    # SE-Pacific stratocumulus deck off Peru and the cloud band along the coast
    "goes_g16_g18_E20240804T120506_W20240804T120508:G16": (-92.0, -77.0, -25.0, -10.0),
}
LOCAL_SIZE_DEG = 12.0
LOCAL_MAX_CENTRAL_ANGLE = 40.0                     # degrees of arc from the sub-satellite point
LOCAL_SEARCH = dict(lon=(-45.0, 45.0), lat=(-45.0, 45.0), step=2.5, min_operational=30)
LOCAL_ARROWS_ACROSS = 32                           # dense arrows per panel width (sets the thinning stride)
# Arrow length grows sub-linearly with speed so that weak winds stay visible (colour gives the speed):
# length = REF x spacing x (speed / 75th-pct speed)^GAMMA, clipped to [MIN, MAX] x spacing.
LOCAL_ARROW_REF_SPACING = 2.3
LOCAL_ARROW_GAMMA = 0.5
LOCAL_ARROW_MIN_SPACING = 1.0
LOCAL_ARROW_MAX_SPACING = 3.4
LOCAL_SPEED_MAX: float | None = None               # None: 99th pct of LayerFlow QC'd speeds, rounded up to 10
LOCAL_ARROW_WIDTH_MM = 0.16
LOCAL_SHOW_REJECTED = True                         # rejected vectors (QI below threshold) in light grey
REJECTED_COLOR = "#a8a8a8"
LOCAL_TITLES = {"operational": "Operational", "layerwise_deepflow_opencv": "DeepFlow (P)",
                "nn_layerwise_flowdiffuser": "FlowDiffuser (P)", "pressure_conditioned_flow": "LayerFlow"}
LOCAL_TITLE_PT = 6.5
LEGEND_PT = 6.5
BOX_COLOR = "#000000"
BOX_PT = 0.9

# Layout (mm); FIG_WIDTH_MM = 90 (one case per half row) or 180
FIG_WIDTH_MM = 90.0
RETENTION_ABLATION = False                         # True: add the LayerFlow ablation panel (a)
RETENTION_FIG = (90.0, 68.0)                       # single panel; (180, 62) with the ablation panel

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "qc" / CASE
MM = 1.0 / 25.4
STYLE = {m[0]: m for m in METHODS}


# =============================================================================
# Paths / loading
# =============================================================================


def qc_dir(tracker: str) -> Path:
    return PROJECT_ROOT / "outputs" / CASE / "qc_selfsup" / qc_name(tracker)


def qc_npz(tracker: str) -> Path:
    return qc_dir(tracker) / f"{qc_name(tracker)}_selfsup_qc.npz"


def product_path(tracker: str) -> Path:
    base = PROJECT_ROOT / "outputs" / CASE / "pressure_assignment"
    if is_goes():
        return base / f"{qc_name(tracker)}_{GOES_GEOMETRY}_final_amv_goes_pressure.npz"
    return base / f"{tracker}_final_amv_oca_pressure.npz"


def load_qi(tracker: str) -> pd.DataFrame:
    with np.load(qc_npz(tracker), allow_pickle=False) as d:
        meta = json.loads(str(d["metadata_json"].ravel()[0]))
        if "tau_ms" not in meta:
            raise RuntimeError(f"{tracker}: QI is still the old percentile version -- rerun run_quality_control.py")
        return pd.DataFrame({"row": d["row_center_original"].astype(np.int64),
                             "col": d["col_center_original"].astype(np.int64),
                             "qi": d["quality_index"].astype(np.float64)})


def load_dense(tracker: str) -> pd.DataFrame:
    from validation.compare_era5 import load_pressure_amv_npz

    prod = load_pressure_amv_npz(product_path(tracker))
    prod["row"] = np.rint(prod["row_center_original"]).astype(np.int64)
    prod["col"] = np.rint(prod["col_center_original"]).astype(np.int64)
    if is_goes():
        lon, lat = goes_surface_lonlat(prod["row_center_original"].to_numpy(np.float64),
                                       prod["col_center_original"].to_numpy(np.float64))
        prod["longitude"], prod["latitude"] = lon, lat
    prod = prod.loc[np.isfinite(prod[["longitude", "latitude", "u_amv", "v_amv"]].to_numpy()).all(axis=1)]
    out = prod[["row", "col", "longitude", "latitude", "u_amv", "v_amv"]].merge(
        load_qi(tracker).drop_duplicates(["row", "col"]), on=["row", "col"], how="left")
    print(f"  {tracker}: {len(out):,} vectors, QI matched {np.isfinite(out['qi']).mean():.1%}, "
          f"QI >= {QI_MIN:.0f}: {(out['qi'] >= QI_MIN).mean():.1%}")
    return out


def goes_surface_lonlat(rows: np.ndarray, cols: np.ndarray):
    from preprocessing.goes_geolocation import (
        ecef_to_geodetic_lonlat_on_height_ellipsoid, fixed_grid_to_ecef, scan_angles_at_pixels)

    c = _GOES_CENTER
    sx, sy = scan_angles_at_pixels(c.x_scan_angle, c.y_scan_angle, rows, cols)
    ex, ey, ez, valid = fixed_grid_to_ecef(sx, sy, 0.0, c.projection_attrs)
    lon, lat = ecef_to_geodetic_lonlat_on_height_ellipsoid(ex, ey, ez, 0.0, c.projection_attrs)
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    lon[~np.asarray(valid, bool)] = np.nan
    return lon, lat


def load_goes_image():
    global GEOS, _GOES_CENTER
    from data_io.goes import discover_goes_frames, frame_for, load_goes_frame

    center = load_goes_frame(frame_for(discover_goes_frames(case_data_dir()), SAT, "center"),
                             apply_quality_mask=False)
    _GOES_CENTER = center
    a = center.projection_attrs
    GEOS = pwd.Geos(float(a.get("longitude_of_projection_origin", pwd.GOES_GEOS[SAT].sub_lon)),
                    float(a.get("perspective_point_height", 35786023.0)), float(a.get("semi_major_axis", 6378137.0)),
                    float(a.get("semi_minor_axis", 6356752.31414)), str(a.get("sweep_angle_axis", "x")))
    xa = np.asarray(center.x_scan_angle, dtype=np.float64) * GEOS.height
    ya = np.asarray(center.y_scan_angle, dtype=np.float64) * GEOS.height
    dx, dy = xa[1] - xa[0], ya[1] - ya[0]
    image = pwd.Image(np.asarray(center.brightness_temperature_k, dtype=np.float32),
                      (xa[0] - dx / 2, xa[-1] + dx / 2, ya[-1] + dy / 2, ya[0] - dy / 2))
    return image, pd.Timestamp(center.scan_midpoint)


def load_goes_operational() -> pd.DataFrame:
    """Operational GOES winds with QI: decoded BUFR (QI without forecast) when available,
    otherwise the netCDF DMW with DQF = 0 -> qi = 100, DQF != 0 -> qi = 0."""

    from data_io.goes_dmw import discover_goes_dmw_files, dmw_basic_valid_mask, load_goes_dmw

    files = sorted((case_data_dir() / "official_amv_qi").glob("*_qi.csv"))
    if files:
        b = pd.read_csv(files[0])
        b = b.loc[b["satellite"].astype(str).str.upper() == SAT].copy()
        rad = np.radians(b["wind_direction_deg"].to_numpy(float))
        spd = b["wind_speed_ms"].to_numpy(float)
        d = pd.DataFrame({"longitude": b["longitude"].to_numpy(float), "latitude": b["latitude"].to_numpy(float),
                          "u_amv": -spd * np.sin(rad), "v_amv": -spd * np.cos(rad),
                          "pressure_hpa": b["pressure_pa"].to_numpy(float) / 100.0,
                          "qi": b["qi_without_forecast"].to_numpy(float)})
        d = d.loc[d["pressure_hpa"].between(*GOES_DMW_PRESSURE_RANGE)].reset_index(drop=True)
        print(f"  operational BUFR ({files[0].name}): {len(d):,} vectors, QI >= {QI_MIN_OPERATIONAL:.0f}: "
              f"{(d['qi'] >= QI_MIN_OPERATIONAL).mean():.1%}")
        return d

    raw = load_goes_dmw(discover_goes_dmw_files(case_data_dir(), satellites=[SAT])[0].path, satellite=SAT)
    keep = dmw_basic_valid_mask(raw) & raw["pressure_hpa"].between(*GOES_DMW_PRESSURE_RANGE).to_numpy()
    d = raw.loc[keep].rename(columns={"u_ms": "u_amv", "v_ms": "v_amv"}).copy()
    d["qi"] = np.where(d["dqf"].to_numpy() == 0, 100.0, 0.0)
    print(f"  operational DMW: {len(d):,} vectors, DQF = 0: {(d['qi'] > 0).mean():.1%}")
    return d.reset_index(drop=True)


def load_case_image():
    if is_goes():
        return load_goes_image()
    from data_io.case_observations import load_case_observations
    from data_io.product_discovery import infer_amv_valid_time_from_path

    inputs = read_case_inputs(PROJECT_ROOT, CASE)
    data_case_dir = case_data_dir()
    valid_time = pd.Timestamp(inputs["frame_times"][-1] if inputs.get("frame_times") else
                              infer_amv_valid_time_from_path(data_case_dir, offset_minutes=VALID_OFFSET_MINUTES))
    obs = load_case_observations(case_name=CASE, data_case_dir=data_case_dir, frame_count=4,
                                 frame_times=inputs.get("frame_times") or None,
                                 interval_minutes=inputs.get("interval_minutes") or 15.0,
                                 aoi=inputs.get("aoi"), channel="IR10.8", load_oca_data=False)
    times = pd.to_datetime(list(obs.frame_times))
    k = int(np.argmin(np.abs((times - valid_time).total_seconds())))
    bt = np.asarray(obs.images[k], dtype=np.float32).copy()
    if getattr(obs, "valid_masks", None) is not None:
        bt[~np.asarray(obs.valid_masks[k], dtype=bool)] = np.nan
    return pwd.image_from_lonlat_grid(bt, np.asarray(obs.lon_grid), np.asarray(obs.lat_grid), GEOS), valid_time


def load_operational(valid_time) -> pd.DataFrame:
    if is_goes():
        return load_goes_operational()
    from data_io.official_amv import load_official_amv_file
    from data_io.product_discovery import find_official_amv_file
    from validation.compare_era5 import standardize_official_amv

    data_case_dir = case_data_dir()
    path = find_official_amv_file(data_case_dir / "eumetsat_amv" / "native",
                                  expected_time=valid_time.to_pydatetime(), tolerance_minutes=90.0)
    off = standardize_official_amv(load_official_amv_file(path))
    q = off["official_quality"].to_numpy(dtype=float)
    off["qi"] = q * 100.0 if np.nanmedian(q) <= 1.5 else q
    print(f"  operational: {len(off):,} vectors, QI >= {QI_MIN:.0f}: {(off['qi'] >= QI_MIN).mean():.1%}")
    return off


# =============================================================================
# Stage: status
# =============================================================================




# =============================================================================
# Stage: retention
# =============================================================================


def frame_axes(ax) -> None:
    ax.tick_params(direction="in", top=True, right=True, width=0.5, length=2.5, pad=2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)




def thresholds(tracker: str) -> pd.DataFrame | None:
    op_tag = f"operational_{SAT.lower()}" if is_goes() else "operational"
    path = (PROJECT_ROOT / "outputs" / CASE / "qc_selfsup" / f"{op_tag}_qi_thresholds.csv"
            if tracker == "operational" else qc_dir(tracker) / "validation_era5_qi_thresholds.csv")
    return pd.read_csv(path) if path.is_file() else None


def draw_markers(ax, thr: pd.DataFrame | None, color: str) -> None:
    if thr is None:
        return
    for subset, marker in (("qi>=60", "o"), ("qi>=80", "^")):
        r = thr.loc[thr["subset"] == subset]
        if not r.empty:
            ax.plot(r["fraction"].iloc[0] * 100, r["mvd"].iloc[0], marker, ms=3.4, mfc="white",
                    mec=color, mew=0.8, zorder=5)




# =============================================================================
# Stage: coverage
# =============================================================================


class PixelMap:
    """Native pixel (row, col) <-> projection (x, y), from the image extent."""

    def __init__(self, image):
        self.n_rows, self.n_cols = image.bt.shape
        left, right, bottom, top = image.extent
        self.dx = (right - left) / self.n_cols
        self.dy = (bottom - top) / self.n_rows
        self.left, self.top = left, top

    def xy(self, row, col):
        return self.left + (np.asarray(col) + 0.5) * self.dx, self.top + (np.asarray(row) + 0.5) * self.dy

    def rowcol(self, x, y):
        return (np.asarray(y) - self.top) / self.dy - 0.5, (np.asarray(x) - self.left) / self.dx - 0.5


def block_field(rows, cols, weights, block: int, pm: PixelMap, earth: np.ndarray | None):
    """Smoothed per-block fraction (earth is None -> presence of >= 1 point)."""

    from scipy.ndimage import gaussian_filter

    nb_r, nb_c = -(-pm.n_rows // block), -(-pm.n_cols // block)
    r = np.clip(np.asarray(rows, dtype=np.int64) // block, 0, nb_r - 1)
    c = np.clip(np.asarray(cols, dtype=np.int64) // block, 0, nb_c - 1)
    num = np.zeros((nb_r, nb_c))
    np.add.at(num, (r, c), weights)
    if earth is None:
        num = (num > 0).astype(float)
        den = np.ones_like(num)
    else:
        den = earth
    disk = earth if earth is not None else np.ones_like(num)
    num_s = gaussian_filter(num, SMOOTH_BLOCKS, mode="constant")
    den_s = gaussian_filter(den.astype(float), SMOOTH_BLOCKS, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        field = np.where(disk > 0, num_s / den_s, np.nan)
    rc = (np.arange(nb_r) + 0.5) * block - 0.5
    cc = (np.arange(nb_c) + 0.5) * block - 0.5
    X, Y = pm.xy(rc[:, None] * np.ones((1, nb_c)), np.ones((nb_r, 1)) * cc[None, :])
    return X, Y, field


def earth_blocks(image, block: int) -> np.ndarray:
    ok = np.isfinite(image.bt)
    nb_r, nb_c = -(-ok.shape[0] // block), -(-ok.shape[1] // block)
    pad = np.zeros((nb_r * block, nb_c * block), bool)
    pad[:ok.shape[0], :ok.shape[1]] = ok
    return pad.reshape(nb_r, block, nb_c, block).sum(axis=(1, 3)).astype(float)


def box_outline(box) -> tuple[np.ndarray, np.ndarray]:
    lo0, lo1, la0, la1 = box
    t = np.linspace(0, 1, 60)
    lon = np.concatenate([lo0 + (lo1 - lo0) * t, np.full_like(t, lo1), lo1 - (lo1 - lo0) * t, np.full_like(t, lo0)])
    lat = np.concatenate([np.full_like(t, la0), la0 + (la1 - la0) * t, np.full_like(t, la1), la1 - (la1 - la0) * t])
    return GEOS.forward(lon, lat)


def central_angle(lon, lat) -> np.ndarray:
    lon, lat = np.radians(np.asarray(lon, float) - GEOS.sub_lon), np.radians(np.asarray(lat, float))
    return np.degrees(np.arccos(np.clip(np.cos(lat) * np.cos(lon), -1.0, 1.0)))


def choose_box(dense: pd.DataFrame, op: pd.DataFrame) -> tuple[float, float, float, float]:
    """Illustrative box near the sub-satellite point.

    Requirements: all corners within LOCAL_MAX_CENTRAL_ANGLE; >= min_operational
    operational vectors.  Score = mix x agreement x complexity x visibility:
      mix         1 - 2|f - 0.5|, f = LayerFlow fraction with QI >= QI_MIN (both kept and rejected present)
      agreement   1 / (1 + median |V_op - V_dense| / 3 m/s) at the operational vectors (same layer tracked)
      complexity  circular variance of the retained dense wind directions (curved / converging flow)
      visibility  min(1, 75th-pct speed / 8 m/s)
    """

    s = LOCAL_SIZE_DEG
    lon, lat = dense["longitude"].to_numpy(), dense["latitude"].to_numpy()
    kept = (dense["qi"] >= QI_MIN).to_numpy()
    u, v = dense["u_amv"].to_numpy(), dense["v_amv"].to_numpy()
    spd = np.hypot(u, v)
    opq = op.loc[op["qi"] >= QI_MIN_OPERATIONAL]
    pair = opq[["row", "col", "longitude", "latitude", "u_amv", "v_amv"]].merge(
        dense[["row", "col", "u_amv", "v_amv"]], on=["row", "col"], suffixes=("_op", "_d"))
    pvd = np.hypot(pair["u_amv_op"] - pair["u_amv_d"], pair["v_amv_op"] - pair["v_amv_d"]).to_numpy()
    plon, plat = pair["longitude"].to_numpy(), pair["latitude"].to_numpy()
    rows = []
    for lo0 in GEOS.sub_lon + np.arange(LOCAL_SEARCH["lon"][0], LOCAL_SEARCH["lon"][1] - s + 1e-6, LOCAL_SEARCH["step"]):
        for la0 in np.arange(LOCAL_SEARCH["lat"][0], LOCAL_SEARCH["lat"][1] - s + 1e-6, LOCAL_SEARCH["step"]):
            if central_angle([lo0, lo0 + s, lo0, lo0 + s], [la0, la0, la0 + s, la0 + s]).max() > LOCAL_MAX_CENTRAL_ANGLE:
                continue
            inside = (lon >= lo0) & (lon < lo0 + s) & (lat >= la0) & (lat < la0 + s)
            pin = (plon >= lo0) & (plon < lo0 + s) & (plat >= la0) & (plat < la0 + s)
            if inside.sum() < 5000 or pin.sum() < LOCAL_SEARCH["min_operational"]:
                continue
            f = kept[inside].mean()
            k = inside & kept
            if k.sum() < 500:
                continue
            ang = np.arctan2(v[k], u[k])
            complexity = 1.0 - np.hypot(np.cos(ang).mean(), np.sin(ang).mean())
            med_vd = float(np.median(pvd[pin]))
            p75 = float(np.percentile(spd[inside], 75))
            score = (1 - 2 * abs(f - 0.5)) / (1 + med_vd / 3.0) * complexity * min(1.0, p75 / 8.0)
            rows.append({"box": (float(lo0), float(lo0 + s), float(la0), float(la0 + s)), "score": score,
                         "kept_frac": f, "op_vs_dense_mvd": med_vd, "complexity": complexity, "p75_speed": p75,
                         "n_op": int(pin.sum())})
    if not rows:
        raise RuntimeError("no local box found; set LOCAL_BOX")
    cand = pd.DataFrame(rows).sort_values("score", ascending=False)
    print("top local-box candidates:")
    with pd.option_context("display.width", 200):
        print(cand.head(8).round(3).to_string(index=False))
    return cand["box"].iloc[0]


def local_vectors(df: pd.DataFrame, xlim, ylim, stride: int | None, qi_min: float = QI_MIN,
                  rejected: bool = False):
    d = df.loc[(df["qi"] < qi_min) | df["qi"].isna()] if rejected else df.loc[df["qi"] >= qi_min]
    if stride:
        d = d.loc[(d["row"] % stride == 0) & (d["col"] % stride == 0)]
    ok, x, y, U, V = pwd.project_vectors(GEOS, d["longitude"], d["latitude"], d["u_amv"], d["v_amv"])
    ok &= (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
    return x[ok], y[ok], U[ok], V[ok]


def local_speed_scale(vmax: float):
    cmap = plt.get_cmap(pwd.SPEED_CMAP).resampled(pwd.SPEED_LEVELS)
    cmap.set_over(cmap(pwd.SPEED_LEVELS - 1))
    from matplotlib.colors import BoundaryNorm
    norm = BoundaryNorm(np.linspace(0.0, vmax, pwd.SPEED_LEVELS + 1), pwd.SPEED_LEVELS)
    step = 5 if vmax <= 30 else 10
    return cmap, norm, list(np.arange(0, vmax + 1e-6, step))


def arrow_lengths(U, V, spacing_mm: float, ref_speed: float):
    """Unit direction x display length (mm), length ~ speed^GAMMA."""

    speed = np.hypot(U, V)
    length = LOCAL_ARROW_REF_SPACING * spacing_mm * (speed / ref_speed) ** LOCAL_ARROW_GAMMA
    length = np.clip(length, LOCAL_ARROW_MIN_SPACING * spacing_mm, LOCAL_ARROW_MAX_SPACING * spacing_mm)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(U / speed) * length, np.nan_to_num(V / speed) * length


def draw_local(ax, image, vec, title: str, xlim, ylim, vmax: float, spacing_mm: float, ref_speed: float,
               rej=None) -> None:
    ax.set_facecolor("white")
    ax.imshow(image.bt, extent=image.extent, origin="upper", cmap=pwd.BT_CMAP, norm=Normalize(*pwd.BT_RANGE),
              interpolation="nearest", alpha=IR_ALPHA, zorder=1)
    if rej is not None and len(rej[0]):
        Ur, Vr = arrow_lengths(rej[2], rej[3], spacing_mm, ref_speed)
        qr = ax.quiver(rej[0], rej[1], Ur, Vr, color=REJECTED_COLOR, angles="uv", scale_units="inches",
                       scale=1.0 / MM, units="inches", width=LOCAL_ARROW_WIDTH_MM * 0.8 * MM,
                       headwidth=3.0, headlength=3.6, headaxislength=3.3, minshaft=1, minlength=0,
                       pivot="middle", linewidth=0, zorder=2.5)
        qr.set_rasterized(True)
    x, y, U, V = vec
    speed = np.hypot(U, V)
    cmap, norm, _ = local_speed_scale(vmax)
    order = np.argsort(speed, kind="stable")
    Ul, Vl = arrow_lengths(U, V, spacing_mm, ref_speed)
    q = ax.quiver(x[order], y[order], Ul[order], Vl[order], np.minimum(speed, vmax * 1.001)[order],
                  cmap=cmap, norm=norm, angles="uv", scale_units="inches",
                  scale=1.0 / MM, units="inches",
                  width=LOCAL_ARROW_WIDTH_MM * MM, headwidth=3.0, headlength=3.6, headaxislength=3.3,
                  minshaft=1, minlength=0, pivot="middle", linewidth=0, zorder=3)
    q.set_rasterized(True)
    coast = pwd.projected_lines(GEOS, pwd.coastlines_lonlat())
    if coast:
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(coast, colors="#303030", linewidths=pwd.COASTLINE_PT, zorder=4))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.text(0.03, 0.97, title, transform=ax.transAxes, ha="left", va="top", zorder=6, fontsize=LOCAL_TITLE_PT,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])




_CTX: dict = {}
AGREE_MIN_N = 100
# Upper-right panel: "igra" = MVD vs IGRA radiosondes vs QI threshold (evaluate_quality_igra.py);
# "operational" = MVD between dense and collocated operational AMVs.
RIGHT_PANEL = "igra"
IGRA_MIN_LEVELS = 30


def igra_curve_file(tracker: str) -> Path:
    base = PROJECT_ROOT / "outputs" / CASE / "comparison" / (f"amv_vs_igra_{SAT.lower()}" if is_goes() else "amv_vs_igra")
    name = "operational" if tracker == "operational" else qc_name(tracker)
    return base / "qi_curves" / f"igra_qi_curve_{name}.csv"


def draw_igra(ax) -> None:
    for tracker, label, color, dash, lw in METHODS:
        f = igra_curve_file(tracker)
        if not f.is_file():
            print("  no IGRA QI curve:", f)
            continue
        c = pd.read_csv(f)
        ok = c["n_levels"] >= IGRA_MIN_LEVELS
        ax.plot(c.loc[ok, "qi_min"], c.loc[ok, "mvd"], color=color, ls=dash, lw=lw)
    ax.set_xlim(0, 100)
    for q in (60, 80):
        ax.axvline(q, color="#b0b0b0", lw=0.6, ls=(0, (3, 2)), zorder=0)
    frame_axes(ax)
    ax.set_xlabel("QI threshold")


def agreement_curve(tracker: str) -> pd.DataFrame | None:
    """Cumulative MVD between dense vectors with QI >= q and the collocated operational AMVs
    (operational QI >= QI_MIN_OPERATIONAL; GOES: DQF = 0), same pixel.  Cached per tracker."""

    cache = qc_dir(tracker) / "agreement_operational_curve.csv"
    npz = qc_npz(tracker)
    if cache.is_file() and npz.is_file() and cache.stat().st_mtime > npz.stat().st_mtime:
        return pd.read_csv(cache)
    if not npz.is_file() or not product_path(tracker).is_file():
        print("  agreement: missing", tracker)
        return None
    d = _CTX["dense"].get(tracker)
    if d is None:
        d = load_dense(tracker)
    op = _CTX["op"]
    ref = op.loc[op["qi"] >= QI_MIN_OPERATIONAL, ["row", "col", "u_amv", "v_amv"]]
    pair = ref.merge(d[["row", "col", "u_amv", "v_amv", "qi"]].drop_duplicates(["row", "col"]),
                     on=["row", "col"], suffixes=("_op", "_d"))
    vd = np.hypot(pair["u_amv_d"] - pair["u_amv_op"], pair["v_amv_d"] - pair["v_amv_op"]).to_numpy()
    qi = pair["qi"].to_numpy()
    ok = np.isfinite(vd) & np.isfinite(qi)
    vd, qi = vd[ok], qi[ok]
    rows = []
    for q in np.arange(0.0, 100.0, 1.0):
        keep = qi >= q
        rows.append({"qi_min": q, "n": int(keep.sum()), "mvd": float(vd[keep].mean()) if keep.any() else np.nan})
    curve = pd.DataFrame(rows)
    curve.to_csv(cache, index=False)
    k = qi >= QI_MIN
    summary = pd.DataFrame([{"tracker": tracker, "n_pairs": int(vd.size), "mvd_all": float(vd.mean()),
                             "n_kept": int(k.sum()), "mvd_kept": float(vd[k].mean()) if k.any() else np.nan,
                             "n_rejected": int((~k).sum()), "mvd_rejected": float(vd[~k].mean()) if (~k).any() else np.nan}])
    summary.to_csv(qc_dir(tracker) / "agreement_operational_summary.csv", index=False)
    print("  agreement with operational:", summary.round(3).to_dict("records")[0])
    return curve


def draw_agreement(ax) -> None:
    for tracker, label, color, dash, lw in METHODS:
        if tracker == "operational":
            continue
        c = agreement_curve(tracker)
        if c is None:
            continue
        ok = c["n"] >= AGREE_MIN_N
        ax.plot(c.loc[ok, "qi_min"], c.loc[ok, "mvd"], color=color, ls=dash, lw=lw)
    ax.set_xlim(0, 100)
    for q in (60, 80):
        ax.axvline(q, color="#b0b0b0", lw=0.6, ls=(0, (3, 2)), zorder=0)
    frame_axes(ax)
    ax.set_xlabel("QI threshold")


def coverage_block_height() -> float:
    s = FIG_WIDTH_MM / 90.0
    disk = (FIG_WIDTH_MM - 2 * 1.0 * s - 2.0 * s) / 2.0
    return 1.0 * s + disk + (10.0 if DISK_MODE == "qi_map" else 14.5)


# ---- QI curves: cumulative MVD vs QI threshold (top) + QI distribution (bottom) ----
CURVE_MIN_N = 200
# Bottom panel: "classes" = share of vectors per QI class (grouped bars, classes of the QI map);
# "retained" = fraction with QI >= q; "pdf" = fine histogram.
# "prob" = probability of each QI interval (0-1, histogram outline); "cdf" = P(QI <= q);
# "share" = % per QI bin as a line.
QI_BOTTOM = "prob"
QI_PROB_BIN = 10.0
QI_SHARE_BIN = 2.0
QI_CLASSES = (0, 20, 40, 60, 80, 100)
CURVE_MIN_FRACTION = 0.005


def qi_curve_files(tracker: str) -> tuple[Path, Path]:
    if tracker == "operational":
        base = PROJECT_ROOT / "outputs" / CASE / "qc_selfsup"
        tag = f"operational_{SAT.lower()}" if is_goes() else "operational"
        return base / f"{tag}_qi_curve.csv", base / f"{tag}_qi_hist.csv"
    return qc_dir(tracker) / "validation_era5_qi_curve.csv", qc_dir(tracker) / "validation_era5_qi_hist.csv"


def draw_qi_curves(ax_mvd, ax_pdf, legend: bool = True) -> None:
    for tracker, label, color, dash, lw in METHODS:
        fc, fh = qi_curve_files(tracker)
        if not fc.is_file() or not fh.is_file():
            print("  no QI curve:", tracker, fc)
            continue
        c, h = pd.read_csv(fc), pd.read_csv(fh)
        ok = (c["n"] >= CURVE_MIN_N) & (c["fraction"] >= CURVE_MIN_FRACTION)
        ax_mvd.plot(c.loc[ok, "qi_min"], c.loc[ok, "mvd"], color=color, ls=dash, lw=lw, label=label)
        if QI_BOTTOM == "prob":
            edges = np.arange(0.0, 100.0 + QI_PROB_BIN, QI_PROB_BIN)
            cnt = np.asarray([h.loc[(h["bin_left"] >= lo) & (h["bin_left"] < hi), "count"].sum()
                              for lo, hi in zip(edges[:-1], edges[1:])], float)
            prob = cnt / max(cnt.sum(), 1.0)
            ax_pdf.step(edges, np.r_[prob, prob[-1]], where="post", color=color, ls=dash, lw=lw)
        elif QI_BOTTOM == "cdf":
            cum = np.cumsum(h["count"].to_numpy(float)) / max(h["count"].sum(), 1)
            ax_pdf.plot(h["bin_right"], cum, color=color, ls=dash, lw=lw)
        elif QI_BOTTOM == "share":
            edges = np.arange(0.0, 100.0 + QI_SHARE_BIN, QI_SHARE_BIN)
            share = [h.loc[(h["bin_left"] >= lo) & (h["bin_left"] < hi), "count"].sum()
                     for lo, hi in zip(edges[:-1], edges[1:])]
            share = 100.0 * np.asarray(share) / max(h["count"].sum(), 1)
            ax_pdf.plot(0.5 * (edges[:-1] + edges[1:]), share, color=color, ls=dash, lw=lw)
        elif QI_BOTTOM == "classes":
            k = [m[0] for m in METHODS].index(tracker)
            nb, width = len(METHODS), (QI_CLASSES[1] - QI_CLASSES[0]) / (len(METHODS) + 1.5)
            share = [h.loc[(h["bin_left"] >= lo) & (h["bin_left"] < hi), "count"].sum() for lo, hi in
                     zip(QI_CLASSES[:-1], QI_CLASSES[1:])]
            share = 100.0 * np.asarray(share) / max(h["count"].sum(), 1)
            centers = 0.5 * (np.asarray(QI_CLASSES[:-1]) + np.asarray(QI_CLASSES[1:]))
            xs = centers + (k - (nb - 1) / 2.0) * width
            hatched = dash != "-"
            ax_pdf.bar(xs, share, width=width * 0.92, color="white" if hatched else color, edgecolor=color,
                       linewidth=0.5, hatch="//////" if hatched else None, zorder=2)
        elif QI_BOTTOM == "retained":
            ax_pdf.plot(c["qi_min"], c["fraction"] * 100.0, color=color, ls=dash, lw=lw)
        else:
            x = np.r_[h["bin_left"].to_numpy(), h["bin_right"].iloc[-1]]
            y = np.r_[h["density"].to_numpy(), h["density"].iloc[-1]]
            ax_pdf.step(x, y * 100.0, where="post", color=color, ls=dash, lw=lw * 0.8)
    for ax in (ax_mvd, ax_pdf):
        ax.set_xlim(0, 100)
        for q in (60, 80):
            ax.axvline(q, color="#b0b0b0", lw=0.6, ls=(0, (3, 2)), zorder=0)
        frame_axes(ax)
    ax_mvd.set_ylabel("MVD (m s$^{-1}$)")
    ax_mvd.tick_params(labelbottom=False)
    if QI_BOTTOM in ("prob", "cdf"):
        ax_pdf.set_xlabel("QI threshold (top) / QI (bottom)")
        ax_pdf.set_ylabel("Probability" if QI_BOTTOM == "prob" else "Cumulative prob.")
        ax_pdf.set_ylim(0, None if QI_BOTTOM == "prob" else 1.0)
        if QI_BOTTOM == "prob":
            ax_pdf.set_xticks(np.arange(0, 101, 20))
    elif QI_BOTTOM == "share":
        ax_pdf.set_xlabel("QI threshold (top) / QI (bottom)")
        ax_pdf.set_ylabel("Vectors (%)")
        ax_pdf.set_ylim(bottom=0)
    elif QI_BOTTOM == "classes":
        ax_pdf.set_xlabel("QI threshold (top) / QI class (bottom)")
        ax_pdf.set_ylabel("Vectors (%)")
        ax_pdf.set_xticks(QI_CLASSES)
        ax_pdf.grid(False)
    elif QI_BOTTOM == "retained":
        ax_pdf.set_xlabel("QI threshold")
        ax_pdf.set_ylabel("Retained (%)")
        ax_pdf.set_ylim(0, 100)
    else:
        ax_pdf.set_xlabel("QI threshold (top) / QI (bottom)")
        ax_pdf.set_ylabel("PDF (%)")
    for ax in (ax_mvd, ax_pdf):                      # aligned y labels
        ax.yaxis.set_label_coords(-0.085, 0.5)
    ax_pdf.yaxis.set_major_locator(plt.MaxNLocator(3))
    if legend:
        ax_mvd.legend(frameon=False, loc="lower left", ncol=2, fontsize=LEGEND_PT, handlelength=1.8,
                      columnspacing=0.8, labelspacing=0.2, borderaxespad=0.3)




def set_case(case: str, sat: str | None = None) -> None:
    global CASE, OUTPUT_DIR, LOCAL_BOX, SAT, GEOS
    CASE = case
    SAT = (sat or GOES_SAT_DEFAULT).upper() if is_goes() else None
    if not is_goes():
        GEOS = pwd.SEVIRI_GEOS
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "qc" / (f"{case}_{SAT.lower()}" if SAT else case)
    key = f"{case}:{SAT}" if SAT else case
    LOCAL_BOX = LOCAL_BOXES.get(key, LOCAL_BOXES.get(case))


SHOW_CASE_LABELS = True


def case_label() -> str:
    return f"GOES-{SAT[1:]}" if is_goes() else "SEVIRI"      # G16 -> GOES-16


def stage_coverage(fig=None, X0: float = 0.0, Y0: float = 0.0) -> float:
    """QI disk + 2x2 local winds.  Draws into `fig` at (X0, Y0) mm when given; returns the block height."""
    from matplotlib.collections import LineCollection

    pwd.configure_fonts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("loading image / products ...")
    image, valid_time = load_case_image()
    pm = PixelMap(image)
    op = load_operational(valid_time)
    orow, ocol = pm.rowcol(*GEOS.forward(op["longitude"].to_numpy(), op["latitude"].to_numpy()))
    ok = np.isfinite(orow) & np.isfinite(ocol)
    op = op.loc[ok].copy()
    op["row"] = np.rint(orow[ok]).astype(np.int64)
    op["col"] = np.rint(ocol[ok]).astype(np.int64)
    dense = {t: load_dense(t) for t in COVERAGE_METHODS if t != "operational"}
    if BEFORE_QC_TRACKER not in dense:
        dense[BEFORE_QC_TRACKER] = load_dense(BEFORE_QC_TRACKER)

    box = LOCAL_BOX or choose_box(dense[BEFORE_QC_TRACKER], op)
    print("local box (lon_min, lon_max, lat_min, lat_max):", box)

    # ---- layout --------------------------------------------------------------
    s = FIG_WIDTH_MM / 90.0
    pad, gap, inner = 1.0 * s, 2.0 * s, 1.0 * s
    disk = (FIG_WIDTH_MM - 2 * pad - gap) / 2.0
    panel = (disk - inner) / 2.0
    legend_h, cbar_h = (10.0 if DISK_MODE == "qi_map" else 14.5), 2.0
    W, H = FIG_WIDTH_MM, pad + disk + legend_h
    own = fig is None
    if own:
        fig = plt.figure(figsize=(W * MM, H * MM))
    FW, FH = fig.get_size_inches() / MM
    y0 = Y0 + legend_h
    ax_d = pwd.disk_axes(fig, [(X0 + pad) / FW, y0 / FH, disk / FW, disk / FH], GEOS, "white")
    xr = X0 + pad + disk + gap
    axes_l = [fig.add_axes([(xr + (k % 2) * (panel + inner)) / FW, (y0 + (1 - k // 2) * (panel + inner)) / FH,
                            panel / FW, panel / FH]) for k in range(4)]

    # ---- disk: IR + coverage contours ----------------------------------------
    im = ax_d.imshow(image.bt, extent=image.extent, origin="upper", cmap=pwd.BT_CMAP,
                     norm=Normalize(*pwd.BT_RANGE), interpolation="nearest", alpha=IR_ALPHA, zorder=1)
    im.set_clip_path(ax_d._limb)
    pwd.decorate(ax_d, GEOS, on_image=False)
    handles, stats = [], []
    if DISK_MODE == "qi_map":
        from matplotlib.colors import BoundaryNorm
        d = dense[QI_MAP_TRACKER]
        grid = np.full(image.bt.shape, np.nan, np.float32)
        ok = (d["row"].between(0, pm.n_rows - 1) & d["col"].between(0, pm.n_cols - 1)).to_numpy()
        grid[d["row"].to_numpy()[ok], d["col"].to_numpy()[ok]] = d["qi"].to_numpy()[ok]
        qcmap = plt.get_cmap(QI_MAP_CMAP).resampled(len(QI_MAP_BOUNDS) - 1)
        qnorm = BoundaryNorm(QI_MAP_BOUNDS, qcmap.N)
        qim = ax_d.imshow(grid, extent=image.extent, origin="upper", cmap=qcmap, norm=qnorm,
                          interpolation="nearest", zorder=2)
        qim.set_clip_path(ax_d._limb)
        qim.set_rasterized(True)
        stats = [{"method": STYLE[QI_MAP_TRACKER][1], "qi>=60": float(np.nanmean(grid >= 60)),
                  "qi>=80": float(np.nanmean(grid >= 80)),
                  "disk_fraction_with_vector": float(np.isfinite(grid).sum() / np.isfinite(image.bt).sum())}]
        cax_q = fig.add_axes([(X0 + pad + 3.0) / FW, (Y0 + legend_h - 1.3 - cbar_h) / FH, (disk - 6.0) / FW, cbar_h / FH])
        qbar = mcolorbar.ColorbarBase(cax_q, cmap=qcmap, norm=qnorm, orientation="horizontal",
                                      ticks=list(QI_MAP_BOUNDS), spacing="proportional")
        qbar.outline.set_linewidth(0.5)
        cax_q.tick_params(width=0.5, length=2.0, pad=1.0, labelsize=LEGEND_PT)
        cax_q.minorticks_off()
        qbar.solids.set_rasterized(True)
        qbar.set_label(f"QI ({STYLE[QI_MAP_TRACKER][1]})", fontsize=LEGEND_PT, labelpad=1.0)
    else:
        earth = earth_blocks(image, BLOCK_PX)

        def contour(X, Y, F, level, color, dash, lw, label):
            cs = ax_d.contour(X, Y, np.nan_to_num(F, nan=0.0), levels=[level], colors=[color],
                              linewidths=lw, linestyles=[dash], zorder=6)
            for coll in (getattr(cs, "collections", None) or [cs]):
                coll.set_clip_path(ax_d._limb)
            handles.append(Line2D([], [], color=color, ls=dash, lw=lw, label=label))

        presence = COVERAGE_MODE == "presence"
        earth_c = earth_blocks(image, COVERAGE_BLOCK_PX)

        def presence_field(rows, cols):
            X, Y, F = block_field(rows, cols, np.ones(len(rows)), COVERAGE_BLOCK_PX, pm, None)
            return X, Y, np.where(earth_c > 0, F, np.nan)

        majority = COVERAGE_MODE == "majority"

        def counts(rows, cols):
            nb_r, nb_c = earth_c.shape
            r = np.clip(np.asarray(rows, dtype=np.int64) // COVERAGE_BLOCK_PX, 0, nb_r - 1)
            c = np.clip(np.asarray(cols, dtype=np.int64) // COVERAGE_BLOCK_PX, 0, nb_c - 1)
            out = np.zeros((nb_r, nb_c))
            np.add.at(out, (r, c), 1.0)
            return out

        def ratio_field(num, den, valid):
            from scipy.ndimage import gaussian_filter
            num_s = gaussian_filter(num * valid, SMOOTH_BLOCKS, mode="constant")
            den_s = gaussian_filter(den * valid, SMOOTH_BLOCKS, mode="constant")
            with np.errstate(invalid="ignore", divide="ignore"):
                F = np.where(valid & (den_s > 0), num_s / den_s, 0.0)
            nb_r, nb_c = earth_c.shape
            rc = (np.arange(nb_r) + 0.5) * COVERAGE_BLOCK_PX - 0.5
            cc = (np.arange(nb_c) + 0.5) * COVERAGE_BLOCK_PX - 0.5
            X, Y = pm.xy(rc[:, None] * np.ones((1, nb_c)), np.ones((nb_r, 1)) * cc[None, :])
            return X, Y, np.where(earth_c > 0, F, np.nan)

        def majority_after(d, qi_min, operational: bool):
            n_all = counts(d["row"], d["col"])
            kept = d.loc[d["qi"] >= qi_min]
            n_ok = counts(kept["row"], kept["col"])
            if operational:
                valid = n_all >= MIN_OPERATIONAL_VECTORS
            else:                                   # nested in the product's own before-QC coverage
                valid = ratio_field(n_all, earth_c, earth_c > 0)[2] >= MAJORITY_LEVEL
            return ratio_field(n_ok, n_all, np.nan_to_num(valid, nan=0).astype(bool) & (earth_c > 0))

        d0 = dense[BEFORE_QC_TRACKER]
        if majority:
            X, Y, F = ratio_field(counts(d0["row"], d0["col"]), earth_c, earth_c > 0)
            contour(X, Y, F, MAJORITY_LEVEL, BEFORE_QC_COLOR, "-", CONTOUR_PT, f"{STYLE[BEFORE_QC_TRACKER][1]}, before QC")
        elif presence:
            X, Y, F = presence_field(d0["row"], d0["col"])
            contour(X, Y, F, PRESENCE_LEVEL, BEFORE_QC_COLOR, "-", CONTOUR_PT, f"{STYLE[BEFORE_QC_TRACKER][1]}, before QC")
        else:
            X, Y, F = block_field(d0["row"], d0["col"], np.ones(len(d0)), BLOCK_PX, pm, earth)
            contour(X, Y, F, CONTOUR_LEVEL, BEFORE_QC_COLOR, "-", CONTOUR_PT, f"{STYLE[BEFORE_QC_TRACKER][1]}, before QC")
        stats = [{"method": "before QC", "area_fraction_inside_contour":
                  float(np.nanmean(F >= (MAJORITY_LEVEL if majority else PRESENCE_LEVEL if presence else CONTOUR_LEVEL)))}]
        for tracker in DISK_CONTOUR_METHODS:
            _, label, color, dash, lw = STYLE[tracker]
            if majority:
                is_op = tracker == "operational"
                X, Y, F = majority_after(op if is_op else dense[tracker], QI_MIN_OPERATIONAL if is_op else QI_MIN, is_op)
                level = MAJORITY_LEVEL
            elif presence:
                d = op if tracker == "operational" else dense[tracker]
                kept = d.loc[d["qi"] >= QI_MIN]
                X, Y, F = presence_field(kept["row"], kept["col"])
                level = PRESENCE_LEVEL
            elif tracker == "operational":
                kept = op.loc[op["qi"] >= QI_MIN]
                X, Y, F = block_field(kept["row"], kept["col"], np.ones(len(kept)), OPERATIONAL_BLOCK_PX, pm, None)
                level = OPERATIONAL_LEVEL
                F = np.where(earth_blocks(image, OPERATIONAL_BLOCK_PX) > 0, F, np.nan)
            else:
                d = dense[tracker]
                k = (d["qi"] >= QI_MIN).to_numpy()
                X, Y, F = block_field(d["row"][k], d["col"][k], np.ones(k.sum()), BLOCK_PX, pm, earth)
                level = CONTOUR_LEVEL
            contour(X, Y, F, level, color, dash, CONTOUR_PT * (1.3 if lw > 1.2 else 1.0), label)
            stats.append({"method": label, "area_fraction_inside_contour": float(np.nanmean(F >= level))})
    bx, by = box_outline(box)
    xs, ys = bx[np.isfinite(bx)], by[np.isfinite(by)]
    half = 0.5 * max(xs.max() - xs.min(), ys.max() - ys.min())        # square window in the satellite view
    xc, yc = 0.5 * (xs.max() + xs.min()), 0.5 * (ys.max() + ys.min())
    xlim, ylim = (xc - half, xc + half), (yc - half, yc + half)
    ax_d.plot([xlim[0], xlim[1], xlim[1], xlim[0], xlim[0]], [ylim[0], ylim[0], ylim[1], ylim[1], ylim[0]],
              color=BOX_COLOR, lw=BOX_PT, zorder=8)
    print(pd.DataFrame(stats).round(3).to_string(index=False))

    if handles:
        leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=((X0 + pad) / FW, (Y0 + legend_h - 0.3) / FH),
                         ncol=1, frameon=False, handlelength=2.0, labelspacing=0.15,
                         handletextpad=0.5, borderaxespad=0.0, fontsize=LEGEND_PT)
        leg.set_in_layout(False)

    # ---- local panels -------------------------------------------------------
    stride = max(1, int(round(2 * half / abs(pm.dx) / LOCAL_ARROWS_ACROSS)))
    vmax = LOCAL_SPEED_MAX
    if vmax is None:
        ref = local_vectors(dense[BEFORE_QC_TRACKER], xlim, ylim, None)
        vmax = float(max(20.0, np.ceil(np.percentile(np.hypot(ref[2], ref[3]), 99) / 10.0) * 10.0))
    ref_all = local_vectors(dense[BEFORE_QC_TRACKER], xlim, ylim, None, -np.inf)
    ref_speed = max(1.0, float(np.percentile(np.hypot(ref_all[2], ref_all[3]), 75)))
    spacing_mm = panel / LOCAL_ARROWS_ACROSS
    print(f"  local window: {2 * half / 1000:.0f} km, stride {stride} px, speed max {vmax:.0f} m/s")
    for ax, tracker in zip(axes_l, COVERAGE_METHODS):
        df = op if tracker == "operational" else dense[tracker]
        is_op = tracker == "operational"
        args = (None, QI_MIN_OPERATIONAL) if is_op else (stride, QI_MIN)
        vec = local_vectors(df, xlim, ylim, *args)
        rej = local_vectors(df, xlim, ylim, *args, rejected=True) if LOCAL_SHOW_REJECTED else None
        draw_local(ax, image, vec, LOCAL_TITLES.get(tracker, STYLE[tracker][1]), xlim, ylim, vmax,
                   spacing_mm, ref_speed, rej)
        if rej is not None:
            print(f"  local {tracker}: rejected {len(rej[0]):,} (grey)")
        print(f"  local {tracker}: {len(vec[0]):,} arrows drawn")

    cmap, norm, ticks = local_speed_scale(vmax)
    cax = fig.add_axes([(xr + 2.0) / FW, (Y0 + legend_h - 1.3 - cbar_h) / FH, (disk - 4.0) / FW, cbar_h / FH])
    bar = mcolorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal", extend="max",
                                 extendfrac=0.03, ticks=ticks, spacing="proportional")
    bar.outline.set_linewidth(0.5)
    cax.tick_params(width=0.5, length=2.0, pad=1.0, labelsize=LEGEND_PT)
    cax.minorticks_off()
    bar.solids.set_rasterized(True)
    bar.set_label(pwd.SPEED_LABEL, fontsize=LEGEND_PT, labelpad=1.0)

    if own:
        stem = OUTPUT_DIR / "qc_coverage"
        fig.savefig(stem.with_suffix(".pdf"), dpi=pwd.RASTER_DPI)
        fig.savefig(stem.with_suffix(".png"), dpi=600)
        plt.close(fig)
        print("wrote", stem.with_suffix(".pdf"), f"({W:.0f} x {H:.1f} mm; disk {disk:.1f} mm, panels {panel:.1f} mm)")
    _CTX.update({"pm": pm, "op": op, "dense": dense})
    return H


def stage_figure(cases: list[str], sat: str | None = None) -> None:
    """One row per case: [QI disk + 2x2 local winds] | [MVD vs QI threshold over QI PDF]; 180 mm wide."""

    pwd.configure_fonts()
    hb, gap = coverage_block_height(), 3.0
    W = 2 * FIG_WIDTH_MM
    H = len(cases) * hb + (len(cases) - 1) * gap
    fig = plt.figure(figsize=(W * MM, H * MM))
    for k, case in enumerate(cases):
        set_case(case, sat)
        y0 = H - (k + 1) * hb - k * gap
        print(f"\n--- row {k + 1}: {case}")
        stage_coverage(fig, X0=0.0, Y0=y0)
        if SHOW_CASE_LABELS:     # bold case label in the empty upper-left corner of the disk (as in the IGRA figure)
            fig.text(0.6 / W, (y0 + hb - 0.4) / H, case_label(), ha="left", va="top", fontweight="bold")
        # right block, 2 x 2:  [MVD vs ERA5 | MVD vs operational] over [QI probability | legend]
        left, right = FIG_WIDTH_MM + 12.0, W - 6.0
        bottom, top = y0 + 9.0, y0 + hb - 2.0
        cgap, rgap = 3.0, 2.0
        cw = (right - left - cgap) / 2.0
        h_low = 0.36 * (top - bottom - rgap)
        h_up = top - bottom - rgap - h_low
        y_up = bottom + h_low + rgap
        ax_pdf = fig.add_axes([left / W, bottom / H, cw / W, h_low / H])
        ax_mvd = fig.add_axes([left / W, y_up / H, cw / W, h_up / H], sharex=ax_pdf)
        ax_op = fig.add_axes([(left + cw + cgap + 2.0) / W, y_up / H, (cw - 2.0) / W, h_up / H])
        draw_qi_curves(ax_mvd, ax_pdf, legend=False)
        for ax in (ax_mvd, ax_pdf):
            ax.yaxis.set_label_coords(-0.17, 0.5)
        (draw_igra if RIGHT_PANEL == "igra" else draw_agreement)(ax_op)
        ax_op.yaxis.tick_right()
        ax_op.tick_params(axis="y", left=True, right=True, labelleft=False, labelright=True)
        ax_op.set_xticks(np.arange(0, 101, 20))
        ax_op.set_xlabel("QI", fontsize=LEGEND_PT, labelpad=0.5)
        ax_op.tick_params(labelsize=LEGEND_PT, pad=1)
        for ax, txt in ((ax_mvd, "vs ERA5"), (ax_op, "vs IGRA" if RIGHT_PANEL == "igra" else "vs operational")):
            ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top", fontsize=LEGEND_PT,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
        ax_pdf.set_xlabel("QI", fontsize=LEGEND_PT)
        h, l = ax_mvd.get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=LEGEND_PT - 0.5, handlelength=1.6,
                   columnspacing=0.6, handletextpad=0.4, labelspacing=0.15, borderaxespad=0.0,
                   bbox_to_anchor=((left + cw + cgap + cw / 2) / W, (bottom - 1.0) / H))
    out = PROJECT_ROOT / "outputs" / "figures" / "qc"
    out.mkdir(parents=True, exist_ok=True)
    tag = "_".join(c.removeprefix("test_")[:24] for c in cases) + (f"_{sat.lower()}" if sat else "")
    stem = out / ("qc_figure_" + tag)
    fig.savefig(stem.with_suffix(".pdf"), dpi=pwd.RASTER_DPI)
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)
    print("wrote", stem.with_suffix(".pdf"), f"({W:.0f} x {H:.1f} mm)")


def main(argv: list[str]) -> int:
    """Render the quality-assessment composite."""
    args = list(argv[1:])
    sat = None
    if "--sat" in args:
        k = args.index("--sat")
        sat = args[k + 1].upper()
        del args[k:k + 2]
    cases = [a for a in args if a != "figure"]
    if not cases:
        cases = ["test_2018-01-12T11", "goes_g16_g18_E20240804T120506_W20240804T120508"]
    set_case(cases[0], sat)
    stage_figure(cases, sat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
