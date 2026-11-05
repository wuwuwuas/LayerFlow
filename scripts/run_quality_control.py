"""Self-supervised quality control for dense AMVs (no ERA5 / NWP / radiosondes).

Idea
----
Each dense wind is the mean of K temporal components (SEVIRI: C1/C2/C3).  If
the component errors are independent with per-axis variance sigma^2, their
dispersion around the component mean

    S = sum_k |V_k - V_bar|^2     ~   sigma^2 * chi^2_{2(K-1)}

is an unbiased (but very noisy) per-pixel estimate of sigma^2.  A small MLP is
trained to predict log sigma^2 from internal features only (image texture,
brightness-constancy residual, wind-field smoothness, cloud-product
consistency, geometry) by minimising the chi-square negative log-likelihood

    L = (K - 1) log sigma^2 + S / (2 sigma^2).

The network prediction is then used as a prior and updated with the pixel's
own dispersion (inverse-gamma / empirical Bayes; one scalar alpha fitted by
marginal likelihood):

    sigma^2_post = ((alpha - 1) sigma^2_NN + S/2) / (alpha + K - 2).

Expected RMS vector error of the final (component-mean) wind:  EE = sqrt(2/K) * sigma.

The fit is done per case (test-time optimisation; no weights are transferred
between cases).  Two-fold spatial cross-fitting gives every point a prediction
from a model that did not see it; training runs until the training NLL has
converged (learning-rate halving on plateau; no validation split).  The quality index is point-wise: the
posterior probability that the tracking error of the component-mean wind is
below a tolerance tau,

    QI = 100 * P(|eps| <= tau) = 100 * [1 - (1 + K tau^2 / (2 b))^(-a)],
    a = alpha + K - 1,   b = (alpha - 1) sigma^2_NN + S/2,

with tau = delta * dx / dt (delta = sub-pixel displacement accuracy, 0.15 px;
dx = sub-satellite pixel size; dt = image interval): SEVIRI 3 km / 15 min and
GOES 2 km / 10 min both give 3.33 m/s per pixel, i.e. tau = 0.5 m/s.  Each
vector's QI depends only on its own features, component spread and K, so the
fraction retained at QI >= 60 / 80 varies from case to case.  (The within-case
percentile of EE_post is also saved as qi_percentile.)

No external data are used to build the QC.  ERA5 is read only in the final
validation block (retention / reliability curves; the operational AMVs ranked
by their own QI are shown for reference).

Run from the project root:

    python scripts/run_quality_control.py                                   # defaults below
    python scripts/run_quality_control.py test_2018-01-12T11 pressure_conditioned_flow raw_deepflow_opencv
    python scripts/run_quality_control.py goes_g16_g18_E20240804T120506_W20240804T120508 --sat G16 pressure_conditioned_flow
    python scripts/run_quality_control.py test_2018-01-12T11 --rescore pressure_conditioned_flow   # point-wise QI only

GOES: K = 2 components (C1 centre->previous, C2 centre->next) read from the
<tracker>_<sat>_<geometry>_final_native.npz; ACHP pressure of the three frames
gives the pressure-track range; ERA5 validation reuses the existing PCF cache.

Outputs: outputs/<case>/qc_selfsup/<tracker>/
    <tracker>_selfsup_qc.npz         row/col + quality_index, ee_post, ee_nn, ee_spread, K
    features_importance.csv          permutation importance (held-out NLL increase)
    validation_era5_*.csv / *.png    retention + reliability curves (all points)
    training_log.csv
"""

from __future__ import annotations

import os

from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.name.lower() in {"examples", "notebooks"}:
    PROJECT_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# =============================================================================
# Settings
# =============================================================================
CASE_NAME = os.environ.get("AMV_CASE", "test_2018-01-12T11")
TRACKERS = ["pressure_conditioned_flow"]       # LayerFlow; add e.g. "raw_deepflow_opencv", "nn_layerwise_flowdiffuser"
SUB_SATELLITE_LON = 0.0                          # SEVIRI (GOES zenith comes from the product)
GOES_GEOMETRY = "cloud_height"
# (component, image index of the anchor frame, image index of the target frame)
SEVIRI_COMPONENTS = (("c1", 0, 1), ("c2", 1, 2), ("c3", 2, 3))
GOES_COMPONENTS = (("c1", 1, 0), ("c2", 1, 2))  # images = [previous, center, next]

# Point-wise QI tolerance: tau = QI_DELTA_PX * pixel size / image interval.
# delta = 0.15 px lies between the ~0.1 px RMS random error of cross-correlation
# PIV (Westerweel et al., 1997) and the 0.1-0.5 px endpoint errors of the best
# optical-flow methods (Baker et al., 2011).
QI_DELTA_PX = 0.15
# Speed-relative part of the tolerance: tau_i = max(tau, QI_REL_TOL * |V_i|).
# Displacement errors of optical flow grow with the displacement itself (e.g. the
# speed-binned endpoint errors of MPI Sintel), and operational QIs use
# speed-dependent tolerances; without it fast jet-stream winds are penalised for
# their absolute error.  0 = absolute tolerance only.  CLI: --rel 0.05
QI_REL_TOL = 0.03                                # with the height term; 0 = absolute tolerance only

# Height-assignment term.  The component winds share the assigned height, so a
# height error moves all of them alike and is invisible in S.  Its effect is
# |dV/dp|^2 sigma_p^2 / K_p: dV/dp = local vertical shear of the dense field
# (ridge regression of u, v on the assigned pressure in a SHEAR_WINDOW_PX window),
# sigma_p^2 = height-assignment noise, learned like sigma^2 from the dispersion of
# the K_p component pressures (chi-square with K_p - 1 dof, cross-fitting, EB).
HEIGHT_TERM = True
# sigma_p^2 estimator: "local" = local method of moments, mean of S_p/(K_p - 1) over a
# SIGMA_P_WINDOW_PX window; "nn" uses a second network and empirical Bayes.
HEIGHT_SIGMA_MODE = "local"
# Windows in km (sensor independent; converted with the sub-satellite pixel size,
# SEVIRI 3 km -> 15 / 7 px, ABI 2 km -> 23 / 11 px).
SHEAR_WINDOW_KM = 45.0
SIGMA_P_WINDOW_KM = 21.0


def window_px(km: float, kind: str) -> int:
    n = int(round(km / SSP_PIXEL_KM[kind]))
    return n + 1 if n % 2 == 0 else n
SHEAR_RIDGE_HPA = 10.0                            # ridge: single-layer windows -> shear ~ 0
SPEED_BINS = (0.0, 10.0, 20.0, 30.0, np.inf)
SSP_PIXEL_KM = {"seviri": 3.0, "goes": 2.0}      # IR sub-satellite sampling
SEVIRI_INTERVAL_S = 900.0
GOES_INTERVAL_S_FALLBACK = 600.0                 # full disk, mode 6


def is_goes(case: str) -> bool:
    from runtime_options import read_case_inputs
    record = read_case_inputs(PROJECT_ROOT, case)
    if record:
        return record["sensor"] == "goes"
    return case.lower().startswith("goes")


def paths(case: str, tracker: str, sat: str | None = None) -> dict:
    out = PROJECT_ROOT / "outputs" / case
    if is_goes(case):
        name = f"{tracker}_{sat.lower()}"
        return {
            "case": case, "sat": sat, "kind": "goes", "out": out,
            "final": out / "trackers" / name / "final" / f"{name}_{GOES_GEOMETRY}_final_native.npz",
            "product": out / "pressure_assignment" / f"{name}_{GOES_GEOMETRY}_final_amv_goes_pressure.npz",
            "era5_dense": None, "era5_official": None,
            "result": out / "qc_selfsup" / name, "name": name,
        }
    era5_dir = out / "comparison" / "amv_vs_era5" / "era5_collocated"
    return {
        "case": case, "sat": None, "kind": "seviri", "out": out,
        "final": out / "trackers" / tracker / "final" / f"{tracker}_final_native.npz",
        "product": out / "pressure_assignment" / f"{tracker}_final_amv_oca_pressure.npz",
        "era5_dense": era5_dir / "of_dense_piv_oca_pressure_era5_collocated.npz",
        "era5_official": era5_dir / "official_amv_era5_collocated.npz",
        "result": out / "qc_selfsup" / tracker, "name": tracker,
    }


# Two-fold spatial cross-fitting (blocks of BLOCK_PX x BLOCK_PX pixels).
BLOCK_PX = 64
SEED = 0
MAX_TRAIN_POINTS = 2_000_000                     # per fold

# MLP
HIDDEN = (64, 64, 64)
# Training runs until the training NLL has converged (no validation split).
# Every new minimum of the epoch NLL (lower than the best so far by more than
# MIN_DELTA nats/sample) is checkpointed.  No new minimum for PATIENCE_LR epochs
# -> learning rate halved (not below LR_MIN); no new minimum for PATIENCE_STOP
# epochs -> converged.  The checkpoint of the lowest NLL is used afterwards.
WARMUP_EPOCHS = 200                              # fixed lr, no checkpoint / no stopping before this
MAX_EPOCHS = 1000                                # safety limit only
MIN_DELTA = 1e-4
PATIENCE_LR = 5
PATIENCE_STOP = 20
LR_MIN = 1e-5
BATCH = 8192
LR = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = "auto"                                  # "auto", "cuda", "cpu"

# Feature scales (pixels)
GRAD_SIGMA_PX = 1.5
TEXTURE_WINDOW_PX = 7
WIND_GRAD_SIGMA_PX = 2.0
PRESSURE_WINDOW_PX = 7

RETENTION = np.round(np.arange(0.1, 1.0001, 0.05), 3)


# =============================================================================
# Data
# =============================================================================


def load_inputs(case: str, sat: str | None = None) -> dict:
    if is_goes(case):
        return load_inputs_goes(case, sat)
    from data_io.case_observations import load_case_observations
    from runtime_options import read_case_inputs

    record = read_case_inputs(PROJECT_ROOT, case)
    data_case_dir = Path(record["data_dir"]) if record else PROJECT_ROOT / "data" / case.removeprefix("test_")
    interval = float(record.get("interval_minutes") or 15.0)
    obs = load_case_observations(case_name=case, data_case_dir=data_case_dir, frame_count=4,
                                 frame_times=record.get("frame_times") or None, aoi=record.get("aoi"),
                                 interval_minutes=interval, channel="IR10.8", load_oca_data=True)
    images = np.asarray(obs.images, dtype=np.float32)
    valid = np.asarray(obs.valid_masks, dtype=bool)
    images[~valid] = np.nan
    pressure = None if obs.frame_pressure_hpa is None else np.asarray(obs.frame_pressure_hpa, dtype=np.float32)
    cloud = None if obs.frame_cloud_probability is None else np.asarray(obs.frame_cloud_probability, dtype=np.float32)
    return {"images": images, "pressure": pressure, "cloud": cloud, "components": SEVIRI_COMPONENTS, "center": 1,
            "lon": np.asarray(obs.lon_grid, dtype=np.float32), "lat": np.asarray(obs.lat_grid, dtype=np.float32),
            "dt_s": interval * 60.0}


def load_inputs_goes(case: str, sat: str) -> dict:
    from data_io.goes import discover_goes_frames, frame_for, load_goes_frame
    from runtime_options import read_case_inputs

    record = read_case_inputs(PROJECT_ROOT, case)
    frames = discover_goes_frames(record.get("data_dir", PROJECT_ROOT / "data" / case))
    roles = [frame_for(frames, sat, role) for role in ("previous", "center", "next")]
    try:
        t = [pd.Timestamp(r.scan_midpoint) for r in roles]
        dt_s = float(np.mean([abs((t[1] - t[0]).total_seconds()), abs((t[2] - t[1]).total_seconds())]))
    except Exception:  # noqa: BLE001
        dt_s = GOES_INTERVAL_S_FALLBACK
    print(f"GOES {sat} image interval: {dt_s:.0f} s")
    loaded = [load_goes_frame(r, apply_quality_mask=False) for r in roles]
    images = np.stack([np.asarray(f.brightness_temperature_k, np.float32) for f in loaded])
    pressure = np.stack([np.asarray(f.cloud_top_pressure_hpa, np.float32) for f in loaded])
    return {"images": images, "pressure": pressure, "cloud": None, "components": GOES_COMPONENTS, "center": 1,
            "lon": None, "lat": None, "dt_s": dt_s}


def grid_accessor(final: np.lib.npyio.NpzFile, shape: tuple[int, int]):
    """Return a function key -> 2-D grid array, whatever the stored layout."""

    h, w = shape
    rows = np.asarray(final["row_center_original"])
    if rows.ndim == 2 and rows.shape == shape:
        return lambda key: np.asarray(final[key])
    if rows.size == h * w:
        return lambda key: np.asarray(final[key]).reshape(h, w)
    r = np.rint(rows).astype(np.int64).ravel()
    c = np.rint(np.asarray(final["col_center_original"])).astype(np.int64).ravel()

    def to_grid(key: str) -> np.ndarray:
        values = np.asarray(final[key]).ravel()
        out = np.full(shape, np.nan if values.dtype.kind == "f" else 0, dtype=values.dtype)
        out[r, c] = values
        return out
    return to_grid


# =============================================================================
# Internal features (no external data)
# =============================================================================


def _nanfilter(values: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    ok = np.isfinite(values)
    num = gaussian_filter(np.where(ok, values, 0.0).astype(np.float32), sigma)
    den = gaussian_filter(ok.astype(np.float32), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (num / den).astype(np.float32)


def _local_std(values: np.ndarray, size: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter

    ok = np.isfinite(values)
    v = np.where(ok, values, 0.0).astype(np.float32)
    n = uniform_filter(ok.astype(np.float32), size)
    m = uniform_filter(v, size)
    m2 = uniform_filter(v * v, size)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = m / n
        var = np.maximum(m2 / n - mean * mean, 0.0)
    return np.sqrt(var).astype(np.float32)


def _box_mean(values: np.ndarray, size: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter

    ok = np.isfinite(values)
    num = uniform_filter(np.where(ok, values, 0.0).astype(np.float64), size, mode="constant")
    den = uniform_filter(ok.astype(np.float64), size, mode="constant")
    # A rolling sum can leave a tiny positive residual after its last valid
    # sample. Require at least half a sample to distinguish real support.
    supported = den * (int(size) ** values.ndim) >= 0.5
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(supported, num / den, np.nan)


def local_shear(u: np.ndarray, v: np.ndarray, p: np.ndarray, size: int) -> np.ndarray:
    """|dV/dp| (m/s per hPa): ridge regression of u, v on the assigned pressure in a window."""

    ok = np.isfinite(u) & np.isfinite(v) & np.isfinite(p)
    u, v, p = (np.where(ok, a, np.nan).astype(np.float64) for a in (u, v, p))
    mp, mu, mv = _box_mean(p, size), _box_mean(u, size), _box_mean(v, size)
    var_p = np.maximum(_box_mean(p * p, size) - mp ** 2, 0.0)
    denom = var_p + SHEAR_RIDGE_HPA ** 2
    return np.hypot((_box_mean(u * p, size) - mu * mp) / denom, (_box_mean(v * p, size) - mv * mp) / denom)


def image_texture_features(image: np.ndarray) -> dict[str, np.ndarray]:
    from scipy.ndimage import uniform_filter

    smooth = _nanfilter(image, GRAD_SIGMA_PX)
    gy, gx = np.gradient(smooth)
    win = TEXTURE_WINDOW_PX
    jxx = uniform_filter(np.nan_to_num(gx * gx), win)
    jyy = uniform_filter(np.nan_to_num(gy * gy), win)
    jxy = uniform_filter(np.nan_to_num(gx * gy), win)
    tr = jxx + jyy
    det_term = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4 * jxy * jxy, 0.0))
    lam_min = 0.5 * (tr - det_term)
    lam_max = 0.5 * (tr + det_term)
    with np.errstate(invalid="ignore", divide="ignore"):
        coherence = (lam_max - lam_min) / (lam_max + lam_min + 1e-6)
    return {
        "bt": image,
        "grad_mag": np.hypot(gx, gy).astype(np.float32),
        "struct_min_eig": np.log1p(np.maximum(lam_min, 0)).astype(np.float32),     # aperture problem
        "struct_coherence": coherence.astype(np.float32),
        "bt_local_std": _local_std(image, win),
    }


def warp_residual(img0: np.ndarray, img1: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """|I1(x + d) - I0(x)|: brightness-constancy residual of one component."""

    import cv2

    h, w = img0.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = (xx + np.nan_to_num(dx)).astype(np.float32)
    map_y = (yy + np.nan_to_num(dy)).astype(np.float32)
    warped = cv2.remap(np.nan_to_num(img1).astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    res = np.abs(warped - img0)
    res[~np.isfinite(dx) | ~np.isfinite(dy)] = np.nan
    return res.astype(np.float32)


def local_zenith_deg(lat: np.ndarray, lon: np.ndarray, sub_lon: float) -> np.ndarray:
    r_e, r_geo = 6378.137, 42164.0
    cos_g = np.cos(np.radians(lat)) * np.cos(np.radians(lon - sub_lon))
    d = np.sqrt(r_e**2 + r_geo**2 - 2 * r_e * r_geo * cos_g)
    return np.degrees(np.arccos(np.clip((r_geo * cos_g - r_e) / d, -1, 1))).astype(np.float32)


def build_point_table(inputs: dict, p: dict) -> tuple[pd.DataFrame, list[str]]:
    images = inputs["images"]
    shape = images.shape[1:]
    product = np.load(p["product"], allow_pickle=False)
    rows = np.rint(product["row_center_original"]).astype(np.int64)
    cols = np.rint(product["col_center_original"]).astype(np.int64)
    n = rows.size
    print(f"product points: {n:,}")

    final = np.load(p["final"], allow_pickle=False)
    grid = grid_accessor(final, shape)

    # --- components: target dispersion S and K -------------------------------------------
    components = inputs["components"]
    center = inputs["center"]
    comp_u, comp_v, comp_ok = [], [], []
    for c, _, _ in components:
        u = grid(f"u_{c}_ms")[rows, cols].astype(np.float64)
        v = grid(f"v_{c}_ms")[rows, cols].astype(np.float64)
        ok = grid(f"valid_{c}_mask")[rows, cols].astype(bool) & np.isfinite(u) & np.isfinite(v)
        comp_u.append(np.where(ok, u, 0.0)); comp_v.append(np.where(ok, v, 0.0)); comp_ok.append(ok)
    U, V, OK = np.stack(comp_u), np.stack(comp_v), np.stack(comp_ok)
    K = OK.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ub, vb = (U * OK).sum(0) / K, (V * OK).sum(0) / K
    S = (((U - ub) ** 2 + (V - vb) ** 2) * OK).sum(0)

    table = {"row": rows, "col": cols, "K": K, "S": S,
             "u": np.asarray(product["u_ms"], np.float64), "v": np.asarray(product["v_ms"], np.float64)}

    # --- features -----------------------------------------------------------------------
    feats: dict[str, np.ndarray] = {}
    tex = image_texture_features(images[center])
    for name, arr in tex.items():
        feats[name] = arr[rows, cols]
    feats["bt_temporal_change"] = np.nanmean(np.abs(np.diff(images, axis=0)), axis=0)[rows, cols]

    residuals, end_pressure, anchor_pressure = [], [], {}
    h, w = shape
    for c, i0, i1 in components:
        dx, dy = grid(f"dx_{c}_pixels").astype(np.float32), grid(f"dy_{c}_pixels").astype(np.float32)
        residuals.append(warp_residual(images[i0], images[i1], dx, dy)[rows, cols])
        if inputs["pressure"] is not None:            # pressure at the tracked end point
            rr = np.clip(np.rint(rows + np.nan_to_num(dy[rows, cols])), 0, h - 1).astype(np.int64)
            cc = np.clip(np.rint(cols + np.nan_to_num(dx[rows, cols])), 0, w - 1).astype(np.int64)
            end_pressure.append(inputs["pressure"][i1][rr, cc])
            anchor_pressure[i0] = inputs["pressure"][i0][rows, cols]
        del dx, dy
    feats["warp_residual_mean"] = np.nanmean(np.stack(residuals), axis=0)

    u_grid, v_grid = grid("u_ms").astype(np.float32), grid("v_ms").astype(np.float32)
    us, vs = _nanfilter(u_grid, WIND_GRAD_SIGMA_PX), _nanfilter(v_grid, WIND_GRAD_SIGMA_PX)
    uy, ux = np.gradient(us); vy, vx = np.gradient(vs)
    feats["wind_grad_mag"] = np.sqrt(ux**2 + uy**2 + vx**2 + vy**2)[rows, cols]
    feats["wind_local_std"] = np.hypot(_local_std(u_grid, 7), _local_std(v_grid, 7))[rows, cols]
    feats["speed"] = np.hypot(table["u"], table["v"])

    feats["pressure"] = np.asarray(product["pressure_hpa"], np.float32)
    pcs = [np.asarray(product[f"pressure_{c}_hpa"], np.float32) for c, _, _ in components
           if f"pressure_{c}_hpa" in product.files]
    if not pcs and end_pressure:                      # GOES: ACHP at the anchor and the tracked end points
        pcs = list(anchor_pressure.values()) + end_pressure
    if pcs:
        with np.errstate(all="ignore"):
            feats["pressure_track_range"] = np.nanmax(np.stack(pcs), 0) - np.nanmin(np.stack(pcs), 0)
    if inputs["pressure"] is not None:
        feats["pressure_local_std"] = _local_std(inputs["pressure"][center], PRESSURE_WINDOW_PX)[rows, cols]
    # Pressures averaged into the assigned pressure: SEVIRI = mean of the valid
    # component-start OCA pressures (postprocess.height_assignment, "mean_valid_components");
    # GOES = centre-frame ACHP at the anchor only (single value).
    sevir_mean = any(f"pressure_{c}_hpa" in product.files for c, _, _ in components)
    if pcs:                                           # height-assignment noise target
        P = np.stack(pcs).astype(np.float64)
        okp = np.isfinite(P)
        table["K_p"] = okp.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            pm = np.where(okp, P, 0.0).sum(0) / table["K_p"]
        table["S_p"] = np.where(okp, (P - pm) ** 2, 0.0).sum(0)
        table["S_p"][table["K_p"] < 2] = np.nan
        table["J_avg"] = np.maximum(table["K_p"], 1) if sevir_mean else np.ones(n, np.int64)
    pg, ug, vg = (np.full(shape, np.nan) for _ in range(3))
    pg[rows, cols] = np.asarray(product["pressure_hpa"], np.float64)
    ug[rows, cols], vg[rows, cols] = table["u"], table["v"]          # assigned product winds
    table["shear"] = local_shear(ug, vg, pg, window_px(SHEAR_WINDOW_KM, p["kind"]))[rows, cols]
    if "S_p" in table:
        sp2 = np.full(shape, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            sp2[rows, cols] = table["S_p"] / np.maximum(table["K_p"] - 1, 1)
        table["sigma_p2_local"] = _box_mean(sp2, window_px(SIGMA_P_WINDOW_KM, p["kind"]))[rows, cols]
        del sp2
    del pg, ug, vg
    for key in ("cloud_probability", "cloud_mask", "pressure_dqf", "height_dqf"):
        if key in product.files:
            feats[key] = np.asarray(product[key], np.float32)

    if inputs["lat"] is not None:
        lat, lon = inputs["lat"][rows, cols], inputs["lon"][rows, cols]
        feats["abs_lat"] = np.abs(lat)
        feats["zenith"] = local_zenith_deg(lat, lon, SUB_SATELLITE_LON)
    else:
        feats["abs_lat"] = np.abs(np.asarray(product["latitude"], np.float32))
        if "local_zenith_angle_deg" in product.files:
            feats["zenith"] = np.asarray(product["local_zenith_angle_deg"], np.float32)

    for name, arr in feats.items():
        table[f"f_{name}"] = np.asarray(arr, np.float32)
    frame = pd.DataFrame(table)
    feature_cols = [c for c in frame.columns if c.startswith("f_")]
    keep = (frame["K"] >= 2) & np.isfinite(frame["S"])
    frame = frame.loc[keep].reset_index(drop=True)
    # Missing features -> median (rare; e.g. image edges).
    for c in feature_cols:
        med = float(np.nanmedian(frame[c]))
        frame[c] = frame[c].fillna(med)
    print(f"points with >= 2 components: {len(frame):,}; features: {len(feature_cols)}")
    return frame, feature_cols


def assign_folds(frame: pd.DataFrame) -> np.ndarray:
    block = (frame["row"].to_numpy() // BLOCK_PX) * 100_000 + frame["col"].to_numpy() // BLOCK_PX
    uniq = np.unique(block)
    lab = np.random.default_rng(SEED).integers(0, 2, uniq.size)
    return lab[np.searchsorted(uniq, block)]


# =============================================================================
# Model
# =============================================================================


def chi2_nll(log_var, S, K):
    import torch

    return ((K - 1.0) * log_var + S / (2.0 * torch.exp(log_var))).mean()


def train_mlp(X, S, K, log_rows: list, fold: int):
    import torch
    from torch import nn

    device = ("cuda" if torch.cuda.is_available() else "cpu") if DEVICE == "auto" else DEVICE
    torch.manual_seed(SEED + fold)
    layers, d = [], X.shape[1]
    for h in HIDDEN:
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers += [nn.Linear(d, 1)]
    model = nn.Sequential(*layers).to(device)
    with torch.no_grad():                         # start at the global mean variance
        model[-1].bias.fill_(float(np.log(np.mean(S / (2.0 * (K - 1.0))) + 1e-6)))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    Xt, St, Kt = t(X), t(S), t(K)
    converged = False
    best, best_epoch, best_state = np.inf, 0, None
    since_best, since_lr_change = 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(Xt.shape[0], device=device)
        total = 0.0
        for i in range(0, Xt.shape[0], BATCH):
            idx = perm[i:i + BATCH]
            loss = chi2_nll(model(Xt[idx]).squeeze(1), St[idx], Kt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * idx.numel()
        nll_epoch = total / Xt.shape[0]
        lr_now = opt.param_groups[0]["lr"]
        event = ""
        if epoch <= WARMUP_EPOCHS:
            event = "warm-up"
        elif nll_epoch < best - MIN_DELTA:
            best, best_epoch, since_best, since_lr_change = nll_epoch, epoch, 0, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            event = "best (saved)"
        else:
            since_best += 1
            since_lr_change += 1
            if since_best >= PATIENCE_STOP:
                converged, event = True, "converged"
            elif since_lr_change >= PATIENCE_LR and lr_now > LR_MIN * 1.0001:
                new_lr = max(lr_now * 0.5, LR_MIN)
                for g in opt.param_groups:
                    g["lr"] = new_lr
                since_lr_change, event = 0, f"lr -> {new_lr:.2e}"
        log_rows.append({"fold": fold, "epoch": epoch, "train_nll": nll_epoch, "best_nll": best,
                         "lr": lr_now, "event": event, "converged": converged})
        print(f"  fold {fold} epoch {epoch:3d}  train NLL {nll_epoch:.5f}  best {best:.5f} (ep {best_epoch})  "
              f"lr {lr_now:.2e}  {event}")
        if converged:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    if best_state is None:                        # MAX_EPOCHS <= WARMUP_EPOCHS: keep the last state
        best, best_epoch = nll_epoch, epoch
    print(f"  fold {fold}: {'converged' if converged else 'NOT converged (MAX_EPOCHS reached)'} after {epoch} epochs; "
          f"using epoch {best_epoch} (NLL {best:.5f})")
    model.eval()
    return model, device


def predict_log_var(model, device, X: np.ndarray) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 262_144):
            out.append(model(torch.as_tensor(X[i:i + 262_144], dtype=torch.float32, device=device)).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def fit_prior_strength(var_nn: np.ndarray, S: np.ndarray, K: np.ndarray) -> float:
    """Inverse-gamma prior strength alpha by marginal likelihood (compound gamma)."""

    from scipy.special import gammaln

    k = K - 1.0
    S = np.maximum(S, 1e-8)
    best_a, best_ll = None, -np.inf
    for a in np.exp(np.linspace(np.log(1.05), np.log(200.0), 120)):
        beta2 = 2.0 * (a - 1.0) * var_nn                    # 2*beta, prior mean = var_nn
        ll = ((k - 1) * np.log(S) - gammaln(k) + a * np.log(beta2) - gammaln(a) + gammaln(a + k)
              - (a + k) * np.log(beta2 + S)).mean()
        if ll > best_ll:
            best_a, best_ll = a, ll
    return float(best_a)


def permutation_importance(model, device, X, S, K, names) -> pd.DataFrame:
    import torch

    def nll(Xm):
        lv = torch.as_tensor(predict_log_var(model, device, Xm))
        return float(chi2_nll(lv, torch.as_tensor(S, dtype=torch.float32), torch.as_tensor(K, dtype=torch.float32)))

    base = nll(X)
    rng = np.random.default_rng(SEED)
    rows = []
    for j, name in enumerate(names):
        Xp = X.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        rows.append({"feature": name.removeprefix("f_"), "nll_increase": nll(Xp) - base})
    return pd.DataFrame(rows).sort_values("nll_increase", ascending=False)


# =============================================================================
# Validation against ERA5 (test blocks only; not used for training)
# =============================================================================


def _retention(score: np.ndarray, vd: np.ndarray, name: str, *, higher_is_better: bool = False) -> pd.DataFrame:
    order = np.argsort(-score if higher_is_better else score, kind="stable")
    vd_sorted = vd[order]
    csum, csq = np.cumsum(vd_sorted), np.cumsum(vd_sorted ** 2)
    rows = []
    for f in RETENTION:
        n = max(1, int(round(f * vd.size)))
        rows.append({"score": name, "retention": f, "n_retained": n,
                     "mvd": csum[n - 1] / n, "rmsvd": float(np.sqrt(csq[n - 1] / n))})
    return pd.DataFrame(rows)


def era5_table(p: dict, tracker: str) -> pd.DataFrame:
    if p["kind"] == "goes":
        from validation.goes_existing_era5_cache import build_amv_vs_existing_goes_era5_cache, resolve_existing_pcf_cache

        cache = resolve_existing_pcf_cache(PROJECT_ROOT, p["case"], p["sat"], GOES_GEOMETRY)
        table, _ = build_amv_vs_existing_goes_era5_cache(p["product"], cache, amv_source=tracker)
        return table
    from validation.compare_era5 import build_pressure_amv_vs_era5_table

    return build_pressure_amv_vs_era5_table(p["product"], p["era5_dense"], join_mode="row_col_pressure",
                                            amv_source=tracker)


QI_CURVE_STEP = 1.0
QI_HIST_BIN = 1.0


def qi_curve_tables(qi: np.ndarray, vd: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MVD of the vectors with QI >= q (cumulative) and the QI histogram (density)."""

    qi, vd = np.asarray(qi, float), np.asarray(vd, float)
    ok = np.isfinite(qi) & np.isfinite(vd)
    qi, vd = qi[ok], vd[ok]
    rows = []
    for q in np.arange(0.0, 100.0, QI_CURVE_STEP):
        keep = qi >= q
        rows.append({"qi_min": q, "n": int(keep.sum()), "fraction": float(keep.mean()),
                     "mvd": float(vd[keep].mean()) if keep.any() else np.nan,
                     "rmsvd": float(np.sqrt(np.mean(vd[keep] ** 2))) if keep.any() else np.nan})
    edges = np.arange(0.0, 100.0 + QI_HIST_BIN, QI_HIST_BIN)
    count, _ = np.histogram(np.clip(qi, 0, 100), bins=edges)
    hist = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": count,
                         "density": count / max(count.sum(), 1) / QI_HIST_BIN})
    return pd.DataFrame(rows), hist


def pointwise_qi(var_post: np.ndarray, K: np.ndarray, alpha: float, tau, extra_var=0.0) -> np.ndarray:
    """100 * P(|eps| <= tau) under the inverse-gamma posterior (closed form).

    extra_var: additional vector-error variance of the mean wind (height term); it is
    added to the posterior scale (moment-matched inverse gamma, same shape).
    """

    K = np.asarray(K, dtype=np.float64)
    a = alpha + K - 1.0
    var_eff = np.asarray(var_post, dtype=np.float64) + K * np.asarray(extra_var, dtype=np.float64) / 2.0
    b = var_eff * (a - 1.0)
    return 100.0 * (1.0 - (1.0 + K * tau ** 2 / (2.0 * b)) ** (-a))


def era5_validation(frame: pd.DataFrame, scores: dict[str, np.ndarray], p: dict, tracker: str,
                    official_curve: pd.DataFrame | None, qi: np.ndarray | None = None,
                    speed: np.ndarray | None = None) -> None:
    table = era5_table(p, tracker)
    if "era5_valid_mask" in table.columns:
        table = table.loc[table["era5_valid_mask"].astype(bool)]
    era = pd.DataFrame({
        "row": np.rint(table["row_center_original"]).astype(np.int64),
        "col": np.rint(table["col_center_original"]).astype(np.int64),
        "vd": np.hypot(table["u_amv"] - table["u_era5"], table["v_amv"] - table["v_era5"]),
    })
    m = frame[["row", "col"]].copy()
    for name, sc in scores.items():
        m[name] = sc
    if qi is not None:
        m["_qi"] = qi
    if speed is not None:
        m["_speed"] = speed
    m = m.merge(era, on=["row", "col"], how="inner")
    m = m.loc[np.isfinite(m["vd"])]
    vd = m["vd"].to_numpy()
    print(f"ERA5-collocated points: {len(m):,}")
    if qi is not None:
        rows = [{"subset": "all", "n": len(m), "fraction": 1.0, "mvd": float(vd.mean()),
                 "rmsvd": float(np.sqrt(np.mean(vd ** 2)))}]
        for thr in (60.0, 80.0):
            keep = m["_qi"].to_numpy() >= thr
            if keep.any():
                rows.append({"subset": f"qi>={thr:.0f}", "n": int(keep.sum()), "fraction": float(keep.mean()),
                             "mvd": float(vd[keep].mean()), "rmsvd": float(np.sqrt(np.mean(vd[keep] ** 2)))})
        thr_table = pd.DataFrame(rows)
        thr_table.to_csv(p["result"] / "validation_era5_qi_thresholds.csv", index=False)
        curve, hist = qi_curve_tables(m["_qi"].to_numpy(), vd)
        curve.to_csv(p["result"] / "validation_era5_qi_curve.csv", index=False)
        hist.to_csv(p["result"] / "validation_era5_qi_hist.csv", index=False)
        print("\nPoint-wise QI thresholds vs ERA5:")
        print(thr_table.round(3).to_string(index=False))
        if speed is not None:
            b = pd.cut(m["_speed"], SPEED_BINS, right=False)
            rows = []
            for interval, g in m.groupby(b, observed=True):
                keep = g["_qi"] >= 60.0
                row = {"speed_bin": str(interval), "n": len(g), "frac_qi60": float(keep.mean()),
                       "mvd_all": float(g["vd"].mean()),
                       "mvd_qi60": float(g.loc[keep, "vd"].mean()) if keep.any() else np.nan,
                       "mvd_rejected": float(g.loc[~keep, "vd"].mean()) if (~keep).any() else np.nan}
                for name in [k for k in scores if k.startswith("ee_")]:      # within-bin ranking quality
                    row[f"rank_{name}"] = float(_retention(g[name].to_numpy(), g["vd"].to_numpy(), name)["mvd"].mean())
                rows.append(row)
            sb = pd.DataFrame(rows)
            sb.to_csv(p["result"] / "validation_era5_speed_bins.csv", index=False)
            print("\nQI >= 60 by wind speed (m/s):")
            print(sb.round(3).to_string(index=False))

    curves = [_retention(m[name].to_numpy(), vd, name) for name in scores]
    curves.append(_retention(np.random.default_rng(SEED).random(len(m)), vd, "random"))
    if qi is not None:
        curves.append(_retention(m["_qi"].to_numpy(), vd, "qi", higher_is_better=True))
    ret = pd.concat(curves, ignore_index=True)
    if official_curve is not None:
        ret = pd.concat([ret, official_curve], ignore_index=True)
    summary = ret.groupby("score")["mvd"].mean().rename("mean_mvd_over_retention").sort_values()
    print("\nRetention summary (mean MVD over retained fractions; lower = better):")
    print(summary.round(3).to_string())
    print("\nMVD at selected retained fractions:")
    print(ret.loc[ret["retention"].isin([0.2, 0.4, 0.6, 0.8, 1.0])]
          .pivot(index="retention", columns="score", values="mvd").round(3).to_string())
    ret.to_csv(p["result"] / "validation_era5_retention.csv", index=False)
    summary.to_csv(p["result"] / "validation_era5_retention_summary.csv")

    rel = []
    for name in [s for s in scores if s.startswith("ee_")]:
        q = pd.qcut(m[name], 10, labels=False, duplicates="drop")
        g = m.groupby(q)
        rel.append(pd.DataFrame({"score": name, "predicted_ee": g[name].mean(),
                                 "observed_rmsvd": g["vd"].apply(lambda x: float(np.sqrt(np.mean(x ** 2)))),
                                 "observed_mvd": g["vd"].mean(), "n": g.size()}))
    rel = pd.concat(rel, ignore_index=True)
    rel.to_csv(p["result"] / "validation_era5_reliability.csv", index=False)



def official_retention_curve(case: str, p: dict) -> pd.DataFrame | None:
    """Operational AMVs ranked by their own QI (reference curve; SEVIRI only, DMW has no QI)."""

    if p["kind"] == "goes":
        return None
    try:
        from data_io.official_amv import load_official_amv_file
        from data_io.product_discovery import find_official_amv_file, infer_amv_valid_time_from_path
        from validation.compare_era5 import build_official_amv_vs_era5_table

        from runtime_options import read_case_inputs
        inputs = read_case_inputs(PROJECT_ROOT, case)
        data_dir = Path(inputs.get("data_dir", PROJECT_ROOT / "data" / case.removeprefix("test_")))
        valid_time = (pd.Timestamp(inputs["frame_times"][-1]).to_pydatetime() if inputs.get("frame_times")
                      else infer_amv_valid_time_from_path(data_dir, offset_minutes=45))
        path = find_official_amv_file(data_dir / "eumetsat_amv" / "native",
                                      expected_time=valid_time,
                                      tolerance_minutes=90.0, verbose=False)
        t = build_official_amv_vs_era5_table(load_official_amv_file(path), p["era5_official"])
        if "era5_valid_mask" in t.columns:
            t = t.loc[t["era5_valid_mask"].astype(bool)]
        t = t.loc[np.isfinite(t["official_quality"])]
        vd = np.hypot(t["u_amv"] - t["u_era5"], t["v_amv"] - t["v_era5"]).to_numpy()
        ok = np.isfinite(vd)
        q = t["official_quality"].to_numpy(dtype=float)[ok]
        if np.nanmedian(q) <= 1.5:
            q = q * 100.0
        vd = vd[ok]
        print(f"operational AMVs with QI and ERA5: {ok.sum():,}")
        rows = [{"subset": "all", "n": int(vd.size), "fraction": 1.0, "mvd": float(vd.mean()),
                 "rmsvd": float(np.sqrt(np.mean(vd ** 2)))}]
        for thr in (60.0, 80.0):
            keep = q >= thr
            rows.append({"subset": f"qi>={thr:.0f}", "n": int(keep.sum()), "fraction": float(keep.mean()),
                         "mvd": float(vd[keep].mean()), "rmsvd": float(np.sqrt(np.mean(vd[keep] ** 2)))})
        out_dir = p["out"] / "qc_selfsup"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / "operational_qi_thresholds.csv", index=False)
        curve, hist = qi_curve_tables(q, vd)
        curve.to_csv(out_dir / "operational_qi_curve.csv", index=False)
        hist.to_csv(out_dir / "operational_qi_hist.csv", index=False)
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        spd = np.hypot(t["u_amv"], t["v_amv"]).to_numpy()[ok]
        b = pd.cut(spd, SPEED_BINS, right=False)
        sb = []
        for interval in b.categories:
            g = np.asarray(b == interval)
            if g.any():
                keep = g & (q >= 60.0)
                sb.append({"speed_bin": str(interval), "n": int(g.sum()), "frac_qi60": float(keep.sum() / g.sum()),
                           "mvd_all": float(vd[g].mean()),
                           "mvd_qi60": float(vd[keep].mean()) if keep.any() else np.nan,
                           "mvd_rejected": float(vd[g & ~keep].mean()) if (g & ~keep).any() else np.nan})
        pd.DataFrame(sb).to_csv(out_dir / "operational_speed_bins.csv", index=False)
        print("operational QI >= 60 by wind speed (m/s):")
        print(pd.DataFrame(sb).round(3).to_string(index=False))
        return _retention(q, vd, "operational_qi", higher_is_better=True)
    except Exception as exc:  # noqa: BLE001
        print("operational reference curve skipped:", type(exc).__name__, exc)
        return None


# =============================================================================
# Main
# =============================================================================


def run_tracker(case: str, tracker: str, inputs: dict, official_curve: pd.DataFrame | None,
                sat: str | None = None) -> None:
    p = paths(case, tracker, sat)
    p["result"].mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print("\n" + "=" * 90 + f"\n{case} | {tracker}" + (f" | {sat}" if sat else ""))
    frame, feature_cols = build_point_table(inputs, p)
    frame["fold"] = assign_folds(frame)

    X_all = frame[feature_cols].to_numpy(np.float32)
    X_all = (X_all - X_all.mean(0)) / (X_all.std(0) + 1e-6)      # unsupervised standardisation
    S_all, K_all = frame["S"].to_numpy(np.float32), frame["K"].to_numpy(np.float32)
    fold = frame["fold"].to_numpy()

    # Two-fold cross-fitting: each half is predicted by the model trained on the other half.
    rng = np.random.default_rng(SEED)
    log_rows: list = []

    def crossfit(X, S, Keff, valid, tag):
        log_var = np.full(len(frame), np.nan, dtype=np.float64)
        models = []
        for f in (0, 1):
            train_idx = np.flatnonzero((fold == f) & valid)
            train_idx = rng.choice(train_idx, min(train_idx.size, MAX_TRAIN_POINTS), replace=False)
            start = len(log_rows)
            model, device = train_mlp(X[train_idx], S[train_idx], Keff[train_idx], log_rows, f)
            for row in log_rows[start:]:
                row["model"] = tag
            held_out = np.flatnonzero(fold != f)
            log_var[held_out] = predict_log_var(model, device, X[held_out])
            models.append((model, device))
        return log_var, models

    print("--- tracking-error model (wind-component dispersion)")
    log_var, models = crossfit(X_all, S_all, K_all, np.ones(len(frame), bool), "wind")
    print(f"[{time.perf_counter() - t0:6.1f} s] cross-fitted on {models[0][1]}")

    var_nn = np.exp(log_var)
    sub = rng.choice(len(frame), min(len(frame), 1_000_000), replace=False)
    alpha = fit_prior_strength(var_nn[sub], S_all[sub].astype(np.float64), K_all[sub].astype(np.float64))
    var_post = ((alpha - 1.0) * var_nn + S_all / 2.0) / (alpha + K_all - 2.0)
    scale = np.sqrt(2.0 / K_all)
    spread = np.sqrt(S_all / (2.0 * (K_all - 1.0)))
    scores = {"ee_post": scale * np.sqrt(var_post), "ee_nn": scale * np.sqrt(var_nn), "ee_spread": scale * spread}

    # Height-assignment term: sigma_p^2 from the component-pressure dispersion (scalar,
    # chi-square with K_p - 1 dof -> effective K = (K_p + 1) / 2 in the shared code).
    height_var = np.zeros(len(frame))
    alpha_p = np.nan
    var_p_post = np.full(len(frame), np.nan)
    if HEIGHT_TERM and "S_p" in frame.columns:
        Kp = frame["K_p"].to_numpy(np.float64)
        Sp = frame["S_p"].to_numpy(np.float64)
        if HEIGHT_SIGMA_MODE == "local":
            print(f"--- height noise: local mean of S_p/(K_p-1) over {window_px(SIGMA_P_WINDOW_KM, p['kind'])} px")
            var_p_post = np.maximum(frame["sigma_p2_local"].to_numpy(np.float64), 0.0)
            var_p_post = np.where(np.isfinite(var_p_post), var_p_post, np.nanmedian(var_p_post))
        else:
            print("--- height-noise model (network + empirical Bayes)")
            valid_p = np.isfinite(Sp) & (Kp >= 2)
            Keff_p = ((np.maximum(Kp, 2) + 1.0) / 2.0).astype(np.float32)
            cols_p = [i for i, c in enumerate(feature_cols) if c != "f_pressure_track_range"]
            Xp = np.ascontiguousarray(X_all[:, cols_p])
            log_var_p, _ = crossfit(Xp, np.nan_to_num(Sp).astype(np.float32), Keff_p, valid_p, "pressure")
            var_p_nn = np.exp(log_var_p)
            sub_p = rng.choice(np.flatnonzero(valid_p), min(int(valid_p.sum()), 1_000_000), replace=False)
            alpha_p = fit_prior_strength(var_p_nn[sub_p], Sp[sub_p], Keff_p[sub_p].astype(np.float64))
            var_p_post = np.where(valid_p, ((alpha_p - 1.0) * var_p_nn + np.nan_to_num(Sp) / 2.0)
                                  / (alpha_p + Keff_p - 2.0), var_p_nn)
        shear = np.nan_to_num(frame["shear"].to_numpy(np.float64))
        J_avg = frame["J_avg"].to_numpy(np.float64) if "J_avg" in frame.columns else np.ones(len(frame))
        height_var = shear ** 2 * var_p_post / J_avg
        # Ablation: same pressure noise, spatially constant shear (median) -> is the local shear informative?
        height_var_const = np.median(shear[shear > 0]) ** 2 * var_p_post / J_avg if (shear > 0).any() else height_var
        print(f"height term: alpha_p = {alpha_p:.2f}; median sigma_p = {np.median(np.sqrt(var_p_post)):.1f} hPa; "
              f"median height error = {np.median(np.sqrt(height_var)):.2f} m/s")
    scores["ee_total"] = np.sqrt(scores["ee_post"] ** 2 + height_var)
    if HEIGHT_TERM and "S_p" in frame.columns:
        scores["ee_total_constshear"] = np.sqrt(scores["ee_post"] ** 2 + height_var_const)
    pd.DataFrame(log_rows).to_csv(p["result"] / "training_log.csv", index=False)
    rank = scores["ee_post"].argsort(kind="stable").argsort(kind="stable")
    qi_percentile = (100.0 * (1.0 - rank / max(len(rank) - 1, 1))).astype(np.float32)
    tau = QI_DELTA_PX * SSP_PIXEL_KM[p["kind"]] * 1000.0 / float(inputs["dt_s"])
    speed_all = frame["f_speed"].to_numpy(np.float64)
    tau_i = np.maximum(tau, QI_REL_TOL * speed_all)
    quality_index = pointwise_qi(var_post, K_all, alpha, tau_i, height_var).astype(np.float32)
    print(f"prior strength alpha = {alpha:.2f}; tau = {QI_DELTA_PX} px -> {tau:.3f} m/s")
    for thr in (60.0, 80.0):
        print(f"  QI >= {thr:.0f}: {np.mean(quality_index >= thr):.1%} of {len(quality_index):,} vectors")

    held = np.flatnonzero(fold == 1)[:200_000]            # predicted by the fold-0 model
    imp = permutation_importance(models[0][0], models[0][1], X_all[held], S_all[held], K_all[held], feature_cols)
    imp.to_csv(p["result"] / "features_importance.csv", index=False)
    print("Permutation importance (held-out NLL increase):")
    print(imp.round(4).to_string(index=False))

    np.savez(p["result"] / f"{p['name']}_selfsup_qc.npz",
             row_center_original=frame["row"].to_numpy(np.int32), col_center_original=frame["col"].to_numpy(np.int32),
             quality_index=quality_index, qi_percentile=qi_percentile, speed_ms=speed_all.astype(np.float32),
             ee_post=scores["ee_post"].astype(np.float32),
             ee_nn=scores["ee_nn"].astype(np.float32), ee_spread=scores["ee_spread"].astype(np.float32),
             ee_total=scores["ee_total"].astype(np.float32), height_var=height_var.astype(np.float32),
             ee_total_constshear=scores.get("ee_total_constshear", scores["ee_total"]).astype(np.float32),
             S_p=frame["S_p"].to_numpy(np.float32) if "S_p" in frame else np.full(len(frame), np.nan, np.float32),
             K_p=frame["K_p"].to_numpy(np.int8) if "K_p" in frame else np.zeros(len(frame), np.int8),
             J_avg=frame["J_avg"].to_numpy(np.int8) if "J_avg" in frame else np.ones(len(frame), np.int8),
             sigma_p_post=np.sqrt(var_p_post).astype(np.float32),
             shear=np.nan_to_num(frame["shear"].to_numpy(np.float32)) if "shear" in frame else np.zeros(len(frame), np.float32),
             valid_component_count=frame["K"].to_numpy(np.int8), fold=fold.astype(np.int8),
             metadata_json=np.asarray([json.dumps({"alpha": alpha, "alpha_p": alpha_p, "height_term": HEIGHT_TERM,
                                                   "height_sigma_mode": HEIGHT_SIGMA_MODE,
                                                   "local_pressure_variance_version": 2,
                                                   "sigma_p_window_km": SIGMA_P_WINDOW_KM, "shear_window_km": SHEAR_WINDOW_KM,
                                                   "shear_ridge_hpa": SHEAR_RIDGE_HPA,
                                                   "tau_ms": tau, "delta_px": QI_DELTA_PX, "rel_tol": QI_REL_TOL,
                                                   "dt_s": float(inputs["dt_s"]), "features": feature_cols, "hidden": HIDDEN,
                                                   "max_epochs": MAX_EPOCHS,
                                                   "epochs_run": int(pd.DataFrame(log_rows)["epoch"].max()), "case": case, "tracker": tracker})]))
    print("saved", p["result"] / f"{p['name']}_selfsup_qc.npz")

    try:
        era5_validation(frame, scores, p, tracker, official_curve, qi=quality_index, speed=speed_all)
    except FileNotFoundError as exc:
        print("ERA5 validation skipped:", exc)
    print(f"[{time.perf_counter() - t0:6.1f} s] {tracker} done")


def interval_seconds(case: str, sat: str | None) -> float:
    from runtime_options import read_case_inputs
    record = read_case_inputs(PROJECT_ROOT, case)
    if not is_goes(case):
        return float(record.get("interval_minutes") or 15.0) * 60.0
    try:
        from data_io.goes import discover_goes_frames, frame_for

        frames = discover_goes_frames(record.get("data_dir", PROJECT_ROOT / "data" / case))
        t = [pd.Timestamp(frame_for(frames, sat, r).scan_midpoint) for r in ("previous", "center", "next")]
        return float(np.mean([abs((t[1] - t[0]).total_seconds()), abs((t[2] - t[1]).total_seconds())]))
    except Exception:  # noqa: BLE001
        return GOES_INTERVAL_S_FALLBACK


def repair_local_pressure_variance(data: dict, shape: tuple[int, int], kind: str, meta: dict) -> None:
    """Repair legacy local-variance boundary artifacts without refitting the MLP.

    The pressure-noise and shear terms are not MLP input features. Keep valid
    saved statistics unchanged; repair unsupported windows and negative noise.
    """
    rows = data["row_center_original"].astype(np.int64)
    cols = data["col_center_original"].astype(np.int64)
    grid = np.full(shape, np.nan)
    grid[rows, cols] = data["S_p"].astype(np.float64) / np.maximum(data["K_p"].astype(np.float64) - 1, 1)
    size = window_px(float(meta.get("sigma_p_window_km", SIGMA_P_WINDOW_KM)), kind)
    local = _box_mean(grid, size)[rows, cols]
    unsupported = ~np.isfinite(local)
    negative = local < 0
    local = np.maximum(local, 0.0)
    fallback = np.nanmedian(local)
    if not np.isfinite(fallback):
        raise ValueError("No valid local pressure-dispersion samples are available for the height term")
    local[unsupported] = fallback
    repair = (unsupported | negative | ~np.isfinite(data["sigma_p_post"])
              | ~np.isfinite(data["height_var"]) | (data["height_var"] < 0))
    shear = data["shear"].astype(np.float64)
    divisor = np.maximum(data["J_avg"].astype(np.float64), 1)
    height = shear ** 2 * local / divisor
    data["height_var"][repair] = height[repair]
    data["sigma_p_post"][repair] = np.sqrt(local[repair])
    data["ee_total"][repair] = np.sqrt(data["ee_post"][repair].astype(np.float64) ** 2 + height[repair])
    if "ee_total_constshear" in data:
        constant = np.median(shear[shear > 0]) if (shear > 0).any() else 0.0
        data["ee_total_constshear"][repair] = np.sqrt(
            data["ee_post"][repair].astype(np.float64) ** 2 + constant ** 2 * local[repair] / divisor[repair])
    meta.update(local_pressure_variance_version=2, local_pressure_variance_repaired_points=int(repair.sum()),
                local_pressure_variance_no_support_points=int(unsupported.sum()))
    print(f"Repaired local pressure variance: {int(repair.sum()):,} points; "
          f"{int(unsupported.sum()):,} windows use the case-median variance")


def rescore(case: str, tracker: str, sat: str | None, official_curve: pd.DataFrame | None = None) -> None:
    """Rewrite quality_index of an existing *_selfsup_qc.npz with the point-wise QI (no retraining)."""

    p = paths(case, tracker, sat)
    name = f"{tracker}_{sat.lower()}" if sat else tracker
    path = p["result"] / f"{name}_selfsup_qc.npz"
    legacy = p["result"] / f"{tracker}_selfsup_qc.npz"          # GOES files of the first version
    if not path.is_file() and legacy.is_file():
        path = legacy
    if not path.is_file():
        print("missing", path)
        return
    with np.load(path, allow_pickle=False) as d:
        data = {k: np.asarray(d[k]) for k in d.files}
    meta = json.loads(str(data["metadata_json"].ravel()[0]))
    if (meta.get("height_term") and meta.get("height_sigma_mode") == "local"
            and meta.get("local_pressure_variance_version", 1) < 2):
        with np.load(p["final"], allow_pickle=False) as final:
            shape = final["valid_mask"].shape
        if len(shape) != 2:
            raise ValueError("Local pressure-variance repair requires the dense native grid")
        repair_local_pressure_variance(data, shape, p["kind"], meta)
    K = data["valid_component_count"].astype(np.float64)
    var_post = data["ee_post"].astype(np.float64) ** 2 * K / 2.0
    dt_s = interval_seconds(case, sat)
    tau = QI_DELTA_PX * SSP_PIXEL_KM[p["kind"]] * 1000.0 / dt_s
    if "qi_percentile" not in data:
        data["qi_percentile"] = data["quality_index"]
    if "speed_ms" not in data:                       # older files: speed from the product
        with np.load(p["product"], allow_pickle=False) as prod:
            ps = pd.DataFrame({"row": np.rint(prod["row_center_original"]).astype(np.int64),
                               "col": np.rint(prod["col_center_original"]).astype(np.int64),
                               "speed": np.hypot(prod["u_ms"], prod["v_ms"])}).drop_duplicates(["row", "col"])
        key = pd.DataFrame({"row": data["row_center_original"].astype(np.int64),
                            "col": data["col_center_original"].astype(np.int64)})
        data["speed_ms"] = key.merge(ps, on=["row", "col"], how="left")["speed"].to_numpy(np.float32)
    speed = data["speed_ms"].astype(np.float64)
    tau_i = np.maximum(tau, QI_REL_TOL * np.nan_to_num(speed, nan=0.0))
    extra = data["height_var"].astype(np.float64) if "height_var" in data else 0.0
    data["quality_index"] = pointwise_qi(var_post, K, float(meta["alpha"]), tau_i, extra).astype(np.float32)
    meta.update({"tau_ms": tau, "delta_px": QI_DELTA_PX, "rel_tol": QI_REL_TOL, "dt_s": dt_s})
    data["metadata_json"] = np.asarray([json.dumps(meta)])
    np.savez(path, **data)
    q = data["quality_index"]
    print(f"{name}: tau={tau:.3f} m/s, rel={QI_REL_TOL}, QI>=60 {np.mean(q >= 60):.1%}, "
          f"QI>=80 {np.mean(q >= 80):.1%} -> {path}")
    frame = pd.DataFrame({"row": data["row_center_original"].astype(np.int64),
                          "col": data["col_center_original"].astype(np.int64)})
    scores = {k: data[k].astype(np.float64) for k in ("ee_post", "ee_nn", "ee_spread", "ee_total", "ee_total_constshear")
              if k in data}
    try:
        era5_validation(frame, scores, p, tracker, official_curve, qi=q, speed=speed)
    except FileNotFoundError as exc:
        print("ERA5 validation skipped:", exc)


def main(argv: list[str]) -> int:
    args = list(argv[1:])
    sat = None
    if "--sat" in args:
        k = args.index("--sat")
        sat = args[k + 1].upper()
        del args[k:k + 2]
    only_rescore = "--rescore" in args
    if only_rescore:
        args.remove("--rescore")
    if "--rel" in args:
        global QI_REL_TOL
        k = args.index("--rel")
        QI_REL_TOL = float(args[k + 1])
        del args[k:k + 2]
    print("QI tolerance: delta =", QI_DELTA_PX, "px; relative =", QI_REL_TOL)
    case = args[0] if args else CASE_NAME
    trackers = args[1:] if len(args) > 1 else TRACKERS
    if is_goes(case) and sat is None:
        sat = "G16"
    if only_rescore:
        official_curve = official_retention_curve(case, paths(case, trackers[0], sat))
        for tracker in trackers:
            rescore(case, tracker, sat, official_curve)
        return 0
    inputs = load_inputs(case, sat)
    official_curve = official_retention_curve(case, paths(case, trackers[0], sat))
    for tracker in trackers:
        p = paths(case, tracker, sat)
        if not p["final"].is_file() or not p["product"].is_file():
            print("missing final native or product, skip:", p["final"], p["product"])
            continue
        run_tracker(case, tracker, inputs, official_curve, sat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
