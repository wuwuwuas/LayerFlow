"""Attach the operational QI (NOAA BUFR, QI without forecast) to the GOES DMW-vs-ERA5 table.

The public ABI L2 DMW netCDF has only a DQF; the QI is in the BUFR stream (NCEP GDAS
satwnd dump, NOAA NNJA archive).  The decoded BUFR records are matched one-to-one to
the netCDF DMW winds that already carry ERA5 collocations, by position, pressure and
wind (tolerances below), and the operational QI curve / histogram / thresholds are
written in the same format as for SEVIRI.

    python scripts/match_operational_quality_goes.py [--sat G16]

Inputs
    data/<case>/official_amv_qi/goes16_goes18_c14_*_qi.csv      (decoded BUFR)
    outputs/<case>/official_dmw_vs_era5/official_dmw_vs_era5_collocated.csv.gz
Outputs (outputs/<case>/qc_selfsup/)
    operational_<sat>_bufr_matched.csv, operational_<sat>_qi_curve.csv,
    operational_<sat>_qi_hist.csv, operational_<sat>_qi_thresholds.csv,
    operational_<sat>_speed_bins.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_quality_control as qc  # noqa: E402

PROJECT_ROOT = qc.PROJECT_ROOT
CASE = "goes_g16_g18_E20240804T120506_W20240804T120508"
MATCH_KM = 3.0                  # position tolerance
MATCH_HPA = 2.0                 # pressure tolerance
MATCH_MS = 0.5                  # wind-vector tolerance (both are rounded in BUFR)


def column(frame: pd.DataFrame, *names: str) -> str:
    for n in names:
        if n in frame.columns:
            return n
    raise KeyError(f"none of {names} in {list(frame.columns)}")


def load_bufr(sat: str) -> pd.DataFrame:
    files = sorted((PROJECT_ROOT / "data" / CASE / "official_amv_qi").glob("*_qi.csv"))
    if not files:
        raise FileNotFoundError(PROJECT_ROOT / "data" / CASE / "official_amv_qi")
    b = pd.read_csv(files[0])
    b = b.loc[b["satellite"].astype(str).str.upper() == sat].copy()
    rad = np.radians(b["wind_direction_deg"].to_numpy(float))       # meteorological "from" direction
    spd = b["wind_speed_ms"].to_numpy(float)
    b["u"], b["v"] = -spd * np.sin(rad), -spd * np.cos(rad)
    b["p"] = b["pressure_pa"].to_numpy(float) / 100.0
    b["qi"] = b["qi_without_forecast"].astype(float)
    print(f"BUFR {sat}: {len(b):,} winds ({files[0].name})")
    return b.reset_index(drop=True)


def load_dmw_era5(sat: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs" / CASE / "official_dmw_vs_era5" / "official_dmw_vs_era5_collocated.csv.gz"
    f = pd.read_csv(path)
    if "satellite" in f.columns:
        f = f.loc[f["satellite"].astype(str).str.upper() == sat]
    out = pd.DataFrame({
        "lat": f[column(f, "latitude", "lat")].to_numpy(float),
        "lon": f[column(f, "longitude", "lon")].to_numpy(float),
        "p": f[column(f, "pressure_hpa", "pressure")].to_numpy(float),
        "u": f[column(f, "u_amv", "u_ms", "u")].to_numpy(float),
        "v": f[column(f, "v_amv", "v_ms", "v")].to_numpy(float),
        "u_era5": f[column(f, "u_era5", "era5_u_ms", "u_era5_ms")].to_numpy(float),
        "v_era5": f[column(f, "v_era5", "era5_v_ms", "v_era5_ms")].to_numpy(float),
    })
    for flag in ("era5_valid_mask", "era5_collocated"):
        if flag in f.columns:
            out["era5_ok"] = f[flag].astype(bool).to_numpy()
            break
    else:
        out["era5_ok"] = np.isfinite(out["u_era5"]) & np.isfinite(out["v_era5"])
    if "dqf" in f.columns:
        out["dqf"] = f["dqf"].to_numpy()
    print(f"DMW netCDF {sat}: {len(out):,} winds ({path.name})")
    return out.reset_index(drop=True)


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]) * 6371.0


def match(b: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    from scipy.spatial import cKDTree

    tree = cKDTree(xyz(d["lat"].to_numpy(), d["lon"].to_numpy()))
    cand = tree.query_ball_point(xyz(b["latitude"].to_numpy(), b["longitude"].to_numpy()), r=MATCH_KM)
    rows = []
    bp, bu, bv = b["p"].to_numpy(), b["u"].to_numpy(), b["v"].to_numpy()
    dp, du, dv = d["p"].to_numpy(), d["u"].to_numpy(), d["v"].to_numpy()
    for i, js in enumerate(cand):
        if not js:
            continue
        js = np.asarray(js)
        ok = (np.abs(dp[js] - bp[i]) <= MATCH_HPA) & (np.hypot(du[js] - bu[i], dv[js] - bv[i]) <= MATCH_MS)
        if ok.any():
            js = js[ok]
            cost = np.abs(dp[js] - bp[i]) / MATCH_HPA + np.hypot(du[js] - bu[i], dv[js] - bv[i]) / MATCH_MS
            rows.append((i, int(js[np.argmin(cost)]), float(cost.min())))
    m = pd.DataFrame(rows, columns=["i_bufr", "j_dmw", "cost"]).sort_values("cost")
    m = m.drop_duplicates("j_dmw").drop_duplicates("i_bufr")          # one-to-one
    return m


def main(argv: list[str]) -> int:
    sat = "G16"
    if "--sat" in argv:
        sat = argv[argv.index("--sat") + 1].upper()
    b, d = load_bufr(sat), load_dmw_era5(sat)
    m = match(b, d)
    print(f"matched {len(m):,} = {len(m) / len(b):.1%} of BUFR, {len(m) / len(d):.1%} of netCDF")
    t = d.iloc[m["j_dmw"].to_numpy()].reset_index(drop=True)
    t["qi"] = b["qi"].to_numpy()[m["i_bufr"].to_numpy()]
    if "dqf" in t.columns:
        print("DQF of matched winds:", t["dqf"].value_counts().to_dict())
        um = np.setdiff1d(np.arange(len(d)), m["j_dmw"].to_numpy())
        print("DQF of unmatched netCDF winds:", d.iloc[um]["dqf"].value_counts().to_dict())
    out_dir = PROJECT_ROOT / "outputs" / CASE / "qc_selfsup"
    out_dir.mkdir(parents=True, exist_ok=True)
    t.to_csv(out_dir / f"operational_{sat.lower()}_bufr_matched.csv", index=False)

    e = t.loc[t["era5_ok"] & t["p"].between(100.0, 1000.0)]
    vd = np.hypot(e["u"] - e["u_era5"], e["v"] - e["v_era5"]).to_numpy()
    q = e["qi"].to_numpy()
    ok = np.isfinite(vd) & np.isfinite(q)
    vd, q, spd = vd[ok], q[ok], np.hypot(e["u"], e["v"]).to_numpy()[ok]
    curve, hist = qc.qi_curve_tables(q, vd)
    curve.to_csv(out_dir / f"operational_{sat.lower()}_qi_curve.csv", index=False)
    hist.to_csv(out_dir / f"operational_{sat.lower()}_qi_hist.csv", index=False)
    rows = [{"subset": "all", "n": int(vd.size), "fraction": 1.0, "mvd": float(vd.mean()),
             "rmsvd": float(np.sqrt(np.mean(vd ** 2)))}]
    for thr in (60.0, 80.0):
        k = q >= thr
        rows.append({"subset": f"qi>={thr:.0f}", "n": int(k.sum()), "fraction": float(k.mean()),
                     "mvd": float(vd[k].mean()), "rmsvd": float(np.sqrt(np.mean(vd[k] ** 2)))})
    thr = pd.DataFrame(rows)
    thr.to_csv(out_dir / f"operational_{sat.lower()}_qi_thresholds.csv", index=False)
    print("\noperational (BUFR QI without forecast) vs ERA5:")
    print(thr.round(3).to_string(index=False))
    sb = []
    for interval, g in pd.DataFrame({"s": spd, "vd": vd, "q": q}).groupby(
            pd.cut(spd, qc.SPEED_BINS, right=False), observed=True):
        k = g["q"] >= 60
        sb.append({"speed_bin": str(interval), "n": len(g), "frac_qi60": float(k.mean()),
                   "mvd_all": float(g["vd"].mean()), "mvd_qi60": float(g.loc[k, "vd"].mean()) if k.any() else np.nan,
                   "mvd_rejected": float(g.loc[~k, "vd"].mean()) if (~k).any() else np.nan})
    sb = pd.DataFrame(sb)
    sb.to_csv(out_dir / f"operational_{sat.lower()}_speed_bins.csv", index=False)
    print("\nby wind speed:")
    print(sb.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
