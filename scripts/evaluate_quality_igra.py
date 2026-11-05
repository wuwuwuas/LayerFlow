"""MVD against IGRA radiosondes as a function of the QI threshold (cumulative, QI >= q).

Re-uses the point-level IGRA geometry caches written by evaluate_igra_seviri.py /
evaluate_igra_goes.py (same 100 km / Gaussian 50 km / +-60 min / +-25 hPa / drift
settings), joins them once with the vectors and their QI, and aggregates the
station-level winds for every threshold q exactly like the IGRA comparison
(distance-weighted mean, >= MIN_DENSE / MIN_OPERATIONAL vectors per level).
At q = 0 the result equals the all-points IGRA metric (printed as a check).

Dense products: self-supervised QI.  Operational: SEVIRI = EUMETSAT QI;
GOES = BUFR QI without forecast (via outputs/<case>/qc_selfsup/operational_<sat>_bufr_matched.csv
from match_operational_quality_goes.py), on the DQF = 0 winds used in the GOES IGRA comparison.

    python scripts/evaluate_quality_igra.py test_2018-01-12T11
    python scripts/evaluate_quality_igra.py goes_g16_g18_E20240804T120506_W20240804T120508 --sat G16

Output: outputs/<case>/comparison/amv_vs_igra[_<sat>]/qi_curves/igra_qi_curve_<name>.csv
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_quality_control as qc  # noqa: E402  (PROJECT_ROOT / src path)

PROJECT_ROOT = qc.PROJECT_ROOT
SEVIRI_TRACKERS = ["pressure_conditioned_flow", "raw_deepflow_opencv", "layerwise_deepflow_opencv",
                   "nn_raw_flowdiffuser", "nn_layerwise_flowdiffuser"]
GOES_TRACKERS = ["pressure_conditioned_flow", "raw_deepflow_opencv", "layerwise_deepflow_opencv",
                 "nn_raw_flowdiffuser", "nn_layerwise_flowdiffuser"]
GOES_GEOMETRY = "cloud_height"
DMW_PRESSURE_RANGE = (100.0, 1000.0)
MIN_DENSE = 3
MIN_OPERATIONAL = 1
Q_STEP = 5.0


def comparison_dir(case: str, sat: str | None) -> Path:
    base = PROJECT_ROOT / "outputs" / case / "comparison"
    return base / (f"amv_vs_igra_{sat.lower()}" if sat else "amv_vs_igra")


def newest(paths: list[Path]) -> Path | None:
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def curve_from_geometry(frame: pd.DataFrame, geometry: pd.DataFrame, min_points: int) -> pd.DataFrame:
    from validation.compare_igra import add_amv_point_keys

    keys = ["_amv_key_kind", "_amv_key1", "_amv_key2", "_amv_key3"]
    f = add_amv_point_keys(frame, include_pressure=True)[keys + ["u_amv", "v_amv", "qi"]]
    j = geometry[keys + ["station_id", "sounding_time", "level_index", "u_sonde", "v_sonde", "weight"]].merge(
        f, on=keys, how="inner")
    for c in ("u_amv", "v_amv", "u_sonde", "v_sonde", "weight", "qi"):
        j[c] = pd.to_numeric(j[c], errors="coerce")
    j = j.loc[np.isfinite(j[["u_amv", "v_amv", "u_sonde", "v_sonde", "weight"]]).all(axis=1) & (j["weight"] > 0)]
    j["level"] = pd.factorize(j["station_id"].astype(str) + "|" + j["sounding_time"].astype(str) + "|"
                              + j["level_index"].astype(str))[0]
    j["uw"], j["vw"] = j["u_amv"] * j["weight"], j["v_amv"] * j["weight"]
    nlev = int(j["level"].max()) + 1 if len(j) else 0
    lev = j["level"].to_numpy()
    us = np.zeros(nlev); vs = np.zeros(nlev)
    us[lev], vs[lev] = j["u_sonde"].to_numpy(), j["v_sonde"].to_numpy()
    station = pd.Series(j["station_id"].astype(str).to_numpy(), index=lev).groupby(level=0).first()
    rows = []
    q_all = j["qi"].to_numpy()
    for q in np.arange(0.0, 100.0, Q_STEP):
        k = np.ones(q_all.size, bool) if q == 0 else (q_all >= q)
        lv = lev[k]
        w = np.bincount(lv, weights=j["weight"].to_numpy()[k], minlength=nlev)
        n = np.bincount(lv, minlength=nlev)
        uw = np.bincount(lv, weights=j["uw"].to_numpy()[k], minlength=nlev)
        vw = np.bincount(lv, weights=j["vw"].to_numpy()[k], minlength=nlev)
        ok = (n >= min_points) & (w > 0)
        vd = np.hypot(uw[ok] / w[ok] - us[ok], vw[ok] / w[ok] - vs[ok])
        rows.append({"qi_min": q, "n_levels": int(ok.sum()),
                     "n_stations": int(station.reindex(np.flatnonzero(ok)).nunique()) if ok.any() else 0,
                     "mvd": float(vd.mean()) if vd.size else np.nan})
    return pd.DataFrame(rows)


def dense_frame(case: str, sat: str | None, tracker: str) -> tuple[pd.DataFrame, str] | None:
    from validation.compare_igra import load_pressure_amv_npz

    name = f"{tracker}_{sat.lower()}" if sat else tracker
    prod = (PROJECT_ROOT / "outputs" / case / "pressure_assignment" /
            (f"{name}_{GOES_GEOMETRY}_final_amv_goes_pressure.npz" if sat else f"{tracker}_final_amv_oca_pressure.npz"))
    qnpz = PROJECT_ROOT / "outputs" / case / "qc_selfsup" / name / f"{name}_selfsup_qc.npz"
    if not prod.is_file() or not qnpz.is_file():
        print("  missing", prod if not prod.is_file() else qnpz)
        return None
    frame = load_pressure_amv_npz(prod, tracker=tracker, amv_source="oca_pressure")
    with np.load(qnpz, allow_pickle=False) as d:
        qi = pd.DataFrame({"r": d["row_center_original"].astype(np.int64), "c": d["col_center_original"].astype(np.int64),
                           "qi": d["quality_index"].astype(np.float64)}).drop_duplicates(["r", "c"])
    key = pd.DataFrame({"r": np.rint(frame["row_center_original"]).astype(np.int64),
                        "c": np.rint(frame["col_center_original"]).astype(np.int64)})
    frame["qi"] = key.merge(qi, on=["r", "c"], how="left")["qi"].to_numpy()
    return frame, name


def seviri_operational(case: str) -> pd.DataFrame:
    from data_io.official_amv import load_official_amv_file
    from data_io.product_discovery import find_official_amv_file, infer_amv_valid_time_from_path
    from validation.compare_igra import standardize_official_amv

    from runtime_options import read_case_inputs
    inputs = read_case_inputs(PROJECT_ROOT, case)
    data_dir = Path(inputs.get("data_dir", PROJECT_ROOT / "data" / case.removeprefix("test_")))
    valid_time = (pd.Timestamp(inputs["frame_times"][-1]).to_pydatetime() if inputs.get("frame_times")
                  else infer_amv_valid_time_from_path(data_dir, offset_minutes=45))
    path = find_official_amv_file(data_dir / "eumetsat_amv" / "native",
                                  expected_time=valid_time,
                                  tolerance_minutes=90.0, verbose=False)
    f = standardize_official_amv(load_official_amv_file(path))
    q = pd.to_numeric(f["official_quality"], errors="coerce").to_numpy(float)
    f["qi"] = q * 100.0 if np.nanmedian(q) <= 1.5 else q
    return f


def goes_operational(case: str, sat: str) -> pd.DataFrame:
    """Same frame (same order -> same official_id) as compare_amv_igra_goes.load_official_frame, plus BUFR QI."""

    from data_io.goes_dmw import discover_goes_dmw_files, dmw_basic_valid_mask, load_goes_dmw
    import match_operational_quality_goes as gq

    raw = load_goes_dmw(discover_goes_dmw_files(PROJECT_ROOT / "data" / case, satellites=[sat])[0].path, satellite=sat)
    keep = dmw_basic_valid_mask(raw) & (raw["dqf"].to_numpy() == 0) & raw["pressure_hpa"].between(*DMW_PRESSURE_RANGE).to_numpy()
    dmw = raw.loc[keep].reset_index(drop=True)
    f = pd.DataFrame({"official_id": np.arange(len(dmw)), "longitude": dmw["longitude"].to_numpy(np.float64),
                      "latitude": dmw["latitude"].to_numpy(np.float64), "pressure_hpa": dmw["pressure_hpa"].to_numpy(np.float64),
                      "u_amv": dmw["u_ms"].to_numpy(np.float64), "v_amv": dmw["v_ms"].to_numpy(np.float64)})
    mpath = PROJECT_ROOT / "outputs" / case / "qc_selfsup" / f"operational_{sat.lower()}_bufr_matched.csv"
    if not mpath.is_file():
        raise FileNotFoundError(f"{mpath} -- run match_operational_quality_goes.py first")
    m = pd.read_csv(mpath)
    b = pd.DataFrame({"latitude": m["lat"], "longitude": m["lon"], "p": m["p"], "u": m["u"], "v": m["v"]})
    d = pd.DataFrame({"lat": f["latitude"], "lon": f["longitude"], "p": f["pressure_hpa"], "u": f["u_amv"], "v": f["v_amv"]})
    old = (gq.MATCH_KM, gq.MATCH_HPA, gq.MATCH_MS)
    gq.MATCH_KM, gq.MATCH_HPA, gq.MATCH_MS = 1.0, 0.5, 0.2          # same records -> near-exact match
    pairs = gq.match(b, d)
    gq.MATCH_KM, gq.MATCH_HPA, gq.MATCH_MS = old
    f["qi"] = np.nan
    f.loc[pairs["j_dmw"].to_numpy(), "qi"] = m["qi"].to_numpy()[pairs["i_bufr"].to_numpy()]
    print(f"  operational DMW (DQF=0): {len(f):,}; with BUFR QI: {np.isfinite(f['qi']).sum():,}")
    return f


def main(argv: list[str]) -> int:
    from validation.compare_igra import amv_geometry_fingerprint, load_collocation_npz

    args = list(argv[1:])
    sat = None
    if "--sat" in args:
        k = args.index("--sat")
        sat = args[k + 1].upper()
        del args[k:k + 2]
    case = args[0] if args else "test_2018-01-12T11"
    goes = case.lower().startswith("goes")
    if goes and sat is None:
        sat = "G16"
    cdir = comparison_dir(case, sat if goes else None)
    gdir, out = cdir / "geometry_cache", cdir / "qi_curves"
    out.mkdir(parents=True, exist_ok=True)
    allp = cdir / "metrics" / "metrics_all_points.csv"
    ref = pd.read_csv(allp) if allp.is_file() else None

    jobs = [("operational", None)] + [(t, None) for t in (GOES_TRACKERS if goes else SEVIRI_TRACKERS)]
    for tracker, _ in jobs:
        print(f"\n--- {tracker}")
        if tracker == "operational":
            frame = goes_operational(case, sat) if goes else seviri_operational(case)
            name, prefix, min_pts = "operational", "official_amv_geometry_", MIN_OPERATIONAL
        else:
            got = dense_frame(case, sat if goes else None, tracker)
            if got is None:
                continue
            frame, name = got
            prefix, min_pts = "oca_pressure_geometry_", MIN_DENSE
        fp = amv_geometry_fingerprint(frame.drop(columns=["qi"]))
        path = newest(sorted(gdir.glob(f"{prefix}{fp}_*.npz")))
        if path is None:
            path = newest(sorted(gdir.glob(f"{prefix}*.npz"))) if tracker == "operational" else None
        if path is None:
            print("  no geometry cache for", name, "in", gdir, "(run the IGRA comparison first)")
            continue
        geometry = load_collocation_npz(path)
        curve = curve_from_geometry(frame, geometry, min_pts)
        curve.to_csv(out / f"igra_qi_curve_{name}.csv", index=False)
        msg = f"  q=0: MVD {curve['mvd'].iloc[0]:.3f} ({curve['n_levels'].iloc[0]} levels)"
        if ref is not None and "subset" in ref.columns:
            comp = "official_amv" if tracker == "operational" else None
            sel = ref.loc[ref["subset"] == "all_valid"]
            sel = sel.loc[sel["comparison"] == comp] if comp else sel.loc[sel["tracker"].astype(str).str.startswith(tracker)]
            if not sel.empty:
                msg += f"; IGRA all-points metric {float(sel['mean_vector_difference'].iloc[0]):.3f}"
        print(msg)
        print(curve.loc[curve["qi_min"].isin([0, 20, 40, 60, 80, 90])].round(3).to_string(index=False))
    print("\nSaved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
