"""Ablation of the number of pressure layers K in LayerFlow (SEVIRI and GOES-16).

K = 4 is the default configuration (inner boundaries 300 / 500 / 700 hPa).
The K-1 inner boundaries are evenly spaced between 300 and 700 hPa (K = 2: 500 hPa);
the outer limits (50 / 1100 hPa) and all other settings (soft sigma 50 hPa,
background, solver) are unchanged:

    K = 2: 500                K = 4: 300, 500, 700   (default)
    K = 3: 300, 700           K = 5: 300, 433, 567, 700

The unlayered reference (K = 1) is the original DeepFlow (raw_deepflow_opencv /
raw_deepflow_opencv_g16), evaluated with the same code.

    # tracking (reuses the validated run scripts in scripts/; K = 4 is not rerun)
    python scripts/ablate_layers.py run  seviri --k 2 3 5
    python scripts/ablate_layers.py run  goes   --k 2 3 5
    # ERA5 evaluation of K = 2..5 (all + pressure ranges) -> CSV + LaTeX
    python scripts/ablate_layers.py eval

Trackers: pressure_conditioned_flow_k<K> (SEVIRI), pressure_conditioned_flow_k<K>_g16
(GOES); K = 4 maps to the existing pressure_conditioned_flow(_g16).
Outputs: outputs/layer_ablation.csv, outputs/layer_ablation.tex,
         outputs/layer_ablation_runtime.csv
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import importlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_quality_control as qc  # noqa: E402  (PROJECT_ROOT, src path, ERA5 tables)

PROJECT_ROOT = qc.PROJECT_ROOT
EXAMPLES_DIR = PROJECT_ROOT / "scripts"          # validated run scripts
SEVIRI_CASE = "test_2018-01-12T11"
GOES_CASE = "goes_g16_g18_E20240804T120506_W20240804T120508"
GOES_SAT = "G16"
BASE_TRACKER = "pressure_conditioned_flow"
MAIN_K = 4
K_ALL = (2, 3, 4, 5)
P_TOP, P_BOTTOM = 50.0, 1100.0
INNER_BOUNDARIES = {1: (), 2: (500.0,), 3: (300.0, 700.0), 4: (300.0, 500.0, 700.0),
                    5: (300.0, 433.3, 566.7, 700.0)}       # hPa, evenly spaced in 300-700
REFERENCE = {"SEVIRI": "raw_deepflow_opencv", "GOES-16": "raw_deepflow_opencv"}   # unlayered DeepFlow (K = 1)
PRESSURE_RANGES = (("high", 0.0, 400.0), ("middle", 400.0, 700.0), ("low", 700.0, 1100.0))  # high, middle and low pressure ranges


def layer_edges(k: int) -> list[float]:
    return [P_TOP, *INNER_BOUNDARIES[k], P_BOTTOM]


def layer_config(k: int):
    from preprocessing.layerwise_image import PressureLayer
    from tracking.pressure_conditioned_flow import default_pressure_layer_config

    e = layer_edges(k)
    layers = tuple(PressureLayer(f"k{k}_{j + 1}_{e[j]:g}_{e[j + 1]:g}", e[j], e[j + 1]) for j in range(k))
    cfg = replace(default_pressure_layer_config(), layers=layers)
    if k == MAIN_K:                                   # must reproduce the main-text configuration
        ref = [(l.p_min_hpa, l.p_max_hpa) for l in default_pressure_layer_config().layers]
        assert [(l.p_min_hpa, l.p_max_hpa) for l in layers] == ref, (layers, ref)
    return cfg


def tracker_name(k: int) -> str:
    return BASE_TRACKER if k == MAIN_K else f"{BASE_TRACKER}_k{k}"


def log_runtime(sensor: str, k: int, seconds: float) -> None:
    f = PROJECT_ROOT / "outputs" / "layer_ablation_runtime.csv"
    row = pd.DataFrame([{"sensor": sensor, "K": k, "runtime_min": seconds / 60.0,
                         "time": pd.Timestamp.now().isoformat(timespec="seconds")}])
    row.to_csv(f, mode="a", header=not f.is_file(), index=False)


def run_seviri(k: int) -> None:
    sys.path.insert(0, str(EXAMPLES_DIR))
    base = importlib.import_module('run_layerflow_seviri')
    from runtime_options import read_case_inputs
    inputs = read_case_inputs(PROJECT_ROOT, SEVIRI_CASE)
    out = PROJECT_ROOT / "outputs" / SEVIRI_CASE
    base.CASE_NAME = SEVIRI_CASE
    base.DATA_CASE_DIR = Path(inputs.get("data_dir", PROJECT_ROOT / "data" / SEVIRI_CASE.removeprefix("test_")))
    base.FRAME_TIMES = tuple(pd.Timestamp(t).to_pydatetime() for t in inputs.get("frame_times", ())) or None
    base.FRAME_INTERVAL_MINUTES = inputs.get("interval_minutes") or 15.0
    base.AOI = inputs.get("aoi")
    base.OUTPUT_DIR = out
    base.TRACKER_ROOT_DIR = out / "trackers"
    base.PRESSURE_OUTPUT_DIR = out / "pressure_assignment"
    base.TRACKER_NAME = tracker_name(k)
    base.LAYER_CONFIG = layer_config(k)
    print(f"\n##### SEVIRI K = {k}: {layer_edges(k)} hPa -> {base.TRACKER_NAME}")
    t0 = time.perf_counter()
    base.main()
    log_runtime("SEVIRI", k, time.perf_counter() - t0)


def run_goes(k: int) -> None:
    sys.path.insert(0, str(EXAMPLES_DIR))
    base = importlib.import_module('run_layerflow_goes')
    out = PROJECT_ROOT / "outputs" / GOES_CASE
    base.CASE_NAME = GOES_CASE
    base.DATA_CASE_DIR = PROJECT_ROOT / "data" / GOES_CASE
    base.OUTPUT_DIR = out
    base.TRACKER_ROOT_DIR = out / "trackers"
    base.PRESSURE_OUTPUT_DIR = out / "pressure_assignment"
    base.SATELLITES = (GOES_SAT,)
    base.TRACKER_TEMPLATE = tracker_name(k) + "_{satellite_lower}"
    base.SAVE_DIAGNOSTIC_FIGURES = False
    base.LAYER_CONFIG = layer_config(k)
    print(f"\n##### GOES-16 K = {k}: {layer_edges(k)} hPa -> {base.TRACKER_TEMPLATE}")
    t0 = time.perf_counter()
    base.main()
    log_runtime("GOES-16", k, time.perf_counter() - t0)


# ---------------------------------------------------------------- evaluation
def evaluate(case: str, k: int, sat: str | None) -> dict | None:
    tracker = REFERENCE["GOES-16" if sat else "SEVIRI"] if k == 1 else tracker_name(k)
    p = qc.paths(case, tracker, sat)
    if not p["product"].is_file():
        print("  missing", p["product"])
        return None
    t = qc.era5_table(p, tracker)
    if "era5_valid_mask" in t.columns:
        t = t.loc[t["era5_valid_mask"].astype(bool)]
    e = pd.DataFrame({"row": np.rint(t["row_center_original"]).astype(np.int64),
                      "col": np.rint(t["col_center_original"]).astype(np.int64),
                      "u": t["u_amv"].to_numpy(float), "v": t["v_amv"].to_numpy(float),
                      "ue": t["u_era5"].to_numpy(float), "ve": t["v_era5"].to_numpy(float)})
    with np.load(p["product"], allow_pickle=False) as prod:
        pr = pd.DataFrame({"row": np.rint(prod["row_center_original"]).astype(np.int64),
                           "col": np.rint(prod["col_center_original"]).astype(np.int64),
                           "p": np.asarray(prod["pressure_hpa"], float)}).drop_duplicates(["row", "col"])
    e = e.merge(pr, on=["row", "col"], how="left")
    vd = np.hypot(e["u"] - e["ue"], e["v"] - e["ve"]).to_numpy()
    s, se = np.hypot(e["u"], e["v"]).to_numpy(), np.hypot(e["ue"], e["ve"]).to_numpy()
    ok = np.isfinite(vd) & np.isfinite(s) & np.isfinite(se)
    vd, s, se, pp = vd[ok], s[ok], se[ok], e["p"].to_numpy()[ok]
    row = {"sensor": "GOES-16" if sat else "SEVIRI", "K": k, "tracker": tracker,
           "edges": "/".join(f"{x:g}" for x in layer_edges(k)),
           "n": int(vd.size), "mvd": float(vd.mean()), "rmsvd": float(np.sqrt(np.mean(vd ** 2))),
           "mae_speed": float(np.mean(np.abs(s - se))), "r_speed": float(np.corrcoef(s, se)[0, 1])}
    for name, lo, hi in PRESSURE_RANGES:
        g = (pp >= lo) & (pp < hi)
        row[f"mvd_{name}"] = float(vd[g].mean()) if g.any() else np.nan
        row[f"n_{name}"] = int(g.sum())
    print(f"  K={k}: n={row['n']:,} MVD {row['mvd']:.3f} RMSVD {row['rmsvd']:.3f} "
          + " ".join(f"{n} {row[f'mvd_{n}']:.2f}" for n, *_ in PRESSURE_RANGES))
    return row


def latex(t: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{c l cccc ccc}", r"\toprule",
             r" & & & & & & \multicolumn{3}{c}{MVD by pressure (hPa)} \\", r"\cmidrule(lr){7-9}",
             r"$K$ & Boundaries (hPa) & MVD & RMSVD & MAE$_{\mathrm{speed}}$ & $r_{\mathrm{speed}}$"
             r" & $<$400 & 400--700 & $\geq$700 \\", r"\midrule"]
    for k, (sensor, g) in enumerate(t.groupby("sensor", sort=False)):
        if k:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{9}}{{l}}{{\textit{{{sensor}}}}} \\")
        for _, r in g.iterrows():
            inner = r["edges"].split("/")[1:-1]
            b = "--" if not inner else ", ".join(inner)
            lines.append(f"{int(r['K'])} & {b} & {r['mvd']:.3f} & {r['rmsvd']:.3f} & {r['mae_speed']:.3f} & "
                         f"{r['r_speed']:.3f} & {r['mvd_high']:.2f} & {r['mvd_middle']:.2f} & {r['mvd_low']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def stage_eval(ks) -> None:
    rows = []
    for case, sat in ((SEVIRI_CASE, None), (GOES_CASE, GOES_SAT)):
        print(f"\n=== {'GOES-16' if sat else 'SEVIRI'}")
        for k in [1, *[x for x in ks if x != 1]]:
            try:
                r = evaluate(case, k, sat)
            except Exception as exc:  # noqa: BLE001
                print(f"  K={k}: failed ({type(exc).__name__}: {exc})")
                r = None
            if r is not None:
                rows.append(r)
    if not rows:
        return
    t = pd.DataFrame(rows)
    out = PROJECT_ROOT / "outputs" / "layer_ablation"
    t.to_csv(out.with_suffix(".csv"), index=False)
    out.with_suffix(".tex").write_text(latex(t), encoding="utf-8")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("\n" + t.round(3).to_string(index=False))
    print(f"\nwrote {out.with_suffix('.csv')} and {out.with_suffix('.tex')}")


def main(argv: list[str]) -> int:
    args = list(argv[1:])
    ks = list(K_ALL)
    if "--k" in args:
        i = args.index("--k")
        ks = [int(a) for a in args[i + 1:] if a.isdigit()]
        del args[i:]
    stage = args[0] if args else "eval"
    if stage == "run":
        sensor = args[1].lower() if len(args) > 1 else "seviri"
        for k in ks:
            if k == MAIN_K:
                print(f"K = {MAIN_K} is the main-text run ({BASE_TRACKER}); skipped")
                continue
            (run_goes if sensor.startswith("goes") else run_seviri)(k)
        return 0
    if stage == "edges":
        for k in ks:
            print(k, layer_edges(k))
        return 0
    stage_eval(ks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
