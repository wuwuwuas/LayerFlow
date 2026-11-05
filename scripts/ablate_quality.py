"""Quality-indicator component ablation (no retraining; ERA5 only).

Four nested variants of the point-wise QI, evaluated at the same QI thresholds
at the standard thresholds (60 / 80) with the retained fraction:

    spread     raw ensemble spread S/(2(K-1)) as a plug-in variance (no network, no prior)
    tracking   network variance + empirical-Bayes posterior (tracking term only)
    +height    tracking + height-assignment term, absolute tolerance only (rho = 0)
    full       +height with tau_i = max(tau, rho |V|)  (= quality_index in the npz)

for all vectors and per wind-speed bin (MVD with the retained fraction), together with
the operational product (its own QI) and both products without QC.

    python scripts/ablate_quality.py test_2018-01-12T11 test_2018-12-01T01 \
        goes_g16_g18_E20240804T120506_W20240804T120508 --sat G16 [--tracker pressure_conditioned_flow]

    python scripts/ablate_quality.py plot        # redraw the retention figure only

Outputs
    outputs/<case>/qc_selfsup/<name>/ablation.csv, ablation_retention.csv   (per case)
    outputs/qc_ablation_retention_<tracker>.csv
    outputs/figures/qc_ablation_retention_<tracker>.pdf / .png   (MVD vs retained fraction)
    outputs/qc_ablation_<tracker>.csv / .tex              (all cases, one double-column table)
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_quality_control as qc  # noqa: E402

TRACKER = "pressure_conditioned_flow"
THRESHOLDS = (60.0, 80.0)
PLUGIN_ALPHA = 1.0e6             # alpha -> infinity: known variance (no prior)
VARIANTS = ("spread", "tracking", "+height", "full")
CASE_LABELS = {"test_2018-01-12T11": "SEVIRI day", "test_2018-12-01T01": "SEVIRI night"}


def case_label(case: str, sat: str | None) -> str:
    if qc.is_goes(case):
        return f"GOES-{sat[1:]}" if sat else "GOES"
    return CASE_LABELS.get(case, case)


def load(case: str, tracker: str, sat: str | None):
    p = qc.paths(case, tracker, sat)
    path = p["result"] / f"{p['name']}_selfsup_qc.npz"
    if not path.is_file():
        legacy = p["result"] / f"{tracker}_selfsup_qc.npz"
        path = legacy if legacy.is_file() else path
    if not path.is_file():
        print("missing", path)
        return None, None
    with np.load(path, allow_pickle=False) as d:
        data = {k: np.asarray(d[k]) for k in d.files}
    if "height_var" not in data or "S_p" not in data:
        print(f"  {path.name} has no height term (older QC run): rerun run_quality_control.py for this case, skip")
        return None, None
    if "speed_ms" not in data:                       # older files: speed from the product (as in --rescore)
        with np.load(p["product"], allow_pickle=False) as prod:
            ps = pd.DataFrame({"row": np.rint(prod["row_center_original"]).astype(np.int64),
                               "col": np.rint(prod["col_center_original"]).astype(np.int64),
                               "speed": np.hypot(prod["u_ms"], prod["v_ms"])}).drop_duplicates(["row", "col"])
        key = pd.DataFrame({"row": data["row_center_original"].astype(np.int64),
                            "col": data["col_center_original"].astype(np.int64)})
        data["speed_ms"] = key.merge(ps, on=["row", "col"], how="left")["speed"].to_numpy(np.float32)
        print("  speed_ms not in npz: taken from the product")
    return p, data


def era5_vd(p: dict, tracker: str, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(index into the QC points, vector difference to ERA5); same join as era5_validation."""
    table = qc.era5_table(p, tracker)
    if "era5_valid_mask" in table.columns:
        table = table.loc[table["era5_valid_mask"].astype(bool)]
    era = pd.DataFrame({"row": np.rint(table["row_center_original"]).astype(np.int64),
                        "col": np.rint(table["col_center_original"]).astype(np.int64),
                        "vd": np.hypot(table["u_amv"] - table["u_era5"], table["v_amv"] - table["v_era5"])})
    key = pd.DataFrame({"row": rows.astype(np.int64), "col": cols.astype(np.int64), "i": np.arange(rows.size)})
    m = key.merge(era, on=["row", "col"], how="inner")          # inner join, as in the main validation
    m = m.loc[np.isfinite(m["vd"])]
    return m["i"].to_numpy(), m["vd"].to_numpy()


def variant_qi(data: dict, meta: dict) -> dict[str, np.ndarray]:
    K = data["valid_component_count"].astype(np.float64)
    var_post = data["ee_post"].astype(np.float64) ** 2 * K / 2.0
    var_spread = data["ee_spread"].astype(np.float64) ** 2 * K / 2.0
    h = data["height_var"].astype(np.float64) if "height_var" in data else np.zeros_like(K)
    speed = np.nan_to_num(data["speed_ms"].astype(np.float64), nan=0.0)
    alpha, tau, rho = float(meta["alpha"]), float(meta["tau_ms"]), float(meta.get("rel_tol", qc.QI_REL_TOL))
    tau_i = np.maximum(tau, rho * speed)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {
            "spread": qc.pointwise_qi(np.maximum(var_spread, 1e-12), K, PLUGIN_ALPHA, tau),
            "tracking": qc.pointwise_qi(var_post, K, alpha, tau),
            "+height": qc.pointwise_qi(var_post, K, alpha, tau, h),
            "full": qc.pointwise_qi(var_post, K, alpha, tau_i, h),
        }
    stored = data["quality_index"].astype(np.float64)
    diff = np.nanmax(np.abs(out["full"] - stored))
    print(f"  check: full vs stored quality_index, max |diff| = {diff:.3g}"
          + ("" if diff < 0.05 else "  <-- npz was scored with other settings (run --rescore first)"))
    print(f"  alpha {alpha:.2f}, tau {tau:.3f} m/s, rho {rho}")
    return out


def mvd_top(qi: np.ndarray, vd: np.ndarray, frac: float) -> float:
    """MVD of the fraction `frac` of vectors with the highest QI (same retention for all variants)."""
    if vd.size == 0:
        return float("nan")
    n = max(1, int(round(frac * vd.size)))
    return float(vd[np.argsort(-qi, kind="stable")[:n]].mean())


CURVES: list[pd.DataFrame] = []      # retention curves collected by run()
RETENTION_GRID = np.round(np.arange(0.05, 1.0001, 0.01), 3)
FAST_MS = 20.0                   # fast winds for the lower panels of the retention figure


def retention_curve(qi: np.ndarray, vd: np.ndarray) -> np.ndarray:
    """MVD of the vectors with the highest QI, for each retained fraction in RETENTION_GRID."""
    if vd.size == 0:
        return np.full(RETENTION_GRID.shape, np.nan)
    order = np.argsort(-qi, kind="stable")
    csum = np.cumsum(vd[order])
    n = np.maximum(1, np.rint(RETENTION_GRID * vd.size).astype(int))
    return csum[n - 1] / n


SPEED_EDGES = (0.0, 10.0, 20.0, 30.0, np.inf)       # m/s
SPEED_NAMES = ("0--10", "10--20", "20--30", r"$\geq$30")


def evaluate(qi: np.ndarray, vd: np.ndarray, speed: np.ndarray, ref_frac: dict | None = None) -> dict:
    """All vectors and per speed bin: no-QC MVD, and MVD / retained fraction at QI >= 60, 80."""
    row = {"n": int(vd.size), "mvd_all": float(vd.mean())}
    bins = [(speed >= lo) & (speed < hi) for lo, hi in zip(SPEED_EDGES[:-1], SPEED_EDGES[1:])]
    for j, g in enumerate(bins):
        row[f"b{j}_share"] = float(g.mean())
        row[f"b{j}_all"] = float(vd[g].mean()) if g.any() else np.nan
    for thr in THRESHOLDS:
        t = f"{thr:.0f}"
        k = qi >= thr
        row[f"frac{t}"] = float(k.mean())
        row[f"mvd{t}"] = float(vd[k].mean()) if k.any() else np.nan
        if ref_frac is not None:      # MVD at the retention of the full QI at this threshold
            row[f"mvd_ret{t}"] = mvd_top(qi, vd, ref_frac[thr])
        for j, g in enumerate(bins):
            row[f"b{j}_frac{t}"] = float(k[g].mean()) if g.any() else np.nan
            row[f"b{j}_kept{t}"] = float(vd[g & k].mean()) if (g & k).any() else np.nan
            row[f"b{j}_rej{t}"] = float(vd[g & ~k].mean()) if (g & ~k).any() else np.nan
    return row


def operational(case: str, tracker: str, sat: str | None):
    """(QI, VD to ERA5, speed) of the operational AMVs; same selection as the main-text numbers."""
    if qc.is_goes(case):            # BUFR QI without forecast matched to DQF = 0 DMW (match_operational_quality_goes.py)
        f = qc.PROJECT_ROOT / "outputs" / case / "qc_selfsup" / f"operational_{sat.lower()}_bufr_matched.csv"
        if not f.is_file():
            print("  no operational table:", f, "(run match_operational_quality_goes.py)")
            return None
        t = pd.read_csv(f)
        t = t.loc[t["era5_ok"].astype(bool) & t["p"].between(100.0, 1000.0)]
        vd = np.hypot(t["u"] - t["u_era5"], t["v"] - t["v_era5"]).to_numpy()
        q, spd = t["qi"].to_numpy(float), np.hypot(t["u"], t["v"]).to_numpy()
    else:                           # EUMETSAT QI (as in official_retention_curve)
        from data_io.official_amv import load_official_amv_file
        from data_io.product_discovery import find_official_amv_file, infer_amv_valid_time_from_path
        from validation.compare_era5 import build_official_amv_vs_era5_table

        p = qc.paths(case, tracker, sat)
        from runtime_options import read_case_inputs
        inputs = read_case_inputs(qc.PROJECT_ROOT, case)
        data_dir = Path(inputs.get("data_dir", qc.PROJECT_ROOT / "data" / case.removeprefix("test_")))
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
        q = t["official_quality"].to_numpy(dtype=float)
        if np.nanmedian(q) <= 1.5:
            q = q * 100.0
        spd = np.hypot(t["u_amv"], t["v_amv"]).to_numpy()
    ok = np.isfinite(vd) & np.isfinite(q) & np.isfinite(spd)
    print(f"  operational AMVs with QI and ERA5: {ok.sum():,}")
    return q[ok], vd[ok], spd[ok]


def run(case: str, tracker: str, sat: str | None) -> pd.DataFrame | None:
    p, data = load(case, tracker, sat)
    if data is None:
        return None
    meta = json.loads(str(data["metadata_json"].ravel()[0]))
    label = case_label(case, sat)
    print(f"\n=== {label}  ({p['name']})")
    qis = variant_qi(data, meta)
    idx, vd = era5_vd(p, tracker, data["row_center_original"], data["col_center_original"])
    speed = data["speed_ms"].astype(np.float64)[idx]
    print(f"  ERA5-collocated points: {vd.size:,}")
    qfull = qis["full"][idx]
    ref = {thr: float(np.mean(qfull[np.isfinite(qfull)] >= thr)) for thr in THRESHOLDS}
    rows = []
    try:
        op = operational(case, tracker, sat)
    except Exception as exc:  # noqa: BLE001
        print(f"  operational skipped ({type(exc).__name__}: {exc})")
        op = None
    if op is not None:
        rows.append({"case": label, "variant": "operational", **evaluate(*op)})
    for v in VARIANTS:
        q = qis[v][idx]
        ok = np.isfinite(q)
        rows.append({"case": label, "variant": v, **evaluate(q[ok], vd[ok], speed[ok], ref)})
    out = pd.DataFrame(rows)
    out.to_csv(p["result"] / "ablation.csv", index=False)
    # MVD-retention curves (all winds and fast winds), ranked by each configuration's QI
    curves = []
    fast = speed >= FAST_MS
    for v in VARIANTS:
        q = qis[v][idx]
        ok = np.isfinite(q)
        for subset, sel in (("all", ok), ("fast", ok & fast)):
            curves.append(pd.DataFrame({"case": label, "variant": v, "subset": subset,
                                        "retention": RETENTION_GRID, "mvd": retention_curve(q[sel], vd[sel])}))
    for thr in THRESHOLDS:                            # where the final QI thresholds lie on the curves
        for subset, sel in (("all", np.isfinite(qfull)), ("fast", np.isfinite(qfull) & fast)):
            curves.append(pd.DataFrame({"case": [label], "variant": [f"full_qi{thr:.0f}"], "subset": [subset],
                                        "retention": [float(np.mean(qfull[sel] >= thr))], "mvd": [np.nan]}))
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(p["result"] / "ablation_retention.csv", index=False)
    CURVES.append(curves)
    show = ["variant", "mvd_all"] + [f"b{j}_all" for j in range(len(SPEED_NAMES))]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print("  no QC: MVD all and per speed bin (0-10, 10-20, 20-30, >=30)")
        print(out[show].round(2).to_string(index=False))
        for thr in THRESHOLDS:
            t = f"{thr:.0f}"
            cols = ["variant", f"mvd{t}", f"frac{t}"] + [c for j in range(len(SPEED_NAMES))
                                                        for c in (f"b{j}_kept{t}", f"b{j}_frac{t}")]
            if f"mvd_ret{t}" in out:
                cols.insert(3, f"mvd_ret{t}")
            print(f"  QI >= {t}: MVD (retained) all and per speed bin")
            print(out[cols].round(3).to_string(index=False))
    return out


NAMES = {"operational": "Operational", "spread": "Ensemble spread", "tracking": "Tracking term",
         "+height": "+ height term", "full": "+ $\\rho$ (full)"}


def _cell(mvd: float, frac: float) -> str:
    return "--" if not np.isfinite(mvd) else f"{mvd:.2f} ({100 * frac:.1f})"


def latex_table(table: pd.DataFrame) -> str:
    """One double-column table: QI >= 60 | QI >= 80, each All + four speed bins, 'MVD (retained %)'.
    Per case: operational without / with QC, LayerFlow without QC, then the four QI variants."""
    nb = len(SPEED_NAMES)
    ncol = 1 + 2 * (nb + 1)
    sub = " & ".join(["All", *SPEED_NAMES])
    lines = [r"\begin{tabular}{l " + " ".join(["c" * (nb + 1)] * 2) + "}", r"\toprule",
             rf" & \multicolumn{{{nb + 1}}}{{c}}{{QI\,$\geq$\,60}} & \multicolumn{{{nb + 1}}}{{c}}{{QI\,$\geq$\,80}} \\",
             rf"\cmidrule(lr){{2-{nb + 2}}}\cmidrule(lr){{{nb + 3}-{ncol}}}",
             rf" & {sub} & {sub} \\", r"\midrule"]

    def no_qc(r) -> str:
        half = " & ".join([f"{r['mvd_all']:.2f}"] + [f"{r[f'b{j}_all']:.2f}" for j in range(nb)])
        return f"{half} & {half}"

    def with_qc(r) -> str:
        cells = []
        for thr in THRESHOLDS:
            t = f"{thr:.0f}"
            cells.append(_cell(r[f"mvd{t}"], r[f"frac{t}"]))
            cells += [_cell(r[f"b{j}_kept{t}"], r[f"b{j}_frac{t}"]) for j in range(nb)]
        return " & ".join(cells)

    for k, (case, g) in enumerate(table.groupby("case", sort=False)):
        if k:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{{case}}}}} \\")
        op = g.loc[g["variant"] == "operational"]
        if not op.empty:
            r = op.iloc[0]
            lines.append(rf"Operational, unfiltered & {no_qc(r)} \\")
            lines.append(rf"Operational QI & {with_qc(r)} \\")
        lf = g.loc[g["variant"] != "operational"]
        lines.append(rf"LayerFlow, unfiltered & {no_qc(lf.iloc[0])} \\")
        for _, r in lf.iterrows():
            lines.append(rf"\quad {NAMES[r['variant']]} & {with_qc(r)} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


CURVE_STYLE = {"spread": ("Ensemble spread", "#7F7F7F", (0, (3, 1.5)), 1.0),
               "tracking": ("Tracking term", "#2F6FD6", "-", 1.0),
               "+height": ("+ height term", "#F28E2B", "-", 1.0),
               "full": (r"+ $\rho$ (full)", "#D62728", "-", 1.4)}
FIG_W_CM, FIG_H_CM, FONT_PT = 9.0, 8.2, 7.5          # single-column figure
REF_COLOR = "#BDBDBD"                                 # threshold reference lines (dotted)


def plot_retention(curves: pd.DataFrame, tracker: str) -> None:
    """2 x 2: columns = cases, rows = all winds / fast winds; MVD against the retained fraction.
    Dotted lines: retained fraction of the final QI at QI >= 80 and >= 60 within each subset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = 1 / 2.54
    cases = list(dict.fromkeys(curves["case"]))
    with plt.rc_context({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
                         "font.size": FONT_PT, "axes.linewidth": 0.5, "mathtext.fontset": "stix",
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.major.size": 2.0, "ytick.major.size": 2.0,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.pad": 1.5, "ytick.major.pad": 1.5}):
        fig, axes = plt.subplots(2, len(cases), figsize=(FIG_W_CM * cm, FIG_H_CM * cm), squeeze=False,
                                 sharex=True)
        for r, subset in enumerate(("all", "fast")):
            for c, case in enumerate(cases):
                ax = axes[r, c]
                g = curves[(curves["case"] == case) & (curves["subset"] == subset)]
                for v, (lab, col, ls, lw) in CURVE_STYLE.items():
                    h = g[g["variant"] == v]
                    ax.plot(100 * h["retention"], h["mvd"], color=col, ls=ls, lw=lw, label=lab, zorder=3)
                ax.set_xlim(0, 100)
                ax.set_xticks([0, 25, 50, 75, 100])
                ax.tick_params(top=True, right=True)
                ymin, ymax = ax.get_ylim()
                ymax = ymax + 0.22 * (ymax - ymin)        # headroom for the threshold labels
                for thr_name, txt in (("full_qi80", r"QI$\geq$80"), ("full_qi60", r"QI$\geq$60")):
                    x = g.loc[g["variant"] == thr_name, "retention"]
                    if len(x):
                        xv = 100 * float(x.iloc[0])
                        ax.axvline(xv, color=REF_COLOR, lw=0.7, ls=(0, (1, 1.2)), zorder=1)
                        ax.text(xv - 1.2, ymax - 0.03 * (ymax - ymin), txt, rotation=90, ha="right", va="top",
                                fontsize=FONT_PT - 1.5, color="#8C8C8C", zorder=2)
                ax.set_ylim(ymin, ymax)
                if r == 0:
                    ax.set_title(case.replace(" day", ""), pad=2.5, fontsize=FONT_PT + 0.5)
                if r == 1:
                    ax.set_xlabel("Retained fraction (%)", labelpad=1.5)
                if c == 0:
                    ax.set_ylabel("All winds\nMVD (m s$^{-1}$)" if subset == "all"
                                  else f"Winds $\\geq${FAST_MS:.0f} m s$^{{-1}}$\nMVD (m s$^{{-1}}$)",
                                  labelpad=2, linespacing=1.35)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, handlelength=2.0,
                   columnspacing=1.5, handletextpad=0.5, labelspacing=0.25, bbox_to_anchor=(0.56, 1.0),
                   fontsize=FONT_PT)
        fig.subplots_adjust(left=0.165, right=0.965, bottom=0.105, top=0.845, wspace=0.24, hspace=0.12)
        fig.align_ylabels(axes[:, 0])
        out = qc.PROJECT_ROOT / "outputs" / "figures" / f"qc_ablation_retention_{tracker}"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.with_suffix(".pdf"))
        fig.savefig(out.with_suffix(".png"), dpi=600)
        plt.close(fig)
    print("wrote", out.with_suffix(".pdf"))


def main(argv: list[str]) -> int:
    args = list(argv[1:])
    sat, tracker = None, TRACKER
    if "--sat" in args:
        k = args.index("--sat")
        sat = args[k + 1].upper()
        del args[k:k + 2]
    if "--tracker" in args:
        k = args.index("--tracker")
        tracker = args[k + 1]
        del args[k:k + 2]
    if args and args[0] == "plot":                  # redraw the figure from the saved curves
        plot_retention(pd.read_csv(qc.PROJECT_ROOT / "outputs" / f"qc_ablation_retention_{tracker}.csv"), tracker)
        return 0
    cases = args or [qc.CASE_NAME]
    tables = []
    for case in cases:
        s = (sat or "G16") if qc.is_goes(case) else None
        try:
            t = run(case, tracker, s)
        except Exception as exc:  # noqa: BLE001  (one bad case must not stop the others)
            print(f"  {case}: failed ({type(exc).__name__}: {exc})")
            t = None
        if t is not None:
            tables.append(t)
    if not tables:
        return 1
    allt = pd.concat(tables, ignore_index=True)
    out = qc.PROJECT_ROOT / "outputs" / f"qc_ablation_{tracker}"
    allt.to_csv(out.with_suffix(".csv"), index=False)
    out.with_suffix(".tex").write_text(latex_table(allt), encoding="utf-8")
    print(f"\nwrote {out.with_suffix('.csv')} and {out.with_suffix('.tex')}")
    curves = pd.concat(CURVES, ignore_index=True)
    curves.to_csv(qc.PROJECT_ROOT / "outputs" / f"qc_ablation_retention_{tracker}.csv", index=False)
    plot_retention(curves, tracker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
