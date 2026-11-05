"""IGRA radiosonde comparison summary: MVD per method for all / high / mid / low levels.

One row of four panels per case (dot plot, methods on the y axis, labels only on
the first panel).  Point = MVD from metrics_all_points.csv; optional error bar =
95 % station-block bootstrap interval computed from the collocation NPZ files
(stations are resampled with replacement, all their levels together).

Run from the project root:

    python scripts/plot_radiosonde_validation.py

Output: outputs/figures/igra_summary/igra_mvd_summary.pdf/.png (+ CSV of values/CIs).
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# =============================================================================
# Settings
# =============================================================================
# One row per case.  "igra_dir" = COMPARISON_DIR of evaluate_igra_seviri.py
# (contains metrics/ and igra_collocations/).
# "aliases" maps the METHODS keys (SEVIRI names) to this case's tracker names.
GOES_CASE = "goes_g16_g18_E20240804T120506_W20240804T120508"
CASES = (
    {"label": "SEVIRI", "igra_dir": PROJECT_ROOT / "outputs" / "test_2018-01-12T11" / "comparison" / "amv_vs_igra",
     "operational": "official", "suffix": "_oca_pressure", "aliases": {}},
    {"label": "GOES-16", "igra_dir": PROJECT_ROOT / "outputs" / GOES_CASE / "comparison" / "amv_vs_igra_g16",
     "operational": "official", "suffix": "_goes_pressure",
     "aliases": {}},
)

# key in metrics CSV ("tracker" column), legend label, colour, filled marker?
METHODS = (
    ("official", "Operational", "#000000", True),
    ("raw_deepflow_opencv", "DeepFlow", "#2a78d6", True),
    ("layerwise_deepflow_opencv", "DeepFlow (P)", "#2a78d6", False),
    ("nn_raw_flowdiffuser", "FlowDiffuser", "#1baf7a", True),
    ("nn_layerwise_flowdiffuser", "FlowDiffuser (P)", "#1baf7a", False),
    ("pressure_conditioned_flow", "LayerFlow", "#e34948", True),
)
SUBSETS = (
    ("all_valid", "All levels"),
    ("pressure_high_lt400", "High (<400 hPa)"),
    ("pressure_mid_400_700", "Middle (400–700 hPa)"),
    ("pressure_low_gt700", "Low (>700 hPa)"),
)
LAYER_OF_SUBSET = {"pressure_high_lt400": "high", "pressure_mid_400_700": "middle", "pressure_low_gt700": "low"}

# "all_points" (no QC) or "qc_qi60_speed5" (QC subset; dense products use the
# self-supervised QI, operational products their own QI / DQF).  CLI: "qc".
METRIC_SET = "all_points"
BOOTSTRAP = True                 # needs igra_collocations/*.npz; skipped silently if absent
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0
X_LIMITS: tuple[float, float] | None = None   # None: shared per row, auto
LCFLOW_GUIDE = True              # faint vertical line at the LayerFlow value
SHOW_CASE_LABELS = True          # case name above the method labels of each row

FIG_WIDTH_MM = 180.0
ROW_HEIGHT_MM = 27.0
LABEL_MM = 30.0                  # method-label column
GAP_MM = 2.5
RIGHT_MM, BOTTOM_MM, TOP_MM = 1.5, 7.5, 7.5
FONT_NAME, FONT_PATH, FONT_SIZE_PT = "Times New Roman", None, 8.0

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "igra_summary"
PNG_DPI = 600
MM = 1.0 / 25.4


# =============================================================================
# Data
# =============================================================================


def resolved(case: dict[str, Any], tracker: str) -> str:
    return case.get("aliases", {}).get(tracker, tracker)


def collocation_path(case: dict[str, Any], tracker: str) -> Path:
    name = "official_amv" if tracker == case["operational"] else f"{resolved(case, tracker)}{case['suffix']}"
    return case["igra_dir"] / "igra_collocations" / f"{name}_{METRIC_SET}_collocated.npz"


def station_bootstrap(table: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float]:
    vd = np.hypot(table["u_amv"] - table["u_sonde"], table["v_amv"] - table["v_sonde"]).to_numpy()
    codes, stations = pd.factorize(table["station_id"].astype(str))
    sums = np.bincount(codes, weights=vd, minlength=len(stations))
    counts = np.bincount(codes, minlength=len(stations)).astype(float)
    draws = rng.integers(0, len(stations), size=(BOOTSTRAP_SAMPLES, len(stations)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def load_case(case: dict[str, Any]) -> pd.DataFrame:
    metrics_dir = case["igra_dir"] / "metrics"
    if not (metrics_dir / f"metrics_{METRIC_SET}.csv").is_file():
        metrics_dir = case["igra_dir"]             # allow pointing directly at a metrics folder
    m = pd.read_csv(metrics_dir / f"metrics_{METRIC_SET}.csv")
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for tracker, label, _, _ in METHODS:
        tab = None
        path = collocation_path(case, tracker)
        if BOOTSTRAP and path.is_file():
            with np.load(path, allow_pickle=False) as data:
                tab = pd.DataFrame({k: data[k] for k in ("station_id", "u_amv", "v_amv", "u_sonde", "v_sonde",
                                                        "pressure_layer") if k in data.files})
            tab = tab.loc[np.isfinite(tab[["u_amv", "v_amv", "u_sonde", "v_sonde"]]).all(axis=1)]
        for subset, _ in SUBSETS:
            r = m.loc[(m["tracker"] == resolved(case, tracker)) & (m["subset"] == subset)]
            if r.empty:
                continue
            r = r.iloc[0]
            lo = hi = np.nan
            if tab is not None:
                sel = tab if subset == "all_valid" else tab.loc[tab["pressure_layer"].astype(str) == LAYER_OF_SUBSET[subset]]
                if len(sel) > 1:
                    lo, hi = station_bootstrap(sel, rng)
            rows.append({"case": case["label"], "tracker": tracker, "label": label, "subset": subset,
                         "n": int(r["sample_count"]), "stations": int(r["station_count"]),
                         "mvd": float(r["mean_vector_difference"]), "ci_low": lo, "ci_high": hi,
                         "rmsvd": float(r["vector_difference_rmse"]),
                         "speed_bias": float(r["speed_bias_amv_minus_sonde"])})
    return pd.DataFrame(rows)


# =============================================================================
# Plot
# =============================================================================


def configure_fonts() -> None:
    name = FONT_NAME
    if FONT_PATH:
        font_manager.fontManager.addfont(FONT_PATH)
        name = font_manager.FontProperties(fname=FONT_PATH).get_name()
    else:
        try:
            font_manager.findfont(font_manager.FontProperties(family=FONT_NAME), fallback_to_default=False)
        except ValueError:
            name = "Liberation Serif"
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [name, "Liberation Serif", "DejaVu Serif"],
        "font.size": FONT_SIZE_PT, "mathtext.fontset": "stix", "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.0, "ytick.major.size": 0.0,
        "xtick.major.pad": 1.5, "ytick.major.pad": 2.0, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def plot(values: pd.DataFrame, out: Path) -> None:
    cases = list(dict.fromkeys(values["case"]))
    n_rows = len(cases)
    fig_h = TOP_MM + n_rows * ROW_HEIGHT_MM + (n_rows - 1) * (TOP_MM + BOTTOM_MM) + BOTTOM_MM
    fig = plt.figure(figsize=(FIG_WIDTH_MM * MM, fig_h * MM))
    panel_w = (FIG_WIDTH_MM - LABEL_MM - RIGHT_MM - (len(SUBSETS) - 1) * GAP_MM) / len(SUBSETS)
    ypos = {t: i for i, (t, *_) in enumerate(METHODS)}
    for row, case in enumerate(cases):
        v = values.loc[values["case"] == case]
        lo = np.nanmin(np.r_[v["mvd"], v["ci_low"]])
        hi = np.nanmax(np.r_[v["mvd"], v["ci_high"]])
        pad = 0.08 * (hi - lo)
        xlim = X_LIMITS or (np.floor((lo - pad) * 2) / 2, np.ceil((hi + pad) * 2) / 2)
        top_mm = TOP_MM + row * (ROW_HEIGHT_MM + TOP_MM + BOTTOM_MM)
        bottom = (fig_h - top_mm - ROW_HEIGHT_MM) / fig_h
        for k, (subset, title) in enumerate(SUBSETS):
            x0 = LABEL_MM + k * (panel_w + GAP_MM)
            ax = fig.add_axes([x0 / FIG_WIDTH_MM, bottom, panel_w / FIG_WIDTH_MM, ROW_HEIGHT_MM / fig_h])
            s = v.loc[v["subset"] == subset]
            for i in range(len(METHODS)):
                ax.axhline(i, color="#eeeeee", lw=0.4, zorder=0)
            lc = s.loc[s["tracker"] == "pressure_conditioned_flow", "mvd"]
            if LCFLOW_GUIDE and not lc.empty:
                ax.axvline(lc.iloc[0], color="#e34948", lw=0.5, alpha=0.35, zorder=0)
            for tracker, label, color, filled in METHODS:
                r = s.loc[s["tracker"] == tracker]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = ypos[tracker]
                if np.isfinite(r["ci_low"]):
                    ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=color, lw=0.8, solid_capstyle="butt", zorder=2)
                size = 4.0 if tracker == "pressure_conditioned_flow" else 3.3
                ax.plot(r["mvd"], y, marker="o", ms=size, mec=color, mew=0.9,
                        mfc=color if filled else "white", ls="none", zorder=3)
            ax.set_ylim(len(METHODS) - 0.5, -0.5)
            ax.set_xlim(*xlim)
            ax.set_yticks(range(len(METHODS)))
            ax.set_yticklabels([lab for _, lab, _, _ in METHODS] if k == 0 else [])
            ax.tick_params(axis="x", which="both", direction="in", top=True)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%g"))
            for spine in ax.spines.values():          # full frame, as in the height-profile figure
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_color("#333333")
            ax.set_xlabel("MVD (m s$^{-1}$)", labelpad=1.5)
            # Header: subset title + sample sizes (dense / operational).
            dense = s.loc[s["tracker"] == "pressure_conditioned_flow"]
            oper = s.loc[s["tracker"] == METHODS[0][0]]
            n_txt = ""
            if not dense.empty and not oper.empty:
                # N (stations), dense / operational -- explained in the caption.
                n_txt = (f"\n{int(dense['n'].iloc[0])} / {int(oper['n'].iloc[0])} "
                         f"({int(dense['stations'].iloc[0])} / {int(oper['stations'].iloc[0])})")
            ax.set_title(f"{title}{n_txt}", fontsize=FONT_SIZE_PT, pad=1.5, linespacing=1.05)
            if k == 0 and SHOW_CASE_LABELS:
                ax.text(-(LABEL_MM - 1.0) / panel_w, 1.0, case, transform=ax.transAxes, ha="left", va="bottom",
                        fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=PNG_DPI)
    plt.close(fig)
    print("wrote", out.with_suffix(".pdf"))


def main() -> int:
    global METRIC_SET
    if "qc" in sys.argv[1:]:
        METRIC_SET = "qc_qi60_speed5"
    tag = "" if METRIC_SET == "all_points" else "_qc"
    print("metric set:", METRIC_SET)
    configure_fonts()
    frames = []
    for c in CASES:
        path = c["igra_dir"] / "metrics" / f"metrics_{METRIC_SET}.csv"
        if not path.is_file():
            path = c["igra_dir"] / f"metrics_{METRIC_SET}.csv"
        if not path.is_file():
            print(f"skip {c['label']}: no metrics file in {c['igra_dir']}")
            continue
        if path.stat().st_size < 10:
            print(f"skip {c['label']}: {path} is empty (the IGRA comparison produced no collocations)")
            continue
        frames.append(load_case(c))
    if not frames:
        print("Nothing to plot.")
        return 1
    values = pd.concat(frames, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    values.to_csv(OUTPUT_DIR / f"igra_mvd_summary{tag}.csv", index=False)
    with pd.option_context("display.width", 200):
        print(values.round(3).to_string(index=False))
    plot(values, OUTPUT_DIR / f"igra_mvd_summary{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
