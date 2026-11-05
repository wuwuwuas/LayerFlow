"""Height-dependent error profiles: MVD and speed bias versus assigned pressure.

For each case (SEVIRI 2018-01-12, SEVIRI 2018-12-01, GOES-16, GOES-18) and each
method, ERA5 differences are profiled with a sliding pressure window (no coarse
high/mid/low bins).  A narrow side panel shows the assigned-pressure PDF of the
operational product and of LCFlow (OCA / ACHP pressure).

Pressure on the vertical axis is each product's own assigned pressure, i.e. the
profile contains both tracking and height-assignment error.  All optical-flow
methods share the OCA (SEVIRI) / ACHP (GOES) pressure, so their differences are
tracking only.

Two stages (run from the project root):

    python scripts/plot_pressure_error_profiles.py compute   # tables -> profile CSVs (slow part)
    python scripts/plot_pressure_error_profiles.py plot      # CSVs -> figures (seconds)
    python scripts/plot_pressure_error_profiles.py           # both
    python scripts/plot_pressure_error_profiles.py compute goes   # only cases whose key contains "goes"

Outputs in outputs/figures/height_profiles/:
    profiles_<case>.csv                one row per method x pressure window
    overall_<case>.csv                 all-valid N / MVD / bias (check against the ERA5 tables)
    height_profile_<case>.pdf/.png     [MVD | speed bias | PDF] for one case
    height_profile_legend.pdf/.png     shared legend
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# =============================================================================
# User settings
# =============================================================================
GOES_CASE = "goes_g16_g18_E20240804T120506_W20240804T120508"
GOES_GEOMETRY = "cloud_height"
CASES = (
    {"key": "seviri_20180112", "kind": "seviri", "case": "test_2018-01-12T11", "label": "SEVIRI 2018-01-12 11:45 UTC"},
    {"key": "seviri_20181201", "kind": "seviri", "case": "test_2018-12-01T01", "label": "SEVIRI 2018-12-01 01:45 UTC"},
    {"key": "goes_g16", "kind": "goes", "case": GOES_CASE, "satellite": "G16", "label": "GOES-16 2024-08-04"},
    {"key": "goes_g18", "kind": "goes", "case": GOES_CASE, "satellite": "G18", "label": "GOES-18 2024-08-04"},
)

# Methods: key, legend label, colour, line width, dash, tracker candidates
# (SEVIRI name; GOES uses "<name>_<sat>").  "operational" is special.
# Original images = solid, pressure-conditioned (layerwise) images = dashed,
# same colour per algorithm.  Operational black, LCFlow red and thicker.
METHODS = (
    {"key": "operational", "label": "Operational", "color": "#000000", "lw": 1.0, "dash": None},
    {"key": "deepflow", "label": "DeepFlow", "color": "#2a78d6", "lw": 0.85, "dash": None,
     "trackers": ("raw_deepflow_opencv",)},
    {"key": "deepflow_pc", "label": "DeepFlow (P)", "color": "#2a78d6", "lw": 0.85, "dash": (3.2, 1.6),
     "trackers": ("layerwise_deepflow_opencv",)},
    {"key": "flowdiffuser", "label": "FlowDiffuser", "color": "#1baf7a", "lw": 0.85, "dash": None,
     "trackers": ("nn_raw_flowdiffuser",)},
    {"key": "flowdiffuser_pc", "label": "FlowDiffuser (P)", "color": "#1baf7a", "lw": 0.85, "dash": (3.2, 1.6),
     "trackers": ("nn_layerwise_flowdiffuser",)},
    {"key": "lcflow", "label": "LayerFlow", "color": "#e34948", "lw": 1.5, "dash": None,
     "trackers": ("pressure_conditioned_flow",)},
)
PDF_METHODS = ("operational", "lcflow")        # pressure PDFs in the side panel
# The second PDF is the cloud-product pressure (OCA / ACHP) at the dense
# retrieval points - shared by all optical-flow methods - so it gets its own
# neutral style and a product label instead of the LCFlow colour.
# Both PDFs are translucent areas (with a thin edge of the same colour).
PDF_OPERATIONAL_COLOR, PDF_OPERATIONAL_ALPHA = "#6e6e6e", 0.40
PDF_CLOUD_COLOR, PDF_CLOUD_ALPHA = "#c27c2c", 0.40
# Where the PDF key goes: "shared" = second row of the shared legend figure
# (full words), "panel" = small key inside each PDF panel (short words), "none".
PDF_LEGEND = "shared"
PDF_SHARED_LABELS = ("Operational assigned pressure", "OCA / ACHP cloud-top pressure")
PDF_LABELS = {"seviri": ("Oper.", "OCA"), "goes": ("Oper.", "ACHP")}   # panel mode: (operational, cloud product)

# Sliding window.
P_CENTERS = np.arange(125.0, 975.1, 10.0)
WINDOW_HPA = 50.0                               # full width
MIN_SAMPLES = 200                               # fewer -> no point drawn
P_LIMITS = (1000.0, 100.0)                      # axis (bottom, top)

# Layout (mm).  Width ~88 mm so two cases fit side by side in a 180 mm row.
FIG_WIDTH_MM = 90.0                             # two cases side by side = 180 mm
FIG_HEIGHT_MM = 50.0
PANEL_FRACTIONS = (0.41, 0.34, 0.25)            # MVD | bias | PDF share of the axes width
LEFT_MM, RIGHT_MM, BOTTOM_MM, TOP_MM, GAP_MM = 10.5, 2.5, 7.5, 1.5, 2.6
MVD_LIMITS: tuple[float, float] | None = None   # None: auto
BIAS_LIMITS: tuple[float, float] | None = None  # None: symmetric auto

FONT_NAME = "Times New Roman"
FONT_PATH: str | None = None
FONT_SIZE_PT = 8.0

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "height_profiles"
PNG_DPI = 600
MM = 1.0 / 25.4

# =============================================================================
# Profiles
# =============================================================================


def sliding_profile(pressure: np.ndarray, vector_diff: np.ndarray, speed_bias: np.ndarray) -> pd.DataFrame:
    ok = np.isfinite(pressure) & np.isfinite(vector_diff) & np.isfinite(speed_bias)
    p = pressure[ok].astype(np.float64)
    order = np.argsort(p)
    p = p[order]
    vd = vector_diff[ok].astype(np.float64)[order]
    sb = speed_bias[ok].astype(np.float64)[order]
    csum = lambda a: np.concatenate([[0.0], np.cumsum(a)])  # noqa: E731
    s_vd, s_vd2, s_sb, s_sb2 = csum(vd), csum(vd * vd), csum(sb), csum(sb * sb)
    lo = np.searchsorted(p, P_CENTERS - WINDOW_HPA / 2, side="left")
    hi = np.searchsorted(p, P_CENTERS + WINDOW_HPA / 2, side="left")
    n = (hi - lo).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mvd = (s_vd[hi] - s_vd[lo]) / n
        bias = (s_sb[hi] - s_sb[lo]) / n
        sd_vd = np.sqrt(np.maximum((s_vd2[hi] - s_vd2[lo]) / n - mvd**2, 0.0))
        sd_sb = np.sqrt(np.maximum((s_sb2[hi] - s_sb2[lo]) / n - bias**2, 0.0))
        density = n / (len(p) * WINDOW_HPA)                    # PDF per hPa
    frame = pd.DataFrame({
        "pressure_center_hpa": P_CENTERS, "n": n.astype(int),
        "mvd": mvd, "mvd_ci95": 1.96 * sd_vd / np.sqrt(n),
        "speed_bias": bias, "speed_bias_ci95": 1.96 * sd_sb / np.sqrt(n),
        "pressure_pdf": density,
    })
    thin = frame["n"] < MIN_SAMPLES
    frame.loc[thin, ["mvd", "mvd_ci95", "speed_bias", "speed_bias_ci95"]] = np.nan
    return frame


def diagnostics(table: pd.DataFrame, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mask is None:
        mask = np.ones(len(table), dtype=bool)
    t = table.loc[mask]
    vd = np.hypot(t["u_amv"] - t["u_era5"], t["v_amv"] - t["v_era5"]).to_numpy(np.float64)
    sb = (np.hypot(t["u_amv"], t["v_amv"]) - np.hypot(t["u_era5"], t["v_era5"])).to_numpy(np.float64)
    return t["pressure_hpa"].to_numpy(np.float64), vd, sb


# =============================================================================
# Tables (server, project functions)
# =============================================================================


def seviri_tables(case: dict[str, Any]) -> dict[str, Callable[[], pd.DataFrame]]:
    from data_io.official_amv import load_official_amv_file
    from data_io.product_discovery import find_official_amv_file, infer_amv_valid_time_from_path
    from validation.compare_era5 import (
        build_era5_comparison_masks,
        build_official_amv_vs_era5_table,
        build_pressure_amv_vs_era5_table,
    )

    out_dir = PROJECT_ROOT / "outputs" / case["case"]
    from runtime_options import read_case_inputs
    inputs = read_case_inputs(PROJECT_ROOT, case["case"])
    data_dir = Path(inputs.get("data_dir", PROJECT_ROOT / "data" / case["case"].removeprefix("test_")))
    era5_dir = out_dir / "comparison" / "amv_vs_era5" / "era5_collocated"
    valid_time = (pd.Timestamp(inputs["frame_times"][-1]).to_pydatetime() if inputs.get("frame_times")
                  else infer_amv_valid_time_from_path(data_dir, offset_minutes=45))

    def operational() -> pd.DataFrame:
        path = find_official_amv_file(data_dir / "eumetsat_amv" / "native", expected_time=valid_time,
                                      tolerance_minutes=90.0, verbose=False)
        table = build_official_amv_vs_era5_table(load_official_amv_file(path),
                                                 era5_dir / "official_amv_era5_collocated.npz")
        return table.loc[build_era5_comparison_masks(table)["all_valid"]]

    def tracker(name: str) -> Callable[[], pd.DataFrame]:
        def load() -> pd.DataFrame:
            product = out_dir / "pressure_assignment" / f"{name}_final_amv_oca_pressure.npz"
            table = build_pressure_amv_vs_era5_table(product, era5_dir / "of_dense_piv_oca_pressure_era5_collocated.npz",
                                                     join_mode="row_col_pressure", amv_source=name)
            return table.loc[build_era5_comparison_masks(table)["all_valid"]]
        load.path = out_dir / "pressure_assignment" / f"{name}_final_amv_oca_pressure.npz"  # type: ignore[attr-defined]
        return load

    loaders: dict[str, Callable[[], pd.DataFrame]] = {"operational": operational}
    for method in METHODS:
        if method["key"] == "operational":
            continue
        for name in method["trackers"]:
            candidate = tracker(name)
            if candidate.path.is_file():  # type: ignore[attr-defined]
                loaders[method["key"]] = candidate
                break
        else:
            print(f"  [{case['key']}] no product for {method['key']} (tried {method['trackers']})")
    return loaders


def goes_tables(case: dict[str, Any]) -> dict[str, Callable[[], pd.DataFrame]]:
    from validation.goes_existing_era5_cache import build_amv_vs_existing_goes_era5_cache, resolve_existing_pcf_cache

    sat = case["satellite"].upper()
    out_dir = PROJECT_ROOT / "outputs" / case["case"]

    def operational() -> pd.DataFrame:
        path = out_dir / "official_dmw_vs_era5" / "official_dmw_vs_era5_collocated.csv.gz"
        frame = pd.read_csv(path)

        def column(*names: str) -> str:
            for name in names:
                if name in frame.columns:
                    return name
            raise KeyError(f"{path.name}: none of {names} found; columns = {list(frame.columns)}")

        if "satellite" in frame.columns:
            frame = frame.loc[frame["satellite"].astype(str).str.upper() == sat]
        for flag in ("era5_valid_mask", "era5_collocated"):
            if flag in frame.columns:
                frame = frame.loc[frame[flag].astype(bool)]
                break
        return pd.DataFrame({
            "pressure_hpa": frame[column("pressure_hpa", "pressure")],
            "u_amv": frame[column("u_amv", "u_ms", "u")],
            "v_amv": frame[column("v_amv", "v_ms", "v")],
            "u_era5": frame[column("u_era5", "era5_u_ms", "u_era5_ms")],
            "v_era5": frame[column("v_era5", "era5_v_ms", "v_era5_ms")],
        }).dropna()

    cache = resolve_existing_pcf_cache(PROJECT_ROOT, case["case"], sat, GOES_GEOMETRY)

    def tracker(name: str) -> Callable[[], pd.DataFrame]:
        product = out_dir / "pressure_assignment" / f"{name}_{sat.lower()}_{GOES_GEOMETRY}_final_amv_goes_pressure.npz"

        def load() -> pd.DataFrame:
            table, _ = build_amv_vs_existing_goes_era5_cache(product, cache, amv_source=name)
            ok = table["era5_valid_mask"].astype(bool) if "era5_valid_mask" in table else np.ones(len(table), bool)
            return table.loc[ok & np.isfinite(table["u_era5"]) & np.isfinite(table["u_amv"])]
        load.path = product  # type: ignore[attr-defined]
        return load

    loaders: dict[str, Callable[[], pd.DataFrame]] = {"operational": operational}
    for method in METHODS:
        if method["key"] == "operational":
            continue
        for name in method["trackers"]:
            candidate = tracker(name)
            if candidate.path.is_file():  # type: ignore[attr-defined]
                loaders[method["key"]] = candidate
                break
        else:
            print(f"  [{case['key']}] no product for {method['key']} (tried {method['trackers']})")
    return loaders


def compute_case(case: dict[str, Any], loaders: dict[str, Callable[[], pd.DataFrame]], out_dir: Path) -> None:
    profiles, overall = [], []
    for key, load in loaders.items():
        t = time.perf_counter()
        table = load()
        p, vd, sb = diagnostics(table)
        prof = sliding_profile(p, vd, sb)
        prof.insert(0, "method", key)
        profiles.append(prof)
        overall.append({"method": key, "n": int(np.isfinite(vd).sum()), "mvd": float(np.nanmean(vd)),
                        "rmsvd": float(np.sqrt(np.nanmean(vd**2))), "speed_bias": float(np.nanmean(sb))})
        print(f"  [{case['key']}] {key:13s} N={overall[-1]['n']:>11,}  MVD={overall[-1]['mvd']:.3f}  "
              f"bias={overall[-1]['speed_bias']:+.3f}  ({time.perf_counter() - t:.1f} s)")
        del table
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(profiles, ignore_index=True).to_csv(out_dir / f"profiles_{case['key']}.csv", index=False)
    pd.DataFrame(overall).to_csv(out_dir / f"overall_{case['key']}.csv", index=False)


# =============================================================================
# Plotting
# =============================================================================


def configure_fonts() -> str:
    name = FONT_NAME
    if FONT_PATH:
        font_manager.fontManager.addfont(FONT_PATH)
        name = font_manager.FontProperties(fname=FONT_PATH).get_name()
    else:
        try:
            font_manager.findfont(font_manager.FontProperties(family=FONT_NAME), fallback_to_default=False)
        except ValueError:
            for fallback in ("Liberation Serif", "Nimbus Roman", "DejaVu Serif"):
                try:
                    font_manager.findfont(font_manager.FontProperties(family=fallback), fallback_to_default=False)
                    name = fallback
                    break
                except ValueError:
                    continue
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [name, "Liberation Serif", "DejaVu Serif"],
        "font.size": FONT_SIZE_PT, "mathtext.fontset": "stix",
        "axes.linewidth": 0.5, "axes.edgecolor": "#333333",
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "xtick.labelsize": FONT_SIZE_PT, "ytick.labelsize": FONT_SIZE_PT,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return name


def method_style(key: str) -> dict[str, Any]:
    return next(m for m in METHODS if m["key"] == key)


def line(ax, x, y, style) -> None:
    kw = dict(color=style["color"], lw=style["lw"], solid_capstyle="round", dash_capstyle="round",
              zorder=4 if style["key"] == "lcflow" else 3)
    if style["dash"]:
        kw["dashes"] = style["dash"]
    ax.plot(x, y, **kw)


def plot_case(case: dict[str, Any], profiles: pd.DataFrame, out_dir: Path) -> None:
    fig = plt.figure(figsize=(FIG_WIDTH_MM * MM, FIG_HEIGHT_MM * MM))
    width = FIG_WIDTH_MM - LEFT_MM - RIGHT_MM - 2 * GAP_MM
    height = FIG_HEIGHT_MM - BOTTOM_MM - TOP_MM
    x = LEFT_MM
    axes = []
    for frac in PANEL_FRACTIONS:
        w = width * frac
        axes.append(fig.add_axes([x / FIG_WIDTH_MM, BOTTOM_MM / FIG_HEIGHT_MM, w / FIG_WIDTH_MM, height / FIG_HEIGHT_MM]))
        x += w + GAP_MM
    ax_mvd, ax_bias, ax_pdf = axes
    present = set(profiles["method"])
    draw_order = [m["key"] for m in METHODS if m["key"] in present and m["key"] != "lcflow"]
    if "lcflow" in present:
        draw_order.append("lcflow")
    for key in draw_order:
        prof = profiles.loc[profiles["method"] == key]
        style = method_style(key)
        y = prof["pressure_center_hpa"].to_numpy()
        line(ax_mvd, prof["mvd"].to_numpy(), y, style)
        line(ax_bias, prof["speed_bias"].to_numpy(), y, style)
    # PDF panel: operational pressure (grey area) and cloud-product pressure at the
    # dense retrieval points (OCA / ACHP, brown line).
    from matplotlib.patches import Patch

    op_label, cloud_label = PDF_LABELS.get(case.get("kind", "seviri"), ("Operational", "Cloud product"))
    handles = []
    for key in PDF_METHODS:
        if key not in present:
            continue
        prof = profiles.loc[profiles["method"] == key]
        y = prof["pressure_center_hpa"].to_numpy()
        pdf = prof["pressure_pdf"].to_numpy() * 100.0              # % per hPa
        if key == "operational":
            color, alpha, label = PDF_OPERATIONAL_COLOR, PDF_OPERATIONAL_ALPHA, op_label
        else:
            color, alpha, label = PDF_CLOUD_COLOR, PDF_CLOUD_ALPHA, cloud_label
        ax_pdf.fill_betweenx(y, 0, pdf, color=color, alpha=alpha, lw=0, zorder=1)
        ax_pdf.plot(pdf, y, color=color, lw=0.5, zorder=2)
        handles.append(Patch(facecolor=color, alpha=alpha, edgecolor=color, lw=0.5, label=label))
    if PDF_LEGEND == "panel" and handles:
        ax_pdf.legend(handles=handles, loc="center right", frameon=False, handlelength=1.0, handleheight=0.7,
                      handletextpad=0.3, borderaxespad=0.15, labelspacing=0.3)

    ax_bias.axvline(0.0, color="#9a9a9a", lw=0.5, zorder=1)
    for ax in axes:
        ax.set_ylim(*P_LIMITS)
        ax.set_yticks(np.arange(200, 1001, 200))
        ax.yaxis.set_minor_locator(MultipleLocator(100))
        ax.grid(axis="y", which="major", color="#ececec", lw=0.4, zorder=0)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.tick_params(which="minor", length=1.2, width=0.4)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color("#333333")
    ax_mvd.set_ylabel("Pressure (hPa)", labelpad=1.5)
    ax_bias.tick_params(labelleft=False)
    ax_pdf.tick_params(labelleft=False)
    ax_mvd.set_xlabel("MVD (m s$^{-1}$)", labelpad=1.5)
    ax_bias.set_xlabel("Speed bias (m s$^{-1}$)", labelpad=1.5)
    ax_pdf.set_xlabel("PDF (% hPa$^{-1}$)", labelpad=1.5)
    if MVD_LIMITS:
        ax_mvd.set_xlim(*MVD_LIMITS)
    else:
        ax_mvd.set_xlim(0, np.nanmax(profiles["mvd"]) * 1.08)
    if BIAS_LIMITS:
        ax_bias.set_xlim(*BIAS_LIMITS)
    else:
        b = np.nanmax(np.abs(profiles["speed_bias"])) * 1.12
        ax_bias.set_xlim(-b, b)
    pdf_max = np.nanmax(profiles.loc[profiles["method"].isin(PDF_METHODS), "pressure_pdf"]) * 100.0
    ax_pdf.set_xlim(0, pdf_max * 1.1)
    ax_pdf.xaxis.set_major_locator(plt.MaxNLocator(2))
    ax_pdf.xaxis.set_major_formatter(plt.FormatStrFormatter("%g"))
    ax_mvd.xaxis.set_major_locator(plt.MaxNLocator(4, prune="lower"))   # no "0" next to the "1000" label
    ax_bias.xaxis.set_major_locator(plt.MaxNLocator(4))
    for ax in (ax_mvd, ax_bias):
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%g"))
    save(fig, out_dir / f"height_profile_{case['key']}")


def plot_legend(out_dir: Path) -> None:
    from matplotlib.patches import Patch

    shared_pdf = PDF_LEGEND == "shared"
    fig = plt.figure(figsize=(FIG_WIDTH_MM * 2 * MM, (9.0 if shared_pdf else 5.0) * MM))
    handles = []
    for m in METHODS:
        kw = dict(color=m["color"], lw=m["lw"])
        if m["dash"]:
            kw["dashes"] = m["dash"]
        handles.append(Line2D([], [], label=m["label"], **kw))
    common = dict(frameon=False, handlelength=2.6, columnspacing=1.3, handletextpad=0.5, borderaxespad=0.0)
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.72 if shared_pdf else 0.5),
               ncol=len(handles), **common)
    if shared_pdf:
        patches = [
            Patch(facecolor=PDF_OPERATIONAL_COLOR, alpha=PDF_OPERATIONAL_ALPHA, edgecolor=PDF_OPERATIONAL_COLOR,
                  lw=0.5, label=f"{PDF_SHARED_LABELS[0]} (PDF)"),
            Patch(facecolor=PDF_CLOUD_COLOR, alpha=PDF_CLOUD_ALPHA, edgecolor=PDF_CLOUD_COLOR,
                  lw=0.5, label=f"{PDF_SHARED_LABELS[1]} (PDF)"),
        ]
        fig.legend(handles=patches, loc="center", bbox_to_anchor=(0.5, 0.25), ncol=2,
                   **{**common, "handlelength": 1.6, "columnspacing": 2.5})
    save(fig, out_dir / "height_profile_legend")


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=PNG_DPI)
    plt.close(fig)
    print("  wrote", stem.with_suffix(".pdf").name, "+ .png")


def main(argv: Sequence[str]) -> int:
    stage = argv[1].lower() if len(argv) > 1 else "all"
    only = argv[2].lower() if len(argv) > 2 else ""          # e.g. "goes" or "seviri_20180112"
    if stage in {"compute", "all"}:
        for case in CASES:
            if only and only not in case["key"]:
                continue
            print(f"\n[{case['key']}] {case['label']}")
            loaders = seviri_tables(case) if case["kind"] == "seviri" else goes_tables(case)
            compute_case(case, loaders, OUTPUT_DIR)
    if stage in {"plot", "all"}:
        print("Font:", configure_fonts())
        for case in CASES:
            path = OUTPUT_DIR / f"profiles_{case['key']}.csv"
            if not path.is_file():
                print("  missing", path.name, "- run 'compute' first")
                continue
            plot_case(case, pd.read_csv(path), OUTPUT_DIR)
        plot_legend(OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
