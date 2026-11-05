"""Full-disk qualitative wind-field figure: IR image | operational | LayerFlow.

Two rows (SEVIRI, GOES) x three columns, drawn in the satellite's own
geostationary view (no resampling of the IR image):

    (a) SEVIRI IR10.8 [BT bar]   (b) operational AMV   (c) LayerFlow  [pressure bar]
    (d) GOES-16 C14   [BT bar]   (e) operational DMW   (f) LayerFlow  [pressure bar]

Vectors are coloured by cloud-top pressure and/or wind speed (COLOR_MODES);
one figure set per mode.  Each row is saved as its own figure (ROW_WIDTH_MM wide, default 180 mm) and the
two rows are also stacked into one 2 x 3 figure.  Insert at 100 % size: all
text is FONT_SIZE_PT.

Run from the project root:

    python scripts/plot_wind_fields.py
    python scripts/plot_wind_fields.py --cmap turbo

Notes
  * Dense vector layers are rasterized inside the PDF (RASTER_DPI) so the PDF
    stays small; text, outlines and colorbars stay vector.
  * LayerFlow is thinned on its native pixel grid (row % s == 0 and col % s == 0)
    for display only.
  * Vector positions use image navigation: SEVIRI product lon/lat, GOES LayerFlow
    row/col (its cloud_height lon/lat are parallax-corrected and would not sit
    on the clouds in the IR image), DMW / EUMETSAT AMV lon/lat.
  * Coastlines: Natural Earth 1:50m shapefile at COASTLINE_FILE (read with
    pyshp, which cartopy installs).  Skipped if absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colorbar as mcolorbar  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import BoundaryNorm, Normalize  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# =============================================================================
# User settings
# =============================================================================
SEVIRI_CASE = "test_2018-01-12T11"
SEVIRI_VALID_OFFSET_MINUTES = 45
SEVIRI_LCFLOW_PRODUCT = "pressure_conditioned_flow_final_amv_oca_pressure.npz"

GOES_CASE = "goes_g16_g18_E20240804T120506_W20240804T120508"
GOES_SATELLITE = "G16"
GOES_GEOMETRY = "cloud_height"
GOES_DMW_REQUIRE_DQF_ZERO = True
GOES_DMW_PRESSURE_RANGE = (100.0, 1000.0)

# LayerFlow display thinning on the native grid.  10 px = 30 km for SEVIRI;
# 15 px = 30 km for GOES ABI (2 km), i.e. the same ground spacing.
LCFLOW_STRIDE_PX = {"SEVIRI": 10, "GOES": 15}

# Pressure colouring.
PRESSURE_CMAP_CHOICES = {
    "jet": "dark blue (high cloud) -> dark red (low cloud); default",
    "turbo": "improved jet, smoother cyan-green band",
    "rainbow": "violet -> red, lighter ends",
    "nipy_spectral": "most hue changes; 1000 hPa end near white",
    "gist_rainbow_r": "saturated rainbow without dark ends",
    "Spectral_r": "diverging blue -> yellow -> red",
    "RdYlBu_r": "diverging blue -> white -> red",
    "coolwarm": "diverging blue -> grey -> red",
    "viridis": "perceptually uniform",
    "plasma": "perceptually uniform",
    "cividis": "perceptually uniform, CVD safe",
}
PRESSURE_CMAP = "jet"
PRESSURE_REVERSE = True                   # True: high cloud red, low cloud blue
PRESSURE_RANGE = (100.0, 1000.0)
PRESSURE_LEVELS = 255
PRESSURE_TICKS = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
PRESSURE_LABEL = "Pressure (hPa)"

# Vector colour: one figure set is written per entry ("pressure" and/or "speed").
COLOR_MODES = ("speed",)  # Wind-speed colouring
SPEED_CMAP = "jet"
SPEED_MAX_MS = 50.0               # upper colour limit; faster winds keep the top colour (~99th percentile)
SPEED_LEVELS = 255
SPEED_TICKS = (0, 10, 20, 30, 40, 50)
SPEED_LABEL = "Wind speed (m s$^{-1}$)"

# IR image: cold = black.
BT_CMAP = "gray"
BT_RANGE = (190.0, 310.0)
BT_TICKS = (200, 220, 240, 260, 280, 300)
BT_LABEL = "Brightness temperature (K)"

# Arrows.  At these densities arrows overlap into streaks (the intended "dense
# field" look); one fixed physical scale for all vector panels.
UNIT_SPEED_MS = 30.0
UNIT_LENGTH_MM = 1.6
ARROW_WIDTH_MM = 0.07
ARROW_HEAD = 3.0                  # head width/length in shaft widths; 0 = plain line segments
OPERATIONAL_ARROW_WIDTH_MM = 0.07

# Layout (mm).  One row: [lat labels] IR disk [BT bar] disk disk [pressure bar].
ROW_WIDTH_MM = 180.0
DISK_GAP_MM = 1.5
CBAR_THICKNESS_MM = 2.6
GEO_LABEL_BLOCK_MM = 6.0          # room for latitude labels left of the IR disk
CBAR_BLOCK_BT_MM = 13.0           # BT bar + ticks + label, right of the IR disk
CBAR_BLOCK_P_MM = 14.5            # pressure bar + ticks ("1000") + label
CBAR_LENGTH_FRACTION = 0.82       # of the disk diameter
ROW_PAD_MM = 0.8                  # above and below the disks
ROW_GAP_MM = 2.0                  # between rows in the stacked figure

FONT_NAME = "Times New Roman"
FONT_PATH: str | None = None
FONT_SIZE_PT = 8.0

DISK_EDGE_COLOR = "#000000"
DISK_EDGE_PT = 0.6
GRATICULE_STEP_DEG = 30.0
GRATICULE_COLOR = "#7a7a7a"
GRATICULE_PT = 0.35
COASTLINE_FILE = PROJECT_ROOT / "data" / "natural_earth" / "ne_50m_coastline.shp"
COASTLINE_COLOR = "#303030"
COASTLINE_PT = 0.35
COASTLINE_ON_IMAGE_COLOR = "#d9d9d9"   # coastlines over the (dark) IR image
PANEL_LABELS = False                   # "(a)".."(f)" in the corner
# Geographic labels on the IR disk only: latitudes outside its left limb,
# longitudes along the equator inside the disk.  "image", "none".
GEO_LABELS = "image"
GEO_LABEL_LATS = (-60, -30, 0, 30, 60)
GEO_LABEL_LON_OFFSETS = (-30, 0, 30)            # relative to the sub-satellite longitude

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "windfield_disk"
PNG_DPI = 600
RASTER_DPI = 600

MM = 1.0 / 25.4

# =============================================================================
# Geostationary projection (PROJ "geos": x, y = scan angle * satellite height)
# =============================================================================


@dataclass(frozen=True)
class Geos:
    sub_lon: float
    height: float          # satellite height above the ellipsoid (m)
    a: float
    b: float
    sweep: str             # "y" for SEVIRI, "x" for GOES ABI

    def forward(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lon = np.radians(np.asarray(lon, dtype=np.float64) - self.sub_lon)
        lat = np.radians(np.asarray(lat, dtype=np.float64))
        e2 = 1.0 - (self.b / self.a) ** 2
        big_h = self.height + self.a
        lat_c = np.arctan((self.b / self.a) ** 2 * np.tan(lat))
        r_c = self.b / np.sqrt(1.0 - e2 * np.cos(lat_c) ** 2)
        sx = big_h - r_c * np.cos(lat_c) * np.cos(lon)
        sy = r_c * np.cos(lat_c) * np.sin(lon)
        sz = r_c * np.sin(lat_c)
        visible = big_h * (big_h - sx) >= sy**2 + sz**2 * (self.a / self.b) ** 2
        norm = np.sqrt(sx**2 + sy**2 + sz**2)
        if self.sweep == "x":
            x, y = np.arcsin(sy / norm), np.arctan(sz / sx)
        else:
            x, y = np.arctan(sy / sx), np.arcsin(sz / norm)
        x, y = x * self.height, y * self.height
        x[~visible] = np.nan
        y[~visible] = np.nan
        return x, y

    @property
    def limb_radii(self) -> tuple[float, float]:
        big_h = self.height + self.a
        return np.arcsin(self.a / big_h) * self.height, np.arcsin(self.b / big_h) * self.height


SEVIRI_GEOS = Geos(0.0, 35785831.0, 6378169.0, 6356583.8, "y")
GOES_GEOS = {
    "G16": Geos(-75.0, 35786023.0, 6378137.0, 6356752.31414, "x"),
    "G18": Geos(-137.0, 35786023.0, 6378137.0, 6356752.31414, "x"),
}


def project_vectors(geos: Geos, lon, lat, u, v, x=None, y=None):
    """Positions and wind components in projection coordinates.

    Direction from a 60 s displacement on the sphere, length = wind speed, so
    that quiver(angles="uv") on an equal-aspect axis draws the true direction.
    """

    lon, lat, u, v = (np.asarray(a, dtype=np.float64) for a in (lon, lat, u, v))
    if x is None:
        x, y = geos.forward(lon, lat)
    dt = 60.0
    lat2 = lat + np.degrees(v * dt / 6371000.0)
    lon2 = lon + np.degrees(u * dt / (6371000.0 * np.cos(np.radians(lat))))
    x1, y1 = geos.forward(lon, lat)
    x2, y2 = geos.forward(lon2, lat2)
    dx, dy = x2 - x1, y2 - y1
    norm = np.hypot(dx, dy)
    speed = np.hypot(u, v)
    with np.errstate(invalid="ignore", divide="ignore"):
        U, V = dx / norm * speed, dy / norm * speed
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(U) & np.isfinite(V)
    return ok, np.asarray(x), np.asarray(y), U, V


# =============================================================================
# Drawing
# =============================================================================


@dataclass
class Image:
    bt: np.ndarray
    extent: tuple[float, float, float, float]     # left, right, bottom, top (projection m)


@dataclass
class Vectors:
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    pressure: np.ndarray

    def __len__(self) -> int:
        return int(self.x.size)


@dataclass
class RowData:
    label: str
    geos: Geos
    image: Image
    operational: Vectors
    lcflow: Vectors
    info: dict[str, Any]


def configure_fonts() -> str:
    name = FONT_NAME
    if FONT_PATH:
        font_manager.fontManager.addfont(FONT_PATH)
        name = font_manager.FontProperties(fname=FONT_PATH).get_name()
    else:
        try:
            font_manager.findfont(font_manager.FontProperties(family=FONT_NAME), fallback_to_default=False)
        except ValueError:
            for fallback in ("Liberation Serif", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"):
                try:
                    font_manager.findfont(font_manager.FontProperties(family=fallback), fallback_to_default=False)
                    name = fallback
                    break
                except ValueError:
                    continue
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [name, "Liberation Serif", "DejaVu Serif"],
        "font.size": FONT_SIZE_PT,
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.5,
        "xtick.labelsize": FONT_SIZE_PT,
        "ytick.labelsize": FONT_SIZE_PT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": RASTER_DPI,
    })
    return name


COLOR_MODE = COLOR_MODES[0]       # current mode while rendering


def speed_cmap_norm():
    cmap = plt.get_cmap(SPEED_CMAP).resampled(SPEED_LEVELS)
    cmap.set_over(cmap(SPEED_LEVELS - 1))
    bounds = np.linspace(0.0, SPEED_MAX_MS, SPEED_LEVELS + 1)
    return cmap, BoundaryNorm(bounds, SPEED_LEVELS)


def vector_colour_spec() -> dict[str, Any]:
    if COLOR_MODE == "speed":
        cmap, norm = speed_cmap_norm()
        return dict(cmap=cmap, norm=norm, ticks=SPEED_TICKS, label=SPEED_LABEL, invert=False, extend="max")
    cmap, norm = pressure_cmap_norm()
    return dict(cmap=cmap, norm=norm, ticks=PRESSURE_TICKS, label=PRESSURE_LABEL, invert=True, extend="both")


def pressure_cmap_norm():
    cmap = plt.get_cmap(PRESSURE_CMAP)
    if PRESSURE_REVERSE:
        cmap = cmap.reversed()
    cmap = cmap.resampled(PRESSURE_LEVELS)
    cmap.set_under(cmap(0))
    cmap.set_over(cmap(PRESSURE_LEVELS - 1))
    bounds = np.linspace(PRESSURE_RANGE[0], PRESSURE_RANGE[1], PRESSURE_LEVELS + 1)
    return cmap, BoundaryNorm(bounds, PRESSURE_LEVELS)


_COAST: list[np.ndarray] | None = None


def coastlines_lonlat() -> list[np.ndarray]:
    global _COAST
    if _COAST is not None:
        return _COAST
    _COAST = []
    path = Path(COASTLINE_FILE)
    if not path.is_file():
        print(f"[coastlines] skipped: {path} not found")
        return _COAST
    try:
        import shapefile  # pyshp (installed with cartopy)

        for shape in shapefile.Reader(str(path)).shapes():
            points = np.asarray(shape.points, dtype=np.float64)
            starts = list(shape.parts) + [len(points)]
            _COAST.extend(points[a:b] for a, b in zip(starts[:-1], starts[1:]) if b - a > 1)
        print(f"[coastlines] {len(_COAST):,} lines from {path.name}")
    except Exception as exc:  # noqa: BLE001
        print(f"[coastlines] skipped ({type(exc).__name__}: {exc})")
    return _COAST


def projected_lines(geos: Geos, lines: list[np.ndarray]) -> list[np.ndarray]:
    out = []
    for line in lines:
        x, y = geos.forward(line[:, 0], line[:, 1])
        ok = np.isfinite(x)
        if ok.sum() < 2:
            continue
        breaks = np.nonzero(np.diff(ok.astype(np.int8)) != 0)[0] + 1
        for seg_x, seg_y, seg_ok in zip(np.split(x, breaks), np.split(y, breaks), np.split(ok, breaks)):
            if seg_ok[0] and seg_x.size > 1:
                out.append(np.column_stack([seg_x, seg_y]))
    return out


def graticule(step: float) -> list[np.ndarray]:
    lines = []
    lat = np.linspace(-89.0, 89.0, 400)
    for lon in np.arange(-180.0, 180.0, step):
        lines.append(np.column_stack([np.full_like(lat, lon), lat]))
    lon = np.linspace(-180.0, 180.0, 800)
    for la in np.arange(-90.0 + step, 90.0, step):
        lines.append(np.column_stack([lon, np.full_like(lon, la)]))
    return lines


def disk_axes(fig: plt.Figure, rect: list[float], geos: Geos, facecolor: str) -> plt.Axes:
    ax = fig.add_axes(rect)
    rx, ry = geos.limb_radii
    pad = 1.01
    ax.set_xlim(-rx * pad, rx * pad)
    ax.set_ylim(-ry * pad, ry * pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    disk = Ellipse((0, 0), 2 * rx, 2 * ry, facecolor=facecolor, edgecolor="none", zorder=0)
    ax.add_patch(disk)
    ax._limb = Ellipse((0, 0), 2 * rx, 2 * ry, facecolor="none", edgecolor=DISK_EDGE_COLOR,
                       linewidth=DISK_EDGE_PT, zorder=10)
    ax.add_patch(ax._limb)
    return ax


def decorate(ax: plt.Axes, geos: Geos, on_image: bool) -> None:
    grat = LineCollection(projected_lines(geos, graticule(GRATICULE_STEP_DEG)),
                          colors=GRATICULE_COLOR, linewidths=GRATICULE_PT, linestyles=(0, (1.0, 1.5)), zorder=4)
    grat.set_clip_path(ax._limb)
    ax.add_collection(grat)
    coast = projected_lines(geos, coastlines_lonlat())
    if coast:
        lc = LineCollection(coast, colors=COASTLINE_ON_IMAGE_COLOR if on_image else COASTLINE_COLOR,
                            linewidths=COASTLINE_PT, zorder=5)
        lc.set_clip_path(ax._limb)
        ax.add_collection(lc)


def draw_image(ax: plt.Axes, image: Image) -> None:
    im = ax.imshow(image.bt, extent=image.extent, origin="upper", cmap=BT_CMAP,
                   norm=Normalize(*BT_RANGE), interpolation="nearest", zorder=1)
    im.set_clip_path(ax._limb)


def draw_vectors(ax: plt.Axes, vectors: Vectors, width_mm: float) -> None:
    if not len(vectors):
        return
    speed = np.hypot(vectors.u, vectors.v)
    if COLOR_MODE == "speed":
        value = np.minimum(speed, SPEED_MAX_MS * 1.001)
        order = np.argsort(speed, kind="stable")                # fast winds on top
    else:
        value = np.clip(vectors.pressure, *PRESSURE_RANGE)
        order = np.argsort(-vectors.pressure, kind="stable")    # high clouds on top
    spec = vector_colour_spec()
    cmap, norm = spec["cmap"], spec["norm"]
    head = dict(headwidth=ARROW_HEAD, headlength=ARROW_HEAD * 1.2, headaxislength=ARROW_HEAD * 1.1) \
        if ARROW_HEAD > 0 else dict(headwidth=1.0, headlength=0.0, headaxislength=0.0)
    q = ax.quiver(vectors.x[order], vectors.y[order], vectors.u[order], vectors.v[order],
                  value[order], cmap=cmap, norm=norm,
                  angles="uv", scale_units="inches", scale=UNIT_SPEED_MS / (UNIT_LENGTH_MM * MM),
                  units="inches", width=width_mm * MM, minshaft=1, minlength=0, pivot="middle",
                  linewidth=0, zorder=3, **head)
    q.set_rasterized(True)
    q.set_clip_path(ax._limb)


def draw_colorbar(fig, rect, cmap, norm, ticks, label, side: str, invert: bool, extend: str) -> None:
    cax = fig.add_axes(rect)
    bar = mcolorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical", extend=extend,
                                 extendfrac=0.03, ticks=list(ticks), spacing="proportional")
    bar.outline.set_linewidth(0.5)
    cax.yaxis.set_ticks_position(side)
    cax.yaxis.set_label_position(side)
    cax.tick_params(width=0.5, length=2.0, pad=1.5, labelsize=FONT_SIZE_PT)
    cax.minorticks_off()
    bar.set_label(label, labelpad=1.5)
    bar.solids.set_rasterized(True)
    bar.solids.set_edgecolor("face")
    if invert:
        cax.invert_yaxis()


def row_geometry() -> dict[str, float]:
    label = GEO_LABEL_BLOCK_MM if GEO_LABELS != "none" else 0.0
    disk = (ROW_WIDTH_MM - label - CBAR_BLOCK_BT_MM - CBAR_BLOCK_P_MM - 4 * DISK_GAP_MM) / 3.0
    return {"disk": disk, "height": disk + 2 * ROW_PAD_MM}


def lon_text(lon: float) -> str:
    lon = ((lon + 180.0) % 360.0) - 180.0
    if abs(lon) < 1e-6 or abs(abs(lon) - 180.0) < 1e-6:
        return f"{abs(lon):g}°"
    return f"{abs(lon):g}°{'E' if lon > 0 else 'W'}"


def lat_text(lat: float) -> str:
    return "0°" if abs(lat) < 1e-6 else f"{abs(lat):g}°{'N' if lat > 0 else 'S'}"


def geo_labels(ax: plt.Axes, geos: Geos) -> None:
    import matplotlib.patheffects as pe

    rx, _ = geos.limb_radii
    gap = 0.02 * rx
    lon = np.linspace(geos.sub_lon - 89.9, geos.sub_lon + 89.9, 2000)
    for lat in GEO_LABEL_LATS:
        x, y = geos.forward(lon, np.full_like(lon, float(lat)))
        if not np.isfinite(x).any():
            continue
        k = int(np.nanargmin(x))
        ax.text(x[k] - gap, y[k], lat_text(lat), ha="right", va="center", clip_on=False)
    halo = [pe.withStroke(linewidth=1.6, foreground="white")]
    for offset in GEO_LABEL_LON_OFFSETS:
        lo = geos.sub_lon + offset
        x, y = geos.forward(np.array([lo]), np.array([0.0]))
        if np.isfinite(x[0]):
            ax.text(x[0] + gap, y[0] - gap, lon_text(lo), ha="left", va="top", color="black",
                    path_effects=halo, zorder=12)


def draw_row(fig: plt.Figure, row: RowData, y0_mm: float, fig_w: float, fig_h: float, labels: str) -> None:
    g = row_geometry()
    d = g["disk"]
    bottom = (y0_mm + ROW_PAD_MM) / fig_h
    length = d * CBAR_LENGTH_FRACTION
    cb_bottom = (y0_mm + ROW_PAD_MM + 0.5 * (d - length)) / fig_h

    x = GEO_LABEL_BLOCK_MM if GEO_LABELS != "none" else 0.0
    axes = [disk_axes(fig, [x / fig_w, bottom, d / fig_w, d / fig_h], row.geos, "black")]
    x += d + DISK_GAP_MM
    draw_colorbar(fig, [x / fig_w, cb_bottom, CBAR_THICKNESS_MM / fig_w, length / fig_h],
                  plt.get_cmap(BT_CMAP), Normalize(*BT_RANGE), BT_TICKS, BT_LABEL,
                  side="right", invert=False, extend="both")
    x += CBAR_BLOCK_BT_MM + DISK_GAP_MM
    for _ in range(2):
        axes.append(disk_axes(fig, [x / fig_w, bottom, d / fig_w, d / fig_h], row.geos, "white"))
        x += d + DISK_GAP_MM
    spec = vector_colour_spec()
    draw_colorbar(fig, [x / fig_w, cb_bottom, CBAR_THICKNESS_MM / fig_w, length / fig_h],
                  spec["cmap"], spec["norm"], spec["ticks"], spec["label"], side="right",
                  invert=spec["invert"], extend=spec["extend"])

    draw_image(axes[0], row.image)
    decorate(axes[0], row.geos, on_image=True)
    if GEO_LABELS == "image":
        geo_labels(axes[0], row.geos)
    for ax, vec, w in ((axes[1], row.operational, OPERATIONAL_ARROW_WIDTH_MM), (axes[2], row.lcflow, ARROW_WIDTH_MM)):
        decorate(ax, row.geos, on_image=False)
        draw_vectors(ax, vec, w)
    if PANEL_LABELS:
        for ax, lab in zip(axes, labels):
            ax.text(0.0, 1.0, f"({lab})", transform=ax.transAxes, ha="left", va="top")


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), dpi=RASTER_DPI)
    fig.savefig(stem.with_suffix(".png"), dpi=PNG_DPI)
    plt.close(fig)
    print("  wrote", stem.with_suffix(".pdf").name, "+ .png")


def render(rows: list[RowData], out_dir: Path) -> None:
    global COLOR_MODE
    g = row_geometry()
    labels = ["abc", "def"]
    total_h = len(rows) * g["height"] + (len(rows) - 1) * ROW_GAP_MM
    for mode in COLOR_MODES:
        COLOR_MODE = mode
        for k, row in enumerate(rows):
            fig = plt.figure(figsize=(ROW_WIDTH_MM * MM, g["height"] * MM))
            draw_row(fig, row, 0.0, ROW_WIDTH_MM, g["height"], labels[k])
            save(fig, out_dir / f"windfield_disk_row_{row.label.lower()}_{mode}")
        fig = plt.figure(figsize=(ROW_WIDTH_MM * MM, total_h * MM))
        for k, row in enumerate(rows):
            y0 = total_h - (k + 1) * g["height"] - k * ROW_GAP_MM
            draw_row(fig, row, y0, ROW_WIDTH_MM, total_h, labels[k])
        save(fig, out_dir / f"windfield_disk_2x3_{mode}")
    print(f"  layout: disk diameter {g['disk']:.1f} mm, row {ROW_WIDTH_MM:.0f} x {g['height']:.1f} mm, "
          f"2x3 figure {ROW_WIDTH_MM:.0f} x {total_h:.1f} mm")


# =============================================================================
# Data (server, project loaders)
# =============================================================================


def lattice(rows: np.ndarray, cols: np.ndarray, stride: int) -> np.ndarray:
    r = np.rint(rows).astype(np.int64)
    c = np.rint(cols).astype(np.int64)
    return (r % stride == 0) & (c % stride == 0)


def image_from_lonlat_grid(bt: np.ndarray, lon: np.ndarray, lat: np.ndarray, geos: Geos) -> Image:
    """Projection extent of a full-disk grid from its own navigation (handles flips)."""

    n_rows, n_cols = bt.shape
    x, y = geos.forward(lon, lat)
    rr, cc = np.nonzero(np.isfinite(x))
    # Least-squares x = x0 + dx * col, y = y0 + dy * row.
    sel = slice(None, None, max(1, rr.size // 200000))
    dx, x0 = np.polyfit(cc[sel], x[rr, cc][sel], 1)
    dy, y0 = np.polyfit(rr[sel], y[rr, cc][sel], 1)
    left, right = x0 - 0.5 * dx, x0 + (n_cols - 0.5) * dx
    top, bottom = y0 - 0.5 * dy, y0 + (n_rows - 0.5) * dy
    return Image(bt, (left, right, bottom, top))


def load_seviri_row() -> RowData:
    from data_io.case_observations import load_case_observations
    from data_io.official_amv import load_official_amv_file
    from data_io.product_discovery import find_official_amv_file, infer_amv_valid_time_from_path
    from validation.compare_era5 import load_pressure_amv_npz, standardize_official_amv

    geos = SEVIRI_GEOS
    from runtime_options import read_case_inputs
    inputs = read_case_inputs(PROJECT_ROOT, SEVIRI_CASE)
    data_case_dir = Path(inputs.get("data_dir", PROJECT_ROOT / "data" / SEVIRI_CASE.removeprefix("test_")))
    valid_time = pd.Timestamp(inputs["frame_times"][-1] if inputs.get("frame_times") else
                              infer_amv_valid_time_from_path(data_case_dir, offset_minutes=SEVIRI_VALID_OFFSET_MINUTES))
    obs = load_case_observations(case_name=SEVIRI_CASE, data_case_dir=data_case_dir, frame_count=4,
                                 frame_times=inputs.get("frame_times") or None,
                                 interval_minutes=inputs.get("interval_minutes") or 15.0,
                                 aoi=inputs.get("aoi"), channel="IR10.8", load_oca_data=False)
    times = pd.to_datetime(list(obs.frame_times))
    frame = int(np.argmin(np.abs((times - valid_time).total_seconds())))
    bt = np.asarray(obs.images[frame], dtype=np.float32).copy()
    if getattr(obs, "valid_masks", None) is not None:
        bt[~np.asarray(obs.valid_masks[frame], dtype=bool)] = np.nan
    image = image_from_lonlat_grid(bt, np.asarray(obs.lon_grid), np.asarray(obs.lat_grid), geos)

    path = find_official_amv_file(data_case_dir / "eumetsat_amv" / "native",
                                  expected_time=valid_time.to_pydatetime(), tolerance_minutes=90.0)
    off = standardize_official_amv(load_official_amv_file(path))
    ok, x, y, U, V = project_vectors(geos, off["longitude"], off["latitude"], off["u_amv"], off["v_amv"])
    operational = Vectors(x[ok], y[ok], U[ok], V[ok], off["pressure_hpa"].to_numpy(np.float64)[ok])

    prod = load_pressure_amv_npz(PROJECT_ROOT / "outputs" / SEVIRI_CASE / "pressure_assignment" / SEVIRI_LCFLOW_PRODUCT)
    keep = lattice(prod["row_center_original"].to_numpy(), prod["col_center_original"].to_numpy(),
                   LCFLOW_STRIDE_PX["SEVIRI"])
    keep &= np.isfinite(prod[["longitude", "latitude", "u_amv", "v_amv", "pressure_hpa"]].to_numpy()).all(axis=1)
    p = prod.loc[keep]
    ok, x, y, U, V = project_vectors(geos, p["longitude"], p["latitude"], p["u_amv"], p["v_amv"])
    lcflow = Vectors(x[ok], y[ok], U[ok], V[ok], p["pressure_hpa"].to_numpy(np.float64)[ok])
    info = {"time": str(times[frame]), "operational": len(operational), "lcflow_all": len(prod),
            "lcflow_drawn": len(lcflow), "stride": LCFLOW_STRIDE_PX["SEVIRI"]}
    return RowData("SEVIRI", geos, image, operational, lcflow, info)


def load_goes_row() -> RowData:
    from data_io.goes import discover_goes_frames, frame_for, load_goes_frame
    from data_io.goes_dmw import discover_goes_dmw_files, dmw_basic_valid_mask, load_goes_dmw
    from preprocessing.goes_geolocation import (
        ecef_to_geodetic_lonlat_on_height_ellipsoid, fixed_grid_to_ecef, scan_angles_at_pixels)
    from validation.compare_era5 import load_pressure_amv_npz

    sat = GOES_SATELLITE.upper()
    case_dir = PROJECT_ROOT / "data" / GOES_CASE
    center = load_goes_frame(frame_for(discover_goes_frames(case_dir), sat, "center"), apply_quality_mask=False)
    attrs = center.projection_attrs
    geos = Geos(float(attrs.get("longitude_of_projection_origin", GOES_GEOS[sat].sub_lon)),
                float(attrs.get("perspective_point_height", 35786023.0)),
                float(attrs.get("semi_major_axis", 6378137.0)), float(attrs.get("semi_minor_axis", 6356752.31414)),
                str(attrs.get("sweep_angle_axis", "x")))
    xa = np.asarray(center.x_scan_angle, dtype=np.float64) * geos.height
    ya = np.asarray(center.y_scan_angle, dtype=np.float64) * geos.height
    dx, dy = xa[1] - xa[0], ya[1] - ya[0]
    image = Image(np.asarray(center.brightness_temperature_k, dtype=np.float32),
                  (xa[0] - dx / 2, xa[-1] + dx / 2, ya[-1] + dy / 2, ya[0] - dy / 2))

    dmw_file = discover_goes_dmw_files(case_dir, satellites=[sat])[0]
    raw = load_goes_dmw(dmw_file.path, satellite=sat)
    keep = dmw_basic_valid_mask(raw)
    if GOES_DMW_REQUIRE_DQF_ZERO:
        keep &= raw["dqf"].to_numpy() == 0
    keep &= raw["pressure_hpa"].between(*GOES_DMW_PRESSURE_RANGE).to_numpy()
    dmw = raw.loc[keep]
    ok, x, y, U, V = project_vectors(geos, dmw["longitude"], dmw["latitude"], dmw["u_ms"], dmw["v_ms"])
    operational = Vectors(x[ok], y[ok], U[ok], V[ok], dmw["pressure_hpa"].to_numpy(np.float64)[ok])

    name = f"pressure_conditioned_flow_{sat.lower()}_{GOES_GEOMETRY}_final_amv_goes_pressure.npz"
    prod = load_pressure_amv_npz(PROJECT_ROOT / "outputs" / GOES_CASE / "pressure_assignment" / name)
    rows = prod["row_center_original"].to_numpy(np.float64)
    cols = prod["col_center_original"].to_numpy(np.float64)
    keep = lattice(rows, cols, LCFLOW_STRIDE_PX["GOES"])
    keep &= np.isfinite(prod[["u_amv", "v_amv", "pressure_hpa"]].to_numpy()).all(axis=1)
    rows, cols, p = rows[keep], cols[keep], prod.loc[keep]
    sx, sy = scan_angles_at_pixels(center.x_scan_angle, center.y_scan_angle, rows, cols)
    ex, ey, ez, valid = fixed_grid_to_ecef(sx, sy, 0.0, attrs)
    lon, lat = ecef_to_geodetic_lonlat_on_height_ellipsoid(ex, ey, ez, 0.0, attrs)
    ok, x, y, U, V = project_vectors(geos, lon, lat, p["u_amv"], p["v_amv"], x=sx * geos.height, y=sy * geos.height)
    ok &= valid
    lcflow = Vectors(x[ok], y[ok], U[ok], V[ok], p["pressure_hpa"].to_numpy(np.float64)[ok])
    info = {"time": str(center.scan_midpoint), "operational": len(operational), "lcflow_all": len(prod),
            "lcflow_drawn": len(lcflow), "stride": LCFLOW_STRIDE_PX["GOES"]}
    return RowData("GOES", geos, image, operational, lcflow, info)


def main(argv: Sequence[str]) -> int:
    global PRESSURE_CMAP
    args = list(argv[1:])
    if "--cmap" in args:
        k = args.index("--cmap")
        PRESSURE_CMAP = args[k + 1]
    print("Font:", configure_fonts(), "| colormap:", PRESSURE_CMAP)
    rows = []
    for loader in (load_seviri_row, load_goes_row):
        t = time.perf_counter()
        row = loader()
        print(f"[{time.perf_counter() - t:5.1f} s] {row.label}: {row.info}")
        rows.append(row)
    t = time.perf_counter()
    render(rows, OUTPUT_DIR)
    print(f"[{time.perf_counter() - t:5.1f} s] rendered -> {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
