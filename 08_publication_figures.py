"""
08_publication_figures.py
-------------------------
Every figure used in the manuscript (300-dpi PNG + vector PDF), written to
analysis_out/figures/pub/.

Maps (state-route network from the HSIS geodatabase, EPSG 2927)
  map1_study_area             rural/urban mainline, KA crashes, regions, counties, relief
  map2_county_choropleths     (a) KA rate per 100M VMT, (b) observed/expected under model E
  map3_network_risk           empirical Bayes KA rate on mainline segments
  map4_rural_curves_hotspots  rural curves by radius, EB top 10, four close-ups
  map5_residuals_lisa         (a) SPF Pearson residuals, (b) local Moran clusters

Statistical figures
  fig_rural_curve_rates       KA share of crashes and KA per 100M VMT, rural x curve
  fig_radius_bands            curve vs tangent by radius: bands and splines
  fig_pred_rates              predicted KA rate, tangent vs curves, rural vs urban
  fig_geometry_terms          geometry terms of models A-D
  fig_rootogram               count fit of the NB SPF
  fig_cure                    cumulative residual plots
  fig_robustness              curve x rural across specifications
  fig_moran_scatter           Moran scatterplot of SPF residuals
  fig_hotspot_stability       ranking stability, raw rate vs EB

Run all:  python 08_publication_figures.py
Run some: python 08_publication_figures.py map4 radius   (names containing these words)
"""
import faulthandler
import sys
import warnings

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
from scipy import stats
from shapely.geometry import box

import config as cfg
from gis_layers import north_arrow, scalebar

warnings.filterwarnings("ignore", category=UserWarning)
OUT = cfg.OUT / "figures" / "pub"
OUT.mkdir(parents=True, exist_ok=True)
TAB = cfg.OUT / "tables"
CRS = cfg.CRS
FT_PER_MI = 5280.0

# ------------------------------------------------------------------ style
URBAN, RURAL = "#2a78d6", "#eb6834"            # validated categorical pair
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
NET = "#c9c8c2"                                 # network context
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE = ["#fde3d6", "#f9c2a6", "#f39a71", "#eb6834", "#c7501f", "#9c3c15", "#6e290d"]
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec8a72", "#b23230"])
SEQ_O = LinearSegmentedColormap.from_list("seqo", ORANGE)
SEQ_B = LinearSegmentedColormap.from_list("seqb", BLUE)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.5, "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 110,
    "pdf.fonttype": 42,
})
W1, W2 = 3.5, 7.2      # single / double column width (in)


def step(msg):
    print(f"      · {msg}", flush=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  figures/pub/{name}")


def log_ticks(axis, ticks=(0.5, 0.7, 1, 1.4, 2)):
    """Log axis with plain labels only at the given ticks (no 6x10^-1 minor labels)."""
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_ticklabels([f"{t:g}" for t in ticks])
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


def panel_label(ax, s):
    ax.text(0.0, 1.02, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            color=INK, va="bottom")


# ------------------------------------------------------------------ map helpers
def map_axes(ax):
    ax.set_axis_off()
    ax.set_aspect("equal")


def _old_scalebar(ax, miles=50, loc=(0.05, 0.05)):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    L = miles * FT_PER_MI
    bx, by = x0 + loc[0] * (x1 - x0), y0 + loc[1] * (y1 - y0)
    h = 0.008 * (y1 - y0)
    for k in range(2):
        ax.add_patch(plt.Rectangle((bx + k * L / 2, by), L / 2, h,
                                   fc=INK if k == 0 else "white", ec=INK, lw=0.6, zorder=20))
    for frac, lab in ((0, "0"), (1, f"{miles} mi")):
        ax.text(bx + frac * L, by + 2.2 * h, lab, ha="center", va="bottom", fontsize=6.5,
                color=INK, zorder=20)


def _old_north_arrow(ax, loc=(0.95, 0.88)):
    ax.annotate("N", xy=loc, xytext=(loc[0], loc[1] - 0.09), xycoords="axes fraction",
                ha="center", va="center", fontsize=8, fontweight="bold", color=INK,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0), zorder=20)


def hbar(fig, ax, mappable, label, rect=(0.58, 0.07, 0.38, 0.035), **kw):
    """Compact horizontal colorbar (inside the panel by default, or below it via rect)."""
    cax = ax.inset_axes(list(rect))
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", **kw)
    cb.set_label(label, fontsize=6.5, color=INK2, labelpad=2)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.4)
    return cb


def basemap(ax):
    try:
        import contextily as cx
        cx.add_basemap(ax, crs=f"EPSG:{CRS}", source=cx.providers.CartoDB.PositronNoLabels,
                       attribution_size=5, zorder=0)
        return True
    except Exception:
        return False


def legend_box(ax, handles, title=None, loc="upper left", **kw):
    lg = ax.legend(handles=handles, title=title, loc=loc, frameon=True, framealpha=0.92,
                   edgecolor=GRID, fontsize=6.5, title_fontsize=7, **kw)
    lg.get_frame().set_linewidth(0.5)
    return lg


# ------------------------------------------------------------------ data
def load_network():
    """Segment lines for every year from the geodatabase, keyed like the models."""
    frames = []
    for yr in cfg.YEARS:
        g = gpd.read_file(cfg.GDB, layer=f"{cfg.STATE}{str(yr)[-2:]}road", engine="pyogrio",
                          columns=["ROUTE_ID", "BEGMP"])
        g = g[g.geometry.notna() & ~g.geometry.is_empty].to_crs(CRS)
        g["route"] = g["ROUTE_ID"].astype(str).str.strip()
        g["seg_key"] = g["route"] + "_" + g["BEGMP"].astype(float).round(3).astype(str)
        g["year"] = yr
        frames.append(g[["seg_key", "route", "year", "geometry"]])
    net = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=CRS)
    net["geometry"] = net.geometry.force_2d()
    ids = set(net["route"])
    twins = {r for r in ids if r.endswith("d") and r[:-1] + cfg.INVENTORY_SUFFIX in ids}
    return net[~net["route"].isin(twins)]


def segment_summary():
    sg = pd.read_parquet(cfg.OUT / "data" / "segment_year_fitted.parquet")
    a = sg["alpha_P"].iloc[0]                    # primary SPF (model E)
    sg["var"] = sg["mu_P"] + a * sg["mu_P"] ** 2
    s = sg.groupby("seg_key").agg(y=("n_severe", "sum"), mu=("mu_P", "sum"), var=("var", "sum"),
                                  vmt=("vmt", "sum"), rural=("rural", "max"),
                                  curv=("curv_max", "max"), aadt=("aadt", "mean"),
                                  route=("route", "first"), begmp=("begmp", "first"),
                                  any_curve=("any_curve", "max"))
    w = 1 / (1 + a * s["mu"])
    s["eb"] = w * s["mu"] + (1 - w) * s["y"]
    s["eb_rate"] = s["eb"] / (s["vmt"] / 1e8)
    s["pearson"] = (s["y"] - s["mu"]) / np.sqrt(s["var"])
    return sg, s, a


def route_label(route):
    n = int("".join(ch for ch in route[:3] if ch.isdigit()) or 0)
    if n in (5, 82, 90, 182, 205, 405, 705):
        return f"I-{n}"
    if n in (2, 12, 97, 101, 195, 197, 395, 730):
        return f"US {n}"
    return f"SR {n}"


# ================================================================== GIS maps
import gis_layers as GL


def seg_lines(net, s):
    """Mainline segment lines (latest year) joined to the 3-year segment summary."""
    g = net[net["year"] == cfg.YEARS[-1]].merge(s, left_on="seg_key", right_index=True)
    return g[g["vmt"] > 0]


def cased(ax, gdf, color, lw, z, casing="white", cw=None, **kw):
    """Line with a white casing so it reads over terrain and other lines."""
    if len(gdf) == 0:
        return
    gdf.plot(ax=ax, color=casing, linewidth=(cw if cw is not None else np.asarray(lw) + 1.1),
             zorder=z, **kw)
    gdf.plot(ax=ax, color=color, linewidth=lw, zorder=z + 0.1, **kw)


def map1_study_area(bm, net, cr, s):
    g = seg_lines(net, s)
    fig, ax = plt.subplots(figsize=(W2, 4.9))
    bm.draw(ax, regions=True)
    rel = net[(net["year"] == cfg.YEARS[-1]) & ~net["seg_key"].isin(s.index)]
    rel.plot(ax=ax, color="#8f8d86", lw=0.35, zorder=3)
    cased(ax, g[g["rural"] == 0], URBAN, 0.9, 4)
    cased(ax, g[g["rural"] == 1], RURAL, 0.9, 4)
    ka = cr[(cr["severe"] == 1) & cr["x"].notna()]
    ax.scatter(ka["x"], ka["y"], s=2.2, c="#1b1b1b", alpha=0.55, lw=0, zorder=6)
    bm.region_labels(ax)
    scalebar(ax, 100, loc=(0.70, 0.035))
    north_arrow(ax)
    bm.locator(ax, rect=(0.005, 0.74, 0.20, 0.25))
    legend_box(ax, [Line2D([], [], color=URBAN, lw=2, label="Urban mainline"),
                    Line2D([], [], color=RURAL, lw=2, label="Rural mainline"),
                    Line2D([], [], color="#8f8d86", lw=1, label="Ramps, couplets, other"),
                    Line2D([], [], marker="o", ls="", ms=3, color="#1b1b1b",
                           label=f"KA crash, 2021–23 (n = {len(ka):,})"),
                    Line2D([], [], color=GL.REGION, lw=1, ls="--", label="WSDOT region"),
                    Line2D([], [], color=GL.COUNTY, lw=0.8, label="County")],
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    save(fig, "map1_study_area")


def county_table(bm, net, s):
    g = seg_lines(net, s).copy()
    g["geometry"] = g.geometry.interpolate(0.5, normalized=True)
    j = gpd.sjoin(g[["y", "mu", "vmt", "rural", "geometry"]], bm.wa[["NAME", "geometry"]],
                  predicate="within")
    c = j.groupby("NAME").agg(O=("y", "sum"), E=("mu", "sum"), vmt=("vmt", "sum"),
                             rural_vmt=("vmt", lambda v: v[j.loc[v.index, "rural"] == 1].sum()))
    c["rate"] = c["O"] / (c["vmt"] / 1e8)
    c["smr"] = c["O"] / c["E"]
    c["lo"] = stats.chi2.ppf(0.025, 2 * c["O"]) / 2 / c["E"]
    c["hi"] = stats.chi2.ppf(0.975, 2 * c["O"] + 2) / 2 / c["E"]
    c["sig"] = (c["lo"] > 1) | (c["hi"] < 1)
    c["rural_share"] = c["rural_vmt"] / c["vmt"]
    c.round(3).to_csv(TAB / "tab_county_rates.csv")
    return bm.wa.merge(c, left_on="NAME", right_index=True, how="left")


# small counties: (dx mi, dy mi, rotation) so labels do not collide
COUNTY_LABEL_NUDGE = {"Columbia": (0, 2, 90), "Garfield": (0, 4, 90), "Walla Walla": (-6, 0, 0),
                      "Asotin": (2, -4, 90), "San Juan": (-4, 2, 0), "Island": (3, 0, 0),
                      "Kitsap": (-2, 0, 0), "Wahkiakum": (2, -1, 0), "Pend Oreille": (0, 0, 90)}


def map2_county_choropleths(bm, net, s):
    cg = county_table(bm, net, s)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.15), gridspec_kw={"wspace": 0.02})
    # (a) KA rate, quantile classes
    q = np.unique(np.nanquantile(cg["rate"], np.linspace(0, 1, 6)).round(2))
    cmap_a = ListedColormap(ORANGE[1:len(q)])
    norm_a = BoundaryNorm(q, cmap_a.N)
    # (b) observed / expected, symmetric classes on the log scale
    sb = [0, 0.67, 0.8, 0.91, 1.1, 1.25, 1.5, np.inf]
    cmap_b = ListedColormap(["#184f95", "#6da7ec", "#b7d3f6", "#f0efec", "#f6c3b4", "#ec8a72", "#b23230"])
    norm_b = BoundaryNorm(sb, cmap_b.N)
    for ax, col, cmap, norm in ((axes[0], "rate", cmap_a, norm_a), (axes[1], "smr", cmap_b, norm_b)):
        bm.draw(ax, relief=False, counties=False, regions=False, cities=False, neighbor_labels=False)
        cg.plot(ax=ax, column=col, cmap=cmap, norm=norm, edgecolor="white", lw=0.5, zorder=2,
                missing_kwds=dict(color="#e4e3df", edgecolor="white", lw=0.5))
        bm.reg_lines.plot(ax=ax, color=GL.REGION, lw=0.9, zorder=3)
        for _, r in cg.iterrows():
            p = r.geometry.representative_point()
            dx, dy, rot = COUNTY_LABEL_NUDGE.get(r["NAME"], (0, 0, 0))
            ax.text(p.x + dx * FT_PER_MI, p.y + dy * FT_PER_MI, r["NAME"], fontsize=4.3, ha="center",
                    va="center", rotation=rot, color="#2b2a27", path_effects=GL._halo(1.4), zorder=4)
    sig = cg[cg["sig"] == True]
    if len(sig):
        sig.boundary.plot(ax=axes[1], color="#111111", lw=1.1, zorder=3.5)
    labs_a = [f"{q[i]:.1f}–{q[i + 1]:.1f}" for i in range(len(q) - 1)]
    legend_box(axes[0], [Patch(fc=cmap_a(i), ec="white", label=l) for i, l in enumerate(labs_a)],
               title="KA crashes per 100M VMT (mainline, 2021–23)", loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=3)
    labs_b = ["< 0.67", "0.67–0.80", "0.80–0.91", "0.91–1.10", "1.10–1.25", "1.25–1.50", "> 1.50"]
    hb = [Patch(fc=cmap_b(i), ec="white", label=l) for i, l in enumerate(labs_b)]
    hb.append(Patch(fc="none", ec="#111111", lw=1.3, label="95% CI excludes 1"))
    legend_box(axes[1], hb, title="Observed / expected KA crashes (model E)", loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=3)
    for ax, lab in zip(axes, ("(a)", "(b)")):
        scalebar(ax, 100, loc=(0.60, 0.03), fs=6)
        panel_label(ax, lab)
    save(fig, "map2_county_choropleths")


def map3_network_risk(bm, net, s):
    g = seg_lines(net, s).sort_values("eb_rate")
    bins = np.unique(np.quantile(g["eb_rate"], [0, .2, .4, .6, .8, .95, 1]).round(2))
    cmap = ListedColormap(ORANGE[1:len(bins)])
    norm = BoundaryNorm(bins, cmap.N)
    lw = np.interp(np.log10(g["aadt"].clip(lower=100)), [2, 5.3], [0.6, 2.6])
    fig, ax = plt.subplots(figsize=(W2, 4.9))
    bm.draw(ax, regions=True)
    g.plot(ax=ax, color="white", linewidth=lw + 1.0, zorder=4)
    g.plot(ax=ax, column="eb_rate", cmap=cmap, norm=norm, linewidth=lw, zorder=4.1)
    bm.region_labels(ax)
    scalebar(ax, 100, loc=(0.70, 0.035))
    north_arrow(ax)
    labs = [f"{bins[i]:.1f}–{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]
    labs[-1] += "  (top 5%)"
    legend_box(ax, [Line2D([], [], color=cmap(i), lw=2.6, label=l) for i, l in enumerate(labs)]
               + [Line2D([], [], color="#888", lw=0.7, label="Low AADT"),
                  Line2D([], [], color="#888", lw=2.6, label="High AADT")],
               title="Empirical Bayes KA crashes per 100M VMT, 2021–23", loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=4)
    save(fig, "map3_network_risk")


RADIUS_EDGES = [0, 1000 / 3000, 0.5, 1.0, 2.0, np.inf]
RADIUS_LABS = ["R ≥ 3000 ft", "2000–3000 ft", "1000–2000 ft", "500–1000 ft", "R < 500 ft"]
RADIUS_COLS = ["#f6c3b4", "#f39a71", "#eb6834", "#b23230", "#5c1414"]


def _curve_layer(ax, g, lw0=0.6, step=0.5, z=5):
    for b in range(5):
        sub = g[g["band"] == b]
        if len(sub):
            cased(ax, sub, RADIUS_COLS[b], lw0 + step * b, z + b * 0.2)


def _distinct_top(eb_keys, pts, n=4, min_ft=15 * 5280):
    keep = []
    for k in eb_keys:
        if k not in pts.index:
            continue
        p = pts[k]
        if all(p.distance(pts[q]) > min_ft for q in keep):
            keep.append(k)
        if len(keep) == n:
            break
    return keep


def map4_rural_curves_zoom(bm, net, cr, s):
    eb = pd.read_csv(TAB / "tab_hotspots_eb.csv", index_col=0)
    g = seg_lines(net, s)
    rc = g[(g["rural"] == 1) & (g["curv"] > 0)].copy()
    rc["band"] = pd.cut(rc["curv"], RADIUS_EDGES, labels=False, right=False)
    hot = g[g["seg_key"].isin(eb.index)].dissolve("seg_key")
    pts = hot.geometry.interpolate(0.5, normalized=True)
    rank = {k: i + 1 for i, k in enumerate(eb.index)}
    zoom = _distinct_top(list(eb.index), pts, n=4)
    step(f"{len(rc):,} rural curve segments; {len(pts)} hotspots; close-ups: {zoom}")
    hw, hh = 7 * 5280, 4 * 5280                      # zoom window: 14 x 8 miles

    fig = plt.figure(figsize=(W2, 8.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.3, 1], hspace=0.06)
    top = outer[0].subgridspec(1, 2, width_ratios=[3.2, 1], wspace=0.0)
    ax = fig.add_subplot(top[0])
    lax = fig.add_subplot(top[1])
    lax.axis("off")
    step("statewide base map (terrain tiles)")
    bm.draw(ax, regions=False,
            city_names=[c[0] for c in GL.CITIES if c[0] != "Olympia"])   # Olympia sits among hotspots
    step("statewide layers")
    net[net["year"] == cfg.YEARS[-1]].plot(ax=ax, color="#8f8d86", lw=0.3, zorder=3)
    _curve_layer(ax, rc, lw0=0.5, step=0.35)
    for i, k in enumerate(zoom):
        p = pts[k]
        ax.add_patch(plt.Rectangle((p.x - hw, p.y - hh), 2 * hw, 2 * hh, fill=False,
                                   ec="#111111", lw=1.0, zorder=20))
    # numbered markers for all ten; nudge markers that would overlap (> 6 mi apart)
    placed = []
    for k, p in sorted(pts.items(), key=lambda kv: rank[kv[0]]):
        x, y = p.x, p.y
        for q in placed:
            if np.hypot(x - q[0], y - q[1]) < 6 * FT_PER_MI:
                y = q[1] - 7 * FT_PER_MI
        if (x, y) != (p.x, p.y):
            ax.plot([p.x, x], [p.y, y], color="#111111", lw=0.5, zorder=20.5)
        placed.append((x, y))
        ax.scatter(x, y, s=30, facecolor="white", edgecolor="#111111", lw=0.8, zorder=21)
        ax.text(x, y, str(rank[k]), ha="center", va="center", fontsize=4.8, fontweight="bold", zorder=22)
    # close-up letters: first box corner clear of markers and cities
    avoid = placed + GL.city_points()
    for i, k in enumerate(zoom):
        p = pts[k]
        corners = [(p.x + hw, p.y + hh, "left", "bottom"), (p.x - hw, p.y + hh, "right", "bottom"),
                   (p.x + hw, p.y - hh, "left", "top"), (p.x - hw, p.y - hh, "right", "top")]
        def clear(c):
            cx = c[0] + (3 if c[2] == "left" else -3) * FT_PER_MI
            cy = c[1] + (3 if c[3] == "bottom" else -3) * FT_PER_MI
            return min(np.hypot(cx - a, cy - b) for a, b in avoid)
        c = max(corners, key=clear) if clear(corners[0]) < 7 * FT_PER_MI else corners[0]
        ax.text(c[0], c[1], "abcd"[i], fontsize=7.5, fontweight="bold", ha=c[2], va=c[3],
                color="#111111", path_effects=GL._halo(), zorder=23)
    scalebar(ax, 100, loc=(0.66, 0.035))
    north_arrow(ax)
    h = [Line2D([], [], color=RADIUS_COLS[b], lw=0.9 + 0.5 * b, label=RADIUS_LABS[b]) for b in range(5)]
    h += [Line2D([], [], color="#8f8d86", lw=1, label="Other state routes"),
          Line2D([], [], marker="o", ls="", ms=5.5, mfc="white", mec="#111111", label="Top-10 EB segment"),
          Patch(fc="none", ec="#111111", label="Close-up (a–d)"),
          Line2D([], [], color="#111111", lw=2.5, label="Hotspot segment"),
          Line2D([], [], marker="o", ls="", ms=4, mfc="#111111", mec="white", label="KA crash"),
          Line2D([], [], marker="o", ls="", ms=3, mfc="#9a988f", mec="none", label="Other crash")]
    lg = lax.legend(handles=h, title="Rural curves by radius", loc="upper left", frameon=False,
                    fontsize=6.5, title_fontsize=7, handlelength=2.2)
    lg._legend_box.align = "left"
    tb = "\n".join(f"{rank[k]:>2}  {route_label(r['route'])} MP {float(k.split('_')[-1]):.2f}"
                   for k, r in eb.iterrows())
    lax.text(0.02, 0.02, "Top-10 by EB excess\n" + tb, transform=lax.transAxes, fontsize=5.6,
             family="monospace", color=INK2, va="bottom")
    ka = cr[(cr["severe"] == 1) & cr["x"].notna()]
    allc = cr[cr["x"].notna()]
    bot = outer[1].subgridspec(2, 2, hspace=0.28, wspace=0.05)
    for i, k in enumerate(zoom):
        step(f"close-up {'abcd'[i]}: base map")
        za = fig.add_subplot(bot[i // 2, i % 2])
        p = pts[k]
        ext = (p.x - hw, p.y - hh, p.x + hw, p.y + hh)
        bm.draw(za, extent=ext, regions=False, cities=True, neighbor_labels=False, fade_outside=False,
                offline_relief=False)
        clip = box(*ext)
        step(f"close-up {'abcd'[i]}: layers")
        net[net["year"] == cfg.YEARS[-1]].clip(clip).plot(ax=za, color="#77756e", lw=1.0, zorder=3)
        _curve_layer(za, rc.clip(clip), lw0=1.6, step=0.7)
        cased(za, hot.loc[[k]].clip(clip), "#111111", 3.0, 9, casing="#ffd166", cw=6.5)
        inb = lambda d: d[(d.x > ext[0]) & (d.x < ext[2]) & (d.y > ext[1]) & (d.y < ext[3])]
        a = inb(allc)
        za.scatter(a.x, a.y, s=3, c="#9a988f", lw=0, alpha=0.7, zorder=10)
        kk = inb(ka)
        za.scatter(kk.x, kk.y, s=16, c="#111111", edgecolor="white", lw=0.6, zorder=11)
        r = eb.loc[k]
        za.set_title(f"({'abcd'[i]})  #{rank[k]} {route_label(r['route'])} MP {float(k.split('_')[-1]):.2f}:  "
                     f"{int(r['y'])} KA observed vs {r['mu']:.1f} expected",
                     fontsize=6.5, loc="left", color=INK, pad=2.5)
        scalebar(za, 4, loc=(0.06, 0.08), fs=5.5, bg=True)
        za.set_axis_on()
        za.set_xlabel("")
        za.set_ylabel("")
        za.set_xticks([])
        za.set_yticks([])
        za.grid(False)
        for sp in za.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor("#111111")
            sp.set_linewidth(0.8)
    step("saving PNG and PDF")
    save(fig, "map4_rural_curves_hotspots")


def map5_residuals_lisa(bm, net, s):
    lisa = pd.read_csv(TAB / "segment_residuals_lisa.csv", index_col=0)
    g = seg_lines(net, s).merge(lisa[["lisa"]], left_on="seg_key", right_index=True, how="left")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.2), gridspec_kw={"wspace": 0.02})
    lim = np.nanpercentile(np.abs(g["pearson"]), 98)
    for ax in axes:
        bm.draw(ax, relief=True, cities=False, neighbor_labels=False, regions=True)
        net[net["year"] == cfg.YEARS[-1]].plot(ax=ax, color="#a9a79f", lw=0.3, zorder=3)
    gg = g.assign(a=g["pearson"].abs()).sort_values("a")
    lw_r = np.interp(gg["a"], [0, 1, lim], [0.35, 0.8, 2.2])
    gg.plot(ax=axes[0], color="white", lw=lw_r + 0.6, zorder=4)
    gg.plot(ax=axes[0], column="pearson", cmap=DIV, vmin=-lim, vmax=lim, lw=lw_r, zorder=4.1)
    hbar(fig, axes[0], plt.cm.ScalarMappable(cmap=DIV, norm=plt.Normalize(-lim, lim)),
         "Pearson residual (observed − expected KA crashes)", rect=(0.15, -0.09, 0.70, 0.04))
    cmap = {"High-High": "#b23230", "Low-Low": "#184f95", "High-Low": "#ec8a72", "Low-High": "#6da7ec"}
    for k, c in cmap.items():
        cased(axes[1], g[g["lisa"] == k], c, 1.6, 5)
    n = g["lisa"].value_counts()
    legend_box(axes[1], [Line2D([], [], color=c, lw=2, label=f"{k} (n = {n.get(k, 0):,})")
                         for k, c in cmap.items()], title="Local Moran's I, p < 0.05", loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=2)
    for ax, lab in zip(axes, ("(a)", "(b)")):
        scalebar(ax, 100, loc=(0.60, 0.03), fs=6)
        panel_label(ax, lab)
    save(fig, "map5_residuals_lisa")


# ================================================================== statistical
def coef(tab, model, term):
    try:
        return tab.loc[term, (model, "est")], tab.loc[term, (model, "lo")], tab.loc[term, (model, "hi")]
    except KeyError:
        return (np.nan,) * 3


def fig_geometry_terms():
    t = pd.read_csv(TAB / "tab_2x2_coefficients.csv", header=[0, 1], index_col=0)
    rows = [("Curve (presence / curvature)", "on_curve", "curv", "any_curve", "curv_max"),
            ("Curve × rural", "on_curve:rural", "curv:rural", "any_curve:rural", "curv_max:rural"),
            ("Grade (≥4% / %)", "steep", "abs_grade", "steep", "grade_max"),
            ("Curve × grade", None, "curv:abs_grade", None, "curv_max:grade_max")]
    panels = [("Crash level: odds ratio of KA", "A_logit_binary", "B_logit_continuous", 1, 2),
              ("Segment level: KA rate ratio", "C_spf_binary", "D_spf_continuous", 3, 4)]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, (title, mb, mc, ib, ic) in zip(axes, panels):
        for off, m, idx, col, lab in ((0.14, mb, ib, INK2, "Binary geometry"),
                                      (-0.14, mc, ic, URBAN, "Continuous geometry")):
            for yy, r in zip(y, rows):
                if r[idx] is None:
                    continue
                e, lo, hi = coef(t, m, r[idx])
                if np.isnan(e):
                    continue
                ax.plot([lo, hi], [yy + off] * 2, color=col, lw=1.4, solid_capstyle="round")
                ax.plot(e, yy + off, "o", ms=4.5, color=col, mec="white", mew=0.6,
                        label=lab if yy == y[0] else None)
        ax.axvline(1, color=MUTED, lw=0.8, ls=(0, (3, 2)))
        ax.set_xscale("log")
        log_ticks(ax.xaxis)
        ax.set_xlim(0.6, 2.1)
        ax.set_title(title, loc="left", color=INK)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [r[0] for r in rows])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.55, -0.06))
    save(fig, "fig_geometry_terms")


def fig_pred_rates(sg):
    pr = pd.read_csv(TAB / "pred_rates_interactions.csv")
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.5), sharey=True)
    obs = sg[sg["curv_max"] > 0]
    for ax, gr in zip(axes, (0, 3, 6)):
        for rural, col, lab in ((0, URBAN, "Urban"), (1, RURAL, "Rural")):
            d0 = pr[(pr.grade == gr) & (pr.rural == rural)]
            t0, d = d0[d0.curv == 0], d0[d0.curv > 0]      # tangent vs curves (model E has a step)
            ax.fill_between(d.curv, d.lo, d.hi, color=col, alpha=0.16, lw=0)
            ax.plot(d.curv, d.rate, color=col, lw=1.8)
            if len(t0):
                xo = -0.25 if rural == 0 else -0.55
                ax.errorbar([xo], t0.rate, yerr=[t0.rate - t0.lo, t0.hi - t0.rate], fmt="s", ms=3.5,
                            color=col, mec="white", mew=0.5, elinewidth=1.2, capsize=0, clip_on=False)
            ax.text(d.curv.iloc[-1], d.rate.iloc[-1], f" {lab}", color=INK2, fontsize=7,
                    va="center", ha="left")
            o = obs[obs["rural"] == rural]["curv_max"]
            o = o[o <= d.curv.max()].sample(min(len(o), 600), random_state=1)
            ymin = 0.02 if rural == 0 else 0.06
            ax.plot(o, np.full(len(o), ymin), "|", color=col, ms=4, alpha=0.35,
                    transform=ax.get_xaxis_transform())
        ax.set_title(f"Maximum grade {gr}%", loc="left", color=INK)
        ax.set_xlabel(r"Curvature, 1000/$R$ (ft$^{-1}$)")
        ax.set_xlim(-0.9, pr.curv.max() * 1.25)
        xt = [c for c in range(0, int(pr.curv.max() * 1.25) + 1, 2 if pr.curv.max() < 6 else 4)]
        ax.set_xticks([-0.4] + xt, ["T"] + [str(c) for c in xt])
        top = ax.secondary_xaxis("top", functions=(lambda c: c, lambda c: c))
        ticks = [c for c in (2, 4, 1000 / 150, 8) if c <= pr.curv.max()]
        top.set_xticks(ticks, [f"{1000 / c:.0f}" for c in ticks])
        top.set_xlabel("Radius (ft)", fontsize=6.5, color=MUTED)
        top.tick_params(labelsize=6, colors=MUTED)
    axes[0].set_ylabel("Predicted KA crashes per 100M VMT")
    axes[0].set_ylim(bottom=0)
    save(fig, "fig_pred_rates")


def fig_rural_curve_rates():
    """(a) KA share of crashes and (b) KA crashes per 100M VMT, by rurality and curve status."""
    rc = pd.read_csv(TAB / "tab_rural_curve_rate.csv")
    vr = pd.read_csv(TAB / "tab_vmt_rates.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.4))
    specs = ((axes[0], rc, "on_curve", rc["mean"] * 100, "KA share of crashes (%)", "{:.2f}%"),
             (axes[1], vr, "any_curve", vr["rate_per_100M"], "KA crashes per 100M VMT", "{:.2f}"))
    for ax, d, ccol, val, ylab, fmt in specs:
        d = d.assign(v=val.to_numpy())
        w = 0.36
        for i, (ru, col, lab) in enumerate(((0, URBAN, "Urban"), (1, RURAL, "Rural"))):
            dd = d[d["rural"] == ru].sort_values(ccol)
            x = np.arange(2) + (i - 0.5) * w
            ax.bar(x, dd["v"], w * 0.92, color=col, label=lab)
            for xx, vv in zip(x, dd["v"]):
                ax.text(xx, vv, fmt.format(vv), ha="center", va="bottom", fontsize=6.3, color=INK2)
        ax.set_xticks([0, 1], ["Tangent", "Curve"])
        ax.set_ylabel(ylab)
        ax.set_ylim(0, d["v"].max() * 1.18)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, loc="upper left")
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig_rural_curve_rates")


def fig_radius_bands():
    """Curve vs tangent ratio by radius: radius bands (points) and natural-spline fit (lines)."""
    t = pd.read_csv(TAB / "tab_radius_bands_rural.csv")
    sp_file = TAB / "tab_spline_curve_effect.csv"
    sp = pd.read_csv(sp_file) if sp_file.exists() else None
    order = [("[0.0, 500.0)", 300), ("[500.0, 1000.0)", 707), ("[1000.0, 2000.0)", 1414),
             ("[2000.0, 3000.0)", 2449), ("[3000.0, inf)", 4500)]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6), sharey=True)
    for ax, (m, ttl) in zip(axes, (("logit", "Crash level: odds ratio of KA vs tangent"),
                                   ("SPF", "Segment level: KA crash ratio vs tangent"))):
        for rural, col, off, lab in ((0, URBAN, 0.95, "Urban"), (1, RURAL, 1.05, "Rural")):
            if sp is not None:
                d = sp[(sp.model == m) & (sp.rural == rural)].sort_values("radius_ft")
                ax.fill_between(d.radius_ft, d.lo, d.hi, color=col, alpha=0.13, lw=0)
                ax.plot(d.radius_ft, d.ratio, color=col, lw=1.4)
            for i, (k, xm) in enumerate(order):
                r = t[(t.model == m) & (t.band == k) & (t.rural == rural)]
                if not len(r):
                    continue
                r = r.iloc[0]
                ax.plot([xm * off] * 2, [r.lo, r.hi], color=col, lw=1.1, alpha=0.9)
                ax.plot(xm * off, r.ratio, "o", ms=4.2, color=col, mec="white", mew=0.6,
                        label=f"{lab}, radius band" if i == 0 else None)
        ax.axhline(1, color=MUTED, lw=0.8, ls=(0, (3, 2)))
        ax.set_xscale("log")
        ax.set_xlim(130, 6000)
        ax.xaxis.set_major_locator(FixedLocator([150, 250, 500, 1000, 2000, 5000]))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.invert_xaxis()
        ax.set_xlabel("Curve radius (ft); sharper to the right")
        ax.set_yscale("log")
        log_ticks(ax.yaxis, (0.5, 0.7, 1, 1.4, 2, 3))
        ax.set_ylim(0.35, 4.0)
        ax.set_title(ttl, loc="left", color=INK, fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    if sp is not None:
        h += [Line2D([], [], color=URBAN, lw=1.4), Line2D([], [], color=RURAL, lw=1.4)]
        l += ["Urban, spline", "Rural, spline"]
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01),
               fontsize=6.5)
    save(fig, "fig_radius_bands")


def fig_robustness():
    sens = pd.read_csv(TAB / "tab_sens_exposure_segmentation.csv", header=[0, 1], index_col=0)
    t = pd.read_csv(TAB / "tab_2x2_coefficients.csv", header=[0, 1], index_col=0)
    seg = [("E (primary)", "E: presence + sharpness (primary)"),
           ("C: presence only", "C: presence only"), ("D: sharpness only", "D: sharpness only"),
           ("log VMT offset", "E, log VMT offset"),
           ("log VMT, coefficient estimated", "E, log VMT as covariate"),
           ("+ share on curve, n curves", "E + share on curve, no. of curves"),
           ("length-weighted curvature", "E, length-weighted curvature"),
           ("fixed 0.1-mi segments", "E, fixed 0.1-mi segments")]

    def rp(file, term):
        try:
            r = pd.read_csv(TAB / file, index_col=0).loc[term]
            return np.exp(r.est), np.exp(r.est - 1.96 * r.se), np.exp(r.est + 1.96 * r.se)
        except (FileNotFoundError, KeyError):
            return (np.nan,) * 3

    def rows_for(pres):
        out = []
        for k, lab in seg:
            term = ("any_curve:rural" if pres else
                    ("curv_lenwt:rural" if "length-weighted" in k else "curv_max:rural"))
            if (k, "est") in sens.columns and term in sens.index:
                out.append(("seg", lab, *sens.loc[term, [(k, "est"), (k, "lo"), (k, "hi")]]))
            else:
                out.append(("seg", lab, np.nan, np.nan, np.nan))
        out.append(("seg", "E, random-parameters NB (panel)",
                    *rp("tab_RPNB.csv", "any_curve:rural" if pres else "curv_max:rural")))
        if pres:
            out += [("crash", "A: logit, binary", *coef(t, "A_logit_binary", "on_curve:rural")),
                    ("crash", "A0: without behavioural covariates",
                     *coef(t, "A0_logit_binary_nobehav", "on_curve:rural"))]
        else:
            out += [("crash", "B: logit, continuous", *coef(t, "B_logit_continuous", "curv:rural")),
                    ("crash", "B0: without behavioural covariates",
                     *coef(t, "B0_logit_cont_nobehav", "curv:rural")),
                    ("crash", "B, random-parameters logit", *rp("tab_RPL.csv", "curv:rural"))]
        return pd.DataFrame(out, columns=["level", "spec", "est", "lo", "hi"])

    R = [r.dropna(subset=["est"]).reset_index(drop=True) for r in (rows_for(True), rows_for(False))]
    n = max(len(r) for r in R)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 0.2 * n + 1.1), sharey=False,
                             gridspec_kw={"wspace": 0.95})
    for ax, r, ttl, xl in zip(axes, R, ("(a) Curve presence × rural", "(b) Curvature × rural"),
                              ("Ratio, rural vs urban curve (95% CI)",
                               "Ratio per unit 1000/R (95% CI)")):
        y = np.arange(len(r))[::-1]
        for yy, (_, row) in zip(y, r.iterrows()):
            if np.isnan(row.est):
                continue
            primary = row.spec.startswith("E: presence")
            col = RURAL if primary else (INK2 if row.level == "seg" else URBAN)
            ax.plot([row.lo, row.hi], [yy, yy], color=col, lw=1.4)
            ax.plot(row.est, yy, "o", ms=5 if primary else 4, color=col, mec="white", mew=0.6)
        ax.axvline(1, color=MUTED, lw=0.8, ls=(0, (3, 2)))
        ax.set_yticks(y, r["spec"], fontsize=6.3)
        split = (r["level"] == "crash").to_numpy().argmax()
        ax.axhline(y[split] + 0.5, color=MUTED, lw=0.6)
        ax.set_title(ttl, loc="left", color=INK, fontsize=8)
        ax.set_xlabel(xl, fontsize=7)
        if ax is axes[0]:
            ax.set_xscale("log")
            log_ticks(ax.xaxis, (0.5, 0.7, 1, 1.4, 2))
        ax.grid(axis="y", visible=False)
    save(fig, "fig_robustness")


def fig_rootogram(sg):
    y = sg["n_severe"].to_numpy()
    mu, a = sg["mu_P"].to_numpy(), sg["alpha_P"].iloc[0]
    K = 4                                            # last class is "4 or more"
    ks = np.arange(0, K + 1)
    r = 1 / a
    p = r / (r + mu)
    nb = np.array([stats.nbinom.pmf(k, r, p).sum() for k in ks[:-1]])
    po = np.array([stats.poisson.pmf(k, mu).sum() for k in ks[:-1]])
    nb = np.r_[nb, len(y) - nb.sum()]
    po = np.r_[po, len(y) - po.sum()]
    obs = np.array([(y == k).sum() for k in ks[:-1]] + [(y >= K).sum()])
    labs = [str(k) for k in ks[:-1]] + [f"{K}+"]
    # log-VMT offset alternative (saved by 03_models.py)
    muE, aE = sg["mu_V"].to_numpy(), sg["alpha_V"].iloc[0]
    pE = (1 / aE) / (1 / aE + muE)
    nbE = np.array([stats.nbinom.pmf(k, 1 / aE, pE).sum() for k in ks[:-1]])
    nbE = np.r_[nbE, len(y) - nbE.sum()]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.45), gridspec_kw={"width_ratios": [1.15, 1]})
    # (a) hanging rootogram for the NB model, Poisson expectation overlaid
    ax = axes[0]
    top = np.sqrt(nb)
    ax.bar(ks, np.sqrt(obs), bottom=top - np.sqrt(obs), width=0.62, color=BLUE[1], ec=BLUE[3], lw=0.6,
           label="Observed (hanging from NB)")
    ax.plot(ks, top, "o-", color=RURAL, ms=3.5, lw=1.4, label="Expected, NB (model E)")
    ax.plot(ks, np.sqrt(po), "s--", color=INK2, ms=3, lw=1.0, label="Expected, Poisson")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(ks, labs)
    ax.set_xlabel("KA crashes per segment-year")
    ax.set_ylabel(r"$\sqrt{\mathrm{segment\text{-}years}}$")
    ax.legend(frameon=False, loc="upper right", fontsize=6.3)
    ax.grid(axis="x", visible=False)
    panel_label(ax, "(a)")
    # (b) where the fit differs: sqrt(observed) - sqrt(expected), the bar gaps in (a)
    ax = axes[1]
    w = 0.27
    d_nb, d_po, d_e = (np.sqrt(obs) - np.sqrt(e) for e in (nb, po, nbE))
    ax.bar(ks - w, d_po, width=w, color="#b9b7ae", label="Poisson, same means as E")
    ax.bar(ks, d_nb, width=w, color=RURAL, label="NB, model E")
    ax.bar(ks + w, d_e, width=w, color=URBAN, label="NB, log VMT offset")
    ax.axhline(0, color=INK, lw=0.8)
    for v in (-1, 1):
        ax.axhline(v, color=MUTED, lw=0.6, ls=":")
    ax.text(K + 0.45, 1.05, "±1", fontsize=6, color=MUTED, va="bottom", ha="right")
    ax.set_xlim(-0.6, K + 0.6)
    ax.set_xticks(ks, labs)
    ax.set_xlabel("KA crashes per segment-year")
    ax.set_ylabel(r"$\sqrt{\mathrm{observed}} - \sqrt{\mathrm{expected}}$")
    lim = max(1.5, np.abs(np.r_[d_po, d_nb, d_e]).max() * 1.15)
    ax.set_ylim(-lim, lim)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=6.3, ncol=3)
    ax.grid(axis="x", visible=False)
    panel_label(ax, "(b)")
    fig.tight_layout(w_pad=2.0)
    pd.DataFrame({"count": labs, "observed": obs, "expected_NB": nb.round(1),
                  "expected_Poisson": po.round(1), "expected_NB_VMT_offset": nbE.round(1),
                  "alpha_E": round(a, 4), "alpha_VMT_offset": round(aE, 4)}).to_csv(TAB / "tab_rootogram.csv", index=False)
    save(fig, "fig_rootogram")


def cure(ax, x, res, label, show_label=True):
    o = np.argsort(x, kind="stable")
    xs, rs = x[o], res[o]
    c = np.cumsum(rs)
    s2 = np.cumsum(rs ** 2)
    sd = np.sqrt(s2 * (1 - s2 / s2[-1]))
    ax.fill_between(xs, -2 * sd, 2 * sd, color=GRID, lw=0, label=r"$\pm 2\sigma^*$")
    ax.plot(xs, c, color=URBAN, lw=1.1, label="Cumulative residual")
    ax.axhline(0, color=MUTED, lw=0.6)
    if show_label:
        ax.set_xlabel(label)
    out = (np.abs(c) > 2 * sd).mean()
    ax.text(0.98, 0.04, f"{out:.0%} outside", transform=ax.transAxes, fontsize=6.3,
            color=INK2, va="bottom", ha="right",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    return out


def fig_cure(sg):
    y = sg["n_severe"].to_numpy()
    rows = [("E: presence + sharpness", sg["mu_P"].to_numpy()),
            ("D: sharpness only", sg["mu_D"].to_numpy()),
            ("E with log VMT\noffset", sg["mu_V"].to_numpy())]
    lv = np.log10(sg["vmt"].to_numpy())
    cv = sg["curv_max"].to_numpy()
    m = cv > 0
    fig, axes = plt.subplots(len(rows), 3, figsize=(W2, 2.1 * len(rows) + 0.2), sharex="col")
    summ = {}
    for i, (name, mu) in enumerate(rows):
        res = y - mu
        last = i == len(rows) - 1
        summ[(name.split("\n")[0], "fitted")] = cure(axes[i, 0], mu, res, "Fitted KA crashes", last)
        summ[(name.split("\n")[0], "log VMT")] = cure(axes[i, 1], lv, res,
                                                       r"$\log_{10}$ annual VMT", last)
        summ[(name.split("\n")[0], "curvature")] = cure(axes[i, 2], cv[m], res[m],
                                                         r"Curvature, 1000/$R$ (curves only)", last)
        axes[i, 0].set_ylabel(name + "\n\nCumulative residual", fontsize=7)
    # common y-range per column so the two models are compared on the same scale
    for j in range(3):
        lo = min(a.get_ylim()[0] for a in axes[:, j])
        hi = max(a.get_ylim()[1] for a in axes[:, j])
        for a in axes[:, j]:
            a.set_ylim(lo, hi)
    for ax, lab in zip(axes.flat, [f"({c})" for c in "abcdefghi"]):
        panel_label(ax, lab)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.2)
    pd.Series(summ).rename("share_outside_2sd").to_csv(TAB / "tab_cure_summary.csv")
    save(fig, "fig_cure")


def fig_moran_scatter(net, s):
    from libpysal.weights import KNN, lag_spatial
    lisa = pd.read_csv(TAB / "segment_residuals_lisa.csv", index_col=0)
    if {"x", "y"} <= set(lisa.columns):      # identical points and weights to 05 (same Moran's I)
        pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lisa["x"], lisa["y"]), index=lisa.index, crs=CRS)
    else:
        g = net[net["year"] == cfg.YEARS[-1]].drop_duplicates("seg_key").set_index("seg_key")
        g = g.loc[g.index.intersection(lisa.index)]
        pts = gpd.GeoDataFrame(geometry=g.geometry.interpolate(0.5, normalized=True), crs=CRS)
    z = lisa.loc[pts.index, "pearson"].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
    z = (z - z.mean()) / z.std()
    w = KNN.from_dataframe(pts, k=8)
    w.transform = "r"
    lz = lag_spatial(w, z)
    b = np.polyfit(z, lz, 1)[0]
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    ax.hexbin(z, lz, gridsize=45, bins="log", cmap=SEQ_B, mincnt=1, lw=0.1, edgecolors="white")
    xx = np.linspace(np.percentile(z, 0.5), np.percentile(z, 99.5), 50)
    ax.plot(xx, b * xx, color=RURAL, lw=1.6)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlim(np.percentile(z, 0.5), np.percentile(z, 99.8))
    ax.set_ylim(np.percentile(lz, 0.2), np.percentile(lz, 99.8))
    ax.set_xlabel("Standardized SPF residual")
    ax.set_ylabel("Spatial lag of residual (k = 8)")
    ax.text(0.03, 0.96, f"Moran's I (slope) = {b:.3f}", transform=ax.transAxes, fontsize=7,
            color=INK, va="top")
    save(fig, "fig_moran_scatter")


def fig_hotspot_stability():
    J = pd.read_csv(TAB / "tab_hotspot_threshold_jaccard.csv", index_col=0)
    P = pd.read_csv(TAB / "tab_hotspot_posterior_stability.csv", index_col=0)

    def lab(s):
        s = s.strip("()").replace("'", "")
        a, b = [x.strip() for x in s.split(",")]
        return "EB" if a == "EB" else f"{a}M, ≥{b}"
    labs = [lab(c) for c in J.columns]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.0), gridspec_kw={"width_ratios": [1.25, 1],
                                                                    "wspace": 0.75})
    im = axes[0].imshow(J.to_numpy(), cmap=SEQ_B, vmin=0, vmax=1)
    axes[0].set_xticks(range(len(labs)), labs, rotation=90, fontsize=5.5)
    axes[0].set_yticks(range(len(labs)), labs, fontsize=5.5)
    axes[0].grid(False)
    cb = fig.colorbar(im, ax=axes[0], shrink=0.75, pad=0.02)
    cb.set_label("Jaccard overlap of top-10 lists", fontsize=6.5)
    axes[0].set_xlabel("Min. annual VMT, min. KA crashes", fontsize=6.5)
    x = np.arange(len(P))
    axes[1].bar(x - 0.2, P.iloc[:, 0], 0.38, color=INK2, label="Raw-rate ranking", ec="white", lw=0.5)
    axes[1].bar(x + 0.2, P.iloc[:, 1], 0.38, color=RURAL, label="Empirical Bayes ranking", ec="white", lw=0.5)
    axes[1].set_xticks(x, [str(i) for i in P.index])
    axes[1].set_xlabel("Rank in original top 10")
    axes[1].set_ylabel("P(remains in top 10)")
    axes[1].set_ylim(0, max(0.5, P.to_numpy().max() * 1.15))
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].grid(axis="x", visible=False)
    for ax, l in zip(axes, ("(a)", "(b)")):
        panel_label(ax, l)
    save(fig, "fig_hotspot_stability")


def main(only=None):
    faulthandler.enable()                          # print a trace even on a hard crash
    cr = pd.read_parquet(cfg.OUT / "data" / "crash_level.parquet")
    sg, s, _ = segment_summary()
    net = load_network()
    print(f"network segments {len(net):,}; summarized segments {len(s):,}")
    # segment midpoints with district, for data-driven WSDOT region outlines
    mid = net[net["year"] == cfg.YEARS[-1]].drop_duplicates("seg_key").copy()
    mid["geometry"] = mid.geometry.interpolate(0.5, normalized=True)
    dist = sg.groupby("seg_key")["district"].first()
    mid["district"] = mid["seg_key"].map(dist)
    bm = GL.BaseMap(seg_points=mid.dropna(subset=["district"]))
    jobs = ((map1_study_area, (bm, net, cr, s)), (map2_county_choropleths, (bm, net, s)),
            (map3_network_risk, (bm, net, s)), (map4_rural_curves_zoom, (bm, net, cr, s)),
            (map5_residuals_lisa, (bm, net, s)), (fig_rural_curve_rates, ()), (fig_geometry_terms, ()),
            (fig_pred_rates, (sg,)), (fig_radius_bands, ()), (fig_robustness, ()),
            (fig_rootogram, (sg,)), (fig_cure, (sg,)), (fig_moran_scatter, (net, s)),
            (fig_hotspot_stability, ()))
    todo = [(f, a) for f, a in jobs if not only or any(o in f.__name__ for o in only)]
    for n, (f, args) in enumerate(todo, 1):
        print(f"  [{n}/{len(todo)}] drawing {f.__name__} ...", flush=True)
        try:
            f(*args)
        except BaseException as e:                  # one failing figure should not stop the rest
            import traceback
            print(f"  !! {f.__name__} failed: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)


if __name__ == "__main__":
    # e.g.  python 08_publication_figures.py map4   (draw only figures whose name contains "map4")
    main(sys.argv[1:] or None)
