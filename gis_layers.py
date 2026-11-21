"""
gis_layers.py - cartographic reference layers and map furniture for the figures.

Layers are cached in <project>/gis/ after the first download:
  counties  US Census cartographic boundary file, 1:500k (shoreline-clipped)
  states    US Census cartographic boundary file, 1:500k
  provinces Natural Earth 1:10m admin-1 (British Columbia)
  lakes     Natural Earth 1:10m lakes
  terrain   Esri World Shaded Relief tiles (contextily) when online; otherwise the
            global shaded-relief raster bundled with the `basemap-data` package

If a download fails (offline), counties/states fall back to the county file in
`basemap-data` (pip install basemap-data), so the figures always render.
"""
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

import config as cfg

GIS = cfg.ROOT / "gis"
GIS.mkdir(exist_ok=True)
CRS = cfg.CRS
WA_FIPS, NEIGHBOR_FIPS = "53", {"41": "OREGON", "16": "IDAHO", "30": "MONTANA"}

URLS = {
    "counties": "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip",
    "states": "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip",
    "provinces": "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip",
    "lakes": "https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes.zip",
}

# map palette (muted, so data layers carry the color)
WATER = "#d6e6f2"
LAND = "#f7f6f2"
FADE = "#ffffff"
COUNTY = "#a3a199"
STATE = "#6f6e69"
REGION = "#3d3c39"
LABEL = "#5b5a55"

CITIES = [  # name, lat, lon, label offset (dx, dy) in points
    ("Seattle", 47.6062, -122.3321, (5, 2)), ("Spokane", 47.6588, -117.4260, (5, 2)),
    ("Tacoma", 47.2529, -122.4443, (-5, 3)), ("Vancouver", 45.6387, -122.6615, (5, -6)),
    ("Bellingham", 48.7519, -122.4787, (5, 2)), ("Yakima", 46.6021, -120.5059, (5, 2)),
    ("Tri-Cities", 46.2112, -119.1372, (5, 2)), ("Wenatchee", 47.4235, -120.3103, (5, 2)),
    ("Olympia", 47.0379, -122.9007, (5, -8)), ("Walla Walla", 46.0646, -118.3430, (5, 2)),
    ("Port Angeles", 48.1181, -123.4307, (5, 2)), ("Aberdeen", 46.9754, -123.8157, (5, 2)),
    ("Moses Lake", 47.1301, -119.2781, (5, 2)), ("Pullman", 46.7298, -117.1817, (5, -6)),
    ("Omak", 48.4110, -119.5276, (5, 2)), ("Longview", 46.1382, -122.9382, (-5, 2)),
]

# region label anchors (lon, lat), chosen clear of city labels and dense network
REGION_LABEL_AT = {
    "Northwest": (-121.55, 48.42), "North Central": (-119.85, 48.02), "Olympic": (-123.75, 47.62),
    "Southwest": (-122.05, 45.93), "South Central": (-119.75, 46.42), "Eastern": (-118.15, 47.85),
}


def city_points():
    """City locations in the project CRS (for collision checks)."""
    g = gpd.GeoSeries([Point(c[2], c[1]) for c in CITIES], crs=4326).to_crs(CRS)
    return [(q.x, q.y) for q in g]


def _halo(w=2.2):
    return [pe.withStroke(linewidth=w, foreground="white")]


# ------------------------------------------------------------------ downloads
def _fetch(name):
    """Return a GeoDataFrame for a named layer, downloading once into gis/."""
    zpath = GIS / Path(URLS[name]).name
    if not zpath.exists():
        try:
            req = urllib.request.Request(URLS[name], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                zpath.write_bytes(r.read())
            print(f"  downloaded {zpath.name}")
        except Exception as e:
            print(f"  ({name}: download unavailable - {type(e).__name__}; using fallback)")
            return None
    return gpd.read_file(f"zip://{zpath}")


def _basemap_data_dir():
    try:
        import mpl_toolkits.basemap_data as bd
        return Path(list(bd.__path__)[0])
    except Exception:
        return None


def counties_states():
    """(WA counties, neighbor states, WA outline) in the project CRS."""
    c = _fetch("counties")
    s = _fetch("states")
    if c is not None and s is not None:
        c = c.rename(columns={"STATEFP": "STATE_FIPS", "NAME": "NAME"})
        s = s.rename(columns={"STATEFP": "STATE_FIPS"})
    else:
        d = _basemap_data_dir()
        if d is None:
            raise RuntimeError("No boundary data: connect to the internet once, or "
                               "`pip install basemap-data`.")
        c = gpd.read_file(d / "UScounties.shp")
        s = c.dissolve("STATE_FIPS").reset_index()
    c = c.to_crs(CRS)
    s = s.to_crs(CRS)
    wa = c[c["STATE_FIPS"] == WA_FIPS].copy()
    nb = s[s["STATE_FIPS"].isin(NEIGHBOR_FIPS)].copy()
    nb["label"] = nb["STATE_FIPS"].map(NEIGHBOR_FIPS)
    outline = wa.dissolve()
    return wa, nb, outline, s


def province_bc():
    p = _fetch("provinces")
    if p is None:
        return None
    p = p[p["name"] == "British Columbia"]
    return p.to_crs(CRS) if len(p) else None


def lakes():
    lk = _fetch("lakes")
    return None if lk is None else lk.to_crs(CRS)


def regions(wa_counties, seg_points):
    """WSDOT regions as dissolved counties: each county takes the district most of
    its segments carry (data-driven, so no hand-coded county lists)."""
    j = gpd.sjoin(seg_points[["district", "geometry"]].dropna(), wa_counties[["NAME", "geometry"]],
                  predicate="within")
    major = j.groupby("NAME")["district"].agg(lambda x: x.mode().iloc[0])
    wc = wa_counties.copy()
    wc["district"] = wc["NAME"].map(major)
    wc = wc.dropna(subset=["district"])
    reg = wc.dissolve("district").reset_index()
    reg["name"] = reg["district"].astype(int).map(cfg.DISTRICT_NAMES)
    return reg


def interior_borders(reg):
    """Borders shared by two regions (no coastline), so Puget Sound is not traced
    with heavy dashed lines around every island."""
    from shapely.ops import linemerge, unary_union
    g = list(reg.geometry)
    lines = []
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            shared = g[i].boundary.intersection(g[j].buffer(2))    # 2-ft tolerance
            if not shared.is_empty:
                lines.append(shared)
    if not lines:
        return gpd.GeoSeries([], crs=reg.crs)
    m = unary_union(lines)
    try:
        m = linemerge(m)
    except Exception:
        pass
    return gpd.GeoSeries([m], crs=reg.crs)


# ------------------------------------------------------------------ terrain
USE_TILES = os.environ.get("MAP_TILES", "1") != "0"     # set MAP_TILES=0 to skip tile downloads
TILE_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/"
            "MapServer/tile/{z}/{y}/{x}")
TILE_MAX_ZOOM = 13
R_MERC = 6378137.0
_tiles_failed = False


def _lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / np.pi) / 2.0 * n
    return x, y


def _tile_bounds_merc(x, y, z):
    n = 2 ** z
    span = 2 * np.pi * R_MERC / n
    x0 = -np.pi * R_MERC + x * span
    y1 = np.pi * R_MERC - y * span
    return x0, y1 - span, x0 + span, y1


def _get_tile(z, x, y):
    """One 256-px tile, cached in gis/tiles/. Returns an RGB array or None."""
    from PIL import Image
    p = GIS / "tiles" / str(z) / str(x) / f"{y}.jpg"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(TILE_URL.format(z=z, x=x, y=y), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            p.write_bytes(r.read())
    return np.asarray(Image.open(p).convert("RGB"))


def _progress(done, total, cached, width=28):
    """Single-line text progress bar (works in PowerShell and terminals)."""
    frac = done / total if total else 1
    bar = "#" * int(round(frac * width)) + "-" * (width - int(round(frac * width)))
    sys.stdout.write(f"\r      [{bar}] {done}/{total} tiles ({cached} cached)")
    sys.stdout.flush()


def _download_with_progress(z, xs, ys, workers=6):
    """Fetch all tiles in parallel threads, showing a progress bar. Fails fast on error."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    keys = [(x, y) for y in ys for x in xs]
    cached = sum((GIS / "tiles" / str(z) / str(x) / f"{y}.jpg").exists() for x, y in keys)
    out, done = {}, 0
    _progress(0, len(keys), cached)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_get_tile, z, x, y): (x, y) for x, y in keys}
        for f in as_completed(futs):
            out[futs[f]] = f.result()          # re-raises the first download error
            done += 1
            _progress(done, len(keys), cached)
    sys.stdout.write("\n")
    return out


def _tile_relief(extent, max_zoom, px=1600):
    """Mosaic Esri shaded-relief tiles over `extent` and warp them to the map CRS.
    Returns (grey image 0-1, imshow extent) or None."""
    global _tiles_failed
    if not USE_TILES or _tiles_failed:
        return None
    import rasterio  # noqa: F401  (needed for warp)
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject, transform_bounds
    lon0, lat0, lon1, lat1 = transform_bounds(f"EPSG:{CRS}", "EPSG:4326", *extent)
    tiles_across = px / 256.0
    z = int(np.clip(np.floor(np.log2(360.0 * tiles_across / max(lon1 - lon0, 1e-6))), 3,
                    min(max_zoom, TILE_MAX_ZOOM)))
    xa, ya = _lonlat_to_tile(lon0, lat1, z)
    xb, yb = _lonlat_to_tile(lon1, lat0, z)
    xs, ys = range(int(xa), int(xb) + 1), range(int(ya), int(yb) + 1)
    n = len(xs) * len(ys)
    print(f"      · relief tiles: zoom {z}, {n} tiles (saved in gis/tiles)", flush=True)
    if n > 400:
        return None
    try:
        tiles = _download_with_progress(z, list(xs), list(ys))
        rows = [np.concatenate([tiles[(x, y)] for x in xs], axis=1) for y in ys]
    except Exception as e:
        print(f"\n      · tile download failed ({type(e).__name__}: {e}); continuing without terrain",
              flush=True)
        _tiles_failed = True
        return None
    mos = np.concatenate(rows, axis=0).astype(float).mean(axis=2) / 255.0
    bx0, _, _, by1 = _tile_bounds_merc(xs[0], ys[0], z)
    _, by0, bx1, _ = _tile_bounds_merc(xs[-1], ys[-1], z)
    src_t = from_bounds(bx0, by0, bx1, by1, mos.shape[1], mos.shape[0])
    w = px
    h = int(px * (extent[3] - extent[1]) / (extent[2] - extent[0]))
    dst_t = from_bounds(*extent, w, h)
    out = np.full((h, w), np.nan)
    reproject(mos, out, src_transform=src_t, src_crs="EPSG:3857", dst_transform=dst_t,
              dst_crs=f"EPSG:{CRS}", resampling=Resampling.bilinear, dst_nodata=np.nan)
    return out


def _offline_relief(extent):
    """Grey shaded relief from basemap-data's global raster, warped to the map CRS."""
    d = _basemap_data_dir()
    if d is None:
        return None
    try:
        import rasterio  # noqa: F401
        from PIL import Image
        from rasterio.transform import from_origin
        from rasterio.warp import Resampling, reproject, transform_bounds
        Image.MAX_IMAGE_PIXELS = None
        im = np.asarray(Image.open(d / "shadedrelief.jpg")).astype(float)
        ppd = im.shape[1] / 360.0
        lon0, lat0, lon1, lat1 = transform_bounds(f"EPSG:{CRS}", "EPSG:4326", *extent)
        lon0, lon1, lat0, lat1 = lon0 - 1, lon1 + 1, lat0 - 1, lat1 + 1
        c0, c1 = int((lon0 + 180) * ppd), int((lon1 + 180) * ppd)
        r0, r1 = int((90 - lat1) * ppd), int((90 - lat0) * ppd)
        sub = im[r0:r1, c0:c1]
        R, G, B = sub[..., 0], sub[..., 1], sub[..., 2]
        water = (B > R + 18) & (B > G)
        lum = 0.30 * R + 0.59 * G + 0.11 * B
        lum = (lum - np.percentile(lum[~water], 2)) / (np.ptp(np.percentile(lum[~water], [2, 98])) + 1e-9)
        lum = np.clip(lum, 0, 1)
        src_t = from_origin(-180 + c0 / ppd, 90 - r0 / ppd, 1 / ppd, 1 / ppd)
        w, h = 900, int(900 * (extent[3] - extent[1]) / (extent[2] - extent[0]))
        dst_t = from_origin(extent[0], extent[3], (extent[2] - extent[0]) / w, (extent[3] - extent[1]) / h)
        out = np.zeros((2, h, w), dtype=float)
        for k, band in enumerate((lum, water.astype(float))):
            reproject(band, out[k], src_transform=src_t, src_crs="EPSG:4326", dst_transform=dst_t,
                      dst_crs=f"EPSG:{CRS}", resampling=Resampling.bilinear)
        return out[0], out[1] > 0.5
    except Exception as e:
        print(f"  (offline relief unavailable: {e})")
        return None


def terrain(ax, extent, strength=0.55, allow_offline=True, max_zoom=11):
    """Shaded relief under everything: Esri tiles when reachable, bundled raster otherwise."""
    g = _tile_relief(extent, max_zoom)
    if g is not None:
        g = np.where(np.isnan(g), 1.0, g)
        g = 1 - strength * (1 - g)
        ax.imshow(g, extent=(extent[0], extent[2], extent[1], extent[3]), cmap="gray", vmin=0, vmax=1,
                  zorder=0, interpolation="bilinear")
        return "tiles"
    if not allow_offline:
        return None
    r = _offline_relief(extent)
    if r is None:
        return None
    lum, water = r
    rgb = np.dstack([1 - strength * (1 - (0.35 + 0.65 * lum))] * 3)
    wc = np.array([int(WATER[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    rgb[water] = wc
    ax.imshow(rgb, extent=(extent[0], extent[2], extent[1], extent[3]), zorder=0, interpolation="bilinear")
    return "offline"


# ------------------------------------------------------------------ base map
class BaseMap:
    """Loads reference layers once and draws a consistent Washington base map."""

    def __init__(self, seg_points=None):
        self.wa, self.nb, self.outline, self.states = counties_states()
        self.bc = province_bc()
        self.lakes = lakes()
        self.reg = regions(self.wa, seg_points) if seg_points is not None else None
        self.reg_lines = interior_borders(self.reg) if self.reg is not None else None
        b = self.outline.total_bounds
        pad = 0.035 * (b[2] - b[0])
        self.extent = (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)
        # land mask (shoreline-clipped states + BC) -> water = everything else.
        # Painting water on top of the relief makes sounds, rivers and the ocean read as water.
        big = box(b[0] - 0.4 * (b[2] - b[0]), b[1] - 0.4 * (b[3] - b[1]),
                  b[2] + 0.4 * (b[2] - b[0]), b[3] + 0.4 * (b[3] - b[1]))
        parts = [self.states.clip(big)]
        if self.bc is not None:
            parts.append(self.bc.clip(big))
        self.land = pd.concat(parts).union_all().buffer(0)
        self._big = big

    def draw(self, ax, extent=None, relief=True, counties=True, regions=True, cities=True, city_names=None,
             neighbor_labels=True, fade_outside=True, offline_relief=True):
        ext = extent or self.extent
        ax.set_facecolor(WATER)
        clip = box(*ext)
        land_drawn = False
        if relief:
            land_drawn = terrain(ax, ext, allow_offline=offline_relief) is not None
        if not land_drawn:          # flat land colour if no terrain is available
            for g in (self.nb, self.bc, self.wa):
                if g is not None:
                    g.clip(clip).plot(ax=ax, color=LAND, lw=0, zorder=0.5)
        if fade_outside:            # push neighbours back so Washington reads first
            outside = gpd.GeoSeries([clip.difference(self.outline.union_all())], crs=CRS)
            outside.plot(ax=ax, color=FADE, alpha=0.55, lw=0, zorder=1)
        water = clip.intersection(self._big).difference(self.land)
        if not water.is_empty:
            gpd.GeoSeries([water], crs=CRS).plot(ax=ax, color=WATER, lw=0, zorder=1.25)
            gpd.GeoSeries([water], crs=CRS).boundary.plot(ax=ax, color="#9fb6c8", lw=0.25, zorder=1.26)
        if self.lakes is not None:
            self.lakes.clip(clip).plot(ax=ax, color=WATER, lw=0, zorder=1.27)
        if counties:
            self.wa.boundary.plot(ax=ax, color=COUNTY, lw=0.4, zorder=1.5)
        # land borders with neighbours (dark), coastline left to the water edge (light)
        self.nb.boundary.clip(clip).intersection(self.outline.union_all().buffer(3000)) \
            .plot(ax=ax, color=STATE, lw=0.9, zorder=1.7)
        self.nb.boundary.clip(clip).plot(ax=ax, color="#a9a8a2", lw=0.5, zorder=1.6)
        if self.bc is not None:
            self.bc.boundary.clip(clip).plot(ax=ax, color="#a9a8a2", lw=0.5, zorder=1.6)
            self.bc.boundary.clip(clip).intersection(self.outline.union_all().buffer(3000)) \
                .plot(ax=ax, color=STATE, lw=0.9, zorder=1.7)
        if regions and self.reg_lines is not None:
            self.reg_lines.clip(clip).plot(ax=ax, color=REGION, lw=0.9, linestyle="--", zorder=8)
        if neighbor_labels:
            self._neighbor_labels(ax, ext)
        if cities:
            self.city_labels(ax, ext, names=city_names)
        ax.set_xlim(ext[0], ext[2])
        ax.set_ylim(ext[1], ext[3])
        ax.set_aspect("equal")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_axis_off()

    def _neighbor_labels(self, ax, ext):
        kw = dict(fontsize=7, color=LABEL, style="italic", ha="center", va="center",
                  path_effects=_halo(), zorder=9)
        x0, y0, x1, y1 = ext
        ax.text(x0 + 0.52 * (x1 - x0), y0 + 0.03 * (y1 - y0), "OREGON", **kw)
        ax.text(x1 - 0.022 * (x1 - x0), y0 + 0.45 * (y1 - y0), "IDAHO", rotation=90, **kw)
        ax.text(x0 + 0.55 * (x1 - x0), y1 - 0.025 * (y1 - y0), "BRITISH COLUMBIA", **kw)
        ax.text(x0 + 0.035 * (x1 - x0), y0 + 0.40 * (y1 - y0), "PACIFIC\nOCEAN", **dict(kw, color="#5b7a96"))

    def city_labels(self, ax, ext, names=None, size=6.3):
        pts = gpd.GeoDataFrame(CITIES, columns=["name", "lat", "lon", "off"],
                               geometry=[Point(c[2], c[1]) for c in CITIES], crs=4326).to_crs(CRS)
        x0, y0, x1, y1 = ext
        for _, r in pts.iterrows():
            if names and r["name"] not in names:
                continue
            x, y = r.geometry.x, r.geometry.y
            if not (x0 < x < x1 and y0 < y < y1):
                continue
            ax.plot(x, y, "s", ms=2.6, mfc="white", mec="#222222", mew=0.6, zorder=21)
            ax.annotate(r["name"], (x, y), xytext=r["off"], textcoords="offset points",
                        ha="right" if r["off"][0] < 0 else "left",
                        fontsize=size, color="#222222", path_effects=_halo(), zorder=21)

    def region_labels(self, ax):
        if self.reg is None:
            return
        at = gpd.GeoSeries([Point(*REGION_LABEL_AT[n]) if n in REGION_LABEL_AT else None
                            for n in self.reg["name"]], crs=4326).to_crs(CRS)
        for (_, r), q in zip(self.reg.iterrows(), at):
            p = q if q is not None and not q.is_empty else r.geometry.representative_point()
            ax.text(p.x, p.y, r["name"].upper(), fontsize=6.2, color=REGION, ha="center",
                    va="center", fontweight="bold", alpha=0.75, path_effects=_halo(2.5), zorder=9)

    def locator(self, ax, rect=(0.0, 0.70, 0.22, 0.30), zoom_box=None):
        """Inset: Pacific Northwest with Washington (and optionally a zoom box) highlighted."""
        ia = ax.inset_axes(rect)
        st = self.states.to_crs(5070)
        lower48 = st[~st["STATE_FIPS"].isin(["02", "15", "60", "66", "69", "72", "78"])]
        lower48.plot(ax=ia, color="#e9e8e3", edgecolor="white", lw=0.3)
        lower48[lower48["STATE_FIPS"] == WA_FIPS].plot(ax=ia, color="#3d3c39", lw=0)
        if zoom_box is not None:
            gpd.GeoSeries([box(*zoom_box)], crs=CRS).to_crs(5070).boundary.plot(ax=ia, color="#c7501f", lw=1)
        b = lower48.total_bounds
        ia.set_xlim(b[0], b[2])
        ia.set_ylim(b[1], b[3])
        ia.set_aspect("equal")
        ia.set_xticks([])
        ia.set_yticks([])
        ia.set_xlabel("")
        ia.set_ylabel("")
        for s in ia.spines.values():
            s.set_visible(True)
            s.set_edgecolor("#b9b7ae")
            s.set_linewidth(0.5)
        ia.set_facecolor("white")
        return ia


def scalebar(ax, miles, loc=(0.04, 0.05), fs=6.5, bg=False):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    L = miles * 5280.0
    bx, by = x0 + loc[0] * (x1 - x0), y0 + loc[1] * (y1 - y0)
    h = 0.010 * (y1 - y0)
    if bg:      # white card behind the bar so it reads over lines and points
        from matplotlib.patches import FancyBboxPatch
        px, py = 0.035 * (x1 - x0), 0.03 * (y1 - y0)
        ax.add_patch(FancyBboxPatch((bx - px, by - py), L + 2 * px, 9 * h + py,
                                    boxstyle="round,pad=0,rounding_size=%f" % (0.01 * (x1 - x0)),
                                    fc="white", ec="none", alpha=0.85, zorder=29))
    for k in range(4):
        ax.add_patch(plt.Rectangle((bx + k * L / 4, by), L / 4, h, fc="#222222" if k % 2 == 0 else "white",
                                   ec="#222222", lw=0.5, zorder=30))
    for frac, lab in ((0, "0"), (0.5, f"{miles / 2:g}"), (1, f"{miles:g} mi")):
        ax.text(bx + frac * L, by + 1.9 * h, lab, ha="center", va="bottom", fontsize=fs, color="#222222",
                path_effects=_halo(), zorder=30)


def north_arrow(ax, loc=(0.965, 0.80), size=0.06):
    x, y = loc
    ax.annotate("", xy=(x, y + size), xytext=(x, y), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="wedge,tail_width=0.5", fc="#222222", ec="#222222", lw=0.5),
                zorder=30)
    ax.text(x, y - 0.012, "N", transform=ax.transAxes, ha="center", va="top", fontsize=7,
            fontweight="bold", color="#222222", path_effects=_halo(), zorder=30)


def county_stats(wa_counties, seg, crashes=None):
    """Aggregate segment-years to counties (by segment midpoint)."""
    pts = gpd.GeoDataFrame(seg, geometry="geometry", crs=CRS)
    j = gpd.sjoin(pts, wa_counties[["NAME", "geometry"]], predicate="within", how="inner")
    return j
