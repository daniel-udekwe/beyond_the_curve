"""
01_build_datasets.py
--------------------
Builds the analysis files for WA 2021-2023 from the HSIS extract:

  data/crash_level.parquet    one row per state-route crash linked to the inventory
  data/segment_year.parquet   one row per roadway segment x year
  data/segment_fixed.parquet  same, on fixed-length (0.1 mi) segments  [sensitivity]
  logs/linkage_log.txt        counts at every filtering / linkage step

Linking rules (documented in the paper, Section 3.2):
  * key = ROUTE_ID + milepost. Crashes on both directions of a divided highway are
    coded to the increasing ('...i') ROUTE_ID; the decreasing-roadway inventory
    records ('...d' with an '...i' twin) are therefore dropped from road, curve and
    grade files, and the few crashes coded to such '...d' IDs are re-keyed to '...i'.
    Ramp/couplet IDs that exist only as '...d' are kept.
  * crash -> road segment and crash -> curve: interval join, BEGMP <= MP < ENDMP
  * grade: the grade file holds vertical PVIs. The grade in force between two
    PVIs is the 'grade ahead' of the earlier PVI, so PVIs are converted to grade
    intervals; crashes take the interval containing them, segments take the max
    (and length-weighted mean) |grade| over overlapping intervals.
  * segment curvature: sharpest overlapping curve (min radius) plus number of
    overlapping curves, share of length on curve, length-weighted curvature.
  * AADT, lanes, speed, rurality, district come from the road file of the SAME year;
    zeros in AADT/lanes/speed are treated as missing.
"""
import re

import numpy as np
import pandas as pd

import config as cfg

LOG = []


def log(msg):
    print(msg)
    LOG.append(str(msg))


# ------------------------------------------------------------------ loading
def load(kind, year):
    path = cfg.raw_file(kind, year)
    df = None
    for enc in ("utf-8-sig", "latin-1"):      # utf-8-sig strips a byte-order mark
        try:
            df = pd.read_csv(path, low_memory=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    # strip any stray BOM / whitespace from header names
    df.columns = [str(c).replace("\ufeff", "").replace("ï»¿", "").strip() for c in df.columns]
    cmap = {v: k for k, v in cfg.COLS[kind].items() if v in df.columns}
    missing = [v for v in cfg.COLS[kind].values() if v not in df.columns]
    if missing:
        log(f"  [{path.name}] columns not found: {missing}")
    df = df.rename(columns=cmap)[list(cmap.values())].copy()
    df["year"] = year
    if "route" in df:
        r = df["route"]
        df["route"] = r.where(r.isna(), r.astype(str).str.strip()).astype(object)
    log(f"  loaded {path.name}: {len(df):,} rows")
    return df


def num(s, zero_is_missing=False, lo=None, hi=None):
    x = pd.to_numeric(s, errors="coerce")
    if zero_is_missing:
        x = x.where(x != 0)
    if lo is not None:
        x = x.where(x >= lo)
    if hi is not None:
        x = x.where(x <= hi)
    return x


def parse_angle(s):
    """'005*05.1' (deg*min) or decimal degrees -> decimal degrees."""
    def one(v):
        if pd.isna(v):
            return np.nan
        v = str(v).strip()
        m = re.match(r"^(\d+)\*(\d+(?:\.\d+)?)$", v)
        if m:
            return float(m.group(1)) + float(m.group(2)) / 60
        try:
            return float(v)
        except ValueError:
            return np.nan
    return s.map(one)


# ----------------------------------------------------- direction handling
def twin_d_ids(ids):
    """'...d' route IDs whose '...i' twin exists (decreasing roadway of a divided highway)."""
    ids = set(ids.dropna())
    return {r for r in ids if r.endswith("d") and (r[:-1] + cfg.INVENTORY_SUFFIX) in ids}


def drop_twin_d(df, twins, name):
    m = df["route"].isin(twins)
    log(f"  {name}: dropped {m.sum():,} decreasing-roadway records ('...d' with '...i' twin)")
    return df[~m].copy()


# ----------------------------------------------------- interval utilities
def interval_join_points(points, intervals, mp_col="mp", cols=()):
    """Attributes of the interval containing each point (route, mp).
    Where intervals overlap, the one with the latest BEGMP <= mp is used."""
    pts = points[["route", mp_col]].dropna().reset_index().rename(columns={"index": "_pid"})
    pts = pts.sort_values(mp_col)
    pts["route"] = pts["route"].astype(object)
    iv = intervals.dropna(subset=["route", "begmp", "endmp"]).sort_values("begmp").copy()
    iv["route"] = iv["route"].astype(object)
    m = pd.merge_asof(pts, iv, left_on=mp_col, right_on="begmp", by="route",
                      direction="backward")
    inside = (m[mp_col] >= m["begmp"]) & (m[mp_col] < m["endmp"])
    m.loc[~inside, list(cols)] = np.nan
    return m.set_index("_pid").reindex(points.index)


def overlaps(segs, ivs, id_col):
    """All (segment, interval) pairs that overlap, with overlap length (miles).
    segs need route, begmp, endmp, seg_id; ivs need route, begmp, endmp, id_col."""
    out = []
    seg_by_route = {r: g.sort_values("begmp") for r, g in segs.groupby("route")}
    for route, cg in ivs.groupby("route"):
        sg = seg_by_route.get(route)
        if sg is None or sg.empty:
            continue
        sb, se = sg["begmp"].to_numpy(), sg["endmp"].to_numpy()
        sid = sg["seg_id"].to_numpy()
        for cid, cb, ce in cg[[id_col, "begmp", "endmp"]].itertuples(index=False):
            lo = np.searchsorted(se, cb, side="right")
            hi = np.searchsorted(sb, ce, side="left")
            for j in range(lo, hi):
                ov = min(se[j], ce) - max(sb[j], cb)
                if ov > 0:
                    out.append((sid[j], cid, ov))
    return pd.DataFrame(out, columns=["seg_id", id_col, "ov_mi"])


# ----------------------------------------------------------- preparation
def prep_road(road, year):
    road["begmp"], road["endmp"] = num(road["begmp"]), num(road["endmp"])
    road = road[road["endmp"] > road["begmp"]].copy()
    road["seg_id"] = [f"{year}_{i}" for i in range(len(road))]
    road["len_mi"] = road["endmp"] - road["begmp"]
    road["aadt"] = num(road["aadt"], zero_is_missing=True)
    road["lanes"] = num(road["lanes"], zero_is_missing=True)
    road["speed"] = num(road["speed"], zero_is_missing=True)
    road["vmt"] = road["aadt"] * road["len_mi"] * 365
    fc = road["func_cls"].astype("string")
    rc = road["rodwycls"].astype("string") if "rodwycls" in road else fc
    rural = fc.str.startswith(cfg.RURAL_PREFIX)
    rural = rural.where(fc.notna(), rc.str.startswith(cfg.RURAL_PREFIX))
    road["rural"] = rural.astype("Float64").astype(float)
    road["district"] = num(road["district"], zero_is_missing=True)
    road["divided"] = (road["divided"] == "Y").astype(float).where(road["divided"].notna())
    log(f"  road: {len(road):,} segments; AADT missing {road['aadt'].isna().mean():.1%}, "
        f"speed missing {road['speed'].isna().mean():.1%}, lanes missing "
        f"{road['lanes'].isna().mean():.1%}, rurality missing {road['rural'].isna().mean():.1%}")
    return road


def prep_curve(curve):
    for c in ("begmp", "endmp", "radius", "length", "superel"):
        if c in curve:
            curve[c] = num(curve[c])
    curve["radius"] = curve["radius"].where(curve["radius"] >= cfg.MIN_RADIUS_FT)
    curve["angle"] = parse_angle(curve["angle"])
    n0 = len(curve)
    curve = curve[(curve["endmp"] > curve["begmp"]) & curve["radius"].notna()].copy()
    log(f"  curves: kept {len(curve):,} of {n0:,} (dropped zero-length or radius < "
        f"{cfg.MIN_RADIUS_FT} ft / missing)")
    curve["curve_id"] = np.arange(len(curve))
    curve["curv"] = cfg.CURV_SCALE / curve["radius"]
    return curve


def grade_intervals(grade):
    """PVI points -> intervals with the grade in force (percent, signed)."""
    g = grade.copy()
    g["mp"] = num(g["mp"])
    for c in ("g_back", "g_ahead"):
        g[c] = num(g[c]).where(lambda x: x.abs() <= cfg.MAX_ABS_GRADE)
    g = g.dropna(subset=["route", "mp"]).sort_values(["route", "mp"])
    g = g.drop_duplicates(["route", "mp"], keep="last")
    rows = []
    for route, d in g.groupby("route"):
        mp, ga, gb = d["mp"].to_numpy(), d["g_ahead"].to_numpy(), d["g_back"].to_numpy()
        # before the first PVI: its grade back
        rows.append((route, max(0.0, mp[0] - cfg.MAX_GRADE_SPAN_MI), mp[0], gb[0]))
        for k in range(len(mp)):
            end = mp[k + 1] if k + 1 < len(mp) else mp[k] + cfg.MAX_GRADE_SPAN_MI
            end = min(end, mp[k] + cfg.MAX_GRADE_SPAN_MI)
            g_k = ga[k] if not np.isnan(ga[k]) else (gb[k + 1] if k + 1 < len(mp) else np.nan)
            rows.append((route, mp[k], end, g_k))
    iv = pd.DataFrame(rows, columns=["route", "begmp", "endmp", "grade"])
    iv = iv[(iv["endmp"] > iv["begmp"]) & iv["grade"].notna()].copy()
    iv["abs_grade"] = iv["grade"].abs()
    iv["grade_id"] = np.arange(len(iv))
    log(f"  grade: {len(g):,} PVIs -> {len(iv):,} grade intervals")
    return iv


def recode_crash(df):
    sev = df["severity"].map({k: v[1] for k, v in cfg.SEVERITY_MAP.items()})
    df["kabco"] = df["severity"].map({k: v[0] for k, v in cfg.SEVERITY_MAP.items()})
    df["sev4"] = sev                                   # 0 O, 1 B/C, 2 A, 3 K
    df["severe"] = (sev >= 2).astype(float).where(sev.notna())
    df["alcohol"] = df["alcohol"].isin(cfg.ALCOHOL_YES).astype(int)
    light = df["light"].astype("string")
    df["dark"] = light.str.startswith(cfg.DARK_PREFIX).fillna(False).astype(int)
    df["dark_unlit"] = light.isin(cfg.DARK_UNLIT_CODES).fillna(False).astype(int)
    df["wet"] = df["surface"].isin(cfg.WET_CODES).astype(int)
    df["workzone"] = df["workzone"].isin(cfg.WORKZONE_YES).astype(int)
    df["ramp"] = df["rd_type"].astype("string").str.contains(cfg.RAMP_PATTERN) \
        .fillna(False).astype(int)
    df["multi_veh"] = (num(df["numvehs"]) > 1).astype(int) if "numvehs" in df else 0
    for c in ("x", "y"):                     # coordinates are only needed for maps
        df[c] = num(df[c]) if c in df else np.nan
    return df


def unit_summary(unit):
    unit["spd"] = num(unit["spdlimit"], lo=15, hi=80)
    unit["heavy"] = unit["vehtype"].isin(cfg.HEAVY_VEH).astype(int)
    unit["caseno"] = unit["caseno"].astype(str)
    return unit.groupby("caseno").agg(unit_speed=("spd", "max"), heavy_veh=("heavy", "max"))


# ----------------------------------------------------------- per-year build
def build_year(year):
    log(f"\n=== {year} ===")
    crash = recode_crash(load("crash", year))
    gy = cfg.GEOMETRY_YEAR.get(year, year)
    if gy != year:
        log(f"  curve/grade inventories taken from the {gy} release")
    road, curve, grade = load("road", year), load("curve", gy), load("grade", gy)
    units = unit_summary(load("unit", year))

    # ---- direction handling (see module docstring)
    twins = twin_d_ids(road["route"])
    road = drop_twin_d(road, twins, "road")
    curve = drop_twin_d(curve, twins, "curve")
    grade = drop_twin_d(grade, twins, "grade")
    road, curve, giv = prep_road(road, year), prep_curve(curve), grade_intervals(grade)

    crash["mp"] = num(crash["mp"])
    n0 = len(crash)
    crash = crash[crash["route"].notna() & crash["mp"].notna()].copy()
    log(f"  crashes total {n0:,}; with ROUTE_ID + milepost {len(crash):,}")
    kept = set(road["route"].dropna())
    rekey = crash["route"].map(lambda r: isinstance(r, str) and r.endswith("d") and r not in kept
                               and (r[:-1] + cfg.INVENTORY_SUFFIX) in kept).astype(bool)
    crash.loc[rekey, "route"] = crash.loc[rekey, "route"].map(lambda r: r[:-1] + cfg.INVENTORY_SUFFIX)
    log(f"  crashes re-keyed from '...d' to '...i': {rekey.sum():,}")
    log(f"  crash SEVERITY Unknown (excluded from models): {crash['severe'].isna().sum():,}")

    # ================= crash-level file
    seg_cols = ["seg_id", "len_mi", "aadt", "vmt", "lanes", "speed", "rural", "district",
                "divided", "func_cls", "shld_w"]
    seg_cols = [c for c in seg_cols if c in road]
    j = interval_join_points(crash, road[["route", "begmp", "endmp"] + seg_cols], cols=seg_cols)
    for c in seg_cols:
        crash[c] = j[c]
    log(f"  crashes linked to a road segment: {crash['seg_id'].notna().sum():,} "
        f"({crash['seg_id'].notna().mean():.1%})")
    # rurality fallback from the crash record; speed fallback from unit records
    rr = crash["rur_urb_crash"].astype("string").str.startswith(cfg.RURAL_PREFIX)
    crash["rural"] = crash["rural"].fillna(rr.astype("Float64").astype(float))
    crash["caseno"] = crash["caseno"].astype(str)
    crash = crash.join(units, on="caseno")
    crash["speed_src"] = np.where(crash["speed"].notna(), "road",
                                  np.where(crash["unit_speed"].notna(), "unit", "none"))
    crash["speed"] = crash["speed"].fillna(crash["unit_speed"])
    log(f"  crash speed source: {crash['speed_src'].value_counts().to_dict()}")

    ccols = ["curv", "radius", "length", "angle", "superel"]
    jc = interval_join_points(crash, curve[["route", "begmp", "endmp"] + ccols], cols=ccols)
    crash["on_curve"] = jc["curv"].notna().astype(int)
    crash["curv"] = jc["curv"].fillna(0.0)
    crash["radius"] = jc["radius"]
    crash["curve_len_ft"] = jc["length"].fillna(0.0)
    crash["curve_angle"] = jc["angle"].fillna(0.0)
    crash["superel"] = jc["superel"]

    jg = interval_join_points(crash, giv[["route", "begmp", "endmp", "abs_grade"]],
                              cols=["abs_grade"])
    crash["abs_grade"] = jg["abs_grade"]
    for cut in cfg.GRADE_CUTS:
        crash[f"steep_{cut:g}"] = (crash["abs_grade"] >= cut).astype(int) \
            .where(crash["abs_grade"].notna())
    crash["steep"] = crash[f"steep_{cfg.STEEP_GRADE:g}"]
    lk = crash["seg_id"].notna()
    log(f"  linked crashes: on curve {crash.loc[lk, 'on_curve'].mean():.1%}; "
        f"grade available {crash.loc[lk, 'abs_grade'].notna().mean():.1%}; "
        f"steep (>= {cfg.STEEP_GRADE}%) {crash.loc[lk, 'steep'].mean():.1%}")
    # agreement with officer-coded curve (RD_CHAR1) as a linkage sanity check
    if "rd_char" in crash:
        oc = crash.loc[lk, "rd_char"].astype("string").str.startswith("Curve")
        ok = oc.notna()
        log(f"  inventory curve vs officer-coded curve agreement: "
            f"{(oc[ok] == crash.loc[lk, 'on_curve'][ok].astype(bool)).mean():.1%}")

    # ================= segment-year file
    seg = road.set_index("seg_id")
    ov = overlaps(road, curve, "curve_id").merge(
        curve[["curve_id", "curv", "length", "angle", "superel"]], on="curve_id")
    if len(ov):
        g = ov.groupby("seg_id")
        sharp = ov.loc[g["curv"].idxmax()].set_index("seg_id")
        seg["curv_max"] = sharp["curv"]
        seg["curve_len_ft"] = sharp["length"]
        seg["curve_angle"] = sharp["angle"]
        seg["superel_sharpest"] = sharp["superel"]
        seg["n_curves"] = g.size()
        seg["curve_ov_mi"] = g["ov_mi"].sum()
        ov["w"] = ov["curv"] * ov["ov_mi"]
        seg["curv_lenwt"] = ov.groupby("seg_id")["w"].sum() / seg["len_mi"]
    for c in ("curv_max", "curve_len_ft", "curve_angle", "n_curves", "curve_ov_mi", "curv_lenwt"):
        seg[c] = seg[c].fillna(0.0) if c in seg else 0.0
    seg["share_on_curve"] = (seg["curve_ov_mi"] / seg["len_mi"]).clip(upper=1)
    seg["any_curve"] = (seg["n_curves"] > 0).astype(int)

    og = overlaps(road, giv, "grade_id").merge(giv[["grade_id", "abs_grade"]], on="grade_id")
    if len(og):
        gg = og.groupby("seg_id")
        seg["grade_max"] = gg["abs_grade"].max()
        og["w"] = og["abs_grade"] * og["ov_mi"]
        seg["grade_mean"] = og.groupby("seg_id")["w"].sum() / og.groupby("seg_id")["ov_mi"].sum()
    else:
        seg["grade_max"] = seg["grade_mean"] = np.nan
    for cut in cfg.GRADE_CUTS:
        seg[f"steep_{cut:g}"] = (seg["grade_max"] >= cut).astype(int).where(seg["grade_max"].notna())

    lc = crash.dropna(subset=["seg_id"])
    seg["n_crash"] = lc.groupby("seg_id").size()
    seg["n_severe"] = lc[lc["severe"] == 1].groupby("seg_id").size()
    seg["n_unknown_sev"] = lc[lc["severe"].isna()].groupby("seg_id").size()
    for c in ("n_crash", "n_severe", "n_unknown_sev"):
        seg[c] = seg[c].fillna(0).astype(int)
    seg["year"] = year
    seg = seg.reset_index()

    # ---- diagnostics: where AADT is missing, and how much inventory actually links
    mainline = seg["route"].astype(str).str.fullmatch(r"\d{3}[id]")
    nov = seg["vmt"].isna() | (seg["vmt"] <= 0)
    log(f"  AADT missing: mainline {nov[mainline].mean():.1%} of {mainline.sum():,} segs; "
        f"related roadways (ramps, couplets, ...) {nov[~mainline].mean():.1%} of "
        f"{(~mainline).sum():,} segs")
    log(f"  severe crashes on segments without AADT: {seg.loc[nov, 'n_severe'].sum():,} of "
        f"{seg['n_severe'].sum():,} (mainline {seg.loc[nov & mainline, 'n_severe'].sum():,})")
    log(f"  mainline road-miles on curve: "
        f"{seg.loc[mainline, 'curve_ov_mi'].sum() / seg.loc[mainline, 'len_mi'].sum():.1%}; "
        f"curves overlapping a kept segment: {ov['curve_id'].nunique() if len(ov) else 0:,} of "
        f"{len(curve):,}; grade intervals overlapping: "
        f"{og['grade_id'].nunique() if len(og) else 0:,} of {len(giv):,}")
    log(f"  segments {len(seg):,}; with VMT>0 {(seg['vmt'] > 0).sum():,}; "
        f"with grade {seg['grade_max'].notna().mean():.1%}; any curve {seg['any_curve'].mean():.1%}; "
        f"severe crashes on segments {seg['n_severe'].sum():,}")
    return crash, seg, road, curve, giv


# --------------------------------------------- fixed-length re-segmentation
def fixed_segments(road, curve, giv, crash, year, step=cfg.FIXED_SEG_LEN_MI):
    rows = []
    for route, rg in road.groupby("route"):
        lo, hi = rg["begmp"].min(), rg["endmp"].max()
        edges = np.arange(np.floor(lo / step) * step, hi + step, step)
        rows += [(route, b, e) for b, e in zip(edges[:-1], edges[1:])]
    fx = pd.DataFrame(rows, columns=["route", "begmp", "endmp"])
    fx["seg_id"] = [f"F{year}_{i}" for i in range(len(fx))]
    fx["len_mi"] = step
    mid = fx.assign(mp=(fx["begmp"] + fx["endmp"]) / 2)
    attr = ["aadt", "lanes", "speed", "rural", "district"]
    j = interval_join_points(mid, road[["route", "begmp", "endmp"] + attr], cols=attr)
    for a in attr:
        fx[a] = j[a].values
    fx = fx.dropna(subset=["aadt"]).copy()
    fx["vmt"] = fx["aadt"] * fx["len_mi"] * 365
    fx = fx.set_index("seg_id")
    ov = overlaps(fx.reset_index(), curve, "curve_id").merge(curve[["curve_id", "curv"]],
                                                              on="curve_id")
    fx["curv_max"] = ov.groupby("seg_id")["curv"].max()
    fx["curv_max"] = fx["curv_max"].fillna(0.0)
    og = overlaps(fx.reset_index(), giv, "grade_id").merge(giv[["grade_id", "abs_grade"]],
                                                           on="grade_id")
    fx["grade_max"] = og.groupby("seg_id")["abs_grade"].max()
    jc = interval_join_points(crash, fx.reset_index()[["route", "begmp", "endmp", "seg_id"]],
                              cols=["seg_id"])
    cr = crash.assign(fseg=jc["seg_id"]).dropna(subset=["fseg"])
    fx["n_severe"] = cr[cr["severe"] == 1].groupby("fseg").size()
    fx["n_severe"] = fx["n_severe"].fillna(0).astype(int)
    fx["year"] = year
    return fx.reset_index()


def main():
    crashes, segs, fixed = [], [], []
    for y in cfg.YEARS:
        crash, seg, road, curve, giv = build_year(y)
        crashes.append(crash)
        segs.append(seg)
        fixed.append(fixed_segments(road, curve, giv, crash, y))
    crash = pd.concat(crashes, ignore_index=True)
    seg = pd.concat(segs, ignore_index=True)
    fx = pd.concat(fixed, ignore_index=True)

    # stable segment key across years (for panel random effects / clustering)
    seg["seg_key"] = seg["route"].astype(str) + "_" + seg["begmp"].round(3).astype(str)
    crash["district_name"] = crash["district"].map(cfg.DISTRICT_NAMES)
    for d in (crash, seg, fx):
        for c in d.columns:
            if str(d[c].dtype) == "string":
                d[c] = d[c].astype(object)

    crash.to_parquet(cfg.OUT / "data" / "crash_level.parquet", index=False)
    seg.to_parquet(cfg.OUT / "data" / "segment_year.parquet", index=False)
    fx.to_parquet(cfg.OUT / "data" / "segment_fixed.parquet", index=False)
    log(f"\nTOTAL linked crash rows {crash['seg_id'].notna().sum():,} of {len(crash):,}; "
        f"segment-years {len(seg):,}; fixed bins {len(fx):,}")
    log(f"segments whose key appears in all {len(cfg.YEARS)} years: "
        f"{(seg.groupby('seg_key')['year'].nunique() == len(cfg.YEARS)).mean():.1%}")
    (cfg.OUT / "logs" / "linkage_log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
