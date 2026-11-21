"""
05_spatial_hotspots.py
----------------------
(1) Formal test of residual spatial clustering (Reviewer 3, RQ3)
    Global Moran's I (KNN k=8, 999 permutations) on, in turn:
      a. raw severe-crash rate per VMT
      b. Pearson residuals of an exposure-only NB (log AADT + log length + year)
      c. Pearson residuals of the full SPF (model D)
    The drop from (a)/(b) to (c) shows how much clustering the covariates
    explain; any remaining I > 0 is residual clustering. Local Moran (LISA)
    clusters of (c) are mapped.

(2) Hotspot ranking stability (Reviewer 3)
    * Empirical Bayes (Hauer) excess severe crashes from model D, pooled
      2021-2023 - threshold-free ranking used as the primary screen
    * raw-rate top-10 under every VMT x minimum-crash threshold pair, compared
      with Jaccard overlap
    * year-by-year top-10 overlap
    * posterior simulation: probability each segment appears in the top 10
      when counts are redrawn from their NB posterior

Requires geopandas, pyogrio, libpysal, esda.
"""
import itertools

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local

import config as cfg
from utils import fit_nb, save

# ---- geometry of road segments (layer + key names confirmed from schema report)
GDB_ROAD_LAYER = {2021: "wa21road", 2022: "wa22road", 2023: "wa23road"}
GDB_KEYS = {"route": "ROUTE_ID", "begmp": "BEGMP"}
CRS_PROJ = 2927


def segment_points():
    """One representative point per seg_key (midpoint of the polyline)."""
    frames = []
    for yr, layer in GDB_ROAD_LAYER.items():
        g = gpd.read_file(cfg.GDB, layer=layer, engine="pyogrio",
                          columns=list(GDB_KEYS.values())).to_crs(CRS_PROJ)
        g = g.rename(columns={v: k for k, v in GDB_KEYS.items()})
        g["route"] = g["route"].astype(str).str.strip()
        g["seg_key"] = g["route"] + "_" + g["begmp"].astype(float).round(3).astype(str)
        g = g[g.geometry.notna() & ~g.geometry.is_empty]
        g["geometry"] = g.geometry.force_2d().line_merge().interpolate(0.5, normalized=True)
        frames.append(g[["seg_key", "geometry"]])
    pts = pd.concat(frames).drop_duplicates("seg_key")
    return gpd.GeoDataFrame(pts, crs=CRS_PROJ)


def pearson_by_segment(df, mu, alpha):
    var = mu + alpha * mu ** 2
    a = df.assign(mu=mu, var=var).groupby("seg_key").agg(
        y=("n_severe", "sum"), mu=("mu", "sum"), var=("var", "sum"),
        vmt=("vmt", "sum"), rural=("rural", "max"), any_curve=("any_curve", "max"),
        route=("route", "first"))
    a["pearson"] = (a["y"] - a["mu"]) / np.sqrt(a["var"])
    a["rate"] = a["y"] / (a["vmt"] / 1e8)
    return a


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else np.nan


def main():
    sg = pd.read_parquet(cfg.OUT / "data" / "segment_year_fitted.parquet")
    alpha = sg["alpha_P"].iloc[0]

    # exposure-only NB for comparison
    m0 = fit_nb("n_severe ~ log_aadt + log_len + C(year)", sg, "district")
    mu0 = m0.predict(sg, offset=np.zeros(len(sg)))

    segD = pearson_by_segment(sg, sg["mu_P"], alpha)
    seg0 = pearson_by_segment(sg, mu0, m0.params["alpha"])

    pts = segment_points().set_index("seg_key")
    common = segD.index.intersection(pts.index)
    print(f"segments with geometry: {len(common):,} of {len(segD):,}")
    gd = gpd.GeoDataFrame(segD.loc[common], geometry=pts.loc[common, "geometry"],
                          crs=CRS_PROJ)
    gd["pearson0"] = seg0.loc[common, "pearson"]
    w = KNN.from_dataframe(gd, k=8)
    w.transform = "r"

    rows = []
    for col, lab in [("rate", "raw rate per 100M VMT"),
                     ("pearson0", "residual, exposure-only NB"),
                     ("pearson", "residual, full SPF (E)")]:
        x = gd[col].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
        mi = Moran(x, w, permutations=999)
        rows.append([lab, mi.I, mi.EI, mi.z_sim, mi.p_sim])
    save(pd.DataFrame(rows, columns=["variable", "Moran_I", "E[I]", "z", "p_perm"])
         .set_index("variable"), "tab_moran", "%.4f")

    lisa = Moran_Local(gd["pearson"].to_numpy(), w, permutations=999, seed=cfg.SEED)
    q = np.where(lisa.p_sim < 0.05, lisa.q, 0)
    gd["lisa"] = pd.Series(q, index=gd.index).map(
        {0: "not significant", 1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"})
    gd["x"], gd["y"] = gd.geometry.x, gd.geometry.y      # same points as the Moran test
    gd.drop(columns="geometry").to_csv(cfg.OUT / "tables" / "segment_residuals_lisa.csv")

    # ========================= hotspots
    rc = segD[(segD["rural"] == 1) & (segD["any_curve"] == 1)].copy()
    wgt = 1 / (1 + alpha * rc["mu"])
    rc["eb"] = wgt * rc["mu"] + (1 - wgt) * rc["y"]
    rc["eb_excess"] = rc["eb"] - rc["mu"]
    top_eb = rc.nlargest(cfg.TOP_K, "eb_excess")
    save(top_eb[["route", "y", "mu", "eb", "eb_excess", "vmt"]], "tab_hotspots_eb", "%.3f")

    # raw-rate rankings across thresholds (annual VMT thresholds -> 3-yr totals)
    nyr = len(cfg.YEARS)
    tops = {}
    for v, k in itertools.product(cfg.HOTSPOT_VMT_GRID, cfg.HOTSPOT_MINCRASH_GRID):
        s = rc[(rc["vmt"] >= v * nyr) & (rc["y"] >= k)]
        tops[(v / 1e6, k)] = list(s.nlargest(cfg.TOP_K, "rate").index)
    tops[("EB", "-")] = list(top_eb.index)
    keys = list(tops)
    J = pd.DataFrame([[jaccard(tops[a], tops[b]) for b in keys] for a in keys],
                     index=[str(k) for k in keys], columns=[str(k) for k in keys])
    save(J, "tab_hotspot_threshold_jaccard", "%.2f")

    # year-to-year stability
    yr_tops = {}
    for yr in cfg.YEARS:
        d = sg[(sg["year"] == yr) & (sg["rural"] == 1) & (sg["any_curve"] == 1)]
        d = d[(d["vmt"] >= 0.05e6) & (d["n_severe"] >= 1)]
        yr_tops[yr] = set(d.assign(rate=d["n_severe"] / d["vmt"]).nlargest(cfg.TOP_K, "rate")["seg_key"])
    yy = pd.DataFrame([[jaccard(yr_tops[a], yr_tops[b]) for b in cfg.YEARS] for a in cfg.YEARS],
                      index=cfg.YEARS, columns=cfg.YEARS)
    save(yy, "tab_hotspot_year_jaccard", "%.2f")

    # posterior simulation of top-10 membership
    rng = np.random.default_rng(cfg.SEED)
    shape = 1 / alpha + rc["y"].to_numpy()
    rate_ = (1 / alpha + rc["mu"].to_numpy()) / rc["mu"].to_numpy()
    base = rc[(rc["vmt"] >= 0.05e6 * nyr)]
    idx = rc.index.get_indexer(base.index)
    cnt_raw = pd.Series(0, index=base.index)
    cnt_eb = pd.Series(0, index=rc.index)
    S = 1000
    for _ in range(S):
        lam = rng.gamma(shape, 1 / rate_)
        ystar = rng.poisson(lam)
        r = pd.Series(ystar[idx] / base["vmt"].to_numpy(), index=base.index)
        r = r[ystar[idx] >= 2]
        cnt_raw[r.nlargest(cfg.TOP_K).index] += 1
        ebs = wgt.to_numpy() * rc["mu"].to_numpy() + (1 - wgt.to_numpy()) * ystar - rc["mu"].to_numpy()
        cnt_eb[rc.index[np.argsort(-ebs)[:cfg.TOP_K]]] += 1
    raw_p = (cnt_raw / S).reindex(tops[(0.05, 2)]).reset_index(drop=True)
    eb_p = (cnt_eb / S).reindex(top_eb.index).reset_index(drop=True)
    stab = pd.concat({"P(top10) raw-rate, original top10": raw_p,
                      "P(top10) EB, original top10": eb_p}, axis=1)
    stab.index = stab.index + 1
    save(stab, "tab_hotspot_posterior_stability", "%.3f")


if __name__ == "__main__":
    main()
