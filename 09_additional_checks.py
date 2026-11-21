"""
09_additional_checks.py
-----------------------
Additional robustness checks on the central findings. Run after 03 and 05.

1. Curve classification
   a. agreement between the inventory curve indicator and the officer-coded
      roadway character (RD_CHAR1), overall and separately for rural and urban
      crashes (sensitivity, specificity, Cohen's kappa)
   b. crash-level model A re-estimated with the officer-coded curve indicator,
      and on crashes where inventory and officer coding agree

2. Flexible curvature (segment level and crash level)
   Natural cubic regression splines in log radius, applied to curved segments
   only (zero on tangents), added to the presence-only model, with and without
   curve length and central angle, common to both settings and separately by
   rurality. Likelihood-ratio tests against the presence-only model, and the
   implied curve-vs-tangent ratio over radius for rural and urban settings.

3. Spatially correlated omitted factors
   Model E with WSDOT-region fixed effects and with county fixed effects, so the
   curve x rural contrast is identified within regions / within counties.

4. Empirical Bayes uncertainty
   For each rural curved segment: posterior (gamma) 90% interval of the expected
   three-year KA count, P(expected > SPF prediction) and P(expected > 2x SPF
   prediction). Reported for the EB top 10 and as network counts.

Outputs (analysis_out/tables): tab_curve_agreement, tab_officer_curve_logit,
tab_spline_lr, tab_spline_curve_effect, tab_fixed_effects, tab_eb_uncertainty
"""
import importlib

import geopandas as gpd
import numpy as np
import pandas as pd
import patsy
from scipy import stats

import config as cfg
from utils import fit_logit, fit_nb, save, coef_table, fit_stats

M = importlib.import_module("03_models")
S5 = importlib.import_module("05_spatial_hotspots")

SPLINE_DF = 4
RADII = [int(round(r)) for r in np.geomspace(150, 5000, 30)]


# ------------------------------------------------------------------ helpers
def tcrit(df):
    return stats.t.ppf(0.975, df["district"].nunique() - 1)


def lin(m, w, tq, G):
    """exp(w'b) with interval; w is a Series over parameter names."""
    names = m.params.index
    w = w.reindex(names).fillna(0.0).to_numpy()
    b = float(w @ m.params.to_numpy())
    se = float(np.sqrt(w @ m.cov_params().to_numpy() @ w))
    return np.exp(b), np.exp(b - tq * se), np.exp(b + tq * se), 2 * stats.t.sf(abs(b / se), G - 1)


def spline_basis(radius, design=None):
    """Centered natural cubic spline basis in log radius (ft)."""
    x = np.log(np.asarray(radius, float))
    if design is None:
        dm = patsy.dmatrix(f"cr(x, df={SPLINE_DF}, constraints='center') - 1", {"x": x})
        return np.asarray(dm), dm.design_info
    return np.asarray(patsy.build_design_matrices([design], {"x": x})[0]), design


def add_spline(df, curv_col, curve_flag):
    on = df[curve_flag].to_numpy() == 1
    rad = np.where(on, cfg.CURV_SCALE / df[curv_col].where(df[curv_col] > 0, np.nan), np.nan)
    B, info = spline_basis(rad[on])
    cols = [f"s{j}" for j in range(B.shape[1])]
    for j, c in enumerate(cols):
        v = np.zeros(len(df))
        v[on] = B[:, j]
        df[c] = v
        df[c + "r"] = v * df["rural"].to_numpy()
    return df, cols, info


def agreement(inv, off):
    t = pd.crosstab(inv, off)
    a, b = t.loc[1, 1], t.loc[1, 0]   # inventory curve: officer curve / straight
    c, d = t.loc[0, 1], t.loc[0, 0]
    n = a + b + c + d
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / n ** 2
    return {"n": n, "officer_curve_share": (a + c) / n, "inventory_curve_share": (a + b) / n,
            "agreement": po, "sensitivity (inv | officer curve)": a / (a + c),
            "specificity (inv | officer straight)": d / (b + d), "kappa": (po - pe) / (1 - pe)}


# ------------------------------------------------------------------ main
def main():
    cr, sg = M.prep()
    sg["zero"] = 0.0
    out = cfg.OUT / "tables"

    # ======================= 1. curve classification
    rc = cr["rd_char"].astype("string").str.strip()
    known = rc.notna() & (rc != "") & ~rc.str.contains("Unknown", case=False, na=True)
    cr["off_curve"] = np.where(known, rc.str.startswith("Curve").astype(float), np.nan)
    ck = cr[cr["off_curve"].notna()].copy()
    ck["off_curve"] = ck["off_curve"].astype(int)
    rows = {"All": agreement(ck["on_curve"], ck["off_curve"]),
            "Urban": agreement(ck.loc[ck.rural == 0, "on_curve"], ck.loc[ck.rural == 0, "off_curve"]),
            "Rural": agreement(ck.loc[ck.rural == 1, "on_curve"], ck.loc[ck.rural == 1, "off_curve"])}
    for lab, sub in (("Urban, KA crashes", ck[(ck.rural == 0) & (ck.severe == 1)]),
                     ("Rural, KA crashes", ck[(ck.rural == 1) & (ck.severe == 1)])):
        rows[lab] = agreement(sub["on_curve"], sub["off_curve"])
    save(pd.DataFrame(rows).T, "tab_curve_agreement", "%.4f")

    mA = fit_logit(M.F_A, ck, "district")
    mOff = fit_logit(M.F_A.replace("on_curve", "off_curve"), ck, "district")
    agree = ck[ck["on_curve"] == ck["off_curve"]]
    mAg = fit_logit(M.F_A, agree, "district")
    tq, G = tcrit(ck), ck["district"].nunique()
    res = []
    for lab, m, v in (("A, inventory curve (same sample)", mA, "on_curve"),
                      ("A, officer-coded curve", mOff, "off_curve"),
                      ("A, crashes where both codings agree", mAg, "on_curve")):
        urb = lin(m, pd.Series({v: 1.0}), tq, G)
        rur = lin(m, pd.Series({v: 1.0, f"{v}:rural": 1.0}), tq, G)
        inter = lin(m, pd.Series({f"{v}:rural": 1.0}), tq, G)
        res.append([lab, int(m.nobs), *urb, *rur, *inter])
    cols = ["model", "n"] + [f"{a}_{b}" for a in ("urban", "rural", "ratio") for b in ("or", "lo", "hi", "p")]
    save(pd.DataFrame(res, columns=cols).set_index("model"), "tab_officer_curve_logit", "%.4f")
    print("  curve classification done")

    # ======================= 2. flexible curvature (splines in log radius)
    sg, sc, info_sg = add_spline(sg, "curv_max", "any_curve")
    base = ("n_severe ~ any_curve + any_curve:rural + grade_max" + M.SG_COMMON)
    lenang = " + curve_len_k + curve_angle"
    common = " + " + " + ".join(sc)
    byrur = common + " + " + " + ".join(c + "r" for c in sc)
    lr_rows = []
    fits = {}
    for tag, extra in (("without length/angle", ""), ("with length/angle", lenang)):
        m0 = fit_nb(base + extra, sg, "district")
        m1 = fit_nb(base + extra + common, sg, "district")
        m2 = fit_nb(base + extra + byrur, sg, "district")
        fits[tag] = m2
        k = len(sc)
        for lab, ma, mb, df_ in (("spline (common) vs presence only", m1, m0, k),
                                 ("spline by rurality vs presence only", m2, m0, 2 * k),
                                 ("spline by rurality vs common spline", m2, m1, k)):
            lr = 2 * (ma.llf - mb.llf)
            lr_rows.append(["SPF", tag, lab, lr, df_, stats.chi2.sf(lr, df_), ma.aic, mb.aic])
    # crash level
    cr2 = cr.copy()
    cr2, cc, info_cr = add_spline(cr2, "curv", "on_curve")
    cbase = "severe ~ on_curve + on_curve:rural + abs_grade" + M.CR_COMMON + M.BEHAV
    l0 = fit_logit(cbase, cr2, "district")
    l1 = fit_logit(cbase + " + " + " + ".join(cc), cr2, "district")
    l2 = fit_logit(cbase + " + " + " + ".join(cc) + " + " + " + ".join(c + "r" for c in cc), cr2, "district")
    for lab, ma, mb, df_ in (("spline (common) vs presence only", l1, l0, len(cc)),
                             ("spline by rurality vs presence only", l2, l0, 2 * len(cc)),
                             ("spline by rurality vs common spline", l2, l1, len(cc))):
        lr = 2 * (ma.llf - mb.llf)
        lr_rows.append(["logit", "", lab, lr, df_, stats.chi2.sf(lr, df_), ma.aic, mb.aic])
    save(pd.DataFrame(lr_rows, columns=["model", "controls", "test", "LR", "df", "p", "AIC_larger",
                                        "AIC_smaller"]), "tab_spline_lr", "%.4f")

    # implied curve-vs-tangent ratio over radius (spline by rurality)
    eff = []
    for lab, m, info, cols_, flag, d in (("SPF", fits["without length/angle"], info_sg, sc, "any_curve", sg),
                                         ("logit", l2, info_cr, cc, "on_curve", cr2)):
        tq, G = tcrit(d), d["district"].nunique()
        B, _ = spline_basis(RADII, info)
        for i, r in enumerate(RADII):
            for rural in (0, 1):
                w = {flag: 1.0}
                if rural:
                    w[f"{flag}:rural"] = 1.0
                for j, c in enumerate(cols_):
                    w[c] = B[i, j]
                    if rural:
                        w[c + "r"] = B[i, j]
                eff.append([lab, r, rural, *lin(m, pd.Series(w), tq, G)])
    save(pd.DataFrame(eff, columns=["model", "radius_ft", "rural", "ratio", "lo", "hi", "p"]),
         "tab_spline_curve_effect", "%.4f")
    print("  spline models done")

    # ======================= 3. regional / county fixed effects
    fe_rows = []
    tq, G = tcrit(sg), sg["district"].nunique()
    mE = fit_nb(M.F_E, sg, "district")
    specs = [("E (primary)", mE, "district")]
    specs.append(("E + region fixed effects", fit_nb(M.F_E + " + C(district)", sg, "district"), "district"))
    try:
        pts = S5.segment_points().set_index("seg_key")
        wa = gpd.read_file(f"zip://{cfg.ROOT / 'gis' / 'cb_2023_us_county_500k.zip'}")
        wa = wa[wa["STATEFP"] == "53"].to_crs(pts.crs)
        j = gpd.sjoin(pts.reset_index(), wa[["NAME", "geometry"]], predicate="within")
        sg["county"] = sg["seg_key"].map(j.drop_duplicates("seg_key").set_index("seg_key")["NAME"])
        sgc = sg.dropna(subset=["county"])
        specs.append(("E + county fixed effects", fit_nb(M.F_E + " + C(county)", sgc, "district"), "district"))
        specs.append(("E + county FE, county clusters",
                      fit_nb(M.F_E + " + C(county)", sgc, "county"), "county"))
    except Exception as e:
        print(f"  (county fixed effects skipped: {type(e).__name__}: {e})")
    for lab, m, grp in specs:
        Gm = sg[grp].nunique() if grp in sg else G
        tqm = stats.t.ppf(0.975, Gm - 1)
        r_int = lin(m, pd.Series({"any_curve:rural": 1.0}), tqm, Gm)
        r_rur = lin(m, pd.Series({"any_curve": 1.0, "any_curve:rural": 1.0}), tqm, Gm)
        r_urb = lin(m, pd.Series({"any_curve": 1.0}), tqm, Gm)
        fe_rows.append([lab, int(m.nobs), *r_int, *r_rur, *r_urb, m.params["alpha"], m.aic])
    cols = ["model", "n"] + [f"{a}_{b}" for a in ("ratio", "rural", "urban") for b in ("irr", "lo", "hi", "p")] \
        + ["alpha", "aic"]
    save(pd.DataFrame(fe_rows, columns=cols).set_index("model"), "tab_fixed_effects", "%.4f")
    print("  fixed-effects models done")

    # ======================= 4. EB posterior uncertainty
    fitted = pd.read_parquet(cfg.OUT / "data" / "segment_year_fitted.parquet")
    a = fitted["alpha_P"].iloc[0]
    seg = fitted.groupby("seg_key").agg(y=("n_severe", "sum"), mu=("mu_P", "sum"),
                                        rural=("rural", "max"), any_curve=("any_curve", "max"),
                                        route=("route", "first"))
    seg = seg[(seg.rural == 1) & (seg.any_curve == 1)].copy()
    shape = 1 / a + seg["y"]
    rate = 1 + 1 / (a * seg["mu"])                     # posterior of the 3-yr expected count
    seg["eb"] = shape / rate
    seg["lo90"] = stats.gamma.ppf(0.05, shape, scale=1 / rate)
    seg["hi90"] = stats.gamma.ppf(0.95, shape, scale=1 / rate)
    seg["p_gt_mu"] = stats.gamma.sf(seg["mu"], shape, scale=1 / rate)
    seg["p_gt_2mu"] = stats.gamma.sf(2 * seg["mu"], shape, scale=1 / rate)
    seg["excess"] = seg["eb"] - seg["mu"]
    top = seg.nlargest(cfg.TOP_K, "excess")
    summ = pd.DataFrame({
        "segments": [len(seg)],
        "P(>SPF) >= 0.90": [(seg.p_gt_mu >= 0.90).sum()],
        "P(>SPF) >= 0.95": [(seg.p_gt_mu >= 0.95).sum()],
        "P(>2xSPF) >= 0.50": [(seg.p_gt_2mu >= 0.50).sum()],
        "P(>2xSPF) >= 0.80": [(seg.p_gt_2mu >= 0.80).sum()],
        "top10 also P(>SPF)>=0.90": [(top.p_gt_mu >= 0.90).sum()],
    })
    save(top[["route", "y", "mu", "eb", "lo90", "hi90", "p_gt_mu", "p_gt_2mu", "excess"]],
         "tab_eb_uncertainty", "%.3f")
    save(summ, "tab_eb_threshold_counts", "%.0f")
    print("done")


if __name__ == "__main__":
    main()
