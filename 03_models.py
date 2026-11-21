"""
03_models.py
------------
Main and robustness models. All crash-level models use ONE common analytic
sample; all segment-level models use ONE common analytic sample, so fit
statistics are only ever compared within a unit of analysis.

A. 2x2 design that isolates binarisation (Reviewers 1 & 3)
     A  crash-level logit, binary geometry
     B  crash-level logit, continuous geometry
     C  segment-level NB SPF, binary geometry
     D  segment-level NB SPF, continuous geometry   (primary)
   Segment models enter exposure as log AADT + log segment length with
   estimated coefficients (no offset); the log-VMT offset form is a
   sensitivity model (CURE plots and the rootogram reject it).
   A vs B and C vs D differ ONLY in geometry coding.
   Each logit is also run without alcohol/lighting/weather (A0, B0) to show
   how much the geometry estimates move when behavioural covariates are
   dropped - the omitted-variable check for the SPF (Reviewers 1 & 2).

B. Count-model diagnostics (R1): Poisson vs NB LR test, ZINB and hurdle NB,
   Vuong tests, observed vs predicted zeros.

C. Exposure / segmentation sensitivity (R1, R4): log(VMT) as free covariate,
   log(AADT)+log(length) separately, offset + log(length), fixed 0.1-mi
   segments, length-weighted curvature, share of segment on curve.

D. Threshold sensitivity (R1): grade cut-points 2/3/4/6 %, radius bands.

E. Severity specification (R1): ordered logit on 4 classes.

F. Inference with few clusters (R3): district (G=6, t(G-1)), wild cluster
   bootstrap (Webb weights), route clusters, segment clusters.

G. Interaction interpretation (R3): predicted severe crashes per 100M VMT and
   predicted P(severe) across curvature for rural/urban and grade levels.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from patsy import build_design_matrices
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

import config as cfg
from utils import (load_crash, load_seg, fit_logit, fit_nb, fit_poisson,
                   refit_cluster, score_wild_bootstrap, coef_table, fit_stats,
                   save, lr_test_alpha, vuong)

# ------------------------------------------------------------- formulas
BEHAV = " + alcohol + dark + wet"
CR_COMMON = " + workzone + ramp + rural + speed + lanes + C(year)"
F_A = "severe ~ on_curve + on_curve:rural + steep" + CR_COMMON + BEHAV
F_B = ("severe ~ curv + curv:rural + abs_grade + curv:abs_grade + curve_len_k"
       " + curve_angle" + CR_COMMON + BEHAV)

EXPO = " + log_aadt + log_len"
SG_COMMON = EXPO + " + rural + lanes + speed + C(year)"
F_C = "n_severe ~ any_curve + any_curve:rural + steep" + SG_COMMON
F_D = ("n_severe ~ curv_max + curv_max:rural + grade_max + curv_max:grade_max"
       " + curve_len_k + curve_angle" + SG_COMMON)

# E nests C and D: curve presence (any_curve) and sharpness (curv_max) together.
# It is the primary SPF; C and D are its presence-only and sharpness-only restrictions.
F_E = F_D + " + any_curve + any_curve:rural"
KEY_TERMS_D = ["curv_max", "curv_max:rural", "grade_max", "curv_max:grade_max"]
KEY_TERMS_C = ["any_curve", "any_curve:rural"]
KEY_TERMS_E = KEY_TERMS_C + ["curv_max", "curv_max:rural"]
TCRIT = None   # set in main: t(G-1) critical value


def prep():
    cr = load_crash()
    cr["curve_len_k"] = cr["curve_len_ft"] / 1000
    need = ["severe", "on_curve", "curv", "abs_grade", "steep", "workzone", "ramp",
            "rural", "speed", "lanes", "alcohol", "dark", "wet", "curve_len_k",
            "curve_angle", "district", "route", "year"]
    cr = cr.dropna(subset=need).copy()

    sg = load_seg()
    n0 = len(sg)
    sg = sg[(sg["vmt"] > 0)].copy()
    print(f"segment-years {n0:,}; VMT>0 {len(sg):,}; missing grade {sg['grade_max'].isna().sum():,}; "
          f"missing speed {sg['speed'].isna().sum():,}; missing lanes {sg['lanes'].isna().sum():,}")
    sg["steep"] = (sg["grade_max"] >= cfg.STEEP_GRADE).astype(int)
    sg["curve_len_k"] = sg["curve_len_ft"] / 1000
    sg["log_vmt"] = np.log(sg["vmt"])
    sg["log_len"] = np.log(sg["len_mi"])
    sg["log_aadt"] = np.log(sg["aadt"])
    sg["zero"] = 0.0            # no offset: exposure is estimated
    need = ["n_severe", "any_curve", "curv_max", "grade_max", "rural", "lanes",
            "speed", "district", "route", "seg_key", "log_vmt", "log_aadt", "log_len"]
    sg = sg.dropna(subset=need).copy()
    return cr, sg


def main():
    cr, sg = prep()
    print(f"crash-level n={len(cr):,}; segment-years n={len(sg):,}")
    fits, coefs = [], []

    # ================= A. 2x2 design
    mA = fit_logit(F_A, cr, "district")
    mB = fit_logit(F_B, cr, "district")
    mA0 = fit_logit(F_A.replace(BEHAV, ""), cr, "district")
    mB0 = fit_logit(F_B.replace(BEHAV, ""), cr, "district")
    mC = fit_nb(F_C, sg, "district")
    mD = fit_nb(F_D, sg, "district")
    mE = fit_nb(F_E, sg, "district")
    for m, n in [(mA, "A_logit_binary"), (mB, "B_logit_continuous"),
                 (mA0, "A0_logit_binary_nobehav"), (mB0, "B0_logit_cont_nobehav"),
                 (mC, "C_spf_binary"), (mD, "D_spf_continuous"), (mE, "E_spf_presence_sharpness")]:
        fits.append(fit_stats(m, n))
        coefs.append(coef_table(m, label=n))
    save(pd.concat(coefs, axis=1), "tab_2x2_coefficients")
    fit = pd.DataFrame(fits).set_index("model")
    fit["unit"] = ["crash"] * 4 + ["segment"] * 3
    save(fit, "tab_2x2_fit")      # compare AIC only within the same unit


    # ================= A2. presence vs sharpness (segment level, model E)
    from scipy import stats as st
    G = sg["district"].nunique()
    tq = st.t.ppf(0.975, G - 1)

    def lincomb(m, terms):
        """exp(sum of coefficients) with a t(G-1) interval from the clustered covariance."""
        b = m.params[terms].sum()
        V = m.cov_params().loc[terms, terms].to_numpy().sum()
        se = np.sqrt(V)
        return np.exp(b), np.exp(b - tq * se), np.exp(b + tq * se), 2 * st.t.sf(abs(b / se), G - 1)

    e_nocurv = fit_nb(F_E.replace("curv_max + curv_max:rural + ", "")
                      .replace(" + curv_max:grade_max", ""), sg, "district")
    e_nopres = mD
    ps = []
    for lab, m, terms in [("C: curve, urban", mC, ["any_curve"]),
                          ("C: curve, rural", mC, ["any_curve", "any_curve:rural"]),
                          ("C: rural / urban curve ratio", mC, ["any_curve:rural"]),
                          ("E: curve, urban", mE, ["any_curve"]),
                          ("E: curve, rural", mE, ["any_curve", "any_curve:rural"]),
                          ("E: rural / urban curve ratio", mE, ["any_curve:rural"]),
                          ("E: +1 curvature, urban", mE, ["curv_max"]),
                          ("E: +1 curvature, rural", mE, ["curv_max", "curv_max:rural"]),
                          ("D: +1 curvature, urban", mD, ["curv_max"]),
                          ("D: +1 curvature, rural", mD, ["curv_max", "curv_max:rural"])]:
        ps.append([lab, *lincomb(m, terms)])
    pres = pd.DataFrame(ps, columns=["quantity", "ratio", "lo", "hi", "p"]).set_index("quantity")
    lr_s = 2 * (mE.llf - e_nocurv.llf)
    lr_p = 2 * (mE.llf - e_nopres.llf)
    pres.loc["LR: sharpness terms in E (df 3)"] = [lr_s, np.nan, np.nan, st.chi2.sf(lr_s, 3)]
    pres.loc["LR: presence terms in E (df 2)"] = [lr_p, np.nan, np.nan, st.chi2.sf(lr_p, 2)]
    save(pres, "tab_presence_vs_sharpness", "%.4f")

    # ================= B. count diagnostics
    mP = fit_poisson(F_E, sg, "district")
    lr, p_lr = lr_test_alpha(mE, mP)
    y = sg["n_severe"].to_numpy()
    infl = sm.add_constant(np.column_stack([sg["log_vmt"], sg["rural"]]))
    off0 = np.zeros(len(sg))
    diag = {"LR_alpha": lr, "p_LR_alpha (boundary)": p_lr, "alpha": mE.params["alpha"]}
    try:
        zinb = ZeroInflatedNegativeBinomialP(y, mE.model.exog, exog_infl=infl,
                                             offset=off0, p=2)
        rz = zinb.fit(disp=0, maxiter=500, method="bfgs")
        diag.update({"ZINB_aic": rz.aic, "ZINB_bic": rz.bic})
        z, pz = vuong(rz.model.loglikeobs(rz.params), mE.model.loglikeobs(mE.params.to_numpy()))
        diag.update({"Vuong_ZINB_vs_NB_z": z, "Vuong_p": pz})
    except Exception as e:
        diag["ZINB_error"] = repr(e)
    try:
        # hurdle NB built from its two parts, because statsmodels' HurdleCountModel
        # does not take an offset: logit(any severe crash) + zero-truncated NB2
        from statsmodels.discrete.truncated_model import TruncatedLFNegativeBinomialP
        Xh = mE.model.exog                       # exposure terms already included
        zpart = sm.Logit((y > 0).astype(int), Xh).fit(disp=0, maxiter=200)
        pos = y > 0
        tm = TruncatedLFNegativeBinomialP(y[pos], mE.model.exog[pos],
                                          offset=off0[pos], p=2)
        t0 = tm.fit(start_params=mE.params.to_numpy(), disp=0, maxiter=20000, method="nm")
        tpart = tm.fit(start_params=t0.params, disp=0, maxiter=1000, method="bfgs")
        if not np.isfinite(tpart.llf) or tpart.llf < t0.llf:
            tpart = t0
        ll_h = zpart.llf + tpart.llf
        k_h = len(zpart.params) + len(tpart.params)
        diag.update({"Hurdle_loglik": ll_h, "Hurdle_aic": -2 * ll_h + 2 * k_h,
                     "Hurdle_bic": -2 * ll_h + np.log(len(y)) * k_h})
    except Exception as e:
        diag["Hurdle_error"] = repr(e)
    diag.update({"NB_aic": mE.aic, "NB_bic": mE.bic, "Poisson_aic": mP.aic})
    # observed vs predicted zeros
    mu = mE.predict(sg, offset=off0)
    a = mE.params["alpha"]
    p0_nb = (1 + a * mu) ** (-1 / a)
    diag.update({"zeros_observed": (y == 0).mean(), "zeros_pred_NB": p0_nb.mean(),
                 "zeros_pred_Poisson": np.exp(-mP.predict(sg, offset=off0)).mean()})
    save(pd.Series(diag).to_frame("value"), "tab_count_diagnostics", "%.4f")

    # ================= C. exposure / segmentation sensitivity
    sens = {}
    base_rhs = F_E.split("~")[1]
    rhs_noexpo = base_rhs.replace(EXPO, "")
    sens["E (primary)"] = mE
    sens["C: presence only"] = mC
    sens["D: sharpness only"] = mD
    mV = fit_nb("n_severe ~" + rhs_noexpo, sg, "district", offset="log_vmt")
    sens["log VMT offset"] = mV
    sens["log VMT, coefficient estimated"] = fit_nb("n_severe ~ log_vmt +" + rhs_noexpo, sg,
                                                    "district")
    sens["length-weighted curvature"] = fit_nb(F_E.replace("curv_max", "curv_lenwt"),
                                               sg, "district")
    sens["+ share on curve, n curves"] = fit_nb(F_E + " + share_on_curve + n_curves",
                                                sg, "district")
    fx = pd.read_parquet(cfg.OUT / "data" / "segment_fixed.parquet")
    fx = fx[fx["vmt"] > 0].copy()
    fx["log_vmt"] = np.log(fx["vmt"])
    fx["log_aadt"] = np.log(fx["aadt"])
    fx["zero"] = 0.0
    if "any_curve" not in fx:
        fx["any_curve"] = (fx["curv_max"] > 0).astype(int)
    # equal lengths: log length is constant, so only log AADT enters
    sens["fixed 0.1-mi segments"] = fit_nb(
        "n_severe ~ curv_max + curv_max:rural + grade_max + curv_max:grade_max"
        " + any_curve + any_curve:rural" + SG_COMMON.replace(" + log_len", ""),
        fx.dropna(subset=["rural", "lanes", "speed", "grade_max", "district", "log_aadt"]),
        "district")
    sens_fit = pd.DataFrame([fit_stats(m, k) for k, m in sens.items()
                             if k != "fixed 0.1-mi segments"]).set_index("model")
    save(sens_fit, "tab_sens_fit")
    save(pd.concat([coef_table(m, label=k) for k, m in sens.items()], axis=1),
         "tab_sens_exposure_segmentation")

    # ================= D. thresholds
    thr = []
    for cut in cfg.GRADE_CUTS:
        cr["steep_x"] = cr[f"steep_{cut:g}"]
        sg["steep_x"] = (sg["grade_max"] >= cut).astype(int)
        ra = fit_logit(F_A.replace("steep", "steep_x"), cr, "district")
        rc = fit_nb(F_C.replace("steep", "steep_x"), sg, "district")
        thr.append([cut, np.exp(ra.params["steep_x"]), ra.pvalues["steep_x"],
                    np.exp(rc.params["steep_x"]), rc.pvalues["steep_x"],
                    cr["steep_x"].mean(), sg["steep_x"].mean()])
    save(pd.DataFrame(thr, columns=["cut_%", "logit_OR", "logit_p", "spf_IRR", "spf_p",
                                    "share_crashes", "share_segments"]).set_index("cut_%"),
         "tab_sens_grade_cuts")

    bands = pd.IntervalIndex.from_breaks(cfg.RADIUS_BANDS_FT, closed="left")
    cr["rband"] = pd.cut(cr["radius"], bands).astype(str).where(cr["on_curve"] == 1, "tangent")
    sg["rband"] = pd.cut(cfg.CURV_SCALE / sg["curv_max"].replace(0, np.nan), bands) \
        .astype(str).where(sg["curv_max"] > 0, "tangent")
    # rural-specific radius bands: band effect vs tangent, separately for urban and rural
    rbr_sg = fit_nb("n_severe ~ C(rband, Treatment('tangent')) * rural + grade_max"
                    + SG_COMMON, sg, "district")
    rbr_cr = fit_logit("severe ~ C(rband, Treatment('tangent')) * rural + steep"
                       + CR_COMMON + BEHAV, cr, "district")
    rows = []
    for lab_m, m in (("SPF", rbr_sg), ("logit", rbr_cr)):
        bands_ = [t for t in m.params.index if t.startswith("C(rband") and ":" not in t]
        for t in bands_:
            band = t.split("[T.")[1].rstrip("]")
            for rural in (0, 1):
                terms = [t] + ([f"{t}:rural"] if rural else [])
                r_, lo_, hi_, p_ = lincomb(m, terms)
                n_ = int(((sg["rband"] if lab_m == "SPF" else cr["rband"]) == band)
                         [(sg["rural"] if lab_m == "SPF" else cr["rural"]) == rural].sum())
                rows.append([lab_m, band, rural, r_, lo_, hi_, p_, n_])
    save(pd.DataFrame(rows, columns=["model", "band", "rural", "ratio", "lo", "hi", "p", "n"]),
         "tab_radius_bands_rural")

    # ================= E. ordered logit (4 classes)
    # OrderedModel cannot take an intercept, so year enters as explicit dummies
    od = cr.dropna(subset=["sev4"]).copy()
    od["sev4"] = od["sev4"].astype(int)
    ydum = []
    for yr in cfg.YEARS[1:]:
        od[f"y{yr}"] = (od["year"] == yr).astype(int)
        ydum.append(f"y{yr}")
    om = OrderedModel.from_formula(
        "sev4 ~ 0 + curv + curv:rural + abs_grade + curv:abs_grade + curve_len_k"
        " + curve_angle + workzone + ramp + rural + speed + lanes + alcohol + dark + wet + "
        + " + ".join(ydum), od, distr="logit")
    ro = om.fit(method="bfgs", disp=0, maxiter=1000)
    save(coef_table(ro, label="ordered_logit"), "tab_sens_ordered_logit")

    # ================= F. inference with few clusters
    infs = []
    for lab, m, terms in (("C", mC, KEY_TERMS_C), ("D", mD, KEY_TERMS_D), ("E", mE, KEY_TERMS_E)):
        d = pd.DataFrame({
            "coef": m.params[terms],
            "p_district_t(G-1)": m.pvalues[terms],
            "p_wild_cluster_district": score_wild_bootstrap(m, sg, "district", terms),
            "p_route": refit_cluster(m, sg, "route").pvalues[terms],
            "p_segment": refit_cluster(m, sg, "seg_key").pvalues[terms],
        })
        d.index = [f"{lab}: {t}" for t in d.index]
        infs.append(d)
    save(pd.concat(infs), "tab_inference_clusters", "%.4f")

    # ================= G. predicted rates for the interactions
    grid = np.linspace(0, sg.loc[sg["curv_max"] > 0, "curv_max"].quantile(0.99), 60)
    ref = sg[sg["curv_max"] > 0].median(numeric_only=True)     # curve attributes of curved segments
    # exposure fixed at the median AADT and length of rural / urban segments;
    # predicted crashes are converted to a rate per 100M VMT at that exposure
    ref_ex = sg.groupby("rural")[["aadt", "len_mi"]].median()
    rng = np.random.default_rng(cfg.SEED)
    draws = rng.multivariate_normal(mE.params, mE.cov_params(), 2000)
    rows = []
    for rural in (0, 1):
        for gr in (0, 3, 6):
            on = (grid > 0).astype(int)                # grid[0] = tangent
            nd = pd.DataFrame({"curv_max": grid, "any_curve": on, "rural": rural, "grade_max": gr,
                               "curve_len_k": ref["curve_len_k"] * on,
                               "curve_angle": ref["curve_angle"] * on,
                               "lanes": 2, "speed": ref["speed"], "year": cfg.YEARS[-1],
                               "log_aadt": np.log(ref_ex.loc[rural, "aadt"]),
                               "log_len": np.log(ref_ex.loc[rural, "len_mi"])})
            vmt_ref = ref_ex.loc[rural, "aadt"] * 365 * ref_ex.loc[rural, "len_mi"]
            Xg = mE.model.data.orig_exog.design_info      # patsy design for new data
            Xn = np.asarray(build_design_matrices([Xg], nd)[0])
            eta = draws[:, :Xn.shape[1]] @ Xn.T
            mu = np.exp(eta) / vmt_ref * 1e8            # KA crashes per 100M VMT
            for i, c in enumerate(grid):
                rows.append([rural, gr, c, np.median(mu[:, i]),
                             *np.percentile(mu[:, i], [2.5, 97.5])])
    pr = pd.DataFrame(rows, columns=["rural", "grade", "curv", "rate", "lo", "hi"])
    pr.to_csv(cfg.OUT / "tables" / "pred_rates_interactions.csv", index=False)


    # keep fitted D for the spatial script
    sg["mu_P"] = mE.predict(sg, offset=off0)          # primary SPF (E)
    sg["alpha_P"] = mE.params["alpha"]
    sg["mu_D"] = mD.predict(sg, offset=off0)          # sharpness-only, for the CURE comparison
    sg["alpha_D"] = mD.params["alpha"]
    # log-VMT offset alternative, kept for the CURE / rootogram comparison
    sg["mu_V"] = mV.predict(sg, offset=sg["log_vmt"])
    sg["alpha_V"] = mV.params["alpha"]
    pd.Series(ref_ex.stack()).to_csv(cfg.OUT / "tables" / "pred_rates_reference_exposure.csv")
    sg.to_parquet(cfg.OUT / "data" / "segment_year_fitted.parquet", index=False)
    with open(cfg.OUT / "logs" / "model_summaries.txt", "w") as f:
        for m, n in [(mA, "A"), (mB, "B"), (mC, "C"), (mD, "D"), (mE, "E")]:
            f.write(f"\n\n######## {n}\n{m.summary()}")


if __name__ == "__main__":
    main()
