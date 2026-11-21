"""
02_descriptives.py
------------------
Descriptive tables requested by the reviewers.

  tab_desc_crash        N, mean, SD, min, max for the crash-level sample      (R1)
  tab_desc_segment      same for the segment-year sample                       (R1, R4)
  tab_seglen            segment-length distribution and curves per segment     (R1, R4)
  tab_count_outcome     severe-crash count distribution, zeros, var/mean      (R1)
  tab_bivariate         chi-square table (old Table 2), all years
  tab_rural_curve_rate  crash-level severe share by rural x curve
  tab_vmt_rates         exposure-normalised rates by rural x curve (old Table 3)
"""
import numpy as np
import pandas as pd
from scipy import stats

import config as cfg
from utils import load_crash, load_seg, save


def summ(df, cols, labels):
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        rows.append([labels.get(c, c), len(s), s.mean(), s.std(), s.min(), s.max()])
    return pd.DataFrame(rows, columns=["Variable", "N", "Mean", "SD", "Min", "Max"]).set_index("Variable")


def main():
    cr = load_crash()
    sg = load_seg()
    sg = sg[sg["vmt"] > 0].copy()

    # ---- crash-level descriptive table
    cr_cols = ["severe", "on_curve", "curv", "curve_len_ft", "curve_angle", "abs_grade",
               "steep", "workzone", "ramp", "rural", "speed", "lanes", "aadt",
               "alcohol", "dark", "wet"]
    lab = {"severe": "Severe/fatal (KA)", "on_curve": "On horizontal curve",
           "curv": "Curvature (1000/R, ft)", "curve_len_ft": "Curve length (ft)",
           "curve_angle": "Central angle (deg)", "abs_grade": "Absolute grade (%)",
           "steep": "Steep grade (>=4%)", "workzone": "Work zone", "ramp": "Ramp",
           "rural": "Rural", "speed": "Posted speed (mph)", "lanes": "Lane count",
           "aadt": "AADT", "alcohol": "Alcohol involved", "dark": "Dark",
           "wet": "Wet/icy surface", "len_mi": "Segment length (mi)",
           "vmt": "Annual VMT (millions)", "n_severe": "Severe crashes",
           "n_crash": "All crashes", "curv_max": "Max curvature (1000/R)",
           "grade_max": "Max absolute grade (%)", "n_curves": "Curves overlapping segment",
           "share_on_curve": "Share of length on curve"}
    save(summ(cr, [c for c in cr_cols if c in cr], lab), "tab_desc_crash")

    sg["vmt_m"] = sg["vmt"] / 1e6
    sg_cols = ["len_mi", "aadt", "vmt_m", "lanes", "speed", "rural", "curv_max",
               "curve_len_ft", "curve_angle", "n_curves", "share_on_curve",
               "grade_max", "n_crash", "n_severe"]
    lab["vmt_m"] = lab["vmt"]
    save(summ(sg, [c for c in sg_cols if c in sg], lab), "tab_desc_segment")

    # ---- segment length & curves per segment (R4)
    q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    seglen = pd.DataFrame({
        "len_mi": sg["len_mi"].quantile(q),
        "n_curves (all)": sg["n_curves"].quantile(q),
        "n_curves (curved segs)": sg.loc[sg["n_curves"] > 0, "n_curves"].quantile(q),
        "share_on_curve (curved segs)": sg.loc[sg["n_curves"] > 0, "share_on_curve"].quantile(q),
    })
    seglen.loc["share segs with >1 curve"] = [np.nan, (sg["n_curves"] > 1).mean(),
                                              (sg.loc[sg["n_curves"] > 0, "n_curves"] > 1).mean(),
                                              np.nan]
    save(seglen, "tab_seglen")

    # ---- count outcome (R1)
    y = sg["n_severe"]
    dist = y.clip(upper=4).value_counts().sort_index()
    dist.index = [str(i) if i < 4 else "4+" for i in dist.index]
    co = pd.DataFrame({"segments": dist, "share": dist / len(y)})
    co.loc["mean"] = [y.mean(), np.nan]
    co.loc["variance"] = [y.var(), np.nan]
    co.loc["var/mean"] = [y.var() / y.mean(), np.nan]
    co.loc["prop. zeros"] = [(y == 0).mean(), np.nan]
    save(co, "tab_count_outcome", floatfmt="%.4f")

    # ---- bivariate (same analytic sample as the crash models)
    d = cr.dropna(subset=["severe"])
    rows = []
    for f, name in [("on_curve", "On curve"), ("steep", "Steep grade (>=4%)"),
                    ("workzone", "Work zone"), ("ramp", "Ramp")]:
        tab = pd.crosstab(d[f], d["severe"])
        chi2, p, _, _ = stats.chi2_contingency(tab, correction=False)
        rows.append([name, d.loc[d[f] == 1, "severe"].mean(), d.loc[d[f] == 0, "severe"].mean(),
                     int((d[f] == 1).sum()), chi2, p])
    save(pd.DataFrame(rows, columns=["Feature", "Rate present", "Rate absent", "n present",
                                     "chi2", "p"]).set_index("Feature"), "tab_bivariate", "%.4f")

    rc = d.groupby(["rural", "on_curve"])["severe"].agg(["mean", "size"])
    save(rc, "tab_rural_curve_rate", "%.4f")

    # ---- exposure-normalised rates (descriptive threshold)
    s = sg[sg["vmt"] >= cfg.VMT_MIN_DESC]
    g = s.groupby(["rural", "any_curve"]).agg(severe=("n_severe", "sum"),
                                             vmt_m=("vmt", lambda v: v.sum() / 1e6))
    g["rate_per_100M"] = g["severe"] / g["vmt_m"] * 100
    save(g, "tab_vmt_rates", "%.4f")


if __name__ == "__main__":
    main()
