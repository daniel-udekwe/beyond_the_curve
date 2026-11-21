"""
07_check_2023_geometry.py
-------------------------
Diagnoses the 2023 curve/grade inventories, which link to the 2023 road file
less completely than 2021-22 do. Writes logs/geometry_2023_check.txt with:

 A. For 2023 crashes: on-curve share, grade coverage, and agreement with the
    officer-coded RD_CHAR1 (curve / grade) when geometry comes from
      (i) the 2023 inventories   vs   (ii) the 2022 inventories.
    Whichever agrees better with the officer coding is the better-referenced source.
 B. Curves present in both releases (same ROUTE_ID, radius, length): how far
    their BEGMP moved between 2022 and 2023.
 C. 2023 curve/grade ROUTE_IDs that match no 2023 road record.
"""
import importlib

import numpy as np
import pandas as pd

import config as cfg

B = importlib.import_module("01_build_datasets")
OUT = []


def say(s=""):
    print(s)
    OUT.append(str(s))


def geometry(year, twins):
    curve = B.prep_curve(B.drop_twin_d(B.load("curve", year), twins, "curve"))
    giv = B.grade_intervals(B.drop_twin_d(B.load("grade", year), twins, "grade"))
    return curve, giv


def main():
    road = B.load("road", 2023)
    twins = B.twin_d_ids(road["route"])
    road = B.prep_road(B.drop_twin_d(road, twins, "road"), 2023)
    crash = B.recode_crash(B.load("crash", 2023))
    crash["mp"] = B.num(crash["mp"])
    crash = crash[crash["route"].notna() & crash["mp"].notna()].copy()
    j = B.interval_join_points(crash, road[["route", "begmp", "endmp", "seg_id"]], cols=["seg_id"])
    crash = crash[j["seg_id"].notna()].copy()
    rd = crash["rd_char"].astype("string")
    known = rd.notna() & (rd != "Unknown")
    off_curve = rd.str.startswith("Curve")
    off_grade = rd.str.contains("Grade|Hillcr|Sag", regex=True)

    say("A. 2023 crashes linked to the 2023 road file: "
        f"{len(crash):,}; with officer RD_CHAR1 known: {known.sum():,}")
    say(f"   officer-coded curve share: {off_curve[known].mean():.1%}; "
        f"officer-coded grade/crest/sag share: {off_grade[known].mean():.1%}")
    for gy in (2023, 2022):
        curve, giv = geometry(gy, twins)
        jc = B.interval_join_points(crash, curve[["route", "begmp", "endmp", "curv"]], cols=["curv"])
        on = jc["curv"].notna()
        jg = B.interval_join_points(crash, giv[["route", "begmp", "endmp", "abs_grade"]],
                                    cols=["abs_grade"])
        g = jg["abs_grade"]
        agree_c = (on[known] == off_curve[known].astype(bool)).mean()
        sens_c = on[known & off_curve].mean()          # inventory says curve | officer says curve
        spec_c = (~on[known & ~off_curve]).mean()
        gk = known & g.notna()
        steep2 = g >= 2.0
        agree_g = (steep2[gk] == off_grade[gk].astype(bool)).mean()
        say(f"\n   geometry from {gy} inventories:")
        say(f"     on-curve share {on.mean():.1%}; grade coverage {g.notna().mean():.1%}; "
            f"steep (>=4%) {(g >= 4).mean():.1%}")
        say(f"     curve agreement with officer {agree_c:.1%} "
            f"(sensitivity {sens_c:.1%}, specificity {spec_c:.1%})")
        say(f"     grade>=2% vs officer grade/crest/sag agreement {agree_g:.1%}")

    # ---- B. milepost shift of curves present in both releases
    c22, _ = geometry(2022, twins)
    c23, _ = geometry(2023, twins)
    k = ["route", "radius", "length"]
    m = c23[k + ["begmp"]].merge(c22[k + ["begmp"]], on=k, suffixes=("_23", "_22"))
    m["shift"] = (m["begmp_23"] - m["begmp_22"]).abs()
    m = m.sort_values("shift").drop_duplicates(k + ["begmp_23"])
    say(f"\nB. curves matched across releases on route+radius+length: {len(m):,} "
        f"(of {len(c23):,} in 2023, {len(c22):,} in 2022)")
    if len(m):
        q = m["shift"].quantile([.5, .75, .9, .99])
        say(f"   |BEGMP shift| miles: median {q[.5]:.3f}, p75 {q[.75]:.3f}, p90 {q[.9]:.3f}, "
            f"p99 {q[.99]:.3f}; share shifted > 0.05 mi: {(m['shift'] > .05).mean():.1%}")

    # ---- C. unmatched route IDs
    roads = set(road["route"].dropna())
    for name, d in (("curve", c23), ("grade", geometry(2023, twins)[1])):
        miss = d.loc[~d["route"].isin(roads), "route"]
        say(f"\nC. 2023 {name} records on ROUTE_IDs absent from the 2023 road file: "
            f"{len(miss):,} ({miss.nunique():,} IDs); examples: {list(miss.value_counts().index[:12])}")

    (cfg.OUT / "logs" / "geometry_2023_check.txt").write_text("\n".join(OUT), encoding="utf-8")
    print("\n-> logs/geometry_2023_check.txt")


if __name__ == "__main__":
    main()
