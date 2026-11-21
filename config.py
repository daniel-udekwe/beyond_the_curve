"""
config.py - single place for paths, years, thresholds, column names and codes.
Column names and codes confirmed against schema_report.txt (HSIS WA 2021-2023).
"""
import os
from pathlib import Path

# ----------------------------------------------------------------- paths
# Project root = the folder that holds these scripts, Excel\ and the HSIS .gdb.
# Set HSIS_ROOT to point somewhere else without editing this file.
ROOT = Path(os.environ.get("HSIS_ROOT", Path(__file__).resolve().parent))
RAW = ROOT / "Excel"
GDB = ROOT / "MN_WA_2026_065.gdb"
OUT = ROOT / "analysis_out"
for sub in ("data", "tables", "figures", "logs"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)

STATE = "wa"
YEARS = [2021, 2022, 2023]           # years with road + curve + grade files
CRS = 2927                           # NAD83(HARN) / Washington South (ftUS)
# Which release of the curve/grade inventories to use for each crash year.
# 2023 uses the 2022 release: for 2023 crashes it agrees better with officer-coded
# curves (sensitivity 56% vs 44%) and covers more crashes with grade (91.7% vs 87.7%);
# see logs/geometry_2023_check.txt.
GEOMETRY_YEAR = {2021: 2021, 2022: 2022, 2023: 2022}


def raw_file(kind, year):
    """kind in {crash, unit, road, curve, grade}"""
    return RAW / f"{STATE}{str(year)[-2:]}{kind}.csv"


# ------------------------------------------------------- column mapping
# standard_name: source_name
COLS = {
    "crash": {
        "caseno":   "CASENO",
        "route":    "ROUTE_ID",       # e.g. 099i, 005LX21856i  (links to road/curve/grade)
        "mp":       "MILEPOST",
        "severity": "SEVERITY",       # most severe injury in the crash (text)
        "weather":  "WEATHER",
        "light":    "LIGHT",
        "surface":  "RDSURF",
        "alcohol":  "ALCFLAG",        # 0/1
        "workzone": "WKZONE",
        "rd_type":  "RD_TYPE",        # MAINLINE / ... RAMP / COUPLET ...
        "numvehs":  "NUMVEHS",
        "acctype":  "ACCTYPE",
        "rur_urb_crash": "RUR_URB",   # fallback only; rurality comes from the road file
        "rd_char":  "RD_CHAR1",       # officer-coded curve/grade (descriptive check)
        "x":        "GPS_LATX",       # state plane ft (EPSG 2927)
        "y":        "GPS_LATY",
    },
    "unit": {
        "caseno":   "CASENO",
        "vehtype":  "VEHTYPE",
        "spdlimit": "SPDLIMIT",
    },
    "road": {
        "route":    "ROUTE_ID",
        "begmp":    "BEGMP",
        "endmp":    "ENDMP",
        "aadt":     "TOTAL_AADT",     # two-way AADT of the same year; 0 = missing
        "lanes":    "LANE_COUNT",     # 0 = missing
        "speed":    "SPEED_LIMIT",    # 0 = missing
        "func_cls": "FUNCTIONAL_CLASS",  # 'Rural ...' / 'Urban ...'
        "rodwycls": "RODWYCLS",
        "divided":  "DIVIDED_INDICATOR",
        "med_barr": "MEDIAN_BARRIER",
        "access":   "ACCESS_CONTROL",
        "shld_w":   "RIGHT_SHOULDER_WIDTH",
        "road_w":   "ROAD_WIDTH",
        "district": "DISTRICT_NUMBER",   # 1-6 (0 = ferries/unknown)
    },
    "curve": {
        "route":    "ROUTE_ID",
        "begmp":    "BEGMP",
        "endmp":    "ENDMP",
        "radius":   "Curve_Radius",   # ft; 0 = missing
        "length":   "Curve_Length",   # ft
        "angle":    "Curve_Central_Angle",  # 'DDD*MM.m' (2021-22) or decimal deg (2023)
        "superel":  "Super_Elevation",
        "direction": "Curve_Direction",
    },
    "grade": {                        # vertical PVI points
        "route":    "ROUTE_ID",
        "mp":       "MILEPOST",
        "g_back":   "F__Grade_Back",
        "g_ahead":  "F__Grade_Ahead",
        "vc_len":   "Curve_Length",   # vertical curve length, ft
    },
}

# ------------------------------------------------------- code mappings
# crash SEVERITY text -> (KABCO, ordered class 0..3); 'Unknown' -> excluded
SEVERITY_MAP = {
    "Dead at Scene":            ("K", 3),
    "Dead on Arrival":          ("K", 3),
    "Died in Hospital":         ("K", 3),
    "Suspected Serious Injury": ("A", 2),
    "Suspected Minor Injury":   ("B", 1),
    "Possible Injury":          ("C", 1),
    "No Apparent Injury":       ("O", 0),
}
RAMP_PATTERN = "RAMP"                       # RD_TYPE contains this -> ramp
WET_CODES = {"Wet", "Ice", "Snow/Slush", "Standing Water"}       # RDSURF
DARK_PREFIX = "Dark"                        # LIGHT startswith -> dark
DARK_UNLIT_CODES = {"Dark-No Street Lights"}
ALCOHOL_YES = {1, 1.0, "1"}
WORKZONE_YES = {"Within WorkZone"}          # 'In External Traffic Backup' not counted
RURAL_PREFIX = "Rural"                      # FUNCTIONAL_CLASS / RODWYCLS startswith
HEAVY_VEH = {"Truck & Trailer", "Truck (Flatbad,Van,etc)", "Truck - Double Trailer Combinations",
             "Truck Tractor", "Truck Tractor & Semi-Trailer", "Bus or Motor Stage", "School Bus"}
DISTRICT_NAMES = {1: "Northwest", 2: "North Central", 3: "Olympic", 4: "Southwest",
                  5: "South Central", 6: "Eastern"}

# Directional records: crashes on both sides of a divided highway are coded to the
# increasing ('...i') ROUTE_ID, while the road inventory also carries separate
# decreasing-roadway records ('...d'). TOTAL_AADT is two-way, so keeping the 'd'
# records would double-count exposure on segments that receive no crashes.
INVENTORY_SUFFIX = "i"

# ------------------------------------------------------- cleaning rules
MAX_ABS_GRADE = 15.0            # PVI grades beyond this are treated as coding errors
MAX_GRADE_SPAN_MI = 5.0         # grade ahead of the last PVI is carried at most this far
MIN_RADIUS_FT = 50              # radius 0 / implausibly small -> missing

# ------------------------------------------------------- analysis settings
STEEP_GRADE = 4.0               # primary binary threshold (%)
GRADE_CUTS = [2.0, 3.0, 4.0, 6.0]
RADIUS_BANDS_FT = [0, 500, 1000, 2000, 3000, float("inf")]
CURV_SCALE = 1000               # curvature reported as 1000/R (1/ft)
VMT_MIN_DESC = 0.1e6            # descriptive rates (annual VMT)
HOTSPOT_VMT_GRID = [0.05e6, 0.1e6, 0.25e6, 0.5e6]
HOTSPOT_MINCRASH_GRID = [1, 2, 3]
TOP_K = 10
FIXED_SEG_LEN_MI = 0.1          # fixed-length re-segmentation sensitivity
SEED = 20261120
