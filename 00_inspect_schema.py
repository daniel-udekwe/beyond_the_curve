"""
00_inspect_schema.py
--------------------
Writes a schema report for every CSV in the HSIS Excel/ folder so the
analysis pipeline can be written against the real column names and codes.

Usage (from the 2026-065 folder):
    python 00_inspect_schema.py path/to/Excel

Output: schema_report.txt in the current folder. Send that file back.
Nothing in the data folder is modified.
"""
import sys
from pathlib import Path

import pandas as pd

MAX_LEVELS = 40      # print full value counts for columns with <= this many levels
N_EXAMPLES = 6       # example values for higher-cardinality columns


def read_csv(path):
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode {path}")


def describe(df, name, out):
    out.write("=" * 90 + "\n")
    out.write(f"FILE: {name}   rows={len(df):,}   cols={df.shape[1]}\n")
    out.write("=" * 90 + "\n")
    for col in df.columns:
        s = df[col]
        n_miss = s.isna().sum()
        n_uniq = s.nunique(dropna=True)
        out.write(f"\n-- {col}  dtype={s.dtype}  missing={n_miss:,} "
                  f"({n_miss / max(len(s), 1):.1%})  unique={n_uniq:,}\n")
        if n_uniq <= MAX_LEVELS:
            vc = s.value_counts(dropna=False).sort_index()
            for k, v in vc.items():
                out.write(f"     {k!r}: {v:,}\n")
        else:
            if pd.api.types.is_numeric_dtype(s):
                d = s.describe(percentiles=[.01, .25, .5, .75, .99])
                out.write("     " + "  ".join(f"{k}={v:.4g}" for k, v in d.items()) + "\n")
            ex = s.dropna().drop_duplicates().head(N_EXAMPLES).tolist()
            out.write(f"     examples: {ex}\n")
    out.write("\n\n")


def describe_gdb(gdb, out):
    """List every layer in the file geodatabase with its fields and CRS."""
    try:
        import pyogrio
    except ImportError:
        out.write("GDB: pyogrio not installed (pip install pyogrio) - skipped\n\n")
        return
    out.write("=" * 90 + f"\nGEODATABASE: {gdb.name}\n" + "=" * 90 + "\n")
    for name, gtype in pyogrio.list_layers(gdb):
        try:
            info = pyogrio.read_info(gdb, layer=name)
            out.write(f"\nLAYER {name}  geometry={gtype}  features={info['features']:,}  "
                      f"crs={info['crs']}\n")
            out.write(f"   fields: {list(info['fields'])}\n")
            sample = pyogrio.read_dataframe(gdb, layer=name, max_features=3,
                                            read_geometry=False)
            out.write(sample.to_string(max_cols=40) + "\n")
        except Exception as e:
            out.write(f"\nLAYER {name}  ERROR: {e}\n")
    out.write("\n\n")


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Excel")
    files = sorted(folder.glob("*.csv"))
    if not files:
        sys.exit(f"No CSV files found in {folder}")
    gdbs = sorted(folder.parent.glob("*.gdb"))
    with open("schema_report.txt", "w", encoding="utf-8") as out:
        for g in gdbs:
            print(f"Reading {g.name} ...")
            describe_gdb(g, out)
        for f in files:
            print(f"Reading {f.name} ...")
            try:
                describe(read_csv(f), f.name, out)
            except Exception as e:  # keep going if one file fails
                out.write(f"FILE: {f.name}  ERROR: {e}\n\n")
    print("Done -> schema_report.txt")


if __name__ == "__main__":
    main()
