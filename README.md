# Beyond the Curve

Analysis code for *Beyond the Curve: Curve Presence, Curve Sharpness, and Rural
Context in Fatal and Serious-Injury Crashes on Washington State Highways*.

The pipeline links Highway Safety Information System (HSIS) crash, roadway,
horizontal-curve and vertical-grade files for Washington State (2021–2023),
estimates crash-level logit and segment-level negative binomial safety
performance functions, runs the robustness, spatial and screening analyses,
and draws every figure in the paper.

## Data

The HSIS data are not included and may not be redistributed. They can be
requested from the Federal Highway Administration through the HSIS program
(<https://www.hsisinfo.org>). The code expects this layout in the project root:

```
Excel/                    wa{21,22,23}{crash,unit,road,curve,grade}.csv
MN_WA_2026_065.gdb/       HSIS geodatabase with the wa21road, wa22road, wa23road route layers
```

File names, column names and codes are set in `config.py`. To keep the data
elsewhere, set the environment variable `HSIS_ROOT` to the folder that contains
`Excel/` and the `.gdb`.

## Setup

Python 3.10 or later.

```
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
```

## Run

```
python run_all.py
```

or step by step:

| Step | Script | What it does |
|---|---|---|
| 1 | `01_build_datasets.py` | links crashes, roadway, curve and grade files; writes `analysis_out/data/*.parquet` |
| 2 | `02_descriptives.py` | descriptive and bivariate tables |
| 3 | `03_models.py` | models A–E, count diagnostics, exposure and segmentation sensitivity, few-cluster inference, predicted rates |
| 4 | `04_random_parameters.py` | random-parameters logit and panel negative binomial (simulated ML, PyTorch; slow) |
| 5 | `05_spatial_hotspots.py` | Moran's I, local Moran clusters, empirical Bayes screening and ranking stability |
| 6 | `09_additional_checks.py` | officer-coded curve status, spline curvature, region and county fixed effects, EB posterior uncertainty |
| 7 | `08_publication_figures.py` | all manuscript figures |

`00_inspect_schema.py` prints the columns and codes of the raw files, and
`07_check_2023_geometry.py` compares the 2022 and 2023 curve inventories
against officer-coded curve status; both are one-off checks.
Figures can be redrawn selectively, e.g. `python 08_publication_figures.py map4 radius`.

## Outputs

All outputs are written to `analysis_out/` (not tracked):

- `figures/pub/` – the 14 manuscript figures (PNG and PDF)
- `tables/` – CSV tables behind every number in the manuscript
- `data/` – linked datasets and fitted values
- `logs/` – linkage log and model summaries

Map layers (US Census cartographic boundaries, Natural Earth, Esri World Shaded
Relief tiles) are downloaded once into `gis/`. Set `MAP_TILES=0` to draw maps
without relief tiles.

## Files

| File | Purpose |
|---|---|
| `config.py` | paths, years, column names, code mappings, thresholds |
| `utils.py` | model fitting (logit, NB with fallbacks), cluster and wild-bootstrap inference, table output |
| `gis_layers.py` | base map: boundaries, water, relief, labels, scale bar, locator |
