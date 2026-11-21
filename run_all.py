"""Runs the pipeline in order; stops at the first failing step."""
import importlib
import time

STEPS = ["01_build_datasets", "02_descriptives", "03_models",
         "04_random_parameters", "05_spatial_hotspots", "09_additional_checks",
         "08_publication_figures"]

for s in STEPS:
    t = time.time()
    print(f"\n######## {s}")
    importlib.import_module(s).main()
    print(f"######## {s} done in {time.time() - t:.0f}s")
