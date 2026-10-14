"""Produce the per-dataset cell-count tables (docs/V2_DESIGN.md, Section 1).

For every dataset and group column: n, share, bad count, bad rate, and the
expected number of (group x bad) points in calibration / test sets of the
sizes planned in the design, plus the binomial SE of a 90 % coverage estimate
in the smallest cell.
Writes thesis_experiments/results/v2/cell_counts_<dataset>.csv and prints a summary.
"""
import os, sys, math
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v2.data import LOADERS, cell_counts

PLAN = {  # (n_calib, n_test) planned per dataset
    "sgc": (350, 300),
    "gmsc": (10000, 2000),
    "taiwan": (6000, 10000),
}
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "v2")
os.makedirs(OUT, exist_ok=True)

for key, loader in LOADERS.items():
    X, y, meta = loader()
    n_cal, n_test = PLAN[key]
    print(f"\n=== {meta['name']}: n={len(y)}, bad rate={y.mean():.3f}")
    for n in meta["notes"]:
        print("  -", n)
    tabs = []
    for gc in meta["group_cols"]:
        t = cell_counts(X, y, gc, meta["group_levels"].get(gc))
        t["exp_bad_in_calib"] = (t["share_bad_cell"] * n_cal).round(1)
        t["exp_bad_in_test"] = (t["share_bad_cell"] * n_test).round(1)
        t["exp_group_in_test"] = (t["share"] * n_test).round(0)
        t["se_cov90_group_test_pp"] = (100 * (0.09 / (t["share"] * n_test)) ** 0.5).round(1)
        t["se_cov90_badcell_test_pp"] = (100 * (0.09 / (t["share_bad_cell"] * n_test).clip(lower=1)) ** 0.5).round(1)
        tabs.append(t)
    tab = pd.concat(tabs, ignore_index=True)
    tab.to_csv(os.path.join(OUT, f"cell_counts_{meta['name']}.csv"), index=False)
    pd.set_option("display.width", 220)
    print(tab[["group", "label", "n", "share", "bad_rate", "exp_bad_in_calib", "exp_bad_in_test",
               "exp_group_in_test", "se_cov90_group_test_pp", "se_cov90_badcell_test_pp"]].round(3).to_string(index=False))
