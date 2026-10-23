"""Multi-attribute analysis (GMSC: age_3way + income_hi + dependents_any).

Usage: python multi_attribute_table.py <results_dir_gmsc_multi> <out_prefix>
Writes <out_prefix>_selection.csv (selection rates per regime/variant, per-seed 'none') and <out_prefix>_flags.csv
(per-label flag rates for Adaptive, per seed).
"""
import sys, glob, pandas as pd

d, pre = sys.argv[1], sys.argv[2]
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{d}/*.txt"))]); df["conditional"] = df["conditional"].astype(str)
ov = df[df.attribute_idx == -1]
sel = [c for c in df.columns if c.startswith("selected")]; fl = [c for c in df.columns if c.startswith("flag")]
sel_t = ov[ov.method.isin(["Adaptive", "Adaptive1"])].groupby(["method", "conditional"])[sel].mean().round(3)
sel_t.to_csv(pre + "_selection.csv")
A = ov[ov.method == "Adaptive"]
A.set_index(["conditional", "seed"])[fl].round(3).to_csv(pre + "_flags.csv")
print(sel_t.to_string())
