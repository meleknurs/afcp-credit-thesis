"""Sample-size sweep tables (joint train=cal vs calibration-only), Taiwan and GMSC.

Usage: python sweep_tables.py <afcp_results_dir_of_dataset> <taiwan|gmsc> <out.csv>
Reads the per-run files written by v2/run_afcp.py and writes one row per (regime, bias, design, n_train, n_calib).
Taiwan: women = sex_female level 1; GMSC: youngest group = age_3way level 0 (<=35).
"""
import sys, glob, pandas as pd

d, ds, out = sys.argv[1], sys.argv[2], sys.argv[3]
nte = "nte500" if ds == "taiwan" else "nte1000"
fs = [f for f in glob.glob(f"{d}/*{nte}_cond*") if "_nn_" in f]
df = pd.concat([pd.read_csv(f) for f in fs]); df["conditional"] = df["conditional"].astype(str)
df["design"] = (df.n_train == df.n_calib).map({True: "joint", False: "cal-only"})
if ds == "taiwan": sel, grp, lvl = "selected_sex_female", "sex_female", 1
else:              sel, grp, lvl = "selected_age_3way", "age_3way", 0
key = ["conditional", "bias_type", "bias_eps", "bias_scope", "design", "n_train", "n_calib"]
ov = df[df.attribute_idx == -1]
rows = []
for k, g in ov.groupby(key):
    A, M = g[g.method == "Adaptive"], g[g.method == "Marginal"]
    s = df[(df.attribute_name == grp) & df[key].eq(pd.Series(k, index=key)).all(axis=1)]
    r = dict(zip(key, k)); r["n_seeds"] = M.seed.nunique()
    r["cov_all_Marginal"] = M.Coverage.mean(); r["size_Marginal"] = M.Length.mean()
    for m in ["Marginal", "Partial", "Exhaustive", "Adaptive"]:
        r[f"cov_group_{m}"] = s[(s.method == m) & (s.attribute_level == lvl)].Coverage.mean()
    r["cov_other_Marginal"] = s[(s.method == "Marginal") & (s.attribute_level != lvl)].Coverage.mean()
    r["AFCP_selects_group"] = A[sel].mean(); r["AFCP_selects_none"] = A.selected_none.mean()
    rows.append(r)
pd.DataFrame(rows).round(3).sort_values(key).to_csv(out, index=False)
print("wrote", out, len(rows), "rows from", len(fs), "files")
