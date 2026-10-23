"""RQ4: does the conclusion depend on how the group is defined?

Usage: python group_def_table.py <rq4_groupdef_dir> <default_results_dir_root> <out.csv>
For each (dataset, candidate definition, model, regime) reports, averaged over seeds: Marginal worst-group coverage and
coverage gap for the *evaluated* definition equal to the candidate, size of the smallest group (n_count), AFCP selection rate
and Adaptive-vs-Marginal change in worst-group coverage and gap. One AFCP run per (dataset, definition): the definition is passed
as the AFCP candidate attribute (run_afcp.py dataset@column) and evaluated in the same run.
"""
import sys, glob, os, pandas as pd

d, out = sys.argv[1], sys.argv[3]
df = pd.concat([pd.read_csv(f) for f in glob.glob(f"{d}/*.txt")]); df["conditional"] = df["conditional"].astype(str)
rows = []
for (ds, cand, model, cond), g in df.groupby(["dataset", "cand_def", "model", "conditional"]):
    sel_col = f"selected_{cand}"
    ov = g[g.attribute_idx == -1]; grp = g[g.attribute_name == cand]
    def wg(meth):  # worst-group coverage and gap per seed, then mean
        x = grp[grp.method == meth].groupby("seed").Coverage.agg(["min", "max"])
        return x["min"].mean(), (x["max"] - x["min"]).mean()
    row = dict(dataset=ds, definition=cand, model=model, conditional=cond, n_seeds=g.seed.nunique())
    row["smallest_group_test_n"] = grp[grp.method == "Marginal"].groupby("seed").n_count.min().mean()
    for m in ["Marginal", "Partial", "Exhaustive", "Adaptive", "Adaptive1"]:
        if (grp.method == m).any():
            row[f"worst_{m}"], row[f"gap_{m}"] = wg(m)
    for m in ["Adaptive", "Adaptive1"]:
        a = ov[ov.method == m]
        if len(a): row[f"selects_{m}"] = a[sel_col].mean(); row[f"none_{m}"] = a.selected_none.mean()
    row["set_size_Marginal"] = ov[ov.method == "Marginal"].Length.mean()
    rows.append(row)
pd.DataFrame(rows).round(3).sort_values(["dataset", "model", "conditional", "definition"]).to_csv(out, index=False)
print("wrote", out, len(rows), "rows")
