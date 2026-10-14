"""EDA for the UCI Taiwan credit-card default data (docs/CREDIT_TAIWAN.md)."""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v2.data import load_taiwan, REPO_ROOT

X, y, meta = load_taiwan()
raw = pd.read_csv(os.path.join(REPO_ROOT, "data", "default_credit_card_clients.csv"))
out_fig = os.path.join(REPO_ROOT, "docs", "figures", "taiwan_eda_sensitive_attributes.png")
out_md = os.path.join(REPO_ROOT, "thesis_experiments", "results", "v2", "eda_taiwan_tables.md")

lines = []
P = lambda s="": lines.append(s)
P(f"n = {len(y)}, default rate = {y.mean():.4f}, missing values = {int(raw.isna().sum().sum())}")
P(); P("Raw code frequencies (before recoding):")
for c in ["SEX", "EDUCATION", "MARRIAGE"]:
    P(f"  {c}: " + ", ".join(f"{int(k)}:{int(v)}" for k, v in raw[c].value_counts().sort_index().items()))
P(f"  AGE: min {raw.AGE.min()}, median {raw.AGE.median()}, max {raw.AGE.max()}")
P("  PAY_0 codes: " + ", ".join(f"{int(k)}:{int(v)}" for k, v in raw["PAY_0"].value_counts().sort_index().items()))
P(); P("Default rate by group (after recoding):")
for gc in meta["group_cols"]:
    t = y.groupby(X[gc]).agg(["count", "mean"])
    t.index = [meta["group_levels"][gc].get(i, i) for i in t.index]
    P(f"  {gc}: " + "; ".join(f"{i}: n={int(r['count'])}, default={r['mean']:.3f}" for i, r in t.iterrows()))
P(); P("Default rate by age decade:")
dec = pd.cut(X["AGE"], bins=[20, 30, 40, 50, 60, 80], right=False, labels=["20-29", "30-39", "40-49", "50-59", "60+"])
t = y.groupby(dec).agg(["count", "mean"])
for i, r in t.iterrows(): P(f"  {i}: n={int(r['count'])}, default={r['mean']:.3f}")
P(); P("Intersection sex x age_3way (n, default rate):")
ct = pd.crosstab([X["sex_female"]], X["age_3way"], values=y, aggfunc=["count", "mean"]).round(3)
P(ct.to_string())
open(out_md, "w").write("\n".join(lines)); print("\n".join(lines))

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for lab, m, col in [("no default", y == 0, "#4C72B0"), ("default", y == 1, "#DD8452")]:
    ax[0, 0].hist(X.loc[m, "AGE"], bins=range(20, 80, 2), alpha=0.6, label=lab, color=col, density=True)
ax[0, 0].set_title("Age distribution by outcome"); ax[0, 0].set_xlabel("age"); ax[0, 0].legend()
for i, gc in enumerate(["sex_female", "marriage_grp", "education_grp"]):
    a = ax.flat[i + 1]
    t = y.groupby(X[gc]).mean(); n = X[gc].value_counts().sort_index()
    labels = [meta["group_levels"][gc].get(k, k) for k in t.index]
    bars = a.bar(labels, t.values, color="#4C72B0")
    for b, k in zip(bars, t.index): a.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"n={int(n[k])}", ha="center", fontsize=8)
    a.axhline(y.mean(), ls="--", color="grey", lw=1); a.set_ylim(0, 0.35); a.set_title(f"Default rate by {gc}")
    a.tick_params(axis="x", labelrotation=15)
fig.suptitle("Taiwan credit-card default data: sensitive attributes (dashed = overall default rate)")
fig.tight_layout(); fig.savefig(out_fig, dpi=150); print("figure ->", out_fig)
