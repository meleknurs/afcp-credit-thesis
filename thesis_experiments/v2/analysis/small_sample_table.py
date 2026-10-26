"""Small-sample, many-seed analysis of AFCP's selection behaviour (Taiwan).

Summarises the 100-seed runs at n_train = n_calib in {100, 250, 500, 1000}
(Zhou & Sesia-style sample sizes), clean data and hidden-defaults injection
(eps = 0.5, train + calibration), marginal regime, neural network.

For each (bias, n) cell the table reports the number of seeds, AFCP's
sex-selection rate (fraction of test points for which sex was selected, as in
all other result tables, averaged over seeds) with a 95 % bootstrap confidence
interval over seeds, the share of seeds that select sex for at least half of
the test points, the per-label flag rates, women's and men's coverage under
each method, and set size.

Usage: python small_sample_table.py <afcp_results_dir_of_taiwan> <out.csv>
"""
import glob
import sys

import numpy as np
import pandas as pd


def bootstrap_ci(x, n_boot=10000, seed=0):
    """95 % percentile bootstrap interval for the mean of x over seeds."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


d, out = sys.argv[1], sys.argv[2]
SIZES = [100, 250, 500, 1000]
fs = [f for n in SIZES for f in glob.glob(f"{d}/taiwan_nn_ntr{n}_ncal{n}_nte500_condFalse_td0.0_sl0.1_seed*.txt")]
df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
df["bias"] = np.where(df.bias_type.fillna("none").eq("none") | df.bias_type.isna(), "none", "hidden_defaults")

rows = []
for (bias, n), g in df.groupby(["bias", "n_train"]):
    ov = g[g.attribute_idx == -1]
    A, M = ov[ov.method == "Adaptive"], ov[ov.method == "Marginal"]
    sex = g[g.attribute_name == "sex_female"]
    m = len(A)
    lo, hi = bootstrap_ci(A.selected_sex_female)
    r = dict(bias=bias, n_train=n, n_calib=n, n_test=500, n_seeds=m,
             selects_sex=A.selected_sex_female.mean(), selects_sex_ci_low=lo, selects_sex_ci_high=hi,
             seeds_selecting_sex=(A.selected_sex_female >= 0.5).mean(),
             selects_none=A.selected_none.mean(),
             flag_label0_sex=A.flag_label0_sex_female.mean(),
             flag_label1_sex=A.flag_label1_sex_female.mean(),
             cal_auroc=A.cal_auroc.mean())
    for meth in ["Marginal", "Partial", "Exhaustive", "Adaptive"]:
        s = sex[sex.method == meth]
        r[f"cov_women_{meth}"] = s[s.attribute_level == 1].Coverage.mean()
        r[f"cov_men_{meth}"] = s[s.attribute_level == 0].Coverage.mean()
    r["cov_women_Marginal_sd"] = sex[(sex.method == "Marginal") & (sex.attribute_level == 1)].Coverage.std()
    r["size_Marginal"] = M.Length.mean()
    r["size_Adaptive"] = A.Length.mean()
    rows.append(r)

res = pd.DataFrame(rows).sort_values(["bias", "n_train"], ascending=[False, True])
res.round(3).to_csv(out, index=False)
print(f"wrote {out}: {len(res)} rows from {len(fs)} files")
print(res.round(3).to_string(index=False))
