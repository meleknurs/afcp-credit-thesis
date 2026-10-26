"""Figure for the small-sample, 100-seed Taiwan analysis.

Left: AFCP's sex-selection rate against sample size, clean data vs hidden
defaults (eps = 0.5), with 95 % bootstrap intervals over seeds. Right: women's coverage
under Marginal and Partial for the same runs.

Usage: python small_sample_figure.py <taiwan_small_sample_summary.csv> <out.png>
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

inp, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(inp)
style = {"none": dict(color="black", marker="o", label="clean data (ε = 0)"),
         "hidden_defaults": dict(color="tab:red", marker="s", label="hidden defaults, ε = 0.5")}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for bias, g in df.groupby("bias"):
    g = g.sort_values("n_train")
    s = style[bias]
    yerr = [g.selects_sex - g.selects_sex_ci_low, g.selects_sex_ci_high - g.selects_sex]
    ax1.errorbar(g.n_train, g.selects_sex, yerr=yerr, capsize=3, **s)
    ax2.plot(g.n_train, g.cov_women_Marginal, linestyle="-", **s)
    ax2.plot(g.n_train, g.cov_women_Partial, linestyle="--", color=s["color"], marker=s["marker"],
             markerfacecolor="white", label=s["label"] + " — Partial")
ax1.set_xscale("log"); ax1.set_xticks([100, 250, 500, 1000]); ax1.set_xticklabels([100, 250, 500, 1000])
ax1.set_ylim(0, 0.5); ax1.set_xlabel("n_train = n_calib"); ax1.set_ylabel("AFCP selects sex (mean over 100 seeds)")
ax1.set_title("Selection rate, 95 % bootstrap interval"); ax1.grid(alpha=0.3); ax1.legend(loc="upper left", fontsize=8)
ax2.set_xscale("log"); ax2.set_xticks([100, 250, 500, 1000]); ax2.set_xticklabels([100, 250, 500, 1000])
ax2.axhline(0.9, color="gray", linestyle=":", linewidth=1)
ax2.set_ylim(0.82, 0.96); ax2.set_xlabel("n_train = n_calib"); ax2.set_ylabel("coverage — women (clean test)")
ax2.set_title("Women's coverage: Marginal (solid) vs Partial (dashed)"); ax2.grid(alpha=0.3); ax2.legend(fontsize=7)
fig.suptitle("Taiwan, NN, marginal calibration, n_test = 500, 100 seeds per point", fontsize=10)
fig.tight_layout()
fig.savefig(out, dpi=150, metadata={"Software": None})
print("figure ->", out)
