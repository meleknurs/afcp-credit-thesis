"""
Combine the per-seed calibration-size sweep results (mc_give_me_credit_sweep.py)
into a single 5-seed summary table for the manuscript's Table 4.

Run this AFTER generating sweep_summary_ntest1000_seed{seed}.txt for
seed in {2023, 2024, 2025, 2026, 2027} via:

    python thesis_experiments/mc_give_me_credit_sweep.py 2023
    python thesis_experiments/mc_give_me_credit_sweep.py 2024
    python thesis_experiments/mc_give_me_credit_sweep.py 2025
    python thesis_experiments/mc_give_me_credit_sweep.py 2026
    python thesis_experiments/mc_give_me_credit_sweep.py 2027

Then:
    python thesis_experiments/combine_sweep_results.py
"""
import os
import glob
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(SCRIPT_DIR, "results", "give_me_credit", "sample_size_sweep")
RESULTS_DIR = os.path.join(SWEEP_DIR, "raw")

SEEDS = [2023, 2024, 2025, 2026, 2027]

frames = []
missing = []
for seed in SEEDS:
    path = os.path.join(RESULTS_DIR, f"sweep_summary_ntest1000_seed{seed}.txt")
    if not os.path.exists(path):
        missing.append(seed)
        continue
    df = pd.read_csv(path)
    df["seed"] = seed
    frames.append(df)

if missing:
    print(f"WARNING: missing seeds {missing} -- run mc_give_me_credit_sweep.py for these first.")

if not frames:
    raise SystemExit("No sweep_summary files found. Nothing to combine.")

all_df = pd.concat(frames, ignore_index=True)

# Drop SKIPPED rows (n_train_calib too large for available data) before averaging
all_df["afcp_selected_age"] = pd.to_numeric(all_df["afcp_selected_age"], errors="coerce")

summary = (
    all_df.groupby("n_train_calib")["afcp_selected_age"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={
        "mean": "afcp_selected_age_mean",
        "std": "afcp_selected_age_std",
        "count": "n_seeds",
    })
)
summary["afcp_selected_age_mean_pct"] = (summary["afcp_selected_age_mean"] * 100).round(1)
summary["afcp_selected_age_std_pct"] = (summary["afcp_selected_age_std"] * 100).round(1)

out_path = os.path.join(SWEEP_DIR, "summary_5seed_combined.csv")
summary.to_csv(out_path, index=False)

print(f"Seeds combined: {sorted(all_df['seed'].unique().tolist())}")
print(summary.to_string(index=False))
print(f"\nWritten to: {out_path}")
print("\nFor Table 4, report as: afcp_selected_age_mean_pct% +/- afcp_selected_age_std_pct%")
