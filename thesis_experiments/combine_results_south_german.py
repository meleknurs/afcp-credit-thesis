"""
Combine South German Credit experiment results across all seeds and configurations.

Reads all sgc_*.txt result files, groups by configuration (age25, agemedian, pstatus),
computes mean ± std for each of the 5 methods (Adaptive, Adaptive1, Marginal, Partial,
Exhaustive), calculates coverage/length deltas vs. Marginal, and includes AFCP's
attribute selection rate.

Outputs:
    - thesis_experiments/results/south_german_credit/summary_by_config.csv (publication-ready summary)

Usage:
    python combine_results_south_german.py
"""

import os
import glob
import pandas as pd
import numpy as np

RESULTS_DIR = "thesis_experiments/results/south_german_credit/raw"
OUTPUT_CSV = os.path.join("thesis_experiments/results/south_german_credit", "summary_by_config.csv")


def main():
    """Combine all result files and generate summary CSV."""
    
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "sgc_*.txt")))
    
    if len(files) == 0:
        print(f"[Error] No result files found in {RESULTS_DIR}")
        return
    
    print(f"[Combine] Found {len(files)} result files")
    
    # Parse and group by config
    results_by_config = {}
    for fname in files:
        basename = os.path.basename(fname)
        parts = basename.replace(".txt", "").split("_")
        config = parts[1]  # age25, agemedian, pstatus
        seed_str = parts[-1]  # seed2023
        seed = int(seed_str.replace("seed", ""))
        
        if config not in results_by_config:
            results_by_config[config] = []
        
        df = pd.read_csv(fname)
        df["seed"] = seed
        results_by_config[config].append(df)
        print(f"  {basename:70s} | config={config:11s} seed={seed}")
    
    # Generate summary rows
    summary_rows = []
    
    for config in sorted(results_by_config.keys()):
        dfs = results_by_config[config]
        combined = pd.concat(dfs, ignore_index=True)
        
        # Overall results - all 5 methods
        overall = combined[combined["attribute_idx"] == -1.0]
        
        # Calculate selection rate for Adaptive (overall only)
        adaptive_overall = overall[overall["method"] == "Adaptive"]
        sel_col = [c for c in combined.columns if c.startswith("selected_")][0]
        selection_rate = None
        if len(adaptive_overall) > 0:
            sel_values = adaptive_overall[sel_col].dropna()
            if len(sel_values) > 0:
                selection_rate = sel_values.mean() * 100
        
        # Get mean ± std for each method
        method_stats = {}
        for method in ["Adaptive", "Adaptive1", "Marginal", "Partial", "Exhaustive"]:
            method_data = overall[overall["method"] == method]
            if len(method_data) > 0:
                method_stats[method] = {
                    "cov_mean": method_data["Coverage"].mean(),
                    "cov_std": method_data["Coverage"].std(),
                    "len_mean": method_data["Length"].mean(),
                    "len_std": method_data["Length"].std(),
                    "selection_rate": selection_rate if method == "Adaptive" else None,
                }
        
        # Create rows for overall comparison
        for method in ["Adaptive", "Adaptive1", "Marginal", "Partial", "Exhaustive"]:
            if method in method_stats:
                stats = method_stats[method]
                delta_cov = method_stats[method]["cov_mean"] - method_stats["Marginal"]["cov_mean"]
                delta_len = method_stats[method]["len_mean"] - method_stats["Marginal"]["len_mean"]
                
                row = {
                    "Config": config,
                    "Level": "Overall",
                    "Method": method,
                    "Coverage (mean ± sd)": f"{stats['cov_mean']:.4f}±{stats['cov_std']:.4f}",
                    "Length (mean ± sd)": f"{stats['len_mean']:.3f}±{stats['len_std']:.3f}",
                    "ΔCoverage_vs_Marginal": f"{delta_cov:+.4f}",
                    "ΔLength_vs_Marginal": f"{delta_len:+.3f}",
                }
                
                # Add selection rate for Adaptive only
                if method == "Adaptive" and stats["selection_rate"] is not None:
                    row["Selected_Attr_%"] = f"{stats['selection_rate']:.1f}"
                
                summary_rows.append(row)
    
    # Write summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n[Combine] Summary written to: {OUTPUT_CSV}")
    print(f"[Combine] Ready for Section 2 tables (copy-paste compatible)\n")
    
    # Print summary to console
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()