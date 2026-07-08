"""Exploratory data analysis for the Give Me Some Credit dataset.

Loads the Kaggle Give Me Some Credit training data, removes the index column,
and produces the summary statistics and figures documented in docs/CREDIT_GMSC.md.

Usage:
    python eda_give_me_credit.py

Outputs:
    docs/figures/gmsc_eda_sensitive_attributes.png
    stdout: summary tables
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

LOCAL_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "cs-training.csv")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")


def load_gmsc() -> pd.DataFrame:
    """Load Give Me Some Credit training data from local CSV."""
    try:
        df = pd.read_csv(LOCAL_CSV)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        return df
    except FileNotFoundError:
        print(f"Error: {LOCAL_CSV} not found")
        sys.exit(1)


def summarize(df: pd.DataFrame) -> None:
    """Print summary statistics for GMSC dataset."""
    print("\n" + "=" * 80)
    print("GIVE ME SOME CREDIT - DATASET SUMMARY")
    print("=" * 80)
    
    # Data shape
    print(f"\nDataset shape: {df.shape[0]:,} observations, {df.shape[1]} columns")
    
    # Target variable
    target_col = "SeriousDlqin2yrs"
    print(f"\n[Target: {target_col}]")
    print(f"Default rate (SeriousDlqin2yrs=1): {df[target_col].mean():.4f}")
    print(f"Distribution: {df[target_col].value_counts().to_dict()}")
    
    # Age
    print(f"\n[Age]")
    age_col = "age"
    print(f"Mean: {df[age_col].mean():.1f}, Median: {df[age_col].median():.0f}, Min: {df[age_col].min()}, Max: {df[age_col].max()}")
    age_zero = (df[age_col] <= 0).sum()
    print(f"Age <= 0 count: {age_zero}")
    if age_zero > 0:
        print(f"  Note: Observations with age ≤ 0 are likely data-entry errors.")
    
    # Age bins
    age_median = df[age_col].median()
    df["age_bin_median"] = (df[age_col] <= age_median).astype(int)
    df["age_bin_40"] = (df[age_col] >= 40).astype(int)
    
    print(f"\n[Age Binning]")
    print(f"Median split (≤{age_median:.0f} vs >{age_median:.0f}): {dict(df['age_bin_median'].value_counts().sort_index())}")
    print(f"Policy-based split (≥40 vs <40): {dict(df['age_bin_40'].value_counts().sort_index())}")
    
    # Default rates by age
    print(f"\n[Default Rate Across Age Groups]")
    print(f"Median split (≤{age_median:.0f} vs >{age_median:.0f}):")
    for bin_val in sorted(df["age_bin_median"].unique()):
        label = f"≤{age_median:.0f}" if bin_val == 1 else f">{age_median:.0f}"
        default_rate = df[df["age_bin_median"] == bin_val][target_col].mean()
        n = (df["age_bin_median"] == bin_val).sum()
        print(f"  {label}: {default_rate:.4f} (n={n:,})")
    
    print(f"Policy-based split (≥40 vs <40):")
    for bin_val in sorted(df["age_bin_40"].unique()):
        label = "≥40" if bin_val == 1 else "<40"
        default_rate = df[df["age_bin_40"] == bin_val][target_col].mean()
        n = (df["age_bin_40"] == bin_val).sum()
        print(f"  {label}: {default_rate:.4f} (n={n:,})")
    
    # 10-year age groups
    print(f"\n[Default Rate by 10-Year Age Groups]")
    age_groups = pd.cut(df[age_col], 
                        bins=[0, 30, 40, 50, 60, 70, 150],
                        labels=["<30", "30–39", "40–49", "50–59", "60–69", "70+"],
                        include_lowest=True)
    age_group_rates = df.groupby(age_groups, observed=True)[target_col].agg(['count', 'mean'])
    age_group_rates.columns = ['n', 'default_rate']
    print(age_group_rates.to_string())
    
    print(f"\nObserved pattern: Default risk decreases approximately monotonically with age.")
    print(f"No obvious natural breakpoint is visible.")
    
    # Missing values
    print(f"\n[Missing Values]")
    missing = df.isnull().sum()
    missing_count = missing.sum()
    if missing_count > 0:
        print(f"Total missing values: {missing_count:,}")
        print(missing[missing > 0].to_string())
        print(f"\nNote: These variables require preprocessing before model training.")
        print(f"The specific imputation strategy is described in the experimental setup.")
    else:
        print("No missing values")


def make_figure(df: pd.DataFrame, out_path: str) -> None:
    """Create EDA figure with 4 panels."""
    target_col = "SeriousDlqin2yrs"
    age_col = "age"
    age_median = df[age_col].median()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Age distribution by default status
    ax = axes[0, 0]
    for target_val in sorted(df[target_col].unique()):
        label = f"{'Default' if target_val == 1 else 'No Default'}"
        ax.hist(df[df[target_col] == target_val][age_col], alpha=0.6, label=label, bins=30)
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    ax.set_title("Age Distribution by Default Status")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Default rate by 10-year age groups
    ax = axes[0, 1]
    age_groups = pd.cut(df[age_col], 
                        bins=[0, 30, 40, 50, 60, 70, 150],
                        labels=["<30", "30–39", "40–49", "50–59", "60–69", "70+"],
                        include_lowest=True)
    
    age_group_rates = df.groupby(age_groups, observed=True)[target_col].mean()
    x_pos = range(len(age_group_rates))
    ax.plot(x_pos, age_group_rates, marker='o', linewidth=2, markersize=8, color='steelblue')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(age_group_rates.index, rotation=45)
    ax.set_ylabel("Default Rate")
    ax.set_xlabel("Age Group")
    ax.set_title("Default Rate by Age Group\n(Monotonic decline, no natural breakpoint)")
    ax.grid(alpha=0.3)
    
    # Add reference lines
    ax.axvline(x=1.5, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Policy-based (~40)')
    ax.axvline(x=2.5, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Median split (~52)')
    ax.legend(fontsize=8)
    
    # Plot 3: Debt ratio distribution
    ax = axes[1, 0]
    ax.hist(df["DebtRatio"], bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Debt Ratio")
    ax.set_ylabel("Frequency")
    ax.set_title("Debt Ratio Distribution")
    ax.grid(alpha=0.3)
    
    # Plot 4: Monthly income distribution (log scale)
    ax = axes[1, 1]
    income_valid = df[df["MonthlyIncome"] > 0]["MonthlyIncome"]
    ax.hist(np.log10(income_valid), bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Log10(Monthly Income)")
    ax.set_ylabel("Frequency")
    ax.set_title("Monthly Income Distribution (log scale)")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Figure saved to: {out_path}")
    plt.close()


if __name__ == "__main__":
    print("Loading data...")
    df = load_gmsc()
    print(f"✓ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    summarize(df)
    
    fig_path = os.path.join(FIG_DIR, "gmsc_eda_sensitive_attributes.png")
    print("\nCreating figure...")
    make_figure(df, fig_path)
    
    print("\n" + "=" * 80)
    print("EDA COMPLETE")
    print("=" * 80)