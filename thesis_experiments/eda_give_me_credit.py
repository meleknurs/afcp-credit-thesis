"""
Exploratory Data Analysis for Give Me Some Credit dataset.

Examines the structure, missing values, and sensitive attributes
of the Give Me Some Credit credit default prediction dataset.
Used to inform design of AFCP fairness experiments.

Sensitive-attribute splits evaluated:
  - Policy-based split: age >= 40 vs < 40 (policy-relevant threshold)
  - Median split: age <= 52 vs > 52 (data-driven, balanced groups)

Primary analysis uses the split with stronger fairness justification based on EDA.

Key finding: This dataset contains age as the primary usable sensitive attribute.
Unlike South German Credit, GMSC does not include gender/sex variables.
Other potential attributes (MonthlyIncome, NumberOfDependents) have substantial
missing values and require preprocessing decisions.

Output: Figure saved to docs/figures/gmsc_eda_sensitive_attributes.png
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Set path
LOCAL_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "cs-training.csv")

print("=" * 80)
print("GIVE ME SOME CREDIT - EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# Load data
print("\n[Data Loading]")
try:
    print("Attempting to load cs-training.csv from data/gmsc/...")
    df = pd.read_csv(LOCAL_CSV)
    print(f"✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Drop index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print(f"✓ Dropped Unnamed: 0 index column. New shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: {LOCAL_CSV} not found.")
    sys.exit(1)

# Display basic info
print(f"\nDataset shape: {df.shape}")
print(f"\nColumn names:\n{list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")

# Check target variable
print(f"\n[Target Variable: SeriousDlqin2yrs]")
target_col = "SeriousDlqin2yrs"
if target_col not in df.columns:
    print(f"Error: {target_col} not found in columns")
    sys.exit(1)

print(f"Value counts:\n{df[target_col].value_counts()}")
print(f"Distribution: {df[target_col].value_counts(normalize=True)}")

# Check age column
print(f"\n[Age Column]")
age_col = "age"
if age_col not in df.columns:
    print(f"Warning: {age_col} not found")
    age_col = None
else:
    print(f"Age - Mean: {df[age_col].mean():.1f}, Median: {df[age_col].median():.1f}, Min: {df[age_col].min()}, Max: {df[age_col].max()}")
    age_zero_count = (df[age_col] <= 0).sum()
    print(f"Age <= 0 count: {age_zero_count}")
    if age_zero_count > 0:
        print("  Note: Observations with age ≤ 0 are likely data-entry errors.")
    print(f"Missing values: {df[age_col].isna().sum()}")

# Create age bins (2-way: policy-based and data-driven)
if age_col:
    print(f"\n[Age Binning: Two splits for sensitivity analysis]")
    age_median = df[age_col].median()
    print(f"Age statistics - Median: {age_median:.0f}, Min: {df[age_col].min()}, Max: {df[age_col].max()}")
    
    # Policy-based split: 40-year threshold
    df["age_bin_40"] = (df[age_col] >= 40).astype(int)
    print(f"\nPolicy-based split (age_bin_40): 1 = age >= 40, 0 = age < 40")
    print(f"Rationale: Policy-relevant age threshold inspired by age-discrimination literature")
    print(df["age_bin_40"].value_counts().sort_index())
    print(f"Proportions:\n{df['age_bin_40'].value_counts(normalize=True).sort_index()}")
    
    # Data-driven median split
    df["age_bin_median"] = (df[age_col] <= age_median).astype(int)
    print(f"\nMedian split (age_bin_median): 1 = age <= {age_median:.0f}, 0 = age > {age_median:.0f}")
    print(f"Rationale: Data-driven split with balanced group sizes")
    print(df["age_bin_median"].value_counts().sort_index())
    print(f"Proportions:\n{df['age_bin_median'].value_counts(normalize=True).sort_index()}")
    
    # Default rate across age groups
    print(f"\n[Default Rate Across Age Groups]")
    print(f"Policy-based split (age ≥40 vs <40):")
    for bin_val in sorted(df["age_bin_40"].unique()):
        label = "≥40" if bin_val == 1 else "<40"
        subset = df[df["age_bin_40"] == bin_val]
        default_rate = (subset[target_col] == 1).mean()
        print(f"  {label}: {default_rate:.3f} (n={len(subset)})")
    
    print(f"Median split (age ≤52 vs >52):")
    for bin_val in sorted(df["age_bin_median"].unique()):
        label = f"≤{age_median:.0f}" if bin_val == 1 else f">{age_median:.0f}"
        subset = df[df["age_bin_median"] == bin_val]
        default_rate = (subset[target_col] == 1).mean()
        print(f"  {label}: {default_rate:.3f} (n={len(subset)})")
    
    # 5-year age group analysis (to identify natural breakpoints)
    print(f"\n[Default Rate by 5-Year Age Groups]")
    age_groups = pd.cut(df[age_col], 
                        bins=[0, 25, 30, 35, 40, 45, 50, 55, 60, 65, 150],
                        labels=["<25", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65+"],
                        include_lowest=True)
    
    default_by_group = df.groupby(age_groups, observed=True)[target_col].agg(['sum', 'count'])
    default_by_group.columns = ['defaults', 'total']
    default_by_group['default_rate'] = default_by_group['defaults'] / default_by_group['total']
    
    print(default_by_group[['total', 'defaults', 'default_rate']])
    print("\nObserved pattern: Default risk decreases approximately monotonically with age.")
    print("No obvious natural breakpoint is visible; median split (52 years) and policy-based split (40 years)")
    print("are evaluated in downstream fairness analysis.")

# Check for missing values
print(f"\n[Missing Values]")
missing = df.isnull().sum()
missing_count = missing.sum()
if missing_count > 0:
    print(f"Total missing values: {missing_count}")
    print(missing[missing > 0])
    print("\nNote: MonthlyIncome and NumberOfDependents have substantial missing values.")
    print("These variables require preprocessing before model training.")
    print("The specific imputation strategy is described in the experimental setup.")
else:
    print("No missing values")

# Summary statistics
print(f"\n[Summary Statistics]")
print(df.describe())

# Create EDA figure
print(f"\n[Creating EDA Figure]")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Age distribution by target
if age_col:
    ax = axes[0, 0]
    for target_val in sorted(df[target_col].unique()):
        label = f"{'Default' if target_val == 1 else 'No Default'}"
        ax.hist(df[df[target_col] == target_val][age_col], alpha=0.6, label=label, bins=30)
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    ax.set_title("Age Distribution by Default Status")
    ax.legend()
    ax.grid(alpha=0.3)

# Plot 2: Default rate by 5-year age groups (line chart to show trend)
if age_col:
    ax = axes[0, 1]
    age_groups = pd.cut(df[age_col], 
                        bins=[0, 25, 30, 35, 40, 45, 50, 55, 60, 65, 150],
                        labels=["<25", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65+"],
                        include_lowest=True)
    
    default_by_group = df.groupby(age_groups, observed=True)[target_col].agg(['sum', 'count'])
    default_by_group.columns = ['defaults', 'total']
    default_by_group['default_rate'] = default_by_group['defaults'] / default_by_group['total']
    
    # Plot line
    age_group_labels = list(default_by_group.index)
    x_pos = range(len(age_group_labels))
    ax.plot(x_pos, default_by_group['default_rate'], marker='o', linewidth=2, markersize=8, color='steelblue')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(age_group_labels, rotation=45)
    ax.set_ylabel("Default Rate")
    ax.set_xlabel("Age Group")
    ax.set_title("Default Rate by Age Group\n(Monotonic decline, no natural breakpoint)")
    ax.grid(alpha=0.3)
    
    # Add vertical lines at candidate thresholds
    ax.axvline(x=3.5, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Policy-based (~40)')
    ax.axvline(x=5.5, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Median split (~52)')
    ax.legend(fontsize=8)

# Plot 3: Debt ratio distribution
if "DebtRatio" in df.columns:
    ax = axes[1, 0]
    ax.hist(df["DebtRatio"], bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Debt Ratio")
    ax.set_ylabel("Frequency")
    ax.set_title("Debt Ratio Distribution")
    ax.grid(alpha=0.3)

# Plot 4: Monthly income distribution (log scale for visibility)
if "MonthlyIncome" in df.columns:
    ax = axes[1, 1]
    income_valid = df[df["MonthlyIncome"] > 0]["MonthlyIncome"]
    ax.hist(np.log10(income_valid), bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Log10(Monthly Income)")
    ax.set_ylabel("Frequency")
    ax.set_title("Monthly Income Distribution (log scale)")
    ax.grid(alpha=0.3)

plt.tight_layout()
output_fig = os.path.join(os.path.dirname(__file__), "..", "docs", "figures", "gmsc_eda_sensitive_attributes.png")
os.makedirs(os.path.dirname(output_fig), exist_ok=True)
plt.savefig(output_fig, dpi=150, bbox_inches="tight")
print(f"✓ Figure saved to: {output_fig}")
plt.close()

print("\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)