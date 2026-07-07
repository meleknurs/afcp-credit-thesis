"""Exploratory data analysis for the South German Credit dataset.

Loads the dataset (via ucimlrepo, falling back to a local .asc file),
maps German column names to English (Groemping 2019), and produces the
summary statistics and figures documented in docs/CREDIT_SOUTH_GERMAN.md.

Usage:
    python eda_south_german.py

Outputs:
    docs/figures/sgc_eda_sensitive_attributes.png
    stdout: summary tables
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# German -> English column names, following Groemping (2019),
# read_SouthGermanCredit.R and the accompanying code table.
COL_MAP = {
    "laufkont": "status",
    "laufzeit": "duration",
    "moral": "credit_history",
    "verw": "purpose",
    "hoehe": "amount",
    "sparkont": "savings",
    "beszeit": "employment_duration",
    "rate": "installment_rate",
    "famges": "personal_status_sex",
    "buerge": "other_debtors",
    "wohnzeit": "present_residence",
    "verm": "property",
    "alter": "age",
    "weitkred": "other_installment_plans",
    "wohn": "housing",
    "bishkred": "number_credits",
    "beruf": "job",
    "pers": "people_liable",
    "telef": "telephone",
    "gastarb": "foreign_worker",
    "kredit": "credit_risk",
}

LOCAL_ASC = os.path.join(os.path.dirname(__file__), "..", "data", "SouthGermanCredit.asc")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")


def load_south_german() -> pd.DataFrame:
    """Load the South German Credit dataset with English column names."""
    try:
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=522)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    except Exception as exc:  # network unavailable etc. -> local fallback
        print(f"[info] ucimlrepo failed ({exc}); using local file {LOCAL_ASC}")
        df = pd.read_csv(LOCAL_ASC, sep=" ")
    df = df.rename(columns=COL_MAP)
    # Guard against silent column reordering: sensitive-attribute indices
    # are hard-coded downstream (see docs/CREDIT_SOUTH_GERMAN.md, sec. 1.2).
    assert list(df.columns).index("personal_status_sex") == 8
    assert list(df.columns).index("age") == 12
    assert list(df.columns).index("foreign_worker") == 19
    assert list(df.columns).index("credit_risk") == 20
    return df


def summarize(df: pd.DataFrame) -> None:
    """Print the summary tables used in the EDA section of the docs."""
    print("Shape:", df.shape)
    print("Missing values:", int(df.isna().sum().sum()))
    print("\nTarget distribution (credit_risk, 0=bad / 1=good):")
    print(df["credit_risk"].value_counts().sort_index())

    print("\n--- age ---")
    print(df["age"].describe().round(2))
    for label, grouping in [
        ("age <= 25 vs > 25 (Kamiran & Calders 2009 cut)", pd.cut(df["age"], [0, 25, 100])),
        ("median split (33)", pd.cut(df["age"], [0, 33, 100])),
        ("quartiles", pd.qcut(df["age"], 4)),
    ]:
        print(f"\n{label}:")
        print(df.groupby(grouping, observed=True)["credit_risk"].agg(["count", "mean"]).round(3))

    print("\n--- personal_status_sex ---")
    print("1=male div/sep, 2=female non-single OR male single, 3=male mar/wid, 4=female single")
    print(df.groupby("personal_status_sex")["credit_risk"].agg(["count", "mean"]).round(3))

    print("\n--- foreign_worker (1=yes, 2=no) ---")
    print(df.groupby("foreign_worker")["credit_risk"].agg(["count", "mean"]).round(3))


def make_figure(df: pd.DataFrame, out_path: str) -> None:
    """Create the four-panel sensitive-attribute EDA figure."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    for k, lbl in [(0, "bad"), (1, "good")]:
        ax.hist(df[df.credit_risk == k]["age"], bins=25, alpha=0.6, label=f"credit_risk={k} ({lbl})")
    ax.axvline(25, color="red", ls="--", lw=1, label="age=25 cut")
    ax.axvline(33, color="gray", ls=":", lw=1, label="median=33")
    ax.set_title("Age distribution by credit risk")
    ax.set_xlabel("age")
    ax.legend(fontsize=8)

    def rate_bar(ax, grouping, title, color, xlabel=None):
        tab = df.groupby(grouping, observed=True)["credit_risk"].agg(["count", "mean"])
        bars = ax.bar([str(i) for i in tab.index], tab["mean"], color=color)
        for b, (n, m) in zip(bars, tab.itertuples(index=False)):
            ax.text(b.get_x() + b.get_width() / 2, m + 0.01, f"n={n}", ha="center", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.tick_params(axis="x", labelsize=8)

    rate_bar(
        axes[0, 1],
        "personal_status_sex",
        "Good-credit rate by personal_status_sex",
        "steelblue",
        "1=m div/sep  2=f non-single/m single  3=m mar/wid  4=f single",
    )
    rate_bar(axes[1, 0], "foreign_worker", "Good-credit rate by foreign_worker (1=yes, 2=no)", "indianred")
    rate_bar(axes[1, 1], pd.qcut(df["age"], 4), "Good-credit rate by age quartile", "seagreen")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"\nFigure saved to {out_path}")


if __name__ == "__main__":
    data = load_south_german()
    summarize(data)
    make_figure(data, os.path.join(FIG_DIR, "sgc_eda_sensitive_attributes.png"))