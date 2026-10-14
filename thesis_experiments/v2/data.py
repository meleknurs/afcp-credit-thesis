"""
v2 data loading for the three credit datasets (see docs/V2_DESIGN.md, Sections 1-2).

Every loader returns (X, y, meta):
    X     pandas DataFrame: model features + group-label columns (ints)
    y     pandas Series, 1 = bad / default (the rare, positive class), 0 = good
    meta  dict with keys
            name, group_cols (list of group-label columns), group_levels
            ({col: {level: label}}), continuous_cols (to standardise),
            categorical_cols (kept as integer codes, not standardised),
            notes (list of preprocessing statements for the thesis)
Group labels are built on the full data before any sampling (Phase-1 convention)
and are passed to the model as features (Zhou & Sesia design).
"""
import os
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

SGC_COLS = {
    "laufkont": "status", "laufzeit": "duration", "moral": "credit_history",
    "verw": "purpose", "hoehe": "amount", "sparkont": "savings",
    "beszeit": "employment_duration", "rate": "installment_rate",
    "famges": "personal_status_sex", "buerge": "other_debtors",
    "wohnzeit": "present_residence", "verm": "property", "alter": "age",
    "weitkred": "other_installment_plans", "wohn": "housing",
    "bishkred": "number_credits", "beruf": "job", "pers": "people_liable",
    "telef": "telephone", "gastarb": "foreign_worker", "kredit": "credit_risk",
}


def load_sgc():
    df = pd.read_csv(os.path.join(DATA_DIR, "SouthGermanCredit.asc"), sep=r"\s+")
    df = df.rename(columns=SGC_COLS)
    # Grömping (2019): kredit 1 = good, 0 = bad. Recode so that bad = 1.
    y = (df["credit_risk"] == 0).astype(int).rename("bad")
    X = df.drop(columns=["credit_risk"]).copy()
    X["age_le25"] = (X["age"] <= 25).astype(int)          # 1 = young (unprivileged, AIF360 convention)
    X["age_le33"] = (X["age"] <= 33).astype(int)          # median split, Phase-1 robustness check
    meta = dict(
        name="south_german_credit",
        group_cols=["age_le25", "age_le33", "personal_status_sex"],
        candidate_cols=["age_le25"],
        group_levels={
            "age_le25": {0: ">25", 1: "<=25"},
            "age_le33": {0: ">33", 1: "<=33"},
            "personal_status_sex": {1: "male divorced/separated", 2: "female non-single or male single",
                                    3: "male married/widowed", 4: "female single"},
        },
        continuous_cols=["duration", "amount", "age"],
        categorical_cols=[c for c in X.columns if c not in ["duration", "amount", "age",
                                                            "age_le25", "age_le33"]],
        notes=[
            "South German Credit (Grömping 2019, UCI 522), corrected coding; Phase 0 used the "
            "Statlog german.data-numeric file, replaced because of the documented coding errors.",
            "Target recoded: bad = 1 (kredit = 0), good = 0.",
            "Groups: age <=25 vs >25 (Kamiran & Calders 2009 / AIF360); age <=33 (median) as "
            "robustness check; personal_status_sex level 2 is a mixed-sex category and is "
            "reported as such, not relabelled. foreign_worker tested in Phase 0 and dropped "
            "(52 test points, no detectable disparity).",
            "All other variables kept as Grömping's ordinal/categorical integer codes.",
        ],
    )
    return X, y, meta


def load_gmsc(drop_sentinel=True):
    df = pd.read_csv(os.path.join(DATA_DIR, "cs-training.csv"))
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    n0 = len(df)
    delinq = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse",
              "NumberOfTimes90DaysLate"]
    notes = [f"Give Me Some Credit (Kaggle), {n0} rows loaded."]
    sent = (df[delinq] >= 96).any(axis=1)
    if drop_sentinel:
        df = df[~sent]
        notes.append(f"Dropped {int(sent.sum())} rows with sentinel codes 96/98 in the delinquency "
                     "counters (present in all three counters simultaneously; undocumented by Kaggle).")
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates()
    notes.append(f"Dropped {n_dup} exact duplicate rows.")
    n_age0 = int((df["age"] <= 0).sum())
    df = df[df["age"] > 0]
    notes.append(f"Dropped {n_age0} rows with age <= 0.")
    df = df.reset_index(drop=True)

    y = df["SeriousDlqin2yrs"].astype(int).rename("bad")
    X = df.drop(columns=["SeriousDlqin2yrs"]).copy()
    # Missing-value indicators; imputation itself is done per split with training medians
    X["MonthlyIncome_missing"] = X["MonthlyIncome"].isna().astype(int)
    X["NumberOfDependents_missing"] = X["NumberOfDependents"].isna().astype(int)
    # Heavy-tailed ratios: log1p (applied before per-split standardisation)
    for c in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]:
        X[c] = np.log1p(X[c])
    notes.append("log1p applied to RevolvingUtilizationOfUnsecuredLines, DebtRatio, MonthlyIncome; "
                 "MonthlyIncome / NumberOfDependents imputed per split with the training median / 0 "
                 "plus missing indicators.")
    X["age_3way"] = pd.cut(X["age"], bins=[0, 35, 55, 150], labels=[0, 1, 2], include_lowest=True).astype(int)
    X["age_lt25"] = (X["age"] < 25).astype(int)
    X["age_ge60"] = (X["age"] >= 60).astype(int)
    notes.append("Groups: age_3way (<=35 / 36-55 / >55, Phase-1 primary); age <25 (Kozodoi et al. "
                 "2022 comparability row, ~1.4 % of rows); age >=60 (sensitivity, non-standard).")
    meta = dict(
        name="give_me_credit",
        group_cols=["age_3way", "age_lt25", "age_ge60"],
        candidate_cols=["age_3way"],
        group_levels={"age_3way": {0: "<=35", 1: "36-55", 2: ">55"},
                      "age_lt25": {0: ">=25", 1: "<25"},
                      "age_ge60": {0: "<60", 1: ">=60"}},
        continuous_cols=[c for c in X.columns if c not in
                         ["age_3way", "age_lt25", "age_ge60", "MonthlyIncome_missing", "NumberOfDependents_missing"]],
        categorical_cols=["MonthlyIncome_missing", "NumberOfDependents_missing"],
        impute={"MonthlyIncome": "median", "NumberOfDependents": 0},
        notes=notes,
    )
    return X, y, meta


def load_taiwan():
    df = pd.read_csv(os.path.join(DATA_DIR, "default_credit_card_clients.csv"))
    df = df.drop(columns=["ID"])
    y = df["default payment next month"].astype(int).rename("bad")
    X = df.drop(columns=["default payment next month"]).copy()
    n_edu = int(X["EDUCATION"].isin([0, 5, 6]).sum()); n_mar = int((X["MARRIAGE"] == 0).sum())
    X.loc[X["EDUCATION"].isin([0, 5, 6]), "EDUCATION"] = 4
    X.loc[X["MARRIAGE"] == 0, "MARRIAGE"] = 3
    X["sex_female"] = (X["SEX"] == 2).astype(int)
    X["education_grp"] = X["EDUCATION"].astype(int)       # 1 grad, 2 univ, 3 high school, 4 others
    X["marriage_grp"] = X["MARRIAGE"].astype(int)         # 1 married, 2 single, 3 others
    X["age_lt25"] = (X["AGE"] < 25).astype(int)
    X["age_3way"] = pd.cut(X["AGE"], bins=[0, 35, 55, 150], labels=[0, 1, 2], include_lowest=True).astype(int)
    # candidate versions for AFCP's selection step: small "others" levels merged (docs/CREDIT_TAIWAN.md 1.4)
    X["marriage_cand"] = X["marriage_grp"].replace({3: 2}).astype(int)      # others -> single
    X["education_cand"] = X["education_grp"].replace({4: 3}).astype(int)    # others -> high school
    pay = [c for c in X.columns if c.startswith("PAY_") and not c.startswith("PAY_AMT")]
    amt = [c for c in X.columns if c.startswith("BILL_AMT") or c.startswith("PAY_AMT")] + ["LIMIT_BAL"]
    meta = dict(
        name="taiwan_credit",
        group_cols=["sex_female", "marriage_grp", "education_grp", "age_lt25", "age_3way"],
        candidate_cols=["sex_female", "marriage_cand", "education_cand", "age_lt25"],
        group_levels={"sex_female": {0: "male", 1: "female"},
                      "marriage_cand": {1: "married", 2: "single/others"},
                      "education_cand": {1: "graduate", 2: "university", 3: "high school/others"},
                      "marriage_grp": {1: "married", 2: "single", 3: "others"},
                      "education_grp": {1: "graduate", 2: "university", 3: "high school", 4: "others"},
                      "age_lt25": {0: ">=25", 1: "<25"},
                      "age_3way": {0: "<=35", 1: "36-55", 2: ">55"}},
        continuous_cols=amt + ["AGE"] + pay,
        categorical_cols=["SEX", "EDUCATION", "MARRIAGE", "marriage_cand", "education_cand"],
        notes=[
            "Default of Credit Card Clients (Yeh & Lien 2009, UCI 350, CC BY 4.0); all 30,000 rows used.",
            f"EDUCATION codes 0/5/6 ({n_edu} rows) folded into 4 'others'; MARRIAGE 0 ({n_mar} rows) "
            "folded into 3 'others' (undocumented codes, common recode).",
            "PAY_0..PAY_6 kept numeric with -2 / -1 / 0 retained (no consumption / paid in full / revolving).",
            "Groups: sex (primary, Le Quy et al. 2022); marriage; education; age <25; age_3way "
            "(same cut as GMSC for cross-dataset comparison).",
        ],
    )
    return X, y, meta


LOADERS = {"sgc": load_sgc, "gmsc": load_gmsc, "taiwan": load_taiwan}


def cell_counts(X, y, group_col, levels=None):
    """Counts and bad rate per (group level × label) on the data given."""
    g = X[group_col]
    rows = []
    for lvl in sorted(g.unique()):
        m = g == lvl
        rows.append(dict(group=group_col, level=lvl,
                         label=(levels or {}).get(lvl, str(lvl)),
                         n=int(m.sum()), share=m.mean(),
                         n_bad=int(y[m].sum()), bad_rate=y[m].mean(),
                         share_bad_cell=(y[m].sum()) / len(y)))
    return pd.DataFrame(rows)
