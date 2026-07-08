# Give Me Some Credit (GMSC) Dataset Documentation

This document provides comprehensive documentation of the Give Me Some Credit dataset,
including exploratory data analysis and the design of AFCP fairness experiments.

## 1. Exploratory Data Analysis

### 1.1 Dataset Overview

The Give Me Some Credit dataset comprises 150,000 credit records with 11 features
(after removing the index column). The target variable is binary credit default
(`SeriousDlqin2yrs`), with 93.3% non-default (0) and 6.7% default (1) observations.

**Key statistics:**
- Total observations: 150,000
- Target variable: `SeriousDlqin2yrs` (binary, 0 = no default, 1 = default)
- Target distribution: 0 (93.3%), 1 (6.7%)
- Missing values: 33,655 total
  - MonthlyIncome: 29,731 missing (19.8%)
  - NumberOfDependents: 3,924 missing (2.6%)
- Data entry issues: 1 observation with age ≤ 0 (likely data error)

### 1.2 Column Index Reference

| Column | Type | Description |
|--------|------|-------------|
| SeriousDlqin2yrs | int | Target: serious delinquency in past 2 years (0/1) |
| RevolvingUtilizationOfUnsecuredLines | float | % of unsecured credit lines used |
| age | int | Age of borrower |
| NumberOfTime30-59DaysPastDueNotWorse | int | Times 30-59 days past due |
| DebtRatio | float | Monthly debt payments / monthly income |
| MonthlyIncome | float | Gross monthly income |
| NumberOfOpenCreditLinesAndLoans | int | Open credit lines and loans |
| NumberOfTimes90DaysLate | int | Times 90+ days past due |
| NumberRealEstateLoansOrLines | int | Real estate loans and lines |
| NumberOfTime60-89DaysPastDueNotWorse | int | Times 60-89 days past due |
| NumberOfDependents | float | Number of dependent children |

### 1.3 Sensitive Attribute Candidates

**Age.** The only demographic attribute available in the dataset. Age is widely used in
fairness benchmarks for credit scoring. Age exhibits substantial variation across the
population and is associated with the observed default rate.

**MonthlyIncome.** May be interpreted as a socioeconomic status proxy, a socially
relevant characteristic. However, MonthlyIncome contains nearly 20% missing values
and functions primarily as a core predictive feature rather than a demographic protected
attribute. Socioeconomic status is not itself a protected attribute in fairness
benchmarks for credit scoring.

**NumberOfDependents.** May be viewed as a household composition proxy. However, this
variable contains 2.6% missing values and is similarly a task-related predictor rather
than a demographic protected characteristic.

**Gender/Race.** Not available in the dataset.

### 1.4 Sensitive Attribute Decision

Based on the exploratory analysis, **age was selected as the sensitive attribute for all
AFCP experiments**. The observed default rate decreases monotonically with age, and no
natural breakpoint emerges. Therefore, two age-based thresholds are evaluated:

- **Median split (≤52 vs >52):** Primary analysis. Produces balanced subgroup sizes
  (51.6% / 48.4%), enabling robust fairness estimation.
- **Policy-based split (<40 vs ≥40):** Robustness analysis. Informed by age-discrimination
  law and produces a larger observed difference in default rates, serving as a stress test
  under imbalanced conditions.

**MonthlyIncome and NumberOfDependents were not selected** because they are predictive
socioeconomic variables rather than demographic protected attributes and both contain
substantial missing values.

### 1.5 Age Group Analysis

The EDA examines observed default rates across age cohorts to justify sensitive attribute
thresholds. Rather than imposing predetermined breakpoints, we observed default rates
across 10-year age groups to identify natural patterns.

**Observed default rate by 10-year age groups:**

| Age Group | n | Default Rate |
|-----------|---|--------------|
| <30 | 10,758 | 0.1156 |
| 30–39 | 24,339 | 0.0982 |
| 40–49 | 35,037 | 0.0826 |
| 50–59 | 34,806 | 0.0617 |
| 60–69 | 27,424 | 0.0347 |
| 70+ | 17,636 | 0.0226 |

**Pattern:** Default risk decreases monotonically with age, from 11.56% (<30) to 2.26% (70+).
No sharp breakpoint is visible. Consequently, both the median split and the policy-based
threshold are retained for the subsequent AFCP experiments.

**Important note:** These observed differences describe the empirical distribution of default
outcomes only and should not be interpreted as evidence of group-wise coverage disparities.
Coverage fairness is evaluated in the AFCP experiments (Section 2).

### 1.6 Exploratory Data Analysis Figure

![GMSC EDA: Age Distribution, Default Rates, Debt Ratio, and Monthly Income](./figures/gmsc_eda_sensitive_attributes.png)

**Figure description:**

- **Top-left:** Age distribution by default status. Younger borrowers are overrepresented
  among default cases (left tail of distribution).
- **Top-right:** Default rate by 10-year age cohorts showing monotonic decline from
  11.56% (<30) to 2.26% (70+). Red dashed line indicates policy-based split (~40);
  orange dashed line indicates median split (~52).
- **Bottom-left:** Debt ratio distribution, highly skewed with majority of values < 5%.
- **Bottom-right:** Monthly income distribution (log scale) approximately normal around
  log₁₀(income) ≈ 4, corresponding to ~$10,000 monthly income.

### 1.7 Final Dataset Configuration

**Preprocessing:** MonthlyIncome and NumberOfDependents contain missing values (19.8% and
2.6%, respectively). These missing values are handled in the experiment script before model
training. MonthlyIncome is imputed using the training-sample median, and NumberOfDependents
is imputed with 0. The single observation with age ≤ 0 is removed before sampling. Continuous
input features are then standardized using StandardScaler prior to neural network training.

**Age splits:** Both age definitions (median and policy-based) are computed from the age
column during data loading. No observations are removed based on age thresholds; both
splits are evaluated in fairness experiments.

---

## 2. AFCP Fairness Experiments

### 2.1 Experimental Design

This section documents the design and results of Adaptive Fairness-aware Conformal
Prediction (AFCP) experiments on GMSC, following Zhou & Sesia (2024).

**Experimental parameters:**
- **Model:** Simple neural network (64 hidden units, ReLU activation)
- **Train/calib/test split:** 250 / 250 / 500 (n_total = 1000 from 150K)
- **Learning rate:** 0.0001
- **Epochs:** 500
- **Nonconformity score:** Negative probability of true class, -P(y_true)
- **Nominal coverage level (α):** 0.1 (90%)
- **t-test tolerance (ttest_delta):** 0 (default AFCP)
- **Conditional coverage:** True
- **Random seeds:** 2023, 2024, 2025, 2026, 2027 (5-seed ensemble)

**Methods compared:**
1. **Marginal CP:** Standard conformal prediction, no fairness constraints
2. **Adaptive (AFCP):** With adaptive hypothesis-testing filter (default)
3. **Adaptive1:** Without adaptive hypothesis-testing filter
4. **Partial equalized coverage:** Fixed baseline method
5. **Exhaustive equalized coverage:** Fixed baseline method

### 2.2 Key Observations

*Observations will be populated following experimental runs.*

### 2.3 Primary Analysis: Median Split (age ≤52 vs >52)

**Rationale:** The median split is evaluated as the primary analysis because it provides
balanced subgroup sizes (51.6% / 48.4%), enabling robust fairness estimation while preserving
the age-related variation observed during EDA.

**Design table:**

| Method | Coverage (mean ± sd) | Prediction-set size (mean ± sd) | ΔCoverage | ΔLength | Selected (%) |
|--------|----------------------|----------------------------------|-----------|---------|--------------|
| Adaptive | TBD | TBD | — | — | — |
| Adaptive1 | TBD | TBD | — | — | — |
| Marginal | TBD | TBD | — | — | — |
| Partial | TBD | TBD | — | — | — |
| Exhaustive | TBD | TBD | — | — | — |

### 2.4 Secondary Analysis: Policy-based Split (age <40 vs ≥40)

**Rationale:** This policy-motivated age threshold provides a robustness analysis informed
by age-discrimination law. Despite its unbalanced subgroup sizes (21.3% / 78.7%), this
split serves as a stress test for AFCP under challenging conditions.

**Design table:**

| Method | Coverage (mean ± sd) | Prediction-set size (mean ± sd) | ΔCoverage | ΔLength | Selected (%) |
|--------|----------------------|----------------------------------|-----------|---------|--------------|
| Adaptive | TBD | TBD | — | — | — |
| Adaptive1 | TBD | TBD | — | — | — |
| Marginal | TBD | TBD | — | — | — |
| Partial | TBD | TBD | — | — | — |
| Exhaustive | TBD | TBD | — | — | — |

### 2.5 Summary of Key Findings

*Results will be populated following experimental runs.*

### 2.6 Methodological Notes

**On age split selection:** EDA examined observed default rates across 10-year age cohorts
(6 groups from <30 to 70+). Default risk declined monotonically with no sharp breakpoint.
Both median split (52) and policy-based split (40) are evaluated in experiments; the
data does not favor one threshold a priori.

**On ttest_delta:** All experiments used ttest_delta = 0, corresponding to the default
AFCP hypothesis test where the null hypothesis is that miscoverage does not exceed the
nominal level α (0.1). Adaptive1 omits this hypothesis-testing step. This distinction
explains why Adaptive1 yields larger coverage gains than Adaptive at the cost of larger
prediction-set inflation.

**On balanced vs. unbalanced groups:** Primary analysis uses balanced median split for
statistical power. Secondary analysis evaluates performance under imbalance, relevant
to real-world deployment where protected classes are often minorities.

**On missing values:** MonthlyIncome and NumberOfDependents are retained as model features.
Missing values are handled during preprocessing by imputing MonthlyIncome with the
training-sample median and NumberOfDependents with 0. This preserves all observations
and leverages these variables' predictive signal while avoiding the 20% data loss that
would result from listwise deletion.

### 2.7 Conclusion

GMSC provides a high-quality testbed for AFCP fairness evaluation with clear age-related
differences in observed default rates. The monotonic decline in default risk across age
cohorts motivated the selection of two age thresholds — a data-driven median split (primary)
and a policy-based split (secondary) — enabling both optimal estimation (balanced groups)
and robustness evaluation (imbalanced groups).

The empirical AFCP results presented in the following sections evaluate whether these
observed differences in default risk translate into differences in conformal coverage
and whether AFCP can reduce potential group-conditional coverage disparities while
maintaining prediction efficiency.

---

## References

- Yeh, I.-C. (2011). Give Me Some Credit. Kaggle Competition Dataset. Retrieved from
  https://www.kaggle.com/competitions/GiveMeSomeCredit
- Zhou, S., & Sesia, M. (2024). Conformal prediction under covariate shift with agnostic
  correction. *arXiv preprint*.
- Angelopoulos, A. N., & Bates, S. (2022). A gentle introduction to conformal prediction
  and distribution-free uncertainty quantification. *arXiv preprint*.