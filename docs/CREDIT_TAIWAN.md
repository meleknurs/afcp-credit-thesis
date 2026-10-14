# Default of Credit Card Clients (Taiwan) — Dataset Documentation

Source: Yeh, I-C., & Lien, C-H. (2009). *The comparisons of data mining techniques for
the predictive accuracy of probability of default of credit card clients.* Expert Systems
with Applications 36(2), 2473–2480. UCI Machine Learning Repository id 350,
https://doi.org/10.24432/C55S3H, licence CC BY 4.0. File in this repository:
`data/default_credit_card_clients.csv` (converted from the UCI `.xls`; SHA-256 in
`data/default_credit_card_clients.sha256`). Added in v2 (14 Oct 2026); see
`docs/V2_DESIGN.md`, Section 1, for why.

## 1. Exploratory data analysis

### 1.1 Overview

- 30,000 credit-card clients (Taiwan, April–September 2005), 23 features, no missing values.
- Target `default payment next month` (1 = default): **22.12 %** default rate — the most
  balanced of the three thesis datasets (SGC 30 %, GMSC 6.7 %).
- Features: credit limit (`LIMIT_BAL`), four demographic fields (`SEX`, `EDUCATION`,
  `MARRIAGE`, `AGE`), six monthly repayment-status codes (`PAY_0`, `PAY_2`…`PAY_6`), six
  monthly bill amounts (`BILL_AMT1–6`) and six monthly payment amounts (`PAY_AMT1–6`).

### 1.2 Undocumented codes and recoding

| field | documented codes (UCI) | undocumented codes found | treatment |
|---|---|---|---|
| `EDUCATION` | 1 graduate, 2 university, 3 high school, 4 others | 0 (14), 5 (280), 6 (51) | folded into 4 "others" (common recode; Le Quy et al. 2022 report the same 468-row "others" group) |
| `MARRIAGE` | 1 married, 2 single, 3 others | 0 (54) | folded into 3 "others" (377 rows) |
| `PAY_0…PAY_6` | −1 pay duly, 1–9 months of delay | −2 (2,759 in `PAY_0`), 0 (14,737) | kept as numeric; −2 = no consumption, 0 = revolving credit (Yeh & Lien replication notes) |

All 30,000 rows are kept (some fairness papers drop rows without stating which:
Kozodoi et al. 2022 use 23,531, Hurlin et al. 2022 use 25,983).

### 1.3 Sensitive attribute candidates

| attribute | groups (n) | default rate | note |
|---|---|---|---|
| **sex** (primary) | male 11,888 / female 18,112 | 0.242 / 0.208 | convention in the fairness literature (Le Quy et al. 2022) |
| marriage | married 13,659 / single 15,964 / others 377 | 0.235 / 0.209 / 0.236 | "others" too small for group-conditional calibration |
| education | graduate 10,585 / university 14,030 / high school 4,917 / others 468 | 0.192 / 0.237 / 0.252 / 0.071 | monotone in education level; "others" small and atypical |
| age < 25 | 2,685 (9 %) | 0.272 vs 0.216 | Kozodoi et al. (2022) convention |
| age 3-way (≤35 / 36–55 / >55) | 16,809 / 12,347 / 844 | 0.214 / 0.228 / 0.265 | same cut as GMSC for cross-dataset comparison; >55 small (2.8 %) |

Default rate by age decade: 20–29 0.228, 30–39 0.203, 40–49 0.230, 50–59 0.249, 60+ 0.283
(U-shaped, unlike GMSC's monotone decline). Intersection sex × age: women aged >55 have the
highest default rate (0.296, n = 405), men aged 36–55 the second highest (0.255).

### 1.4 Decisions

- Primary sensitive attribute: **sex**. Candidate set for AFCP's multi-attribute selection:
  sex, marriage, education, age < 25 (and age 3-way in a secondary configuration).
- For the AFCP candidate set the two small "others" levels are merged into the nearest
  level (marriage others → single; education others → high school) so that no candidate
  group has fewer than ≈50 (group × default) calibration points; the unmerged levels are
  reported in the EDA only.
- Class imbalance is left as is (no resampling), as in the fairness benchmarks.
- Cell counts for the planned split sizes (n_train = n_calib = 6,000, n_test = 10,000):
  `thesis_experiments/results/v2/cell_counts_taiwan_credit.csv`. Smallest primary cell
  (male × default) ≈ 575 calibration / 958 test points → coverage SE ≈ 1 pp.

### 1.5 Figure

![Taiwan EDA: age distribution and default rates by sex, marriage, education](./figures/taiwan_eda_sensitive_attributes.png)

Produced by `thesis_experiments/v2/eda_taiwan.py`; tables in
`thesis_experiments/results/v2/eda_taiwan_tables.md`.

## 2. Experiments

*(v2 — to be filled as the baseline, calibration-sweep and bias-injection runs complete.)*

## References

- Yeh, I-C., & Lien, C-H. (2009). Expert Systems with Applications, 36(2), 2473–2480.
- Le Quy, T., Roy, A., Iosifidis, V., Zhang, W., & Ntoutsi, E. (2022). A survey on datasets
  for fairness-aware machine learning. WIREs Data Mining and Knowledge Discovery, 12(3).
- Kozodoi, N., Jacob, J., & Lessmann, S. (2022). Fairness in credit scoring: Assessment,
  implementation and profit implications. European Journal of Operational Research, 297(3).
