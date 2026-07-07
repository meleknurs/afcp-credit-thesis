# South German Credit — Dataset Analysis & Experiments

This document records all thesis-specific decisions and results for the
South German Credit dataset. See `CREDIT_ANALYSIS.md` for the cross-dataset
overview and `SETUP_LOG.md` for the replication of the original AFCP
implementation.

**Dataset:** South German Credit (Grömping 2019), UCI ML Repository id=522.
Corrected version of the widely used Statlog German Credit dataset.
Reference: Grömping, U. (2019). *South German Credit Data: Correcting a
Widely Used Data Set.* Reports in Mathematics, Physics and Chemistry,
Report 4/2019, Beuth University of Applied Sciences Berlin.

**Source file:** `SouthGermanCredit.asc` (space-separated, German column
names; English names mapped according to Grömping's `read_SouthGermanCredit.R`
and the accompanying code table).

---

## 1. Exploratory data analysis

### 1.1 Overview

- 1,000 observations, 20 features + 1 target, no missing values.
- Target `kredit` / `credit_risk`: 0 = bad (n=300), 1 = good (n=700).
  No transformation is required; the coding is used as-is
  (`num_classes=2`, ClassNNet).
- **Sampling design (Grömping 2019):** the data are a stratified sample of
  1,000 credits (1973–1975, a large regional bank in southern Germany) in
  which bad credits are heavily oversampled: the sources report an actual
  bad-credit prevalence of about 5%, versus 30% in the sample. The 30/70
  split is therefore a design artifact, not the population distribution.
  This is directly relevant to the thesis discussion of label/selection
  bias in credit data.
- Numeric features: `duration` (median 18 months), `amount` (median 2,320,
  right-skewed, max 18,424; per Grömping 2019 the values are the result of
  an unknown monotonic transformation, not actual DM amounts), `age`
  (median 33, range 19–75).
- All remaining features are ordinal/categorical integer codes
  (2–10 levels; see code table). Per Grömping's `read_SouthGermanCredit.R`,
  these level codes are not arbitrary: they are the P2 scores from Häußler
  (1979), constructed so that larger codes indicate better creditworthiness.
  Treating them as ordinal integer inputs to the classifier is therefore a
  defensible encoding choice.
- **Coding reversal warning:** in the corrected data, the codes of
  `people_liable` (`pers`) and `foreign_worker` (`gastarb`) are the
  *opposite* of those in Open Data LMU (2010) and the original UCI German
  Credit release. Numerical comparisons with older fairness studies that
  use these variables must account for this.

### 1.2 Column index reference (0-based, `.asc` order)

| Index | Column | Index | Column |
|---|---|---|---|
| 0 | status | 11 | property |
| 1 | duration | 12 | **age** |
| 2 | credit_history | 13 | other_installment_plans |
| 3 | purpose | 14 | housing |
| 4 | amount | 15 | number_credits |
| 5 | savings | 16 | job |
| 6 | employment_duration | 17 | people_liable |
| 7 | installment_rate | 18 | telephone |
| 8 | **personal_status_sex** | 19 | **foreign_worker** |
| 9 | other_debtors | 20 | credit_risk (target) |
| 10 | present_residence | | |

These indices are hard-coded explicitly in `mc_south_german.py` to avoid
accidental index mismatches. Runtime assertions additionally verify that the
expected columns appear at the intended positions, providing a safeguard if
the column layout changes (e.g., due to encoding steps or dataset revisions).

### 1.3 Sensitive attribute candidates

**age (idx 12).**
Younger applicants have a markedly lower good-credit rate:

| Grouping | Group | n | good-credit rate |
|---|---|---|---|
| ≤25 vs >25 | ≤ 25 | 190 | 0.579 |
| | > 25 | 810 | 0.728 |
| median split | ≤ 33 | 514 | 0.648 |
| | > 33 | 486 | 0.755 |

Quartile analysis shows a broadly monotone increase in good-credit rate with
age (0.630 → 0.671 → 0.767 → 0.742).

**personal_status_sex (idx 8).**
Codes: 1 = male divorced/separated (n=50, rate 0.600),
2 = female non-single OR male single (n=310, 0.648),
3 = male married/widowed (n=548, 0.734),
4 = female single (n=92, 0.728).

Sex cannot be recovered from this variable: code 2 conflates
"female non-single" and "male single", and female widows are not listed in
any of the female categories of the corrected code table (Grömping 2019).
The variable is therefore not used as a sex attribute.

**foreign_worker (idx 19).**
1 = yes (n=37, good rate 0.892), 2 = no (n=963, 0.693). The corrected coding
resolves an implausibility of the original Statlog release, in which more
than 90% of debtors appeared to be foreign workers (Grömping 2019); in the
corrected data only 3.7% are. This makes the minority group far too small
for stable calibration-based subgroup analysis (~18 calibration observations
under a 50/50 split, fewer still under label-conditional calibration).

### 1.4 Sensitive attribute decisions

1. **Primary attribute: age (binned).** Age is the most widely used fairness
   attribute for German Credit in the literature (e.g., Kamiran & Calders
   2009, who use the ≤25 cut; Calmon et al. 2017), the group difference is
   substantial, and the variable is cleanly coded. The binning method
   is examined with both a fixed cut (≤25 vs. >25) and a median split, as a
   sensitivity analysis. Age bins are defined per dataset (no common cut with GMSC is
   possible: median age 33 here vs. 51 in GMSC).

2. **Secondary analysis: personal_status_sex (4 levels).** Included not as a
   sex attribute but as a *personal status* attribute, to examine whether
   AFCP detects coverage disparities across marital/status groups. The
   smallest group (code 1, n=50) yields ~25 calibration observations,
   which connects this analysis to the thesis's sample-size sensitivity
   findings: AFCP's t-test is not expected to reach significance for groups
   of this size.

3. **Excluded: foreign_worker.** Deliberately excluded on methodological
   grounds: the minority group (n=37) is far below any plausible threshold
   for AFCP's calibration-based t-test. Documented here for completeness.

Since AFCP selects at most one sensitive attribute during each calibration
procedure, prioritizing a single well-justified primary attribute is
consistent with both the algorithmic design and the experimental objectives
of this thesis (Zhou & Sesia 2024).

### 1.5 EDA figure

![South German Credit sensitive attributes EDA](../docs/figures/sgc_eda_sensitive_attributes.png)

*Figure 1: Age distributions by credit risk (upper left); good-credit rates
by personal status (upper right), foreign worker status (lower left), and
age quartile (lower right). Sample sizes (n) labeled on each bar.*

### 1.6 Final dataset configuration

The experiments use the following configuration:

| Component | Choice |
|---|---|
| Features | All 20 predictor variables |
| Target | `credit_risk` (binary: 0=bad, 1=good) |
| Primary sensitive attribute | `age` (binned: ≤25 vs. >25, or median split) |
| Secondary analysis | `personal_status_sex` (4 levels, personal status) |
| Excluded | `foreign_worker` (n=37 minority group) |

---

## 2. Experiments

*(to be added: run configurations, per-run results, commit references)*