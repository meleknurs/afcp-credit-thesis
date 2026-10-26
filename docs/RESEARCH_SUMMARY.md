# Research Summary

This document summarises the experimental work of the thesis in chronological
order, from the pilot study to the v2 experiments. Each step is described in
four parts: **what was done**, **why**, **results** and **interpretation**.
All figures are taken from `docs/CREDIT_*.md` and from the summary files in
`thesis_experiments/results/v2/`; the design rationale is given in
`docs/V2_DESIGN.md`.

## Overview

Standard conformal prediction (CP) guarantees 90 % coverage **on average**,
but coverage for a particular group (e.g. women) can be considerably lower.
Adaptively Fair Conformal Prediction (AFCP; Zhou & Sesia, 2024) aims to detect
such a group automatically and to equalise its coverage. This thesis tests,
on credit-scoring data with **controlled, semi-synthetic label bias**, when
AFCP detects the affected group and why coverage-equalisation methods
sometimes fail to restore coverage.

---

## Step 0 — Pilot study (July 2026)

**What.** Two datasets (South German Credit, Give Me Some Credit), five
methods (Marginal, Partial, Exhaustive, AFCP-Adaptive, Adaptive1), five
seeds. The pilot comprised five experiments:

1. SGC, age ≤ 25 vs > 25.
2. SGC, age median split (≤ 33). Changing the age threshold changed the
   results (group-definition sensitivity).
3. GMSC, three age groups (≤ 35 / 36–55 / > 55).
4. GMSC sample-size sweep (training and calibration sets enlarged jointly,
   2,000 → 8,000): in two of five seeds (2023, 2027) AFCP selected age from
   4,000 onwards; in the other three it never did. Because training and
   calibration sizes increased together, the two effects are confounded.
5. GMSC multi-attribute analysis (age + income + dependents): the final
   selection was always empty. Age was selected for label 0 and income for
   label 1, and the implementation takes the **intersection** across labels
   (label veto).

**Why.** To apply Zhou & Sesia's method to credit data.

**Results.** Overall coverage ≈ 90 %; worst-group coverage slightly lower
(GMSC 0.875). AFCP selected no attribute in most runs.

**Interpretation.** The pilot does not show that AFCP fails; rather, three
factors prevented a meaningful test:

1. The neural network, trained for 500 epochs, overfitted: predicted
   probabilities were close to 0 or 1, nonconformity scores tied, and the
   selection t-test became uninformative.
2. The implementation used label-conditional calibration
   (`conditional=True`) by default. Combined with the rare default class
   (6.7 %), this produced a mean set size of 1.88 and triggered the label
   veto.
3. There was **no disparity to detect**: on clean data, group differences
   were 1–2 percentage points.

## Step 1 — v2 design

**What.** Three datasets: SGC (small sample), GMSC (rare default class) and
the **Taiwan** credit-card data (main test bed: 30,000 clients, 22 % default,
four demographic attributes). The neural network was retrained with early
stopping and dropout, and logistic regression was added as a second
classifier. Both calibration regimes (marginal and label-conditional) are
reported explicitly. Ten seeds (2023–2032). Split sizes were derived from the
smallest (group × label) cell.

**Why.** To address each of the three pilot limitations.

**Results.** On Taiwan, AUROC ≈ 0.76 and mean set size ≈ 1.34 under the
marginal regime, i.e. the prediction sets are informative.

**Interpretation.** Results now depend on the data rather than on silent
defaults of the implementation.

## Step 2 — Baseline on clean data

**What.** All four methods on the Taiwan data without injected bias.

**Results.** Coverage: Marginal 0.902, Partial 0.908, Exhaustive 0.906,
AFCP 0.906. Women 0.900, men 0.904: **no disparity by sex**, and AFCP never
selects sex (0.00). GMSC: age ≤ 35 0.894, 36–55 0.903, > 55 0.907. SGC:
0.89 vs 0.90, set size 1.45–1.69; AFCP selects age in 3 of 10 seeds.

**Interpretation.** The control condition is clean: AFCP raises no false
alarm, so changes in the following steps can be attributed to the injected
bias.

## Step 3 — Label-bias injection (two directions)

**What.** A fraction ε of women's labels was corrupted in the training and
calibration data; the test set was always kept clean.

- **Stigma** (non-default → default): the model over-predicts default for
  women.
- **Hidden defaults** (default → non-default): the model misses defaults
  among women.

Zhou & Sesia introduce bias in synthetic data by randomising a group's
labels; here the bias is introduced into real data as a dose-controlled
historical labelling error.

**Results (Taiwan, marginal regime).**

- Hidden defaults: women's coverage falls from 0.900 to 0.847 (ε = 0.5) and
  to **0.801 (ε = 0.7)**, while men's coverage stays at ≈ 0.89–0.90; the gap
  reaches 0.12.
- Stigma: women's coverage **rises** (0.900 → 0.951 at ε = 0.3), but set size
  grows from 1.31 to 1.74. At high doses (ε = 0.4–0.5) coverage falls again
  (non-monotone).

**Interpretation.** The two bias directions manifest differently: hidden
defaults reduce coverage, whereas stigma reduces informativeness.

## Step 4 — Do the equalisation methods restore coverage?

**Results.** When the calibration set is also corrupted, Partial,
Exhaustive and AFCP **do not restore** women's coverage (0.79–0.81 at
ε = 0.7). When the calibration set is clean (training-only injection, hidden
defaults, ε = 0.5), Partial and Exhaustive reach 0.904 and 0.905 (compared
with 0.85 and 0.84 when both training and calibration data are corrupted).

**Interpretation.** Equalised coverage is equalised with respect to the
**calibration labels**, not the true labels. This is the central
methodological finding of the thesis.

## Step 5 — Label-conditional regime

**Results.** Hidden defaults, ε = 0.7: women's coverage is 0.86 (0.80 under
the marginal regime); Partial and Exhaustive restore it to ≈ 0.90 (the
remaining defaults are genuine). Mean set size ≈ 1.7. AFCP's selection rate
is low (≤ 0.2), and the per-label flags differ (0.20 for label 0 vs 0.40 for
label 1), so the **label veto is the limiting factor** here.

**Interpretation.** The label-conditional regime yields fairer coverage but
less informative sets (coverage–informativeness trade-off). The veto
mechanism observed in the pilot is reproduced here under controlled
conditions.

## Step 6 — Can AFCP detect the bias?

**Results (Taiwan, marginal regime).** AFCP's rate of selecting sex increases
with ε: from 0 to 0.6 for hidden defaults and from 0 to 0.4 for stigma.
Selection is seed-dependent (all-or-nothing per seed). At ε = 0 the rate is
0.00, i.e. there are no false alarms.

The finding **does not replicate on GMSC**: the selection rate is 0.20 at
every dose, always for the same two seeds (2023, 2027). The hypothesis that
calibration contamination suppresses selection is **not supported** (the
rate is 0.10 under training-only injection as well).

**Interpretation.** AFCP's detection power depends on the dataset; with a
rare default class and near-singleton sets, the selection test is weak. The
sensitivity to `ttest_delta` and `sig_level` remains to be examined (see
Open items).

## Step 7 — Adaptive1 (selection without the hypothesis test)

**Results.** Taiwan: sex selection rate 0.1 / 0.6 / 0.7 at ε = 0 / 0.3 / 0.7,
compared with 0 / 0.4 / 0.6 for Adaptive. The rate of selecting no attribute
is ≈ 0. Coverage and set size are unchanged.

**Interpretation.** Without the hypothesis test, AFCP reduces to a "select
the worst group" rule rather than a detector, and it produces false alarms at
ε = 0 (0.10).

## Step 8 — Robustness to the classifier (logistic regression)

**Results.** At ε = 0.2 the neural network and logistic regression agree to
within ≈ 1 percentage point, so the findings are not specific to the neural
network. Stigma at ε = 0.5 (logistic regression): women's coverage 0.82,
i.e. under-coverage at high doses.

## Step 9 — Sample-size sweep (pilot experiment repeated in v2)

All pilot experiments were repeated under the v2 design; which of them are
included in the thesis is decided afterwards.

**What.** Two designs: (a) **joint**, training and calibration sets enlarged
together to 2,000 / 4,000 / 6,000 / 8,000 (as in the pilot, confounded);
(b) **calibration-only**, training set fixed at 6,000 and calibration set
1,000 / 2,000 / 4,000 / 6,000. Both calibration regimes, v2 neural network.
GMSC (no bias, 5 seeds, test set 1,000) and Taiwan (10 seeds, test set 500;
no bias and hidden defaults at ε = 0.5).

**Why.** In the pilot, AFCP began selecting age as the sample grew, but
because the training set grew as well, the cause could not be identified.

**Results — GMSC (AFCP age-selection rate, per seed).**

- `conditional=True` (the pilot's regime): seeds **2023 and 2027 select at
  every size; seeds 2024–2026 never select** (joint 2,000 to 8,000 and
  calibration-only 2,000–4,000 alike). Mean selection rate 0.4.
- The only exception: with a calibration set of 1,000 (training 6,000), seed
  2023 does not select; it selects from 2,000 onwards.
- `conditional=False`: seed 2027 always selects; seed 2023 only at
  6,000/6,000 and 8,000/8,000; the others never.
- In the pilot, the same two seeds (2023, 2027) were the ones that selected.

**Results — Taiwan.**

- Without bias, AFCP rarely selects sex (0–0.2 at every size), so there is no
  increase in false alarms with sample size.
- Hidden defaults, ε = 0.5 (marginal regime): selection rate 0.4–0.5 with
  **no clear trend in size** (calibration 1,000: 0.50, 2,000: 0.40, 4,000:
  0.50; joint 2,000: 0.27, 4,000: 0.50, 8,000: 0.40). Women's coverage is
  ≈ 0.83–0.85 at every size, men's 0.87–0.90.
- Label-conditional regime: selection rate 0.10–0.20, flat.

**Interpretation.** The pilot's observation that AFCP selects more often as
the sample grows is not supported: selection depends mainly on **which seed,
i.e. which data sample, is drawn**, and additional calibration data does not
change it materially (at most at the smallest size). The confounding in the
pilot sweep therefore did not drive its result, and the calibration-only
design now allows this to be stated. On Taiwan the effect is already large
(women ≈ 0.85), and more data does not restore coverage because the
calibration labels are corrupted.

Files: `thesis_experiments/results/v2/gmsc_sweep_summary.csv`,
`thesis_experiments/results/v2/taiwan_sweep_summary.csv`.

## Step 10 — GMSC multi-attribute analysis (age + income + dependents)

**What.** The pilot's multi-attribute experiment repeated on GMSC with the
v2 setup: three candidate attributes (age in three groups, income high/low,
dependents yes/no) instead of one; Marginal, Partial, Exhaustive, AFCP and
Adaptive1; five seeds.

**Why.** In the pilot, AFCP always returned an empty selection with multiple
attributes; the aim is to determine whether this is a property of the method
or an artefact of the overfitted neural network.

**Results.** The pilot's "always empty" behaviour **does not replicate**. The
veto mechanism still exists (an attribute can be eliminated when only one
label flags it) but it is not systematic.

**Interpretation.** The empty selections in the pilot were most likely caused
by the overfitted network combined with the silent label-conditional default,
rather than by the method itself.

Files: `thesis_experiments/results/v2/afcp/gmsc_multi/` (10 files).

## Step 11 — Group definition (RQ4): do results depend on how groups are defined?

**What.** On clean data (**no bias injected**), the same attribute was
defined with different cut-offs and all methods were run for each
definition: SGC age ≤ 25 vs ≤ 33; GMSC age < 25 vs ≥ 60; Taiwan age < 25 vs
age 3-way (≤ 35 / 36–55 / > 55). 120 runs in total (SGC 10 seeds × NN/LR × both regimes,
GMSC 5 seeds, Taiwan 10 seeds).

**Why.** The protected group is not given a priori in the data; it is
necessary to know whether the worst group and the ranking of methods change
with the cut-off.

**Results (worst-group coverage and gap; target 0.90).**

- **SGC (NN, marginal regime):** ≤ 25: smallest test group ≈ 57; Marginal
  worst 0.871, gap 0.042; Partial 0.882; AFCP 0.873; Adaptive1 0.879.
  ≤ 33: ≈ 142; Marginal 0.879, gap 0.035; Partial 0.880; AFCP 0.878;
  Adaptive1 0.887. Differences between methods are ≤ ≈ 0.015, i.e. within
  noise. Logistic regression gives the same picture.
- **GMSC:** age < 25: on average only **≈ 8 test observations** (out of
  500). Marginal worst 0.803, gap 0.137; Partial/Exhaustive 0.832, gap 0.108.
  Age ≥ 60: ≈ 160 observations; Marginal worst 0.893, gap 0.017. A large
  disparity appears only under the very small group definition.
- **Taiwan:** age < 25: ≈ 45 observations; Marginal worst 0.883, gap 0.029.
  Age 3-way: smallest group (> 55) ≈ 14 observations; Marginal 0.883, gap
  0.060.
- **AFCP's selection rate** is almost independent of the definition within a
  dataset, model and regime (SGC 0.3, GMSC 0.2/0.4, Taiwan 0.5): selection is
  driven by the seed rather than the definition (one exception: SGC, LR,
  `conditional=True`, ≤ 33: 0.195).

**Interpretation.**

1. The group definition strongly changes the worst-group figure, but mainly
   **because the group becomes smaller**: with a test set of ≈ 500, the
   standard error of coverage in a group of 8–14 observations is very large.
2. The equalisation methods improve on Marginal by ≈ 1–3 percentage points
   in the smallest groups (GMSC < 25: 0.803 → 0.832), but with n ≈ 8 this is
   not reliable and is reported as such.
3. With a test set of 500, the standard error of group coverage is ≈ 2–4
   percentage points; the differences between methods are below this level.

Table: `thesis_experiments/results/v2/rq4_groupdef_summary.csv`; script:
`thesis_experiments/v2/analysis/group_def_table.py`.

## Step 12 — Small samples with 100 seeds: estimating the detection rate precisely

**What.** All selection rates above rest on 5–10 seeds, which resolves rates
only in steps of 0.1–0.2. The Taiwan baseline and the hidden-defaults
configuration (ε = 0.5, training + calibration) were therefore rerun with
**100 seeds** at small sample sizes in the style of Zhou & Sesia's
experiments, n_train = n_calib ∈ {100, 250, 500, 1,000}, neural network,
marginal regime, test set 500: 800 runs in total.

**Why.** To state AFCP's false-alarm rate and detection rate with confidence
intervals rather than as single digits, and to see whether detection improves
with sample size once the estimate is precise enough.

**Results.**

- Clean data: AFCP's selection rate for sex is 0.03–0.07 at every size (95 %
  bootstrap interval at n = 1,000: 0.02–0.11). The false-alarm rate does not
  grow with n.
- Hidden defaults: the selection rate rises monotonically with n,
  0.07 → 0.15 → 0.24 → 0.31 (intervals at n = 100 and n = 1,000:
  0.02–0.12 and 0.22–0.40, which do not overlap). Per seed the rate is
  almost always exactly 0 or 1, so it can be read as the share of seeds in
  which sex is selected. The per-label flags are
  equal for both labels, so the label veto is not the limiting factor here;
  the t-test itself has little power with ≤ 1,000 calibration points.
- At n = 100 the sets are wide (1.68 labels) and women are covered at 0.90
  even under bias; as n grows the sets tighten (1.27) and the bias appears as
  under-coverage (women 0.857 vs 0.905 on clean data).
- Partial and Exhaustive do not restore women's coverage (0.873 at
  n = 1,000), confirming Step 4 at small sample sizes with 100 seeds.

**Interpretation.** With enough seeds, AFCP's detection rate does increase
with sample size, and its false-alarm rate stays low. The pilot's reading
(Step 0) was therefore directionally right but could not be established with
5 seeds, and the 10-seed sweeps of Step 9 could not resolve it either. What
remains unchanged is the central finding: detection does not translate into
corrected coverage when the calibration labels are themselves biased.

Files: `thesis_experiments/results/v2/taiwan_small_sample_summary.csv`,
`docs/figures/taiwan_small_sample_selection.png`; details in
`docs/CREDIT_TAIWAN.md`, Section 2.9.

## Research questions

- **RQ1.** How does label bias manifest in coverage and prediction-set size?
- **RQ2.** Do equalised-coverage methods correct it? (Only when the
  calibration labels are clean.)
- **RQ3.** When does AFCP detect the bias, and when does it remain silent?
  (Calibration regime, label veto, dataset.)
- **RQ4.** Are the results sensitive to the definition of the groups?

## Open items

- Sensitivity of AFCP's selection to `ttest_delta` and `sig_level`.
- Selection rates based on 5–10 seeds (Steps 2–11) are reported with
  caution; Step 12 shows what 100 seeds resolve at small sample sizes, and
  the same design at n = 6,000 would complete the picture.
- The multi-attribute runs use the main AFCP procedure (Algorithm 1: one
  attribute per placeholder label, then the intersection over labels). The
  two-attribute extension of Zhou & Sesia (Appendix A2), which lets each
  label nominate up to two attributes before the intersection, is not yet
  implemented; it is planned for the GMSC and Taiwan multi-attribute
  configurations.

## References

Zhou, Y., & Sesia, M. (2024). Conformal classification with equalized
coverage for adaptively selected groups. *Advances in Neural Information
Processing Systems (NeurIPS)*. arXiv:2405.15106.
