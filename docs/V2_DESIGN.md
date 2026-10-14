# Thesis experiments v2 — experimental design

Status: v2 agreed 14 Oct 2026 (open items decided, see below), based on the
literature review in `docs/literature/experiment_design_review.md`.

## 0. Why a v2

Phase 0 (Mar–Jul, `archive/phase0_early_trials/`) and Phase 1 (Jul,
`thesis_experiments/`, BIAS 2026 paper) share one pipeline inherited from the
AFCP replication. Diagnostics on 13–14 Oct 2026 (`bias_meter.py`) and the
literature review show that inherited choices, not the data, drive the
headline results:

| inherited choice | effect | source |
|---|---|---|
| label-conditional calibration (`conditional=True`) — the silent default of the AFCP code; the paper's main text presents marginal sets (App. A3 covers the conditional variant) | with a 6.7 % positive class nearly every set contains both labels → mean set size 1.88; RQ3 unanswerable. Marginal calibration gives ~90 % singletons but under-covers the rare class (30 % average minority coverage on tabular benchmarks) | AFCP code; Singh et al. 2026; Tursunbadalov 2026; Ding et al. 2023 |
| 500 epochs, no dropout, no early stopping, no checkpoint | p̂(default) ≈ 0 for most true defaulters → nonconformity scores tie at the maximum → AFCP's label-1 selection test has constant indicators (p = 1) → label intersection always empty → `selected_none` | AFCP code; meter runs 14 Oct (50 vs 500 epochs) |
| n_test = 500 (AFCP code default) | smallest age group ≈ 67 test points → coverage SE ≈ 3.7 pp; group gaps of 3–10 pp unresolvable | binomial arithmetic; Cresswell et al. 2025 flag the same problem at 91 points |
| GMSC sentinels 96/98 (269 rows) scaled as numeric; `german.data-numeric` → South German Credit switch undocumented | data-quality items no fairness paper documents either | Grömping 2019; computed from `cs-training.csv` |

Phase 1 results stay as reported (git tag `v1-paper`); the thesis presents
them as the replication phase and v2 as the redesign. The guiding principle
from the review: **every guarantee is a per-(group × label)-cell guarantee;
design from the smallest cell outward.**

## 1. Datasets, roles, group definitions

| dataset | n | positive (bad) rate | primary attribute (convention) | further candidate attributes | smallest (group × bad) cell | role |
|---|---|---|---|---|---|---|
| South German Credit (UCI 522, CC BY 4.0) | 1,000 | 30 % | age ≤ 25 vs > 25 (AIF360 / Kamiran & Calders 2009; 190 / 810) | `famges` (personal status; level 2 is mixed-sex, report as such) | ≈ 8 % → ~24 points per test split | small-n regime: baseline only, many splits |
| Give Me Some Credit (Kaggle) | 150,000 | 6.7 % | age ≤ 35 / 36–55 / > 55 (Phase 1); age < 25 (Kozodoi et al. 2022) as comparability row only | age ≥ 60 as sensitivity (non-standard) | ≈ 0.15 % (age < 25) / ≈ 1 % (≥ 60) | rare-class regime: baseline + marginal-vs-conditional comparison; large calibration/test only |
| Default of Credit Card Clients, Taiwan (UCI 350, CC BY 4.0) | 30,000 | 22 % | sex (Le Quy et al. 2022: 24.2 % vs 20.8 % default) | marriage, education, age < 25 | ≈ 9.6 % (male × default) | **main test bed**: multi-attribute selection, bias injection, sensitivity |

Decisions
- Taiwan added: balanced enough to avoid the rare-class degeneracy, four
  demographic attributes so AFCP's selection step does real work, cell sizes
  that resolve 3–5 pp gaps in one split, free licence.
- Foreign worker (SGC): tested in Phase 0 (52 test points, no detectable
  disparity); reported as tested-and-dropped.
- GMSC age group (decided): Phase-1 three-level split stays primary for
  continuity; the literature's < 25 cut (cell ≈ 0.12 % after cleaning) is
  reported as a comparability row with its cell size stated.
- Taiwan: the 'others' levels of marriage (377 rows) and education (468 rows,
  7 % bad) are too small to serve as AFCP candidate groups; they are merged
  into the nearest level for the candidate-attribute set (marriage others →
  single; education others → high school) and reported separately in the EDA.
- A per-dataset **cell-count table** (train / calib / test counts for every
  group × label cell) is produced by the data script and goes in the thesis.

## 2. Data preparation (documented per dataset in `docs/CREDIT_*.md`)

- SGC: Grömping (2019) corrected coding; response recoded so "bad" = 1 is the
  positive class; no other changes. One paragraph in the thesis on why
  `german.data-numeric` (Phase 0) was replaced.
- GMSC: drop the 269 rows with sentinel 96/98 in the delinquency counters
  (all three counters carry it simultaneously) — decided: drop;
  drop age = 0; de-duplicate (609 rows); MonthlyIncome and NumberOfDependents
  → training median plus a missing indicator; log1p-transform
  RevolvingUtilization, DebtRatio, MonthlyIncome. State that no fairness
  paper documents these steps.
- Taiwan: EDUCATION 0/5/6 → 4 ("others"), MARRIAGE 0 → 3; PAY_0…PAY_6 kept as
  numeric with −2 / −1 / 0 retained as distinct values (documented meanings:
  no consumption / paid in full / revolving); all 30,000 rows used.
- No resampling or reweighting for class imbalance (fairness benchmarks do
  not).
- Group labels built on the full data before sampling; sensitive attributes
  passed to the model as features (AFCP design); both stated explicitly.
- Data files: the two UCI sets committed with CC BY attribution; GMSC as a
  download script + row count + hash.

## 3. Classifiers and training protocol

- Primary: AFCP's `ClassNNet` (256-256-128-64-2, ReLU), Adam lr 1e-4, batch
  25, but with **a validation slice carved from the training half**, early
  stopping on validation log-loss (patience 20, as in Lu et al. 2022), dropout
  on, best-epoch checkpoint. The calibration set is never used for model
  selection. Chosen epoch logged per run.
- Secondary (same CP pipeline): L2 logistic regression; shallow gradient
  boosting (≤ 100 trees, depth 3). Rationale: 2026 tabular benchmarks show
  the imbalance / calibration findings are architecture-independent, and LR
  is the credit-industry reference.
- **Degeneracy check logged for every run** (calibration set): AUROC and
  log-loss (sanity ranges: SGC 0.78–0.81, GMSC ≈ 0.86 non-linear / 0.69 raw LR,
  Taiwan 0.75–0.77), share of points with p̂(y_true) < 1e-6, share of tied
  nonconformity scores per (group × label) cell, variance of the LOO
  miscoverage indicator per cell (0 ⇒ AFCP test undefined).
- Nonconformity score: randomised APS/ARC as in the AFCP code; randomisation
  stated.

## 4. Conformal setup and sizes

- α = 0.10 throughout.
- Calibration regime is an explicit experimental factor: **marginal**
  (`conditional=False`) and **label-conditional** (`conditional=True`), every
  method under both. Report per-class and per-(group × class) coverage, set
  size, singleton rate, {0,1}-set rate for each.
- Methods: Marginal, Partial, Exhaustive, AFCP (Adaptive). Adaptive1 kept as
  the "selection-power ceiling" arm in one diagnostic table only.
- Split-size rule: **≥ 50 calibration and ≥ 100 test points in the smallest
  (group × label) cell for a label-conditional run** (Ding et al. 2023:
  usable regime 20–40 per class; Vadlamani et al. 2025 over-coverage bound
  1/(|cell|+1)). Derived sizes:
  - Taiwan: n_train = n_calib = 6,000 for the main table; test ≈ 10,000;
    calibration sweep {500, 1,000, 2,000, 5,000, 10,000} with test fixed.
  - GMSC: n_train = n_calib ≥ 10,000, test ≥ 30,000 (meter-based analyses);
    full-AFCP runs at n_test = 2,000 with the resulting SE stated.
  - SGC: 350 / 350 / 300, ≥ 100 random splits, per-split gaps paired across
    splits; thesis states that only ≥ 10 pp gaps are resolvable.
- Cost note: full AFCP ≈ 2 s per test point per adaptive method. Large test
  sets are used with the fast meter (marginal CP + analytic reproduction of
  the AFCP selection test); full AFCP runs use n_test = 1,000–2,000 and ≥ 10
  seeds, parallelised on the laptop; GitHub Actions for the Taiwan grid if
  the laptop budget is exceeded (decided).
- Seeds: 10 (2023–2032) for all full runs; 100 splits for SGC via the meter.
- AFCP settings: `ttest_delta` ∈ {0, 0.02, 0.05} and t-test level ∈ {0.05,
  0.10} as a sensitivity sweep on Taiwan (code default 0.10; paper says 5 %).
  Report per-label flag frequencies; an "any-label" selection ablation is run
  and labelled as a deviation from Zhou & Sesia.

## 5. Bias injection (Taiwan primary; GMSC secondary)

Mechanisms, following the review's effectiveness ranking:

| arm | mechanism | injected into | grid | seeds | status |
|---|---|---|---|---|---|
| **label bias (primary)** | one-sided: fraction f of the disadvantaged group's *good* labels flipped to *bad* (Ceccon et al. 2025; Bias on Demand measurement bias on Y) | (a) train only; (b) train + calibration (Einbinder et al. 2022 protocol); test always clean | f ∈ {0, 0.05, 0.10, 0.20, 0.30, 0.40} | 10 | to implement (replaces the symmetric Bernoulli(0.5) noise of the 13 Oct pilot, which the literature and our pilot show only inflates sets) |
| representation bias | retain share r of the disadvantaged group in training | train only | r ∈ {1, 0.5, 0.2, 0.1} | ≥ 30 | framed as a variance experiment (Pombal et al. 2022; Zhioua et al. 2025) |
| measurement bias (negative control) | zero the payment-history fields for a fraction of the group | train + calibration | {0, 0.5, 1.0} | 10 | implemented; no disparity on GMSC pilot — kept as control |
| score-shift (optional) | shift the logit of the group by β before thresholding (Legast et al. 2026) | train + calibration | β grid tbd | 10 | only if time permits |

- Same data partition reused across bias levels within a seed (Legast et al.)
  so dose–response curves are clean.
- Target group: Taiwan — chosen from the baseline table (lowest coverage
  group under marginal CP); GMSC — age ≤ 35 (Phase 1 worst group).
- Outputs: dose–response plots of (i) marginal CP coverage of the target
  group's defaulters, (ii) worst-group coverage and Δ_Cov per method, (iii)
  AFCP selection rate and per-label flag rate, (iv) set-size / singleton
  gaps — against f, mean ± SD over seeds.
- Testable prediction (from the review): under one-sided flipping, marginal
  CP loses coverage on the target group's defaulters first; equalized-
  coverage methods recover it at the price of set-size / singleton gaps
  (Cresswell et al.); AFCP's selection rate rises with f only once the
  (group × label) cell is large enough for the t-test.
- Contribution statement: no published study crosses group-dependent label
  bias with group-conditional conformal coverage, and none injects bias into
  Taiwan or GMSC.

## 6. Metrics and statistics

Per split: marginal coverage; per-class coverage; per-(group × class)
coverage; mean set size; singleton, {0,1} and empty-set rates; accuracy given
singleton; AFCP attribute selection frequency and per-label flag frequency.
Summaries: Δ_Cov and Δ_Size (max pairwise group gaps, Cresswell et al. 2025),
singleton-rate gap, worst-group coverage, CovGap (Ding et al.).
Statistics: mean ± SD and 2 SE over seeds; percentile-bootstrap CIs over
seeds; paired Wilcoxon signed-rank for method-vs-method contrasts; group-vs-
group within a method tested on the per-split gap paired across seeds;
Clopper–Pearson intervals for per-split rates; min/max worst-group coverage
across seeds reported.

## 7. Mapping to the research questions

| RQ | v2 evidence |
|---|---|
| RQ1 marginal CP under-covers groups? | baseline tables, 3 datasets × 3 models × 2 calibration regimes, with cell-count table and resolvable-gap statement |
| RQ2 which method reduces disparity? | same tables; paired tests; calibration sweep on Taiwan reproducing AFCP's sample-size curve on real data |
| RQ3 fairness vs informativeness | singleton / set-size gaps now meaningful under the marginal regime; Cresswell's criteria computed directly |
| new: detection power of AFCP under controlled bias | dose–response curves (Section 5); mechanistic explanation of Phase-1 non-selection (degeneracy, label veto) |

## 8. Order of work

1. agree this document → 2. data scripts + EDA + cell-count tables (Taiwan
new; GMSC fixes; SGC documented) → 3. classifiers with early stopping and
degeneracy logging, verified with the meter on all three datasets → 4.
baseline full-AFCP runs (parallel on the laptop) → 5. label-bias injection:
meter screening over the full grid, then full runs → 6. sensitivity
(ttest_delta, test level, model) → 7. tables and figures.

Each step ends with an update to the matching `docs/CREDIT_*.md` (new
`docs/CREDIT_TAIWAN.md`) and a commit. Meter outputs:
`thesis_experiments/results/<dataset>/bias_meter/`; full runs keep the
Phase-1 naming with suffix `_bias<type>_g<k>_eps<ε>_<scope>`.
