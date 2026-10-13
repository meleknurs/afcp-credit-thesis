# AFCP Credit Thesis

> **Status:** Work in progress (Master's thesis, 2026)

This repository contains the code and documentation for the MSc Data Science
thesis:

**Fairness-Aware Conformal Prediction in Credit Scoring:
Comparing Coverage Equalization Methods Across Demographic Groups**

Meleknur Özgü — HSLU Lucerne, 2026

---

## Structure

```
afcp-credit-thesis/
├── AFCP/                    # Original implementation (kept unchanged)
├── thesis_experiments/      # Thesis-specific experiment scripts
│   ├── eda_south_german.py / eda_give_me_credit.py
│   ├── mc_south_german.py / mc_give_me_credit.py / mc_give_me_credit_sweep.py
│   ├── combine_results_south_german.py / combine_sweep_results.py
│   └── results/             # Seed-level outputs and summaries backing the manuscript tables
│       ├── south_german_credit/
│       └── give_me_credit/
├── docs/
│   ├── SETUP_LOG.md
│   ├── CREDIT_SOUTH_GERMAN.md
│   ├── CREDIT_GMSC.md
│   └── figures/
├── tools/                   # Helper scripts (not part of the analysis)
├── data/                    # Local datasets (not tracked by Git)
├── requirements.txt
└── .gitignore
```

The `AFCP/` directory is treated as the original reference implementation and
remains unchanged. All thesis-specific code is developed outside this directory.

---

## Base method

This work builds upon the Adaptive Fair Conformal Prediction (AFCP) framework
introduced by

Zhou, F., & Sesia, M. (2024).
*Conformal Classification with Equalized Coverage for Adaptively Selected Groups.*

Paper: https://arxiv.org/abs/2405.15106
Code: https://github.com/FionaZ3696/Adaptively-Fair-Conformal-Prediction

---

## Datasets

Datasets are intentionally excluded from the repository due to licensing and
distribution restrictions. Download them separately and place them in the
`data/` directory.

- **South German Credit** (Grömping 2019, corrected version of the UCI
  Statlog German Credit dataset):
  https://archive.ics.uci.edu/dataset/522

- **Give Me Some Credit** (Kaggle):
  https://www.kaggle.com/c/GiveMeSomeCredit

---

## Requirements

```bash
pip install -r requirements.txt
```

---

## Documentation

- `docs/SETUP_LOG.md` — Documents the complete replication of the original
  AFCP implementation, verifying that the code runs correctly before any
  thesis-specific modifications are introduced.

- `docs/CREDIT_SOUTH_GERMAN.md` — South German Credit: exploratory data
  analysis, choice of sensitive attributes, experimental design and results
  (primary age ≤25 split, median-split robustness check, exploratory
  personal-status analysis).

- `docs/CREDIT_GMSC.md` — Give Me Some Credit: exploratory data analysis,
  choice of age as the sensitive attribute, main three-level age comparison,
  joint training/calibration sample-size sweep, and exploratory
  multi-attribute analysis.

Only the result files that back the manuscript's tables are tracked; see the
comments in `.gitignore` for the mapping from tables to files.

---

## Replication

The commands below reproduce the experiments provided by the original
repository without any thesis-specific modifications.

```bash
# Synthetic multiclass experiment
cd AFCP/experiments
python mc_sim.py 1000 0.0001 500 2023 0 0.1 0.9 True 0.2

# Nursery real-data experiment
pip install ucimlrepo
python mc_real.py 1000 0.0001 500 2023 0 True
```

See `docs/SETUP_LOG.md` for the full replication log and expected outputs.

---

## Repository philosophy

The goal of this repository is to maintain a strict separation between

1. replication of the original AFCP implementation; and
2. thesis-specific developments.

Accordingly,

- the `AFCP/` directory contains the original implementation and is kept
  unchanged whenever possible;
- all newly developed code is placed outside the original source tree;
- all methodological decisions are documented separately from the replication
  process.