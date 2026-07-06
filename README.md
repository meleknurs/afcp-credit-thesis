# AFCP Credit Thesis

This repository contains the code and documentation for the MSc Data Science
thesis:

**Fairness-Aware Conformal Prediction in Credit Scoring:
Comparing Coverage Equalization Methods Across Demographic Groups**

Meleknur Özgü — HSLU Lucerne, 2026

---

## Structure

```
afcp-credit-thesis/
├── AFCP/                    # Original Zhou & Sesia (2024) implementation (unchanged)
├── thesis_experiments/      # Thesis-specific experiment scripts
├── docs/
│   ├── SETUP_LOG.md         # Replication log for the original implementation
│   └── CREDIT_ANALYSIS.md  # Thesis-specific datasets, decisions, and results
├── data/                    # Local datasets (not tracked by Git, download separately)
├── results/                 # Local experiment outputs (not tracked by Git)
├── requirements.txt
└── .gitignore
```

The `AFCP/` directory is treated as the original reference implementation and
remains unchanged. All thesis-specific code is developed outside this directory.

---

## Base method

Zhou, F. & Sesia, M. (2024). *Conformal Classification with Equalized
Coverage for Adaptively Selected Groups.*

- Paper: https://arxiv.org/abs/2405.15106
- Code: https://github.com/FionaZ3696/Adaptively-Fair-Conformal-Prediction

---

## Datasets

Datasets are not tracked by Git. Download them separately and place them in
the `data/` directory.

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

- `docs/SETUP_LOG.md` — Step-by-step replication of the original synthetic
  and Nursery experiments, verifying that the implementation runs correctly
  before any thesis-specific modifications are introduced.

- `docs/CREDIT_ANALYSIS.md` — Thesis-specific datasets, preprocessing
  decisions, methodological choices, and empirical results. Updated
  incrementally as the analysis progresses.

---

## Replication

To reproduce the original experiments:

```bash
# Synthetic multiclass experiment
cd AFCP/experiments
python mc_sim.py 1000 0.0001 500 2023 0 0.1 0.9 True 0.2

# Nursery real-data experiment
pip install ucimlrepo
python mc_real.py 1000 0.0001 500 2023 0 True
```

See `docs/SETUP_LOG.md` for the full replication log and expected outputs.