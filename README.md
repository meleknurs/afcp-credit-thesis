# AFCP Credit Thesis

> **Status:** Work in progress (MSc thesis, 2026)

This repository contains the code, results and documentation for the MSc Data
Science thesis

**Fairness-Aware Conformal Prediction in Credit Scoring:
Comparing Coverage Equalization Methods Across Demographic Groups**

Meleknur Özgü — Lucerne University of Applied Sciences and Arts (HSLU), 2026

**Start here:** [`docs/RESEARCH_SUMMARY.md`](docs/RESEARCH_SUMMARY.md) gives a
step-by-step account of the experiments, from the pilot study to the v2
results, and maps them to the research questions.

---

## Research questions

- **RQ1.** How does group-dependent label bias manifest in the coverage and
  size of conformal prediction sets?
- **RQ2.** Do equalized-coverage methods (Partial, Exhaustive, AFCP) correct
  the resulting under-coverage?
- **RQ3.** When does AFCP detect the affected group, and when does it remain
  silent?
- **RQ4.** Are the conclusions sensitive to how the groups are defined?

---

## Base method

The thesis builds on Adaptively Fair Conformal Prediction (AFCP):

Zhou, Y., & Sesia, M. (2024). *Conformal Classification with Equalized
Coverage for Adaptively Selected Groups.* Advances in Neural Information
Processing Systems (NeurIPS).
Paper: https://arxiv.org/abs/2405.15106 ·
Code: https://github.com/FionaZ3696/Adaptively-Fair-Conformal-Prediction

The original implementation is included unchanged in `AFCP/`. All
thesis-specific code is kept outside that directory, so that the original
work and the contributions of this thesis remain clearly separated.

---

## Repository structure

```
afcp-credit-thesis/
├── AFCP/                         # Original Zhou & Sesia implementation (unchanged)
├── thesis_experiments/
│   ├── v2/                       # Main experiments (v2 design)
│   │   ├── data.py               #   loaders and preprocessing for the three datasets
│   │   ├── models.py             #   neural network (early stopping) and logistic regression
│   │   ├── run_afcp.py           #   full conformal run: Marginal, Partial, Exhaustive, AFCP, Adaptive1
│   │   ├── meter.py              #   fast disparity meter (screening of bias configurations)
│   │   ├── eda_taiwan.py, make_cell_tables.py
│   │   └── analysis/             #   summary tables (sample-size sweep, multi-attribute, RQ4)
│   ├── mc_*.py, eda_*.py,        # Pilot study (July 2026) scripts
│   │   combine_*.py, bias_meter.py
│   └── results/
│       ├── v2/                   # v2 results and summary tables
│       ├── south_german_credit/  # Pilot study results
│       └── give_me_credit/       # Pilot study results
├── archive/phase0_early_trials/  # Earliest trials (March–July 2026), kept for transparency
├── docs/                         # Design, dataset documentation, figures
├── tools/                        # Helper scripts for running batches (not part of the analysis)
├── data/                         # Datasets (see below)
├── .github/workflows/            # GitHub Actions workflow for large v2 run grids
└── requirements.txt
```

---

## Documentation

| Document | Content |
|---|---|
| [`docs/RESEARCH_SUMMARY.md`](docs/RESEARCH_SUMMARY.md) | Chronological summary of all experiments and findings |
| [`docs/V2_DESIGN.md`](docs/V2_DESIGN.md) | Experimental design of v2: why it was needed, datasets, classifiers, conformal setup, bias injection, metrics |
| [`docs/CREDIT_TAIWAN.md`](docs/CREDIT_TAIWAN.md) | Taiwan credit-card data: EDA, preprocessing and v2 experiments (main test bed) |
| [`docs/CREDIT_GMSC.md`](docs/CREDIT_GMSC.md) | Give Me Some Credit: EDA, preprocessing, pilot and v2 experiments |
| [`docs/CREDIT_SOUTH_GERMAN.md`](docs/CREDIT_SOUTH_GERMAN.md) | South German Credit: EDA, preprocessing, pilot and v2 experiments |
| [`docs/SETUP_LOG.md`](docs/SETUP_LOG.md) | Replication of the original AFCP implementation before any thesis-specific changes |

Only result files that back reported tables are tracked; the comments in
`.gitignore` document which files are kept.

---

## Datasets

| Dataset | Source | Licence | In repository |
|---|---|---|---|
| Default of Credit Card Clients (Taiwan) | Yeh & Lien (2009), [UCI ML Repository id 350](https://doi.org/10.24432/C55S3H) | CC BY 4.0 | Yes: `data/default_credit_card_clients.csv` (converted from the UCI `.xls`; checksum in `.sha256`) |
| South German Credit | Grömping (2019), [UCI ML Repository id 522](https://archive.ics.uci.edu/dataset/522) | CC BY 4.0 | No: download `SouthGermanCredit.asc` and place it in `data/` |
| Give Me Some Credit | [Kaggle competition](https://www.kaggle.com/c/GiveMeSomeCredit) | Kaggle competition rules | No: download `cs-training.csv` and place it in `data/` |

---

## Installation

Tested with Python 3.11.

```bash
pip install -r requirements.txt
```

---

## Running the experiments

v2 full conformal run (one dataset × model × seed × configuration):

```bash
python thesis_experiments/v2/run_afcp.py <dataset> <model> <seed> <config> <conditional> [key=value ...]
```

See the docstring of `run_afcp.py` for the available datasets, configurations
and options. Large grids are run with the GitHub Actions workflow in
`.github/workflows/afcp_runs.yml`.

Replication of the original AFCP experiments (no thesis-specific changes):

```bash
cd AFCP/experiments
python mc_sim.py 1000 0.0001 500 2023 0 0.1 0.9 True 0.2   # synthetic data
python mc_real.py 1000 0.0001 500 2023 0 True              # Nursery data
```

The expected outputs are documented in `docs/SETUP_LOG.md`.

---

## References

- Grömping, U. (2019). *South German Credit Data: Correcting a Widely Used
  Data Set.* Reports in Mathematics, Physics and Chemistry 4/2019, Beuth
  University of Applied Sciences Berlin.
- Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques
  for the predictive accuracy of probability of default of credit card
  clients. *Expert Systems with Applications*, 36(2), 2473–2480.
- Zhou, Y., & Sesia, M. (2024). Conformal classification with equalized
  coverage for adaptively selected groups. *NeurIPS 2024*.
