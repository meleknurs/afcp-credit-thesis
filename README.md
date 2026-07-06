# AFCP Credit Thesis

This repository contains the code and documentation for the MSc Data Science 
thesis:

**Fairness-Aware Conformal Prediction in Credit Scoring:
Comparing Coverage Equalization Methods Across Demographic Groups**

Meleknur Özgü — HSLU Lucerne, 2026

---

## Structure

- `AFCP/` — Original implementation by Zhou & Sesia (2024), unchanged.
- `thesis_experiments/` — Thesis-specific experiment scripts.
- `docs/` — Setup log and analysis documentation.
- `data/` — Local datasets (not tracked by Git, download separately).
- `results/` — Local experiment outputs (not tracked by Git).

## Base method

Zhou, F. & Sesia, M. (2024). *Conformal Classification with Equalized 
Coverage for Adaptively Selected Groups.*
Code: https://github.com/FionaZ3696/Adaptively-Fair-Conformal-Prediction

## Datasets

- South German Credit (Grömping 2019): https://archive.ics.uci.edu/dataset/522
- Give Me Some Credit: https://www.kaggle.com/c/GiveMeSomeCredit

## Requirements

See `requirements.txt`. Install with:
```bash
pip install -r requirements.txt
```