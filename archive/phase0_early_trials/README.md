# Phase 0 — early trials (March – early July 2026)

Archived, read-only. These are the first credit-scoring experiments run directly
inside the original AFCP `experiments/` folder, before the thesis repository
(`thesis_experiments/`) was created on 6 July 2026. They are kept for
transparency and because they explain two decisions in the thesis:

1. **Why `foreign_worker` was dropped as a sensitive attribute**
   (it was promised in the preliminary study). On the UCI German Credit
   numeric data (n_train_calib = 700, n_test = 300, seed 2023, ttest_delta = 0.05)
   marginal CP covered foreign workers at 0.942 vs 0.935 for non-foreign
   workers: no disparity to correct, and AFCP never selected the attribute.
2. **Why the GMSC training/calibration sample-size sweep exists.** The first
   sweep (3 July, seed 2023, ttest_delta = 0.05, n_test = 1000) showed AFCP
   selecting `age` only from n_train_calib = 16,000 (n_calib = 8,000) upward
   (`gmsc_sweep_summary_ntest1000_seed2023.txt`). The July re-run with
   ttest_delta = 0 and five seeds (manuscript Table 4) moved the threshold to
   4,000/4,000 for seeds 2023 and 2027 and found no selection at any size for
   seeds 2024–2026.

All runs: single seed (2023), five-layer NN from Zhou & Sesia, 500 epochs,
label-conditional calibration (`conditional=True`), ttest_delta = 0.05.

## Files

| file | what |
|---|---|
| `results/german_credit_*` | UCI German Credit (`german.data-numeric`, not the South German Credit used from July on). `german_credit`: age 3-way (<=25 / 25–50 / >50); `_foreign`: foreign worker; `_sex`: personal status & sex (4 levels); `_multi`: all three as candidate attributes. |
| `results/gmsc_ndata{4000,5000}_*` | first GMSC runs (30 June), age as sensitive attribute, 500 test points |
| `results/gmsc_sweep_*` | first GMSC sample-size sweep (3 July), n_test = 1000 |
| `results/nursery_*` | replication of the paper's Nursery experiment (14 March) |
| `scripts/` | the scripts as they were at the time (they expect to live inside `AFCP/experiments/`) |

Result-file columns follow the original AFCP output format; `selected_<idx>`
is the fraction of test points for which attribute `<idx>` was selected.
