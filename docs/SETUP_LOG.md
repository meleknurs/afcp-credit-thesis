# AFCP Credit Thesis — Setup & Replication Log

This document records the complete setup process required to reproduce the
original Adaptively Fair Conformal Prediction (AFCP) implementation by Zhou &
Sesia (2024).

Its purpose is to document every step needed to verify that the original code
runs correctly before any thesis-specific modifications are introduced.

All thesis-specific datasets, experimental adaptations, preprocessing
decisions, and methodological choices are documented separately in
`V2_DESIGN.md`, the dataset documents `CREDIT_*.md` and `RESEARCH_SUMMARY.md`.

Project:
**Fairness-aware conformal prediction applied to credit scoring**

Base method:
Zhou & Sesia (2024), *Conformal Classification with Equalized Coverage for
Adaptively Selected Groups*

---

# Repository structure

The repository is intentionally organized so that the original AFCP
implementation remains untouched. All thesis-specific code is placed outside
the original source tree, making it immediately clear which components
originate from Zhou & Sesia (2024) and which were developed for this thesis.

```
afcp-credit-thesis/
├── AFCP/                        # Original Zhou & Sesia repository (unchanged)
│   ├── experiments/
│   ├── SelectiveCI_fairness/
│   └── third_party/
│
├── thesis_experiments/          # Thesis-specific experiment scripts
│
├── docs/
│   └── SETUP_LOG.md
│
├── data/                        # Local datasets (not tracked by Git)
├── results/                     # Local experiment outputs (not tracked by Git)
│
├── README.md
├── requirements.txt
└── .gitignore
```

The `AFCP/` directory is treated as the original reference implementation. Thesis-specific code is developed outside this directory to preserve a clear separation between the original implementation and the contributions of this thesis.

---

# Step 0 — Fresh clone of the original repository

**What**

Cloned the original authors' repository into a clean project directory.

```bash
git clone https://github.com/FionaZ3696/Adaptively-Fair-Conformal-Prediction.git afcp-credit-thesis
cd afcp-credit-thesis
```

**Why**

Start from a clean, unmodified copy of the authors' implementation so that the
replication remains trustworthy and every thesis-specific contribution can be
clearly distinguished from the original work.

---

# Step 1 — Read the README

**What**

Read the project documentation before executing any code.

```bash
cat README.md
```

**Why**

Understand the intended workflow, required dependencies, and the purpose of
each experiment before reproducing the original implementation.

**Key observations**

- `mc_sim.py` reproduces the synthetic multiclass experiments.
- `mc_real.py` reproduces the Nursery dataset experiments.
- `od_sim.py` and `od_real.py` correspond to the outlier detection experiments.
  They fall outside the scope of this thesis and were not considered.
- Required packages: numpy, scipy, scikit-learn, torch, tqdm, pandas,
  matplotlib, torchmetrics, statsmodels.

---

# Step 2 — Install dependencies

**What**

Installed all required Python packages.

```bash
pip install numpy scipy scikit-learn torch tqdm pandas matplotlib torchmetrics statsmodels
```

**Why**

The repository depends on these libraries. Installing the documented
dependencies ensures that the code can be executed without modification.

---

# Step 3 — Understand the mc_sim.py interface

**What**

Executed the script without parameters, inspected the error message, and then
read the beginning of the source code to determine the required command-line
arguments.

```bash
python mc_sim.py
head -40 mc_sim.py
sed -n '34,60p' mc_sim.py
```

**Why**

The script requires exactly 9 command-line arguments and provides no usage
message. Understanding every input parameter is necessary before reproducing
the experiments.

| Parameter | Meaning |
|---|---|
| n_train_calib | Training + calibration sample size |
| lr | Learning rate |
| n_epoch | Number of training epochs |
| seed | Random seed |
| ttest_delta | AFCP tolerance parameter (α + ttest_delta is the selection threshold) |
| delta0 | Synthetic data: majority-group uncertainty proportion |
| delta1 | Synthetic data: minority-group uncertainty proportion |
| conditional_ | Label-conditional coverage (True/False) |
| perc_1 | Synthetic data: minority-group proportion |

---

# Step 4 — First successful replication (synthetic data)

**What**

Executed the synthetic multiclass experiment using the authors' default
configuration.

```bash
python mc_sim.py 1000 0.0001 500 2023 0 0.1 0.9 True 0.2
```

**Output file**
`AFCP/experiments/results/multiclass/ndata1000_lr0.0001_delta10.9_ttestdelta0.0_conditionalTrue_seed2023.txt`

**Why**

Verify that the original implementation runs correctly on generated data
before applying it to real credit datasets.

**Synthetic data structure (`data_gen.py`, `mc_sim.py`)**

The synthetic data is not a saved file; it is generated fresh in memory on
every run. Key properties:

- p = 10 features per observation (7 pure random noise + 3 meaningful)
- The last three columns are evaluated as candidate sensitive attributes:
  - `X[:,-1]` = **group** — the real sensitive attribute; with perc_1=0.2,
    20% are group 1 (minority/hard), 80% are group 0 (majority/easy)
  - `X[:,-2]` = noise/decoy attribute (values cycle 1..5, no bias)
  - `X[:,-3]` = noise/decoy attribute (binned into 6..9, no bias)
- K = 6 classes
- Bias mechanism: group 1 has ~90% uncertain labels (delta1=0.9);
  group 0 has only ~10% uncertain labels (delta0=0.1)
- The code comment summarises this as: `group = [easy, hard]`

The 3 candidate sensitive attributes appear as `selected_7`, `selected_8`,
`selected_9` in the output. Only index 9 (group) carries real bias; indices
7 and 8 are decoys to test whether AFCP is misled.

**Result**

The experiment completed successfully. The resulting behaviour is consistent
with Zhou & Sesia (2024):

- Exhaustive always returns the largest prediction sets (set size = 6.0,
  i.e. all K=6 classes).
- AFCP and Marginal produce substantially smaller prediction sets (~2.0–2.3).
- Marginal exhibits undercoverage for some sensitive groups.
- Set size ordering: Marginal < AFCP ≈ Adaptive1 < Partial < Exhaustive.

**Note on the synthetic data model**

The default `data_model` in the code (delta0/delta1 parameterisation) is not
identical to the decision-tree model described in Appendix A7.1.1 of the
paper (Eq. A27). The paper also does not provide exact command-line parameters
for its figures. As a result, single runs will not reproduce the paper's exact
numerical values. This is a reproducibility gap in the original repository,
not an error in the replication. The synthetic experiment is used here solely
to confirm that the pipeline runs and produces qualitatively correct behaviour.

---

# Step 5 — Replication of the real-data experiment (Nursery dataset)

**What**

Executed the real-data multiclass experiment using the Nursery dataset and
the authors' default configuration.

```bash
pip install ucimlrepo
python mc_real.py 1000 0.0001 500 2023 0 True
```

**Output file**
`AFCP/experiments/results/multiclass/real/ndata1000_lr0.0001_ttestdelta0.0_conditionalTrue_seed2023.txt`

**Why**

The synthetic experiment confirms that the implementation runs on generated
data. Replicating the real-data experiment verifies that the pipeline functions
correctly on an actual dataset before any thesis-specific modifications are
introduced.

**Parameters**

| Parameter | Value | Meaning |
|---|---|---|
| n_train_calib | 1000 | Training + calibration sample size |
| lr | 0.0001 | Learning rate |
| n_epoch | 500 | Training epochs |
| seed | 2023 | Random seed |
| ttest_delta | 0 | AFCP tolerance parameter |
| conditional_ | True | Label-conditional coverage |

Note: delta0, delta1, and perc_1 are absent because mc_real.py uses real
data; these parameters are only required by the synthetic data generator.

**Data**

Nursery dataset, loaded automatically via `fetch_ucirepo` (UCI ML Repository).
12,960 observations, multiclass (5 labels), 8 features.
Five sensitive attributes are designated following Zhou & Sesia (2024):
Parents' occupation (idx 0), Number of children (idx 3), Financial standing
(idx 5), Social conditions (idx 6), Health status (idx 7).

**Result**

The experiment completed successfully. Overall results (test n=500):

| Method | Coverage | Set size |
|---|---|---|
| Marginal | 0.880 | 2.03 |
| Adaptive (AFCP) | 0.910 | 3.39 |
| Adaptive1 | 0.934 | 3.48 |
| Partial | 0.972 | 3.76 |
| Exhaustive | 1.000 | 4.00 |

Group-level inspection revealed that Marginal severely undercovered one
minority subgroup (attribute index 0 / Parents' occupation, level 1, n=20,
coverage 0.450). AFCP returned a substantially higher coverage for the same
group (0.950) without formally selecting any attribute (selected = 0 for all
attributes), consistent with the expectation that a calibration set of ~500
is insufficient for the t-test to reach significance. This pattern is also
observed in the thesis credit datasets.

The resulting behaviour is consistent with Zhou & Sesia (2024) Figure 5 and
Table A9:

- Exhaustive is maximally conservative (coverage 1.0, set size 4.0).
- Marginal undercovers the smallest minority subgroup (coverage 0.450).
- Set size ordering: Marginal < AFCP ≈ Adaptive1 < Partial < Exhaustive.

---

# Note — Bug identified in original implementation

During replication, a possible copy-paste error was identified in the
`selected` attribute reporting of the original `mc_real.py`.

The `selected` columns for attributes 3 and 6 incorrectly check for `[0]`
instead of their respective attribute indices:

```python
# Current (original code):
df['selected_3'] = np.mean([1 if x == [0] else 0 for x in k_hat])
df['selected_6'] = np.mean([1 if x == [0] else 0 for x in k_hat])

# Should be:
df['selected_3'] = np.mean([1 if x == [3] else 0 for x in k_hat])
df['selected_6'] = np.mean([1 if x == [6] else 0 for x in k_hat])
```

With `idx_list = [0, 3, 5, 6, 7]` (Nursery experiment), attributes 3
(Number of children) and 6 (Social conditions) are never correctly tracked
in the summary output. Both columns instead report the selection rate of
attribute 0.

This does not affect prediction sets or coverage results — only the
reporting of which attribute was selected.

The issue was reported to the original authors:
https://github.com/FionaZ3696/Adaptively-Fair-Conformal-Prediction/issues/1

The following message was submitted:

> While replicating your experiments as part of my MSc thesis, I noticed
> what appears to be a small copy-paste error in the `complete_df_real`
> function in `mc_real.py`.
>
> The `selected_3` and `selected_6` columns seem to check for `[0]` instead
> of their respective attribute indices.
>
> Since `idx_list = [0, 3, 5, 6, 7]` in the Nursery experiment, attributes 3
> (Number of children) and 6 (Social conditions) are otherwise never correctly
> tracked in the summary output — the corresponding columns instead report the
> selection rate of attribute 0.
>
> As far as I can tell, this only affects the reporting of the selected
> attribute and does not affect the prediction sets or coverage results.
>
> If I'm overlooking something about the intended implementation, I'd be very
> interested to know.

All thesis-specific scripts (`mc_south_german.py`, `mc_give_me_credit.py`)
use the corrected logic from the outset.

---

# Replication status

Completed

- Fresh clone of the original repository.
- Repository structure inspected.
- README reviewed.
- Dependencies installed.
- `mc_sim.py` interface understood.
- Successful replication of the synthetic multiclass experiment (`mc_sim.py`).
- Successful replication of the real-data experiment (`mc_real.py`, Nursery).
- Replicated behaviour agrees with Zhou & Sesia (2024) in both cases.

The replication of the original AFCP implementation is complete.

Subsequent thesis-specific datasets, methodological adaptations, and empirical
analyses are documented separately (see `RESEARCH_SUMMARY.md`).