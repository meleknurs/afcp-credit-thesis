# AFCP Credit Thesis — Setup & Replication Log

This document records the complete setup process required to reproduce the

original Adaptive Fair Conformal Prediction (AFCP) implementation by Zhou &

Sesia (2024).

Its purpose is to document every step needed to verify that the original code

runs correctly before any thesis-specific modifications are introduced.

All thesis-specific datasets, experimental adaptations, preprocessing

decisions, and methodological choices are documented separately.

Project:

**Fairness-aware conformal prediction applied to credit scoring**

Base method:

Zhou & Sesia (2024), *Conformal Classification with Equalized Coverage for

Adaptively Selected Groups*

---

# Repository structure

The repository is intentionally organized so that the original AFCP

implementation remains untouched. All thesis-specific code is placed outside the

original source tree, making it immediately clear which components originate

from Zhou & Sesia (2024) and which were developed for this thesis.

```

afcp-credit-thesis/

├── AFCP/                       # Original Zhou & Sesia repository (unchanged)

│   ├── experiments/

│   ├── SelectiveCI_fairness/

│   └── third_party/

│

├── thesis_experiments/         # Thesis-specific experiment scripts

│

├── docs/

│   └── SETUP_LOG.md

│

├── data/                       # Local datasets (not tracked by Git)

├── results/                    # Local experiment outputs

│

├── README.md

├── requirements.txt

└── .gitignore

```

The `AFCP/` directory should remain unchanged throughout the project whenever

possible. Any new experiments, datasets, or adaptations should be implemented

outside the original repository structure.

---

# Step 0 — Fresh clone of the original repository

**What**

Cloned the original authors' repository into a clean project directory.

```bash

cd ~/Desktop

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

- `od_sim.py` and `od_real.py` correspond to the paper's outlier detection experiments. They fall outside the scope of this thesis and were therefore not considered.

- Required packages include:

  - numpy

  - scipy

  - scikit-learn

  - torch

  - tqdm

  - pandas

  - matplotlib

  - torchmetrics

  - statsmodels

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

The script does not provide default values. Understanding every input parameter

is necessary before reproducing the experiments.

| Parameter | Meaning |

|-----------|---------|

| n_train_calib | Training + calibration sample size |

| lr | Learning rate |

| n_epoch | Number of training epochs |

| seed | Random seed |

| ttest_delta | AFCP tolerance parameter |

| delta0 | Synthetic data parameter |

| delta1 | Synthetic data parameter |

| conditional_ | Label-conditional coverage |

| perc_1 | Minority-group proportion |

---

# Step 4 — First successful replication

**What**

Executed the synthetic multiclass experiment using the authors' default

configuration.

```bash

python mc_sim.py 1000 0.0001 500 2023 0 0.1 0.9 True 0.2

```

**Why**

The first objective is to verify that the original implementation runs correctly

before making any modifications for the thesis.

**Result**

The experiment completed successfully and generated the expected output file.

The resulting behaviour is consistent with the findings reported by Zhou &

Sesia (2024):

- Exhaustive always returns the largest prediction sets.

- AFCP and Marginal produce substantially smaller prediction sets.

- Marginal exhibits undercoverage for some sensitive groups.

- AFCP's adaptive group-selection mechanism is active.

This confirms that the implementation is functioning correctly and reproduces

the qualitative behaviour described in the original paper.

---

# Replication status

Completed

- Fresh clone of the original repository.

- Repository structure inspected.

- README reviewed.

- Dependencies installed.

- `mc_sim.py` interface understood.

- Successful replication of the synthetic multiclass experiment.

- Replicated behaviour agrees with Zhou & Sesia (2024).

Next step

- Reproduce the original real-data experiment (`mc_real.py`, Nursery dataset)

  before introducing any thesis-specific datasets or modifications.