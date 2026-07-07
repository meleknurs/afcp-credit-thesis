"""
Multiclass AFCP experiment on the South German Credit dataset.

Replicates the structure of mc_real.py (Zhou & Sesia 2024) but adapted
to the South German Credit dataset with credit-risk fairness focus.

Usage:
    python mc_south_german.py <n_train_calib> <lr> <n_epoch> <seed> \
                              <ttest_delta> <conditional_> <run_config>

Parameters:
    n_train_calib (int): Training + calibration sample size
    lr (float): Learning rate
    n_epoch (int): Number of training epochs
    seed (int): Random seed
    ttest_delta (float): AFCP t-test tolerance parameter; 0 tests whether miscoverage exceeds alpha.
    conditional_ (bool): Label-conditional coverage (True/False)
    run_config (str): Sensitive-attribute configuration:
                      "age25" (age ≤25 vs >25),
                      "agemedian" (age ≤33 vs >33),
                      "pstatus" (personal_status_sex, 4 levels)

Example:
    python mc_south_german.py 500 0.0001 500 2023 0 True age25

Outputs:
    results/multiclass/real/sgc_<run_config>_ndata<n>_....txt
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import shutil

import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split

# Set path to find AFCP modules, regardless of where script is called from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "AFCP"))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from third_party.utils import *
from SelectiveCI_fairness.methods import (
    Marginal_Selection,
    Exhaustive_Selection,
    Partial_Selection,
    Adaptive_Selection,
)
from SelectiveCI_fairness.black_box import Blackbox
from SelectiveCI_fairness.networks import ClassNNet
from SelectiveCI_fairness.evals import eval_psets, eval_by_att_mc


#########################
# Experiment parameters #
#########################

print("Number of arguments:", len(sys.argv), "arguments.")
print("Argument List:", str(sys.argv))

if len(sys.argv) != 8:
    print("Error: incorrect number of parameters.")
    print("Usage: python mc_south_german.py <n_train_calib> <lr> <n_epoch>",
          "<seed> <ttest_delta> <conditional_> <run_config>")
    quit()

n_train_calib = int(sys.argv[1])
lr = float(sys.argv[2])
n_epoch = int(sys.argv[3])
seed = int(sys.argv[4])
ttest_delta = float(sys.argv[5])
conditional_ = eval(sys.argv[6])
run_config = sys.argv[7]

# Validate run_config
valid_configs = ["age25", "agemedian", "pstatus"]
if run_config not in valid_configs:
    print(f"Error: run_config must be one of {valid_configs}")
    quit()

# Fixed experiment parameters
num_workers = 0
batch_size = 25
beta = 0
alpha = 0.1  # nominal miscoverage level
n_test = 500

# Note: with South German Credit (n=1000 total), recommend n_train_calib=500.
# Ensure we have enough data for train/calib + test split.

###############
# Output file #
###############

outdir = "results/multiclass/real/"
os.makedirs(outdir, exist_ok=True)

outfile_name = (
    "sgc_"
    + run_config
    + "_ndata"
    + str(n_train_calib)
    + "_lr"
    + str(lr)
    + "_ttestdelta"
    + str(ttest_delta)
    + "_conditional"
    + str(conditional_)
    + "_seed"
    + str(seed)
)

outfile = outdir + outfile_name + ".txt"
print("Output file: {:s}".format(outfile), end="\n")

modeldir = "models/multiClass/real/" + outfile_name

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)


#########################
# Auxiliary functions   #
#########################

colors = [[31, 120, 180], [51, 160, 44], [250, 159, 181]]
colors = [(r / 255, g / 255, b / 255) for (r, g, b) in colors]


def plot_loss(train_loss, val_loss):
    x = np.arange(1, len(train_loss) + 1)
    plt.figure()
    plt.plot(x, train_loss, color=colors[0], label="Training loss", linewidth=2)
    plt.plot(x, val_loss, color=colors[1], label="Validation loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.title("Evolution of the training, validation and test loss")
    plt.show()


def criterion(outputs, inputs, targets):
    targets = targets.to(torch.long)
    return Loss(outputs, targets)


#################
# Load Data     #
#################

# Determine local file path (works even if script called from any directory)
LOCAL_ASC = os.path.join(os.path.dirname(__file__), "..", "data", "SouthGermanCredit.asc")

# Attempt to fetch from UCI; fall back to local .asc file
try:
    print("Attempting to fetch South German Credit (id=522) from UCI repository...")
    from ucimlrepo import fetch_ucirepo
    sgc = fetch_ucirepo(id=522)
    X = sgc.data.features
    y = sgc.data.targets
    print("Successfully fetched from UCI.")
except Exception as e:
    print(f"UCI fetch failed ({e}); loading from local file at {LOCAL_ASC}")
    try:
        X = pd.read_csv(LOCAL_ASC, sep=r"\s+")
        # Last column is target; separate it
        y = X.iloc[:, -1]
        X = X.iloc[:, :-1]
        # Rename columns to English (same mapping as in eda_south_german.py)
        col_map = {
            "laufkont": "status",
            "laufzeit": "duration",
            "moral": "credit_history",
            "verw": "purpose",
            "hoehe": "amount",
            "sparkont": "savings",
            "beszeit": "employment_duration",
            "rate": "installment_rate",
            "famges": "personal_status_sex",
            "buerge": "other_debtors",
            "wohnzeit": "present_residence",
            "verm": "property",
            "alter": "age",
            "weitkred": "other_installment_plans",
            "wohn": "housing",
            "bishkred": "number_credits",
            "beruf": "job",
            "pers": "people_liable",
            "telef": "telephone",
            "gastarb": "foreign_worker",
            "kredit": "credit_risk",
        }
        X.columns = [col_map.get(c, c) for c in X.columns]
        y.name = "credit_risk"
        print(f"Loaded from data/SouthGermanCredit.asc: n={len(X)}")
    except FileNotFoundError:
        print("Error: data/SouthGermanCredit.asc not found.")
        quit()

# Verify data
print(f"Data shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target classes: {np.unique(y)}")
assert X.shape[0] == y.shape[0], "Feature and target shape mismatch"

# Convert target to numeric (0, 1)
y = pd.Series(y).astype(int)

# Feature engineering: create sensitive-attribute binned versions
# All numeric/coded features remain as-is (per Grömping 2019, P2-score ordinals)
# We add new binary columns for age binning

# Verify age column exists and get its index
assert "age" in X.columns, "Column 'age' not found"
age_idx = list(X.columns).index("age")

# Create age bins
X["age_bin_25"] = (X["age"] > 25).astype(int)  # 0 if <= 25, 1 if > 25
X["age_bin_median"] = (X["age"] > 33).astype(int)  # 0 if <= 33, 1 if > 33

# Get indices of sensitive attributes
age_bin_25_idx = list(X.columns).index("age_bin_25")
age_bin_median_idx = list(X.columns).index("age_bin_median")
personal_status_sex_idx = list(X.columns).index("personal_status_sex")

print(f"\nSensitive attribute indices:")
print(f"  age_bin_25: {age_bin_25_idx}")
print(f"  age_bin_median: {age_bin_median_idx}")
print(f"  personal_status_sex: {personal_status_sex_idx}")

# Determine idx_list based on run_config
if run_config == "age25":
    idx_list = [age_bin_25_idx]
    config_desc = "age ≤25 vs >25"
elif run_config == "agemedian":
    idx_list = [age_bin_median_idx]
    config_desc = "age ≤33 vs >33 (median)"
elif run_config == "pstatus":
    idx_list = [personal_status_sex_idx]
    config_desc = "personal_status_sex (4 levels)"

print(f"\nRun configuration: {run_config}")
print(f"Description: {config_desc}")
print(f"Sensitive attribute indices: {idx_list}\n")

# Convert to numpy; keep all features numeric (no LabelEncoder)
# South German data is already coded; we preserve the ordinal structure
X_array = X.values.astype("float64")
y_array = y.values.astype("float64")

print(f"X shape (with engineered features): {X_array.shape}")
print(f"y shape: {y_array.shape}")
print(f"y distribution: {np.bincount(y_array.astype(int))}")

# Validate sample size
total_available = X_array.shape[0]
total_needed = n_train_calib + n_test
assert total_needed <= total_available, (
    f"Error: n_train_calib ({n_train_calib}) + n_test ({n_test}) = {total_needed} "
    f"exceeds dataset size ({total_available}). "
    f"Recommend n_train_calib <= {total_available - n_test}."
)

# Split into train+calib and test
random_indices = random.sample(range(X_array.shape[0]), n_train_calib + n_test)
X_full = X_array[random_indices, :]
y_full = y_array[random_indices]

X_train_calib, X_test, y_train_calib, y_test = train_test_split(
    X_full, y_full, train_size=n_train_calib, random_state=seed
)

X_train, X_calib, y_train, y_calib = train_test_split(
    X_train_calib, y_train_calib, train_size=0.5, random_state=seed
)

print(f"\nData split:")
print(f"  Train: {X_train.shape[0]}")
print(f"  Calib: {X_calib.shape[0]}")
print(f"  Test: {X_test.shape[0]}")

# Convert to tensors and data loaders
train_dataset = data_utils.TensorDataset(
    torch.Tensor(X_train), torch.Tensor(y_train)
)
calib_dataset = data_utils.TensorDataset(
    torch.Tensor(X_calib), torch.Tensor(y_calib)
)
test_dataset = data_utils.TensorDataset(
    torch.Tensor(X_test), torch.Tensor(y_test)
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=num_workers
)
calib_loader = torch.utils.data.DataLoader(
    calib_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=n_test, num_workers=num_workers, shuffle=False
)

################
# Train models #
################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nIs CUDA available? {torch.cuda.is_available()}")

# Binary classification (credit_risk: 0 or 1)
num_classes = 2
net = ClassNNet(
    num_features=X_train.shape[1],
    num_classes=num_classes,
    device=device,
    use_dropout=False,
)

Loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

bbox_mc = Blackbox(
    net,
    device,
    train_loader,
    batch_size=batch_size,
    max_epoch=n_epoch,
    learning_rate=lr,
    val_loader=calib_loader,
    criterion=criterion,
    optimizer=optimizer,
    verbose=False,
)

bbox_mc_stats = bbox_mc.full_train(save_dir=modeldir)

################
#   Inference  #
################


def complete_df_real(
    df, method=None, is_marg_results=False, is_selective=False, k_hat=None
):
    """Augment results DataFrame with experiment metadata and selected attributes."""
    df["n_data"] = n_train_calib
    df["seed"] = seed
    df["lr"] = lr
    df["batch_size"] = batch_size
    df["alpha"] = alpha
    df["method"] = method
    df["conditional"] = conditional_
    df["run_config"] = run_config
    if beta > 0:
        df["beta_threshold"] = beta * len(X_calib)
    if is_marg_results:
        df["n_count"] = len(y_test)
        df["attribute_idx"] = -1
        df["attribute_level"] = -1
    if is_selective == False:
        for idx in idx_list:
            df[f"selected_{idx}"] = None
    elif is_selective:
        assert k_hat is not None, "input selected attribute list"
        for idx in idx_list:
            df[f"selected_{idx}"] = np.mean([1 if x == [idx] else 0 for x in k_hat])

    return df


results_full = pd.DataFrame()

## Adaptive (AFCP with ttest_delta)
print("\n--- Running Adaptive (AFCP with ttest_delta) ---")
Selective_method = Adaptive_Selection(
    alpha=alpha, random_state=seed, ttest_delta=ttest_delta
)
C_sets_adaptive, k_hat = Selective_method.multiclass_classification(
    X_calib, y_calib, X_test, bbox_mc, idx_list, conditional=conditional_
)

results_marg = eval_psets(C_sets_adaptive, y_test)
results_marg = complete_df_real(
    results_marg,
    method="Adaptive",
    is_marg_results=True,
    is_selective=True,
    k_hat=k_hat,
)
results_cond = eval_by_att_mc(X_test, y_test, idx_list, C_sets_adaptive)
results_cond = complete_df_real(
    results_cond, method="Adaptive", is_selective=True, k_hat=k_hat
)
results_full = pd.concat([results_full, results_marg, results_cond])

## Adaptive1 (Adaptive without t-test filtering, i.e., ttest_delta=None)
print("--- Running Adaptive1 (no ttest_delta) ---")
Selective_method = Adaptive_Selection(alpha=alpha, random_state=seed)
C_sets_adaptive_1, k_hat_1 = Selective_method.multiclass_classification(
    X_calib, y_calib, X_test, bbox_mc, idx_list, conditional=conditional_
)

results_marg = eval_psets(C_sets_adaptive_1, y_test)
results_marg = complete_df_real(
    results_marg,
    method="Adaptive1",
    is_marg_results=True,
    is_selective=True,
    k_hat=k_hat_1,
)
results_cond = eval_by_att_mc(X_test, y_test, idx_list, C_sets_adaptive_1)
results_cond = complete_df_real(
    results_cond, method="Adaptive1", is_selective=True, k_hat=k_hat_1
)
results_full = pd.concat([results_full, results_marg, results_cond])

## Marginal
print("--- Running Marginal ---")
Selective_method = Marginal_Selection(alpha=alpha, random_state=seed)
C_sets_marginal = Selective_method.multiclass_classification(
    X_test,
    X_calib,
    y_calib,
    left_tail=False,
    bbox_mc=bbox_mc,
    conditional=conditional_,
)

results_marg = eval_psets(C_sets_marginal, y_test)
results_marg = complete_df_real(
    results_marg, method="Marginal", is_marg_results=True
)
results_cond = eval_by_att_mc(X_test, y_test, idx_list, C_sets_marginal)
results_cond = complete_df_real(results_cond, method="Marginal")
results_full = pd.concat([results_full, results_marg, results_cond])

## Partial
print("--- Running Partial ---")
Selective_method = Partial_Selection(alpha=alpha, random_state=seed)
C_sets_partial = Selective_method.multiclass_classification(
    X_test,
    X_calib,
    y_calib,
    left_tail=False,
    sensitive_atts_idx=idx_list,
    bbox_mc=bbox_mc,
    conditional=conditional_,
)

results_marg = eval_psets(C_sets_partial, y_test)
results_marg = complete_df_real(
    results_marg, method="Partial", is_marg_results=True
)
results_cond = eval_by_att_mc(X_test, y_test, idx_list, C_sets_partial)
results_cond = complete_df_real(results_cond, method="Partial")
results_full = pd.concat([results_full, results_marg, results_cond])

## Exhaustive
print("--- Running Exhaustive ---")
Selective_method = Exhaustive_Selection(alpha=alpha, random_state=seed)
C_sets_exhaustive = Selective_method.multiclass_classification(
    X_test,
    X_calib,
    y_calib,
    left_tail=False,
    sensitive_atts_idx=idx_list,
    bbox_mc=bbox_mc,
    conditional=conditional_,
)

results_marg = eval_psets(C_sets_exhaustive, y_test)
results_marg = complete_df_real(
    results_marg, method="Exhaustive", is_marg_results=True
)
results_cond = eval_by_att_mc(X_test, y_test, idx_list, C_sets_exhaustive)
results_cond = complete_df_real(results_cond, method="Exhaustive")
results_full = pd.concat([results_full, results_marg, results_cond])

################
# Save Results #
################
results_full.to_csv(outfile, index=False)
print(f"\nResults written to {outfile}\n")
sys.stdout.flush()

# Clean up temp model directory
shutil.rmtree(modeldir, ignore_errors=True)
print("Done.")