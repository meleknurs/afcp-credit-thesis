"""
Multiclass AFCP experiment on the Give Me Some Credit dataset.

Replicates the structure of mc_south_german.py (Zhou & Sesia 2024) with
GMSC-specific preprocessing. The main fairness analysis focuses on age; the
"multi" configuration additionally considers income and number of
dependents as competing candidate sensitive attributes.

Usage:
    python mc_give_me_credit.py <n_train_calib> <lr> <n_epoch> <seed> \
                                <ttest_delta> <conditional_> <run_config>

run_config:
    "age_median"      age <= median vs > median (primary)
    "age_40"          age < 40 vs >= 40 (secondary/robustness)
    "age_3way"        age <=35 / 35-55 / >55 (three-level)
    "multi"           age_3way + income median split + dependents 0 vs 1+

Example:
    python mc_give_me_credit.py 500 0.0001 500 2023 0 True age_median
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random
import shutil

import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Path setup (fixed: REPO_ROOT was previously undefined) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "AFCP"))
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
    print("Usage: python mc_give_me_credit.py <n_train_calib> <lr> <n_epoch>",
          "<seed> <ttest_delta> <conditional_> <run_config>")
    quit()

n_train_calib = int(sys.argv[1])
lr = float(sys.argv[2])
n_epoch = int(sys.argv[3])
seed = int(sys.argv[4])
ttest_delta = float(sys.argv[5])
conditional_ = eval(sys.argv[6])
run_config = sys.argv[7]

valid_configs = ["age_median", "age_40", "age_3way", "age_income", "age_dependents", "multi"]
if run_config not in valid_configs:
    print(f"Error: run_config must be one of {valid_configs}")
    quit()

num_workers = 0
batch_size = 25
beta = 0
alpha = 0.1
n_test = 500

###############
# Output file #
###############

outdir = os.path.join(SCRIPT_DIR, "results", "multiclass", "real")
os.makedirs(outdir, exist_ok=True)

outfile_name = (
    "gmsc_"
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

outfile = os.path.join(outdir, outfile_name + ".txt")
print("Output file: {:s}".format(outfile), end="\n")

modeldir = os.path.join(SCRIPT_DIR, "models", "multiClass", "real", outfile_name)

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

def criterion(outputs, inputs, targets):
    targets = targets.to(torch.long)
    return Loss(outputs, targets)

#################
# Load Data     #
#################

DATA_PATH = os.path.join(REPO_ROOT, "data", "cs-training.csv")
print(f"\nLoading GMSC from {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
print(f"Raw data shape: {df.shape}")

# Remove impossible age entry
df = df[df["age"] > 0].reset_index(drop=True)
print(f"After age > 0 filter: {df.shape}")

# Target
assert "SeriousDlqin2yrs" in df.columns, "Target column not found"
y = df["SeriousDlqin2yrs"].astype(int)
X = df.drop(columns=["SeriousDlqin2yrs"]).copy()

# Sensitive attribute construction (age only)
age_median = X["age"].median()
print(f"Age median: {age_median}")

X["age_bin_median"] = (X["age"] <= age_median).astype(int)
X["age_bin_40"] = (X["age"] < 40).astype(int)
X["age_bin_3way"] = pd.cut(
    X["age"],
    bins=[0, 35, 55, 150],
    labels=[0, 1, 2],
    include_lowest=True
).astype(int)

age_bin_median_idx = list(X.columns).index("age_bin_median")
age_bin_40_idx = list(X.columns).index("age_bin_40")
age_bin_3way_idx = list(X.columns).index("age_bin_3way")

# Additional candidate sensitive attributes for the multi-attribute config.
# Candidate group labels are constructed before the split, using the full
# dataset's income median and zero-imputation for missing dependents, to
# remain consistent with the existing experiment design (age_median/
# age_bin_median follow the same pattern). This means the income threshold
# is not purely training-set-derived -- a known,
# accepted limitation shared with the rest of this study's median-split
# configs, not unique to this attribute. Predictive values (as opposed to
# the group label) are still imputed later, per-split, using training-set
# statistics only.
income_median_global = X["MonthlyIncome"].median()
X["income_bin"] = (X["MonthlyIncome"].fillna(income_median_global) <= income_median_global).astype(int)
X["dependents_bin"] = (X["NumberOfDependents"].fillna(0) > 0).astype(int)  # 0 vs 1+ dependents

income_bin_idx = list(X.columns).index("income_bin")
dependents_bin_idx = list(X.columns).index("dependents_bin")

if run_config == "age_median":
    idx_list = [age_bin_median_idx]
    config_desc = f"age <= {age_median:.0f} vs >{age_median:.0f}"
elif run_config == "age_40":
    idx_list = [age_bin_40_idx]
    config_desc = "age <40 vs >=40"
elif run_config == "age_3way":
    idx_list = [age_bin_3way_idx]
    config_desc = "age <=35 / 35-55 / >55"
elif run_config == "age_income":
    # Diagnostic intermediate step: isolates whether a single extra candidate
    # attribute already suppresses age selection, before testing all three.
    idx_list = [age_bin_3way_idx, income_bin_idx]
    config_desc = "diagnostic: age_3way + income_bin (<=median)"
elif run_config == "age_dependents":
    idx_list = [age_bin_3way_idx, dependents_bin_idx]
    config_desc = "diagnostic: age_3way + dependents_bin (0 vs 1+)"
elif run_config == "multi":
    idx_list = [age_bin_3way_idx, income_bin_idx, dependents_bin_idx]
    config_desc = "multi-attribute: age_3way + income_bin (<=median) + dependents_bin (0 vs 1+)"

attribute_names = {
    age_bin_median_idx: "age_median",
    age_bin_40_idx: "age_40",
    age_bin_3way_idx: "age_3way",
    income_bin_idx: "income",
    dependents_bin_idx: "dependents",
}

print(f"\nRun configuration: {run_config}")
print(f"Description: {config_desc}")
print(f"Sensitive attribute indices: {idx_list}")
print(f"X columns: {list(X.columns)}")

# Validate sample size
total_available = X.shape[0]
total_needed = n_train_calib + n_test
assert total_needed <= total_available, (
    f"n_train_calib + n_test = {total_needed} exceeds dataset size {total_available}"
)

# Sample fixed subset
random_indices = random.sample(range(X.shape[0]), total_needed)
X_full = X.iloc[random_indices].reset_index(drop=True)
y_full = y.iloc[random_indices].reset_index(drop=True)

# Stratified splits: GMSC is highly imbalanced (~6.7% default rate),
# so stratification keeps the default proportion stable across splits.
stratify_full = y_full.astype(int)

X_train_calib, X_test, y_train_calib, y_test = train_test_split(
    X_full, y_full, train_size=n_train_calib, random_state=seed,
    stratify=stratify_full
)

X_train, X_calib, y_train, y_calib = train_test_split(
    X_train_calib, y_train_calib, train_size=0.5, random_state=seed,
    stratify=y_train_calib.astype(int)
)

# Explicit copies to avoid SettingWithCopyWarning during in-place preprocessing
X_train = X_train.copy()
X_calib = X_calib.copy()
X_test = X_test.copy()

# Imputation using training-sample information
income_median = X_train["MonthlyIncome"].median()

for split_name, split_df in [
    ("train", X_train),
    ("calib", X_calib),
    ("test", X_test),
]:
    split_df["MonthlyIncome"] = split_df["MonthlyIncome"].fillna(income_median)
    split_df["NumberOfDependents"] = split_df["NumberOfDependents"].fillna(0)

print(f"\nImputation:")
print(f"  MonthlyIncome median from training set: {income_median:.2f}")
print(f"  NumberOfDependents missing values filled with 0")

# Diagnostic: sensitive-group counts per split. Small calibration subgroups
# reduce the power of the worst-group t-test, which is a candidate
# explanation for suppressed selection under multi-attribute configs.
print(f"\nSensitive-group counts (diagnostic):")
for col in ["age_bin_3way", "income_bin", "dependents_bin"]:
    print(f"\n  {col}")
    print("    train:", X_train[col].value_counts(dropna=False).sort_index().to_dict())
    print("    calib:", X_calib[col].value_counts(dropna=False).sort_index().to_dict())
    print("    test :", X_test[col].value_counts(dropna=False).sort_index().to_dict())

# Standardization using training-sample statistics.
# Age bins are excluded from scaling because AFCP selection and group-wise
# evaluation rely on their raw integer values.
sensitive_cols = ["age_bin_median", "age_bin_40", "age_bin_3way", "income_bin", "dependents_bin"]
continuous_cols = [c for c in X_train.columns if c not in sensitive_cols]

scaler = StandardScaler()
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_calib[continuous_cols] = scaler.transform(X_calib[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

print(f"\nStandardization:")
print(f"  Scaled columns: {len(continuous_cols)} (fit on training set)")
print(f"  Excluded from scaling: {sensitive_cols}")

# Convert to numpy
X_train = X_train.values.astype("float64")
X_calib = X_calib.values.astype("float64")
X_test = X_test.values.astype("float64")
y_train = y_train.values.astype("float64")
y_calib = y_calib.values.astype("float64")
y_test = y_test.values.astype("float64")

print(f"\nData split:")
print(f"  Train: {X_train.shape[0]}")
print(f"  Calib: {X_calib.shape[0]}")
print(f"  Test: {X_test.shape[0]}")
print(f"  y distribution full subset: {np.bincount(y_full.astype(int))}")
print(f"  y train distribution: {np.bincount(y_train.astype(int))}")
print(f"  y calib distribution: {np.bincount(y_calib.astype(int))}")
print(f"  y test distribution: {np.bincount(y_test.astype(int))}")

########################
# Data loaders         #
########################

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
# Train model  #
################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nIs CUDA available? {torch.cuda.is_available()}")

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
# Inference    #
################

def normalize_selection(selection):
    """Coerce a single k_hat entry into a flat list of int attribute indices.
    Diagnostic helper: guards against k_hat elements arriving in an
    unexpected shape (e.g. nested lists, numpy scalars) silently producing
    a wrong selected_* count."""
    if selection is None:
        return []
    arr = np.asarray(selection, dtype=object).reshape(-1)
    normalized = []
    for value in arr:
        try:
            normalized.append(int(value))
        except (TypeError, ValueError):
            continue
    return normalized


def print_k_hat_diagnostics(method_name, k_hat):
    """Diagnostic-only console output (not persisted to CSV). Confirms
    k_hat's actual shape/content before trusting the selected_* summary."""
    print(f"\n{method_name} k_hat diagnostics:")
    print("  type:", type(k_hat))
    print("  length:", len(k_hat))
    print("  first 20:", k_hat[:20])
    counts = {}
    for selection in k_hat:
        key = str(selection)
        counts[key] = counts.get(key, 0) + 1
    print("  selection counts:", counts)


class DiagnosticAdaptiveSelection(Adaptive_Selection):
    """Diagnostic-only wrapper. Does NOT modify Zhou & Sesia's methods.py --
    it only overrides select_the_worst_group to log each per-placeholder-
    label call before returning the original, unmodified result unchanged.
    multiclass_classification() in methods.py calls select_the_worst_group
    once per placeholder label (labels = {0, 1} for GMSC) and then takes the
    INTERSECTION of the per-label selections as the final k_hat. This log
    lets us see the two label-conditional selections separately, to check
    whether "selected_none" reflects genuine non-selection or disagreement
    between the two label-conditional picks. Used for this diagnostic run
    only -- not part of the reported results pipeline."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_selection_log = []

    def select_the_worst_group(self, err_func, att_idx, E, X_aug):
        k_hat = super().select_the_worst_group(err_func, att_idx, E, X_aug)
        self.label_selection_log.append(k_hat)
        return k_hat


def print_label_selection_diagnostics(method_name, selector, num_labels=2):
    """Diagnostic-only console output. Splits the logged per-call selections
    by placeholder-label position (call order is label0, label1, label0,
    label1, ... per test point) and reports the per-label selection counts
    plus the fraction of test points where both labels picked the same
    attribute (== the fraction contributing a non-empty final k_hat)."""
    log = selector.label_selection_log
    print(f"\n{method_name} label-wise selection diagnostics:")
    print(f"  total calls logged: {len(log)} (expected: {num_labels} x n_test)")
    per_label = {lbl: {} for lbl in range(num_labels)}
    for i, k in enumerate(log):
        lbl = i % num_labels
        key = str(k)
        per_label[lbl][key] = per_label[lbl].get(key, 0) + 1
    for lbl, counts in per_label.items():
        print(f"  placeholder label {lbl}: {counts}")
    n_test_points = len(log) // num_labels
    agree = 0
    for t in range(n_test_points):
        picks = [log[t * num_labels + l] for l in range(num_labels)]
        # k_hat entries can be an int (selected attribute) or [] (no
        # selection) -- [] is unhashable, so compare via str() instead of
        # putting the raw values into a set.
        if len(set(str(p) for p in picks)) == 1:
            agree += 1
    print(
        f"  label agreement rate: {agree}/{n_test_points} = "
        f"{agree / n_test_points:.3f}"
    )


def complete_df_real(
    df, method=None, is_marg_results=False, is_selective=False, k_hat=None
):
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
    if is_selective is False:
        for idx in idx_list:
            df[f"selected_{idx}"] = None
    elif is_selective:
        assert k_hat is not None, "input selected attribute list"
        normalized_k_hat = [normalize_selection(x) for x in k_hat]
        # Keep the original index-keyed columns for compatibility with
        # already-aggregated results (canonical/age_3way/multi tables).
        for idx in idx_list:
            df[f"selected_{idx}"] = np.mean([
                idx in selection for selection in normalized_k_hat
            ])
        # Additionally record named columns + explicit non-selection rate,
        # for diagnostic readability and the selection_total sanity check.
        selection_rates = []
        for idx in idx_list:
            name = attribute_names[idx]
            rate = np.mean([
                idx in selection for selection in normalized_k_hat
            ])
            df[f"selected_{name}"] = rate
            selection_rates.append(rate)
        none_rate = np.mean([
            len(selection) == 0 for selection in normalized_k_hat
        ])
        df["selected_none"] = none_rate
        selection_total = none_rate + sum(selection_rates)
        df["selection_total"] = selection_total
        if is_marg_results:
            # Print once per method (marg branch only) to avoid duplicate
            # output when complete_df_real is also called for cond results.
            print(
                f"Selection sanity check ({method}): "
                f"selected attributes + none = {selection_total:.6f}"
            )
    return df

results_full = pd.DataFrame()

## Adaptive
print("\n--- Running Adaptive ---")
Selective_method = DiagnosticAdaptiveSelection(
    alpha=alpha, random_state=seed, ttest_delta=ttest_delta
)
C_sets_adaptive, k_hat = Selective_method.multiclass_classification(
    X_calib, y_calib, X_test, bbox_mc, idx_list, conditional=conditional_
)
try:
    print_k_hat_diagnostics("Adaptive", k_hat)
    print_label_selection_diagnostics("Adaptive", Selective_method)
except Exception as diag_err:
    # Diagnostics must never abort the actual experiment run/result write.
    print(f"[diagnostic print failed, continuing run] {diag_err}")

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

## Adaptive1
print("--- Running Adaptive1 ---")
Selective_method = DiagnosticAdaptiveSelection(alpha=alpha, random_state=seed)
C_sets_adaptive_1, k_hat_1 = Selective_method.multiclass_classification(
    X_calib, y_calib, X_test, bbox_mc, idx_list, conditional=conditional_
)
try:
    print_k_hat_diagnostics("Adaptive1", k_hat_1)
    print_label_selection_diagnostics("Adaptive1", Selective_method)
except Exception as diag_err:
    print(f"[diagnostic print failed, continuing run] {diag_err}")

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

shutil.rmtree(modeldir, ignore_errors=True)
print("Done.")