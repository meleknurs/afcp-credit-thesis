"""
Joint training/calibration sample-size sweep for AFCP on Give Me Some Credit.

For each sample size n_train_calib, trains the classifier, calibrates all
methods (Marginal, Partial, Exhaustive, AFCP-Adaptive, Adaptive1) and records
coverage, set size and AFCP's attribute-selection rate for the age attribute
on a fixed test set. Backs manuscript Table 4; the 5-seed combined summary is
written by combine_sweep_results.py.
"""
import os, sys
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random
import shutil
import tempfile
import time
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
AFCP_ROOT = os.path.join(REPO_ROOT, "AFCP")
if AFCP_ROOT not in sys.path:
    sys.path.insert(0, AFCP_ROOT)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- original Zhou & Sesia code ---
from third_party.utils import *
from SelectiveCI_fairness.methods import Marginal_Selection, Exhaustive_Selection, Partial_Selection, Adaptive_Selection
from SelectiveCI_fairness.black_box import Blackbox
from SelectiveCI_fairness.networks import ClassNNet
from SelectiveCI_fairness.evals import eval_psets, eval_by_att_mc

#########################
# Experiment parameters #
#########################
# Sample-size sweep: run the SAME dataset (GMSC) at increasing calibration
# sizes and record whether the REAL AFCP selects the age attribute at each N.
# This isolates sample size as the variable (same signal, growing power),
# giving a clean test of the sample-size sensitivity claim.
# Keep n_test small and fixed so per-sample AFCP inference cost stays bounded;
# only training/calibration size grows.
N_TRAIN_CALIB_LIST = [4000, 8000, 12000, 16000, 20000, 40000]
n_test = 1000

lr = 0.0001
n_epoch = 200
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2023
ttest_delta = 0
conditional_ = True

num_workers = 0
batch_size = 25
beta = 0
alpha = 0.1

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

###############
# Output file #
###############
outdir = os.path.join(SCRIPT_DIR, "results", "give_me_credit", "sample_size_sweep", "raw")
os.makedirs(outdir, exist_ok=True)
sweep_outfile = os.path.join(outdir, f"sweep_ntest{n_test}_seed{seed}.txt")
summary_outfile = os.path.join(outdir, f"sweep_summary_ntest{n_test}_seed{seed}.txt")
print("Sweep detail file:  {:s}".format(sweep_outfile))
print("Sweep summary file: {:s}".format(summary_outfile))

#########################
# Load Give Me Some Credit (once) #
#########################
DATA_PATH = os.path.join(REPO_ROOT, "data", "cs-training.csv")
assert os.path.exists(DATA_PATH), f"Data not found: {DATA_PATH} -- fix the path"

df_full = pd.read_csv(DATA_PATH, index_col=0)
df_full = df_full.dropna().reset_index(drop=True)
print(f"Dataset size after dropping NAs: {len(df_full)}")

y_col = "SeriousDlqin2yrs"


def build_xy(df):
    """Return X (age column overwritten with group label), y, and age column index."""
    y = df[y_col].values.astype(int)
    feature_df = df.drop(columns=[y_col])
    feature_cols = feature_df.columns.tolist()
    age_idx = feature_cols.index("age")
    X = feature_df.values.astype(float)
    # Bin age into 3 groups: young (<=35)->0, middle (35-55)->1, old (>55)->2
    age_groups = pd.cut(feature_df["age"], bins=[-1, 35, 55, np.inf], labels=[0, 1, 2]).astype(int).values
    X[:, age_idx] = age_groups
    return X, y, age_idx


def run_one(n_train_calib):
    """Run the full AFCP pipeline at a given calibration size. Return result rows."""
    print("\n" + "=" * 60)
    print(f"RUN n_train_calib = {n_train_calib}")
    print("=" * 60)

    N_DATA = n_train_calib + n_test
    if N_DATA > len(df_full):
        print(f"  SKIP: requested {N_DATA} rows but only {len(df_full)} available")
        return None

    df = df_full.sample(n=N_DATA, random_state=seed).reset_index(drop=True)
    X, y, age_idx = build_xy(df)
    idx_list = [age_idx]

    # Split: train_calib vs test, then train vs calib
    X_tc, X_test, y_tc, y_test = train_test_split(
        X, y, train_size=n_train_calib, random_state=seed, stratify=y)
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_tc, y_tc, train_size=0.5, random_state=seed, stratify=y_tc)

    # Scale non-sensitive columns only, fit on train
    non_sensitive = [i for i in range(X.shape[1]) if i != age_idx]
    scaler = StandardScaler()
    X_train[:, non_sensitive] = scaler.fit_transform(X_train[:, non_sensitive])
    X_calib[:, non_sensitive] = scaler.transform(X_calib[:, non_sensitive])
    X_test[:, non_sensitive] = scaler.transform(X_test[:, non_sensitive])

    print(f"  Split -- Train: {len(X_train)}, Calib: {len(X_calib)}, Test: {len(X_test)}")

    train_dataset = data_utils.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    calib_dataset = data_utils.TensorDataset(torch.Tensor(X_calib), torch.Tensor(y_calib))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def criterion(outputs, inputs, targets):
        targets = targets.to(torch.long)
        return Loss(outputs, targets)

    net = ClassNNet(num_features=X_train.shape[1], num_classes=len(set(y_train)), device=device, use_dropout=False)
    Loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Model checkpoints are written to the system temp dir rather than the
    # repository folder, to avoid I/O timeouts on cloud-synced folders.
    modeldir = os.path.join(tempfile.gettempdir(), "afcp_sweep_models", f"seed{seed}", f"n{n_train_calib}")
    bbox_mc = Blackbox(net, device, train_loader, batch_size=batch_size, max_epoch=n_epoch,
                       learning_rate=lr, val_loader=calib_loader, criterion=criterion,
                       optimizer=optimizer, verbose=False)

    save_attempts = 3
    for attempt in range(1, save_attempts + 1):
        try:
            bbox_mc.full_train(save_dir=modeldir)
            break
        except RuntimeError as e:
            if attempt == save_attempts:
                raise
            print(f"  WARNING: checkpoint save failed (attempt {attempt}/{save_attempts}): {e}")
            print("  Retrying in 5s...")
            time.sleep(5)

    selected_col = f"selected_{age_idx}"

    def complete_df(df, method=None, is_marg_results=False, is_selective=False, k_hat=None):
        df["n_train_calib"] = n_train_calib
        df["n_calib"] = len(X_calib)
        df["seed"] = seed
        df["lr"] = lr
        df["alpha"] = alpha
        df["method"] = method
        if is_marg_results:
            df["n_count"] = len(y_test)
            df["attribute_idx"] = -1
            df["attribute_level"] = -1
        if is_selective is False:
            df[selected_col] = None
        elif is_selective:
            assert k_hat is not None
            df[selected_col] = np.mean([1 if x == [age_idx] else 0 for x in k_hat])
        return df

    rows = pd.DataFrame()

    # Adaptive (real AFCP)
    m = Adaptive_Selection(alpha=alpha, random_state=seed, ttest_delta=ttest_delta)
    C_sets, k_hat = m.multiclass_classification(X_calib, y_calib, X_test, bbox_mc, idx_list, conditional=conditional_)
    r_marg = complete_df(eval_psets(C_sets, y_test), method='Adaptive', is_marg_results=True, is_selective=True, k_hat=k_hat)
    r_cond = complete_df(eval_by_att_mc(X_test, y_test, idx_list, C_sets), method='Adaptive', is_selective=True, k_hat=k_hat)
    rows = pd.concat([rows, r_marg, r_cond])
    selected_rate = np.mean([1 if x == [age_idx] else 0 for x in k_hat])
    print(f"  --> AFCP selected age at rate: {selected_rate:.3f}")

    # Marginal (baseline reference)
    m = Marginal_Selection(alpha=alpha, random_state=seed)
    C_sets = m.multiclass_classification(X_test, X_calib, y_calib, left_tail=False, bbox_mc=bbox_mc, conditional=conditional_)
    r_marg = complete_df(eval_psets(C_sets, y_test), method='Marginal', is_marg_results=True)
    r_cond = complete_df(eval_by_att_mc(X_test, y_test, idx_list, C_sets), method='Marginal')
    rows = pd.concat([rows, r_marg, r_cond])

    shutil.rmtree(modeldir, ignore_errors=True)
    return rows, selected_rate


#########################
# Run the sweep         #

#########################
# Resume support: if a previous run already produced results for this seed,
# load them and skip any n_train_calib sizes that already completed
# successfully (status != SKIPPED/FAILED), instead of recomputing from
# scratch after a crash or interruption.
all_rows = pd.DataFrame()
summary = []
done_sizes = set()

if os.path.exists(summary_outfile) and os.path.exists(sweep_outfile):
    try:
        prev_summary = pd.read_csv(summary_outfile)
        prev_rows = pd.read_csv(sweep_outfile)
        ok_mask = ~prev_summary["afcp_selected_age"].astype(str).isin(["SKIPPED", "FAILED"])
        done_sizes = set(prev_summary.loc[ok_mask, "n_train_calib"].tolist())
        if done_sizes:
            print(f"Resuming: found completed sizes {sorted(done_sizes)} in existing output, skipping them.")
            all_rows = prev_rows
            summary = prev_summary.to_dict("records")
    except Exception as e:
        print(f"Could not parse existing results for resume ({e}); starting fresh.")

for n_tc in N_TRAIN_CALIB_LIST:
    if n_tc in done_sizes:
        print(f"Skipping n_train_calib={n_tc} (already completed in a previous run).")
        continue
    try:
        out = run_one(n_tc)
    except RuntimeError as e:
        print(f"  FAILED n_train_calib={n_tc} after retries: {e}")
        summary.append({"n_train_calib": n_tc, "n_calib": None, "afcp_selected_age": "FAILED"})
        pd.DataFrame(summary).to_csv(summary_outfile, index=False)
        continue
    if out is None:
        summary.append({"n_train_calib": n_tc, "n_calib": None, "afcp_selected_age": "SKIPPED"})
        continue
    rows, selected_rate = out
    all_rows = pd.concat([all_rows, rows])
    n_calib = int(rows["n_calib"].iloc[0])
    summary.append({"n_train_calib": n_tc, "n_calib": n_calib, "afcp_selected_age": selected_rate})
    # Write incrementally so partial progress is saved if interrupted
    all_rows.to_csv(sweep_outfile, index=False)
    pd.DataFrame(summary).to_csv(summary_outfile, index=False)

print("\n" + "=" * 60)
print("SWEEP SUMMARY (does real AFCP select age as N grows?)")
print("=" * 60)
print(pd.DataFrame(summary).to_string(index=False))
print(f"\nDetail written to: {sweep_outfile}")
print(f"Summary written to: {summary_outfile}")