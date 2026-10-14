"""
Fast "disparity meter" for semi-synthetic bias injection on GMSC.

Trains the same classifier as mc_give_me_credit.py and reports, WITHOUT the
slow AFCP full-conformal loop:
  * marginal split-conformal coverage / set size per age group on a LARGE
    test set (default 5000) and on the calibration set (leave-one-out),
  * the AFCP selection test, reproduced analytically: for each label y, the
    leave-one-out miscoverage indicators of the label-y calibration subset,
    the worst age group, and the one-sample t-test p-value against alpha
    (AFCP selects an attribute only if the test rejects for every label),
  * model-confidence diagnostics for the target group (mean p(default) for
    true defaulters, singleton rate).

Usage:
    python bias_meter.py <out_csv> <seed[,seed,...]> <n_epoch> <config>[,<config>...]
    config = <bias_type>:<bias_group>:<bias_eps>   e.g. none:0:0  label:0:0.3  measure:0:1.0

Example:
    python bias_meter.py results/give_me_credit/bias_meter/meter_seed2023.csv 2023 500 \
        none:0:0,label:0:0.3,label:0:0.9,measure:0:0.5,measure:0:1.0,represent:0:0.9
"""
import os, sys, random, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
import torch.utils.data as data_utils
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "AFCP"))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from third_party.utils import nonconf_scores_mc, arc_wrapper
from SelectiveCI_fairness.black_box import Blackbox
from SelectiveCI_fairness.networks import ClassNNet

out_csv = sys.argv[1]
seeds = [int(s) for s in sys.argv[2].split(",")]
n_epoch = int(sys.argv[3])
configs = [c.split(":") for c in sys.argv[4].split(",")]
n_train_calib, n_test, lr, batch_size, alpha = 12000, 5000, 0.0001, 25, 0.1
DELINQ_COLS = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate"]

df0 = pd.read_csv(os.path.join(REPO_ROOT, "data", "cs-training.csv"))
if "Unnamed: 0" in df0.columns: df0 = df0.drop(columns=["Unnamed: 0"])
df0 = df0[df0["age"] > 0].reset_index(drop=True)

def run(seed, bias_type, bias_group, bias_eps):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    y = df0["SeriousDlqin2yrs"].astype(int); X = df0.drop(columns=["SeriousDlqin2yrs"]).copy()
    X["age_bin_median"] = (X["age"] <= X["age"].median()).astype(int)
    X["age_bin_40"] = (X["age"] < 40).astype(int)
    X["age_bin_3way"] = pd.cut(X["age"], bins=[0, 35, 55, 150], labels=[0, 1, 2], include_lowest=True).astype(int)
    inc_med = X["MonthlyIncome"].median()
    X["income_bin"] = (X["MonthlyIncome"].fillna(inc_med) <= inc_med).astype(int)
    X["dependents_bin"] = (X["NumberOfDependents"].fillna(0) > 0).astype(int)
    g_idx = list(X.columns).index("age_bin_3way")

    idx = random.sample(range(X.shape[0]), n_train_calib + n_test)
    X_full = X.iloc[idx].reset_index(drop=True); y_full = y.iloc[idx].reset_index(drop=True)

    rng = np.random.RandomState(seed)
    in_group = (X_full["age_bin_3way"].values == bias_group)
    if bias_eps > 0 and bias_type in ("label", "measure"):
        corrupt = in_group & (rng.uniform(size=len(y_full)) < bias_eps)
        if bias_type == "label":
            y_full = y_full.copy(); y_full.loc[corrupt] = rng.binomial(1, 0.5, size=corrupt.sum())
        else:
            X_full = X_full.copy(); X_full.loc[corrupt, DELINQ_COLS] = 0

    X_tc, X_te, y_tc, y_te = train_test_split(X_full, y_full, train_size=n_train_calib, random_state=seed, stratify=y_full.astype(int))
    X_tr, X_ca, y_tr, y_ca = train_test_split(X_tc, y_tc, train_size=0.5, random_state=seed, stratify=y_tc.astype(int))
    if bias_eps > 0 and bias_type == "represent":
        drop = (X_tr["age_bin_3way"].values == bias_group) & (rng.uniform(size=len(X_tr)) < bias_eps)
        X_tr, y_tr = X_tr.loc[~drop], y_tr.loc[~drop]
    X_tr, X_ca, X_te = X_tr.copy(), X_ca.copy(), X_te.copy()
    med = X_tr["MonthlyIncome"].median()
    for d in (X_tr, X_ca, X_te):
        d["MonthlyIncome"] = d["MonthlyIncome"].fillna(med); d["NumberOfDependents"] = d["NumberOfDependents"].fillna(0)
    sens = ["age_bin_median", "age_bin_40", "age_bin_3way", "income_bin", "dependents_bin"]
    cont = [c for c in X_tr.columns if c not in sens]
    sc = StandardScaler(); X_tr[cont] = sc.fit_transform(X_tr[cont]); X_ca[cont] = sc.transform(X_ca[cont]); X_te[cont] = sc.transform(X_te[cont])
    X_tr, X_ca, X_te = [d.values.astype("float64") for d in (X_tr, X_ca, X_te)]
    y_tr, y_ca, y_te = [d.values.astype("float64") for d in (y_tr, y_ca, y_te)]

    device = torch.device("cpu")
    tl = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.Tensor(X_tr), torch.Tensor(y_tr)), batch_size=batch_size)
    cl = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.Tensor(X_ca), torch.Tensor(y_ca)), batch_size=batch_size, shuffle=False)
    net = ClassNNet(num_features=X_tr.shape[1], num_classes=2, device=device, use_dropout=False)
    Loss = nn.CrossEntropyLoss()
    crit = lambda outputs, inputs, targets: Loss(outputs, targets.to(torch.long))
    bbox = Blackbox(net, device, tl, batch_size=batch_size, max_epoch=n_epoch, learning_rate=lr, val_loader=cl,
                    criterion=crit, optimizer=optim.Adam(net.parameters(), lr=lr), verbose=False)
    t0 = time.time(); bbox.full_train(save_dir=os.path.join(SCRIPT_DIR, "models", "bias_meter", f"tmp_seed{seed}")); t_train = time.time() - t0

    # ---- marginal split CP on the big test set
    cs = nonconf_scores_mc(X_ca, y_ca, bbox, alpha=alpha, random_state=seed)
    C_te = arc_wrapper(cs, X_te, bbox, alpha, seed)
    cov_te = np.array([y_te[i] in C_te[i] for i in range(len(y_te))]); size_te = np.array([len(c) for c in C_te])
    g_te = X_te[:, g_idx].astype(int)
    p_te = bbox.net.predict_prob(torch.from_numpy(X_te).float())[:, 1]

    row = dict(seed=seed, bias_type=bias_type, bias_group=bias_group, bias_eps=bias_eps, n_epoch=n_epoch, n_test=n_test,
               train_sec=round(t_train, 1), test_cov_overall=cov_te.mean(), test_size_overall=size_te.mean())
    for g in (0, 1, 2):
        m = g_te == g
        row[f"test_n_g{g}"] = int(m.sum()); row[f"test_cov_g{g}"] = cov_te[m].mean(); row[f"test_size_g{g}"] = size_te[m].mean()
        row[f"test_singleton_g{g}"] = (size_te[m] == 1).mean()
        dflt = m & (y_te == 1)
        row[f"test_cov_defaulters_g{g}"] = cov_te[dflt].mean() if dflt.sum() else np.nan
        row[f"test_mean_pdefault_defaulters_g{g}"] = p_te[dflt].mean() if dflt.sum() else np.nan
    covs = [row[f"test_cov_g{g}"] for g in (0, 1, 2)]
    row["test_worst_cov"] = min(covs); row["test_gap"] = max(covs) - min(covs)

    # ---- AFCP selection test, reproduced on the calibration set (label-conditional, leave-one-out)
    g_ca = X_ca[:, g_idx].astype(int)
    for lab in (0, 1):
        sel = y_ca == lab; s = cs[sel]; g = g_ca[sel]; n = len(s)
        # leave-one-out: point j miscovered if its score exceeds the (1-alpha)(1+1/(n-1)) quantile of the others
        order = np.sort(s); k = int(np.ceil((1 - alpha) * n)) - 1  # index of the LOO threshold in the sorted others (approx.)
        thr_loo = np.where(np.arange(n) <= k, order[min(k + 1, n - 1)], order[k])  # threshold for each rank position
        rank = np.argsort(np.argsort(s)); E = s > thr_loo[rank]
        worst, worst_mis, pval = None, -1, None
        for gg in (0, 1, 2):
            mm = g == gg
            if mm.sum() == 0: continue
            mis = E[mm].mean()
            if mis >= worst_mis: worst_mis, worst = mis, gg
        E_w = E[g == worst].astype(float)
        pval = stats.ttest_1samp(E_w, alpha, alternative="greater").pvalue if len(E_w) > 1 else np.nan
        row[f"calib_n_label{lab}"] = n; row[f"calib_worst_group_label{lab}"] = worst
        row[f"calib_worst_miscov_label{lab}"] = worst_mis; row[f"calib_ttest_p_label{lab}"] = pval
        for gg in (0, 1, 2):
            row[f"calib_miscov_g{gg}_label{lab}"] = E[g == gg].mean() if (g == gg).sum() else np.nan
    row["afcp_would_select"] = int((row["calib_ttest_p_label0"] < 0.1) and (row["calib_ttest_p_label1"] < 0.1)
                                   and row["calib_worst_group_label0"] == row["calib_worst_group_label1"])
    return row

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
rows = []
for seed in seeds:
    for bt, bg, be in configs:
        t0 = time.time()
        r = run(seed, bt, int(bg), float(be)); rows.append(r)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[{time.time()-t0:5.0f}s] seed={seed} {bt} g{bg} eps={be}: test cov g0/g1/g2 = "
              f"{r['test_cov_g0']:.3f}/{r['test_cov_g1']:.3f}/{r['test_cov_g2']:.3f}  gap={r['test_gap']:.3f}  "
              f"calib worst miscov (lab0,lab1)=({r['calib_worst_miscov_label0']:.3f},{r['calib_worst_miscov_label1']:.3f}) "
              f"p=({r['calib_ttest_p_label0']:.3f},{r['calib_ttest_p_label1']:.3f}) select={r['afcp_would_select']}", flush=True)
print("done ->", out_csv)
