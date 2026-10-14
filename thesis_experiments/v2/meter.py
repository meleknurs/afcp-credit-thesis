"""
v2 fast "disparity meter": split-conformal metrics on a large test set plus an
analytic reproduction of AFCP's selection test, for any dataset x model x bias
configuration, WITHOUT the full-conformal AFCP loop (docs/V2_DESIGN.md).

Usage:
  python v2/meter.py <dataset> <model> <out_csv> <seeds> <configs> [n_train n_calib n_test]
    dataset  sgc | gmsc | taiwan
    model    nn | lr
    seeds    e.g. 2023,2024
    configs  comma-separated  <bias_type>:<group_col>:<level>:<eps>:<scope>
             bias_type  none | label (good->bad) | labelneg (bad->good) | represent | measure(gmsc only)
             group_col  a group column of the dataset (e.g. age_3way, sex_female)
             level      the level to bias (target group)
             eps        strength
             scope      train | traincal   (test is always clean)
    e.g.  none:age_3way:0:0:train,label:age_3way:0:0.1:traincal

Outputs one row per (seed, config) with:
  - model diagnostics (AUROC, log-loss, best_epoch, degeneracy shares)
  - marginal-regime split CP on the test set: overall / per-group / per-(group x label)
    coverage, set size, singleton and {0,1} rates
  - label-conditional-regime split CP (per-label quantiles), same metrics
  - AFCP selection test reproduced on the calibration set (LOO miscoverage,
    label-conditional as in the code): worst group, miscoverage, p-value per label,
    indicator variance, `afcp_would_select`
"""
import os, sys, time, random
import numpy as np, pandas as pd, torch
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "..", "AFCP"))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if not hasattr(np, "Inf"):  # NumPy >= 2 removed the alias used by the unmodified AFCP code
    np.Inf = np.inf
from v2.data import LOADERS
from v2.models import FITTERS, degeneracy_report, predict_prob
from third_party.utils import nonconf_scores_mc, ProbAccum

ALPHA = 0.1
DEFAULT_SIZES = {"sgc": (350, 350, 300), "gmsc": (6000, 6000, 5000), "taiwan": (6000, 6000, 10000)}
GMSC_DELINQ = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate"]


def parse_config(c):
    bt, gc, lvl, eps, scope = c.split(":")
    return dict(bias_type=bt, group_col=gc, level=int(lvl), eps=float(eps), scope=scope)


def inject(X, y, cfg, rng, is_train):
    """Return possibly modified copies. One-sided label bias: good -> bad in the target group."""
    X = X.copy(); y = y.copy()
    if cfg["eps"] <= 0 or cfg["bias_type"] == "none":
        return X, y
    tgt = (X[cfg["group_col"]].values == cfg["level"])
    if cfg["bias_type"] == "label":          # "stigma": good -> bad in the target group
        cand = tgt & (y.values == 0)
        flip = cand & (rng.uniform(size=len(y)) < cfg["eps"])
        y.iloc[np.where(flip)[0]] = 1
    elif cfg["bias_type"] == "labelneg":     # "hidden defaults": bad -> good in the target group (Fogliato et al. 2020)
        cand = tgt & (y.values == 1)
        flip = cand & (rng.uniform(size=len(y)) < cfg["eps"])
        y.iloc[np.where(flip)[0]] = 0
    elif cfg["bias_type"] == "measure":
        hit = tgt & (rng.uniform(size=len(y)) < cfg["eps"])
        X.loc[hit, GMSC_DELINQ] = 0
    elif cfg["bias_type"] == "represent":
        if is_train:
            drop = tgt & (rng.uniform(size=len(y)) < cfg["eps"])
            X, y = X.loc[~drop], y.loc[~drop]
    return X, y


def split_cp_sets(scores_cal, scores_test, alpha):
    """Split CP with ARC-style finite-sample level; scores_test: (n_test, 2). Returns boolean (n_test,2)."""
    n = len(scores_cal)
    lvl = min(1.0, (1 - alpha) * (1 + 1 / n))
    q = np.quantile(np.append(scores_cal, np.inf), lvl, method="higher")
    return scores_test <= q, q


def loo_miscoverage(s, alpha):
    n = len(s); order = np.argsort(s); rank = np.empty(n, int); rank[order] = np.arange(n)
    k = int(np.ceil((1 - alpha) * n)) - 1  # threshold index among the n-1 others ≈ same k for large n
    srt = s[order]
    thr = np.where(rank <= k, srt[min(k + 1, n - 1)], srt[k])
    return s > thr


def run_one(ds, model_name, seed, cfg, sizes):
    X_all, y_all, meta = LOADERS[ds]()
    n_tr, n_cal, n_te = sizes
    rng = np.random.RandomState(seed); random.seed(seed); torch.manual_seed(seed)
    idx = random.sample(range(len(y_all)), n_tr + n_cal + n_te)
    X = X_all.iloc[idx].reset_index(drop=True); y = y_all.iloc[idx].reset_index(drop=True)
    X_tc, X_te, y_tc, y_te = train_test_split(X, y, test_size=n_te, random_state=seed, stratify=y)
    X_tr, X_ca, y_tr, y_ca = train_test_split(X_tc, y_tc, test_size=n_cal, random_state=seed, stratify=y_tc)
    # same partition for every bias level within a seed; injection after the split
    X_tr, y_tr = inject(X_tr, y_tr, cfg, np.random.RandomState(seed + 1), is_train=True)
    if cfg["scope"] == "traincal":
        X_ca, y_ca = inject(X_ca, y_ca, cfg, np.random.RandomState(seed + 2), is_train=False)
    # imputation per split with training statistics
    for col, how in meta.get("impute", {}).items():
        v = X_tr[col].median() if how == "median" else how
        for d in (X_tr, X_ca, X_te): d[col] = d[col].fillna(v)
    cont = meta["continuous_cols"]; sc = StandardScaler().fit(X_tr[cont])
    for d in (X_tr, X_ca, X_te): d[cont] = sc.transform(d[cont])
    gcols = meta["group_cols"]; colnames = list(X_tr.columns)
    Xtr, Xca, Xte = [d.values.astype("float64") for d in (X_tr, X_ca, X_te)]
    ytr, yca, yte = [d.values.astype(int) for d in (y_tr, y_ca, y_te)]

    t0 = time.time(); model = FITTERS[model_name](Xtr, ytr.astype(float), seed=seed); t_fit = time.time() - t0
    groups_ca = {g: X_ca[g].values.astype(int) for g in gcols}; groups_te = {g: X_te[g].values.astype(int) for g in gcols}
    row = dict(dataset=ds, model=model_name, seed=seed, **cfg, n_train=len(ytr), n_calib=len(yca), n_test=len(yte),
               fit_sec=round(t_fit, 1), best_epoch=model.info.get("best_epoch", np.nan))
    row.update({f"cal_{k}": v for k, v in degeneracy_report(model, Xca, yca, groups_ca).items()})

    # scores
    s_cal = nonconf_scores_mc(Xca, yca, model, alpha=ALPHA, random_state=seed)
    s_te = np.column_stack([nonconf_scores_mc(Xte, np.full(len(yte), lab), model, alpha=ALPHA, random_state=seed) for lab in (0, 1)])
    p_te = predict_prob(model, Xte)[:, 1]

    def metrics(prefix, sets):
        cov = sets[np.arange(len(yte)), yte]; size = sets.sum(1)
        row[f"{prefix}_cov"] = cov.mean(); row[f"{prefix}_size"] = size.mean()
        row[f"{prefix}_singleton"] = (size == 1).mean(); row[f"{prefix}_both"] = (size == 2).mean()
        for lab in (0, 1):
            row[f"{prefix}_cov_label{lab}"] = cov[yte == lab].mean()
        for g, arr in groups_te.items():
            covs = []
            for lvl in np.unique(arr):
                m = arr == lvl; covs.append(cov[m].mean())
                row[f"{prefix}_cov_{g}_{lvl}"] = cov[m].mean(); row[f"{prefix}_size_{g}_{lvl}"] = size[m].mean()
                row[f"{prefix}_singleton_{g}_{lvl}"] = (size[m] == 1).mean()
                for lab in (0, 1):
                    mm = m & (yte == lab); row[f"{prefix}_cov_{g}_{lvl}_label{lab}"] = cov[mm].mean() if mm.any() else np.nan
            row[f"{prefix}_worst_{g}"] = min(covs); row[f"{prefix}_gap_{g}"] = max(covs) - min(covs)

    sets_m, q = split_cp_sets(s_cal, s_te, ALPHA); metrics("marg", sets_m); row["marg_q"] = q
    sets_c = np.zeros_like(sets_m)
    for lab in (0, 1):
        sets_c[:, lab], _ = split_cp_sets(s_cal[yca == lab], s_te[:, lab], ALPHA)
    metrics("cond", sets_c)

    # AFCP selection test, label-conditional LOO on the calibration set, for each candidate group col
    for g, arr in groups_ca.items():
        flags = []
        for lab in (0, 1):
            sel = yca == lab; s = s_cal[sel]; gg = arr[sel]
            E = loo_miscoverage(s, ALPHA)
            worst, worst_mis = None, -1
            for lvl in np.unique(gg):
                mis = E[gg == lvl].mean()
                if mis >= worst_mis: worst_mis, worst = mis, lvl
            Ew = E[gg == worst].astype(float)
            p = stats.ttest_1samp(Ew, ALPHA, alternative="greater").pvalue if Ew.std() > 0 else 1.0
            row[f"sel_{g}_label{lab}_worst"] = worst; row[f"sel_{g}_label{lab}_miscov"] = worst_mis
            row[f"sel_{g}_label{lab}_p"] = p; row[f"sel_{g}_label{lab}_Evar"] = Ew.var()
            flags.append((p < 0.1, worst))
        row[f"sel_{g}_would_select"] = int(flags[0][0] and flags[1][0] and flags[0][1] == flags[1][1])
        row[f"sel_{g}_anylabel"] = int(flags[0][0] or flags[1][0])
    return row


if __name__ == "__main__":
    ds, model_name, out_csv, seeds, configs = sys.argv[1:6]
    sizes = tuple(int(v) for v in sys.argv[6:9]) if len(sys.argv) >= 9 else DEFAULT_SIZES[ds]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    rows = []
    for seed in [int(s) for s in seeds.split(",")]:
        for c in configs.split(","):
            cfg = parse_config(c); t0 = time.time()
            r = run_one(ds, model_name, seed, cfg, sizes); rows.append(r)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            g = cfg["group_col"]
            print(f"[{time.time()-t0:4.0f}s] {ds} {model_name} seed={seed} {c}: AUC={r['cal_auroc']:.3f} ep={r['best_epoch']} "
                  f"deg_pos={r['cal_share_ptrue_lt_tol_pos']:.3f} | marg cov={r['marg_cov']:.3f} size={r['marg_size']:.2f} "
                  f"worst_{g}={r[f'marg_worst_{g}']:.3f} gap={r[f'marg_gap_{g}']:.3f} | cond size={r['cond_size']:.2f} "
                  f"| sel p=({r[f'sel_{g}_label0_p']:.3f},{r[f'sel_{g}_label1_p']:.3f}) select={r[f'sel_{g}_would_select']}", flush=True)
    print("done ->", out_csv)
