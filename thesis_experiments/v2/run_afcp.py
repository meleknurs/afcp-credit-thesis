"""
v2 full conformal run: Marginal, Partial, Exhaustive, AFCP (Adaptive) and optionally
Adaptive1, on one dataset x model x seed x bias configuration, using the unmodified
Zhou & Sesia implementation in AFCP/.

Usage:
  python v2/run_afcp.py <dataset> <model> <seed> <config> <conditional> [options]
    config       <bias_type>:<group_col>:<level>:<eps>:<scope>   (see v2/meter.py; none:...:0:0:train for baseline)
    conditional  True | False     (label-conditional vs marginal calibration regime)
  options (key=value): n_train n_calib n_test ttest_delta(0) sig_level(0.1) adaptive1(0|1) out_dir

Output: results/v2/afcp/<dataset>/<file>.txt in the Phase-1 column format, plus
columns model, regime, bias_*, ttest_delta, sig_level, best_epoch, cal_auroc,
selected_<name>, selected_none, and per-label flag rates for Adaptive.
"""
import os, sys, time, random
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "..", "AFCP"))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if not hasattr(np, "Inf"):  # NumPy >= 2 removed the alias used by the unmodified AFCP code
    np.Inf = np.inf
from v2.data import LOADERS
from v2.models import FITTERS, degeneracy_report
from v2.meter import parse_config, inject, DEFAULT_SIZES
from SelectiveCI_fairness.methods import Marginal_Selection, Exhaustive_Selection, Partial_Selection, Adaptive_Selection
from SelectiveCI_fairness.evals import eval_psets, eval_by_att_mc

ALPHA = 0.1


class LoggingAdaptive(Adaptive_Selection):
    """Logs each per-label selection (unchanged behaviour), to report per-label flag rates."""
    def __init__(self, *a, sig_level=0.1, **k):
        super().__init__(*a, **k); self.sig_level = sig_level; self.log = []
    def select_the_worst_group(self, err_func, att_idx, E, X_aug):
        k_hat = super().select_the_worst_group(err_func, att_idx, E, X_aug); self.log.append(k_hat); return k_hat


def main():
    ds, model_name, seed, cfg_s, conditional = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5] == "True"
    ds_full = ds; ds, _, cand_override = ds.partition("@")   # e.g. sgc@age_le33  or  gmsc@age_lt25 : override the AFCP candidate attribute(s), "+" separated
    ds_tag = ds_full.replace("@", "-").replace("+", "_")
    opts = dict(kv.split("=") for kv in sys.argv[6:])
    sizes = DEFAULT_SIZES[ds][:2] + (1000,)
    n_tr, n_cal, n_te = int(opts.get("n_train", sizes[0])), int(opts.get("n_calib", sizes[1])), int(opts.get("n_test", sizes[2]))
    ttest_delta = float(opts.get("ttest_delta", 0)); sig_level = float(opts.get("sig_level", 0.1))
    run_adaptive1 = opts.get("adaptive1", "0") == "1"
    cfg = parse_config(cfg_s)
    out_dir = opts.get("out_dir", os.path.join(os.path.dirname(HERE), "results", "v2", "afcp", "rq4_groupdef" if cand_override else ds)); os.makedirs(out_dir, exist_ok=True)
    tag = f"{ds_tag}_{model_name}_ntr{n_tr}_ncal{n_cal}_nte{n_te}_cond{conditional}_td{ttest_delta}_sl{sig_level}_seed{seed}"
    if cfg["eps"] > 0: tag += f"_bias{cfg['bias_type']}_{cfg['group_col']}{cfg['level']}_eps{cfg['eps']}_{cfg['scope']}"
    outfile = os.path.join(out_dir, tag + ".txt"); print("Output:", outfile)

    X_all, y_all, meta = LOADERS[ds]()
    if cand_override:
        meta = dict(meta); meta["candidate_cols"] = cand_override.split("+")
        assert all(c in X_all.columns for c in meta["candidate_cols"]), f"unknown candidate column in {cand_override}"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    idx = random.sample(range(len(y_all)), n_tr + n_cal + n_te)
    X = X_all.iloc[idx].reset_index(drop=True); y = y_all.iloc[idx].reset_index(drop=True)
    X_tc, X_te, y_tc, y_te = train_test_split(X, y, test_size=n_te, random_state=seed, stratify=y)
    X_tr, X_ca, y_tr, y_ca = train_test_split(X_tc, y_tc, test_size=n_cal, random_state=seed, stratify=y_tc)
    X_tr, y_tr = inject(X_tr, y_tr, cfg, np.random.RandomState(seed + 1), is_train=True)
    if cfg["scope"] == "traincal":
        X_ca, y_ca = inject(X_ca, y_ca, cfg, np.random.RandomState(seed + 2), is_train=False)
    for col, how in meta.get("impute", {}).items():
        v = X_tr[col].median() if how == "median" else how
        for d in (X_tr, X_ca, X_te): d[col] = d[col].fillna(v)
    cont = meta["continuous_cols"]; sc = StandardScaler().fit(X_tr[cont])
    for d in (X_tr, X_ca, X_te): d[cont] = sc.transform(d[cont])
    cols = list(X_tr.columns)
    cand = meta["candidate_cols"]; idx_list = [cols.index(c) for c in cand]
    eval_cols = meta["group_cols"]; eval_idx = [cols.index(c) for c in eval_cols]
    Xtr, Xca, Xte = [d.values.astype("float64") for d in (X_tr, X_ca, X_te)]
    ytr, yca, yte = [d.values.astype("float64") for d in (y_tr, y_ca, y_te)]
    print(f"split: train {len(ytr)} calib {len(yca)} test {len(yte)}; candidates {cand} -> idx {idx_list}")

    t0 = time.time(); model = FITTERS[model_name](Xtr, ytr, seed=seed); print(f"fit {model_name}: {time.time()-t0:.0f}s, info {{k: v for k, v in model.info.items() if k != 'val_loss_history'}}")
    deg = degeneracy_report(model, Xca, yca.astype(int), {c: X_ca[c].values.astype(int) for c in cand})
    print("degeneracy:", {k: round(v, 4) for k, v in deg.items() if not k.startswith("deg_")})

    def finish(df, method, k_hat=None, logsel=None):
        df["n_data"] = n_tr + n_cal; df["n_train"] = n_tr; df["n_calib"] = n_cal; df["seed"] = seed; df["alpha"] = ALPHA
        df["method"] = method; df["conditional"] = conditional; df["model"] = model_name; df["dataset"] = ds; df["cand_def"] = cand_override or "default"
        df["run_config"] = cfg_s; df["bias_type"] = cfg["bias_type"]; df["bias_group"] = cfg["group_col"]; df["bias_level"] = cfg["level"]
        df["bias_eps"] = cfg["eps"]; df["bias_scope"] = cfg["scope"]; df["ttest_delta"] = ttest_delta; df["sig_level"] = sig_level
        df["best_epoch"] = model.info.get("best_epoch", np.nan); df["cal_auroc"] = deg["auroc"]; df["cal_deg_pos"] = deg["share_ptrue_lt_tol_pos"]
        if k_hat is not None:
            norm = [[int(v) for v in np.asarray(k, dtype=object).reshape(-1)] for k in k_hat]
            for i, c in zip(idx_list, cand):
                df[f"selected_{c}"] = np.mean([i in s for s in norm])
            df["selected_none"] = np.mean([len(s) == 0 for s in norm])
            if logsel is not None:
                L = logsel.log; nl = 2
                for lab in range(nl):
                    picks = L[lab::nl]
                    for i, c in zip(idx_list, cand):
                        df[f"flag_label{lab}_{c}"] = np.mean([(not isinstance(p, list)) and int(p) == i for p in picks])
        return df

    res = pd.DataFrame()
    def add(method, C, k_hat=None, logsel=None):
        nonlocal res
        m = eval_psets(C, yte); m["n_count"] = len(yte); m["attribute_idx"] = -1; m["attribute_level"] = -1
        c = eval_by_att_mc(Xte, yte, eval_idx, C)
        c["attribute_name"] = [eval_cols[eval_idx.index(int(a))] for a in c["attribute_idx"]]
        res = pd.concat([res, finish(m, method, k_hat, logsel), finish(c, method, k_hat, logsel)])
        print(f"  {method}: coverage {float(m.Coverage.iloc[0]):.3f} size {float(m.Length.iloc[0]):.3f}" +
              (f" selected_none {float(res.selected_none.iloc[-1]):.2f}" if k_hat is not None else ""), flush=True)

    t0 = time.time(); print("--- Adaptive"); S = LoggingAdaptive(alpha=ALPHA, random_state=seed, ttest_delta=ttest_delta, sig_level=sig_level)
    C, k = S.multiclass_classification(Xca, yca, Xte, model, idx_list, conditional=conditional); add("Adaptive", C, k, S)
    print(f"  ({time.time()-t0:.0f}s)")
    if run_adaptive1:
        t0 = time.time(); print("--- Adaptive1"); S1 = LoggingAdaptive(alpha=ALPHA, random_state=seed)
        C, k = S1.multiclass_classification(Xca, yca, Xte, model, idx_list, conditional=conditional); add("Adaptive1", C, k, S1); print(f"  ({time.time()-t0:.0f}s)")
    print("--- Marginal"); C = Marginal_Selection(alpha=ALPHA, random_state=seed).multiclass_classification(Xte, Xca, yca, left_tail=False, bbox_mc=model, conditional=conditional); add("Marginal", C)
    print("--- Partial"); C = Partial_Selection(alpha=ALPHA, random_state=seed).multiclass_classification(Xte, Xca, yca, left_tail=False, sensitive_atts_idx=idx_list, bbox_mc=model, conditional=conditional); add("Partial", C)
    print("--- Exhaustive"); C = Exhaustive_Selection(alpha=ALPHA, random_state=seed).multiclass_classification(Xte, Xca, yca, left_tail=False, sensitive_atts_idx=idx_list, bbox_mc=model, conditional=conditional); add("Exhaustive", C)
    res.to_csv(outfile, index=False); print("Results written to", outfile)


if __name__ == "__main__":
    main()
