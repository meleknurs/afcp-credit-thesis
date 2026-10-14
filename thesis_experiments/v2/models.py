"""
v2 classifiers (docs/V2_DESIGN.md, Section 3).

Both models expose the interface the AFCP code expects: an object with
`.net.predict_prob(torch_tensor) -> np.ndarray (n, 2)`.

- fit_nn: Zhou & Sesia's ClassNNet (256-256-128-64-2), Adam lr 1e-4, batch 25,
  dropout on, early stopping on the log-loss of a validation slice carved from
  the TRAINING data (never the calibration set), patience 20, best-epoch
  checkpoint, max 500 epochs.
- fit_lr: L2 logistic regression (scikit-learn), inputs standardised upstream.

degeneracy_report(model, X, y, groups): AUROC, log-loss, Brier, share of
points with p_hat(y_true) < 1e-6, and per (group x label) cell the share of
such points — the "is the model degenerate" check logged for every run.
"""
import os, sys, copy
import numpy as np, torch, torch.nn as nn, torch.optim as optim
import torch.utils.data as data_utils
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "AFCP"))
from SelectiveCI_fairness.networks import ClassNNet


class _Wrapped:
    """Minimal stand-in for AFCP's Blackbox: only `.net.predict_prob` is used by the CP code."""
    def __init__(self, net, info):
        self.net = net
        self.info = info


class _SkNet:
    def __init__(self, clf):
        self.clf = clf
    def predict_prob(self, inputs):
        x = inputs.cpu().numpy() if hasattr(inputs, "cpu") else np.asarray(inputs)
        return self.clf.predict_proba(x)


def fit_lr(X_tr, y_tr, seed=2023, C=1.0):
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    clf.fit(X_tr, y_tr.astype(int))
    return _Wrapped(_SkNet(clf), dict(model="lr", C=C))


def fit_nn(X_tr, y_tr, seed=2023, lr=1e-4, batch_size=25, max_epoch=500, patience=20,
           val_frac=0.15, use_dropout=True, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cpu")
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=val_frac, random_state=seed,
                                                  stratify=y_tr.astype(int))
    loader = data_utils.DataLoader(data_utils.TensorDataset(torch.Tensor(X_fit), torch.Tensor(y_fit)),
                                   batch_size=batch_size, shuffle=True)
    Xv = torch.Tensor(X_val); yv = torch.Tensor(y_val).long()
    net = ClassNNet(num_features=X_tr.shape[1], num_classes=2, device=device, use_dropout=use_dropout)
    opt = optim.Adam(net.parameters(), lr=lr); loss_fn = nn.CrossEntropyLoss()
    best, best_state, best_epoch, bad = np.inf, None, 0, 0
    hist = []
    for ep in range(1, max_epoch + 1):
        net.train()
        for xb, yb in loader:
            opt.zero_grad(); loss = loss_fn(net(xb), yb.long()); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net(Xv), yv))
        hist.append(vl)
        if vl < best - 1e-5:
            best, best_state, best_epoch, bad = vl, copy.deepcopy(net.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 10 == 0:
            print(f"epoch {ep} val_loss {vl:.4f}")
    net.load_state_dict(best_state); net.eval()
    return _Wrapped(net, dict(model="nn", best_epoch=best_epoch, epochs_run=ep, best_val_loss=best,
                              val_loss_history=hist, dropout=use_dropout))


FITTERS = {"nn": fit_nn, "lr": fit_lr}


def predict_prob(model, X):
    return model.net.predict_prob(torch.from_numpy(np.asarray(X)).float())


def degeneracy_report(model, X, y, groups=None, tol=1e-6):
    """groups: dict name -> integer array aligned with X."""
    p = predict_prob(model, X)
    p1 = p[:, 1]; yt = y.astype(int)
    p_true = np.where(yt == 1, p1, 1 - p1)
    rep = dict(auroc=roc_auc_score(yt, p1), logloss=log_loss(yt, np.clip(p1, 1e-12, 1 - 1e-12)),
               brier=brier_score_loss(yt, p1), share_ptrue_lt_tol=float((p_true < tol).mean()),
               share_ptrue_lt_tol_pos=float((p_true[yt == 1] < tol).mean()) if (yt == 1).any() else np.nan,
               mean_p1_pos=float(p1[yt == 1].mean()), mean_p1_neg=float(p1[yt == 0].mean()))
    for gname, g in (groups or {}).items():
        for lvl in np.unique(g):
            for lab in (0, 1):
                m = (g == lvl) & (yt == lab)
                rep[f"deg_{gname}_{lvl}_label{lab}"] = float((p_true[m] < tol).mean()) if m.sum() else np.nan
    return rep
