"""
AFCP with up to two selected attributes per placeholder label (Zhou & Sesia 2024, Appendix A2).

Subclass of the original Adaptive_Selection; the original code is not modified.
Per placeholder label y:
  round 1: Algorithm 1 on all candidates  -> A1 (or empty)
  round 2 (only if A1 non-empty): Algorithm 1 on candidates \\ {A1} -> A2 (or empty)
  A(y) = {A1} U {A2}
Prediction set (eq. A11): marginal set
    U  set calibrated on the joint cell of A(y)            (Algorithm A1 with protected attributes A(y))
    U  set calibrated on each selected attribute alone      (standard equalized set per attribute)
Final selection = intersection over y of A(y) (eq. A10), reported as in the main method.
With a single candidate, or whenever round 2 selects nothing, this reduces exactly to the main method.
"""
import numpy as np
from tqdm import tqdm
from SelectiveCI_fairness.methods import Adaptive_Selection
from third_party.utils import nonconf_scores_mc, compute_conf_pvals, arc_wrapper


class Adaptive_Selection_A2(Adaptive_Selection):

    def __init__(self, *a, sig_level=0.1, **k):
        super().__init__(*a, **k)
        self.sig_level = sig_level
        self.log = []        # round-1 picks, as LoggingAdaptive records them (per test point x label)
        self.log2 = []       # round-2 picks

    def _pick(self, att_idx, E, X_aug):
        k = self.select_the_worst_group(self.error_func_groupwise, att_idx, E, X_aug)
        return None if k == [] else k

    def multiclass_classification(self, X_calib, Y_calib, X_test, bbox_mc, att_idx,
                                  return_khat=True, conditional=False, left_tail=False):
        n_test = X_test.shape[0]
        labels = np.array(list(set(Y_calib)))
        k_final, C_sets_final = [], []

        calib_scores = nonconf_scores_mc(X_calib, Y_calib, bbox_mc, alpha=self.alpha, random_state=self.random_state)
        test_scores = np.full((n_test, len(labels)), -np.inf)
        for idx, y in enumerate(labels):
            test_scores[:, idx] = nonconf_scores_mc(X_test, np.repeat(y, n_test), bbox_mc, alpha=self.alpha, random_state=self.random_state)

        for i, X in tqdm(enumerate(X_test)):
            k_hat_i = []
            pvals_y = np.full(len(labels), -np.inf)     # cell-calibrated set, per placeholder label
            pvals_add = np.full(len(labels), -np.inf)   # marginal / label-conditional set

            for idx, y in enumerate(labels):
                idx_y = np.where(Y_calib == y)[0] if conditional else np.arange(len(Y_calib))
                X_calib_y, Y_calib_y = X_calib[idx_y], Y_calib[idx_y]
                X_aug, Y_aug = self.augment_data(X_calib_y, Y_calib_y, X, y)
                scores_aug = np.append(calib_scores[idx_y], test_scores[i, idx])
                n_union = X_aug.shape[0]

                E = []
                for j in range(n_union):
                    not_j = [m for m in range(n_union) if m != j]
                    C_temp = arc_wrapper(scores_aug[not_j], X_aug[j][None, :], bbox_mc, self.alpha, self.random_state)
                    E.append(Y_aug[j] not in C_temp[0])

                # round 1 and round 2
                a1 = self._pick(att_idx, E, X_aug); self.log.append([] if a1 is None else a1)
                a2 = None
                if a1 is not None and len(att_idx) > 1:
                    a2 = self._pick([a for a in att_idx if a != a1], E, X_aug)
                self.log2.append([] if a2 is None else a2)
                sel = [a for a in (a1, a2) if a is not None]
                k_hat_i.append(set(sel))

                cal = scores_aug[:-1]
                if not sel:
                    pvals_y[idx] = compute_conf_pvals(scores_aug[-1], cal, left_tail=left_tail)
                else:
                    mask = np.ones(len(cal), bool)
                    for a in sel:
                        mask &= (X_calib_y[:, a] == X[a])
                    pvals_y[idx] = compute_conf_pvals(scores_aug[-1], cal[mask], left_tail=left_tail)
                pvals_add[idx] = compute_conf_pvals(scores_aug[-1], cal, left_tail=left_tail)

            # third term of eq. A11: for every attribute selected under any placeholder label, the standard
            # equalized set for that attribute alone (evaluated for every label)
            C = set(labels[np.where(pvals_y >= self.alpha)[0]]) | set(labels[np.where(pvals_add >= self.alpha)[0]])
            for a in set.union(*k_hat_i) if k_hat_i else set():
                for idx, y in enumerate(labels):
                    idx_y = np.where(Y_calib == y)[0] if conditional else np.arange(len(Y_calib))
                    m1 = X_calib[idx_y][:, a] == X[a]
                    p = compute_conf_pvals(test_scores[i, idx], calib_scores[idx_y][m1], left_tail=left_tail)
                    if p >= self.alpha:
                        C.add(y)
            C_sets_final.append(list(C))
            k_final.append(list(set.intersection(*k_hat_i)))

        return C_sets_final, k_final
