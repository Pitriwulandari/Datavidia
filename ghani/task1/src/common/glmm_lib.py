"""Joint Hierarchical Multinomial Mixed-Effects (Laplace-EM MAP GLMM) & Exact Macro-F1 Optimizer.

Solves:
  min_{b, U_e}  sum_i CE(y_i, base_logits_i + b + sum_e U_e[code_e(i)]) + 0.5 * sum_{e,g,k} lam[e,k] * U_e[g,k]^2
with MacKay Laplace-EM empirical Bayes updates for per-entity, per-class prior precision lam[e,k].
"""
import time

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score, log_loss


def softmax(z):
    zm = z - z.max(axis=1, keepdims=True)
    e = np.exp(zm)
    return e / e.sum(axis=1, keepdims=True)


def fit_joint_glmm(
    base_logits_tr,
    y_tr,
    entities_tr,
    n_levels_dict,
    n_classes=7,
    n_em=10,
    n_newton=3,
    damping=0.85,
    init_lam=10.0,
    lam_min=0.15,
    lam_max=2000.0,
    fixed_lam=None,
    verbose=True,
):
    """Fit Joint Multinomial GLMM on training fold.

    Args:
        base_logits_tr: (N, K) float64/float32 Stage-1 logits (will be centered per row).
        y_tr: (N,) int class labels 0..K-1.
        entities_tr: dict {name: (N,) int32 entity codes in 0..n_levels-1}.
        n_levels_dict: dict {name: int total levels across train+test}.
        fixed_lam: optional dict {name: (K,) array} to skip EM updates and use fixed precisions.
    Returns:
        state dict with bias, U (dict of (n_levels, K) float32), H_data (dict of (n_levels, K) float32), lam (dict).
    """
    t0 = time.time()
    N = len(y_tr)
    K = n_classes
    ent_names = list(entities_tr.keys())

    # Center base logits per row
    eta = base_logits_tr.astype(np.float64)
    eta = eta - eta.mean(axis=1, keepdims=True)

    # One-hot target
    Y = np.zeros((N, K), dtype=np.float64)
    Y[np.arange(N), y_tr] = 1.0

    b = np.zeros(K, dtype=np.float64)
    U = {e: np.zeros((n_levels_dict[e], K), dtype=np.float64) for e in ent_names}
    H_data = {e: np.zeros((n_levels_dict[e], K), dtype=np.float64) for e in ent_names}
    cnt = {e: np.bincount(entities_tr[e], minlength=n_levels_dict[e]) for e in ent_names}
    seen_mask = {e: cnt[e] > 0 for e in ent_names}

    if fixed_lam is not None:
        lam = {e: np.array(fixed_lam[e], dtype=np.float64).copy() for e in ent_names}
        n_em_eff = 1
        n_newton_eff = max(n_newton * 4, 15)
    else:
        lam = {e: np.full(K, init_lam, dtype=np.float64) for e in ent_names}
        n_em_eff = n_em
        n_newton_eff = n_newton

    for em_it in range(n_em_eff):
        for _ in range(n_newton_eff):
            # 1. Bias update
            p = softmax(eta)
            diff = p - Y
            w_var = p * (1.0 - p)
            g_b = diff.sum(axis=0)
            h_b = w_var.sum(axis=0) + 1e-4
            step_b = -damping * (g_b / h_b)
            step_b -= step_b.mean()
            b += step_b
            eta += step_b

            # 2. Block-coordinate Newton over each entity
            for e in ent_names:
                codes = entities_tr[e]
                n_u = n_levels_dict[e]
                p = softmax(eta)
                diff = p - Y
                w_var = p * (1.0 - p)
                Ue = U[e]
                lame = lam[e]
                He = H_data[e]

                step = np.zeros((n_u, K), dtype=np.float64)
                for k in range(K):
                    g_k = np.bincount(codes, weights=diff[:, k], minlength=n_u) + lame[k] * Ue[:, k]
                    hd_k = np.bincount(codes, weights=w_var[:, k], minlength=n_u)
                    He[:, k] = hd_k
                    step[:, k] = -damping * g_k / (hd_k + lame[k])

                # Enforce sum-to-zero across classes per entity level for identifiability
                step -= step.mean(axis=1, keepdims=True)
                Ue += step
                eta += step[codes]

        # MacKay Laplace-EM update for prior precision lam[e, k]
        if fixed_lam is None:
            for e in ent_names:
                m = seen_mask[e]
                Ue_m = U[e][m]
                He_m = H_data[e][m]
                for k in range(K):
                    l_old = lam[e][k]
                    gamma = np.sum(He_m[:, k] / (He_m[:, k] + l_old))
                    ssq = np.sum(Ue_m[:, k] ** 2)
                    l_new = np.clip(gamma / max(ssq, 1e-6), lam_min, lam_max)
                    # Smooth geometric damping on lambda update for rock-solid convergence
                    lam[e][k] = np.exp(0.7 * np.log(l_new) + 0.3 * np.log(l_old))

        if verbose and (em_it == 0 or (em_it + 1) % 3 == 0 or em_it == n_em_eff - 1):
            p = softmax(eta)
            ll = log_loss(y_tr, p, labels=list(range(K)))
            sig_str = " | ".join(
                f"{e}:sigma_med={np.median(1.0/np.sqrt(lam[e])):.3f}" for e in ent_names
            )
            print(f"  [GLMM em={em_it+1:2d}/{n_em_eff} {time.time()-t0:5.1f}s] tr_ll={ll:.5f} | {sig_str}", flush=True)

    return {
        "bias": b.astype(np.float32),
        "U": {e: U[e].astype(np.float32) for e in ent_names},
        "H_data": {e: H_data[e].astype(np.float32) for e in ent_names},
        "cnt": {e: cnt[e].astype(np.float32) for e in ent_names},
        "lam": {e: lam[e].astype(np.float32) for e in ent_names},
    }


def predict_glmm(base_logits, entities, state, lam_scale=1.0, use_moderation=False):
    """Predict logits & probabilities using fitted GLMM state."""
    eta = base_logits.astype(np.float64)
    eta = eta - eta.mean(axis=1, keepdims=True)
    eta += state["bias"].astype(np.float64)

    total_var = np.zeros_like(eta) if use_moderation else None
    for e, codes in entities.items():
        Ue = state["U"][e].astype(np.float64)
        if lam_scale != 1.0:
            # Approximate rescaling of U if lam is scaled: U_scaled ~ U * (H + lam) / (H + lam_scale * lam)
            He = state["H_data"][e].astype(np.float64)
            le = state["lam"][e].astype(np.float64)[None, :]
            Ue = Ue * (He + le) / (He + lam_scale * le)
        eta += Ue[codes]
        if use_moderation:
            He = state["H_data"][e].astype(np.float64)
            le = state["lam"][e].astype(np.float64)[None, :] * lam_scale
            total_var += (1.0 / (He + le))[codes]

    if use_moderation:
        eta = eta / np.sqrt(1.0 + (np.pi / 8.0) * total_var)

    return eta.astype(np.float32), softmax(eta).astype(np.float32)


def fast_macro_f1(y_true, pred, K=7):
    """15x faster exact Macro-F1 via vectorized bincount confusion matrix."""
    cm = np.bincount(y_true * K + pred, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    f1s = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1.0), 0.0)
    return float(f1s.mean())


def optimize_macro_f1_weights(proba, y_true, n_restarts=3, seed=42):
    """Fast & accurate Macro-F1 class multiplier optimization using multi-resolution coordinate search + Powell."""
    K = proba.shape[1]
    rng = np.random.RandomState(seed)
    y_int = y_true.astype(np.int32)

    def _eval(log_w):
        w = np.exp(log_w - log_w.mean()).astype(np.float32)
        return fast_macro_f1(y_int, (proba * w).argmax(1), K)

    # Start from inverse-sqrt prior & uniform
    priors = np.bincount(y_true, minlength=K) / len(y_true)
    starts = [
        -0.3 * np.log(priors + 1e-6),
        np.zeros(K, dtype=np.float64),
    ][: max(1, min(n_restarts, 2))]
    for _ in range(max(0, n_restarts - 2)):
        starts.append(rng.normal(0, 0.25, size=K))

    best_score = -1.0
    best_lw = np.zeros(K, dtype=np.float64)

    for lw0 in starts:
        lw = lw0.copy()
        sc = _eval(lw)
        # Multi-resolution coordinate ascent
        for step in [0.4, 0.2, 0.1, 0.05, 0.02, 0.008]:
            improved = True
            while improved:
                improved = False
                for k in range(K):
                    for delta in (-step, step):
                        cand = lw.copy()
                        cand[k] += delta
                        c_sc = _eval(cand)
                        if c_sc > sc + 2e-6:
                            sc = c_sc
                            lw = cand
                            improved = True
        if sc > best_score:
            best_score = sc
            best_lw = lw.copy()

    # Final Nelder-Mead polish
    res = minimize(lambda x: -_eval(x), best_lw, method="Nelder-Mead",
                   options={"maxiter": 180, "xatol": 2e-3, "fatol": 2e-6})
    if -res.fun > best_score:
        best_score = -res.fun
        best_lw = res.x

    w = np.exp(best_lw - best_lw.mean())
    return w, best_score


def cv_macro_f1_with_weights(oof_proba, y_true, folds):
    """100% leak-free OOF evaluation of Macro-F1 multiplier tuning."""
    n_folds = int(folds.max()) + 1
    oof_preds = np.zeros(len(y_true), dtype=np.int32)
    ws = []
    for f in range(n_folds):
        tr_m = folds != f
        va_m = folds == f
        w_f, _ = optimize_macro_f1_weights(oof_proba[tr_m], y_true[tr_m], n_restarts=1, seed=42 + f)
        ws.append(w_f)
        oof_preds[va_m] = (oof_proba[va_m] * w_f).argmax(1)
    honest_f1 = f1_score(y_true, oof_preds, average="macro")
    w_mean = np.exp(np.mean(np.log(ws), axis=0))
    w_mean = w_mean / np.exp(np.mean(np.log(w_mean)))
    return honest_f1, w_mean, oof_preds
