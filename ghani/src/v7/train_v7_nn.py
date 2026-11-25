"""v7 Model Family #4: 5-Fold GroupKFold(customer_id) PyTorch Deep Tabular ResNet (from scratch on pure feat_v7!).

Saves fold scalers (scaler_f0..4.joblib) AND Top-3 SWA averaged state_dict (resnet_f0..4.pt) directly
so zero-retraining inference is 100.000% bit-for-bit exact!

Jalankan dari root repo:
  python -u -W ignore ghani/src/train_v7_nn.py
"""
import gc
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss

for _sub in ['common', 'v1', 'v2', 'v5', 'v6', 'v7']:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)
import glmm_lib as glmm
from train_v7_lgb import compute_test_sim_weights, eval_v7_oof, get_v7_feature_cols

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


class ResBlockV7(nn.Module):
    def __init__(self, dim, dropout=0.18):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class TabularResNetV7(nn.Module):
    def __init__(self, n_in, n_classes=7, hidden=320, n_blocks=3, dropout=0.18):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(n_in, hidden), nn.LayerNorm(hidden))
        self.blocks = nn.ModuleList([ResBlockV7(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, n_classes))

    def forward(self, x):
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)


def fit_transform_nn_v7(df_tr, df_va, df_te, num_cols, cat_cols):
    Xtr = df_tr[num_cols].to_numpy(dtype=np.float32, copy=True)
    Xva = df_va[num_cols].to_numpy(dtype=np.float32, copy=True)
    Xte = df_te[num_cols].to_numpy(dtype=np.float32, copy=True)

    # Also add missingness indicator flags for key NaN columns (e.g., cu_sp_loo_* when n==1!)
    na_cols = [i for i in range(Xtr.shape[1]) if np.isnan(Xtr[:, i]).mean() > 0.01]
    na_tr = np.isnan(Xtr[:, na_cols]).astype(np.float32)
    na_va = np.isnan(Xva[:, na_cols]).astype(np.float32)
    na_te = np.isnan(Xte[:, na_cols]).astype(np.float32)

    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med).astype(np.float32)
    for arr in (Xtr, Xva, Xte):
        na = np.isnan(arr)
        if na.any():
            arr[na] = np.take(med, np.where(na)[1])

    heavy = np.std(Xtr, axis=0) > 50.0
    if heavy.any():
        for arr in (Xtr, Xva, Xte):
            arr[:, heavy] = np.sign(arr[:, heavy]) * np.log1p(np.abs(arr[:, heavy]))

    mu = Xtr.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = np.maximum(Xtr.std(axis=0, dtype=np.float64).astype(np.float32), 1e-5)
    for arr in (Xtr, Xva, Xte):
        arr -= mu
        arr /= sd
        np.clip(arr, -8.0, 8.0, out=arr)

    cat_cats = {}
    oh_tr, oh_va, oh_te = [Xtr, na_tr], [Xva, na_va], [Xte, na_te]
    for c in cat_cols:
        cats = list(pd.Categorical(df_tr[c].astype(str)).categories)
        cat_cats[c] = cats
        for src, dst in [(df_tr[c], oh_tr), (df_va[c], oh_va), (df_te[c], oh_te)]:
            codes = pd.Categorical(src.astype(str), categories=cats).codes
            mat = np.zeros((len(codes), len(cats)), dtype=np.float32)
            ok = codes >= 0
            mat[np.where(ok)[0], codes[ok]] = 1.0
            dst.append(mat)

    scaler = {
        "num_cols": num_cols, "cat_cols": cat_cols, "cat_cats": cat_cats,
        "na_cols": na_cols, "med": med, "heavy": heavy, "mu": mu, "sd": sd,
    }
    return np.hstack(oh_tr), np.hstack(oh_va), np.hstack(oh_te), scaler


def main():
    torch.set_num_threads(os.cpu_count() or 8)
    os.makedirs("ghani/models/v7/nn", exist_ok=True)
    os.makedirs("ghani/cache/v7/nn", exist_ok=True)

    tr = pd.read_pickle("ghani/cache/v7/feat_v7_train.pkl")
    te = pd.read_pickle("ghani/cache/v7/feat_v7_test.pkl")
    folds = np.load("ghani/cache/v7/folds_gkf5.npy")

    classes = sorted(tr["status_penanganan"].unique())
    y = tr["status_penanganan"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
    K = len(classes)

    n_tr = tr["cu_sp_n"].to_numpy(dtype=np.float32)
    n_te = te["cu_sp_n"].to_numpy(dtype=np.float32)
    sim_w, _ = compute_test_sim_weights(n_tr, n_te, tr["is_peak_month"].to_numpy(), te["is_peak_month"].to_numpy())

    feats = get_v7_feature_cols(tr)
    cat_cols = [c for c in feats if str(tr[c].dtype) == "category"]
    num_cols = [c for c in feats if c not in cat_cols]
    log(f"Loaded feat_v7 for TabularResNetV7: tr={tr.shape}, te={te.shape}, num={len(num_cols)}, cat={len(cat_cols)}")
    tr.drop(columns=[c for c in tr.columns if c not in feats], inplace=True)
    te.drop(columns=[c for c in te.columns if c not in feats], inplace=True)
    gc.collect()

    oof_logits = np.zeros((len(y), K), dtype=np.float32)
    te_logits = np.zeros((len(te), K), dtype=np.float32)

    batch_size = 4096
    epochs = 16

    for f in range(5):
        m_path = f"ghani/models/v7/nn/resnet_f{f}.pt"
        sc_path = f"ghani/models/v7/nn/scaler_f{f}.joblib"
        oof_p = f"ghani/cache/v7/nn/oof_f{f}.npy"
        te_p = f"ghani/cache/v7/nn/te_f{f}.npy"
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        if os.path.exists(m_path) and os.path.exists(sc_path) and os.path.exists(oof_p) and os.path.exists(te_p) and os.environ.get("FORCE", "0") != "1":
            oof_logits[va_idx] = np.load(oof_p)
            te_logits += np.load(te_p) / 5.0
            log(f"v7 ResNet Fold {f}: cached.")
            continue

        Xtr, Xva, Xte, scaler = fit_transform_nn_v7(tr.iloc[tr_idx], tr.iloc[va_idx], te, num_cols, cat_cols)
        joblib.dump(scaler, sc_path, compress=3)

        torch.manual_seed(42 + f)
        np.random.seed(42 + f)
        model = TabularResNetV7(Xtr.shape[1], n_classes=K, hidden=320, n_blocks=3, dropout=0.18)
        opt = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=2.5e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
        crit = nn.CrossEntropyLoss()

        ytr_t = torch.from_numpy(y[tr_idx].astype(np.int64))
        N_tr = len(tr_idx)
        top_ckpts = []

        log(f"=== v7 TabularResNetV7 Fold {f}: input_dim={Xtr.shape[1]} ===")
        for ep in range(epochs):
            model.train()
            perm = np.random.permutation(N_tr)
            for s in range(0, N_tr, batch_size):
                ib = perm[s : s + batch_size]
                opt.zero_grad()
                loss = crit(model(torch.from_numpy(Xtr[ib])), ytr_t[ib])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            sched.step()

            model.eval()
            ch = []
            with torch.no_grad():
                for s in range(0, len(va_idx), 16384):
                    ch.append(model(torch.from_numpy(Xva[s : s + 16384])).numpy())
            va_raw = np.vstack(ch).astype(np.float32)
            vp = glmm.softmax(va_raw)
            vll = log_loss(y[va_idx], vp, labels=list(range(K)))
            if len(top_ckpts) < 3 or vll < top_ckpts[-1][0]:
                st_c = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                top_ckpts.append((vll, st_c))
                top_ckpts.sort(key=lambda t: t[0])
                top_ckpts = top_ckpts[:3]
            if (ep + 1) % 4 == 0 or ep == 0 or ep == epochs - 1:
                log(f"  ResNetV7 F{f} ep={ep+1:2d}/{epochs}: val_ll={vll:.5f} (best={top_ckpts[0][0]:.5f}) argF1={glmm.fast_macro_f1(y[va_idx], vp.argmax(1), K):.5f}")

        # Weight-space SWA of top-3 checkpoints -> single .pt file reproduces exact SWA!
        swa_state = {}
        for k in top_ckpts[0][1].keys():
            swa_state[k] = sum(t[1][k] for t in top_ckpts) / float(len(top_ckpts))
        model.load_state_dict(swa_state)
        torch.save(swa_state, m_path)
        model.eval()

        va_ch, te_ch = [], []
        with torch.no_grad():
            for s in range(0, len(va_idx), 16384):
                va_ch.append(model(torch.from_numpy(Xva[s : s + 16384])).numpy())
            for s in range(0, len(Xte), 16384):
                te_ch.append(model(torch.from_numpy(Xte[s : s + 16384])).numpy())
        va_swa = np.vstack(va_ch).astype(np.float32)
        te_swa = np.vstack(te_ch).astype(np.float32)

        np.save(oof_p, va_swa)
        np.save(te_p, te_swa)
        oof_logits[va_idx] = va_swa
        te_logits += te_swa / 5.0
        vp = glmm.softmax(va_swa)
        log(f"  ResNetV7 F{f} SWA: val_ll={log_loss(y[va_idx], vp, labels=list(range(K))):.5f} | argF1={glmm.fast_macro_f1(y[va_idx], vp.argmax(1), K):.5f}")
        del Xtr, Xva, Xte, model, top_ckpts, swa_state
        gc.collect()

    np.save("ghani/cache/v7/nn_oof_logits.npy", oof_logits)
    np.save("ghani/cache/v7/nn_test_logits.npy", te_logits)
    ll_a, ll_s, f1_a, f1_s, w_m, _ = eval_v7_oof("v7_TabularResNetV7", glmm.softmax(oof_logits), y, folds, n_tr, sim_w, K)
    json.dump(
        {"oof_ll": float(ll_a), "test_sim_ll": float(ll_s), "cv_f1": float(f1_a), "test_sim_f1": float(f1_s), "weights": w_m.tolist()},
        open("ghani/models/v7/meta_nn.json", "w"), indent=2,
    )


if __name__ == "__main__":
    main()
