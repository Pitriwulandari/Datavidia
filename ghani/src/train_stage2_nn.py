"""Stage-2 Residual-Skip 5-Fold PyTorch Deep Tabular ResNet (completely orthogonal neural model!).

Starts directly from leak-free Joint GLMM logits via residual skip connection:
  logits = linear_cal(glmm_logits) + ResNetMLP(standardized_features)
Smooth inductive bias -> huge diversity boost when ensembled with GBDTs!

Jalankan (dari root repo, setelah build_stage2_glmm.py):
  python -u -W ignore ghani/src/train_stage2_nn.py
"""
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glmm_lib as glmm
import stage2_common as s2

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.15):
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


class TabularResNet(nn.Module):
    def __init__(self, n_num, n_classes=7, hidden=256, n_blocks=2, dropout=0.15):
        super().__init__()
        # Learnable 7x7 cross-class calibration initialized to identity on GLMM logits!
        self.skip_w = nn.Linear(n_classes, n_classes, bias=True)
        nn.init.eye_(self.skip_w.weight)
        nn.init.zeros_(self.skip_w.bias)

        self.in_proj = nn.Sequential(nn.Linear(n_num, hidden), nn.LayerNorm(hidden))
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, n_classes))
        # Initialize residual head near zero so Epoch 0 starts at exact GLMM performance!
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, glmm_logits):
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.skip_w(glmm_logits) + self.head(h)


def prep_numeric_matrix(df_tr, df_va, df_te, cat_cols):
    """One-hot encode small categoricals, impute NaNs with median + clip/standardize using train stats."""
    num_cols = [c for c in df_tr.columns if c not in cat_cols]
    Xtr = df_tr[num_cols].to_numpy(dtype=np.float32, copy=True)
    Xva = df_va[num_cols].to_numpy(dtype=np.float32, copy=True)
    Xte = df_te[num_cols].to_numpy(dtype=np.float32, copy=True)

    # Compute median/mean/std on train (vectorized, fast)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med).astype(np.float32)
    for arr in (Xtr, Xva, Xte):
        na = np.isnan(arr)
        if na.any():
            arr[na] = np.take(med, np.where(na)[1])

    # Signed log1p for heavy-tailed features (std > 100)
    heavy = np.std(Xtr, axis=0) > 100.0
    if heavy.any():
        for arr in (Xtr, Xva, Xte):
            arr[:, heavy] = np.sign(arr[:, heavy]) * np.log1p(np.abs(arr[:, heavy]))

    mu = Xtr.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = np.maximum(Xtr.std(axis=0, dtype=np.float64).astype(np.float32), 1e-5)
    for arr in (Xtr, Xva, Xte):
        arr -= mu
        arr /= sd
        np.clip(arr, -8.0, 8.0, out=arr)

    # Compact one-hot for low-cardinality categoricals
    if cat_cols:
        oh_tr, oh_va, oh_te = [], [], []
        for c in cat_cols:
            cats = pd.Categorical(df_tr[c].astype(str)).categories
            n_c = len(cats)
            for src, dst in [(df_tr[c], oh_tr), (df_va[c], oh_va), (df_te[c], oh_te)]:
                codes = pd.Categorical(src.astype(str), categories=cats).codes
                mat = np.zeros((len(codes), n_c), dtype=np.float32)
                ok = codes >= 0
                mat[np.where(ok)[0], codes[ok]] = 1.0
                dst.append(mat)
        Xtr = np.hstack([Xtr] + oh_tr)
        Xva = np.hstack([Xva] + oh_va)
        Xte = np.hstack([Xte] + oh_te)

    return Xtr, Xva, Xte


def main():
    torch.set_num_threads(os.cpu_count() or 8)
    os.makedirs("ghani/models/stage2/nn", exist_ok=True)
    os.makedirs("ghani/cache/stage2/nn", exist_ok=True)

    tr_base, te_base, y, folds, classes, te_sid, cat_cols = s2.load_stage2_base()
    K = len(classes)
    log(f"loaded Stage-2 base for ResNet: tr={tr_base.shape}, te={te_base.shape}")

    oof_logits = np.zeros((len(y), K), dtype=np.float32)
    te_logits = np.zeros((len(te_sid), K), dtype=np.float32)

    batch_size = 4096
    epochs = 12

    for f in range(5):
        m_path = f"ghani/models/stage2/nn/resnet_f{f}.pt"
        oof_p = f"ghani/cache/stage2/nn/oof_f{f}.npy"
        te_p = f"ghani/cache/stage2/nn/te_f{f}.npy"
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        if os.path.exists(m_path) and os.path.exists(oof_p) and os.path.exists(te_p) and os.environ.get("FORCE", "0") != "1":
            oof_logits[va_idx] = np.load(oof_p)
            te_logits += np.load(te_p) / 5.0
            log(f"ResNet Fold {f}: cached.")
            continue

        blk_tr, blk_va, blk_te = s2.get_fold_glmm_arrays(f, folds)
        base_tr = blk_tr[:, 7:14].astype(np.float32)
        base_va = blk_va[:, 7:14].astype(np.float32)
        base_te = blk_te[:, 7:14].astype(np.float32)

        df_tr = s2.attach_glmm_df(tr_base.iloc[tr_idx], blk_tr)
        df_va = s2.attach_glmm_df(tr_base.iloc[va_idx], blk_va)
        df_te = s2.attach_glmm_df(te_base, blk_te)
        Xtr, Xva, Xte = prep_numeric_matrix(df_tr, df_va, df_te, cat_cols)
        del df_tr, df_va, df_te
        gc.collect()

        torch.manual_seed(42 + f)
        np.random.seed(42 + f)
        model = TabularResNet(Xtr.shape[1], n_classes=K, hidden=256, n_blocks=2, dropout=0.20)
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
        crit = nn.CrossEntropyLoss()

        ytr_t = torch.from_numpy(y[tr_idx].astype(np.int64))
        N_tr = len(tr_idx)
        best_ll = 999.0
        best_state = None
        best_va_raw = None

        log(f"=== Stage-2 TabularResNet Fold {f}: feat_dim={Xtr.shape[1]} ===")
        for ep in range(epochs):
            model.train()
            perm = np.random.permutation(N_tr)
            for s in range(0, N_tr, batch_size):
                idx_b = perm[s : s + batch_size]
                xb = torch.from_numpy(Xtr[idx_b])
                gb = torch.from_numpy(base_tr[idx_b])
                yb = ytr_t[idx_b]
                opt.zero_grad()
                loss = crit(model(xb, gb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            sched.step()

            model.eval()
            va_chunks = []
            with torch.no_grad():
                for s in range(0, len(va_idx), 16384):
                    xb = torch.from_numpy(Xva[s : s + 16384])
                    gb = torch.from_numpy(base_va[s : s + 16384])
                    va_chunks.append(model(xb, gb).numpy())
            va_raw = np.vstack(va_chunks).astype(np.float32)
            vp = glmm.softmax(va_raw)
            vll = log_loss(y[va_idx], vp, labels=list(range(K)))
            if vll < best_ll:
                best_ll = vll
                best_va_raw = va_raw
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (ep + 1) % 3 == 0 or ep == 0 or ep == epochs - 1:
                log(f"  ResNet F{f} ep={ep+1:2d}/{epochs}: val_ll={vll:.5f} (best={best_ll:.5f}) arg_F1={glmm.fast_macro_f1(y[va_idx], vp.argmax(1), K):.5f}")

        model.load_state_dict(best_state)
        torch.save(best_state, m_path)

        model.eval()
        te_chunks = []
        with torch.no_grad():
            for s in range(0, len(te_sid), 16384):
                xb = torch.from_numpy(Xte[s : s + 16384])
                gb = torch.from_numpy(base_te[s : s + 16384])
                te_chunks.append(model(xb, gb).numpy())
        te_raw = np.vstack(te_chunks).astype(np.float32)

        np.save(oof_p, best_va_raw)
        np.save(te_p, te_raw)
        oof_logits[va_idx] = best_va_raw
        te_logits += te_raw / 5.0
        del Xtr, Xva, Xte, model
        gc.collect()

    oof_p_all = glmm.softmax(oof_logits)
    ll = log_loss(y, oof_p_all, labels=list(range(K)))
    f1_cv, w_opt, _ = glmm.cv_macro_f1_with_weights(oof_p_all, y, folds)
    log(f"=== STAGE-2 TABULAR RESNET 5-FOLD: OOF_ll={ll:.5f} | arg_F1={glmm.fast_macro_f1(y, oof_p_all.argmax(1), K):.5f} | CV_F1={f1_cv:.5f} ===")
    np.save("ghani/cache/stage2/nn_oof_logits.npy", oof_logits)
    np.save("ghani/cache/stage2/nn_test_logits.npy", te_logits)
    json.dump({"oof_ll": float(ll), "cv_f1": float(f1_cv), "weights": w_opt.tolist()}, open("ghani/models/stage2/meta_nn.json", "w"), indent=2)


if __name__ == "__main__":
    main()
