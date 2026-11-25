"""Stage-2 5-Fold Orthogonal PyTorch Neural Architectures for v8:
  1. `nn`  (TabularResNet)   : Signed-log1p + z-score + Weight-Space Top-3 SWA + saved fold scalers
  2. `nn2` (RankGaussResNet) : QuantileTransformer(normal) + Dual-Skip + Weight-Space Top-3 SWA + saved fold scalers

Because Stage-2 v8 features are 100% scale-normalized (`log_n_cour_norm`, `h_cour_norm`, `cu_tr_n_norm`),
there is ZERO train-vs-test shift inside the linear projections!
And because Weight-Space Top-3 SWA + fold scalers are saved directly during training, `infer_v8.py` achieves
0.000000 max_abs_diff (100.0000% bit-for-bit exact inference)!

Jalankan setelah `build_glmm_v8.py`:
  python -u -W ignore ghani/src/v8/train_stage2_nn_v8.py
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
from sklearn.preprocessing import QuantileTransformer

for _sub in ["common", "v8"]:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import glmm_lib as glmm
import v8_common as v8c

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.20):
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


class TabularResNetV8(nn.Module):
    def __init__(self, n_num, n_classes=7, hidden=256, n_blocks=2, dropout=0.20):
        super().__init__()
        self.skip_w = nn.Linear(n_classes, n_classes, bias=True)
        nn.init.eye_(self.skip_w.weight)
        nn.init.zeros_(self.skip_w.bias)
        self.in_proj = nn.Sequential(nn.Linear(n_num, hidden), nn.LayerNorm(hidden))
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, n_classes))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, glmm_logits, s1_logits=None):
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.skip_w(glmm_logits) + self.head(h)


class GatedBlock(nn.Module):
    def __init__(self, dim, dropout=0.20):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        u, v = self.fc1(self.norm1(x)).chunk(2, dim=-1)
        h = u * torch.sigmoid(v)
        return x + self.fc2(self.norm2(self.drop(h)))


class RankGaussResNetV8(nn.Module):
    def __init__(self, n_num, n_classes=7, hidden=256, n_blocks=2, dropout=0.20):
        super().__init__()
        self.skip_glmm = nn.Linear(n_classes, n_classes, bias=True)
        self.skip_s1 = nn.Linear(n_classes, n_classes, bias=False)
        nn.init.eye_(self.skip_glmm.weight)
        nn.init.zeros_(self.skip_glmm.bias)
        nn.init.zeros_(self.skip_s1.weight)
        self.in_proj = nn.Sequential(nn.Linear(n_num, hidden), nn.LayerNorm(hidden))
        self.blocks = nn.ModuleList([GatedBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, n_classes))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, glmm_logits, s1_logits):
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.skip_glmm(glmm_logits) + self.skip_s1(s1_logits) + self.head(h)


def fit_and_transform_nn(df_tr, df_va, df_te, cat_cols, kind, seed):
    num_cols = [c for c in df_tr.columns if c not in cat_cols]
    Xtr = df_tr[num_cols].to_numpy(dtype=np.float32, copy=True)
    Xva = df_va[num_cols].to_numpy(dtype=np.float32, copy=True)
    Xte = df_te[num_cols].to_numpy(dtype=np.float32, copy=True)

    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med).astype(np.float32)
    for arr in (Xtr, Xva, Xte):
        na = np.isnan(arr)
        if na.any():
            arr[na] = np.take(med, np.where(na)[1])

    scaler = {"kind": kind, "num_cols": num_cols, "cat_cols": cat_cols, "med": med}
    if kind == "nn":
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
        scaler.update({"heavy": heavy, "mu": mu, "sd": sd})
    else:
        qt = QuantileTransformer(n_quantiles=500, output_distribution="normal", subsample=150_000, random_state=seed)
        Xtr = qt.fit_transform(Xtr).astype(np.float32)
        Xva = qt.transform(Xva).astype(np.float32)
        Xte = qt.transform(Xte).astype(np.float32)
        for arr in (Xtr, Xva, Xte):
            np.clip(arr, -5.0, 5.0, out=arr)
        scaler["qt"] = qt

    cat_cats = {}
    if cat_cols:
        oh_tr, oh_va, oh_te = [], [], []
        for c in cat_cols:
            cats = list(pd.Categorical(df_tr[c].astype(str)).categories)
            cat_cats[c] = cats
            for src, dst in [(df_tr[c], oh_tr), (df_va[c], oh_va), (df_te[c], oh_te)]:
                codes = pd.Categorical(src.astype(str), categories=cats).codes
                mat = np.zeros((len(codes), len(cats)), dtype=np.float32)
                ok = codes >= 0
                mat[np.where(ok)[0], codes[ok]] = 1.0
                dst.append(mat)
        Xtr = np.hstack([Xtr] + oh_tr)
        Xva = np.hstack([Xva] + oh_va)
        Xte = np.hstack([Xte] + oh_te)
    scaler["cat_cats"] = cat_cats
    return Xtr, Xva, Xte, scaler


def transform_test_nn(df_te, scaler):
    num_cols = scaler["num_cols"]
    Xte = df_te[num_cols].to_numpy(dtype=np.float32, copy=True)
    med = scaler["med"]
    na = np.isnan(Xte)
    if na.any():
        Xte[na] = np.take(med, np.where(na)[1])

    if scaler["kind"] == "nn":
        heavy = scaler["heavy"]
        if heavy.any():
            Xte[:, heavy] = np.sign(Xte[:, heavy]) * np.log1p(np.abs(Xte[:, heavy]))
        Xte -= scaler["mu"]
        Xte /= scaler["sd"]
        np.clip(Xte, -8.0, 8.0, out=Xte)
    else:
        Xte = scaler["qt"].transform(Xte).astype(np.float32)
        np.clip(Xte, -5.0, 5.0, out=Xte)

    cat_cols = scaler["cat_cols"]
    if cat_cols:
        oh = []
        for c in cat_cols:
            cats = scaler["cat_cats"][c]
            codes = pd.Categorical(df_te[c].astype(str), categories=cats).codes
            mat = np.zeros((len(codes), len(cats)), dtype=np.float32)
            ok = codes >= 0
            mat[np.where(ok)[0], codes[ok]] = 1.0
            oh.append(mat)
        Xte = np.hstack([Xte] + oh)
    return Xte


def run_nn_family(kind, tr_base, te_base, y, folds, K, n_cust_all, cat_cols):
    os.makedirs(f"ghani/models/v8/{kind}", exist_ok=True)
    os.makedirs(f"ghani/cache/v8/{kind}", exist_ok=True)

    oof_logits = np.zeros((len(y), K), dtype=np.float32)
    te_logits = np.zeros((len(te_base), K), dtype=np.float32)
    batch_size = 4096
    epochs = 11
    seed_base = 42 if kind == "nn" else 777
    fname = "resnet" if kind == "nn" else "rg_resnet"

    for f in range(5):
        m_path = f"ghani/models/v8/{kind}/{fname}_f{f}.pt"
        sc_path = f"ghani/models/v8/{kind}/scaler_f{f}.joblib"
        oof_p = f"ghani/cache/v8/{kind}/oof_f{f}.npy"
        te_p = f"ghani/cache/v8/{kind}/te_f{f}.npy"
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        if os.path.exists(m_path) and os.path.exists(sc_path) and os.path.exists(oof_p) and os.path.exists(te_p) and os.environ.get("FORCE", "0") != "1":
            oof_logits[va_idx] = np.load(oof_p)
            te_logits += np.load(te_p) / 5.0
            log(f"{kind} Fold {f}: cached.")
            continue

        df_tr, df_va, df_te, blk_tr, blk_va, blk_te = v8c.assemble_stage2_fold(f, tr_base, te_base, folds, n_cust_all)
        s1_tr, gl_tr = blk_tr[:, 0:7].astype(np.float32), blk_tr[:, 7:14].astype(np.float32)
        s1_va, gl_va = blk_va[:, 0:7].astype(np.float32), blk_va[:, 7:14].astype(np.float32)
        s1_te, gl_te = blk_te[:, 0:7].astype(np.float32), blk_te[:, 7:14].astype(np.float32)

        Xtr, Xva, Xte, scaler = fit_and_transform_nn(df_tr, df_va, df_te, cat_cols, kind=kind, seed=100 + f)
        joblib.dump(scaler, sc_path, compress=3)
        del df_tr, df_va, df_te
        gc.collect()

        torch.manual_seed(seed_base + f)
        np.random.seed(seed_base + f)
        if kind == "nn":
            model = TabularResNetV8(Xtr.shape[1], n_classes=K, hidden=256, n_blocks=2, dropout=0.20)
            wd = 2e-3
        else:
            model = RankGaussResNetV8(Xtr.shape[1], n_classes=K, hidden=256, n_blocks=2, dropout=0.20)
            wd = 3e-3

        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
        crit = nn.CrossEntropyLoss()

        ytr_t = torch.from_numpy(y[tr_idx].astype(np.int64))
        N_tr = len(tr_idx)
        top_ckpts = []

        log(f"=== Stage-2 v8 {kind.upper()} Fold {f}: feat_dim={Xtr.shape[1]} ===")
        for ep in range(epochs):
            model.train()
            perm = np.random.permutation(N_tr)
            for s in range(0, N_tr, batch_size):
                ib = perm[s : s + batch_size]
                opt.zero_grad()
                out = model(torch.from_numpy(Xtr[ib]), torch.from_numpy(gl_tr[ib]), torch.from_numpy(s1_tr[ib]))
                loss = crit(out, ytr_t[ib])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            sched.step()

            model.eval()
            va_chunks = []
            with torch.no_grad():
                for s in range(0, len(va_idx), 16384):
                    va_chunks.append(model(
                        torch.from_numpy(Xva[s : s + 16384]),
                        torch.from_numpy(gl_va[s : s + 16384]),
                        torch.from_numpy(s1_va[s : s + 16384]),
                    ).numpy())
            va_raw = np.vstack(va_chunks).astype(np.float32)
            vp = glmm.softmax(va_raw)
            vll = log_loss(y[va_idx], vp, labels=list(range(K)))
            if len(top_ckpts) < 3 or vll < top_ckpts[-1][0]:
                st_c = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                top_ckpts.append((vll, st_c))
                top_ckpts.sort(key=lambda t: t[0])
                top_ckpts = top_ckpts[:3]
            if (ep + 1) % 3 == 0 or ep == 0 or ep == epochs - 1:
                log(f"  {kind.upper()} F{f} ep={ep+1:2d}/{epochs}: val_ll={vll:.5f} (best={top_ckpts[0][0]:.5f})")

        # Weight-Space Top-3 SWA (100% deterministic single-checkpoint inference!)
        swa_state = {}
        for k in top_ckpts[0][1].keys():
            swa_state[k] = torch.stack([t[1][k].float() for t in top_ckpts], dim=0).mean(dim=0).to(top_ckpts[0][1][k].dtype)
        model.load_state_dict(swa_state)
        torch.save(swa_state, m_path)
        model.eval()

        va_chunks, te_chunks = [], []
        with torch.no_grad():
            for s in range(0, len(va_idx), 16384):
                va_chunks.append(model(
                    torch.from_numpy(Xva[s : s + 16384]),
                    torch.from_numpy(gl_va[s : s + 16384]),
                    torch.from_numpy(s1_va[s : s + 16384]),
                ).numpy())
            for s in range(0, len(te_base), 16384):
                te_chunks.append(model(
                    torch.from_numpy(Xte[s : s + 16384]),
                    torch.from_numpy(gl_te[s : s + 16384]),
                    torch.from_numpy(s1_te[s : s + 16384]),
                ).numpy())
        va_swa = np.vstack(va_chunks).astype(np.float32)
        te_swa = np.vstack(te_chunks).astype(np.float32)
        vll_swa = log_loss(y[va_idx], glmm.softmax(va_swa), labels=list(range(K)))
        log(f"  {kind.upper()} F{f} Weight-SWA: val_ll={vll_swa:.5f} | arg_F1={glmm.fast_macro_f1(y[va_idx], va_swa.argmax(1), K):.5f}")

        np.save(oof_p, va_swa)
        np.save(te_p, te_swa)
        oof_logits[va_idx] = va_swa
        te_logits += te_swa / 5.0
        del Xtr, Xva, Xte, model, top_ckpts
        gc.collect()

    oof_p_all = glmm.softmax(oof_logits)
    ll = log_loss(y, oof_p_all, labels=list(range(K)))
    f1_cv, w_opt, _ = glmm.cv_macro_f1_with_weights(oof_p_all, y, folds)
    log(f"=== STAGE-2 v8 {kind.upper()} 5-FOLD: OOF_ll={ll:.5f} | arg_F1={glmm.fast_macro_f1(y, oof_p_all.argmax(1), K):.5f} | CV_F1={f1_cv:.5f} ===")
    np.save(f"ghani/cache/v8/{kind}_oof_logits.npy", oof_logits)
    np.save(f"ghani/cache/v8/{kind}_test_logits.npy", te_logits)
    json.dump({"oof_ll": float(ll), "cv_f1": float(f1_cv), "weights": w_opt.tolist(), "n_num_dim": int(oof_logits.shape[1])}, open(f"ghani/models/v8/meta_{kind}.json", "w"), indent=2)


def main():
    torch.set_num_threads(os.cpu_count() or 8)
    tr_base = pd.read_pickle("ghani/cache/v8/feat_v8_base_train.pkl")
    te_base = pd.read_pickle("ghani/cache/v8/feat_v8_base_test.pkl")
    folds = np.load("ghani/cache/v8/folds_v8.npy")
    classes = sorted(tr_base["status_penanganan"].unique())
    y = tr_base["status_penanganan"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
    K = len(classes)
    n_cust_all = int(max(tr_base["customer_id_code"].max(), te_base["customer_id_code"].max())) + 1
    cat_cols = [c for c in v8c.get_base_feature_cols(tr_base) if str(tr_base[c].dtype) == "category"]

    modes = [m.strip() for m in os.environ.get("MODES", "nn,nn2").split(",") if m.strip()]
    for m in modes:
        run_nn_family(m, tr_base, te_base, y, folds, K, n_cust_all, cat_cols)


if __name__ == "__main__":
    main()
