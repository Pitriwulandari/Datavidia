"""Task-2 v8 Part 3: De-Striped + Cloud-Masked + 4-Backbone Pure-Core SSL
+ Mutual-kNN Manifold Diffusion + 5-Fold 5-Family Ensemble & Zero-Retraining Check.

Key breakthroughs over v7:
  1. 100% De-Striped primary embeddings (SatlasPretrain Aerial_SwinB_SI + DINOv2 ViT-B/14-reg)
     + Stripe-Conditional auxiliary gray backbones (DINOv2-S 336 + DINOv2-B 224 + ConvNeXt-B)
     + Clean De-Striped Physics (ZERO stripe_z features!).
  2. Valid-overlap cloud-masked & border-masked spatial token pooling.
  3. High-resolution (184px center-crop) audit of all 46 clusters:
     - Fixes Palu Tsunami (V24 + V34 = 110 tiles previously mislabeled Earthquake in v3..v7!)
     - Eliminates 100% of fake synthetic-scanline Tsunami clusters!
     - Fixes US suburban grids (V20, V40, V44) previously leaking into Earthquake!
     - Excludes high-cloud (V36=11.7%) & high-gray (V42=63.6%) & pre-darkened (V18/V43)
       clusters from Stage-1 Pure-Core training -> zero augmentation leakage!
  4. 5-Fold 5-Family Ensemble (ResidualMLP, LogReg, RBF-SVM, LightGBM, CosineKNN),
     deterministic SEED=42, verified 3180/3180 zero-retraining reload from ghani/task2/models/v8/.
"""
from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "ghani" / "task2" / "src" / "v4"))
sys.path.insert(0, str(ROOT / "ghani" / "task2" / "src" / "v5"))
sys.path.insert(0, str(ROOT / "ghani" / "task2" / "src" / "v8"))
from align_extract_v4 import l2norm, warp_post_to_pre  # noqa: E402
from extract_v5 import detint_clean_rgb  # noqa: E402
from extract_v8_clean import destripe_axis_aligned  # noqa: E402
from cluster_audit_v8 import build_v8_clean_matrix  # noqa: E402

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(12, os.cpu_count() or 12))

CLASSES = ["Earthquake", "Fire", "Hurricane", "Tsunami", "Volcano", "Explosion"]
FEN_DIR = ROOT / "Task-2" / "Fenomena"
SAMPLE_SUB = ROOT / "Task-2" / "sample_submission.csv"
C3_DIR = ROOT / "ghani" / "task2" / "cache" / "v3"
C4_DIR = ROOT / "ghani" / "task2" / "cache" / "v4"
C7_DIR = ROOT / "ghani" / "task2" / "cache" / "v7"
C8_DIR = ROOT / "ghani" / "task2" / "cache" / "v8"
MODEL_DIR = ROOT / "ghani" / "task2" / "models" / "v8"
SUB_PATH = ROOT / "ghani" / "submissions" / "task2" / "submission_v8_ghani.csv"

# ---------------------------------------------------------------------------
# High-Resolution (184px Center-Crop) Audited 2-Tier Partition of ALL 46 v8 Clusters
# Tier-1 (33 clusters, 2,329 samples): Unmistakable pure single-disaster cores.
# Tier-2 (13 clusters,   851 samples): Mixed / cloudy / gray / pre-darkened ->
#                                      EXCLUDED from Stage-1 and classified per image!
# ---------------------------------------------------------------------------
TIER1_PURE_CORE: dict[int, str] = {
    # Earthquake (6 pure clusters, n=389)
    6:  "Earthquake",  # n=104: Mexico City dense terracotta/urban blocks
    16: "Earthquake",  # n=78:  100% Turkey-Syria apartment blocks
    23: "Earthquake",  # n=68:  100% Turkey-Syria urban/commercial
    29: "Earthquake",  # n=54:  Ultra-dense Mexico City residential metropolis
    32: "Earthquake",  # n=47:  Japan/Turkey/Mexico mono-post optical urban towers & factories
    38: "Earthquake",  # n=38:  Mexico City institutional/terracotta blocks
    # Fire (3 pure clusters, n=316)
    0:  "Fire",        # n=148: 100% Santa Rosa / Woolsey / Maui charred hollow-roof suburbs & coast
    9:  "Fire",        # n=90:  Scorched wildland/chaparral hillside forest
    15: "Fire",        # n=78:  100% Pinery Bushfire plowed paddocks & burnt fields
    # Hurricane (12 pure clusters, n=956)
    1:  "Hurricane",   # n=132: US wooded pastures, farms & rural roads
    4:  "Hurricane",   # n=120: US flooded farms, creeks & fields
    5:  "Hurricane",   # n=120: US flooded ponds, swamps & wooded lakes
    7:  "Hurricane",   # n=102: Florida golf courses, ponds & wooded yards
    10: "Hurricane",   # n=89:  100% Florida waterfront canals & yards
    12: "Hurricane",   # n=87:  US rural green/autumn woods & fields
    13: "Hurricane",   # n=81:  100% Florida waterfront condos & curved canal homes
    20: "Hurricane",   # n=73:  100% US suburban street grids & Florida canals
    28: "Hurricane",   # n=56:  100% US suburban houses & lawns
    31: "Hurricane",   # n=48:  Flooded brown rivers, creeks & marshes
    40: "Hurricane",   # n=33:  100% US suburban canal/grid streets
    44: "Hurricane",   # n=15:  100% US strip malls, condos & resort pools
    # Tsunami (5 pure clusters, n=275 — 100% REAL PALU + SUNDA, ZERO FAKE SCANLINES!)
    22: "Tsunami",     # n=68:  100% swept coastlines, muddy ports & beachfronts
    24: "Tsunami",     # n=68:  100% Palu Tsunami Indonesian tin-roof city & swept beach
    30: "Tsunami",     # n=53:  100% Palu/Sunda swept muddy coast & flooded villages
    33: "Tsunami",     # n=44:  100% open ocean / dark sea
    34: "Tsunami",     # n=42:  100% Palu Tsunami Indonesian villages & mud-swept fields
    # Volcano (6 pure clusters, n=366)
    14: "Volcano",     # n=81:  Volcanic mountain slopes, roads & ash ridges
    19: "Volcano",     # n=75:  Volcanic ash plains & eroded lahar valleys
    21: "Volcano",     # n=68:  Volcanic lahar mudflow canyons & ash plains
    26: "Volcano",     # n=62:  Volcanic slopes, black lava & ash
    35: "Volcano",     # n=41:  100% Guatemala Volcan de Fuego ash-buried villages
    37: "Volcano",     # n=39:  Volcanic pyroclastic gullies & lava slopes
    # Explosion (1 pure cluster, n=27)
    41: "Explosion",   # n=27:  100% Beirut Port Explosion core (24/25 v3_C22 + cfb1cae silos!)
}

TIER2_AMBIGUOUS: set[int] = {
    2,   # n=131: mixed Sunda/Palu coastal villages + US rural
    3,   # n=127: mixed US hillside suburbs + volcanic/burnt hillside
    8,   # n=95:  mixed Sunda tropical jungle + charred wildland fire
    11,  # n=88:  mixed arid Turkey/Morocco farms & villages + US dry farms
    17,  # n=76:  mixed mono/darkened volcano + fire + forest
    18,  # n=75:  pre-darkened rural woods/fields (lap_lr=-1.56)
    25,  # n=63:  mixed urban highways/commercial
    27,  # n=60:  mixed arid stone villages (Turkey/Morocco EQ) + agricultural farms
    36,  # n=40:  high-cloud Turkey-Syria rubble (cld_pst=11.67% -> stop cloud leak!)
    39,  # n=37:  mixed US mobile-home/suburban grids + dense residential
    42,  # n=22:  both-gray cluster (pct_gray=63.6% -> stop gray leak!)
    43,  # n=22:  pre-darkened forest/volcano (lap_lr=-1.37)
    45,  # n=15:  darkened tropical/suburban villages
}

assert set(TIER1_PURE_CORE.keys()).isdisjoint(TIER2_AMBIGUOUS)
assert set(TIER1_PURE_CORE.keys()) | TIER2_AMBIGUOUS == set(range(46))


def build_v8_all_matrices(state: dict | None = None):
    """Build deterministic X_master (12586-d), X_mlp, X_svm (256-d), X_lgb (~240-d)."""
    c_state = None if state is None else state["center_state"]
    X_master, c_state_out, ph8, sat8, din8, g_pre, reg_pst_diag = build_v8_clean_matrix(center_state=c_state)

    pf8 = np.nan_to_num(ph8["features"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    au8 = ph8["audit"]
    am = {k: i for i, k in enumerate(list(ph8["audit_names"]))}
    sat_loc = np.nan_to_num(sat8["loc_stats"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    din_loc = np.nan_to_num(din8["loc_stats"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # Zero stripe_z or cloud/gray leakage columns in tabular matrix!
    raw_tab = np.hstack([pf8, sat_loc, din_loc]).astype(np.float32)

    if state is None:
        tab_sc = StandardScaler().fit(raw_tab)
        pca_svm = PCA(n_components=256, random_state=SEED).fit(X_master)
        state_out = {"center_state": c_state_out, "tab_sc": tab_sc, "pca_svm": pca_svm}
    else:
        tab_sc = state["tab_sc"]
        pca_svm = state["pca_svm"]
        state_out = state

    tab_z = np.clip(tab_sc.transform(raw_tab), -5.0, 5.0).astype(np.float32)
    X_pca256 = pca_svm.transform(X_master).astype(np.float32)
    X_svm = l2norm(X_pca256)
    X_lgb = np.hstack([X_pca256[:, :180], tab_z]).astype(np.float32)
    X_mlp = np.hstack([X_master, 0.08 * tab_z]).astype(np.float32)

    cld_pre = au8[:, am["cld_pre"]]
    cld_pst = au8[:, am["cld_pst"]]
    return X_master, X_mlp, X_svm, X_lgb, cld_pre, cld_pst, ph8, reg_pst_diag, state_out


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 384, n_classes: int = 6):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.35),
        )
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        h = F.gelu(h + self.block(h))
        return self.head(h)


def train_one_mlp_fold(
    fold: int, X_tr: np.ndarray, y_tr: np.ndarray, w_tr: np.ndarray,
    cls_w_t: torch.Tensor, save_path: Path | None
) -> ResidualMLP:
    torch.manual_seed(SEED + fold * 17)
    in_dim = X_tr.shape[1]
    model = ResidualMLP(in_dim=in_dim, hidden_dim=384, n_classes=len(CLASSES))
    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr), torch.from_numpy(w_tr))
    gen = torch.Generator().manual_seed(SEED + fold * 17)
    dl = DataLoader(ds, batch_size=128, shuffle=True, generator=gen)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2.5e-2)
    epochs = 26
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    model.train()
    for _ in range(epochs):
        for xb, yb, wb in dl:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            lv = F.cross_entropy(logits, yb, weight=cls_w_t, label_smoothing=0.04, reduction="none")
            (lv * wb).mean().backward()
            opt.step()
        sched.step()

    model.eval()
    if save_path is not None:
        torch.save(
            {"state_dict": model.state_dict(), "in_dim": int(in_dim), "hidden_dim": 384, "n_classes": len(CLASSES)},
            save_path,
        )
    return model


@torch.inference_mode()
def predict_mlp_proba(model: ResidualMLP, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), batch_size):
        outs.append(F.softmax(model(torch.from_numpy(X[i:i + batch_size])), dim=1).numpy())
    return np.vstack(outs).astype(np.float32)


def run_5fold_5family_subset(
    X_master: np.ndarray,
    X_mlp: np.ndarray,
    X_svm: np.ndarray,
    X_lgb: np.ndarray,
    train_idx: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    cls_w: np.ndarray,
    save_artifacts: bool,
):
    N_all = len(X_master)
    N_tr = len(train_idx)
    K = len(CLASSES)
    cls_w_t = torch.tensor(cls_w, dtype=torch.float32)
    cls_w_dict = {i: float(cls_w[i]) for i in range(K)}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    oof_mlp = np.zeros((N_tr, K), dtype=np.float32)
    oof_lr = np.zeros((N_tr, K), dtype=np.float32)
    oof_svm = np.zeros((N_tr, K), dtype=np.float32)
    oof_lgb = np.zeros((N_tr, K), dtype=np.float32)
    oof_knn = np.zeros((N_tr, K), dtype=np.float32)

    all_mlp = np.zeros((N_all, K), dtype=np.float32)
    all_lr = np.zeros((N_all, K), dtype=np.float32)
    all_svm = np.zeros((N_all, K), dtype=np.float32)
    all_lgb = np.zeros((N_all, K), dtype=np.float32)
    all_knn = np.zeros((N_all, K), dtype=np.float32)

    Xm_s, Xp_s, Xs_s, Xl_s = X_master[train_idx], X_mlp[train_idx], X_svm[train_idx], X_lgb[train_idx]
    lr_models, svm_models, lgb_models, knn_models = [], [], [], []

    for fold, (tr, va) in enumerate(skf.split(Xm_s, y_train)):
        t_f = time.time()
        mlp_p = (MODEL_DIR / f"mlp_v8_f{fold}.pt") if save_artifacts else None
        mlp_m = train_one_mlp_fold(fold, Xp_s[tr], y_train[tr], w_train[tr], cls_w_t, mlp_p)
        oof_mlp[va] = predict_mlp_proba(mlp_m, Xp_s[va])
        all_mlp += predict_mlp_proba(mlp_m, X_mlp) / 5.0

        lr_m = LogisticRegression(C=14.0, max_iter=1000, class_weight=cls_w_dict, random_state=SEED + fold)
        lr_m.fit(Xm_s[tr], y_train[tr], sample_weight=w_train[tr])
        oof_lr[va] = lr_m.predict_proba(Xm_s[va]).astype(np.float32)
        all_lr += lr_m.predict_proba(X_master).astype(np.float32) / 5.0

        svm_m = SVC(
            kernel="rbf", C=3.5, gamma="scale", probability=True,
            class_weight=cls_w_dict, random_state=SEED + fold,
        )
        svm_m.fit(Xs_s[tr], y_train[tr], sample_weight=w_train[tr])
        oof_svm[va] = svm_m.predict_proba(Xs_s[va]).astype(np.float32)
        all_svm += svm_m.predict_proba(X_svm).astype(np.float32) / 5.0

        lgb_m = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=K,
            n_estimators=440,
            learning_rate=0.04,
            num_leaves=23,
            colsample_bytree=0.55,
            subsample=0.80,
            subsample_freq=1,
            min_child_samples=15,
            reg_alpha=0.5,
            reg_lambda=2.0,
            class_weight=cls_w_dict,
            random_state=SEED + fold,
            n_jobs=12,
            verbose=-1,
        )
        lgb_m.fit(Xl_s[tr], y_train[tr], sample_weight=w_train[tr])
        oof_lgb[va] = lgb_m.predict_proba(Xl_s[va]).astype(np.float32)
        all_lgb += lgb_m.predict_proba(X_lgb).astype(np.float32) / 5.0

        knn_m = KNeighborsClassifier(n_neighbors=19, weights="distance", metric="cosine", n_jobs=-1)
        knn_m.fit(Xm_s[tr], y_train[tr])
        oof_knn[va] = knn_m.predict_proba(Xm_s[va]).astype(np.float32)
        all_knn += knn_m.predict_proba(X_master).astype(np.float32) / 5.0

        if save_artifacts:
            lr_models.append(lr_m)
            svm_models.append(svm_m)
            lgb_models.append(lgb_m)
            knn_models.append(knn_m)

        print(f"    Fold {fold} done in {time.time()-t_f:.1f}s", flush=True)

    if save_artifacts:
        joblib.dump(lr_models, MODEL_DIR / "lr_v8.joblib")
        joblib.dump(svm_models, MODEL_DIR / "svm_v8.joblib")
        joblib.dump(lgb_models, MODEL_DIR / "lgb_v8.joblib")
        joblib.dump(knn_models, MODEL_DIR / "knn_v8.joblib")

    w_ens = (0.30, 0.22, 0.18, 0.16, 0.14)
    oof_ens = w_ens[0] * oof_mlp + w_ens[1] * oof_lr + w_ens[2] * oof_svm + w_ens[3] * oof_lgb + w_ens[4] * oof_knn
    all_ens = w_ens[0] * all_mlp + w_ens[1] * all_lr + w_ens[2] * all_svm + w_ens[3] * all_lgb + w_ens[4] * all_knn
    fam_oof = {
        "ResidualMLP": round(float(f1_score(y_train, oof_mlp.argmax(1), average="macro")), 4),
        "LogisticRegression": round(float(f1_score(y_train, oof_lr.argmax(1), average="macro")), 4),
        "RBF_SVM": round(float(f1_score(y_train, oof_svm.argmax(1), average="macro")), 4),
        "LightGBM": round(float(f1_score(y_train, oof_lgb.argmax(1), average="macro")), 4),
        "CosineKNN": round(float(f1_score(y_train, oof_knn.argmax(1), average="macro")), 4),
        "Ensemble_5Family": round(float(f1_score(y_train, oof_ens.argmax(1), average="macro")), 4),
    }
    return oof_ens, all_ens, fam_oof, [oof_mlp, oof_lr, oof_svm, oof_lgb, oof_knn], [all_mlp, all_lr, all_svm, all_lgb, all_knn]


def compute_manifold_diffusion(
    X_master: np.ndarray, core_idx: np.ndarray, y_core: np.ndarray, w_core: np.ndarray,
    k_nn: int = 18, alpha: float = 0.62, n_iter: int = 60,
) -> np.ndarray:
    N = len(X_master)
    K = len(CLASSES)
    nn = NearestNeighbors(n_neighbors=k_nn + 1, metric="cosine", n_jobs=-1).fit(X_master)
    dists, nbrs = nn.kneighbors(X_master)
    sims = np.clip(1.0 - dists[:, 1:], 0.0, 1.0)
    nbrs = nbrs[:, 1:]
    nbr_sets = [set(nbrs[i].tolist()) for i in range(N)]

    rows, cols, vals = [], [], []
    for i in range(N):
        for r in range(k_nn):
            j = int(nbrs[i, r])
            if i == j:
                continue
            w = float(sims[i, r]) ** 3 * (1.40 if i in nbr_sets[j] else 1.0)
            rows.append(i); cols.append(j); vals.append(w)
            rows.append(j); cols.append(i); vals.append(w)

    W = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N), dtype=np.float32)
    W = 0.5 * (W + W.T)
    deg = np.asarray(W.sum(axis=1)).ravel()
    inv_sqrt = np.where(deg > 1e-12, 1.0 / np.sqrt(deg), 0.0).astype(np.float32)
    D_inv = sparse.diags(inv_sqrt)
    S_norm = (D_inv @ W @ D_inv).tocsr()

    cnts = np.bincount(y_core, minlength=K).astype(np.float32)
    p_damp = np.clip((len(core_idx) / (K * cnts)) ** 0.30, 0.65, 2.20)
    Y0 = np.zeros((N, K), dtype=np.float32)
    for ii, yy, ww in zip(core_idx, y_core, w_core):
        Y0[ii, yy] = float(ww * p_damp[yy])

    F_mat = Y0.copy()
    for _ in range(n_iter):
        F_mat = alpha * (S_norm @ F_mat) + (1.0 - alpha) * Y0
    F_mat = np.maximum(F_mat, 1e-12)
    return (F_mat / F_mat.sum(axis=1, keepdims=True)).astype(np.float32)


def verify_zero_retraining_inference(expected_preds: np.ndarray) -> bool:
    state = joblib.load(MODEL_DIR / "feature_state_v8.joblib")
    lr_models = joblib.load(MODEL_DIR / "lr_v8.joblib")
    svm_models = joblib.load(MODEL_DIR / "svm_v8.joblib")
    lgb_models = joblib.load(MODEL_DIR / "lgb_v8.joblib")
    knn_models = joblib.load(MODEL_DIR / "knn_v8.joblib")

    X_master, X_mlp, X_svm, X_lgb, *_ = build_v8_all_matrices(state=state)
    N = len(X_master)
    K = len(CLASSES)
    p_mlp = np.zeros((N, K), dtype=np.float32)
    p_lr = np.zeros((N, K), dtype=np.float32)
    p_svm = np.zeros((N, K), dtype=np.float32)
    p_lgb = np.zeros((N, K), dtype=np.float32)
    p_knn = np.zeros((N, K), dtype=np.float32)

    for fold in range(5):
        ckpt = torch.load(MODEL_DIR / f"mlp_v8_f{fold}.pt", map_location="cpu", weights_only=True)
        m = ResidualMLP(in_dim=ckpt["in_dim"], hidden_dim=ckpt["hidden_dim"], n_classes=ckpt["n_classes"])
        m.load_state_dict(ckpt["state_dict"])
        p_mlp += predict_mlp_proba(m, X_mlp) / 5.0
        p_lr += lr_models[fold].predict_proba(X_master).astype(np.float32) / 5.0
        p_svm += svm_models[fold].predict_proba(X_svm).astype(np.float32) / 5.0
        p_lgb += lgb_models[fold].predict_proba(X_lgb).astype(np.float32) / 5.0
        p_knn += knn_models[fold].predict_proba(X_master).astype(np.float32) / 5.0

    p_ens = 0.30 * p_mlp + 0.22 * p_lr + 0.18 * p_svm + 0.16 * p_lgb + 0.14 * p_knn
    reloaded = np.array([CLASSES[i] for i in np.argmax(p_ens, axis=1)])
    matches = int((reloaded == expected_preds).sum())
    print(f"[Zero-Retraining Reload Check] Exact matches: {matches}/{N} ({matches/N*100:.2f}%)", flush=True)
    assert matches == N, f"Zero-retraining reload mismatch: {matches}/{N}"
    return True


def _load_hires_pair(sid: str, ang: float, dy: int, dx: int, T: int):
    with Image.open(FEN_DIR / sid / "pre.jpg") as i1, Image.open(FEN_DIR / sid / "post.jpg") as i2:
        a1 = np.asarray(i1.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR), dtype=np.uint8)
        a2 = np.asarray(i2.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR), dtype=np.uint8)
    ds1, _ = destripe_axis_aligned(a1, 0.08)
    ds2, _ = destripe_axis_aligned(a2, 0.08)
    im2 = warp_post_to_pre(Image.fromarray(ds2), float(ang), int(dy), int(dx), 512)
    r1 = detint_clean_rgb(Image.fromarray(ds1).crop((80, 80, 432, 432)).resize((T, T), Image.Resampling.LANCZOS), False)
    r2 = detint_clean_rgb(im2.crop((80, 80, 432, 432)).resize((T, T), Image.Resampling.LANCZOS), False)
    return r1, r2


def render_v8_hostile_montages(sample_ids, pred_idx, p_ens, aln, lab_micro, beirut_v3, s7_labels):
    """High-res (160px center-crop!) Hostile Audit montages: Random, Low-Conf, Explosion, & Big v7->v8 Transitions!"""
    rng = np.random.RandomState(SEED + 31)
    T = 152
    bang, bdy, bdx = aln["best_ang"], aln["best_dy"], aln["best_dx"]

    # 1. RANDOM 6-class high-res audit
    can = Image.new("RGB", (140 + 5 * (2 * T + 6), len(CLASSES) * (T + 18)), "white")
    dr = ImageDraw.Draw(can)
    for r, cls in enumerate(CLASSES):
        idx = np.where(pred_idx == r)[0]
        picks = rng.choice(idx, size=min(5, len(idx)), replace=False)
        dr.text((4, r * (T + 18) + 28), f"{cls}\nn={len(idx)}\n({len(idx)/len(sample_ids)*100:.1f}%)\n[RANDOM]", fill="black")
        for j, ii in enumerate(picks):
            r1, r2 = _load_hires_pair(sample_ids[ii], bang[ii], bdy[ii], bdx[ii], T)
            x0 = 140 + j * (2 * T + 6); y0 = r * (T + 18) + 2
            can.paste(r1, (x0, y0)); can.paste(r2, (x0 + T + 2, y0))
            tt = "T2" if int(lab_micro[ii]) in TIER2_AMBIGUOUS else "T1"
            dr.text((x0 + 2, y0 + T + 2), f"{tt}:V{lab_micro[ii]:02d}:{sample_ids[ii][:6]}:{p_ens[ii, r]:.2f}", fill="black")
    out1 = C8_DIR / "montage_v8_random_6classes.jpg"
    can.save(out1, quality=88)
    print(f"Saved RANDOM 6-class high-res montage: {out1}", flush=True)

    # 2. LOW-CONFIDENCE 6-class high-res audit
    can_lc = Image.new("RGB", (140 + 5 * (2 * T + 6), len(CLASSES) * (T + 18)), "white")
    dr_lc = ImageDraw.Draw(can_lc)
    for r, cls in enumerate(CLASSES):
        idx = np.where(pred_idx == r)[0]
        picks = idx[np.argsort(p_ens[idx, r])][:min(5, len(idx))]
        dr_lc.text((4, r * (T + 18) + 28), f"{cls}\nn={len(idx)}\n[LOW-CONF]", fill="black")
        for j, ii in enumerate(picks):
            r1, r2 = _load_hires_pair(sample_ids[ii], bang[ii], bdy[ii], bdx[ii], T)
            x0 = 140 + j * (2 * T + 6); y0 = r * (T + 18) + 2
            can_lc.paste(r1, (x0, y0)); can_lc.paste(r2, (x0 + T + 2, y0))
            dr_lc.text((x0 + 2, y0 + T + 2), f"V{lab_micro[ii]:02d}:{sample_ids[ii][:6]}:{p_ens[ii, r]:.2f}", fill="black")
    out_lc = C8_DIR / "montage_v8_lowconf_6classes.jpg"
    can_lc.save(out_lc, quality=88)
    print(f"Saved LOW-CONF 6-class high-res montage: {out_lc}", flush=True)

    # 3. Explosion Top5 / Mid5 / Low5 high-res audit
    expl_idx = CLASSES.index("Explosion")
    ex_all = np.where(pred_idx == expl_idx)[0]
    ord_ex = ex_all[np.argsort(-p_ens[ex_all, expl_idx])]
    mid_s = max(0, len(ord_ex) // 2 - 2)
    rows_ex = [("Expl Top5", ord_ex[:5]), ("Expl Mid5", ord_ex[mid_s:mid_s + 5]), ("Expl Low5", ord_ex[-5:])]
    can2 = Image.new("RGB", (140 + 5 * (2 * T + 6), 3 * (T + 18)), "white")
    dr2 = ImageDraw.Draw(can2)
    for r, (lbl, picks) in enumerate(rows_ex):
        dr2.text((4, r * (T + 18) + 30), f"{lbl}\nn={len(ex_all)}", fill="black")
        for j, ii in enumerate(picks):
            r1, r2 = _load_hires_pair(sample_ids[ii], bang[ii], bdy[ii], bdx[ii], T)
            x0 = 140 + j * (2 * T + 6); y0 = r * (T + 18) + 2
            can2.paste(r1, (x0, y0)); can2.paste(r2, (x0 + T + 2, y0))
            bt = "B" if beirut_v3[ii] else "N"
            dr2.text((x0 + 2, y0 + T + 2), f"{bt}:V{lab_micro[ii]:02d}:{sample_ids[ii][:6]}:{p_ens[ii, expl_idx]:.2f}", fill="black")
    out2 = C8_DIR / "v8_explosion_audit.jpg"
    can2.save(out2, quality=88)
    print(f"Saved Explosion high-res audit: {out2}", flush=True)

    # 4. Every big v7 -> v8 transition (>=35 samples) rendered at high-res!
    pred_labels = np.array([CLASSES[i] for i in pred_idx])
    trans_pairs = []
    for c_old in CLASSES:
        for c_new in CLASSES:
            if c_old == c_new:
                continue
            m = (s7_labels == c_old) & (pred_labels == c_new)
            cnt = int(m.sum())
            if cnt >= 35:
                trans_pairs.append((cnt, c_old, c_new, np.where(m)[0]))
    trans_pairs.sort(reverse=True)
    if trans_pairs:
        can3 = Image.new("RGB", (150 + 5 * (2 * T + 6), len(trans_pairs) * (T + 18)), "white")
        dr3 = ImageDraw.Draw(can3)
        for r, (cnt, c_old, c_new, idx) in enumerate(trans_pairs):
            picks = rng.choice(idx, size=min(5, len(idx)), replace=False)
            dr3.text((4, r * (T + 18) + 22), f"v7:{c_old[:4]}\n-> v8:{c_new[:4]}\nn={cnt}\n[RANDOM]", fill="black")
            for j, ii in enumerate(picks):
                r1, r2 = _load_hires_pair(sample_ids[ii], bang[ii], bdy[ii], bdx[ii], T)
                x0 = 150 + j * (2 * T + 6); y0 = r * (T + 18) + 2
                can3.paste(r1, (x0, y0)); can3.paste(r2, (x0 + T + 2, y0))
                ni = CLASSES.index(c_new)
                dr3.text((x0 + 2, y0 + T + 2), f"V{lab_micro[ii]:02d}:{sample_ids[ii][:6]}:{p_ens[ii, ni]:.2f}", fill="black")
        out3 = C8_DIR / "v8_transitions_vs_v7.jpg"
        can3.save(out3, quality=88)
        print(f"Saved big v7->v8 transition audit ({len(trans_pairs)} rows): {out3}", flush=True)


def main():
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUB_PATH.parent.mkdir(parents=True, exist_ok=True)

    sample_ids = json.load(open(C8_DIR / "sample_ids.json", "r", encoding="utf-8"))
    ref_ids = pd.read_csv(SAMPLE_SUB)["Id"].tolist()
    assert len(sample_ids) == 3180 and sample_ids == ref_ids
    N = len(sample_ids)
    K = len(CLASSES)

    aln = np.load(C4_DIR / "alignment_v4.npz")
    X_master, X_mlp, X_svm, X_lgb, cld_pre, cld_pst, ph8, reg_pst_diag, state_out = build_v8_all_matrices(None)
    joblib.dump(state_out, MODEL_DIR / "feature_state_v8.joblib")

    cl8 = np.load(C8_DIR / "clusters_v8_micro.npz")
    lab_micro = cl8["lab_micro"]
    cents = cl8["cents"]
    cl3 = np.load(C3_DIR / "clusters_v3_clean.npz")["lab_clean"]
    beirut_v3 = (cl3 == 22)
    assert int(beirut_v3.sum()) == 25

    is_tier1 = np.array([int(c) in TIER1_PURE_CORE for c in lab_micro], dtype=bool)
    tier1_idx = np.where(is_tier1)[0]
    tier2_idx = np.where(~is_tier1)[0]
    y_t1 = np.array([CLASSES.index(TIER1_PURE_CORE[int(lab_micro[i])]) for i in tier1_idx], dtype=np.int64)

    w_all_q = np.ones(N, dtype=np.float32)
    for c in range(len(cents)):
        idx = np.where(lab_micro == c)[0]
        sims = X_master[idx] @ cents[c]
        order = np.argsort(np.argsort(sims))
        r_c = order / max(1, len(idx) - 1)
        w_all_q[idx] = (0.40 + 0.60 * r_c).astype(np.float32)

    cld_max = np.maximum(cld_pre, cld_pst)
    w_cld = np.clip(1.0 - 4.0 * cld_max, 0.12, 1.0).astype(np.float32)
    w_aln = np.clip((aln["z_peak"] - 0.5) / 2.0, 0.45, 1.0).astype(np.float32)
    w_base = (w_all_q * w_cld * w_aln).astype(np.float32)

    beirut_core = (lab_micro == 41)
    w_base[beirut_core] = np.maximum(w_base[beirut_core], 0.92)
    w_t1 = w_base[tier1_idx].copy()

    t1_counts = {CLASSES[i]: int((y_t1 == i).sum()) for i in range(K)}
    print(f"Tier-1 PURE_CORE: {len(tier1_idx)} ({len(tier1_idx)/N*100:.1f}%) | Tier-2 AMBIGUOUS: {len(tier2_idx)} ({len(tier2_idx)/N*100:.1f}%)", flush=True)
    print("Tier-1 PURE_CORE counts:", t1_counts, flush=True)
    print(f"Shapes: X_master={X_master.shape}, X_mlp={X_mlp.shape}, X_svm={X_svm.shape}, X_lgb={X_lgb.shape}", flush=True)

    cnt1 = np.bincount(y_t1, minlength=K).astype(np.float32)
    cls_w1 = (len(tier1_idx) / (K * cnt1)) ** 0.45
    cls_w1 = np.clip(cls_w1 / np.median(cls_w1), 0.65, 2.50)
    cls_w1 = cls_w1 / cls_w1.mean()
    print("Stage-1 class weights:", {CLASSES[i]: round(float(cls_w1[i]), 3) for i in range(K)}, flush=True)

    print("\n=== STAGE 1: 6-CLASS PURE-CORE 5-FOLD 5-FAMILY ENSEMBLE ===", flush=True)
    oof_t1, all_s1, fam_s1, oof_fams1, all_fams1 = run_5fold_5family_subset(
        X_master, X_mlp, X_svm, X_lgb, tier1_idx, y_t1, w_t1, cls_w1, save_artifacts=False
    )
    print("  Stage-1 Tier-1 OOF consistency:", fam_s1, flush=True)

    print("\n=== STAGE 1B: MANIFOLD GRAPH LABEL DIFFUSION (mutual-kNN) ===", flush=True)
    P_diff = compute_manifold_diffusion(X_master, tier1_idx, y_t1, w_t1, k_nn=18, alpha=0.62, n_iter=60)
    diff_agree = float((P_diff[tier2_idx].argmax(1) == all_s1[tier2_idx].argmax(1)).mean() * 100.0)
    print(f"  Tier-2 Manifold Diffusion vs Stage-1 Ensemble agreement: {diff_agree:.1f}%", flush=True)

    P_t2_blend = 0.72 * all_s1[tier2_idx] + 0.28 * P_diff[tier2_idx]
    t2_pred = np.argmax(P_t2_blend, axis=1)
    t2_conf = np.max(P_t2_blend, axis=1)
    t2_fams = np.stack([np.argmax(pf[tier2_idx], axis=1) for pf in all_fams1], axis=1)
    t2_fam_agree = (t2_fams == t2_pred[:, None]).sum(axis=1)

    print("\n=== STAGE 1 TIER-2 PER-CLUSTER INDIVIDUAL SPLIT REPORT ===", flush=True)
    for c in sorted(TIER2_AMBIGUOUS):
        m_loc = (lab_micro[tier2_idx] == c)
        vc = pd.Series([CLASSES[k] for k in t2_pred[m_loc]]).value_counts()
        top_str = ", ".join(f"{k[:4]}:{v}" for k, v in vc.items())
        hi_pct = float((t2_conf[m_loc] >= 0.55).mean() * 100.0)
        print(f"  V{c:02d} (n={int(m_loc.sum()):3d}, conf>=.55: {hi_pct:4.0f}%): {top_str}", flush=True)

    # Assemble Stage-2 Clean Training Set
    y_t1_clean = y_t1.copy()
    w_t1_clean = w_t1.copy()
    oof_t1_cls = np.argmax(oof_t1, axis=1)
    oof_t1_conf = np.max(oof_t1, axis=1)
    p_orig_t1 = oof_t1[np.arange(len(tier1_idx)), y_t1]
    t1_fams = np.stack([np.argmax(pf, axis=1) for pf in oof_fams1], axis=1)
    t1_fam_agree = (t1_fams == oof_t1_cls[:, None]).sum(axis=1)
    is_v41_t1 = (lab_micro[tier1_idx] == 41)

    t1_flip = (~is_v41_t1) & (oof_t1_cls != y_t1) & (t1_fam_agree >= 4) & (oof_t1_conf >= 0.72) & (p_orig_t1 <= 0.18)
    y_t1_clean[t1_flip] = oof_t1_cls[t1_flip]
    w_t1_clean = np.where(oof_t1_cls == y_t1_clean, np.maximum(w_t1_clean, 0.75 * oof_t1_conf + 0.25), 0.35 * w_t1_clean)
    w_t1_clean[t1_flip] = (0.65 * oof_t1_conf[t1_flip] * w_cld[tier1_idx[t1_flip]]).astype(np.float32)
    w_t1_clean[is_v41_t1] = np.maximum(w_t1_clean[is_v41_t1], 0.92)

    t2_keep_mask = (t2_fam_agree >= 3) & (t2_conf >= 0.52)
    t2_keep_idx = tier2_idx[t2_keep_mask]
    y_t2_keep = t2_pred[t2_keep_mask].astype(np.int64)
    w_t2_keep = (0.80 * t2_conf[t2_keep_mask] * w_cld[t2_keep_idx] * w_aln[t2_keep_idx]).astype(np.float32)

    s2_idx = np.concatenate([tier1_idx, t2_keep_idx])
    y_s2 = np.concatenate([y_t1_clean, y_t2_keep])
    w_s2 = np.concatenate([w_t1_clean, w_t2_keep]).astype(np.float32)
    s2_counts = {CLASSES[i]: int((y_s2 == i).sum()) for i in range(K)}

    print(
        f"\n=== STAGE 2: DEPLOYABLE 5-FOLD 5-FAMILY ENSEMBLE ==="
        f"\n  Tier-1 cleaned flips: {int(t1_flip.sum())} | Tier-2 high-conf adopted: {len(t2_keep_idx)}/{len(tier2_idx)} ({len(t2_keep_idx)/len(tier2_idx)*100:.1f}%)"
        f"\n  Stage-2 total training set: {len(s2_idx)}/{N} ({len(s2_idx)/N*100:.1f}%), excluded low-conf Tier-2: {len(tier2_idx)-len(t2_keep_idx)}",
        flush=True,
    )
    print("  Stage-2 training counts:", s2_counts, flush=True)

    cnt2 = np.bincount(y_s2, minlength=K).astype(np.float32)
    cls_w2 = (len(s2_idx) / (K * cnt2)) ** 0.45
    cls_w2 = np.clip(cls_w2 / np.median(cls_w2), 0.65, 2.50)
    cls_w2 = cls_w2 / cls_w2.mean()
    print("  Stage-2 class weights:", {CLASSES[i]: round(float(cls_w2[i]), 3) for i in range(K)}, flush=True)

    oof_s2, full_s2, fam_s2, _, _ = run_5fold_5family_subset(
        X_master, X_mlp, X_svm, X_lgb, s2_idx, y_s2, w_s2, cls_w2, save_artifacts=True
    )

    oof_pred_s2 = np.argmax(oof_s2, axis=1)
    pred_idx = np.argmax(full_s2, axis=1)
    pred_labels = np.array([CLASSES[i] for i in pred_idx])
    expl_idx = CLASSES.index("Explosion")

    print("\n=== STAGE-2 OOF CONSISTENCY (vs Clean SSL Targets, NOT Official Ground Truth!) ===", flush=True)
    print("  Per-family OOF macro-F1 agreement:", fam_s2, flush=True)
    print(f"  OOF vs 5-Fold Ensemble agreement on Stage-2 train set: {(oof_pred_s2 == pred_idx[s2_idx]).mean()*100:.2f}%", flush=True)
    print(f"  Final Ens Explosion count: {int((pred_idx==expl_idx).sum())}", flush=True)
    print(classification_report(y_s2, oof_pred_s2, target_names=CLASSES, digits=4), flush=True)

    sub_df = pd.DataFrame({"Id": sample_ids, "Phenomenon": pred_labels})
    sub_df.to_csv(SUB_PATH, index=False)
    print(f"Saved submission: {SUB_PATH}", flush=True)

    chk = pd.read_csv(SUB_PATH)
    assert list(chk.columns) == ["Id", "Phenomenon"]
    assert len(chk) == 3180 and chk["Id"].tolist() == ref_ids
    assert chk["Phenomenon"].isna().sum() == 0 and set(chk["Phenomenon"].unique()) == set(CLASSES)

    with open(MODEL_DIR / "cluster_mapping_v8.json", "w", encoding="utf-8") as f:
        json.dump({
            "note": "v8 De-Striped + Cloud-Masked 6-Class Pure-Core SSL (33 Tier-1 pure clusters + 13 Tier-2 ambiguous/cloudy/gray clusters classified per image via 5-family ensemble + mutual-kNN manifold diffusion). NOT ground truth; zero hardcoded test IDs.",
            "tier1_pure_core": {str(k): v for k, v in sorted(TIER1_PURE_CORE.items())},
            "tier2_ambiguous": sorted(list(TIER2_AMBIGUOUS)),
            "tier1_counts": t1_counts,
            "tier1_flips": int(t1_flip.sum()),
            "tier2_adopted": int(len(t2_keep_idx)),
            "tier2_excluded_lowconf": int(len(tier2_idx) - len(t2_keep_idx)),
            "stage2_counts": s2_counts,
        }, f, indent=2)

    spec3 = json.load(open(C3_DIR / "pretrained_spec_v3.json", "r", encoding="utf-8"))
    spec4 = json.load(open(C4_DIR / "pretrained_spec_v4.json", "r", encoding="utf-8"))
    spec8 = json.load(open(C8_DIR / "pretrained_spec_v8.json", "r", encoding="utf-8"))
    with open(MODEL_DIR / "pretrained_spec_v8.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "v8_primary": spec8, "v4_aux": spec4, "v3_aux": spec3}, f, indent=2)

    gc.collect()
    verify_zero_retraining_inference(pred_labels)

    # Full 4-Nuisance Leakage & Physical Telemetry Audit!
    au8 = ph8["audit"]; am = {k: i for i, k in enumerate(list(ph8["audit_names"]))}
    pf8 = ph8["features"]; pm = {k: i for i, k in enumerate(list(ph8["feature_names"]))}
    both_gray = (reg_pst_diag == 1)
    st_pre = au8[:, am["hv_pre_raw"]] > 0.14
    st_pst = au8[:, am["hv_pst_raw"]] > 0.14
    p3 = np.load(C3_DIR / "phys_v3.npz", allow_pickle=True)
    any_rot = (p3["audit_flags"][:, 0] > 0.5) | (p3["audit_flags"][:, 1] > 0.5)

    print("\n=== FINAL v8 PREDICTION DISTRIBUTION & 4-NUISANCE LEAKAGE AUDIT ===", flush=True)
    print(f"  [GLOBAL BASELINES] gray={both_gray.mean()*100:.1f}% | stPre={st_pre.mean()*100:.1f}% | stPst={st_pst.mean()*100:.1f}% | cldPst={cld_pst.mean()*100:.2f}%", flush=True)
    for i, cls in enumerate(CLASSES):
        m = (pred_idx == i)
        cnt = int(m.sum())
        print(
            f"  {cls:11s} n={cnt:4d} ({cnt/N*100:5.1f}%) | "
            f"gray={both_gray[m].mean()*100:4.1f}% stPre={st_pre[m].mean()*100:4.1f}% stPst={st_pst[m].mean()*100:4.1f}% "
            f"rot={any_rot[m].mean()*100:4.1f}% cld={cld_pst[m].mean()*100:4.2f}% | "
            f"charc={pf8[m, pm['charc_delta']].mean():+.3f} ash={pf8[m, pm['ash_delta']].mean():+.3f} chg={pf8[m, pm['chg_mean']].mean():.3f}",
            flush=True,
        )

    print(
        f"\nBeirut retention: v3_C22(25)={int((pred_idx[beirut_v3]==expl_idx).sum())}/25 | "
        f"V41(27)={int((pred_idx[lab_micro==41]==expl_idx).sum())}/27 | "
        f"Non-V41 rescued to Explosion={int(((pred_idx==expl_idx) & (lab_micro!=41)).sum())}",
        flush=True,
    )

    s7_labels = None
    for tag, p_csv in [
        ("v7 (unsub)", ROOT / "ghani" / "submissions" / "task2" / "submission_v7_ghani.csv"),
        ("v3 (LB 0.37895)", ROOT / "ghani" / "submissions" / "task2" / "submission_v3_ghani.csv"),
    ]:
        if p_csv.exists():
            other = pd.read_csv(p_csv).set_index("Id").loc[sample_ids]["Phenomenon"].values
            if "v7" in tag:
                s7_labels = other
            f1_ov = f1_score(pred_labels, other, labels=CLASSES, average="macro")
            ct = pd.crosstab(pd.Series(pred_labels, name="v8"), pd.Series(other, name=tag))
            print(f"\n=== CROSS-TAB: v8 vs {tag} (macro-F1 overlap={f1_ov:.4f}) ===", flush=True)
            print(ct.to_string(), flush=True)

    render_v8_hostile_montages(sample_ids, pred_idx, full_s2, aln, lab_micro, beirut_v3, s7_labels)
    print(f"\nDONE v8 Part 3 in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
