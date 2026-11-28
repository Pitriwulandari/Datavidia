"""Task-2 v7 Part 3: Satellite-Native 6-Class Pure-Core Semi-Supervised Learning
+ Manifold Graph Label Diffusion + 5-Fold 5-Family Ensemble & Zero-Retraining Check.

100% Pure-Vision & Compliant (FAQ Task-2 no. 2):
  - Pure-vision pretrained backbones ONLY (SatlasPretrain Aerial_SwinB_SI ODC-BY + DINOv2 Apache-2.0
    + ConvNeXt-Base Apache-2.0). No text/point/box prompts, no VLM/LLM/CLIP/SAM,
    no external labeled data, zero hardcoded test IDs.
  - Deterministic SEED = 42, 100% zero-retraining inference reload check (3180/3180 exact match).
  - Architecture:
      1. Grayscale-Conditional Mean-Centered Satellite-Native Fusion (14,960-d X_v7_master).
      2. Explicit 2-Tier Partition of 54 visually & physically audited micro-clusters:
         - Tier-1 PURE_CORE (36 pure clusters, 2,247 samples across all 6 classes)
         - Tier-2 AMBIGUOUS/MIXED/CLOUDY (18 clusters, 933 samples EXCLUDED from Stage-1!)
      3. Stage 1: 5-Fold 5-Family Ensemble (ResidualMLP, LogisticRegression, RBF-SVM, LightGBM,
         CosineKNN) trained STRICTLY on Tier-1 PURE_CORE + Manifold Graph Label Diffusion on the
         mutual-kNN graph -> classifies every Tier-2 ambiguous tile individually per image.
      4. Stage 2: Deployable 5-Fold 5-Family Ensemble trained on cleaned Tier-1 + high-confidence
         multi-family consensus Tier-2 samples, saved to ghani/task2/models/v7/ and verified
         3180/3180 exact reload match.
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
sys.path.insert(0, str(ROOT / "ghani" / "task2" / "src" / "v7"))
from align_extract_v4 import l2norm, warp_post_to_pre  # noqa: E402
from extract_v5 import detint_clean_rgb  # noqa: E402
from cluster_audit_v7 import build_v7_master_matrix  # noqa: E402

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
C5_DIR = ROOT / "ghani" / "task2" / "cache" / "v5"
C7_DIR = ROOT / "ghani" / "task2" / "cache" / "v7"
MODEL_DIR = ROOT / "ghani" / "task2" / "models" / "v7"
SUB_PATH = ROOT / "ghani" / "submissions" / "task2" / "submission_v7_ghani.csv"

# ---------------------------------------------------------------------------
# Visually & Physically Audited 2-Tier Partition of ALL 54 v7 Micro-Clusters
# Tier-1 (36 clusters, 2,247 samples): Unmistakable single-disaster cores.
# Tier-2 (18 clusters, 933 samples):   Mixed / ambiguous / cloudy -> EXCLUDED
#                                      from Stage-1 so every tile is classified
#                                      individually per image!
# ---------------------------------------------------------------------------
TIER1_PURE_CORE: dict[int, str] = {
    # Earthquake (9 pure clusters, n=401)
    18: "Earthquake",  # n=68: dense terracotta/urban blocks
    25: "Earthquake",  # n=56: ultra-dense urban metropolis
    28: "Earthquake",  # n=52: 100% Turkey-Syria terracotta apartment blocks
    29: "Earthquake",  # n=50: 100% Turkey-Syria high-rise rubble towers
    32: "Earthquake",  # n=48: dense concrete towers & urban rubble
    37: "Earthquake",  # n=41: Turkey-Syria apartment blocks & courtyards
    38: "Earthquake",  # n=41: Mexico City / dense concrete residential grids
    52: "Earthquake",  # n=23: dense hillside terracotta/concrete city
    53: "Earthquake",  # n=22: 100% Turkey-Syria red-roofed apartment towers
    # Fire (3 pure clusters, n=315)
    0:  "Fire",        # n=126: 100% Santa Rosa / Woolsey charred hollow-roof burnt suburbs
    2:  "Fire",        # n=106: scorched wildfire chaparral/hillside forest
    10: "Fire",        # n=83:  100% Pinery Bushfire plowed paddocks (st_pre=4.74, st_del=+0.05)
    # Hurricane (13 pure clusters, n=919)
    3:  "Hurricane",   # n=101: US rural farms/creeks/suburbs
    4:  "Hurricane",   # n=100: flooded US creeks & wooded pastures
    7:  "Hurricane",   # n=84:  US rural woods & swollen rivers
    8:  "Hurricane",   # n=84:  US suburbs, golf courses & roundabouts
    9:  "Hurricane",   # n=81:  US flooded highways & farmland
    12: "Hurricane",   # n=76:  US flooded dark-water fields & creeks
    14: "Hurricane",   # n=76:  100% Florida waterfront canals
    17: "Hurricane",   # n=72:  US wooded suburban streets
    22: "Hurricane",   # n=63:  flooded brown rivers & creeks
    24: "Hurricane",   # n=60:  US canal & curved suburbs
    26: "Hurricane",   # n=54:  dark flooded ponds & swamps
    41: "Hurricane",   # n=38:  100% Florida lakes, ponds & suburban homes
    48: "Hurricane",   # n=30:  US golf courses & suburban estates
    # Tsunami (4 pure clusters, n=226)
    13: "Tsunami",     # n=76:  100% swept coastlines & open/striped ocean (h_asym=0.072)
    20: "Tsunami",     # n=66:  100% Sunda Strait post-only scanlines (st_del=+2.97)
    27: "Tsunami",     # n=54:  100% Sunda Strait villages & Palu coast (st_del=+1.47)
    47: "Tsunami",     # n=30:  100% Sunda Strait sea & coast scanlines (st_del=+2.20)
    # Volcano (6 pure clusters, n=359)
    1:  "Volcano",     # n=124: volcanic mountain slopes, ridges & lahar gullies
    21: "Volcano",     # n=65:  volcanic valleys, grey ash mudflows & eroded ridges
    23: "Volcano",     # n=61:  volcanic lahar channels & smooth ash plains
    40: "Volcano",     # n=38:  Lower Puna black basalt lava & volcanic ash ridges
    42: "Volcano",     # n=38:  100% Guatemala Volcan de Fuego ash-buried villages
    45: "Volcano",     # n=33:  volcanic ash plains & smooth lahar mudflows
    # Explosion (1 pure cluster, n=27)
    49: "Explosion",   # n=27:  100% Beirut Port Explosion core (23/25 v3_C22, b_sim=0.714)
}

TIER2_AMBIGUOUS: set[int] = {
    5,   # n=91: mixed Lower Puna lava (Volcano) + charred hills (Fire)
    6,   # n=89: mixed US rural (Hurricane) + arid village (Earthquake) + Volcano
    11,  # n=78: mixed Pinery paddocks (Fire) + Sunda scanlines (Tsunami)
    15,  # n=73: mixed US wooded suburbs (Hurricane) + village/trees
    16,  # n=72: mixed Indonesian/Palu village & swept mud (Earthquake vs Tsunami)
    19,  # n=67: mixed hillside burnt houses (Fire) + terraced slopes (Volcano) + urban
    30,  # n=51: mixed urban waterfront (Hurricane + Earthquake + Beirut Explosion!)
    31,  # n=48: high-cloud tropical coast (cld_post=5.93%) -> excluded to stop cloud leak
    33,  # n=46: mixed urban industrial/commercial (Earthquake + coastal Tsunami + Hurricane)
    34,  # n=43: mixed Palu swept mud (Tsunami) + tropical urban grids (Earthquake)
    35,  # n=43: mixed tropical village/urban (Earthquake) + Palu coastal (Tsunami)
    36,  # n=41: mixed wildland forest (Fire vs Volcano ash)
    39,  # n=39: mixed dense urban houses (Earthquake) + US suburban pools (Hurricane)
    43,  # n=35: mixed Florida waterfront (Hurricane) + swept coastline (Tsunami)
    44,  # n=34: mixed wooded suburbs / tropical trees (Hurricane / Volcano)
    46,  # n=31: mixed US suburban grids (Hurricane) + denser residential / wildfire edge
    50,  # n=27: mixed Palu coastal/urban (Earthquake / Tsunami, highest pct_gray=29.6%)
    51,  # n=25: dense white-roofed urban (Earthquake + borderline Beirut Explosion, b_sim=0.460)
}

assert set(TIER1_PURE_CORE.keys()).isdisjoint(TIER2_AMBIGUOUS)
assert set(TIER1_PURE_CORE.keys()) | TIER2_AMBIGUOUS == set(range(54))


def build_v7_all_matrices(state: dict | None = None):
    """Build deterministic X_master (14960-d), X_mlp, X_svm (256-d), X_lgb (~388-d)."""
    c_state = None if state is None else state["center_state"]
    X_master, c_state_out, p5, ap4, p3, sat7 = build_v7_master_matrix(
        center_state=c_state, w_f2=1.50, w_f3=0.50, w_top=0.60
    )

    p5_names = list(p5["feature_names"])
    ap_names = list(ap4["feature_names"])
    p3_names = list(p3["feature_names"])
    p3_map = {k: i for i, k in enumerate(p3_names)}

    pf5 = np.nan_to_num(p5["features"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    apf = np.nan_to_num(ap4["features"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pf3 = np.nan_to_num(p3["features"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    drop_p3 = {
        "valid_frac_pre", "valid_frac_post", "cloud_frac_pre", "cloud_frac_post",
        "blk_frac_pre", "blk_frac_post", "is_gray_pre", "is_gray_post",
    }
    keep_p3 = [i for i, k in enumerate(p3_names) if k not in drop_p3]
    vs4 = np.load(C4_DIR / "aligned_vits_336.npz")
    vb4 = np.load(C4_DIR / "aligned_vitb_224.npz")
    vb5 = np.load(C5_DIR / "detint_vitb_224.npz")
    sat_loc = np.nan_to_num(sat7["loc_stats"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    raw_tab = np.hstack([
        pf5, apf, pf3[:, keep_p3], sat_loc, vb5["loc_stats"], vs4["loc_stats"], vb4["loc_stats"]
    ]).astype(np.float32)

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

    cld_pre = pf3[:, p3_map["cloud_frac_pre"]]
    cld_post = pf3[:, p3_map["cloud_frac_post"]]
    return X_master, X_mlp, X_svm, X_lgb, cld_pre, cld_post, pf5, p5_names, apf, ap_names, p3, state_out


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
    """Train 5-fold 5-family ensemble on `train_idx` and predict probabilities for ALL N samples."""
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
        mlp_p = (MODEL_DIR / f"mlp_v7_f{fold}.pt") if save_artifacts else None
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
        joblib.dump(lr_models, MODEL_DIR / "lr_v7.joblib")
        joblib.dump(svm_models, MODEL_DIR / "svm_v7.joblib")
        joblib.dump(lgb_models, MODEL_DIR / "lgb_v7.joblib")
        joblib.dump(knn_models, MODEL_DIR / "knn_v7.joblib")

    # Balanced 5-family weights (Neural + Linear + Kernel + Tree + Metric)
    w_ens = (0.30, 0.22, 0.18, 0.16, 0.14)
    oof_ens = (
        w_ens[0] * oof_mlp + w_ens[1] * oof_lr + w_ens[2] * oof_svm
        + w_ens[3] * oof_lgb + w_ens[4] * oof_knn
    )
    all_ens = (
        w_ens[0] * all_mlp + w_ens[1] * all_lr + w_ens[2] * all_svm
        + w_ens[3] * all_lgb + w_ens[4] * all_knn
    )
    fam_oof = {
        "ResidualMLP": round(float(f1_score(y_train, oof_mlp.argmax(1), average="macro")), 4),
        "LogisticRegression": round(float(f1_score(y_train, oof_lr.argmax(1), average="macro")), 4),
        "RBF_SVM": round(float(f1_score(y_train, oof_svm.argmax(1), average="macro")), 4),
        "LightGBM": round(float(f1_score(y_train, oof_lgb.argmax(1), average="macro")), 4),
        "CosineKNN": round(float(f1_score(y_train, oof_knn.argmax(1), average="macro")), 4),
        "Ensemble_5Family": round(float(f1_score(y_train, oof_ens.argmax(1), average="macro")), 4),
    }
    oof_fams = [oof_mlp, oof_lr, oof_svm, oof_lgb, oof_knn]
    all_fams = [all_mlp, all_lr, all_svm, all_lgb, all_knn]
    return oof_ens, all_ens, fam_oof, oof_fams, all_fams


def compute_manifold_diffusion(
    X_master: np.ndarray,
    core_idx: np.ndarray,
    y_core: np.ndarray,
    w_core: np.ndarray,
    k_nn: int = 18,
    alpha: float = 0.62,
    n_iter: int = 60,
) -> np.ndarray:
    """Symmetric normalized mutual-kNN harmonic label diffusion from Tier-1 PURE_CORE."""
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
            w = float(sims[i, r]) ** 3
            if i in nbr_sets[j]:
                w *= 1.40
            rows.append(i); cols.append(j); vals.append(w)
            rows.append(j); cols.append(i); vals.append(w)

    W = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N), dtype=np.float32)
    W = 0.5 * (W + W.T)
    deg = np.asarray(W.sum(axis=1)).ravel()
    inv_sqrt = np.where(deg > 1e-12, 1.0 / np.sqrt(deg), 0.0).astype(np.float32)
    D_inv = sparse.diags(inv_sqrt)
    S_norm = (D_inv @ W @ D_inv).tocsr()

    # Gentle prior damping (gamma=0.30, capped) so neither Hurricane swamps nor Explosion explodes
    cnts = np.bincount(y_core, minlength=K).astype(np.float32)
    p_damp = np.clip((len(core_idx) / (K * cnts)) ** 0.30, 0.65, 2.20)
    Y0 = np.zeros((N, K), dtype=np.float32)
    for ii, yy, ww in zip(core_idx, y_core, w_core):
        Y0[ii, yy] = float(ww * p_damp[yy])

    F_mat = Y0.copy()
    for _ in range(n_iter):
        F_mat = alpha * (S_norm @ F_mat) + (1.0 - alpha) * Y0

    # Clamp negative numerical noise & row-normalize
    F_mat = np.maximum(F_mat, 1e-12)
    P_diff = F_mat / F_mat.sum(axis=1, keepdims=True)
    return P_diff.astype(np.float32)


def verify_zero_retraining_inference(expected_preds: np.ndarray) -> bool:
    """Reload every saved .pt and .joblib artifact fresh from MODEL_DIR and assert 100% exact match."""
    state = joblib.load(MODEL_DIR / "feature_state_v7.joblib")
    lr_models = joblib.load(MODEL_DIR / "lr_v7.joblib")
    svm_models = joblib.load(MODEL_DIR / "svm_v7.joblib")
    lgb_models = joblib.load(MODEL_DIR / "lgb_v7.joblib")
    knn_models = joblib.load(MODEL_DIR / "knn_v7.joblib")

    X_master, X_mlp, X_svm, X_lgb, *_ = build_v7_all_matrices(state=state)
    N = len(X_master)
    K = len(CLASSES)
    p_mlp = np.zeros((N, K), dtype=np.float32)
    p_lr = np.zeros((N, K), dtype=np.float32)
    p_svm = np.zeros((N, K), dtype=np.float32)
    p_lgb = np.zeros((N, K), dtype=np.float32)
    p_knn = np.zeros((N, K), dtype=np.float32)

    for fold in range(5):
        ckpt = torch.load(MODEL_DIR / f"mlp_v7_f{fold}.pt", map_location="cpu", weights_only=True)
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


def render_v7_audit_montages(sample_ids, pred_idx, p_ens, aln, lab_micro, beirut_v3, s6_labels):
    """Render RANDOM 6-class montage, Explosion Top/Mid/Low audit, AND big v6->v7 transition audit!"""
    rng = np.random.RandomState(SEED + 21)
    T = 108
    bang, bdy, bdx = aln["best_ang"], aln["best_dy"], aln["best_dx"]

    # 1. RANDOM 6-class montage + LOW-CONFIDENCE row per class
    can = Image.new("RGB", (125 + 6 * (2 * T + 6), len(CLASSES) * (T + 16)), "white")
    dr = ImageDraw.Draw(can)
    for r, cls in enumerate(CLASSES):
        idx = np.where(pred_idx == r)[0]
        picks = rng.choice(idx, size=min(6, len(idx)), replace=False)
        dr.text((4, r * (T + 16) + 22), f"{cls}\nn={len(idx)}\n({len(idx)/len(sample_ids)*100:.1f}%)\n[RANDOM]", fill="black")
        for j, ii in enumerate(picks):
            sid = sample_ids[ii]
            with Image.open(FEN_DIR / sid / "pre.jpg") as i1, Image.open(FEN_DIR / sid / "post.jpg") as i2:
                r1 = detint_clean_rgb(i1.convert("RGB").resize((T, T), Image.Resampling.BILINEAR), False)
                r2 = detint_clean_rgb(warp_post_to_pre(i2.convert("RGB"), float(bang[ii]), int(bdy[ii]), int(bdx[ii]), T), False)
            x0 = 125 + j * (2 * T + 6)
            y0 = r * (T + 16) + 2
            can.paste(r1, (x0, y0))
            can.paste(r2, (x0 + T + 2, y0))
            t_tag = "T2" if int(lab_micro[ii]) in TIER2_AMBIGUOUS else "T1"
            dr.text((x0 + 2, y0 + T + 1), f"{t_tag}:S{lab_micro[ii]:02d}:{sid[:6]}:{p_ens[ii, r]:.2f}", fill="black")
    out1 = C7_DIR / "montage_v7_final_random_6classes.jpg"
    can.save(out1, quality=88)
    print(f"Saved RANDOM 6-class montage: {out1}", flush=True)

    # 2. LOW-CONFIDENCE 6-class audit (Hostile Audit Rule 4.3!)
    can_lc = Image.new("RGB", (125 + 6 * (2 * T + 6), len(CLASSES) * (T + 16)), "white")
    dr_lc = ImageDraw.Draw(can_lc)
    for r, cls in enumerate(CLASSES):
        idx = np.where(pred_idx == r)[0]
        ord_lc = idx[np.argsort(p_ens[idx, r])]
        picks = ord_lc[:min(6, len(ord_lc))]
        dr_lc.text((4, r * (T + 16) + 22), f"{cls}\nn={len(idx)}\n[LOW-CONF]", fill="black")
        for j, ii in enumerate(picks):
            sid = sample_ids[ii]
            with Image.open(FEN_DIR / sid / "pre.jpg") as i1, Image.open(FEN_DIR / sid / "post.jpg") as i2:
                r1 = detint_clean_rgb(i1.convert("RGB").resize((T, T), Image.Resampling.BILINEAR), False)
                r2 = detint_clean_rgb(warp_post_to_pre(i2.convert("RGB"), float(bang[ii]), int(bdy[ii]), int(bdx[ii]), T), False)
            x0 = 125 + j * (2 * T + 6)
            y0 = r * (T + 16) + 2
            can_lc.paste(r1, (x0, y0))
            can_lc.paste(r2, (x0 + T + 2, y0))
            dr_lc.text((x0 + 2, y0 + T + 1), f"S{lab_micro[ii]:02d}:{sid[:6]}:{p_ens[ii, r]:.2f}", fill="black")
    out_lc = C7_DIR / "montage_v7_lowconf_6classes.jpg"
    can_lc.save(out_lc, quality=88)
    print(f"Saved LOW-CONF 6-class montage: {out_lc}", flush=True)

    # 3. Explosion Top6 / Mid6 / Low6 audit
    expl_idx = CLASSES.index("Explosion")
    ex_all = np.where(pred_idx == expl_idx)[0]
    ord_ex = ex_all[np.argsort(-p_ens[ex_all, expl_idx])]
    mid_s = max(0, len(ord_ex) // 2 - 3)
    rows_ex = [("Expl Top6", ord_ex[:6]), ("Expl Mid6", ord_ex[mid_s:mid_s + 6]), ("Expl Low6", ord_ex[-6:])]
    can2 = Image.new("RGB", (125 + 6 * (2 * T + 6), 3 * (T + 16)), "white")
    dr2 = ImageDraw.Draw(can2)
    for r, (lbl, picks) in enumerate(rows_ex):
        dr2.text((4, r * (T + 16) + 26), f"{lbl}\nn={len(ex_all)}", fill="black")
        for j, ii in enumerate(picks):
            sid = sample_ids[ii]
            with Image.open(FEN_DIR / sid / "pre.jpg") as i1, Image.open(FEN_DIR / sid / "post.jpg") as i2:
                r1 = detint_clean_rgb(i1.convert("RGB").resize((T, T), Image.Resampling.BILINEAR), False)
                r2 = detint_clean_rgb(warp_post_to_pre(i2.convert("RGB"), float(bang[ii]), int(bdy[ii]), int(bdx[ii]), T), False)
            x0 = 125 + j * (2 * T + 6)
            y0 = r * (T + 16) + 2
            can2.paste(r1, (x0, y0))
            can2.paste(r2, (x0 + T + 2, y0))
            bt = "B" if beirut_v3[ii] else "N"
            dr2.text((x0 + 2, y0 + T + 1), f"{bt}:S{lab_micro[ii]:02d}:{sid[:6]}:{p_ens[ii, expl_idx]:.2f}", fill="black")
    out2 = C7_DIR / "v7_explosion_audit.jpg"
    can2.save(out2, quality=88)
    print(f"Saved Explosion audit montage: {out2}", flush=True)

    # 4. Hostile Audit: Every big v6 -> v7 transition (>=25 samples) rendered visually!
    pred_labels = np.array([CLASSES[i] for i in pred_idx])
    trans_pairs = []
    for c_old in CLASSES:
        for c_new in CLASSES:
            if c_old == c_new:
                continue
            m = (s6_labels == c_old) & (pred_labels == c_new)
            cnt = int(m.sum())
            if cnt >= 25:
                trans_pairs.append((cnt, c_old, c_new, np.where(m)[0]))
    trans_pairs.sort(reverse=True)
    if trans_pairs:
        can3 = Image.new("RGB", (145 + 6 * (2 * T + 6), len(trans_pairs) * (T + 16)), "white")
        dr3 = ImageDraw.Draw(can3)
        for r, (cnt, c_old, c_new, idx) in enumerate(trans_pairs):
            picks = rng.choice(idx, size=min(6, len(idx)), replace=False)
            dr3.text((4, r * (T + 16) + 18), f"v6:{c_old[:4]}\n-> v7:{c_new[:4]}\nn={cnt}\n[RANDOM]", fill="black")
            for j, ii in enumerate(picks):
                sid = sample_ids[ii]
                with Image.open(FEN_DIR / sid / "pre.jpg") as i1, Image.open(FEN_DIR / sid / "post.jpg") as i2:
                    r1 = detint_clean_rgb(i1.convert("RGB").resize((T, T), Image.Resampling.BILINEAR), False)
                    r2 = detint_clean_rgb(warp_post_to_pre(i2.convert("RGB"), float(bang[ii]), int(bdy[ii]), int(bdx[ii]), T), False)
                x0 = 145 + j * (2 * T + 6)
                y0 = r * (T + 16) + 2
                can3.paste(r1, (x0, y0))
                can3.paste(r2, (x0 + T + 2, y0))
                ni = CLASSES.index(c_new)
                dr3.text((x0 + 2, y0 + T + 1), f"S{lab_micro[ii]:02d}:{sid[:6]}:{p_ens[ii, ni]:.2f}", fill="black")
        out3 = C7_DIR / "v7_transitions_vs_v6.jpg"
        can3.save(out3, quality=88)
        print(f"Saved big v6->v7 transition audit ({len(trans_pairs)} rows): {out3}", flush=True)


def run_anchor_diagnostic(sample_ids, pred_labels):
    """Read-only diagnostic check on visually verified multi-event anchors (NEVER used in training!)."""
    id2i = {s: i for i, s in enumerate(sample_ids)}
    anchors = {
        "Tsunami":    ["6062a3a7", "7e8c19a", "f079ccd", "1fce27c", "380b5d8", "9d8231e", "fb0eba1", "4857cc6", "f3a9842", "b70b27b", "43e578d"],
        "Volcano":    ["eb0a14a", "3d3b0398", "9de751e", "46611e7", "d9e17cb", "5a5f496", "47cfaf0", "67052b1", "d69a2ad", "4288963"],
        "Fire":       ["f4965d34", "abb57b0", "11da794", "41e98f8", "31795bc", "1dff71b", "e1b956d", "bfcc30b", "2f1d841", "75aa976"],
        "Hurricane":  ["541c2e1", "f6cefe8", "2c438ca", "36d1091", "12b9452", "674da12", "46d3cea", "0bc54e8", "1e8c587", "76c97c0"],
        "Earthquake": ["dec05312", "038f53e4", "961fbe1", "d4967d5", "08c7a86c", "74419ba", "bd88fe7", "73d2b1f", "6e76e5d", "80a01d9"],
        "Explosion":  ["d3a1e358", "c963993", "99cb5f0", "73ed7a2", "6726106", "2b0fcc2", "55dc286", "e4a86ed", "b61f21d"],
    }
    print("\n=== READ-ONLY MULTI-EVENT VISUAL ANCHOR DIAGNOSTIC (v7) ===", flush=True)
    tot_ok = tot_n = 0
    for cls, prefs in anchors.items():
        ok = n = 0
        misses = []
        for pf in prefs:
            m = [s for s in sample_ids if s.startswith(pf)]
            if not m:
                continue
            n += 1
            pred = pred_labels[id2i[m[0]]]
            if pred == cls:
                ok += 1
            else:
                misses.append(f"{pf}->{pred}")
        tot_ok += ok
        tot_n += n
        print(f"  {cls:11s}: {ok:2d}/{n:2d}  misses={misses}", flush=True)
    print(f"  TOTAL ANCHORS: {tot_ok}/{tot_n} ({tot_ok/tot_n*100:.1f}%)", flush=True)


def main():
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUB_PATH.parent.mkdir(parents=True, exist_ok=True)

    sample_ids = json.load(open(C7_DIR / "sample_ids.json", "r", encoding="utf-8"))
    ref_ids = pd.read_csv(SAMPLE_SUB)["Id"].tolist()
    assert len(sample_ids) == 3180 and sample_ids == ref_ids
    N = len(sample_ids)
    K = len(CLASSES)

    aln = np.load(C4_DIR / "alignment_v4.npz")
    X_master, X_mlp, X_svm, X_lgb, cld_pre, cld_post, pf5, p5_names, apf, ap_names, p3, state_out = build_v7_all_matrices(None)
    joblib.dump(state_out, MODEL_DIR / "feature_state_v7.joblib")

    cl7 = np.load(C7_DIR / "clusters_v7_micro.npz")
    lab_micro = cl7["lab_micro"]
    cents = cl7["cents"]
    cl3 = np.load(C3_DIR / "clusters_v3_clean.npz")["lab_clean"]
    beirut_v3 = (cl3 == 22)
    assert int(beirut_v3.sum()) == 25

    p5_map = {k: i for i, k in enumerate(p5_names)}
    ap_map = {k: i for i, k in enumerate(ap_names)}

    # 1. Build Tier-1 PURE_CORE vs Tier-2 AMBIGUOUS masks
    is_tier1 = np.array([int(c) in TIER1_PURE_CORE for c in lab_micro], dtype=bool)
    tier1_idx = np.where(is_tier1)[0]
    tier2_idx = np.where(~is_tier1)[0]
    y_t1 = np.array([CLASSES.index(TIER1_PURE_CORE[int(lab_micro[i])]) for i in tier1_idx], dtype=np.int64)

    # Sample quality weights on Tier-1 (centroid rank + cloud suppression + alignment confidence)
    w_all_q = np.ones(N, dtype=np.float32)
    for c in range(len(cents)):
        idx = np.where(lab_micro == c)[0]
        sims = X_master[idx] @ cents[c]
        order = np.argsort(np.argsort(sims))
        r_c = order / max(1, len(idx) - 1)
        w_all_q[idx] = (0.40 + 0.60 * r_c).astype(np.float32)

    cld_max = np.maximum(cld_pre, cld_post)
    w_cld = np.clip(1.0 - 4.0 * cld_max, 0.12, 1.0).astype(np.float32)
    w_aln = np.clip((aln["z_peak"] - 0.5) / 2.0, 0.45, 1.0).astype(np.float32)
    w_base = (w_all_q * w_cld * w_aln).astype(np.float32)

    beirut_core = (lab_micro == 49)
    w_base[beirut_core] = np.maximum(w_base[beirut_core], 0.92)
    w_t1 = w_base[tier1_idx].copy()

    t1_counts = {CLASSES[i]: int((y_t1 == i).sum()) for i in range(K)}
    print(f"Tier-1 PURE_CORE: {len(tier1_idx)} ({len(tier1_idx)/N*100:.1f}%) | Tier-2 AMBIGUOUS: {len(tier2_idx)} ({len(tier2_idx)/N*100:.1f}%)", flush=True)
    print("Tier-1 PURE_CORE counts:", t1_counts, flush=True)
    print(f"Shapes: X_master={X_master.shape}, X_mlp={X_mlp.shape}, X_svm={X_svm.shape}, X_lgb={X_lgb.shape}", flush=True)

    # Moderate balanced class weights (capped so Explosion is protected without flooding Turkey urban!)
    cnt1 = np.bincount(y_t1, minlength=K).astype(np.float32)
    cls_w1 = (len(tier1_idx) / (K * cnt1)) ** 0.45
    cls_w1 = np.clip(cls_w1 / np.median(cls_w1), 0.65, 2.50)
    cls_w1 = cls_w1 / cls_w1.mean()
    print("Stage-1 class weights:", {CLASSES[i]: round(float(cls_w1[i]), 3) for i in range(K)}, flush=True)

    # 2. STAGE 1: Pure-Core 5-Fold 5-Family Ensemble + Manifold Graph Label Diffusion
    print("\n=== STAGE 1: 6-CLASS PURE-CORE 5-FOLD 5-FAMILY ENSEMBLE ===", flush=True)
    oof_t1, all_s1, fam_s1, oof_fams1, all_fams1 = run_5fold_5family_subset(
        X_master, X_mlp, X_svm, X_lgb, tier1_idx, y_t1, w_t1, cls_w1, save_artifacts=False
    )
    print("  Stage-1 Tier-1 OOF consistency:", fam_s1, flush=True)

    print("\n=== STAGE 1B: MANIFOLD GRAPH LABEL DIFFUSION (mutual-kNN) ===", flush=True)
    P_diff = compute_manifold_diffusion(X_master, tier1_idx, y_t1, w_t1, k_nn=18, alpha=0.62, n_iter=60)
    diff_agree = float((P_diff[tier2_idx].argmax(1) == all_s1[tier2_idx].argmax(1)).mean() * 100.0)
    print(f"  Tier-2 Manifold Diffusion vs Stage-1 Ensemble agreement: {diff_agree:.1f}%", flush=True)

    # Blend Stage-1 Ensemble (0.72) + Manifold Diffusion (0.28) on Tier-2
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
        print(f"  S{c:02d} (n={int(m_loc.sum()):2d}, conf>=.55: {hi_pct:4.0f}%): {top_str}", flush=True)

    # 3. Assemble Stage-2 Clean Training Set:
    #    - Tier-1 PURE_CORE (with rare 4-of-5 consensus outlier cleaning, S49 protected)
    #    - High-confidence Tier-2 tiles (>=3/5 families agree AND blend conf >= 0.52)
    y_t1_clean = y_t1.copy()
    w_t1_clean = w_t1.copy()
    oof_t1_cls = np.argmax(oof_t1, axis=1)
    oof_t1_conf = np.max(oof_t1, axis=1)
    p_orig_t1 = oof_t1[np.arange(len(tier1_idx)), y_t1]
    t1_fams = np.stack([np.argmax(pf, axis=1) for pf in oof_fams1], axis=1)
    t1_fam_agree = (t1_fams == oof_t1_cls[:, None]).sum(axis=1)
    is_s49_t1 = (lab_micro[tier1_idx] == 49)

    t1_flip = (~is_s49_t1) & (oof_t1_cls != y_t1) & (t1_fam_agree >= 4) & (oof_t1_conf >= 0.72) & (p_orig_t1 <= 0.18)
    y_t1_clean[t1_flip] = oof_t1_cls[t1_flip]
    w_t1_clean = np.where(oof_t1_cls == y_t1_clean, np.maximum(w_t1_clean, 0.75 * oof_t1_conf + 0.25), 0.35 * w_t1_clean)
    w_t1_clean[t1_flip] = (0.65 * oof_t1_conf[t1_flip] * w_cld[tier1_idx[t1_flip]]).astype(np.float32)
    w_t1_clean[is_s49_t1] = np.maximum(w_t1_clean[is_s49_t1], 0.92)

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

    # Save submission
    sub_df = pd.DataFrame({"Id": sample_ids, "Phenomenon": pred_labels})
    sub_df.to_csv(SUB_PATH, index=False)
    print(f"Saved submission: {SUB_PATH}", flush=True)

    chk = pd.read_csv(SUB_PATH)
    assert list(chk.columns) == ["Id", "Phenomenon"]
    assert len(chk) == 3180 and chk["Id"].tolist() == ref_ids
    assert chk["Phenomenon"].isna().sum() == 0 and set(chk["Phenomenon"].unique()) == set(CLASSES)

    with open(MODEL_DIR / "cluster_mapping_v7.json", "w", encoding="utf-8") as f:
        json.dump({
            "note": "6-Class Pure-Core SSL (36 Tier-1 pure clusters + 18 Tier-2 ambiguous clusters classified per image via Stage-1 5-family ensemble + mutual-kNN manifold diffusion). NOT ground truth; zero hardcoded test IDs.",
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
    spec5 = json.load(open(C5_DIR / "pretrained_spec_v5.json", "r", encoding="utf-8"))
    spec7 = json.load(open(C7_DIR / "pretrained_spec_v7.json", "r", encoding="utf-8"))
    with open(MODEL_DIR / "pretrained_spec_v7.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "v3": spec3, "v4": spec4, "v5": spec5, "v7": spec7}, f, indent=2)

    gc.collect()
    verify_zero_retraining_inference(pred_labels)

    # Final Distribution & Leakage Audit
    audit3 = p3["audit_flags"]
    both_gray = (audit3[:, 2] > 0.5) & (audit3[:, 3] > 0.5)
    any_rot = (audit3[:, 0] > 0.5) | (audit3[:, 1] > 0.5)
    print("\n=== FINAL v7 PREDICTION DISTRIBUTION & LEAKAGE AUDIT ===", flush=True)
    for i, cls in enumerate(CLASSES):
        m = (pred_idx == i)
        cnt = int(m.sum())
        pct = float(cnt / N * 100.0)
        g_pct = float(both_gray[m].mean() * 100.0)
        r_pct = float(any_rot[m].mean() * 100.0)
        c_pct = float(cld_post[m].mean() * 100.0)
        st_d = float(pf5[m, p5_map["stripe_z_delta"]].mean())
        st_z = float(pf5[m, p5_map["stripe_z_post"]].mean())
        ch_g = float(pf5[m, p5_map["charcoal_gain"]].mean())
        lr_m = float(apf[m, ap_map["lap_log_ratio"]].mean())
        print(
            f"  {cls:11s} n={cnt:4d} ({pct:5.1f}%) | gray={g_pct:4.1f}% rot={r_pct:4.1f}% cld_q={c_pct:4.2f}% | "
            f"stDel={st_d:+.2f} stPst={st_z:.2f} charc={ch_g:+.3f} lapR={lr_m:+.3f}",
            flush=True,
        )

    print(
        f"\nBeirut retention: v3_C22(25)={int((pred_idx[beirut_v3]==expl_idx).sum())}/25 | "
        f"S49(27)={int((pred_idx[lab_micro==49]==expl_idx).sum())}/27 | "
        f"Non-S49 rescued to Explosion={int(((pred_idx==expl_idx) & (lab_micro!=49)).sum())}",
        flush=True,
    )

    run_anchor_diagnostic(sample_ids, pred_labels)

    s6_labels = None
    for tag, p_csv in [
        ("v6 (revised unsub)", ROOT / "ghani" / "submissions" / "task2" / "submission_v6_ghani.csv"),
        ("v4 (LB 0.37878)", ROOT / "ghani" / "submissions" / "task2" / "submission_v4_ghani.csv"),
    ]:
        if p_csv.exists():
            other = pd.read_csv(p_csv).set_index("Id").loc[sample_ids]["Phenomenon"].values
            if "v6" in tag:
                s6_labels = other
            ct = pd.crosstab(pd.Series(pred_labels, name="v7"), pd.Series(other, name=tag))
            print(f"\n=== CROSS-TAB: v7 vs {tag} ===", flush=True)
            print(ct.to_string(), flush=True)

    render_v7_audit_montages(sample_ids, pred_idx, full_s2, aln, lab_micro, beirut_v3, s6_labels)
    print(f"\nDONE v7 Part 3 in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
