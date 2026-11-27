"""Task-2 v5 Part 3: Two-Pass Clean-Core 5-Fold 4-Family Ensemble & Zero-Retraining Verification.

100% Pure-Vision & Compliant (FAQ Task-2 no. 2):
  - No text/point/box prompts, no VLM/LLM/CLIP/SAM, no external labeled data, zero hardcoded test IDs.
  - Deterministic SEED = 42, 100% zero-retraining inference reload check (3180/3180 exact match).
  - Uses 62 visually audited De-Tinted micro-clusters (+3 physics-verified KMeans(2) sub-splits on M01, M02, M14)
    and two-pass confident clean-core self-training to eliminate residual intra-cluster & cloud noise.
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "ghani" / "task2" / "src" / "v4"))
sys.path.insert(0, str(ROOT / "ghani" / "task2" / "src" / "v5"))
from align_extract_v4 import l2norm, warp_post_to_pre  # noqa: E402
from extract_v5 import detint_clean_rgb  # noqa: E402

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
MODEL_DIR = ROOT / "ghani" / "task2" / "models" / "v5"
SUB_PATH = ROOT / "ghani" / "submissions" / "task2" / "submission_v5_ghani.csv"

# Base micro-cluster mapping from RANDOM-sample De-Tinted visual audit (M00..M61)
# Sub-splits on M01 (Volcano vs Striped Tsunami), M02 (Volcano ash vs Hurricane),
# and M14 (Volcano hills vs Hurricane coast) are applied deterministically via KMeans(2).
MICRO_TO_CLASS_V5: dict[int, str] = {
    0: "Hurricane",   # n=161: US rural creeks, golf courses, fields
    1: "Tsunami",     # n=129: sub-split -> M01_s1 (88, stripe_z=4.86) Tsunami | M01_s0 (41, lapR=-0.88) Volcano
    2: "Hurricane",   # n=113: sub-split -> M02_s1 (69) Hurricane | M02_s0 (44, volc_ash, lapR=-0.61) Volcano
    3: "Hurricane",   # n=110: flooded US ponds & farms (watI=0.031, lapR=-0.77)
    4: "Earthquake",  # n=108: dense urban residential & terracotta grids
    5: "Earthquake",  # n=103: dense urban blocks
    6: "Fire",        # n=87:  100% charred black/gray wildfire scars & ash (charcoal=+0.21)
    7: "Tsunami",     # n=84:  100% striped Sunda Tsunami Indonesian villages (stripe_z=4.0)
    8: "Hurricane",   # n=83:  US wooded suburbs & Florida canals
    9: "Hurricane",   # n=82:  US flooded suburbs, rivers & industrial
    10: "Tsunami",    # n=80:  100% striped Sunda Tsunami tropical hills/villages (stripe_z=4.85)
    11: "Fire",       # n=77:  scorched brown/gray wildfire forest canopy (exg_drop=0.18, ngray=+0.21)
    12: "Hurricane",  # n=77:  US wooded wetlands & forests
    13: "Hurricane",  # n=75:  100% flooded US pastures & rivers (watI=0.031, lapR=-0.67)
    14: "Hurricane",  # n=72:  sub-split -> M14_s1 (49) US coast/suburbs Hurricane | M14_s0 (23) volcanic hills Volcano
    15: "Hurricane",  # n=71:  100% flooded US fields & ponds (watI=0.035, lapR=-0.90)
    16: "Hurricane",  # n=69:  100% flooded US farms & creeks
    17: "Hurricane",  # n=64:  US coastal farms & rivers
    18: "Volcano",    # n=63:  cloudy post over volcanic slopes (downweighted in Pass 1: cld=11.8%)
    19: "Fire",       # n=63:  100% charred hollow-roof burnt suburbs (charcoal=+0.13, hol_ash=+0.019)
    20: "Earthquake", # n=56:  dense urban residential grids
    21: "Hurricane",  # n=54:  US flooded suburbs & canal houses (corrected from v4 Earthquake!)
    22: "Hurricane",  # n=54:  US creeks & golf/canal suburbs
    23: "Volcano",    # n=54:  100% volcanic crater/ridges/pyroclastic gullies (lapR=-0.49)
    24: "Explosion",  # n=54:  Beirut Port Explosion core + obliterated warehouse 16f2ba6 + 17/25 v3_C22!
    25: "Fire",       # n=52:  100% charred forest & white-ash wildfire scars (corrected from v4 Volcano!)
    26: "Earthquake", # n=52:  urban avenues, stadiums, commercial grids
    27: "Volcano",    # n=51:  100% black basalt lava flows & volcanic ash valleys (Lower Puna/Guatemala)
    28: "Tsunami",    # n=50:  100% striped Sunda Tsunami coastal inundation (stripe_z=6.10 #1!)
    29: "Volcano",    # n=46:  dark volcanic terrain, mudflow & ash (lapR=-0.64)
    30: "Fire",       # n=44:  100% white-ash burned California suburbs (hol_ash=+0.073; corrected from v4 EQ!)
    31: "Hurricane",  # n=39:  100% dark-water flooded US pastures & ponds (watI=0.036)
    32: "Earthquake", # n=39:  dense residential urban blocks
    33: "Earthquake", # n=38:  dense urban industrial & apartment blocks
    34: "Hurricane",  # n=37:  100% flooded US pastures (watI=0.036)
    35: "Fire",       # n=37:  100% gray/black scorched burnt fields & ash (hol_ash=+0.038; corrected from v4 Hurr!)
    36: "Earthquake", # n=37:  dense terracotta-roofed towns
    37: "Tsunami",    # n=36:  100% open ocean & striped coastal sea (stripe_z=3.70; corrected from v4 Fire!)
    38: "Hurricane",  # n=35:  US canal & suburban houses (corrected from v4 Earthquake!)
    39: "Tsunami",    # n=35:  Indonesian/coastal towns & inundation
    40: "Volcano",    # n=36:  arid mountain valleys, volcanic slopes & terraces
    41: "Hurricane",  # n=31:  US rural/suburban creeks & pastures
    42: "Earthquake", # n=31:  dense city blocks
    43: "Earthquake", # n=31:  dense residential urban grids
    44: "Hurricane",  # n=31:  US apartments w/ pools & Florida lake houses (corrected from v4 Earthquake!)
    45: "Tsunami",    # n=29:  striped coastal towns w/ clouds (stripe_z=4.07, downweighted in Pass 1)
    46: "Fire",       # n=28:  charred dark suburban neighborhoods (charcoal=+0.09; corrected from v4 EQ!)
    47: "Hurricane",  # n=29:  US suburbs & lakes (corrected from v4 Earthquake!)
    48: "Tsunami",    # n=25:  swept beaches & misty coastal towns
    49: "Hurricane",  # n=24:  100% Florida golf courses & lakes
    50: "Earthquake", # n=23:  urban/industrial blocks & bridges
    51: "Hurricane",  # n=23:  flooded rural pastures (watI=0.038)
    52: "Tsunami",    # n=22:  100% striped Sunda post on tropical fields (stripe_z=3.72; corrected from v4 Fire!)
    53: "Earthquake", # n=22:  dense concrete cities
    54: "Explosion",  # n=21:  Beirut waterfront/Corniche, Port highway (*d3a1e35, b61f21d) & high-rises!
    55: "Explosion",  # n=17:  Beirut industrial/high-rises under blast smoke/clouds (cloudy tiles downweighted!)
    56: "Volcano",    # n=16:  black lava flow & volcanic channels
    57: "Hurricane",  # n=15:  flooded wooded creeks (watI=0.032, lapR=-0.92)
    58: "Earthquake", # n=16:  dense Mediterranean red-roofed towns
    59: "Fire",       # n=14:  100% white-ash burned houses (hol_ash=+0.098 #1; corrected from v4 Volcano!)
    60: "Hurricane",  # n=13:  100% Florida canal/suburban houses
    61: "Hurricane",  # n=12:  US coastal/golf suburbs (corrected from v4 Earthquake!)
}
assert len(MICRO_TO_CLASS_V5) == 62


def load_all_v5_caches():
    aln = np.load(C4_DIR / "alignment_v4.npz")
    p5 = np.load(C5_DIR / "detint_phys_v5.npz", allow_pickle=True)
    vb5 = np.load(C5_DIR / "detint_vitb_224.npz")
    cm5 = np.load(C5_DIR / "changemap_vits_224.npz")
    vs4 = np.load(C4_DIR / "aligned_vits_336.npz")
    vb4 = np.load(C4_DIR / "aligned_vitb_224.npz")
    ap4 = np.load(C4_DIR / "aligned_phys_v4.npz", allow_pickle=True)
    p3 = np.load(C3_DIR / "phys_v3.npz", allow_pickle=True)
    cx3 = np.load(C3_DIR / "cnxb_gray.npz")
    return aln, p5, vb5, cm5, vs4, vb4, ap4, p3, cx3


def build_v5_feature_matrices(aln, p5, vb5, cm5, vs4, vb4, ap4, p3, cx3, scaler_state=None):
    """Build deterministic X_v5_knn, X_v5_full, X_v5_lgb with leak-free fitted scaler/PCA state."""
    p5_names = list(p5["feature_names"])
    ap_names = list(ap4["feature_names"])
    p3_names = list(p3["feature_names"])
    p3_map = {k: i for i, k in enumerate(p3_names)}

    pf5 = np.nan_to_num(p5["features"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    apf = np.nan_to_num(ap4["features"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pf3 = np.nan_to_num(p3["features"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # Filter out raw cloud/gray/black artifact cols from pf3 so models never learn artifacts
    drop_p3 = {"valid_frac_pre", "valid_frac_post", "cloud_frac_pre", "cloud_frac_post", "blk_frac_pre", "blk_frac_post", "is_gray_pre", "is_gray_post"}
    keep_p3 = [i for i, k in enumerate(p3_names) if k not in drop_p3]
    pf3_clean = pf3[:, keep_p3]

    raw_tab = np.hstack([pf5, apf, pf3_clean, vb5["loc_stats"], vs4["loc_stats"], vb4["loc_stats"]]).astype(np.float32)

    # Compact L2-normalized embedding views
    e_vb5_pre = l2norm(vb5["pmean_pre"].astype(np.float32))
    e_vb5_post = l2norm(vb5["pmean_post"].astype(np.float32))
    e_vb5_del = l2norm(vb5["aligned_delta_emb"].astype(np.float32))
    e_vb5_tpost = l2norm(vb5["topchg_post"].astype(np.float32))
    e_vb5_tdel = l2norm(vb5["topchg_delta"].astype(np.float32))
    e_cm5 = l2norm(cm5["emb"].astype(np.float32))
    e_vs4_pre = l2norm(vs4["pmean_pre"].astype(np.float32))
    e_vs4_post = l2norm(vs4["pmean_post"].astype(np.float32))
    e_vs4_del = l2norm(vs4["aligned_delta_emb"].astype(np.float32))
    e_vs4_tdel = l2norm(vs4["topchg_delta"].astype(np.float32))
    e_vb4_del = l2norm(vb4["aligned_delta_emb"].astype(np.float32))
    e_vb4_tdel = l2norm(vb4["topchg_delta"].astype(np.float32))
    e_cx_pre = l2norm(cx3["pre"].astype(np.float32))
    e_cx_post = l2norm(cx3["post"].astype(np.float32))

    if scaler_state is None:
        tab_sc = StandardScaler().fit(raw_tab)
        p5_sc = StandardScaler().fit(pf5)
        ap_sc = StandardScaler().fit(apf)
        tab_z = np.clip(tab_sc.transform(raw_tab), -5.0, 5.0).astype(np.float32)
        p5_n = l2norm(np.clip(p5_sc.transform(pf5), -4.0, 4.0).astype(np.float32))
        ap_n = l2norm(np.clip(ap_sc.transform(apf), -4.0, 4.0).astype(np.float32))

        pca_pre = PCA(n_components=48, random_state=SEED).fit(np.hstack([e_vb5_pre, e_vs4_pre]))
        pca_post = PCA(n_components=48, random_state=SEED).fit(np.hstack([e_vb5_post, e_vs4_post]))
        pca_del = PCA(n_components=48, random_state=SEED).fit(np.hstack([e_vb5_del, e_vb5_tdel, e_vs4_del, e_cm5]))
        scaler_state = {
            "tab_sc": tab_sc, "p5_sc": p5_sc, "ap_sc": ap_sc,
            "pca_pre": pca_pre, "pca_post": pca_post, "pca_del": pca_del,
        }
    else:
        tab_z = np.clip(scaler_state["tab_sc"].transform(raw_tab), -5.0, 5.0).astype(np.float32)
        p5_n = l2norm(np.clip(scaler_state["p5_sc"].transform(pf5), -4.0, 4.0).astype(np.float32))
        ap_n = l2norm(np.clip(scaler_state["ap_sc"].transform(apf), -4.0, 4.0).astype(np.float32))

    # X_v5_graph: identical geometry to cluster_audit_v5 for Cosine kNN & sub-splits
    X_v5_graph = l2norm(np.hstack([
        0.85 * e_vb5_pre, 1.15 * e_vb5_post, 1.25 * e_vb5_del,
        1.00 * e_vb5_tpost, 1.10 * e_vb5_tdel, 0.95 * e_cm5,
        0.75 * e_vs4_pre, 0.85 * e_vs4_post, 0.95 * e_vs4_del, 0.85 * e_vs4_tdel,
        0.65 * e_vb4_del, 0.65 * e_vb4_tdel, 0.75 * p5_n, 0.45 * ap_n,
    ]).astype(np.float32))

    # X_v5_full: rich multi-modal matrix for ResidualMLP & LogisticRegression (~9300-d)
    X_v5_full = np.hstack([
        e_vb5_pre, e_vb5_post, e_vb5_del, e_vb5_tpost, e_vb5_tdel,
        e_cm5, e_vs4_pre, e_vs4_post, e_vs4_del, e_vs4_tdel,
        0.7 * e_vb4_del, 0.7 * e_vb4_tdel,
        0.5 * e_cx_pre, 0.5 * e_cx_post,
        0.18 * tab_z,
    ]).astype(np.float32)

    # X_v5_lgb: 144 PCA embedding dims + full aligned/de-tinted/FFT physics (~350-d)
    X_v5_lgb = np.hstack([
        scaler_state["pca_pre"].transform(np.hstack([e_vb5_pre, e_vs4_pre])),
        scaler_state["pca_post"].transform(np.hstack([e_vb5_post, e_vs4_post])),
        scaler_state["pca_del"].transform(np.hstack([e_vb5_del, e_vb5_tdel, e_vs4_del, e_cm5])),
        tab_z,
    ]).astype(np.float32)

    cld_pre = pf3[:, p3_map["cloud_frac_pre"]]
    cld_post = pf3[:, p3_map["cloud_frac_post"]]
    return X_v5_graph, X_v5_full, X_v5_lgb, cld_pre, cld_post, pf5, p5_names, apf, ap_names, pf3, p3_names, scaler_state


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


def train_one_mlp_fold(fold: int, X_tr: np.ndarray, y_tr: np.ndarray, w_tr: np.ndarray, cls_w_t: torch.Tensor, save_path: Path | None) -> ResidualMLP:
    torch.manual_seed(SEED + fold * 13)
    in_dim = X_tr.shape[1]
    model = ResidualMLP(in_dim=in_dim, hidden_dim=384, n_classes=len(CLASSES))
    ds = TensorDataset(
        torch.from_numpy(X_tr),
        torch.from_numpy(y_tr),
        torch.from_numpy(w_tr),
    )
    gen = torch.Generator().manual_seed(SEED + fold * 13)
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
            loss = (lv * wb).mean()
            loss.backward()
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


def run_5fold_4family(
    X_graph: np.ndarray,
    X_full: np.ndarray,
    X_lgb: np.ndarray,
    y_target: np.ndarray,
    sample_w: np.ndarray,
    cls_w: np.ndarray,
    save_artifacts: bool,
):
    N = len(y_target)
    cls_w_t = torch.tensor(cls_w, dtype=torch.float32)
    cls_w_dict = {i: float(cls_w[i]) for i in range(len(CLASSES))}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    oof_mlp = np.zeros((N, len(CLASSES)), dtype=np.float32)
    oof_lr = np.zeros((N, len(CLASSES)), dtype=np.float32)
    oof_lgb = np.zeros((N, len(CLASSES)), dtype=np.float32)
    oof_knn = np.zeros((N, len(CLASSES)), dtype=np.float32)

    full_mlp = np.zeros((N, len(CLASSES)), dtype=np.float32)
    full_lr = np.zeros((N, len(CLASSES)), dtype=np.float32)
    full_lgb = np.zeros((N, len(CLASSES)), dtype=np.float32)
    full_knn = np.zeros((N, len(CLASSES)), dtype=np.float32)

    lr_models, lgb_models, knn_models = [], [], []
    for fold, (tr, va) in enumerate(skf.split(X_graph, y_target)):
        t_f = time.time()
        mlp_p = (MODEL_DIR / f"mlp_v5_f{fold}.pt") if save_artifacts else None
        mlp_m = train_one_mlp_fold(fold, X_full[tr], y_target[tr], sample_w[tr], cls_w_t, mlp_p)
        oof_mlp[va] = predict_mlp_proba(mlp_m, X_full[va])
        full_mlp += predict_mlp_proba(mlp_m, X_full) / 5.0

        lr_m = LogisticRegression(C=10.0, max_iter=1000, class_weight=cls_w_dict, random_state=SEED + fold)
        lr_m.fit(X_full[tr], y_target[tr], sample_weight=sample_w[tr])
        oof_lr[va] = lr_m.predict_proba(X_full[va]).astype(np.float32)
        full_lr += lr_m.predict_proba(X_full).astype(np.float32) / 5.0
        if save_artifacts:
            lr_models.append(lr_m)

        lgb_m = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(CLASSES),
            n_estimators=420,
            learning_rate=0.04,
            num_leaves=23,
            subsample_for_bin=50000,
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
        lgb_m.fit(X_lgb[tr], y_target[tr], sample_weight=sample_w[tr])
        oof_lgb[va] = lgb_m.predict_proba(X_lgb[va]).astype(np.float32)
        full_lgb += lgb_m.predict_proba(X_lgb).astype(np.float32) / 5.0
        if save_artifacts:
            lgb_models.append(lgb_m)

        knn_m = KNeighborsClassifier(n_neighbors=17, weights="distance", metric="cosine", n_jobs=-1)
        knn_m.fit(X_graph[tr], y_target[tr])
        oof_knn[va] = knn_m.predict_proba(X_graph[va]).astype(np.float32)
        full_knn += knn_m.predict_proba(X_graph).astype(np.float32) / 5.0
        if save_artifacts:
            knn_models.append(knn_m)

        print(f"    Fold {fold} done in {time.time()-t_f:.1f}s", flush=True)

    if save_artifacts:
        joblib.dump(lr_models, MODEL_DIR / "lr_v5.joblib")
        joblib.dump(lgb_models, MODEL_DIR / "lgb_v5.joblib")
        joblib.dump(knn_models, MODEL_DIR / "knn_v5.joblib")

    oof_ens = 0.35 * oof_mlp + 0.25 * oof_lr + 0.25 * oof_lgb + 0.15 * oof_knn
    full_ens = 0.35 * full_mlp + 0.25 * full_lr + 0.25 * full_lgb + 0.15 * full_knn
    fam_oof = {
        "ResidualMLP": round(float(f1_score(y_target, oof_mlp.argmax(1), average="macro")), 4),
        "LogisticRegression": round(float(f1_score(y_target, oof_lr.argmax(1), average="macro")), 4),
        "LightGBM": round(float(f1_score(y_target, oof_lgb.argmax(1), average="macro")), 4),
        "CosineKNN": round(float(f1_score(y_target, oof_knn.argmax(1), average="macro")), 4),
        "Ensemble_4Family": round(float(f1_score(y_target, oof_ens.argmax(1), average="macro")), 4),
    }
    return oof_ens, full_ens, fam_oof


def verify_zero_retraining_inference(expected_preds: np.ndarray) -> bool:
    """Reload every saved .pt and .joblib artifact fresh from MODEL_DIR and assert 100% exact match."""
    scaler_state = joblib.load(MODEL_DIR / "feature_scaler_v5.joblib")
    lr_models = joblib.load(MODEL_DIR / "lr_v5.joblib")
    lgb_models = joblib.load(MODEL_DIR / "lgb_v5.joblib")
    knn_models = joblib.load(MODEL_DIR / "knn_v5.joblib")

    aln, p5, vb5, cm5, vs4, vb4, ap4, p3, cx3 = load_all_v5_caches()
    X_graph, X_full, X_lgb, *_ = build_v5_feature_matrices(
        aln, p5, vb5, cm5, vs4, vb4, ap4, p3, cx3, scaler_state=scaler_state
    )
    N = len(X_graph)
    p_mlp = np.zeros((N, len(CLASSES)), dtype=np.float32)
    p_lr = np.zeros((N, len(CLASSES)), dtype=np.float32)
    p_lgb = np.zeros((N, len(CLASSES)), dtype=np.float32)
    p_knn = np.zeros((N, len(CLASSES)), dtype=np.float32)

    for fold in range(5):
        ckpt = torch.load(MODEL_DIR / f"mlp_v5_f{fold}.pt", map_location="cpu", weights_only=True)
        m = ResidualMLP(in_dim=ckpt["in_dim"], hidden_dim=ckpt["hidden_dim"], n_classes=ckpt["n_classes"])
        m.load_state_dict(ckpt["state_dict"])
        p_mlp += predict_mlp_proba(m, X_full) / 5.0
        p_lr += lr_models[fold].predict_proba(X_full).astype(np.float32) / 5.0
        p_lgb += lgb_models[fold].predict_proba(X_lgb).astype(np.float32) / 5.0
        p_knn += knn_models[fold].predict_proba(X_graph).astype(np.float32) / 5.0

    p_ens = 0.35 * p_mlp + 0.25 * p_lr + 0.25 * p_lgb + 0.15 * p_knn
    reloaded = np.array([CLASSES[i] for i in np.argmax(p_ens, axis=1)])
    matches = int((reloaded == expected_preds).sum())
    print(f"[Zero-Retraining Reload Check] Exact matches: {matches}/{N} ({matches/N*100:.2f}%)", flush=True)
    assert matches == N, f"Zero-retraining reload mismatch: {matches}/{N}"
    return True


def render_v5_final_montages(sample_ids, pred_idx, p_ens, aln, lab_micro, beirut_v3):
    """Render honest RANDOM-sample De-Tinted Aligned final montages + Explosion audit."""
    rng = np.random.RandomState(SEED + 7)
    T = 108
    bang, bdy, bdx = aln["best_ang"], aln["best_dy"], aln["best_dx"]

    # 1. Final 6 classes: 8 RANDOM samples per class (2 rows x 4 pairs per class? No: 1 row x 6 RANDOM pairs!)
    can = Image.new("RGB", (115 + 6 * (2 * T + 6), len(CLASSES) * (T + 16)), "white")
    dr = ImageDraw.Draw(can)
    for r, cls in enumerate(CLASSES):
        idx = np.where(pred_idx == r)[0]
        picks = rng.choice(idx, size=min(6, len(idx)), replace=False)
        dr.text((4, r * (T + 16) + 26), f"{cls}\nn={len(idx)}\n({len(idx)/len(sample_ids)*100:.1f}%)\n[RANDOM]", fill="black")
        for j, ii in enumerate(picks):
            sid = sample_ids[ii]
            with Image.open(FEN_DIR / sid / "pre.jpg") as i1, Image.open(FEN_DIR / sid / "post.jpg") as i2:
                r1 = detint_clean_rgb(i1.convert("RGB").resize((T, T), Image.Resampling.BILINEAR), False)
                r2 = detint_clean_rgb(warp_post_to_pre(i2.convert("RGB"), float(bang[ii]), int(bdy[ii]), int(bdx[ii]), T), False)
            x0 = 115 + j * (2 * T + 6)
            y0 = r * (T + 16) + 2
            can.paste(r1, (x0, y0))
            can.paste(r2, (x0 + T + 2, y0))
            dr.text((x0 + 2, y0 + T + 1), f"M{lab_micro[ii]:02d}:{sid[:6]}:{p_ens[ii, r]:.2f}", fill="black")
    out1 = C5_DIR / "montage_v5_final_random_6classes.jpg"
    can.save(out1, quality=88)
    print(f"Saved RANDOM 6-class montage: {out1}", flush=True)

    # 2. Explosion audit: Top-6 Conf, Mid-6 Random, Low-6 Border
    expl_idx = CLASSES.index("Explosion")
    ex_all = np.where(pred_idx == expl_idx)[0]
    ord_ex = ex_all[np.argsort(-p_ens[ex_all, expl_idx])]
    mid_start = max(0, len(ord_ex) // 2 - 3)
    rows_ex = [
        ("Expl Top6", ord_ex[:6]),
        ("Expl Mid6", ord_ex[mid_start:mid_start + 6]),
        ("Expl Low6", ord_ex[-6:]),
    ]
    can2 = Image.new("RGB", (115 + 6 * (2 * T + 6), 3 * (T + 16)), "white")
    dr2 = ImageDraw.Draw(can2)
    for r, (lbl, picks) in enumerate(rows_ex):
        dr2.text((4, r * (T + 16) + 26), f"{lbl}\nn={len(ex_all)}", fill="black")
        for j, ii in enumerate(picks):
            sid = sample_ids[ii]
            with Image.open(FEN_DIR / sid / "pre.jpg") as i1, Image.open(FEN_DIR / sid / "post.jpg") as i2:
                r1 = detint_clean_rgb(i1.convert("RGB").resize((T, T), Image.Resampling.BILINEAR), False)
                r2 = detint_clean_rgb(warp_post_to_pre(i2.convert("RGB"), float(bang[ii]), int(bdy[ii]), int(bdx[ii]), T), False)
            x0 = 115 + j * (2 * T + 6)
            y0 = r * (T + 16) + 2
            can2.paste(r1, (x0, y0))
            can2.paste(r2, (x0 + T + 2, y0))
            bt = "B" if beirut_v3[ii] else "N"
            dr2.text((x0 + 2, y0 + T + 1), f"{bt}:M{lab_micro[ii]:02d}:{sid[:6]}:{p_ens[ii, expl_idx]:.2f}", fill="black")
    out2 = C5_DIR / "v5_explosion_audit.jpg"
    can2.save(out2, quality=88)
    print(f"Saved Explosion audit montage: {out2}", flush=True)


def main():
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUB_PATH.parent.mkdir(parents=True, exist_ok=True)

    sample_ids = json.load(open(C5_DIR / "sample_ids.json", "r", encoding="utf-8"))
    ref_ids = pd.read_csv(SAMPLE_SUB)["Id"].tolist()
    assert len(sample_ids) == 3180 and sample_ids == ref_ids

    aln, p5, vb5, cm5, vs4, vb4, ap4, p3, cx3 = load_all_v5_caches()
    X_graph, X_full, X_lgb, cld_pre, cld_post, pf5, p5_names, apf, ap_names, pf3, p3_names, scaler_state = (
        build_v5_feature_matrices(aln, p5, vb5, cm5, vs4, vb4, ap4, p3, cx3, scaler_state=None)
    )
    joblib.dump(scaler_state, MODEL_DIR / "feature_scaler_v5.joblib")

    cl5 = np.load(C5_DIR / "clusters_v5_micro.npz")
    lab_micro = cl5["lab_micro"]
    cents = cl5["cents"]
    cl3 = np.load(C3_DIR / "clusters_v3_clean.npz")["lab_clean"]
    beirut_v3 = (cl3 == 22)
    assert int(beirut_v3.sum()) == 25

    p5_map = {k: i for i, k in enumerate(p5_names)}
    ap_map = {k: i for i, k in enumerate(ap_names)}
    p3_map = {k: i for i, k in enumerate(p3_names)}

    # 1. Construct Pass-1 Pseudo-Targets from 62 Micro-Clusters + 3 Verified KMeans(2) Sub-Splits
    y_p1 = np.array([CLASSES.index(MICRO_TO_CLASS_V5[int(c)]) for c in lab_micro], dtype=np.int64)

    # Sub-split M01: higher stripe_z -> Tsunami (88), lower stripe_z -> Volcano (41)
    m01_idx = np.where(lab_micro == 1)[0]
    m01_sub = KMeans(n_clusters=2, n_init=30, random_state=SEED).fit_predict(X_graph[m01_idx])
    m01_st = [float(pf5[m01_idx[m01_sub == k], p5_map["stripe_z_post"]].mean()) for k in range(2)]
    m01_volc_idx = m01_idx[m01_sub == int(np.argmin(m01_st))]
    assert 30 <= len(m01_volc_idx) <= 55, f"Unexpected M01 volcano split: {len(m01_volc_idx)}"
    y_p1[m01_volc_idx] = CLASSES.index("Volcano")

    # Sub-split M02: lower lap_log_ratio (heavy volcanic ash smoothing) -> Volcano (44), other -> Hurricane (69)
    m02_idx = np.where(lab_micro == 2)[0]
    m02_sub = KMeans(n_clusters=2, n_init=30, random_state=SEED).fit_predict(X_graph[m02_idx])
    m02_lr = [float(apf[m02_idx[m02_sub == k], ap_map["lap_log_ratio"]].mean()) for k in range(2)]
    m02_volc_idx = m02_idx[m02_sub == int(np.argmin(m02_lr))]
    assert 30 <= len(m02_volc_idx) <= 60, f"Unexpected M02 volcano split: {len(m02_volc_idx)}"
    y_p1[m02_volc_idx] = CLASSES.index("Volcano")

    # Sub-split M14: higher lap_log_ratio (arid volcanic hills) -> Volcano (23), other -> Hurricane (49)
    m14_idx = np.where(lab_micro == 14)[0]
    m14_sub = KMeans(n_clusters=2, n_init=30, random_state=SEED).fit_predict(X_graph[m14_idx])
    m14_lr = [float(apf[m14_idx[m14_sub == k], ap_map["lap_log_ratio"]].mean()) for k in range(2)]
    m14_volc_idx = m14_idx[m14_sub == int(np.argmax(m14_lr))]
    assert 15 <= len(m14_volc_idx) <= 35, f"Unexpected M14 volcano split: {len(m14_volc_idx)}"
    y_p1[m14_volc_idx] = CLASSES.index("Volcano")

    # Ensure all 25 v3_C22 Beirut core tiles are in Explosion
    y_p1[beirut_v3] = CLASSES.index("Explosion")
    p1_counts = {cls: int((y_p1 == i).sum()) for i, cls in enumerate(CLASSES)}
    print("Pass-1 pseudo-target counts:", p1_counts, flush=True)

    # 2. Quality & Cloud-Suppression Sample Weighting for Pass 1
    N = len(sample_ids)
    w_p1 = np.ones(N, dtype=np.float32)
    for c in range(len(cents)):
        idx = np.where(lab_micro == c)[0]
        sims = X_graph[idx] @ cents[c]
        order = np.argsort(np.argsort(sims))
        r_c = order / max(1, len(idx) - 1)
        w_p1[idx] = (0.35 + 0.65 * r_c).astype(np.float32)

    # Strong cloud clarity & alignment reliability weighting so cloudy clusters (M18/M45/M55) never leak!
    cld_max = np.maximum(cld_pre, cld_post)
    w_cld = np.clip(1.0 - 4.0 * cld_max, 0.12, 1.0).astype(np.float32)
    w_aln = np.clip((aln["z_peak"] - 0.5) / 2.0, 0.40, 1.0).astype(np.float32)
    w_p1 = (w_p1 * w_cld * w_aln).astype(np.float32)
    # Downweight the 3 high-cloud micro-clusters in Pass 1 so clean-core predicts them!
    w_p1[np.isin(lab_micro, [18, 45, 55])] *= 0.25
    # Protect clear Beirut core anchors
    w_p1[beirut_v3] = np.maximum(w_p1[beirut_v3], 0.90)

    counts1 = np.bincount(y_p1, minlength=len(CLASSES))
    cls_w1 = (N / (len(CLASSES) * counts1)) ** 0.50
    cls_w1 = cls_w1 / cls_w1.mean()
    print(f"Shapes: X_graph={X_graph.shape}, X_full={X_full.shape}, X_lgb={X_lgb.shape}", flush=True)
    print("\n=== PASS 1: NOISE-ROBUST OOF CORE DISCOVERY ===", flush=True)
    oof_p1, _, fam_p1 = run_5fold_4family(X_graph, X_full, X_lgb, y_p1, w_p1, cls_w1, save_artifacts=False)
    print("  Pass-1 OOF consistency (vs Pass-1 pseudo-targets):", fam_p1, flush=True)

    # 3. Pass-2 Confident Clean-Core Target Refinement (fixes intra-cluster noise & cloudy clusters!)
    y_p2 = y_p1.copy()
    w_p2 = w_p1.copy()
    oof_cls1 = np.argmax(oof_p1, axis=1)
    oof_conf1 = np.max(oof_p1, axis=1)
    p_orig = oof_p1[np.arange(N), y_p1]

    # Relabel non-Beirut-anchor samples where Pass-1 OOF strongly disagrees (>=0.62) OR high-cloud clusters (>=0.48)
    cloudy_mc = np.isin(lab_micro, [18, 45, 55])
    flip_norm = (~beirut_v3) & (~cloudy_mc) & (oof_cls1 != y_p1) & (oof_conf1 >= 0.62) & (p_orig <= 0.28)
    flip_cld = (~beirut_v3) & cloudy_mc & (oof_cls1 != y_p1) & (oof_conf1 >= 0.45)
    flip_mask = flip_norm | flip_cld
    y_p2[flip_mask] = oof_cls1[flip_mask]

    # Weight Pass-2 by OOF agreement confidence so pure core dominates training
    agree_core = (oof_cls1 == y_p2)
    w_p2 = np.where(agree_core, np.maximum(w_p2, 0.70 * oof_conf1 + 0.30), 0.30 * w_p2).astype(np.float32)
    w_p2[flip_mask] = (0.65 * oof_conf1[flip_mask] * w_cld[flip_mask]).astype(np.float32)
    w_p2[beirut_v3] = np.maximum(w_p2[beirut_v3], 0.90)

    p2_counts = {cls: int((y_p2 == i).sum()) for i, cls in enumerate(CLASSES)}
    print(f"\n=== PASS 2: CLEAN-CORE RETRAINING (refined {int(flip_mask.sum())} noisy/cloudy samples) ===", flush=True)
    print("  Pass-2 target counts:", p2_counts, flush=True)

    counts2 = np.bincount(y_p2, minlength=len(CLASSES))
    cls_w2 = (N / (len(CLASSES) * counts2)) ** 0.50
    cls_w2 = cls_w2 / cls_w2.mean()
    print("  Pass-2 class weights:", {CLASSES[i]: round(float(cls_w2[i]), 3) for i in range(len(CLASSES))}, flush=True)

    oof_p2, full_p2, fam_p2 = run_5fold_4family(X_graph, X_full, X_lgb, y_p2, w_p2, cls_w2, save_artifacts=True)

    oof_pred = np.argmax(oof_p2, axis=1)
    pred_idx = np.argmax(full_p2, axis=1)
    pred_labels = np.array([CLASSES[i] for i in pred_idx])
    expl_idx = CLASSES.index("Explosion")

    print("\n=== PASS-2 OOF CONSISTENCY DIAGNOSTICS (vs Clean-Core Targets, NOT Official Ground Truth!) ===", flush=True)
    print("  Per-family OOF macro-F1 agreement:", fam_p2, flush=True)
    print(f"  OOF vs 5-Fold Ensemble argmax agreement: {(oof_pred == pred_idx).mean()*100:.2f}%", flush=True)
    print(f"  OOF Explosion={int((oof_pred==expl_idx).sum())}, Ens Explosion={int((pred_idx==expl_idx).sum())}", flush=True)
    print(classification_report(y_p2, oof_pred, target_names=CLASSES, digits=4), flush=True)

    # Save submission & run strict format checks
    sub_df = pd.DataFrame({"Id": sample_ids, "Phenomenon": pred_labels})
    sub_df.to_csv(SUB_PATH, index=False)
    print(f"Saved submission: {SUB_PATH}", flush=True)

    chk = pd.read_csv(SUB_PATH)
    assert list(chk.columns) == ["Id", "Phenomenon"]
    assert len(chk) == 3180 and chk["Id"].tolist() == ref_ids
    assert chk["Phenomenon"].isna().sum() == 0 and set(chk["Phenomenon"].unique()) == set(CLASSES)

    # Save metadata & spec
    with open(MODEL_DIR / "cluster_mapping_v5.json", "w", encoding="utf-8") as f:
        json.dump({
            "note": "62 visually audited De-Tinted micro-clusters + 3 verified KMeans(2) sub-splits + Pass-2 clean-core self-training. NOT ground truth; zero hardcoded test IDs.",
            "micro_to_class_base": {str(k): v for k, v in MICRO_TO_CLASS_V5.items()},
            "sub_splits": {
                "M01_to_Volcano": int(len(m01_volc_idx)),
                "M02_to_Volcano": int(len(m02_volc_idx)),
                "M14_to_Volcano": int(len(m14_volc_idx)),
            },
            "pass1_counts": p1_counts,
            "pass2_flipped_noisy_or_cloudy": int(flip_mask.sum()),
            "pass2_counts": p2_counts,
        }, f, indent=2)

    spec3 = json.load(open(C3_DIR / "pretrained_spec_v3.json", "r", encoding="utf-8"))
    spec4 = json.load(open(C4_DIR / "pretrained_spec_v4.json", "r", encoding="utf-8"))
    spec5 = json.load(open(C5_DIR / "pretrained_spec_v5.json", "r", encoding="utf-8"))
    with open(MODEL_DIR / "pretrained_spec_v5.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "v3": spec3, "v4": spec4, "v5": spec5}, f, indent=2)

    # Mandatory Zero-Retraining Reload Check
    gc.collect()
    verify_zero_retraining_inference(pred_labels)

    # Final Distribution & Leakage Audit
    audit3 = p3["audit_flags"]
    both_gray = (audit3[:, 2] > 0.5) & (audit3[:, 3] > 0.5)
    any_rot = (audit3[:, 0] > 0.5) | (audit3[:, 1] > 0.5)
    print("\n=== FINAL v5 PREDICTION DISTRIBUTION & LEAKAGE AUDIT ===", flush=True)
    for i, cls in enumerate(CLASSES):
        m = (pred_idx == i)
        cnt = int(m.sum())
        pct = float(cnt / N * 100.0)
        g_pct = float(both_gray[m].mean() * 100.0)
        r_pct = float(any_rot[m].mean() * 100.0)
        c_pct = float(cld_post[m].mean() * 100.0)
        st_z = float(pf5[m, p5_map["stripe_z_post"]].mean())
        ch_g = float(pf5[m, p5_map["charcoal_gain"]].mean())
        lr_m = float(apf[m, ap_map["lap_log_ratio"]].mean())
        print(
            f"  {cls:11s} n={cnt:4d} ({pct:5.1f}%) | gray={g_pct:4.1f}% rot={r_pct:4.1f}% cld_q={c_pct:4.2f}% | "
            f"stripeZ={st_z:.2f} charc={ch_g:+.3f} lapR={lr_m:+.3f}",
            flush=True,
        )

    print(
        f"\nBeirut retention: v3_C22(25)={int((pred_idx[beirut_v3]==expl_idx).sum())}/25 | "
        f"M24(54)={int((pred_idx[lab_micro==24]==expl_idx).sum())}/54 | "
        f"M54(21)={int((pred_idx[lab_micro==54]==expl_idx).sum())}/21 | "
        f"M55(17)={int((pred_idx[lab_micro==55]==expl_idx).sum())}/17",
        flush=True,
    )

    for tag, p_csv in [
        ("v4 (LB 0.37878)", ROOT / "ghani" / "submissions" / "task2" / "submission_v4_ghani.csv"),
        ("v3 (LB 0.37895)", ROOT / "ghani" / "submissions" / "task2" / "submission_v3_ghani.csv"),
    ]:
        if p_csv.exists():
            other = pd.read_csv(p_csv).set_index("Id").loc[sample_ids]["Phenomenon"].values
            ct = pd.crosstab(pd.Series(pred_labels, name="v5"), pd.Series(other, name=tag))
            print(f"\n=== CROSS-TAB: v5 vs {tag} ===", flush=True)
            print(ct.to_string(), flush=True)

    render_v5_final_montages(sample_ids, pred_idx, full_p2, aln, lab_micro, beirut_v3)
    print(f"\nDONE v5 Part 3 in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
