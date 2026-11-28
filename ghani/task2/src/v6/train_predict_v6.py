"""Task-2 v6 Part 2: Mean-Centered Multi-Backbone 5-Fold 4-Family Ensemble & Zero-Retraining Check.

100% Pure-Vision & Compliant (FAQ Task-2 no. 2):
  - No text/point/box prompts, no VLM/LLM/CLIP/SAM, no external labeled data, zero hardcoded test IDs.
  - Deterministic SEED = 42, 100% zero-retraining inference reload check (3180/3180 exact match).
  - Built on 0.85+ Mean-Centered Multi-Backbone Fusion (v3+v4+v5 caches), 47 crisp Louvain clusters
    (+3 verified KMeans(2) sub-splits on K08, K09, K26), and Anti-Black-Hole 3-of-4-family consensus
    Pass-2 clean-core refinement.
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
sys.path.insert(0, str(ROOT / "ghani" / "task2" / "src" / "v6"))
from align_extract_v4 import l2norm, warp_post_to_pre  # noqa: E402
from extract_v5 import detint_clean_rgb  # noqa: E402
from cluster_audit_v6 import build_v6_master_matrix  # noqa: E402

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
C6_DIR = ROOT / "ghani" / "task2" / "cache" / "v6"
MODEL_DIR = ROOT / "ghani" / "task2" / "models" / "v6"
SUB_PATH = ROOT / "ghani" / "submissions" / "task2" / "submission_v6_ghani.csv"

# Complete visually audited mapping for all 47 v6 micro-clusters (K00..K46)
# Deterministic KMeans(2) sub-splits applied on K08 (Fire vs Volcano ash),
# K09 (Volcano lava/ash vs Charcoal Fire), and K26 (Palu Tsunami vs Urban Earthquake).
MICRO_TO_CLASS_V6: dict[int, str] = {
    0: "Volcano",     # n=157: volcanic ridges, pyroclastic gullies & ash (6/10 Volcano anchors!)
    1: "Hurricane",   # n=143: Florida waterfront canals & golf suburbs
    2: "Hurricane",   # n=143: flooded US creeks & farms
    3: "Hurricane",   # n=139: rural US/arid creeks, pastures & roads
    4: "Hurricane",   # n=129: flooded US fields, ponds & creeks
    5: "Hurricane",   # n=121: US pine forests & wooded wetlands
    6: "Volcano",     # n=118: volcanic lahar valleys, black lava & ash
    7: "Fire",        # n=105: 100% charred hollow-roof burnt suburbs (7/10 Fire anchors!)
    8: "Volcano",     # n=103: sub-split -> K08_s1(61, lapR=-0.31) Volcano ash | K08_s0(42, lapR=+0.13) Fire hillside
    9: "Fire",        # n=101: sub-split -> K09_s1(51) Charcoal Fire | K09_s0(50, lapR=+0.66) Volcano lava/ash
    10: "Hurricane",  # n=96:  flooded dark-water US pastures & ponds
    11: "Tsunami",    # n=93:  100% striped Sunda Tsunami farmlands (stripe_z=5.0)
    12: "Fire",       # n=88:  scorched brown/gray wildfire forest canopy
    13: "Earthquake", # n=87:  dense urban blocks
    14: "Earthquake", # n=86:  100% Turkey-Syria terracotta apartment towers
    15: "Hurricane",  # n=86:  US rural/suburban farms & ponds
    16: "Tsunami",    # n=84:  100% striped Sunda Tsunami forests/coast (stripe_z=4.3)
    17: "Fire",       # n=69:  scorched dead hillside forest canopy
    18: "Hurricane",  # n=68:  US rural roads & wooded suburbs
    19: "Hurricane",  # n=63:  US commercial/suburban complexes & new construction
    20: "Tsunami",    # n=58:  100% open ocean & striped sea (stripe_z=7.1 #1!)
    21: "Tsunami",    # n=58:  100% swept coastlines & Palu Tsunami (h_asym=0.092 #1!)
    22: "Tsunami",    # n=56:  100% striped Sunda Tsunami villages (stripe_z=4.0)
    23: "Tsunami",    # n=55:  striped Sunda Tsunami hazy/coastal
    24: "Hurricane",  # n=54:  US wooded suburbs
    25: "Hurricane",  # n=58:  US residential suburban grids
    26: "Earthquake", # n=53:  sub-split -> K26_s0(23) Palu Tsunami coast | K26_s1(30) Urban Earthquake
    27: "Earthquake", # n=54:  ultra-dense urban metropolis
    28: "Hurricane",  # n=49:  Florida lake & canal suburbs
    29: "Tsunami",    # n=44:  striped Sunda coastal w/ clouds (stripe_z=3.7, cld=11.1%, downweighted in Pass 1)
    30: "Earthquake", # n=42:  Mexico City / dense terracotta neighborhoods
    31: "Hurricane",  # n=41:  US canal & suburban houses
    32: "Tsunami",    # n=41:  100% striped Sunda Tsunami coast & fields (stripe_z=4.6)
    33: "Hurricane",  # n=41:  US flooded/wind-damaged forest & creeks
    34: "Earthquake", # n=40:  dense urban avenues & commercial
    35: "Earthquake", # n=39:  dense concrete urban grids
    36: "Explosion",  # n=39:  Beirut off-nadir tall towers & industrial port
    37: "Earthquake", # n=38:  100% Turkey-Syria terracotta apartment blocks
    38: "Earthquake", # n=33:  dense residential urban grids
    39: "Earthquake", # n=35:  dense concrete cities
    40: "Hurricane",  # n=31:  US wooded suburban streets
    41: "Explosion",  # n=29:  100% Beirut Port Explosion core (25/25 v3_C22 + cfb1cae Port blast site!)
    42: "Hurricane",  # n=29:  US curved suburban streets & retention ponds
    43: "Earthquake", # n=25:  Mediterranean/Middle-Eastern stone villages
    44: "Earthquake", # n=23:  urban industrial & apartment blocks
    45: "Earthquake", # n=22:  Turkey-Syria red-roofed apartment grids (cld=13.5%, cloudy tiles downweighted!)
    46: "Earthquake", # n=14:  dense terracotta urban roofs
}
assert len(MICRO_TO_CLASS_V6) == 47


def build_v6_all_matrices(state: dict | None = None):
    """Build deterministic X_master (11376-d) and X_lgb (~360-d) with leak-free fitted state."""
    c_state = None if state is None else state["center_state"]
    X_master, c_state_out, p5, ap4, p3 = build_v6_master_matrix(center_state=c_state)

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
    raw_tab = np.hstack([pf5, apf, pf3[:, keep_p3], vb5["loc_stats"], vs4["loc_stats"], vb4["loc_stats"]]).astype(np.float32)

    if state is None:
        tab_sc = StandardScaler().fit(raw_tab)
        pca_m = PCA(n_components=160, random_state=SEED).fit(X_master)
        state_out = {"center_state": c_state_out, "tab_sc": tab_sc, "pca_m": pca_m}
    else:
        tab_sc = state["tab_sc"]
        pca_m = state["pca_m"]
        state_out = state

    tab_z = np.clip(tab_sc.transform(raw_tab), -5.0, 5.0).astype(np.float32)
    X_lgb = np.hstack([pca_m.transform(X_master).astype(np.float32), tab_z]).astype(np.float32)
    X_mlp = np.hstack([X_master, 0.08 * tab_z]).astype(np.float32)

    cld_pre = pf3[:, p3_map["cloud_frac_pre"]]
    cld_post = pf3[:, p3_map["cloud_frac_post"]]
    return X_master, X_mlp, X_lgb, cld_pre, cld_post, pf5, p5_names, apf, ap_names, p3, state_out


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


def run_5fold_4family(X_master, X_mlp, X_lgb, y_target, sample_w, cls_w, save_artifacts: bool):
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
    for fold, (tr, va) in enumerate(skf.split(X_master, y_target)):
        t_f = time.time()
        mlp_p = (MODEL_DIR / f"mlp_v6_f{fold}.pt") if save_artifacts else None
        mlp_m = train_one_mlp_fold(fold, X_mlp[tr], y_target[tr], sample_w[tr], cls_w_t, mlp_p)
        oof_mlp[va] = predict_mlp_proba(mlp_m, X_mlp[va])
        full_mlp += predict_mlp_proba(mlp_m, X_mlp) / 5.0

        lr_m = LogisticRegression(C=10.0, max_iter=1000, class_weight=cls_w_dict, random_state=SEED + fold)
        lr_m.fit(X_master[tr], y_target[tr], sample_weight=sample_w[tr])
        oof_lr[va] = lr_m.predict_proba(X_master[va]).astype(np.float32)
        full_lr += lr_m.predict_proba(X_master).astype(np.float32) / 5.0
        if save_artifacts:
            lr_models.append(lr_m)

        lgb_m = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(CLASSES),
            n_estimators=420,
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
        lgb_m.fit(X_lgb[tr], y_target[tr], sample_weight=sample_w[tr])
        oof_lgb[va] = lgb_m.predict_proba(X_lgb[va]).astype(np.float32)
        full_lgb += lgb_m.predict_proba(X_lgb).astype(np.float32) / 5.0
        if save_artifacts:
            lgb_models.append(lgb_m)

        knn_m = KNeighborsClassifier(n_neighbors=17, weights="distance", metric="cosine", n_jobs=-1)
        knn_m.fit(X_master[tr], y_target[tr])
        oof_knn[va] = knn_m.predict_proba(X_master[va]).astype(np.float32)
        full_knn += knn_m.predict_proba(X_master).astype(np.float32) / 5.0
        if save_artifacts:
            knn_models.append(knn_m)

        print(f"    Fold {fold} done in {time.time()-t_f:.1f}s", flush=True)

    if save_artifacts:
        joblib.dump(lr_models, MODEL_DIR / "lr_v6.joblib")
        joblib.dump(lgb_models, MODEL_DIR / "lgb_v6.joblib")
        joblib.dump(knn_models, MODEL_DIR / "knn_v6.joblib")

    oof_ens = 0.35 * oof_mlp + 0.25 * oof_lr + 0.20 * oof_lgb + 0.20 * oof_knn
    full_ens = 0.35 * full_mlp + 0.25 * full_lr + 0.20 * full_lgb + 0.20 * full_knn
    fam_oof = {
        "ResidualMLP": round(float(f1_score(y_target, oof_mlp.argmax(1), average="macro")), 4),
        "LogisticRegression": round(float(f1_score(y_target, oof_lr.argmax(1), average="macro")), 4),
        "LightGBM": round(float(f1_score(y_target, oof_lgb.argmax(1), average="macro")), 4),
        "CosineKNN": round(float(f1_score(y_target, oof_knn.argmax(1), average="macro")), 4),
        "Ensemble_4Family": round(float(f1_score(y_target, oof_ens.argmax(1), average="macro")), 4),
    }
    oof_fams = [oof_mlp, oof_lr, oof_lgb, oof_knn]
    return oof_ens, full_ens, fam_oof, oof_fams


def verify_zero_retraining_inference(expected_preds: np.ndarray) -> bool:
    """Reload every saved .pt and .joblib artifact fresh from MODEL_DIR and assert 100% exact match."""
    state = joblib.load(MODEL_DIR / "feature_state_v6.joblib")
    lr_models = joblib.load(MODEL_DIR / "lr_v6.joblib")
    lgb_models = joblib.load(MODEL_DIR / "lgb_v6.joblib")
    knn_models = joblib.load(MODEL_DIR / "knn_v6.joblib")

    X_master, X_mlp, X_lgb, *_ = build_v6_all_matrices(state=state)
    N = len(X_master)
    p_mlp = np.zeros((N, len(CLASSES)), dtype=np.float32)
    p_lr = np.zeros((N, len(CLASSES)), dtype=np.float32)
    p_lgb = np.zeros((N, len(CLASSES)), dtype=np.float32)
    p_knn = np.zeros((N, len(CLASSES)), dtype=np.float32)

    for fold in range(5):
        ckpt = torch.load(MODEL_DIR / f"mlp_v6_f{fold}.pt", map_location="cpu", weights_only=True)
        m = ResidualMLP(in_dim=ckpt["in_dim"], hidden_dim=ckpt["hidden_dim"], n_classes=ckpt["n_classes"])
        m.load_state_dict(ckpt["state_dict"])
        p_mlp += predict_mlp_proba(m, X_mlp) / 5.0
        p_lr += lr_models[fold].predict_proba(X_master).astype(np.float32) / 5.0
        p_lgb += lgb_models[fold].predict_proba(X_lgb).astype(np.float32) / 5.0
        p_knn += knn_models[fold].predict_proba(X_master).astype(np.float32) / 5.0

    p_ens = 0.35 * p_mlp + 0.25 * p_lr + 0.20 * p_lgb + 0.20 * p_knn
    reloaded = np.array([CLASSES[i] for i in np.argmax(p_ens, axis=1)])
    matches = int((reloaded == expected_preds).sum())
    print(f"[Zero-Retraining Reload Check] Exact matches: {matches}/{N} ({matches/N*100:.2f}%)", flush=True)
    assert matches == N, f"Zero-retraining reload mismatch: {matches}/{N}"
    return True


def render_v6_final_montages(sample_ids, pred_idx, p_ens, aln, lab_micro, beirut_v3):
    rng = np.random.RandomState(SEED + 11)
    T = 108
    bang, bdy, bdx = aln["best_ang"], aln["best_dy"], aln["best_dx"]

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
            dr.text((x0 + 2, y0 + T + 1), f"K{lab_micro[ii]:02d}:{sid[:6]}:{p_ens[ii, r]:.2f}", fill="black")
    out1 = C6_DIR / "montage_v6_final_random_6classes.jpg"
    can.save(out1, quality=88)
    print(f"Saved RANDOM 6-class montage: {out1}", flush=True)

    expl_idx = CLASSES.index("Explosion")
    ex_all = np.where(pred_idx == expl_idx)[0]
    ord_ex = ex_all[np.argsort(-p_ens[ex_all, expl_idx])]
    mid_s = max(0, len(ord_ex) // 2 - 3)
    rows_ex = [("Expl Top6", ord_ex[:6]), ("Expl Mid6", ord_ex[mid_s:mid_s + 6]), ("Expl Low6", ord_ex[-6:])]
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
            dr2.text((x0 + 2, y0 + T + 1), f"{bt}:K{lab_micro[ii]:02d}:{sid[:6]}:{p_ens[ii, expl_idx]:.2f}", fill="black")
    out2 = C6_DIR / "v6_explosion_audit.jpg"
    can2.save(out2, quality=88)
    print(f"Saved Explosion audit montage: {out2}", flush=True)


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
    print("\n=== READ-ONLY MULTI-EVENT VISUAL ANCHOR DIAGNOSTIC (v6) ===", flush=True)
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

    sample_ids = json.load(open(C6_DIR / "sample_ids.json", "r", encoding="utf-8"))
    ref_ids = pd.read_csv(SAMPLE_SUB)["Id"].tolist()
    assert len(sample_ids) == 3180 and sample_ids == ref_ids

    aln = np.load(C4_DIR / "alignment_v4.npz")
    X_master, X_mlp, X_lgb, cld_pre, cld_post, pf5, p5_names, apf, ap_names, p3, state_out = build_v6_all_matrices(None)
    joblib.dump(state_out, MODEL_DIR / "feature_state_v6.joblib")

    cl6 = np.load(C6_DIR / "clusters_v6_micro.npz")
    lab_micro = cl6["lab_micro"]
    cents = cl6["cents"]
    cl3 = np.load(C3_DIR / "clusters_v3_clean.npz")["lab_clean"]
    beirut_v3 = (cl3 == 22)
    assert int(beirut_v3.sum()) == 25

    p5_map = {k: i for i, k in enumerate(p5_names)}
    ap_map = {k: i for i, k in enumerate(ap_names)}

    # 1. Pass-1 Pseudo-Targets from 47 Micro-Clusters + 3 Verified KMeans(2) Sub-Splits
    y_p1 = np.array([CLASSES.index(MICRO_TO_CLASS_V6[int(c)]) for c in lab_micro], dtype=np.int64)

    # K08: higher lap_log_ratio (+0.13, holA=+0.046) -> Fire hillside (42); lower lap_log_ratio (-0.31) -> Volcano ash (61)
    k08_idx = np.where(lab_micro == 8)[0]
    k08_sub = KMeans(n_clusters=2, n_init=30, random_state=SEED).fit_predict(X_master[k08_idx])
    k08_lr = [float(apf[k08_idx[k08_sub == k], ap_map["lap_log_ratio"]].mean()) for k in range(2)]
    k08_fire_idx = k08_idx[k08_sub == int(np.argmax(k08_lr))]
    assert 30 <= len(k08_fire_idx) <= 55, f"Unexpected K08 fire split: {len(k08_fire_idx)}"
    y_p1[k08_fire_idx] = CLASSES.index("Fire")

    # K09: higher lap_log_ratio (+0.66, volcanic lava/ash slopes) -> Volcano (50); other -> Charcoal Fire (51)
    k09_idx = np.where(lab_micro == 9)[0]
    k09_sub = KMeans(n_clusters=2, n_init=30, random_state=SEED).fit_predict(X_master[k09_idx])
    k09_lr = [float(apf[k09_idx[k09_sub == k], ap_map["lap_log_ratio"]].mean()) for k in range(2)]
    k09_volc_idx = k09_idx[k09_sub == int(np.argmax(k09_lr))]
    assert 35 <= len(k09_volc_idx) <= 65, f"Unexpected K09 volcano split: {len(k09_volc_idx)}"
    y_p1[k09_volc_idx] = CLASSES.index("Volcano")

    # K26: higher holA (+0.017) / higher lap_log_ratio (+0.01) -> Palu Tsunami coast (23); other -> Urban EQ (30)
    k26_idx = np.where(lab_micro == 26)[0]
    k26_sub = KMeans(n_clusters=2, n_init=30, random_state=SEED).fit_predict(X_master[k26_idx])
    k26_lr = [float(apf[k26_idx[k26_sub == k], ap_map["lap_log_ratio"]].mean()) for k in range(2)]
    k26_tsun_idx = k26_idx[k26_sub == int(np.argmax(k26_lr))]
    assert 15 <= len(k26_tsun_idx) <= 35, f"Unexpected K26 tsunami split: {len(k26_tsun_idx)}"
    y_p1[k26_tsun_idx] = CLASSES.index("Tsunami")

    p1_counts = {cls: int((y_p1 == i).sum()) for i, cls in enumerate(CLASSES)}
    print("Pass-1 pseudo-target counts:", p1_counts, flush=True)

    # 2. Sample Weighting (Cloud Suppression + Mixed-Boundary Softening)
    N = len(sample_ids)
    w_p1 = np.ones(N, dtype=np.float32)
    for c in range(len(cents)):
        idx = np.where(lab_micro == c)[0]
        sims = X_master[idx] @ cents[c]
        order = np.argsort(np.argsort(sims))
        r_c = order / max(1, len(idx) - 1)
        w_p1[idx] = (0.35 + 0.65 * r_c).astype(np.float32)

    cld_max = np.maximum(cld_pre, cld_post)
    w_cld = np.clip(1.0 - 4.0 * cld_max, 0.12, 1.0).astype(np.float32)
    w_aln = np.clip((aln["z_peak"] - 0.5) / 2.0, 0.45, 1.0).astype(np.float32)
    w_p1 = (w_p1 * w_cld * w_aln).astype(np.float32)
    # Downweight the 2 high-cloud clusters (K29=11.1%, K45=13.5%) & soften K09_volc boundary slightly
    w_p1[np.isin(lab_micro, [29, 45])] *= 0.30
    w_p1[k09_volc_idx] *= 0.75
    w_p1[beirut_v3] = np.maximum(w_p1[beirut_v3], 0.90)

    counts1 = np.bincount(y_p1, minlength=len(CLASSES))
    cls_w1 = (N / (len(CLASSES) * counts1)) ** 0.58
    cls_w1 = cls_w1 / cls_w1.mean()
    print(f"Shapes: X_master={X_master.shape}, X_mlp={X_mlp.shape}, X_lgb={X_lgb.shape}", flush=True)
    print("\n=== PASS 1: BALANCED OOF CORE DISCOVERY ===", flush=True)
    oof_p1, _, fam_p1, oof_fams1 = run_5fold_4family(X_master, X_mlp, X_lgb, y_p1, w_p1, cls_w1, save_artifacts=False)
    print("  Pass-1 OOF consistency (vs Pass-1 pseudo-targets):", fam_p1, flush=True)

    # 3. Anti-Black-Hole Pass-2 Clean-Core Refinement (requires >=3 of 4 diverse model families to agree!)
    y_p2 = y_p1.copy()
    w_p2 = w_p1.copy()
    oof_cls1 = np.argmax(oof_p1, axis=1)
    oof_conf1 = np.max(oof_p1, axis=1)
    p_orig = oof_p1[np.arange(N), y_p1]

    fam_preds = np.stack([np.argmax(pf, axis=1) for pf in oof_fams1], axis=1)  # (N, 4)
    n_fam_agree = (fam_preds == oof_cls1[:, None]).sum(axis=1)

    cloudy_mc = np.isin(lab_micro, [29, 45])
    # Protect K41 Beirut core; require >=3/4 family agreement AND strong prob to prevent Hurricane black-hole!
    flip_norm = (
        (~beirut_v3) & (~cloudy_mc)
        & (oof_cls1 != y_p1)
        & (n_fam_agree >= 3)
        & (oof_conf1 >= 0.65)
        & (p_orig <= 0.25)
    )
    flip_cld = (~beirut_v3) & cloudy_mc & (oof_cls1 != y_p1) & (n_fam_agree >= 3) & (oof_conf1 >= 0.48)
    flip_mask = flip_norm | flip_cld
    y_p2[flip_mask] = oof_cls1[flip_mask]

    agree_core = (oof_cls1 == y_p2)
    w_p2 = np.where(agree_core, np.maximum(w_p2, 0.70 * oof_conf1 + 0.30), 0.35 * w_p2).astype(np.float32)
    w_p2[flip_mask] = (0.65 * oof_conf1[flip_mask] * w_cld[flip_mask]).astype(np.float32)
    w_p2[beirut_v3] = np.maximum(w_p2[beirut_v3], 0.90)

    p2_counts = {cls: int((y_p2 == i).sum()) for i, cls in enumerate(CLASSES)}
    print(f"\n=== PASS 2: ANTI-BLACK-HOLE CLEAN-CORE RETRAINING (refined {int(flip_mask.sum())} consensus samples) ===", flush=True)
    print("  Pass-2 target counts:", p2_counts, flush=True)

    counts2 = np.bincount(y_p2, minlength=len(CLASSES))
    cls_w2 = (N / (len(CLASSES) * counts2)) ** 0.58
    cls_w2 = cls_w2 / cls_w2.mean()
    print("  Pass-2 class weights:", {CLASSES[i]: round(float(cls_w2[i]), 3) for i in range(len(CLASSES))}, flush=True)

    oof_p2, full_p2, fam_p2, _ = run_5fold_4family(X_master, X_mlp, X_lgb, y_p2, w_p2, cls_w2, save_artifacts=True)

    oof_pred = np.argmax(oof_p2, axis=1)
    pred_idx = np.argmax(full_p2, axis=1)
    pred_labels = np.array([CLASSES[i] for i in pred_idx])
    expl_idx = CLASSES.index("Explosion")

    print("\n=== PASS-2 OOF CONSISTENCY DIAGNOSTICS (vs Clean-Core Targets, NOT Official Ground Truth!) ===", flush=True)
    print("  Per-family OOF macro-F1 agreement:", fam_p2, flush=True)
    print(f"  OOF vs 5-Fold Ensemble argmax agreement: {(oof_pred == pred_idx).mean()*100:.2f}%", flush=True)
    print(f"  OOF Explosion={int((oof_pred==expl_idx).sum())}, Ens Explosion={int((pred_idx==expl_idx).sum())}", flush=True)
    print(classification_report(y_p2, oof_pred, target_names=CLASSES, digits=4), flush=True)

    sub_df = pd.DataFrame({"Id": sample_ids, "Phenomenon": pred_labels})
    sub_df.to_csv(SUB_PATH, index=False)
    print(f"Saved submission: {SUB_PATH}", flush=True)

    chk = pd.read_csv(SUB_PATH)
    assert list(chk.columns) == ["Id", "Phenomenon"]
    assert len(chk) == 3180 and chk["Id"].tolist() == ref_ids
    assert chk["Phenomenon"].isna().sum() == 0 and set(chk["Phenomenon"].unique()) == set(CLASSES)

    with open(MODEL_DIR / "cluster_mapping_v6.json", "w", encoding="utf-8") as f:
        json.dump({
            "note": "47 visually audited Mean-Centered Multi-Backbone micro-clusters + 3 verified KMeans(2) sub-splits + 3-of-4 family consensus Pass-2 refinement. NOT ground truth; zero hardcoded test IDs.",
            "micro_to_class_base": {str(k): v for k, v in MICRO_TO_CLASS_V6.items()},
            "sub_splits": {
                "K08_to_Fire": int(len(k08_fire_idx)),
                "K09_to_Volcano": int(len(k09_volc_idx)),
                "K26_to_Tsunami": int(len(k26_tsun_idx)),
            },
            "pass1_counts": p1_counts,
            "pass2_consensus_flipped": int(flip_mask.sum()),
            "pass2_counts": p2_counts,
        }, f, indent=2)

    spec3 = json.load(open(C3_DIR / "pretrained_spec_v3.json", "r", encoding="utf-8"))
    spec4 = json.load(open(C4_DIR / "pretrained_spec_v4.json", "r", encoding="utf-8"))
    spec5 = json.load(open(C5_DIR / "pretrained_spec_v5.json", "r", encoding="utf-8"))
    with open(MODEL_DIR / "pretrained_spec_v6.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "v3": spec3, "v4": spec4, "v5": spec5}, f, indent=2)

    gc.collect()
    verify_zero_retraining_inference(pred_labels)

    audit3 = p3["audit_flags"]
    both_gray = (audit3[:, 2] > 0.5) & (audit3[:, 3] > 0.5)
    any_rot = (audit3[:, 0] > 0.5) | (audit3[:, 1] > 0.5)
    print("\n=== FINAL v6 PREDICTION DISTRIBUTION & LEAKAGE AUDIT ===", flush=True)
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
        f"K41(29)={int((pred_idx[lab_micro==41]==expl_idx).sum())}/29 | "
        f"K36(39)={int((pred_idx[lab_micro==36]==expl_idx).sum())}/39",
        flush=True,
    )

    run_anchor_diagnostic(sample_ids, pred_labels)

    for tag, p_csv in [
        ("v4 (LB 0.37878)", ROOT / "ghani" / "submissions" / "task2" / "submission_v4_ghani.csv"),
        ("v5 (unsub)", ROOT / "ghani" / "submissions" / "task2" / "submission_v5_ghani.csv"),
    ]:
        if p_csv.exists():
            other = pd.read_csv(p_csv).set_index("Id").loc[sample_ids]["Phenomenon"].values
            ct = pd.crosstab(pd.Series(pred_labels, name="v6"), pd.Series(other, name=tag))
            print(f"\n=== CROSS-TAB: v6 vs {tag} ===", flush=True)
            print(ct.to_string(), flush=True)

    render_v6_final_montages(sample_ids, pred_idx, full_p2, aln, lab_micro, beirut_v3)
    print(f"\nDONE v6 Part 2 in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
