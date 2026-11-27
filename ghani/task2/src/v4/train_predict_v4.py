"""Task-2 v4 Part 2: Cloud-Free Aligned Event-Graph Training, 4-Family Zero-Retraining Ensemble & Submission.

Builds the audited cloud-free multi-backbone + aligned-damage space (X_v4_best),
recovers the 30 visually audited Louvain (res=2.8) communities, trains a 5-fold stratified
4-family ensemble (ResidualMLP + Multinomial LR + LightGBM + Cosine kNN), saves all
zero-retraining artifacts to ghani/task2/models/v4/, verifies 100% reload consistency,
renders aligned 6-class & Explosion audit montages, and outputs submission_v4_ghani.csv.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "12")

import igraph as ig
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
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(12, os.cpu_count() or 12))

CLASSES = ["Earthquake", "Fire", "Hurricane", "Tsunami", "Volcano", "Explosion"]

ROOT = Path(__file__).resolve().parents[4]
FEN_DIR = ROOT / "Task-2" / "Fenomena"
SAMPLE_SUB = ROOT / "Task-2" / "sample_submission.csv"
C3 = ROOT / "ghani" / "task2" / "cache" / "v3"
C4 = ROOT / "ghani" / "task2" / "cache" / "v4"
MODEL_DIR = ROOT / "ghani" / "task2" / "models" / "v4"
SUB_PATH = ROOT / "ghani" / "submissions" / "task2" / "submission_v4_ghani.csv"

DROP_AP = {"ov_frac", "align_sim", "align_z_peak", "align_reliable"}
DROP_P3 = {
    "valid_frac_pre", "valid_frac_post",
    "cloud_frac_pre", "cloud_frac_post",
    "blk_frac_pre", "blk_frac_post",
    "is_gray_pre", "is_gray_post",
}

CLUSTER_TO_CLASS_V4 = {
    0: "Volcano",     # n=156: black lava flows (Lower Puna) & volcanic slopes (lapR=-0.44, canD=-0.057)
    1: "Earthquake",  # n=184: urban roads/commercial/residential blocks
    2: "Tsunami",     # n=133: Indonesian coastal & striped-post Sunda/Palu Tsunami (incl. beach 9d8231e)
    3: "Hurricane",   # n=100: flooded US creeks & bridges (Florence/Harvey, sat=0.205, lapR=-0.36)
    4: "Tsunami",     # n=143: tropical Indonesian coastal/inland villages + striped post (loc90=0.59)
    5: "Fire",        # n=37:  mountain winding wildfire roads & hillside scorch (SoCal/Woolsey)
    6: "Hurricane",   # n=39:  flooded dark-water US farms (dk=56.8%, Florence/Harvey)
    7: "Volcano",     # n=166: tropical hillside villages coated in volcanic ash/smoke (Guatemala/La Palma)
    8: "Earthquake",  # n=203: Turkey-Syria terracotta apartment blocks w/ collapse (lap3=16293)
    9: "Fire",        # n=173: dry plowed bushfire agricultural paddocks & burn scars (Pinery/SoCal)
    10: "Earthquake", # n=61:  dense red-roofed urban neighborhood
    11: "Earthquake", # n=113: dense urban residential grid (Mexico City / Turkey)
    12: "Earthquake", # n=87:  ultra-dense concrete metropolis (Mexico City)
    13: "Hurricane",  # n=128: US wooded suburban streets
    14: "Hurricane",  # n=88:  US suburban residential & commercial plazas
    15: "Hurricane",  # n=147: US Florida/NC canal & lake suburbs (watI=0.026, hAsym=0.062)
    16: "Fire",       # n=103: scorched dead wildfire forest canopy (Portugal/SoCal)
    17: "Hurricane",  # n=134: winding flooded rivers & wetlands (lapR=-0.60)
    18: "Hurricane",  # n=160: wooded US wetlands & rural creeks
    19: "Earthquake", # n=47: urban commercial/campus & Beirut periphery; pure Beirut isolated via v3_C22
    20: "Volcano",    # n=105: volcanic gullies, pyroclastic & lahar mudflows (Guatemala)
    21: "Fire",       # n=151: 100% charred wildfire suburbs (lapR=+0.33, canD=+0.052, sat=0.060)
    22: "Hurricane",  # n=101: flat US agricultural farmland & flooded ponds
    23: "Earthquake", # n=85:  Mediterranean/Middle-Eastern arid stone villages & olive orchards
    24: "Hurricane",  # n=85:  wooded wetlands & lakes (SG4)
    25: "Tsunami",    # n=63:  tropical coastal towns & inundation (SG0, watI=0.024)
    26: "Tsunami",    # n=41:  100% swept ocean beaches & coastlines (hAsym=0.094 #1!)
    27: "Volcano",    # n=84:  volcanic crater rim, ridges & lahar valleys (lapR=-0.65, canD=-0.068 #1!)
    28: "Earthquake", # n=22:  urban terracotta & commercial blocks (lap3=17500)
    29: "Hurricane",  # n=26:  US wooded lakes & retention ponds (lapR=-0.64, watI=0.034)
}

CLUSTER_EVIDENCE_V4 = {
    0: "n=156: black lava flows (Lower Puna) & volcanic slopes (lapR=-0.44, canD=-0.057)",
    1: "n=184: urban roads/commercial/residential blocks",
    2: "n=133: Indonesian coastal & striped-post Sunda/Palu Tsunami (incl. beach 9d8231e)",
    3: "n=100: flooded US creeks & bridges (Florence/Harvey, sat=0.205, lapR=-0.36)",
    4: "n=143: tropical Indonesian coastal/inland villages + striped post (loc90=0.59)",
    5: "n=37: mountain winding wildfire roads & hillside scorch (SoCal/Woolsey)",
    6: "n=39: flooded dark-water US farms (dk=56.8%, Florence/Harvey)",
    7: "n=166: tropical hillside villages coated in volcanic ash/smoke (Guatemala/La Palma)",
    8: "n=203: Turkey-Syria terracotta apartment blocks w/ collapse (lap3=16293)",
    9: "n=173: dry plowed bushfire agricultural paddocks & burn scars (Pinery/SoCal)",
    10: "n=61: dense red-roofed urban neighborhood",
    11: "n=113: dense urban residential grid (Mexico City / Turkey)",
    12: "n=87: ultra-dense concrete metropolis (Mexico City)",
    13: "n=128: US wooded suburban streets",
    14: "n=88: US suburban residential & commercial plazas",
    15: "n=147: US Florida/NC canal & lake suburbs (watI=0.026, hAsym=0.062)",
    16: "n=103: scorched dead wildfire forest canopy (Portugal/SoCal)",
    17: "n=134: winding flooded rivers & wetlands (lapR=-0.60)",
    18: "n=160: wooded US wetlands & rural creeks",
    19: "n=47: urban commercial/campus & Beirut periphery; pure Beirut isolated via v3_C22",
    20: "n=105: volcanic gullies, pyroclastic & lahar mudflows (Guatemala)",
    21: "n=151: 100% charred wildfire suburbs (lapR=+0.33, canD=+0.052, sat=0.060)",
    22: "n=101: flat US agricultural farmland & flooded ponds",
    23: "n=85: Mediterranean/Middle-Eastern arid stone villages & olive orchards",
    24: "n=85: wooded wetlands & lakes (SG4)",
    25: "n=63: tropical coastal towns & inundation (SG0, watI=0.024)",
    26: "n=41: 100% swept ocean beaches & coastlines (hAsym=0.094 #1!)",
    27: "n=84: volcanic crater rim, ridges & lahar valleys (lapR=-0.65, canD=-0.068 #1!)",
    28: "n=22: urban terracotta & commercial blocks (lap3=17500)",
    29: "n=26: US wooded lakes & retention ponds (lapR=-0.64, watI=0.034)",
}


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def warp_post_to_pre(im_post_rgb: Image.Image, ang_deg: float, dy_g16: int, dx_g16: int, out_size: int) -> Image.Image:
    im_r = im_post_rgb.resize((out_size, out_size), Image.Resampling.BILINEAR)
    im_rot = im_r.rotate(float(ang_deg), resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    dx_px = int(round(dx_g16 * out_size / 16.0))
    dy_px = int(round(dy_g16 * out_size / 16.0))
    if dx_px == 0 and dy_px == 0:
        return im_rot
    canvas = Image.new("RGB", (out_size, out_size), (0, 0, 0))
    canvas.paste(im_rot, (dx_px, dy_px))
    return canvas


class ResidualMLP(nn.Module):
    """Regularized residual MLP over multi-modal pure-vision v4 features."""

    def __init__(self, in_dim: int, hidden_dim: int = 384, n_classes: int = 6):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.35),
        )
        self.res_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res_ln = nn.LayerNorm(hidden_dim)
        self.res_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.res_drop = nn.Dropout(0.30)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        r = self.res_drop(self.res_fc2(F.gelu(self.res_ln(self.res_fc1(h)))))
        return self.head(h + r)


def load_all_caches():
    aln = np.load(C4 / "alignment_v4.npz")
    ap = np.load(C4 / "aligned_phys_v4.npz")
    vs4 = np.load(C4 / "aligned_vits_336.npz")
    vb4 = np.load(C4 / "aligned_vitb_224.npz")
    p3 = np.load(C3 / "phys_v3.npz")
    vb3 = np.load(C3 / "vitb_wb.npz")
    cx3 = np.load(C3 / "cnxb_gray.npz")
    vs3 = np.load(C3 / "vits_gray_rot2.npz")
    return aln, ap, vs4, vb4, p3, vb3, cx3, vs3


def build_v4_matrices(aln, ap, vs4, vb4, p3, vb3, cx3, vs3, scaler_state: dict | None = None):
    """Build exact audited X_v4_best, X_full_v4 (MLP/LR), and X_lgb_v4 (LightGBM)."""
    p3n = list(p3["feature_names"])
    pf3 = p3["features"]
    p3_map = {n: i for i, n in enumerate(p3n)}

    apn = list(ap["feature_names"])
    apf = ap["features"]
    ap_map = {n: i for i, n in enumerate(apn)}

    cld_pre = pf3[:, p3_map["cloud_frac_pre"]]
    cld_post = pf3[:, p3_map["cloud_frac_post"]]
    wp = np.clip(1.0 - 6.0 * cld_pre, 0.05, 1.0)[:, None]
    wq = np.clip(1.0 - 6.0 * cld_post, 0.05, 1.0)[:, None]
    wc = np.minimum(wp, wq)
    w_aln = np.clip((aln["z_peak"] - 1.0) / 1.5, 0.25, 1.0)[:, None]

    if scaler_state is None:
        keep_ap = [i for i, n in enumerate(apn) if n not in DROP_AP]
        keep_p3 = [i for i, n in enumerate(p3n) if n not in DROP_P3]
        scaler_ap = StandardScaler().fit(apf[:, keep_ap])
        scaler_loc = StandardScaler().fit(np.hstack([vs4["loc_stats"], vb4["loc_stats"]]))
    else:
        keep_ap = scaler_state["keep_ap"]
        keep_p3 = scaler_state["keep_p3"]
        scaler_ap = scaler_state["scaler_ap"]
        scaler_loc = scaler_state["scaler_loc"]

    X_ap = np.clip(scaler_ap.transform(apf[:, keep_ap]), -6.0, 6.0).astype(np.float32)
    X_loc = np.clip(scaler_loc.transform(np.hstack([vs4["loc_stats"], vb4["loc_stats"]])), -6.0, 6.0).astype(np.float32)

    X_dm4 = l2norm(np.hstack([
        l2norm(vs4["aligned_delta_emb"]) * 1.10,
        l2norm(vs4["topchg_delta"]) * 1.10,
        l2norm(vs4["topchg_post"]) * 0.65,
        l2norm(vb4["aligned_delta_emb"]) * 1.00,
        l2norm(vb4["topchg_delta"]) * 1.00,
        l2norm(vb4["topchg_post"]) * 0.60,
        l2norm(X_ap) * 0.60,
        l2norm(X_loc) * 0.55,
    ]).astype(np.float32))

    X_sc = l2norm(np.hstack([
        l2norm(0.85 * wp * l2norm(vs4["pmean_pre"]) + 1.15 * wq * l2norm(vs4["pmean_post"])) * 1.15,
        l2norm(0.85 * wp * l2norm(vb4["pmean_pre"]) + 1.15 * wq * l2norm(vb4["pmean_post"])) * 1.10,
        l2norm(0.85 * wp * l2norm(vb3["pmean_pre"]) + 1.15 * wq * l2norm(vb3["pmean_post"])) * 1.05,
        l2norm(0.85 * wp * l2norm(cx3["pre"]) + 1.15 * wq * l2norm(cx3["post"])) * 1.10,
        l2norm(0.85 * wp * l2norm(vs3["pmean_pre"]) + 1.15 * wq * l2norm(vs3["pmean_post"])) * 0.85,
    ]).astype(np.float32))

    X_v4_best = l2norm(np.hstack([X_sc * 1.00, X_dm4 * (wc * w_aln) * 0.72]).astype(np.float32))

    tab_raw = np.hstack([
        pf3[:, keep_p3],
        apf[:, keep_ap],
        vs4["loc_stats"],
        vb4["loc_stats"],
        vb3["chamfer_stats"],
        vs3["chamfer_stats"],
    ]).astype(np.float32)

    if scaler_state is None:
        scaler_tab_all = StandardScaler().fit(tab_raw)
        pca_lgb = PCA(n_components=64, random_state=SEED).fit(X_v4_best)
        scaler_state = {
            "scaler_ap": scaler_ap,
            "keep_ap": keep_ap,
            "scaler_loc": scaler_loc,
            "scaler_tab_all": scaler_tab_all,
            "keep_p3": keep_p3,
            "pca_lgb": pca_lgb,
            "classes": CLASSES,
        }
    else:
        scaler_tab_all = scaler_state["scaler_tab_all"]
        pca_lgb = scaler_state["pca_lgb"]

    X_tab_all = np.clip(scaler_tab_all.transform(tab_raw), -8.0, 8.0).astype(np.float32)
    X_chg_v3 = l2norm(np.hstack([l2norm(vb3["changed_emb"]), l2norm(vs3["changed_emb"])]).astype(np.float32)) * wc * 0.25
    X_full_v4 = np.hstack([X_v4_best, X_chg_v3, X_tab_all * 0.10]).astype(np.float32)
    X_lgb_v4 = np.hstack([pca_lgb.transform(X_v4_best).astype(np.float32), X_tab_all]).astype(np.float32)

    return X_v4_best, X_full_v4, X_lgb_v4, wc, cld_pre, cld_post, pf3, p3_map, apf, ap_map, scaler_state


def compute_louvain_v4(X_v4_best: np.ndarray, pf3: np.ndarray, p3_map: dict[str, int]):
    nn_graph = NearestNeighbors(n_neighbors=16, metric="cosine").fit(X_v4_best)
    dists, idxs = nn_graph.kneighbors(X_v4_best)
    edges, weights = [], []
    for i in range(len(X_v4_best)):
        for j, d in zip(idxs[i, 1:], dists[i, 1:]):
            if i < j:
                edges.append((i, int(j)))
                weights.append(max(0.0, 1.0 - float(d)))
    g = ig.Graph(n=len(X_v4_best), edges=edges, directed=False)
    random.seed(SEED)
    part = g.community_multilevel(weights=weights, resolution=2.8)
    lab = np.array(part.membership, dtype=int)

    major = sorted([c for c in set(lab) if (lab == c).sum() >= 15])
    print(f"Louvain res=2.8: {len(major)} major comms, mod={part.modularity:.3f}", flush=True)

    # Mandatory integrity assertions matching visually audited 30 communities
    assert major == list(range(30)), f"Unexpected major communities: {major}"
    assert int((lab == 19).sum()) == 47 and float(pf3[lab == 19, p3_map["lap_var_pre"]].mean()) > 20000.0
    assert int((lab == 21).sum()) == 151 and float(pf3[lab == 21, p3_map["sat_mean_post"]].mean()) < 0.07
    assert (
        int((lab == 8).sum()) == 203
        and int((lab == 0).sum()) == 156
        and int((lab == 26).sum()) == 41
        and int((lab == 15).sum()) == 147
        and int((lab == 2).sum()) == 133
    )

    cents = np.vstack([l2norm(X_v4_best[lab == c].mean(axis=0, keepdims=True)) for c in major])
    lab_clean = lab.copy()
    tiny_idx = np.where(~np.isin(lab, major))[0]
    for i in tiny_idx:
        best_c = int(np.argmax(X_v4_best[i] @ cents.T))
        lab_clean[i] = major[best_c]
    print(f"Assigned {len(tiny_idx)} tiny-community samples to nearest major centroid.", flush=True)
    return lab_clean, cents


def compute_sample_weights(
    X_v4_best: np.ndarray,
    lab_clean: np.ndarray,
    cents: np.ndarray,
    wc: np.ndarray,
    beirut_mask: np.ndarray,
) -> np.ndarray:
    N = len(lab_clean)
    sample_w = np.ones(N, dtype=np.float32)
    for c in range(len(cents)):
        idx = np.where(lab_clean == c)[0]
        sims = X_v4_best[idx] @ cents[c]
        order = np.argsort(np.argsort(sims))
        r_c = order / max(1, len(idx) - 1)
        w = (0.35 + 0.65 * r_c) * (0.30 + 0.70 * wc[idx, 0])
        sample_w[idx] = w.astype(np.float32)
    sample_w[beirut_mask] = np.maximum(sample_w[beirut_mask], 0.85)
    return sample_w


def train_one_mlp_fold(fold: int, X_tr: np.ndarray, y_tr: np.ndarray, w_tr: np.ndarray, cls_w_t: torch.Tensor) -> ResidualMLP:
    torch.manual_seed(SEED + fold)
    np.random.seed(SEED + fold)
    random.seed(SEED + fold)

    in_dim = X_tr.shape[1]
    model = ResidualMLP(in_dim=in_dim, hidden_dim=384, n_classes=len(CLASSES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-2)
    epochs = 35
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr), torch.from_numpy(w_tr))
    gen = torch.Generator().manual_seed(SEED + fold)
    loader = DataLoader(ds, batch_size=128, shuffle=True, generator=gen, num_workers=0)

    model.train()
    for _ in range(epochs):
        for xb, yb, wb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss_vec = F.cross_entropy(logits, yb, weight=cls_w_t, label_smoothing=0.03, reduction="none")
            loss = (loss_vec * wb).mean()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": int(in_dim),
            "hidden_dim": 384,
            "n_classes": len(CLASSES),
        },
        MODEL_DIR / f"mlp_v4_f{fold}.pt",
    )
    return model


@torch.inference_mode()
def predict_mlp_proba(model: ResidualMLP, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), batch_size):
        outs.append(F.softmax(model(torch.from_numpy(X[i:i + batch_size])), dim=1).numpy())
    return np.vstack(outs).astype(np.float32)


def verify_zero_retraining_inference(expected_preds: np.ndarray) -> bool:
    """Reload saved .pt and .joblib artifacts fresh from MODEL_DIR and assert 100% exact match."""
    scaler_state = joblib.load(MODEL_DIR / "feature_scaler_v4.joblib")
    lr_models = joblib.load(MODEL_DIR / "lr_v4.joblib")
    lgb_models = joblib.load(MODEL_DIR / "lgb_v4.joblib")
    knn_models = joblib.load(MODEL_DIR / "knn_v4.joblib")

    aln, ap, vs4, vb4, p3, vb3, cx3, vs3 = load_all_caches()
    X_v4_best, X_full_v4, X_lgb_v4, _, _, _, _, _, _, _, _ = build_v4_matrices(
        aln, ap, vs4, vb4, p3, vb3, cx3, vs3, scaler_state=scaler_state
    )

    N = len(X_v4_best)
    p_mlp = np.zeros((N, len(CLASSES)), dtype=np.float32)
    p_lr = np.zeros((N, len(CLASSES)), dtype=np.float32)
    p_lgb = np.zeros((N, len(CLASSES)), dtype=np.float32)
    p_knn = np.zeros((N, len(CLASSES)), dtype=np.float32)

    for fold in range(5):
        ckpt = torch.load(MODEL_DIR / f"mlp_v4_f{fold}.pt", map_location="cpu", weights_only=True)
        m = ResidualMLP(in_dim=ckpt["in_dim"], hidden_dim=ckpt["hidden_dim"], n_classes=ckpt["n_classes"])
        m.load_state_dict(ckpt["state_dict"])
        p_mlp += predict_mlp_proba(m, X_full_v4) / 5.0
        p_lr += lr_models[fold].predict_proba(X_full_v4).astype(np.float32) / 5.0
        p_lgb += lgb_models[fold].predict_proba(X_lgb_v4).astype(np.float32) / 5.0
        p_knn += knn_models[fold].predict_proba(X_v4_best).astype(np.float32) / 5.0

    p_ens = 0.35 * p_mlp + 0.25 * p_lr + 0.20 * p_lgb + 0.20 * p_knn
    reloaded_preds = np.array([CLASSES[i] for i in np.argmax(p_ens, axis=1)])
    matches = int((reloaded_preds == expected_preds).sum())
    print(f"[Zero-Retraining Reload Check] Exact matches: {matches}/{N} ({matches/N*100:.2f}%)", flush=True)
    assert matches == N, f"Reload mismatch: {matches}/{N}"
    return True


def render_final_montages(
    sample_ids: list[str],
    pred_idx: np.ndarray,
    p_ens: np.ndarray,
    aln,
    loc_v: np.ndarray,
    cld_pre: np.ndarray,
    cld_post: np.ndarray,
    lab_clean: np.ndarray,
    lab_v3: np.ndarray | None = None,
):
    clear = (cld_pre < 0.04) & (cld_post < 0.04)
    T = 110
    can = Image.new("RGB", (115 + 6 * (2 * T + 6), len(CLASSES) * (T + 16)), "white")
    dr = ImageDraw.Draw(can)

    for r, cls_name in enumerate(CLASSES):
        idx = np.where(pred_idx == r)[0]
        cidx = idx[clear[idx]]
        use_idx = cidx if len(cidx) >= 6 else idx

        top_conf = [int(x) for x in use_idx[np.argsort(-p_ens[use_idx, r])][:3]]
        top_chg = [int(x) for x in use_idx[np.argsort(-loc_v[use_idx, 4])] if int(x) not in top_conf][:3]
        picks = top_conf + top_chg

        dr.text((4, r * (T + 16) + 28), f"{cls_name}\nn={len(idx)}\n({len(idx)/len(pred_idx)*100:.1f}%)", fill="black")
        for j, ii in enumerate(picks):
            sid = sample_ids[ii]
            with Image.open(FEN_DIR / sid / "pre.jpg") as im1, Image.open(FEN_DIR / sid / "post.jpg") as im2:
                r1 = im1.convert("RGB").resize((T, T), Image.Resampling.BILINEAR)
                r2 = warp_post_to_pre(im2.convert("RGB"), aln["best_ang"][ii], aln["best_dy"][ii], aln["best_dx"][ii], T)
            x0 = 115 + j * (2 * T + 6)
            y0 = r * (T + 16) + 2
            can.paste(r1, (x0, y0))
            can.paste(r2, (x0 + T + 2, y0))
            tag = "P" if j < 3 else "D"
            dr.text((x0 + 2, y0 + T + 1), f"{tag}:{sid[:7]}:{p_ens[ii, r]:.2f}/l{loc_v[ii,4]:.2f}", fill="black")

    out_montage = C4 / "montage_v4_final_6classes.jpg"
    can.save(out_montage, quality=90)
    print(f"Saved aligned 6-class montage: {out_montage}", flush=True)

    # Explosion audit montage: 2 rows x 6 pairs = 6 highest-confidence + 6 lowest-confidence clear Explosion predictions
    expl_idx = CLASSES.index("Explosion")
    expl_all = np.where(pred_idx == expl_idx)[0]
    expl_clr = expl_all[clear[expl_all]]
    use_ex = expl_clr if len(expl_clr) >= 12 else expl_all
    ord_desc = use_ex[np.argsort(-p_ens[use_ex, expl_idx])]
    hi_ex = [int(x) for x in ord_desc[:6]]
    lo_ex = [int(x) for x in ord_desc[::-1] if int(x) not in hi_ex][:6]

    T2 = 120
    can2 = Image.new("RGB", (110 + 6 * (2 * T2 + 6), 2 * (T2 + 16)), "white")
    dr2 = ImageDraw.Draw(can2)
    for row_i, (r_lbl, picks_ex) in enumerate([("Expl Top6\nConf", hi_ex), ("Expl Low6\nBorder", lo_ex)]):
        dr2.text((4, row_i * (T2 + 16) + 32), f"{r_lbl}\nn={len(expl_all)}", fill="black")
        for j, ii in enumerate(picks_ex):
            sid = sample_ids[ii]
            with Image.open(FEN_DIR / sid / "pre.jpg") as im1, Image.open(FEN_DIR / sid / "post.jpg") as im2:
                r1 = im1.convert("RGB").resize((T2, T2), Image.Resampling.BILINEAR)
                r2 = warp_post_to_pre(im2.convert("RGB"), aln["best_ang"][ii], aln["best_dy"][ii], aln["best_dx"][ii], T2)
            x0 = 110 + j * (2 * T2 + 6)
            y0 = row_i * (T2 + 16) + 2
            can2.paste(r1, (x0, y0))
            can2.paste(r2, (x0 + T2 + 2, y0))
            tag = ("B" if lab_v3[ii] == 22 else "N") if lab_v3 is not None else ""
            dr2.text((x0 + 2, y0 + T2 + 1), f"{tag}:C{lab_clean[ii]}:{sid[:7]}:{p_ens[ii, expl_idx]:.2f}", fill="black")
    out_expl = C4 / "v4_explosion_audit.jpg"
    can2.save(out_expl, quality=90)
    print(f"Saved explosion audit montage: {out_expl}", flush=True)


def main():
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(C4 / "sample_ids.json", "r", encoding="utf-8") as f:
        sample_ids = json.load(f)
    ref_ids = pd.read_csv(SAMPLE_SUB)["Id"].tolist()
    assert len(sample_ids) == 3180 and sample_ids == ref_ids, "Sample ID alignment mismatch!"

    aln, ap, vs4, vb4, p3, vb3, cx3, vs3 = load_all_caches()
    X_v4_best, X_full_v4, X_lgb_v4, wc, cld_pre, cld_post, pf3, p3_map, apf, ap_map, scaler_state = build_v4_matrices(
        aln, ap, vs4, vb4, p3, vb3, cx3, vs3, scaler_state=None
    )
    lab_clean, cents = compute_louvain_v4(X_v4_best, pf3, p3_map)

    audit3 = p3["audit_flags"]
    both_gray = (audit3[:, 2] > 0.5) & (audit3[:, 3] > 0.5)
    any_rot = (audit3[:, 0] > 0.5) | (audit3[:, 1] > 0.5)
    loc_v = vs4["loc_stats"]

    np.savez_compressed(
        C4 / "clusters_v4_clean.npz",
        ids=np.array(sample_ids),
        X_v4_best=X_v4_best,
        lab_clean=lab_clean,
    )
    prof_rows = []
    for c in range(30):
        m = (lab_clean == c)
        prof_rows.append({
            "cluster": c,
            "mapped_class": CLUSTER_TO_CLASS_V4[c],
            "n": int(m.sum()),
            "pct_both_gray": round(float(both_gray[m].mean()) * 100, 1),
            "pct_rot_corner": round(float(any_rot[m].mean()) * 100, 1),
            "cloud_post_pct": round(float(cld_post[m].mean()) * 100, 2),
            "loc90_vits336": round(float(loc_v[m, 4].mean()), 3),
            "lap_log_ratio": round(float(apf[m, ap_map["lap_log_ratio"]].mean()), 3),
            "canny_delta": round(float(apf[m, ap_map["canny_delta"]].mean()), 4),
            "water_inund_frac": round(float(apf[m, ap_map["water_inund_frac"]].mean()), 4),
            "half_asym_max": round(float(apf[m, ap_map["half_asym_max"]].mean()), 4),
        })
    pd.DataFrame(prof_rows).to_csv(C4 / "cluster_profiles_v4_r28.csv", index=False)

    y_pseudo = np.array([CLASSES.index(CLUSTER_TO_CLASS_V4[int(c)]) for c in lab_clean], dtype=np.int64)
    c9_idx = np.where(lab_clean == 9)[0]
    c9_sub = KMeans(n_clusters=2, n_init=30, random_state=SEED).fit_predict(X_v4_best[c9_idx])
    c9_means = [float(pf3[c9_idx[c9_sub == k], p3_map["canny_dens_pre"]].mean()) for k in range(2)]
    ocean_idx = c9_idx[c9_sub == int(np.argmin(c9_means))]
    assert 30 <= len(ocean_idx) <= 60, f"Unexpected C9 ocean split size: {len(ocean_idx)}"
    y_pseudo[ocean_idx] = CLASSES.index("Tsunami")

    lab_v3 = np.load(C3 / "clusters_v3_clean.npz")["lab_clean"]
    beirut_mask = (lab_v3 == 22)
    assert int(beirut_mask.sum()) == 25, f"Unexpected v3_C22 count: {int(beirut_mask.sum())}"
    y_pseudo[beirut_mask] = CLASSES.index("Explosion")

    raw_counts = {cls: int((y_pseudo == i).sum()) for i, cls in enumerate(CLASSES)}
    with open(MODEL_DIR / "cluster_mapping_v4.json", "w", encoding="utf-8") as f:
        json.dump({
            "note": (
                "Unsupervised community-to-class pseudo-targets from visual inspection of aligned montages "
                "and aligned physics/DiD telemetry. NOT ground truth; zero hardcoded test IDs."
            ),
            "cluster_to_class": {str(k): v for k, v in CLUSTER_TO_CLASS_V4.items()},
            "evidence": {str(k): v for k, v in CLUSTER_EVIDENCE_V4.items()},
            "refinements": {
                "c9_ocean_split_to_tsunami": int(len(ocean_idx)),
                "beirut_explosion_v3_c22": int(beirut_mask.sum()),
            },
            "n_per_cluster": {str(c): int((lab_clean == c).sum()) for c in range(30)},
            "class_counts_raw": raw_counts,
        }, f, indent=2)

    # Merge pretrained_spec_v3.json + pretrained_spec_v4.json into MODEL_DIR/pretrained_spec_v4.json
    spec3 = json.load(open(C3 / "pretrained_spec_v3.json", "r", encoding="utf-8"))
    spec4 = json.load(open(C4 / "pretrained_spec_v4.json", "r", encoding="utf-8"))
    merged_spec = {
        "seed": SEED,
        "n_samples": len(sample_ids),
        "alignment": spec4.get("alignment", ""),
        "preprocessing_v3_and_v4": spec3.get("preprocessing", {}),
        "models": spec3.get("models", []) + [
            {"stage": "v4_aligned", **m} for m in spec4.get("models", [])
        ],
    }
    with open(MODEL_DIR / "pretrained_spec_v4.json", "w", encoding="utf-8") as f:
        json.dump(merged_spec, f, indent=2)

    joblib.dump(scaler_state, MODEL_DIR / "feature_scaler_v4.joblib")
    sample_w = compute_sample_weights(X_v4_best, lab_clean, cents, wc, beirut_mask)
    print(
        f"Shapes: X_v4_best={X_v4_best.shape}, X_full_v4={X_full_v4.shape}, X_lgb_v4={X_lgb_v4.shape} | "
        f"sample_w min={sample_w.min():.3f}, mean={sample_w.mean():.3f}",
        flush=True,
    )

    counts = np.bincount(y_pseudo, minlength=len(CLASSES))
    cls_w = (len(y_pseudo) / (len(CLASSES) * counts)) ** 0.55
    cls_w = cls_w / cls_w.mean()
    cls_w_t = torch.tensor(cls_w, dtype=torch.float32)
    cls_w_dict = {i: float(cls_w[i]) for i in range(len(CLASSES))}
    print("Class weights:", {CLASSES[i]: round(float(cls_w[i]), 3) for i in range(len(CLASSES))}, flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    N = len(sample_ids)
    oof_mlp = np.zeros((N, len(CLASSES)), dtype=np.float32)
    oof_lr = np.zeros((N, len(CLASSES)), dtype=np.float32)
    oof_lgb = np.zeros((N, len(CLASSES)), dtype=np.float32)
    oof_knn = np.zeros((N, len(CLASSES)), dtype=np.float32)

    full_mlp = np.zeros((N, len(CLASSES)), dtype=np.float32)
    full_lr = np.zeros((N, len(CLASSES)), dtype=np.float32)
    full_lgb = np.zeros((N, len(CLASSES)), dtype=np.float32)
    full_knn = np.zeros((N, len(CLASSES)), dtype=np.float32)

    lr_models, lgb_models, knn_models = [], [], []
    expl_idx = CLASSES.index("Explosion")

    for fold, (tr, va) in enumerate(skf.split(X_v4_best, y_pseudo)):
        t_f = time.time()

        mlp_m = train_one_mlp_fold(fold, X_full_v4[tr], y_pseudo[tr], sample_w[tr], cls_w_t)
        oof_mlp[va] = predict_mlp_proba(mlp_m, X_full_v4[va])
        full_mlp += predict_mlp_proba(mlp_m, X_full_v4) / 5.0

        lr_m = LogisticRegression(C=12.0, max_iter=1000, class_weight=cls_w_dict, random_state=SEED + fold)
        lr_m.fit(X_full_v4[tr], y_pseudo[tr], sample_weight=sample_w[tr])
        lr_models.append(lr_m)
        oof_lr[va] = lr_m.predict_proba(X_full_v4[va]).astype(np.float32)
        full_lr += lr_m.predict_proba(X_full_v4).astype(np.float32) / 5.0

        lgb_m = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(CLASSES),
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=15,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            random_state=SEED + fold,
            n_jobs=12,
            verbose=-1,
        )
        lgb_m.fit(X_lgb_v4[tr], y_pseudo[tr], sample_weight=sample_w[tr] * cls_w[y_pseudo[tr]])
        lgb_models.append(lgb_m)
        oof_lgb[va] = lgb_m.predict_proba(X_lgb_v4[va]).astype(np.float32)
        full_lgb += lgb_m.predict_proba(X_lgb_v4).astype(np.float32) / 5.0

        w_thr = np.percentile(sample_w[tr], 20)
        tr_core = tr[(sample_w[tr] >= w_thr) | (y_pseudo[tr] == expl_idx)]
        knn_m = KNeighborsClassifier(n_neighbors=19, metric="cosine", weights="distance")
        knn_m.fit(X_v4_best[tr_core], y_pseudo[tr_core])
        knn_models.append(knn_m)
        oof_knn[va] = knn_m.predict_proba(X_v4_best[va]).astype(np.float32)
        full_knn += knn_m.predict_proba(X_v4_best).astype(np.float32) / 5.0

        print(f"  Fold {fold} trained in {time.time() - t_f:.1f}s", flush=True)

    joblib.dump(lr_models, MODEL_DIR / "lr_v4.joblib")
    joblib.dump(lgb_models, MODEL_DIR / "lgb_v4.joblib")
    joblib.dump(knn_models, MODEL_DIR / "knn_v4.joblib")

    oof_ens = 0.35 * oof_mlp + 0.25 * oof_lr + 0.20 * oof_lgb + 0.20 * oof_knn
    p_ens = 0.35 * full_mlp + 0.25 * full_lr + 0.20 * full_lgb + 0.20 * full_knn

    oof_pred = np.argmax(oof_ens, axis=1)
    pred_idx = np.argmax(p_ens, axis=1)
    pred_labels = np.array([CLASSES[i] for i in pred_idx])

    fam_oof = {
        "ResidualMLP": round(float(f1_score(y_pseudo, oof_mlp.argmax(1), average="macro")), 4),
        "LogisticRegression": round(float(f1_score(y_pseudo, oof_lr.argmax(1), average="macro")), 4),
        "LightGBM": round(float(f1_score(y_pseudo, oof_lgb.argmax(1), average="macro")), 4),
        "CosineKNN": round(float(f1_score(y_pseudo, oof_knn.argmax(1), average="macro")), 4),
        "Ensemble_4Family": round(float(f1_score(y_pseudo, oof_pred, average="macro")), 4),
    }
    oof_per_cls = f1_score(y_pseudo, oof_pred, average=None, labels=list(range(len(CLASSES))))
    oof_ens_agree = float((oof_pred == pred_idx).mean())

    print("\n=== OOF CONSISTENCY DIAGNOSTICS (vs Pseudo-Targets, NOT Official Ground Truth!) ===", flush=True)
    print("  Per-family OOF macro-F1 agreement:", fam_oof, flush=True)
    print(f"  OOF vs 5-Fold Ensemble argmax agreement: {oof_ens_agree*100:.2f}%", flush=True)
    print(f"  OOF Explosion={int((oof_pred==expl_idx).sum())}, Ens Explosion={int((pred_idx==expl_idx).sum())}", flush=True)
    print(classification_report(y_pseudo, oof_pred, target_names=CLASSES, digits=4), flush=True)

    sub_df = pd.DataFrame({"Id": sample_ids, "Phenomenon": pred_labels})
    sub_df.to_csv(SUB_PATH, index=False)
    print(f"Saved submission: {SUB_PATH}", flush=True)

    chk = pd.read_csv(SUB_PATH)
    assert list(chk.columns) == ["Id", "Phenomenon"]
    assert len(chk) == 3180
    assert chk["Id"].isna().sum() == 0 and chk["Phenomenon"].isna().sum() == 0
    assert chk["Id"].nunique() == 3180 and chk["Id"].tolist() == ref_ids
    assert set(chk["Phenomenon"].unique()) == set(CLASSES)

    reload_ok = verify_zero_retraining_inference(pred_labels)

    print("\n=== FINAL v4 PREDICTION DISTRIBUTION & LEAKAGE AUDIT ===", flush=True)
    dist_rows = {}
    i_sat = list(p3["feature_names"]).index("sat_mean_post")
    for i, cls in enumerate(CLASSES):
        m = (pred_idx == i)
        cnt = int(m.sum())
        pct = float(cnt / N * 100.0)
        g_pct = float(both_gray[m].mean() * 100.0)
        r_pct = float(any_rot[m].mean() * 100.0)
        c_pct = float(cld_post[m].mean() * 100.0)
        l90 = float(loc_v[m, 4].mean())
        lr_m = float(apf[m, ap_map["lap_log_ratio"]].mean())
        cd_m = float(apf[m, ap_map["canny_delta"]].mean())
        st_m = float(pf3[m, i_sat].mean())
        dist_rows[cls] = {
            "count": cnt, "pct": round(pct, 2),
            "pct_both_gray": round(g_pct, 1), "pct_rot_corner": round(r_pct, 1),
            "mean_cloud_post_pct": round(c_pct, 2), "loc_p90": round(l90, 3),
        }
        print(
            f"  {cls:11s} n={cnt:4d} ({pct:5.1f}%) | gray={g_pct:4.1f}% rot={r_pct:4.1f}% cld_q={c_pct:4.2f}% | "
            f"loc90={l90:.3f} lapR={lr_m:+.3f} canD={cd_m:+.4f} sat={st_m:.3f}",
            flush=True,
        )

    c19_ret = int((pred_idx[lab_clean == 19] == expl_idx).sum())
    c22_ret = int((pred_idx[beirut_mask] == expl_idx).sum())
    print(f"\nBeirut retention: C19(47)={c19_ret}/47 | v3_C22(25)={c22_ret}/25", flush=True)

    for tag, p_csv in [
        ("v3 (LB 0.37895)", ROOT / "ghani" / "submissions" / "task2" / "submission_v3_ghani.csv"),
        ("v1 (LB 0.11158)", ROOT / "ghani" / "submissions" / "task2" / "submission_v1_ghani.csv"),
    ]:
        if p_csv.exists():
            other = pd.read_csv(p_csv).set_index("Id").loc[sample_ids]["Phenomenon"].values
            ct = pd.crosstab(pd.Series(pred_labels, name="v4"), pd.Series(other, name=tag))
            print(f"\n=== CROSS-TAB: v4 vs {tag} ===", flush=True)
            print(ct.to_string(), flush=True)

    render_final_montages(sample_ids, pred_idx, p_ens, aln, loc_v, cld_pre, cld_post, lab_clean, lab_v3=lab_v3)

    summary = {
        "version": "v4",
        "seed": SEED,
        "n_samples": N,
        "elapsed_sec": round(time.time() - t0, 2),
        "oof_consistency_vs_pseudo_targets_NOT_official_macro_f1": {
            "family_macro_f1": fam_oof,
            "per_class_f1": {CLASSES[i]: round(float(oof_per_cls[i]), 4) for i in range(len(CLASSES))},
            "oof_vs_ensemble_argmax_agreement": round(oof_ens_agree, 4),
            "oof_explosion_count": int((oof_pred == expl_idx).sum()),
        },
        "final_class_distribution": dist_rows,
        "c19_beirut_retention": f"{c19_ret}/47",
        "v3_c22_beirut_retention": f"{c22_ret}/25",
        "zero_retraining_reload_exact_match": reload_ok,
        "submission_path": str(SUB_PATH.relative_to(ROOT)),
    }
    with open(MODEL_DIR / "v4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDONE v4 Part 2 in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
