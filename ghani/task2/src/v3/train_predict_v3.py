"""Task-2 v3 Training, 3-Family Ensemble, Zero-Retraining Verification & Submission.

Builds the Cloud-Adaptive Shared-Space representation (X_clean) from cached pure-vision
features (DINOv2 ViT-B/14-reg WB, DINOv2 ViT-S/14 Gray-Rot2, ConvNeXt-Base Gray, Phys),
recovers visually audited Louvain (res=2.6) communities, trains a 5-fold stratified
3-family ensemble (ResidualMLP + Multinomial LR + Cosine kNN), saves all inference-ready
artifacts to ghani/task2/models/v3/, verifies 100% zero-retraining reload consistency,
and outputs ghani/submissions/task2/submission_v3_ghani.csv.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "12")

import igraph as ig
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
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
CACHE_DIR = ROOT / "ghani" / "task2" / "cache" / "v3"
MODEL_DIR = ROOT / "ghani" / "task2" / "models" / "v3"
SUB_PATH = ROOT / "ghani" / "submissions" / "task2" / "submission_v3_ghani.csv"
SAMPLE_SUB = ROOT / "Task-2" / "sample_submission.csv"
FEN_DIR = ROOT / "Task-2" / "Fenomena"

DROP_PHYS_COLS = {
    "valid_frac_pre", "valid_frac_post",
    "cloud_frac_pre", "cloud_frac_post",
    "blk_frac_pre", "blk_frac_post",
    "is_gray_pre", "is_gray_post",
}

CLUSTER_TO_CLASS = {
    0: "Volcano",     # n=189: black lava flows (Lower Puna) & volcanic ash plains
    1: "Earthquake",  # n=142: dense urban grid (Mexico/Turkey/Palu)
    2: "Hurricane",   # n=130: US rural woodland & retention ponds
    3: "Tsunami",     # n=147: swept tropical coastlines & flooded coastal villages
    4: "Fire",        # n=76:  mountain roads & hillside wildfire scorch
    5: "Hurricane",   # n=55:  flooded dark-water US farms (Florence/Harvey)
    6: "Hurricane",   # n=111: US tree-lined suburban grids
    7: "Earthquake",  # n=146: commercial/urban terracotta complexes & blocks
    8: "Fire",        # n=107: dry plowed agricultural bushfire paddocks
    9: "Earthquake",  # n=162: dense residential red-roofed urban blocks
    10: "Earthquake", # n=75:  ultra-dense concrete metropolis (Mexico City)
    11: "Tsunami",    # n=105: tropical terraced villages & muddy coastal inundation
    12: "Volcano",    # n=74:  tropical hillside villages blanketed in gray volcanic ash
    13: "Volcano",    # n=156: volcanic gullies & pyroclastic/lahar valleys (Guatemala)
    14: "Fire",       # n=184: scorched dead forest canopy (lap_delta > 0)
    15: "Hurricane",  # n=121: winding flooded rivers & wetlands
    16: "Hurricane",  # n=102: rural forest & creeks (cos=0.97 to C2)
    17: "Fire",       # n=180: completely charred wildfire suburbs (lap=+6387, sat=0.065)
    18: "Earthquake", # n=172: terracotta-roofed concrete apartment blocks w/ collapse
    19: "Hurricane",  # n=169: US Florida/NC canal & lake suburbs
    20: "Volcano",    # n=120: volcanic slopes & ash/plume smoothing (edge=-0.041)
    21: "Hurricane",  # n=84:  US retention ponds & flooded creek bridges
    22: "Explosion",  # n=25:  Beirut Port Explosion off-nadir high-rises & waterfront (lap=25570)
    23: "Hurricane",  # n=124: flat US agricultural fields & highway interchanges
    24: "Earthquake", # n=69:  Mediterranean olive orchards & stone village compounds
    25: "Tsunami",    # n=61:  ocean beaches & eroded coastal settlements
    26: "Volcano",    # n=51:  tropical jungle smoothed by volcanic ash/smoke (edge=-0.052)
    27: "Tsunami",    # n=39:  open ocean / dark coastal water
}

CLUSTER_EVIDENCE = {
    0: "n=189: black lava flows (Lower Puna) & volcanic ash plains",
    1: "n=142: dense urban grid (Mexico/Turkey/Palu)",
    2: "n=130: US rural woodland & retention ponds",
    3: "n=147: swept tropical coastlines & flooded coastal villages",
    4: "n=76: mountain roads & hillside wildfire scorch",
    5: "n=55: flooded dark-water US farms (Florence/Harvey)",
    6: "n=111: US tree-lined suburban grids",
    7: "n=146: commercial/urban terracotta complexes & blocks",
    8: "n=107: dry plowed agricultural bushfire paddocks",
    9: "n=162: dense residential red-roofed urban blocks",
    10: "n=75: ultra-dense concrete metropolis (Mexico City)",
    11: "n=105: tropical terraced villages & muddy coastal inundation",
    12: "n=74: tropical hillside villages blanketed in gray volcanic ash",
    13: "n=156: volcanic gullies & pyroclastic/lahar valleys (Guatemala)",
    14: "n=184: scorched dead forest canopy (lap_delta > 0)",
    15: "n=121: winding flooded rivers & wetlands",
    16: "n=102: rural forest & creeks (cos=0.97 to C2)",
    17: "n=180: completely charred wildfire suburbs (lap=+6387, sat=0.065)",
    18: "n=172: terracotta-roofed concrete apartment blocks w/ collapse",
    19: "n=169: US Florida/NC canal & lake suburbs",
    20: "n=120: volcanic slopes & ash/plume smoothing (edge=-0.041)",
    21: "n=84: US retention ponds & flooded creek bridges",
    22: "n=25: Beirut Port Explosion off-nadir high-rises & waterfront (lap=25570)",
    23: "n=124: flat US agricultural fields & highway interchanges",
    24: "n=69: Mediterranean olive orchards & stone village compounds",
    25: "n=61: ocean beaches & eroded coastal settlements",
    26: "n=51: tropical jungle smoothed by volcanic ash/smoke (edge=-0.052)",
    27: "n=39: open ocean / dark coastal water",
}


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


class ResidualMLP(nn.Module):
    """Regularized residual MLP over multi-modal pure-vision features."""

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


def build_clean_spaces(phys, vitb, cnxb, vits):
    """Construct cloud-adaptive shared-space scene and clean multi-backbone features."""
    pnames = list(phys["feature_names"])
    pf = phys["features"]
    imap = {n: i for i, n in enumerate(pnames)}
    cld_pre = pf[:, imap["cloud_frac_pre"]]
    cld_post = pf[:, imap["cloud_frac_post"]]

    wp = np.clip(1.0 - 6.0 * cld_pre, 0.05, 1.0)[:, None]
    wq = np.clip(1.0 - 6.0 * cld_post, 0.05, 1.0)[:, None]
    wc = np.minimum(wp, wq)

    X_scene = l2norm(np.hstack([
        l2norm(0.85 * wp * l2norm(vitb["pmean_pre"]) + 1.15 * wq * l2norm(vitb["pmean_post"])) * 1.15,
        l2norm(0.85 * wp * l2norm(vits["pmean_pre"]) + 1.15 * wq * l2norm(vits["pmean_post"])) * 1.10,
        l2norm(0.85 * wp * l2norm(cnxb["pre"]) + 1.15 * wq * l2norm(cnxb["post"])) * 1.05,
    ]).astype(np.float32))

    X_diff = l2norm(np.hstack([
        l2norm(vitb["pmean_post"] - vitb["pmean_pre"]),
        l2norm(vits["pmean_post"] - vits["pmean_pre"]),
    ]).astype(np.float32)) * wc

    X_clean = l2norm(np.hstack([X_scene, 0.30 * X_diff]).astype(np.float32))
    return X_scene, X_clean, wc, cld_pre, cld_post, pf, pnames, imap


def build_full_features(
    X_clean: np.ndarray,
    wc: np.ndarray,
    pf: np.ndarray,
    pnames: list[str],
    vitb,
    vits,
    scaler_state: dict | None = None,
):
    """Build 4564-d multi-modal feature matrix; fits scaler if scaler_state is None."""
    if scaler_state is None:
        keep_phys_idx = [i for i, c in enumerate(pnames) if c not in DROP_PHYS_COLS]
        tab_raw = np.hstack([pf[:, keep_phys_idx], vitb["chamfer_stats"], vits["chamfer_stats"]]).astype(np.float32)
        scaler_tab = StandardScaler().fit(tab_raw)
        tab_weight = 0.12
        chg_weight = 0.35
        scaler_state = {
            "scaler_tab": scaler_tab,
            "keep_phys_idx": keep_phys_idx,
            "tab_weight": tab_weight,
            "chg_weight": chg_weight,
            "classes": CLASSES,
        }
    else:
        keep_phys_idx = scaler_state["keep_phys_idx"]
        scaler_tab = scaler_state["scaler_tab"]
        tab_weight = float(scaler_state["tab_weight"])
        chg_weight = float(scaler_state["chg_weight"])
        tab_raw = np.hstack([pf[:, keep_phys_idx], vitb["chamfer_stats"], vits["chamfer_stats"]]).astype(np.float32)

    X_tab_scaled = np.clip(scaler_tab.transform(tab_raw), -8.0, 8.0).astype(np.float32) * tab_weight
    X_chg = l2norm(np.hstack([l2norm(vitb["changed_emb"]), l2norm(vits["changed_emb"])]).astype(np.float32)) * wc * chg_weight
    X_full = np.hstack([X_clean, X_chg, X_tab_scaled]).astype(np.float32)
    return X_full, scaler_state


def compute_louvain_communities(X_scene: np.ndarray, X_clean: np.ndarray, pf: np.ndarray, imap: dict[str, int]):
    """Build k=15 cosine kNN event graph, run Louvain(res=2.6), and verify community fingerprints."""
    nn_graph = NearestNeighbors(n_neighbors=16, metric="cosine").fit(X_clean)
    dists, idxs = nn_graph.kneighbors(X_clean)
    edges, weights = [], []
    for i in range(len(X_clean)):
        for j, d in zip(idxs[i, 1:], dists[i, 1:]):
            if i < j:
                edges.append((i, int(j)))
                weights.append(max(0.0, 1.0 - float(d)))
    g = ig.Graph(n=len(X_clean), edges=edges, directed=False)
    random.seed(SEED)
    part = g.community_multilevel(weights=weights, resolution=2.6)
    lab = np.array(part.membership, dtype=int)

    major = sorted([c for c in set(lab) if (lab == c).sum() >= 15])
    print(f"Louvain res=2.6: {len(major)} major comms, mod={part.modularity:.3f}", flush=True)

    # Mandatory integrity assertions verifying exact match to visually audited communities
    assert major == list(range(28)), f"Unexpected major communities: {major}"
    assert int((lab == 22).sum()) == 25 and float(pf[lab == 22, imap["lap_var_pre"]].mean()) > 22000.0
    assert int((lab == 17).sum()) == 180 and float(pf[lab == 17, imap["sat_mean_post"]].mean()) < 0.08
    assert int((lab == 0).sum()) == 189 and int((lab == 13).sum()) == 156
    assert int((lab == 18).sum()) == 172 and int((lab == 19).sum()) == 169 and int((lab == 25).sum()) == 61

    cents = np.vstack([l2norm(X_scene[lab == c].mean(axis=0, keepdims=True)) for c in major])
    lab_clean = lab.copy()
    tiny_idx = np.where(~np.isin(lab, major))[0]
    for i in tiny_idx:
        best_c = int(np.argmax(X_scene[i] @ cents.T))
        lab_clean[i] = major[best_c]
    print(f"Assigned {len(tiny_idx)} tiny-community samples to nearest major centroid.", flush=True)
    return lab_clean, cents


def compute_sample_weights(X_scene: np.ndarray, lab_clean: np.ndarray, cents: np.ndarray, wc: np.ndarray) -> np.ndarray:
    """Compute within-community centrality percentile & cloud confidence weights."""
    N = len(lab_clean)
    sample_w = np.ones(N, dtype=np.float32)
    for c in range(len(cents)):
        idx = np.where(lab_clean == c)[0]
        sims = X_scene[idx] @ cents[c]
        order = np.argsort(np.argsort(sims))
        r_c = order / max(1, len(idx) - 1)
        c_pen = wc[idx, 0]
        w = (0.35 + 0.65 * r_c) * (0.30 + 0.70 * c_pen)
        if c == 22:
            w = np.maximum(w, 0.85)
        sample_w[idx] = w.astype(np.float32)
    return sample_w


def train_one_mlp_fold(
    fold: int,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    cls_w_t: torch.Tensor,
) -> ResidualMLP:
    """Train one ResidualMLP fold deterministically and save checkpoint."""
    torch.manual_seed(SEED + fold)
    np.random.seed(SEED + fold)
    random.seed(SEED + fold)

    in_dim = X_tr.shape[1]
    model = ResidualMLP(in_dim=in_dim, hidden_dim=384, n_classes=len(CLASSES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-2)
    epochs = 35
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ds = TensorDataset(
        torch.from_numpy(X_tr),
        torch.from_numpy(y_tr),
        torch.from_numpy(w_tr),
    )
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
    ckpt_path = MODEL_DIR / f"mlp_v3_f{fold}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": int(in_dim),
            "hidden_dim": 384,
            "n_classes": len(CLASSES),
        },
        ckpt_path,
    )
    return model


@torch.inference_mode()
def predict_mlp_proba(model: ResidualMLP, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size])
        outs.append(F.softmax(model(xb), dim=1).numpy())
    return np.vstack(outs).astype(np.float32)


def verify_zero_retraining_inference(expected_preds: np.ndarray) -> bool:
    """Reload saved .pt and .joblib artifacts from MODEL_DIR and assert 100% prediction match."""
    scaler_state = joblib.load(MODEL_DIR / "feature_scaler_v3.joblib")
    lr_models = joblib.load(MODEL_DIR / "lr_v3.joblib")
    knn_models = joblib.load(MODEL_DIR / "knn_v3.joblib")

    phys = np.load(CACHE_DIR / "phys_v3.npz")
    vitb = np.load(CACHE_DIR / "vitb_wb.npz")
    cnxb = np.load(CACHE_DIR / "cnxb_gray.npz")
    vits = np.load(CACHE_DIR / "vits_gray_rot2.npz")

    _, X_clean, wc, _, _, pf, pnames, _ = build_clean_spaces(phys, vitb, cnxb, vits)
    X_full, _ = build_full_features(X_clean, wc, pf, pnames, vitb, vits, scaler_state=scaler_state)

    N = len(X_full)
    p_mlp = np.zeros((N, len(CLASSES)), dtype=np.float32)
    p_lr = np.zeros((N, len(CLASSES)), dtype=np.float32)
    p_knn = np.zeros((N, len(CLASSES)), dtype=np.float32)

    for fold in range(5):
        ckpt = torch.load(MODEL_DIR / f"mlp_v3_f{fold}.pt", map_location="cpu", weights_only=True)
        m = ResidualMLP(in_dim=ckpt["in_dim"], hidden_dim=ckpt["hidden_dim"], n_classes=ckpt["n_classes"])
        m.load_state_dict(ckpt["state_dict"])
        p_mlp += predict_mlp_proba(m, X_full) / 5.0
        p_lr += lr_models[fold].predict_proba(X_full).astype(np.float32) / 5.0
        p_knn += knn_models[fold].predict_proba(X_clean).astype(np.float32) / 5.0

    p_ens = 0.45 * p_mlp + 0.35 * p_lr + 0.20 * p_knn
    reloaded_preds = np.array([CLASSES[i] for i in np.argmax(p_ens, axis=1)])
    matches = int((reloaded_preds == expected_preds).sum())
    print(f"[Zero-Retraining Reload Check] Exact matches: {matches}/{N} ({matches/N*100:.2f}%)", flush=True)
    assert matches == N, f"Reload mismatch! {matches}/{N}"
    return True


def render_final_montage(
    sample_ids: list[str],
    pred_idx: np.ndarray,
    p_ens: np.ndarray,
    cg: np.ndarray,
    cld_pre: np.ndarray,
    cld_post: np.ndarray,
    lab_clean: np.ndarray,
):
    """Render 6-class sanity montage (3 highest-confidence + 3 highest-Chamfer-change) + extra explosion strip."""
    clear = (cld_pre < 0.04) & (cld_post < 0.04)
    T = 110
    can = Image.new("RGB", (115 + 6 * (2 * T + 6), len(CLASSES) * (T + 16)), "white")
    dr = ImageDraw.Draw(can)

    for r, cls_name in enumerate(CLASSES):
        idx = np.where(pred_idx == r)[0]
        cidx = idx[clear[idx]]
        use_idx = cidx if len(cidx) >= 6 else idx

        conf_ord = use_idx[np.argsort(-p_ens[use_idx, r])]
        top_conf = [int(x) for x in conf_ord[:3]]
        chg_ord = use_idx[np.argsort(-cg[use_idx, 4])]
        top_chg = [int(x) for x in chg_ord if int(x) not in top_conf][:3]
        picks = top_conf + top_chg

        dr.text((4, r * (T + 16) + 28), f"{cls_name}\nn={len(idx)}\n({len(idx)/len(pred_idx)*100:.1f}%)", fill="black")
        for j, ii in enumerate(picks):
            sid = sample_ids[ii]
            x0 = 115 + j * (2 * T + 6)
            y0 = r * (T + 16) + 2
            for kk, kind in enumerate(("pre", "post")):
                with Image.open(FEN_DIR / sid / f"{kind}.jpg") as im:
                    can.paste(im.convert("RGB").resize((T, T), Image.Resampling.BILINEAR), (x0 + kk * (T + 2), y0))
            tag = "P" if j < 3 else "D"
            dr.text((x0 + 2, y0 + T + 1), f"{tag}:{sid[:7]}:{p_ens[ii, r]:.2f}/c{cg[ii,0]:.2f}", fill="black")

    out_montage = CACHE_DIR / "montage_v3_final_6classes.jpg"
    can.save(out_montage, quality=90)
    print(f"Saved visual sanity montage: {out_montage}", flush=True)

    expl_idx = CLASSES.index("Explosion")
    extra_expl = np.where((pred_idx == expl_idx) & (lab_clean != 22))[0]
    extra_path = None
    if len(extra_expl) > 0:
        extra_ord = extra_expl[np.argsort(-p_ens[extra_expl, expl_idx])][:8]
        cols = len(extra_ord)
        can2 = Image.new("RGB", (115 + cols * (2 * T + 6), T + 18), "white")
        dr2 = ImageDraw.Draw(can2)
        dr2.text((4, 30), f"Extra Expl\n(non-C22)\nn={len(extra_expl)}", fill="black")
        for j, ii in enumerate(extra_ord):
            sid = sample_ids[int(ii)]
            x0 = 115 + j * (2 * T + 6)
            y0 = 2
            for kk, kind in enumerate(("pre", "post")):
                with Image.open(FEN_DIR / sid / f"{kind}.jpg") as im:
                    can2.paste(im.convert("RGB").resize((T, T), Image.Resampling.BILINEAR), (x0 + kk * (T + 2), y0))
            dr2.text((x0 + 2, y0 + T + 1), f"C{lab_clean[ii]}:{sid[:7]}:{p_ens[ii, expl_idx]:.2f}", fill="black")
        extra_path = CACHE_DIR / "v3_extra_explosion.jpg"
        can2.save(extra_path, quality=90)
        print(f"Saved extra explosion montage ({len(extra_expl)} samples): {extra_path}", flush=True)
    return out_montage, extra_path


def main():
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CACHE_DIR / "sample_ids.json", "r", encoding="utf-8") as f:
        sample_ids = json.load(f)
    ref_ids = pd.read_csv(SAMPLE_SUB)["Id"].tolist()
    assert len(sample_ids) == 3180 and sample_ids == ref_ids, "Sample ID alignment mismatch!"

    phys = np.load(CACHE_DIR / "phys_v3.npz")
    vitb = np.load(CACHE_DIR / "vitb_wb.npz")
    cnxb = np.load(CACHE_DIR / "cnxb_gray.npz")
    vits = np.load(CACHE_DIR / "vits_gray_rot2.npz")

    X_scene, X_clean, wc, cld_pre, cld_post, pf, pnames, imap = build_clean_spaces(phys, vitb, cnxb, vits)
    lab_clean, cents = compute_louvain_communities(X_scene, X_clean, pf, imap)

    audit = phys["audit_flags"]
    both_gray = (audit[:, 2] > 0.5) & (audit[:, 3] > 0.5)
    any_rot = (audit[:, 0] > 0.5) | (audit[:, 1] > 0.5)
    cg = vits["chamfer_stats"]

    # Save cleaned clusters & cluster_profiles_r26.csv
    np.savez_compressed(
        CACHE_DIR / "clusters_v3_clean.npz",
        ids=np.array(sample_ids),
        X_scene=X_scene,
        X_clean=X_clean,
        lab_clean=lab_clean,
    )
    prof_rows = []
    for c in range(28):
        m = (lab_clean == c)
        prof_rows.append({
            "cluster": c,
            "mapped_class": CLUSTER_TO_CLASS[c],
            "n": int(m.sum()),
            "pct_both_gray": round(float(both_gray[m].mean()) * 100, 1),
            "pct_rot_corner": round(float(any_rot[m].mean()) * 100, 1),
            "cloud_post_pct": round(float(cld_post[m].mean()) * 100, 2),
            "chamfer_gray_mean": round(float(cg[m, 0].mean()), 3),
            "chamfer_gray_p90": round(float(cg[m, 4].mean()), 3),
            "lap_delta": round(float(pf[m, imap["lap_var_delta"]].mean()), 1),
            "edge_delta": round(float(pf[m, imap["canny_dens_delta"]].mean()), 4),
            "sat_post": round(float(pf[m, imap["sat_mean_post"]].mean()), 3),
        })
    pd.DataFrame(prof_rows).to_csv(CACHE_DIR / "cluster_profiles_r26.csv", index=False)

    # Save cluster mapping & pretrained spec to MODEL_DIR
    y_pseudo = np.array([CLASSES.index(CLUSTER_TO_CLASS[int(c)]) for c in lab_clean], dtype=np.int64)
    raw_counts = {cls: int((y_pseudo == i).sum()) for i, cls in enumerate(CLASSES)}
    mapping_doc = {
        "note": (
            "Unsupervised community-to-class pseudo-targets based on visual inspection of non-cloudy montages "
            "and physical/Chamfer telemetry. NOT ground truth; zero hardcoded test IDs."
        ),
        "cluster_to_class": {str(k): v for k, v in CLUSTER_TO_CLASS.items()},
        "evidence": {str(k): v for k, v in CLUSTER_EVIDENCE.items()},
        "n_per_cluster": {str(c): int((lab_clean == c).sum()) for c in range(28)},
        "class_counts_raw": raw_counts,
    }
    with open(MODEL_DIR / "cluster_mapping_v3.json", "w", encoding="utf-8") as f:
        json.dump(mapping_doc, f, indent=2)
    shutil.copy2(CACHE_DIR / "pretrained_spec_v3.json", MODEL_DIR / "pretrained_spec_v3.json")

    # Build full feature matrix & sample weights
    X_full, scaler_state = build_full_features(X_clean, wc, pf, pnames, vitb, vits, scaler_state=None)
    joblib.dump(scaler_state, MODEL_DIR / "feature_scaler_v3.joblib")
    sample_w = compute_sample_weights(X_scene, lab_clean, cents, wc)
    print(f"X_full shape={X_full.shape} | sample_w min={sample_w.min():.3f}, mean={sample_w.mean():.3f}", flush=True)

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
    oof_knn = np.zeros((N, len(CLASSES)), dtype=np.float32)

    full_mlp = np.zeros((N, len(CLASSES)), dtype=np.float32)
    full_lr = np.zeros((N, len(CLASSES)), dtype=np.float32)
    full_knn = np.zeros((N, len(CLASSES)), dtype=np.float32)

    lr_models = []
    knn_models = []
    expl_idx = CLASSES.index("Explosion")

    for fold, (tr, va) in enumerate(skf.split(X_full, y_pseudo)):
        t_f = time.time()
        mlp_m = train_one_mlp_fold(fold, X_full[tr], y_pseudo[tr], sample_w[tr], cls_w_t)
        oof_mlp[va] = predict_mlp_proba(mlp_m, X_full[va])
        full_mlp += predict_mlp_proba(mlp_m, X_full) / 5.0

        lr_m = LogisticRegression(C=1.5, max_iter=1000, class_weight=cls_w_dict, random_state=SEED + fold)
        lr_m.fit(X_full[tr], y_pseudo[tr], sample_weight=sample_w[tr])
        lr_models.append(lr_m)
        oof_lr[va] = lr_m.predict_proba(X_full[va]).astype(np.float32)
        full_lr += lr_m.predict_proba(X_full).astype(np.float32) / 5.0

        w_thr = np.percentile(sample_w[tr], 20)
        tr_core = tr[(sample_w[tr] >= w_thr) | (y_pseudo[tr] == expl_idx)]
        knn_m = KNeighborsClassifier(n_neighbors=19, metric="cosine", weights="distance")
        knn_m.fit(X_clean[tr_core], y_pseudo[tr_core])
        knn_models.append(knn_m)
        oof_knn[va] = knn_m.predict_proba(X_clean[va]).astype(np.float32)
        full_knn += knn_m.predict_proba(X_clean).astype(np.float32) / 5.0

        print(f"  Fold {fold} trained in {time.time() - t_f:.1f}s", flush=True)

    joblib.dump(lr_models, MODEL_DIR / "lr_v3.joblib")
    joblib.dump(knn_models, MODEL_DIR / "knn_v3.joblib")

    oof_ens = 0.45 * oof_mlp + 0.35 * oof_lr + 0.20 * oof_knn
    p_ens = 0.45 * full_mlp + 0.35 * full_lr + 0.20 * full_knn

    oof_pred = np.argmax(oof_ens, axis=1)
    pred_idx = np.argmax(p_ens, axis=1)
    pred_labels = np.array([CLASSES[i] for i in pred_idx])

    oof_macro_f1_pseudo = float(f1_score(y_pseudo, oof_pred, average="macro"))
    oof_per_cls = f1_score(y_pseudo, oof_pred, average=None, labels=list(range(len(CLASSES))))
    oof_ens_agree = float((oof_pred == pred_idx).mean())
    print("\n=== OOF AGREEMENT DIAGNOSTICS (vs Pseudo-Targets, NOT Official Ground Truth!) ===", flush=True)
    print(f"  OOF Macro-F1 (pseudo-target consistency): {oof_macro_f1_pseudo:.4f}", flush=True)
    print(f"  OOF vs 5-Fold Ensemble argmax agreement : {oof_ens_agree*100:.2f}%", flush=True)
    print(f"  OOF Explosion count={int((oof_pred==expl_idx).sum())}, Ens Explosion count={int((pred_idx==expl_idx).sum())}", flush=True)
    print(classification_report(y_pseudo, oof_pred, target_names=CLASSES, digits=4), flush=True)

    # Save submission
    sub_df = pd.DataFrame({"Id": sample_ids, "Phenomenon": pred_labels})
    sub_df.to_csv(SUB_PATH, index=False)
    print(f"Saved submission: {SUB_PATH}", flush=True)

    # Submission integrity checks
    chk = pd.read_csv(SUB_PATH)
    assert list(chk.columns) == ["Id", "Phenomenon"]
    assert len(chk) == 3180
    assert chk["Id"].isna().sum() == 0 and chk["Phenomenon"].isna().sum() == 0
    assert chk["Id"].nunique() == 3180
    assert chk["Id"].tolist() == ref_ids
    assert set(chk["Phenomenon"].unique()) == set(CLASSES)

    # Zero-retraining reload test
    reload_ok = verify_zero_retraining_inference(pred_labels)

    # Leakage & sanity table per predicted class
    print("\n=== FINAL v3 PREDICTION DISTRIBUTION & LEAKAGE AUDIT ===", flush=True)
    dist_rows = {}
    for i, cls in enumerate(CLASSES):
        m = (pred_idx == i)
        cnt = int(m.sum())
        pct = float(cnt / N * 100.0)
        g_pct = float(both_gray[m].mean() * 100.0)
        r_pct = float(any_rot[m].mean() * 100.0)
        c_pct = float(cld_post[m].mean() * 100.0)
        ch_m = float(cg[m, 0].mean())
        lp_d = float(pf[m, imap["lap_var_delta"]].mean())
        ed_d = float(pf[m, imap["canny_dens_delta"]].mean())
        st_q = float(pf[m, imap["sat_mean_post"]].mean())
        dist_rows[cls] = {
            "count": cnt, "pct": round(pct, 2),
            "pct_both_gray": round(g_pct, 1), "pct_rot_corner": round(r_pct, 1),
            "mean_cloud_post_pct": round(c_pct, 2), "chamfer_gray_mean": round(ch_m, 3),
        }
        print(
            f"  {cls:11s} n={cnt:4d} ({pct:5.1f}%) | gray={g_pct:4.1f}% rot={r_pct:4.1f}% cld_q={c_pct:4.2f}% | "
            f"ch={ch_m:.3f} lap_d={lp_d:+6.0f} edg_d={ed_d:+.4f} sat_q={st_q:.3f}",
            flush=True,
        )

    c22_ret = int((pred_idx[lab_clean == 22] == expl_idx).sum())
    print(f"\nC22 (Beirut) Explosion retention: {c22_ret}/25", flush=True)

    # Cross-tab vs Ghani v1 and v2
    for tag, p_csv in [
        ("Ghani v1 (LB 0.11158)", ROOT / "ghani" / "submissions" / "task2" / "submission_v1_ghani.csv"),
        ("Ghani v2", ROOT / "ghani" / "submissions" / "task2" / "submission_v2_ghani.csv"),
    ]:
        if p_csv.exists():
            other = pd.read_csv(p_csv).set_index("Id").loc[sample_ids]["Phenomenon"].values
            ct = pd.crosstab(pd.Series(pred_labels, name="v3"), pd.Series(other, name=tag))
            print(f"\n=== CROSS-TAB: v3 vs {tag} ===", flush=True)
            print(ct.to_string(), flush=True)

    render_final_montage(sample_ids, pred_idx, p_ens, cg, cld_pre, cld_post, lab_clean)

    summary = {
        "version": "v3",
        "seed": SEED,
        "n_samples": N,
        "elapsed_sec": round(time.time() - t0, 2),
        "oof_agreement_vs_pseudo_targets_NOT_official_macro_f1": {
            "macro_f1": round(oof_macro_f1_pseudo, 4),
            "per_class_f1": {CLASSES[i]: round(float(oof_per_cls[i]), 4) for i in range(len(CLASSES))},
            "oof_vs_ensemble_argmax_agreement": round(oof_ens_agree, 4),
            "oof_explosion_count": int((oof_pred == expl_idx).sum()),
        },
        "final_class_distribution": dist_rows,
        "c22_beirut_retention": f"{c22_ret}/25",
        "zero_retraining_reload_exact_match": reload_ok,
        "submission_path": str(SUB_PATH.relative_to(ROOT)),
    }
    with open(MODEL_DIR / "v3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDONE v3 Part 2 in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
