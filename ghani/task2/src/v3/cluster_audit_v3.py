"""Task-2 v3 Event-Graph Over-Clustering & Visual Audit Gatekeeper.

Builds F_scene & F_gray_pure (L2-normalized multi-backbone blocks, NO PCA-whitening),
evaluates Louvain community detection, Ward Agglomerative, and KMeans stability,
checks for augmentation leakage (both_gray / rot_corner), exports cluster_profiles_k16.csv,
and renders visual montages (3 nearest-to-centroid + 3 highest-Chamfer-change per cluster).
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[4]
FEN_DIR = ROOT / "Task-2" / "Fenomena"
CACHE_DIR = ROOT / "ghani" / "task2" / "cache" / "v3"


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def zscore_l2(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    return l2norm(((x - mu) / sd).astype(np.float32))


def build_feature_spaces():
    with open(CACHE_DIR / "sample_ids.json", "r", encoding="utf-8") as f:
        sample_ids = json.load(f)

    phys = np.load(CACHE_DIR / "phys_v3.npz")
    vitb = np.load(CACHE_DIR / "vitb_wb.npz")
    cnxb = np.load(CACHE_DIR / "cnxb_gray.npz")
    vits = np.load(CACHE_DIR / "vits_gray_rot2.npz")

    # Exclude raw gray/artifact flags from clustering features to avoid leakage
    p_names = list(phys["feature_names"])
    p_mat = phys["features"]
    drop_cols = {"valid_frac_pre", "valid_frac_post", "cloud_frac_pre", "cloud_frac_post",
                 "blk_frac_pre", "blk_frac_post", "is_gray_pre", "is_gray_post"}
    keep_idx = [i for i, c in enumerate(p_names) if c not in drop_cols]
    phys_block = zscore_l2(p_mat[:, keep_idx])

    ch_wb = zscore_l2(vitb["chamfer_stats"])
    ch_gr = zscore_l2(vits["chamfer_stats"])

    # F_scene: full multi-backbone fusion
    scene_parts = [
        l2norm(vitb["pmean_post"]) * 1.0,
        l2norm(vitb["cls_post"]) * 0.7,
        l2norm(vitb["pmean_pre"]) * 0.6,
        l2norm(cnxb["post"]) * 1.0,
        l2norm(cnxb["pre"]) * 0.6,
        l2norm(vits["pmean_post"]) * 0.9,
        l2norm(vits["pmean_pre"]) * 0.5,
        l2norm(vitb["changed_emb"]) * 0.8,
        l2norm(vits["changed_emb"]) * 0.6,
        ch_wb * 0.35,
        ch_gr * 0.35,
        phys_block * 0.35,
    ]
    F_scene = l2norm(np.hstack(scene_parts).astype(np.float32))

    # F_gray_pure: 100% color-free control (no vitb_wb blocks)
    gray_drop = drop_cols | {c for c in p_names if any(k in c for k in ("exg_", "sat_", "low_sat"))}
    keep_gray = [i for i, c in enumerate(p_names) if c not in gray_drop]
    phys_gray = zscore_l2(p_mat[:, keep_gray])
    gray_parts = [
        l2norm(cnxb["post"]) * 1.0,
        l2norm(cnxb["pre"]) * 0.6,
        l2norm(vits["pmean_post"]) * 1.0,
        l2norm(vits["pmean_pre"]) * 0.55,
        l2norm(vits["changed_emb"]) * 0.7,
        ch_gr * 0.45,
        phys_gray * 0.35,
    ]
    F_gray = l2norm(np.hstack(gray_parts).astype(np.float32))

    audit = phys["audit_flags"]  # (N, 4): rot_pre, rot_post, is_gray_pre, is_gray_post
    both_gray = (audit[:, 2] > 0.5) & (audit[:, 3] > 0.5)
    any_rot = (audit[:, 0] > 0.5) | (audit[:, 1] > 0.5)

    return sample_ids, F_scene, F_gray, phys, vitb, cnxb, vits, both_gray, any_rot


def run_louvain_sweep(X: np.ndarray, both_gray: np.ndarray, any_rot: np.ndarray):
    nn = NearestNeighbors(n_neighbors=16, metric="cosine").fit(X)
    dists, idxs = nn.kneighbors(X)
    edges, weights = [], []
    for i in range(len(X)):
        for j, d in zip(idxs[i, 1:], dists[i, 1:]):
            if i < j:
                edges.append((i, int(j)))
                weights.append(max(0.0, 1.0 - float(d)))
    g = ig.Graph(n=len(X), edges=edges, directed=False)

    print("\n=== LOUVAIN COMMUNITY SWEEP (kNN k=15) ===", flush=True)
    best_res = None
    best_score = -1.0
    results = {}
    for res in [0.8, 1.0, 1.2, 1.5, 1.8]:
        random.seed(SEED)
        part = g.community_multilevel(weights=weights, resolution=res)
        lab = np.array(part.membership, dtype=int)
        comms = sorted([c for c in range(len(part)) if (lab == c).sum() >= 20], key=lambda c: -(lab == c).sum())
        max_gray = max([float(both_gray[lab == c].mean()) for c in comms]) if comms else 1.0
        max_rot = max([float(any_rot[lab == c].mean()) for c in comms]) if comms else 1.0
        sil = float(silhouette_score(X, lab, metric="cosine")) if len(set(lab)) > 1 else 0.0
        sizes = [int((lab == c).sum()) for c in comms]
        print(f"  res={res:.1f} | n_major(>20)={len(comms):2d} (total={len(part):2d}) | mod={part.modularity:.3f} | sil={sil:.3f} | max_gray={max_gray*100:4.1f}% | max_rot={max_rot*100:4.1f}% | sizes={sizes[:10]}", flush=True)
        results[res] = (lab, comms, part.modularity, sil, max_gray, max_rot)
        # Prefer 14..18 major communities
        dist_16 = abs(len(comms) - 16)
        sc = part.modularity - 0.05 * dist_16
        if sc > best_score:
            best_score = sc
            best_res = res
    return best_res, results


def run_kmeans_ward_eval(X: np.ndarray, tag: str, both_gray: np.ndarray, any_rot: np.ndarray):
    print(f"\n=== KMEANS & WARD STABILITY AUDIT ({tag}) ===", flush=True)
    saved = {}
    for K in [12, 16, 20]:
        km_runs = [KMeans(K, n_init=30, random_state=SEED + s).fit_predict(X) for s in range(5)]
        aris = [adjusted_rand_score(km_runs[i], km_runs[j]) for i in range(5) for j in range(i + 1, 5)]
        lab_km = km_runs[0]
        sil_km = float(silhouette_score(X, lab_km, metric="cosine"))
        max_g_km = max(float(both_gray[lab_km == c].mean()) for c in range(K))
        max_r_km = max(float(any_rot[lab_km == c].mean()) for c in range(K))

        lab_wd = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(X)
        sil_wd = float(silhouette_score(X, lab_wd, metric="cosine"))
        max_g_wd = max(float(both_gray[lab_wd == c].mean()) for c in range(K))
        max_r_wd = max(float(any_rot[lab_wd == c].mean()) for c in range(K))

        print(f"  K={K:2d} KMeans: sil={sil_km:.3f} | ARI_5seed={np.mean(aris):.3f} | max_gray={max_g_km*100:4.1f}% | max_rot={max_r_km*100:4.1f}%", flush=True)
        print(f"  K={K:2d} Ward  : sil={sil_wd:.3f} | max_gray={max_g_wd*100:4.1f}% | max_rot={max_r_wd*100:4.1f}%", flush=True)
        saved[K] = (lab_km, sil_km, lab_wd, sil_wd, float(np.mean(aris)))
    return saved


def build_cluster_profile_df(labels: np.ndarray, cluster_ids: list[int], X: np.ndarray,
                             phys, vitb, cnxb, vits, both_gray: np.ndarray, any_rot: np.ndarray) -> pd.DataFrame:
    p_names = list(phys["feature_names"])
    pf = phys["features"]
    idx_map = {n: i for i, n in enumerate(p_names)}

    cos_vitb = (vitb["pmean_pre"] * vitb["pmean_post"]).sum(axis=1)
    cos_vits = (vits["pmean_pre"] * vits["pmean_post"]).sum(axis=1)
    cos_cnxb = (cnxb["pre"] * cnxb["post"]).sum(axis=1)
    cos_pair = (cos_vitb + cos_vits + cos_cnxb) / 3.0

    ch_g = vits["chamfer_stats"]  # mean, std, p50, p75, p90, max, frac35, frac45, spread
    ch_w = vitb["chamfer_stats"]

    rows = []
    for c in cluster_ids:
        m = (labels == c)
        rows.append({
            "cluster": int(c),
            "n": int(m.sum()),
            "pct_both_gray": round(float(both_gray[m].mean()) * 100, 1),
            "pct_rot_corner": round(float(any_rot[m].mean()) * 100, 1),
            "mean_cos_pre_post": round(float(cos_pair[m].mean()), 3),
            "chamfer_gray_mean": round(float(ch_g[m, 0].mean()), 3),
            "chamfer_gray_p90": round(float(ch_g[m, 4].mean()), 3),
            "chamfer_gray_frac45": round(float(ch_g[m, 7].mean()) * 100, 1),
            "chamfer_wb_mean": round(float(ch_w[m, 0].mean()), 3),
            "chamfer_spread": round(float(ch_g[m, 8].mean()), 3),
            "lum_delta": round(float(pf[m, idx_map["lum_mean_delta"]].mean()), 2),
            "dark_post": round(float(pf[m, idx_map["dark_frac_post"]].mean()) * 100, 1),
            "edge_delta": round(float(pf[m, idx_map["canny_dens_delta"]].mean()), 4),
            "laplacian_delta": round(float(pf[m, idx_map["lap_var_delta"]].mean()), 1),
            "sat_post": round(float(pf[m, idx_map["sat_mean_post"]].mean()), 3),
            "cloud_post": round(float(pf[m, idx_map["cloud_frac_post"]].mean()) * 100, 1),
        })
    return pd.DataFrame(rows)


def render_montages(sample_ids: list[str], labels: np.ndarray, cluster_ids: list[int],
                    X: np.ndarray, chg_score: np.ndarray, prefix: str):
    """Render 8 clusters per montage: 3 nearest-to-centroid + 3 highest-Chamfer-change members."""
    T = 92
    # Split into chunks of 8 clusters
    chunks = [cluster_ids[i:i + 8] for i in range(0, len(cluster_ids), 8)]
    saved_paths = []
    for part_idx, c_list in enumerate(chunks):
        can = Image.new("RGB", (115 + 6 * (2 * T + 8), len(c_list) * (T + 14)), "white")
        dr = ImageDraw.Draw(can)
        for r, c in enumerate(c_list):
            idx = np.where(labels == c)[0]
            subX = X[idx]
            cent = l2norm(subX.mean(axis=0, keepdims=True))[0]
            cent_order = idx[np.argsort(-(subX @ cent))]
            top_cent = [int(x) for x in cent_order[:3]]

            chg_order = idx[np.argsort(-chg_score[idx])]
            top_chg = [int(x) for x in chg_order if int(x) not in top_cent][:3]
            picks = top_cent + top_chg

            dr.text((4, r * (T + 14) + 22), f"C{c}\nn={len(idx)}\n3cent+3chg", fill="black")
            for j, ii in enumerate(picks):
                sid = sample_ids[ii]
                x0 = 115 + j * (2 * T + 8)
                y0 = r * (T + 14) + 2
                for k, kind in enumerate(("pre", "post")):
                    with Image.open(FEN_DIR / sid / f"{kind}.jpg") as im:
                        im_r = im.convert("RGB").resize((T, T), Image.Resampling.BILINEAR)
                        can.paste(im_r, (x0 + k * (T + 2), y0))
                tag = "C" if j < 3 else "D"
                dr.text((x0 + 2, y0 + T + 1), f"{tag}:{sid[:8]}", fill="black")

        out_p = CACHE_DIR / f"{prefix}_part{part_idx}.jpg"
        can.save(out_p, quality=88)
        saved_paths.append(out_p)
    return saved_paths


def main():
    sample_ids, F_scene, F_gray, phys, vitb, cnxb, vits, both_gray, any_rot = build_feature_spaces()
    print(f"F_scene shape={F_scene.shape} | F_gray_pure shape={F_gray.shape}", flush=True)

    best_res, louv_res = run_louvain_sweep(F_scene, both_gray, any_rot)
    km_scene = run_kmeans_ward_eval(F_scene, "F_scene", both_gray, any_rot)
    _ = run_kmeans_ward_eval(F_gray, "F_gray_pure (control)", both_gray, any_rot)

    # K=16 selection: pick KMeans or Ward based on higher silhouette (and check leakage)
    lab_km16, sil_km16, lab_wd16, sil_wd16, ari16 = km_scene[16]
    if sil_km16 >= sil_wd16:
        k16_labels = lab_km16
        k16_method = "KMeans"
    else:
        k16_labels = lab_wd16
        k16_method = "Ward"
    k16_ids = sorted(range(16), key=lambda c: -(k16_labels == c).sum())

    # Best Louvain partition around 14-18 major clusters
    louv_lab, louv_major, louv_mod, louv_sil, _, _ = louv_res[best_res]

    # Save cluster profiles & audit npz
    prof_k16 = build_cluster_profile_df(k16_labels, k16_ids, F_scene, phys, vitb, cnxb, vits, both_gray, any_rot)
    prof_k16_path = CACHE_DIR / "cluster_profiles_k16.csv"
    prof_k16.to_csv(prof_k16_path, index=False)

    prof_louv = build_cluster_profile_df(louv_lab, louv_major, F_scene, phys, vitb, cnxb, vits, both_gray, any_rot)
    prof_louv_path = CACHE_DIR / "cluster_profiles_louvain.csv"
    prof_louv.to_csv(prof_louv_path, index=False)

    np.savez_compressed(
        CACHE_DIR / "clusters_v3_audit.npz",
        ids=np.array(sample_ids),
        F_scene=F_scene,
        k16_labels=k16_labels,
        k16_km=lab_km16,
        k16_wd=lab_wd16,
        louv_labels=louv_lab,
        louv_res=np.array([best_res]),
    )

    chg_score = vits["chamfer_stats"][:, 4]  # chamfer_gray_p90
    k16_montages = render_montages(sample_ids, k16_labels, k16_ids, F_scene, chg_score, "montage_v3_k16")
    louv_montages = render_montages(sample_ids, louv_lab, louv_major, F_scene, chg_score, "montage_v3_louvain")

    print(f"\n=== SELECTED K=16 ({k16_method}, ARI={ari16:.3f}) PROFILES ===", flush=True)
    print(prof_k16.to_string(index=False), flush=True)

    print(f"\n=== SELECTED LOUVAIN (res={best_res}, major={len(louv_major)}, mod={louv_mod:.3f}, sil={louv_sil:.3f}) PROFILES ===", flush=True)
    print(prof_louv.to_string(index=False), flush=True)

    print("\n=== SAVED AUDIT ARTIFACTS ===", flush=True)
    print("  K16 CSV     :", prof_k16_path, flush=True)
    print("  Louvain CSV :", prof_louv_path, flush=True)
    for p in k16_montages + louv_montages:
        print("  Montage     :", p, flush=True)

    max_gray_k16 = float(prof_k16["pct_both_gray"].max())
    max_rot_k16 = float(prof_k16["pct_rot_corner"].max())
    print(f"\nLeakage check (K16): max_both_gray={max_gray_k16:.1f}% (limit <=35%), max_rot_corner={max_rot_k16:.1f}% (limit <=65%)", flush=True)


if __name__ == "__main__":
    main()
