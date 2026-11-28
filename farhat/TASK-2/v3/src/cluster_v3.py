"""Task-2 v3: Clustering on Patch-Chamfer Top-Change Delta Embeddings.

Strategy:
1. Use Hurricane detector (from v4_deep) as anchor - 311 samples with precision 1.0
2. Cluster remaining 2869 samples on topchg_delta embeddings
3. Assign cluster-to-class based on physical features (change_stats.csv)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache" / "features"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def l2norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def load_patch_chamfer_features():
    """Load patch-chamfer v3 features."""
    pc_path = CACHE_DIR / "patch_chamfer_v3_dinov2_vits14_336.npz"
    if not pc_path.exists():
        raise FileNotFoundError(f"Patch-chamfer features not found: {pc_path}")
    
    d = np.load(pc_path)
    print(f"[v3] Loaded patch-chamfer: {list(d.keys())}")
    print(f"  chamfer_stats: {d['chamfer_stats'].shape}")
    print(f"  topchg_delta: {d['topchg_delta'].shape}")
    return d


def load_hurricane_labels():
    """Load Hurricane pseudo-labels (anchor v3).

    Prioritas:
    1. `models/hurricane_mask_v3.npy` — mask beku (frozen) yang sudah diturunkan
       dari klaster v3; dipakai agar pipeline reproducible tanpa artefak v3.
    2. `archive/v3/models/cluster_mapping_v3.json` + `cache/features/clusters_v3.npz`
       — jalur derivasi asli.
    """
    frozen_path = MODELS_DIR / "hurricane_mask_v3.npy"
    clusters_path = CACHE_DIR / "clusters_v3.npz"

    if frozen_path.exists():
        is_hurricane = np.load(frozen_path).astype(bool)
        print(f"[v3] Hurricane anchor dari frozen mask: {int(is_hurricane.sum())} sampel")
        v3_clusters = np.load(clusters_path)["labels"] if clusters_path.exists() else None
        return is_hurricane, v3_clusters

    mapping_path = ROOT / "archive" / "v3" / "models" / "cluster_mapping_v3.json"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"v3 mapping tidak ditemukan: {mapping_path}\n"
            f"Juga tidak ada frozen mask: {frozen_path}"
        )

    with open(mapping_path) as f:
        mapping = json.load(f)
    c2c = mapping["cluster_to_class"]

    if not clusters_path.exists():
        raise FileNotFoundError(f"v3 clusters tidak ditemukan: {clusters_path}")
    clusters = np.load(clusters_path)["labels"]

    hurricane_ids = [int(k) for k, v in c2c.items() if v == "Hurricane"]
    is_hurricane = np.isin(clusters, hurricane_ids)

    print(f"[v3] Hurricane clusters dari v3: {hurricane_ids}")
    print(f"[v3] Hurricane samples: {is_hurricane.sum()} / {len(clusters)}")
    return is_hurricane, clusters


def cluster_residual_on_topchg(
    topchg_delta: np.ndarray,
    hurricane_mask: np.ndarray,
    n_pca: int = 128,
    k_nn: int = 15,
    resolution: float = 2.6,
) -> np.ndarray:
    """Cluster non-Hurricane samples on top-change delta embeddings."""
    
    # Filter out Hurricane samples
    residual_idx = np.where(~hurricane_mask)[0]
    X_residual = topchg_delta[residual_idx]
    
    print(f"[v3] Clustering {len(residual_idx)} residual samples on topchg_delta...")
    print(f"  Input dim: {X_residual.shape[1]}")
    
    # L2 normalize
    X_norm = l2norm(X_residual)
    
    # PCA
    if X_norm.shape[1] > n_pca:
        pca = PCA(n_components=n_pca, random_state=SEED)
        X_pca = pca.fit_transform(X_norm)
        print(f"  PCA: {X_norm.shape[1]} -> {n_pca}, explained_var={pca.explained_variance_ratio_.sum():.3f}")
    else:
        X_pca = X_norm
    
    # kNN graph
    nn = NearestNeighbors(n_neighbors=k_nn, metric="cosine").fit(X_pca)
    dists, idxs = nn.kneighbors(X_pca)
    
    # Build igraph
    edges, weights = [], []
    for i in range(len(X_pca)):
        for j, d in zip(idxs[i, 1:], dists[i, 1:]):
            if i < j:
                edges.append((i, int(j)))
                weights.append(float(1.0 - d))
    
    g = ig.Graph(n=len(X_pca), edges=edges, directed=False)
    
    # Louvain
    part = g.community_multilevel(weights=weights, resolution=resolution)
    labels_residual = np.array(part.membership, dtype=int)
    
    print(f"[v3] Louvain: {len(part)} communities, modularity={part.modularity:.3f}")
    
    # Size distribution
    unique, counts = np.unique(labels_residual, return_counts=True)
    sizes = sorted(counts, reverse=True)
    print(f"  Cluster sizes (top 10): {sizes[:10]}")
    
    # Map back to full index
    full_labels = np.full(len(hurricane_mask), -1, dtype=int)  # -1 = Hurricane
    full_labels[residual_idx] = labels_residual
    
    return full_labels, residual_idx, labels_residual, part.modularity


def compute_cluster_profiles(
    sample_ids: list[str],
    labels: np.ndarray,
    chamfer_stats: np.ndarray,
    topchg_delta: np.ndarray,
    change_stats_path: Path,
) -> pd.DataFrame:
    """Compute physical profiles per cluster for mapping."""
    
    # Load change_stats.csv
    cs = pd.read_csv(change_stats_path)
    
    # Map sample_id to index
    id2i = {s: i for i, s in enumerate(sample_ids)}
    
    # Compute per-cluster stats
    unique_labels = sorted([l for l in np.unique(labels) if l >= 0])
    rows = []
    
    for c in unique_labels:
        mask = labels == c
        n = mask.sum()
        if n < 5:
            continue
        
        # Chamfer stats
        chg_mean = chamfer_stats[mask, 0].mean()  # ssim change mean
        chg_p90 = chamfer_stats[mask, 4].mean()   # p90 change
        spread = chamfer_stats[mask, 9].mean()    # epicenter spread
        rad_corr = chamfer_stats[mask, 10].mean()  # radial decay
        
        # Change stats from CSV
        cids = [sample_ids[i] for i in np.where(mask)[0]]
        cs_mask = cs["Id"].isin(cids)
        
        if cs_mask.sum() > 0:
            cs_sub = cs[cs_mask]
            val_delta = cs_sub["val_delta"].mean() if "val_delta" in cs_sub else 0.0
            frac_changed = cs_sub["frac_changed"].mean() if "frac_changed" in cs_sub else 0.0
            edge_delta = cs_sub["edge_delta"].mean() if "edge_delta" in cs_sub else 0.0
        else:
            val_delta = 0.0
            frac_changed = 0.0
            edge_delta = 0.0
        
        rows.append({
            "cluster": c,
            "n": n,
            "chg_mean": round(chg_mean, 3),
            "chg_p90": round(chg_p90, 3),
            "spread": round(spread, 3),
            "rad_corr": round(rad_corr, 3),
            "val_delta": round(val_delta, 3),
            "frac_changed": round(frac_changed, 3),
            "edge_delta": round(edge_delta, 4),
        })
    
    return pd.DataFrame(rows)


def map_clusters_by_v3_prototype(
    sample_ids: list[str],
    labels: np.ndarray,
    change_stats_path: Path,
) -> tuple[dict, pd.DataFrame]:
    """Petakan klaster v3 -> kelas lewat prototipe fisik klaster v3.

    v3 sudah punya mapping klaster->kelas berbasis bukti (lihat
    archive/v3/models/cluster_mapping_v3.json). Tiap klaster v3 dicocokkan
    ke klaster v3 terdekat dalam ruang statistik perubahan yang sudah
    discale, lalu memakai kelas v3 tersebut. Bukti distance + klaster
    terdekat disimpan supaya mapping bisa diaudit.
    """
    cs = pd.read_csv(change_stats_path)
    id2row = {sid: i for i, sid in enumerate(cs["Id"].tolist())}

    delta_cols = [
        "diff_abs", "diff_abs_r", "diff_abs_g", "diff_abs_b",
        "frac_changed", "corr_gray",
        "diff_signed_r", "diff_signed_g", "diff_signed_b",
        "val_delta", "sat_delta", "green_delta", "red_delta",
        "blue_delta", "gray_delta", "dark_delta", "edge_delta",
    ]
    delta_cols = [c for c in delta_cols if c in cs.columns]
    X = np.full((len(sample_ids), len(delta_cols)), np.nan, dtype=np.float64)
    for i, sid in enumerate(sample_ids):
        r = id2row.get(sid)
        if r is not None:
            X[i] = cs.iloc[r][delta_cols].to_numpy(dtype=np.float64)
    if np.isnan(X).any():
        med = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            m = np.isnan(X[:, j])
            X[m, j] = med[j]

    scale = X.std(axis=0)
    scale[scale < 1e-8] = 1.0
    Xs = X / scale

    v3_labels = np.load(CACHE_DIR / "clusters_v3.npz")["labels"]
    with open(ROOT / "archive" / "v3" / "models" / "cluster_mapping_v3.json") as f:
        v3_map = json.load(f)["cluster_to_class"]

    def profile(lab: np.ndarray, ids: list[int]) -> np.ndarray:
        """Return profile matrix ordered by `ids`."""
        return np.stack([Xs[lab == c].mean(axis=0) for c in ids])

    ids5 = sorted(int(c) for c in np.unique(labels) if c >= 0)
    ids3 = sorted(int(c) for c in np.unique(v3_labels))

    P5 = profile(labels, ids5)
    P3 = profile(v3_labels, ids3)

    rows = []
    mapping = {}
    for i, c in enumerate(ids5):
        dist = np.linalg.norm(P3 - P5[i], axis=1)
        j = int(np.argmin(dist))
        c3 = ids3[j]
        cls = v3_map[str(c3)]
        mapping[f"C{c:02d}"] = cls
        rows.append({
            "cluster_v3": c,
            "n5": int((labels == c).sum()),
            "nearest_v3": c3,
            "nearest_class": cls,
            "distance": round(float(dist[j]), 3),
            "mapping": cls,
        })

    print("[v3] Mapping klaster v3 -> kelas via prototipe v3:")
    print(pd.DataFrame(rows).to_string(index=False))
    return mapping, pd.DataFrame(rows)


def map_clusters_by_v3_kmeans(
    sample_ids: list[str],
    labels: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    """Petakan klaster v3 -> kelas dengan re-fit KMeans(7) pada fitur CLS DINOv2.

    v3 menggunakan KMeans(n_clusters=7, seed=42) pada emb_dinov2_s_196
    (pre+post concat = 392 dims). Kita re-fit KMeans(7, seed=42) pada data
    yang sama, lalu cocokkan centroid baru ke centroid v3 (cluster_v3.npz)
    untuk menentukan permutasi label. Akhirnya gunakan mapping v3
    (cluster_to_class) untuk setiap klaster v3 dengan majority vote
    di dalam setiap klaster v3.
    """
    # Load DINOv2 CLS features
    emb_path = CACHE_DIR / "emb_dinov2_s_196.npz"
    d = np.load(emb_path)
    X_cls = np.concatenate([d["pre"], d["post"]], axis=1)  # (3180, 392)
    
    # Load v3 clusters (KMeans(7) on same features)
    v3_labels = np.load(CACHE_DIR / "clusters_v3.npz")["labels"]
    
    # Re-fit KMeans(7) on current data (should match v3 up to permutation)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=7, random_state=SEED, n_init=20)
    km_labels = km.fit_predict(X_cls)
    
    # Match new clusters to v3 clusters by centroid distance
    new_centroids = km.cluster_centers_
    v3_centroids = np.array([X_cls[v3_labels == i].mean(axis=0) for i in range(7)])
    
    # Hungarian assignment (greedy since small)
    used = set()
    new_to_v3 = {}
    for i in range(7):
        dists = np.linalg.norm(new_centroids[i] - v3_centroids, axis=1)
        for j in np.argsort(dists):
            if j not in used:
                new_to_v3[i] = j
                used.add(j)
                break
    
    # v3 mapping
    with open(ROOT / "archive" / "v3" / "models" / "cluster_mapping_v3.json") as f:
        v3_map = json.load(f)["cluster_to_class"]
    
    # Map each v3 cluster by majority v3 class within it
    rows = []
    mapping = {}
    ids5 = sorted(int(c) for c in np.unique(labels) if c >= 0)
    for c in ids5:
        mask = (labels == c)
        # v3 class for each sample in this v3 cluster
        v3_class = np.array([v3_map[str(new_to_v3[km_labels[i]])] for i in np.where(mask)[0]])
        # Majority vote
        cls = pd.Series(v3_class).mode()[0]
        mapping[f"C{c:02d}"] = cls
        # Distribution within cluster
        dist = pd.Series(v3_class).value_counts().to_dict()
        rows.append({
            "cluster_v3": c,
            "n5": int(mask.sum()),
            "majority_class": cls,
            "v3_distribution": str(dist),
        })
    
    print("[v3] Mapping klaster v3 -> kelas via v3 KMeans:")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return mapping, df


def heuristic_mapping(profiles: pd.DataFrame) -> dict:
    """Rule-based cluster-to-class mapping."""
    mapping = {}
    
    for _, row in profiles.iterrows():
        c = int(row["cluster"])
        n = int(row["n"])
        chg = row["chg_mean"]
        spread = row["spread"]
        rad_corr = row["rad_corr"]
        val_d = row["val_delta"]
        frac = row["frac_changed"]
        edge_d = row["edge_delta"]
        
        # Heuristic rules based on Ghani's physical analysis
        # Hurricane: high frac_changed (widespread flooding), negative val_delta (darker post)
        if frac > 0.15 and val_d < -0.05:
            mapping[f"C{c:02d}"] = "Hurricane"
        # Fire: high edge_delta (burn scars), positive val_delta (brighter ash)
        elif edge_d > 0.02 and chg > 0.25:
            mapping[f"C{c:02d}"] = "Fire"
        # Earthquake: localized collapse, high spread, moderate chg
        elif spread > 0.3 and 0.15 < chg < 0.35:
            mapping[f"C{c:02d}"] = "Earthquake"
        # Volcano: high rad_corr (radial lava flow), high chg
        elif rad_corr > 0.3 and chg > 0.3:
            mapping[f"C{c:02d}"] = "Volcano"
        # Explosion: very high chg_p90, localized crater
        elif chg > 0.4 and spread < 0.25:
            mapping[f"C{c:02d}"] = "Explosion"
        # Tsunami: near coast (need geospatial), high water_inund
        elif frac > 0.2 and val_d < -0.1:
            mapping[f"C{c:02d}"] = "Tsunami"
        else:
            mapping[f"C{c:02d}"] = "Unknown"
    
    return mapping


def main():
    # Load sample IDs
    sample_sub_path = ROOT / "data" / "raw" / "sample_submission.csv"
    sample_ids = pd.read_csv(sample_sub_path)["Id"].tolist()
    assert len(sample_ids) == 3180
    
    # Load patch-chamfer features
    pc = load_patch_chamfer_features()
    chamfer_stats = pc["chamfer_stats"]
    topchg_delta = pc["topchg_delta"]
    
    # Load Hurricane labels from v4_deep
    hurricane_mask, v4_clusters = load_hurricane_labels()
    
    # Cluster residual samples
    full_labels, residual_idx, labels_residual, mod = cluster_residual_on_topchg(
        topchg_delta, hurricane_mask,
        n_pca=128, k_nn=15, resolution=2.6
    )
    
    # Save cluster labels
    out_path = CACHE_DIR / "clusters_v3.npz"
    np.savez_compressed(
        out_path,
        labels=full_labels,
        hurricane_mask=hurricane_mask,
        residual_idx=residual_idx,
    )
    print(f"[v3] Saved {out_path}")
    
    # Compute cluster profiles
    change_stats_path = ROOT / "data" / "processed" / "change_stats.csv"
    profiles = compute_cluster_profiles(
        sample_ids, full_labels, chamfer_stats, topchg_delta, change_stats_path
    )

    # Save profiles
    profiles_path = CACHE_DIR / "cluster_profiles_v3.csv"
    profiles.to_csv(profiles_path, index=False)
    print(f"[v3] Saved {profiles_path}")

    # Map clusters -> class.
    # Jika artefak v3 tersedia, derivasi ulang (metode KMeans centroid matching).
    # Jika tidak, pakai frozen mapping yang sudah divalidasi (reproducible).
    v3_mapping_path = ROOT / "archive" / "v3" / "models" / "cluster_mapping_v3.json"
    mapping_path = MODELS_DIR / "cluster_mapping_v3.json"

    if v3_mapping_path.exists():
        mapping, map_evidence = map_clusters_by_v3_kmeans(sample_ids, full_labels)
        mapping["hurricane_anchor"] = "Hurricane"
        with open(mapping_path, "w") as f:
            json.dump(mapping, f, indent=2)
        print(f"[v3] Saved {mapping_path}")

        evidence_path = MODELS_DIR / "cluster_mapping_v3_evidence.csv"
        map_evidence.to_csv(evidence_path, index=False)
        print(f"[v3] Saved {evidence_path}")

        proto_mapping, proto_evidence = map_clusters_by_v3_prototype(
            sample_ids, full_labels, change_stats_path
        )
        proto_mapping["hurricane_anchor"] = "Hurricane"
        proto_path = MODELS_DIR / "cluster_mapping_v3_prototype.json"
        with open(proto_path, "w") as f:
            json.dump(proto_mapping, f, indent=2)
        proto_evidence.to_csv(
            MODELS_DIR / "cluster_mapping_v3_prototype_evidence.csv", index=False
        )
        print(f"[v3] Saved {proto_path}")
    elif mapping_path.exists():
        mapping = json.loads(mapping_path.read_text())
        print(f"[v3] Artefak v3 tidak tersedia -> memakai frozen mapping: {mapping_path}")
    else:
        raise FileNotFoundError(
            f"Tidak bisa menentukan cluster mapping: {v3_mapping_path} dan {mapping_path} "
            "keduanya tidak ada"
        )

    heuristic = heuristic_mapping(profiles)
    heuristic["hurricane_anchor"] = "Hurricane"
    h_path = MODELS_DIR / "cluster_mapping_v3_heuristic.json"
    with open(h_path, "w") as f:
        json.dump(heuristic, f, indent=2)
    print(f"[v3] Saved {h_path}")

    print("\n[v3] Cluster profiles:")
    print(profiles.to_string(index=False))

    # Count per class (using primary mapping)
    class_counts = {}
    for c_name in mapping.values():
        class_counts[c_name] = class_counts.get(c_name, 0) + 1
    print(f"\n[v3] Class distribution (clusters): {class_counts}")

    # Sample distribution will be computed in train_v3.py


if __name__ == "__main__":
    main()
