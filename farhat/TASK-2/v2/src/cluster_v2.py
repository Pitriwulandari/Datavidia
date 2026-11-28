"""Task-2 v2 Part 2: clustering ensemble & pseudo-label.

Menggabungkan:
- DINOv2 ViT-S/14 (features_v3.npz -> 384d PCA 192d)
- DINOv3 ViT-S/16 (emb_dinov3_s_224.npz -> 384d PCA 192d)
- Fisik de-augmented (phys_v2.npz -> 74d StandardScaler)
Total ~458d -> PCA 128d -> kNN graph k=15 cosine -> Louvain res=2.6 (target ~28 komunitas)
Output: cache/features/clusters_v2.npz + models/cluster_mapping_v2.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src import config as C
from src.utils import get_logger, Timer

log = get_logger("cluster_v2")

CLUSTERS_OUT = C.FEATURE_DIR / "clusters_v2.npz"
MAPPING_OUT = C.MODELS / "cluster_mapping_v2.json"

CLASSES = ["Earthquake", "Fire", "Hurricane", "Tsunami", "Volcano", "Explosion"]


def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / n


def build_knn_graph(emb: np.ndarray, k: int = 15) -> csr_matrix:
    n = emb.shape[0]
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1.0)
    rows, cols, vals = [], [], []
    for i in range(n):
        topk = np.argpartition(sim[i], -k)[-k:]
        rows.extend([i] * k)
        cols.extend(topk)
        vals.extend(sim[i, topk])
    W = csr_matrix((vals, (rows, cols)), shape=(n, n))
    W = (W + W.T) * 0.5
    return W


def louvain_communities(W: csr_matrix, resolution: float = 2.6, seed: int = 42):
    try:
        import igraph as ig
    except Exception as e:
        raise RuntimeError("igraph required: uv pip install igraph") from e
    n = W.shape[0]
    # igraph needs edge list with weights
    coo = W.tocoo()
    sources, targets, weights = coo.row.tolist(), coo.col.tolist(), coo.data.tolist()
    g = ig.Graph(directed=False)
    g.add_vertices(n)
    g.add_edges(list(zip(sources, targets)))
    g.es["weight"] = weights
    part = g.community_multilevel(weights=g.es["weight"], resolution=resolution)
    labels = np.zeros(n, dtype=int)
    for i, com in enumerate(part):
        labels[list(com)] = i
    return labels


def extract_embeddings():
    # 1. DINOv2-S (dari features_v3.npz)
    f1 = np.load(C.FEATURE_DIR / "features_v3.npz")
    e1 = f1["features"]  # sudah concat pre+post + PCA + scaler
    log.info("dinov2_s raw: %s", e1.shape)
    pca1 = PCA(192, random_state=42)
    e1 = pca1.fit_transform(e1)

    # 2. DINOv3-S
    f2 = np.load(C.FEATURE_DIR / "emb_dinov3_s_224.npz")
    e2 = np.hstack([f2["pre"], f2["post"]])  # (3180, 768) -> concat pre+post
    log.info("dinov3_s raw: %s", e2.shape)
    pca2 = PCA(192, random_state=42)
    e2 = pca2.fit_transform(e2)

    # 3. Fisik v2
    f3 = np.load(C.FEATURE_DIR / "phys_v2.npz")
    e3 = f3["features"]
    scaler = StandardScaler()
    e3 = scaler.fit_transform(e3)
    log.info("phys_v2 scaled: %s", e3.shape)

    # Normalisasi L2 tiap blok
    e1 = l2norm(e1)
    e2 = l2norm(e2)
    e3 = l2norm(e3)

    # Gabung & PCA final
    E = np.hstack([e1, e2, e3])
    log.info("concat: %s", E.shape)
    pca = PCA(128, random_state=42)
    E = pca.fit_transform(E)
    E = l2norm(E)
    return E


def audit_cluster(pid: str, emb_row: np.ndarray) -> dict:
    return {"pid": pid, "dummy": True}


def main():
    with Timer("cluster_v2", log):
        E = extract_embeddings()
        W = build_knn_graph(E, k=15)
        labels = louvain_communities(W, resolution=2.6)
        n_com = labels.max() + 1
        log.info("komunitas: %d", n_com)

        # Distribusi per komunitas
        ids = pd.read_csv(C.SAMPLE_SUBMISSION)["Id"].tolist()
        dist = pd.DataFrame({"Id": ids, "cluster": labels})
        vc = dist["cluster"].value_counts().sort_index()
        for c, n in vc.items():
            log.info("C%d: %4d sampel", c, n)

        np.savez_compressed(CLUSTERS_OUT, labels=labels, emb=E)
        log.info("tersimpan %s", CLUSTERS_OUT)

        # Mapping awal kosong - perlu audit visual manual
        mapping = {
            "n_clusters": int(n_com),
            "cluster_to_class": {},
            "method": "Louvain res=2.6 on kNN(15) cosine, emb=[dinov2_s(192)+dinov3_s(192)+phys_v2(74)]->PCA128",
            "audit_notes": "Perlu audit visual montase per komunitas (C0..C{})".format(n_com - 1),
            "disclaimer": "Pseudo-label dari clustering unsupervised tanpa ground truth; mapping awal otomatis belum valid."
        }
        with open(MAPPING_OUT, "w") as f:
            json.dump(mapping, f, indent=2)
        log.info("mapping awal %s", MAPPING_OUT)


if __name__ == "__main__":
    main()