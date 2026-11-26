"""Tahap 3: perakitan matriks fitur + clustering tanpa label.

Temuan penting dari diagnostik pada data ini: PCA-whitening pada embedding
DINOv2 justru merusak struktur klaster (stabilitas ARI ~0.0), sedangkan embedding
mentah ternormalisasi L2 memberikan ARI ~0.73 antar-seed. Karena itu pipeline
memakai representasi mentah, dengan blok opsional:

  post    : embedding DINOv2 citra sesudah kejadian (paling informatif)
  pre     : embedding sebelum kejadian
  delta   : post - pre pada ruang embedding
  stats   : 31 statistik perubahan piksel yang distandardisasi

Clustering memakai k-means (n_init tinggi) pada kombinasi blok tersebut. Tidak
ada label/ground truth yang dipakai; k=6 berasal dari aturan kompetisi.
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src import config as C
from src.utils import get_logger, l2norm, set_seed

log = get_logger("features")

BLOCK_WEIGHTS = {"post": 1.0, "pre": 0.6, "delta": 0.4, "stats": C.HANDCRAFTED_WEIGHT}
DEFAULT_BLOCKS = ["post", "stats"]


def load_embeddings(tag: str) -> dict:
    p = C.FEATURE_DIR / f"emb_{tag}.npz"
    if not p.exists():
        cands = sorted(C.FEATURE_DIR.glob(f"emb_{tag}_*.npz"))
        if not cands:
            raise FileNotFoundError(f"embedding tidak ditemukan: {p}")
        p = cands[-1]
    d = np.load(p, allow_pickle=True)
    return {"ids": [str(x) for x in d["ids"]], "pre": d["pre"], "post": d["post"],
            "model": str(d["model"]), "size": int(d["size"])}


def sinkhorn(logp: np.ndarray, r: float = 1.0, c: float = 1.0, iters: int = 60) -> np.ndarray:
    """IPF/Sinkhorn: skalakan matriks positif agar marginal baris/kolom sesuai."""
    p = np.exp(logp - logp.max())
    for _ in range(iters):
        p *= r / (p.sum(1, keepdims=True) + 1e-12)
        p *= c / (p.sum(0, keepdims=True) + 1e-12)
    return p


def build_features(emb_tags: list[str], blocks: list[str] | None = None,
                   state_tag: str = "main") -> dict:
    """Susun matriks fitur; state normalisasi disimpan untuk inference notebook."""
    blocks = list(blocks or DEFAULT_BLOCKS)
    state_path = C.MODELS / f"feature_state_{state_tag}.joblib"
    state = joblib.load(state_path) if state_path.exists() else None
    if state is None:
        state = {"tags": list(emb_tags), "blocks": blocks,
                 "stats_mean": None, "stats_std": None}

    embs = [load_embeddings(t) for t in emb_tags]
    ids = embs[0]["ids"]
    for e in embs[1:]:
        assert e["ids"] == ids, "urutan Id antar-embedding tidak konsisten"

    raw = {
        "pre": [e["pre"] for e in embs],
        "post": [e["post"] for e in embs],
        "delta": [e["post"] - e["pre"] for e in embs],
    }
    parts, names, slices = [], [], []
    col = 0
    for name in blocks:
        if name == "stats":
            df = pd.read_csv(C.CHANGE_STATS).set_index("Id").loc[ids]
            z = df.to_numpy(np.float32)
            if state["stats_mean"] is None:
                state["stats_mean"] = z.mean(0)
                state["stats_std"] = z.std(0) + 1e-6
            z = ((z - state["stats_mean"]) / state["stats_std"]).astype(np.float32)
            b = l2norm(z)
        else:
            b = l2norm(np.hstack([l2norm(m.astype(np.float32)) for m in raw[name]]))
        b = b * BLOCK_WEIGHTS.get(name, 1.0)
        parts.append(b)
        names.append(name)
        slices.append((col, col + b.shape[1]))
        col += b.shape[1]

    X = l2norm(np.concatenate(parts, axis=1)).astype(np.float32)
    if not state_path.exists():
        state_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(state, state_path)
    log.info("fitur: %s dari blok %s (state=%s)", X.shape, names, state_path.name)
    return {"ids": ids, "X": X, "blocks": names, "slices": slices}


def balanced_kmeans(X: np.ndarray, k: int, n_iter: int = 30, tau: float = 0.25,
                    seed: int = 0) -> np.ndarray:
    """k-means dengan marginal klaster seragam (opsional, bukan default)."""
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)].copy()
    n = len(X)
    labels = None
    for _ in range(n_iter):
        d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2) ** 2
        d = d / (d.mean() + 1e-9)
        p = sinkhorn(-d / tau, r=1.0 / n, c=1.0 / k, iters=40)
        new = p.argmax(1)
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = X[m].mean(0)
    return labels


def consensus_cluster(X: np.ndarray, slices: list[tuple[int, int]], k: int = C.N_CLASSES,
                      runs: int = C.CONSENSUS_RUNS, seed: int = C.SEED,
                      keep_blocks: float = 0.8) -> tuple[np.ndarray, np.ndarray, dict]:
    """K-means berulang pada subset blok; keluarkan label final + diagnostik stabilitas."""
    rng = np.random.default_rng(seed)
    n = len(X)
    co = np.zeros((n, n), dtype=np.float32)
    run_labels = []
    for r in range(runs):
        keep = [s for s in slices if rng.random() < keep_blocks] or list(slices)
        cols = np.concatenate([np.arange(a, b) for a, b in keep])
        lab = KMeans(k, n_init=10, random_state=seed + r).fit_predict(X[:, cols])
        run_labels.append(lab)
        for c in range(k):
            m = lab == c
            co[np.ix_(m, m)] += 1
        log.info("run konsensus %2d/%d (blok %d/%d)", r + 1, runs, len(keep), len(slices))
    co /= runs

    # label final: k-means pada seluruh fitur dengan banyak inisialisasi
    final = KMeans(k, n_init=20, random_state=seed).fit_predict(X)
    aris = [adjusted_rand_score(run_labels[i], run_labels[j])
            for i in range(runs) for j in range(i + 1, runs)]
    diag = {"pairwise_ari_mean": float(np.mean(aris)), "pairwise_ari_std": float(np.std(aris)),
            "runs": runs, "keep_blocks": keep_blocks,
            "final_vs_run_ari": float(np.mean([adjusted_rand_score(final, l) for l in run_labels]))}
    return final, co, diag


def cluster_quality(X: np.ndarray, labels: np.ndarray) -> dict:
    vals, cnts = np.unique(labels, return_counts=True)
    return {"silhouette_cosine": float(silhouette_score(X, labels, metric="cosine")),
            "cluster_sizes": {int(v): int(c) for v, c in zip(vals, cnts)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["dinov2_s"])
    ap.add_argument("--blocks", default=",".join(DEFAULT_BLOCKS))
    ap.add_argument("--k", type=int, default=C.N_CLASSES)
    ap.add_argument("--runs", type=int, default=C.CONSENSUS_RUNS)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--state-tag", default=None)
    ap.add_argument("--no-stats", action="store_true")
    args = ap.parse_args()

    set_seed(C.SEED)
    blocks = [b for b in args.blocks.split(",") if b]
    if args.no_stats:
        blocks = [b for b in blocks if b != "stats"]
    feats = build_features(args.encoders, blocks, state_tag=args.state_tag or args.tag)
    labels, co, diag = consensus_cluster(feats["X"], feats["slices"], k=args.k, runs=args.runs)
    qual = cluster_quality(feats["X"], labels)
    out = C.FEATURE_DIR / f"clusters_{args.tag}.npz"
    np.savez_compressed(out, ids=np.array(feats["ids"]), X=feats["X"], labels=labels,
                        co=co, k=args.k, blocks=np.array(feats["blocks"]))
    log.info("klaster -> %s", out)
    log.info("stabilitas: %s", diag)
    log.info("kualitas  : %s", qual)


if __name__ == "__main__":
    main()
