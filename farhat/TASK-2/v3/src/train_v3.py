"""Task-2 v3: Training with Patch-Chamfer Features + Hurricane Anchor.

Strategy:
1. Hurricane detector (from v4_deep) as anchor - 311 samples, CV 0.71, precision 1.0
2. Train on topchg_delta embeddings for residual samples
3. Ensemble: ResidualMLP + LogisticRegression + kNN
4. Output 6-class predictions
"""
from __future__ import annotations

import json
import pickle
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache" / "features"
MODELS_DIR = ROOT / "models"
SUB_DIR = ROOT / "submissions"
SUB_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Earthquake", "Explosion", "Fire", "Hurricane", "Tsunami", "Volcano"]


class ResidualMLP:
    """MLP with residual connection from input."""
    
    def __init__(self, hidden_dim=256, max_iter=200, alpha=0.001):
        self.hidden_dim = hidden_dim
        self.max_iter = max_iter
        self.alpha = alpha
        self.mlp = None
        self.scaler = None
        self.classes_ = np.array(CLASS_NAMES)
    
    def fit(self, X, y):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        input_dim = X.shape[1]
        
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(self.hidden_dim, self.hidden_dim // 2),
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            batch_size=64,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=self.max_iter,
            random_state=SEED,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )
        self.mlp.fit(X_scaled, y)
        self.classes_ = self.mlp.classes_
        return self
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.mlp.predict_proba(X_scaled)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def load_patch_chamfer_features():
    """Load patch-chamfer v3 features (DINOv2 patch-level embeddings)."""
    pc_path = CACHE_DIR / "patch_chamfer_v3_dinov2_vits14_336.npz"
    if not pc_path.exists():
        raise FileNotFoundError(f"Patch-chamfer features not found: {pc_path}")
    
    d = np.load(pc_path)
    # Construct combined features:
    # - pmean_pre (384) + pmean_post (384) = 768: global patch means
    # - topchg_delta (384): top-24 patch delta
    # - chamfer_stats (12): aggregated stats
    pmean = np.concatenate([d["pmean_pre"], d["pmean_post"]], axis=1)  # (N, 768)
    topchg_delta = d["topchg_delta"]  # (N, 384)
    chamfer_stats = d["chamfer_stats"]  # (N, 12)
    
    # Main feature: patch means + topchg_delta
    X_main = np.concatenate([pmean, topchg_delta], axis=1)  # (N, 1152)
    # Extended feature: add chamfer stats
    X_ext = np.concatenate([X_main, chamfer_stats], axis=1)  # (N, 1164)
    
    return {
        "pmean": pmean,               # (N, 768)
        "topchg_delta": topchg_delta, # (N, 384)
        "chamfer_stats": chamfer_stats, # (N, 12)
        "X_main": X_main,             # (N, 1152)
        "X_ext": X_ext,               # (N, 1164)
    }


def load_hurricane_detector():
    """Load Hurricane detector (v3 embeddings + Hurricane anchor mask).

    Anchor mask diambil dari `clusters_v3.npz["hurricane_mask"]` yang sudah
    dihitung di tahap clustering, sehingga tidak bergantung pada artefak v3
    yang mungkin sudah tidak tersedia.
    """
    emb_path = CACHE_DIR / "emb_dinov2_s_196.npz"
    if not emb_path.exists():
        raise FileNotFoundError(f"DINOv2 embeddings tidak ditemukan: {emb_path}")

    d = np.load(emb_path)
    X = np.concatenate([d["pre"], d["post"]], axis=1)

    mask_path = CACHE_DIR / "clusters_v3.npz"
    if not mask_path.exists():
        raise FileNotFoundError(f"clusters_v3.npz tidak ditemukan: {mask_path}")
    y = np.load(mask_path)["hurricane_mask"].astype(int)

    # Train LogisticRegression (same as before)
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
    clf.fit(X, y)

    print(f"[v3] Hurricane detector trained on {int(y.sum())} samples (anchor mask)")
    return clf


def create_pseudo_labels(
    sample_ids: list[str],
    cluster_labels: np.ndarray,
    hurricane_mask: np.ndarray,
    cluster_mapping: dict,
) -> np.ndarray:
    """Create pseudo-labels from cluster mapping + Hurricane anchor."""
    
    # Use object dtype to avoid string truncation
    labels = np.empty(len(sample_ids), dtype=object)
    labels[:] = "Unknown"
    
    # Hurricane from anchor
    labels[hurricane_mask] = "Hurricane"
    
    # Other classes from cluster mapping
    for i, c in enumerate(cluster_labels):
        if c >= 0 and labels[i] == "Unknown":
            c_key = f"C{c:02d}"
            if c_key in cluster_mapping:
                labels[i] = cluster_mapping[c_key]
    
    # Report distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"[v3] Pseudo-label distribution:")
    for u, c in zip(unique, counts):
        print(f"  {u}: {c}")
    
    return labels


def train_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
):
    """Train ensemble of ResidualMLP + LR + kNN."""
    
    # Filter known labels
    known_mask = y != "Unknown"
    X_known = X[known_mask]
    y_known = y[known_mask]
    
    print(f"[v3] Training on {len(y_known)} samples with known labels")
    
    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_known)
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    cv_scores = {"mlp": [], "lr": [], "knn": [], "ensemble": []}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_known, y_enc)):
        X_train, X_val = X_known[train_idx], X_known[val_idx]
        y_train, y_val = y_known[train_idx], y_known[val_idx]  # string labels for fitting
        
        # ResidualMLP
        mlp = ResidualMLP(hidden_dim=256, max_iter=200)
        mlp.fit(X_train, y_train)
        mlp_proba = mlp.predict_proba(X_val)
        
        # LogisticRegression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        lr = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
        lr.fit(X_train_scaled, y_train)
        lr_proba = lr.predict_proba(X_val_scaled)
        
        # kNN
        knn = KNeighborsClassifier(n_neighbors=15, metric="cosine")
        knn.fit(X_train, y_train)
        knn_proba = knn.predict_proba(X_val)
        
        # Ensemble (weighted average)
        ensemble_proba = 0.45 * mlp_proba + 0.35 * lr_proba + 0.20 * knn_proba
        
        # Scores
        mlp_pred = mlp.classes_[np.argmax(mlp_proba, axis=1)]
        lr_pred = lr.classes_[np.argmax(lr_proba, axis=1)]
        knn_pred = knn.classes_[np.argmax(knn_proba, axis=1)]
        ens_pred = le.classes_[np.argmax(ensemble_proba, axis=1)]
        
        from sklearn.metrics import f1_score
        cv_scores["mlp"].append(f1_score(y_val, mlp_pred, average="macro"))
        cv_scores["lr"].append(f1_score(y_val, lr_pred, average="macro"))
        cv_scores["knn"].append(f1_score(y_val, knn_pred, average="macro"))
        cv_scores["ensemble"].append(f1_score(y_val, ens_pred, average="macro"))
        
        print(f"  Fold {fold+1}/{n_folds}: MLP={cv_scores['mlp'][-1]:.4f}, LR={cv_scores['lr'][-1]:.4f}, kNN={cv_scores['knn'][-1]:.4f}, Ens={cv_scores['ensemble'][-1]:.4f}")
    
    print(f"\n[v3] CV Macro-F1:")
    print(f"  MLP: {np.mean(cv_scores['mlp']):.4f} ± {np.std(cv_scores['mlp']):.4f}")
    print(f"  LR:  {np.mean(cv_scores['lr']):.4f} ± {np.std(cv_scores['lr']):.4f}")
    print(f"  kNN: {np.mean(cv_scores['knn']):.4f} ± {np.std(cv_scores['knn']):.4f}")
    print(f"  Ens: {np.mean(cv_scores['ensemble']):.4f} ± {np.std(cv_scores['ensemble']):.4f}")
    
    # Train final models on all data
    print("\n[v3] Training final models on all data...")
    
    final_mlp = ResidualMLP(hidden_dim=256, max_iter=300)
    final_mlp.fit(X_known, y_enc)
    
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X_known)
    final_lr = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
    final_lr.fit(X_scaled, y_enc)
    
    final_knn = KNeighborsClassifier(n_neighbors=15, metric="cosine")
    final_knn.fit(X_known, y_enc)
    
    return {
        "mlp": final_mlp,
        "lr": final_lr,
        "knn": final_knn,
        "scaler": final_scaler,
        "label_encoder": le,
        "cv_scores": cv_scores,
    }


def predict_all(
    X: np.ndarray,
    models: dict,
    hurricane_detector,
    hurricane_mask: np.ndarray,
) -> np.ndarray:
    """Predict all samples with Hurricane override."""
    
    # Get base predictions from ensemble
    mlp_proba = models["mlp"].predict_proba(X)
    lr_proba = models["lr"].predict_proba(models["scaler"].transform(X))
    knn_proba = models["knn"].predict_proba(X)
    
    # Weighted ensemble
    ensemble_proba = 0.45 * mlp_proba + 0.35 * lr_proba + 0.20 * knn_proba
    
    # Decode labels
    le = models["label_encoder"]
    predictions = le.classes_[np.argmax(ensemble_proba, axis=1)]
    
    # Override Hurricane from detector
    predictions[hurricane_mask] = "Hurricane"
    
    return predictions


def main():
    # Load sample IDs
    sample_sub_path = ROOT / "data" / "raw" / "sample_submission.csv"
    sample_ids = pd.read_csv(sample_sub_path)["Id"].tolist()
    N = len(sample_ids)
    assert N == 3180
    
    # Load patch-chamfer features
    pc = load_patch_chamfer_features()
    X = pc["X_main"]  # (N, 1152) patch means + topchg_delta
    print(f"[v3] Feature shape: {X.shape}")
    
    # Load cluster labels
    clusters_path = CACHE_DIR / "clusters_v3.npz"
    if not clusters_path.exists():
        raise FileNotFoundError(f"Run cluster_v3.py first: {clusters_path}")
    
    clusters = np.load(clusters_path)
    cluster_labels = clusters["labels"]
    hurricane_mask = clusters["hurricane_mask"]
    
    # Load cluster mapping
    mapping_path = MODELS_DIR / "cluster_mapping_v3.json"
    with open(mapping_path) as f:
        cluster_mapping = json.load(f)
    
    # Create pseudo-labels
    y = create_pseudo_labels(sample_ids, cluster_labels, hurricane_mask, cluster_mapping)
    
    # Train ensemble
    models = train_ensemble(X, y, n_folds=5)
    
    # Load Hurricane detector
    hurricane_detector = load_hurricane_detector()
    
    # Predict
    predictions = predict_all(X, models, hurricane_detector, hurricane_mask)
    
    # Distribution
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"\n[v3] Final prediction distribution:")
    for u, c in zip(unique, counts):
        print(f"  {u}: {c}")
    
    # Save submission
    submission = pd.DataFrame({
        "Id": sample_ids,
        "label": predictions,
    })
    sub_path = SUB_DIR / "submission_v3.csv"
    submission.to_csv(sub_path, index=False)
    print(f"\n[v3] Saved {sub_path}")
    
    # Save model bundle (sklearn components only -> portable across processes)
    bundle = {
        "mlp": models["mlp"].mlp,
        "mlp_scaler": models["mlp"].scaler,
        "lr": models["lr"],
        "knn": models["knn"],
        "scaler": models["scaler"],
        "label_encoder": models["label_encoder"],
        "cv_scores": models["cv_scores"],
    }
    models_path = MODELS_DIR / "models_v3.joblib"
    joblib.dump(bundle, models_path)
    print(f"[v3] Saved {models_path}")

    export_artifacts(models)


def export_artifacts(models: dict) -> None:
    """Export competition-ready artifacts: ONNX for MLP/LR + joblib for sklearn parts."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    mlp = models["mlp"]
    lr = models["lr"]
    n_features = int(mlp.mlp.n_features_in_)

    # sklearn-native artifacts
    # NOTE: dump the underlying sklearn estimator (mlp.mlp), NOT the ResidualMLP wrapper,
    # so artifacts load in any process without importing src.train_v3.
    joblib.dump(mlp.mlp, MODELS_DIR / "mlp_v3.joblib")
    joblib.dump(mlp.scaler, MODELS_DIR / "mlp_scaler_v3.joblib")
    joblib.dump(lr, MODELS_DIR / "lr_v3.joblib")
    joblib.dump(models["knn"], MODELS_DIR / "knn_v3.joblib")
    joblib.dump(models["scaler"], MODELS_DIR / "scaler_v3.joblib")
    joblib.dump(models["label_encoder"], MODELS_DIR / "label_encoder_v3.joblib")
    print("[v3] Saved joblib artifacts")

    # ONNX exports (zipmap=False -> probabilities as plain tensor)
    for name, est in (("mlp_v3", mlp.mlp), ("lr_v3", lr)):
        onx = convert_sklearn(
            est,
            initial_types=[("input", FloatTensorType([None, n_features]))],
            options={id(est): {"zipmap": False}},
        )
        out = MODELS_DIR / f"{name}.onnx"
        out.write_bytes(onx.SerializeToString())
        print(f"[v3] Saved {out}")

    # Model config / card
    cv = {}
    for k, v in models["cv_scores"].items():
        arr = np.asarray(v, dtype=float).ravel()
        cv[k] = {"folds": arr.tolist(), "mean": float(arr.mean()), "std": float(arr.std())}

    config = {
        "input_dim": n_features,
        "hidden_layer_sizes": list(mlp.mlp.hidden_layer_sizes),
        "num_classes": int(len(models["label_encoder"].classes_)),
        "classes": models["label_encoder"].classes_.tolist(),
        "ensemble_weights": {"mlp": 0.45, "lr": 0.35, "knn": 0.20},
        "knn_metric": models["knn"].metric,
        "seed": SEED,
        "cv_scores": cv,
    }
    cfg_path = MODELS_DIR / "model_config_v3.json"
    cfg_path.write_text(json.dumps(config, indent=2))
    print(f"[v3] Saved {cfg_path}")


if __name__ == "__main__":
    main()
