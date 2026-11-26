"""Pelatihan model, pencarian pengali kelas, dan inferensi tanpa retraining."""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_loader import (
    CLASSES,
    PROCESSED_DIR,
    SAMPLE_SUBMISSION,
    TARGET,
    read_processed,
)
from .features import DROP_COLS, TE_COLS, build_features, encode_categoricals

HOLD_OUT = 0.12
N_ROUNDS = 420

PARAMS = dict(
    objective="multiclass",
    num_class=len(CLASSES),
    metric="multi_logloss",
    learning_rate=0.09,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.45,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=63,
    cat_smooth=20,
    cat_l2=10,
    num_threads=2,
    verbose=-1,
    seed=42,
    force_row_wise=True,
)


# ------------------------------------------------------------------ persiapan data
def load_matrices(drop_cols=DROP_COLS) -> dict:
    """Muat fitur hasil engineering, pisahkan target/ID, dan encode kategorikal.

    Mengembalikan dict berisi ``tr``, ``te``, ``y``, ``train_ids``, ``test_ids``,
    ``raw_id_tr``/``raw_id_te`` (ID mentah untuk target encoding), ``cat_cols``, dan
    ``cat_categories`` (kamus encoding untuk inferensi).
    """
    t0 = time.time()
    tr = read_processed("feat_train")
    te = read_processed("feat_test")
    print(f"  dimuat: train {tr.shape} | test {te.shape}  [{time.time() - t0:.1f}s]")

    if drop_cols:
        tr.drop(columns=[c for c in drop_cols if c in tr.columns], inplace=True)
        te.drop(columns=[c for c in drop_cols if c in te.columns], inplace=True)

    y = pd.Categorical(tr.pop(TARGET), categories=CLASSES).codes.astype(np.int8)
    train_ids = tr.pop("shipment_id").to_numpy()
    test_ids = te.pop("shipment_id").to_numpy()

    # ID mentah harus diambil sebelum di-encode (target encoding memakai nilai asli)
    raw_id_tr = {c: tr[c].to_numpy() for c in TE_COLS if c in tr.columns}
    raw_id_te = {c: te[c].to_numpy() for c in TE_COLS if c in te.columns}

    tr, te, cat_cols, cat_categories = encode_categoricals(tr, te)
    return dict(
        tr=tr,
        te=te,
        y=y,
        train_ids=train_ids,
        test_ids=test_ids,
        raw_id_tr=raw_id_tr,
        raw_id_te=raw_id_te,
        cat_cols=cat_cols,
        cat_categories=cat_categories,
    )


def split_indices(y, hold_out: float = HOLD_OUT, seed: int = 42):
    """Split hold-out terstratifikasi (alasan: biaya 2 core, sampel validasi 149 ribu)."""
    idx = np.arange(len(y))
    return train_test_split(idx, test_size=hold_out, random_state=seed, stratify=y)


def class_weights(y, n_class: int = len(CLASSES)) -> np.ndarray:
    """Bobot kelas berbanding terbalik dengan frekuensi (menyeimbangkan kelas minoritas)."""
    return len(y) / (n_class * np.bincount(y, minlength=n_class))


# ----------------------------------------------------------------------- pelatihan
def train_model(tr_part, va_part, y_tr, y_va, feats, cat_use, params=None,
                n_rounds: int = N_ROUNDS, weights=None, early_stopping: int = 60):
    """Latih LightGBM multiclass dengan early stopping pada logloss validasi."""
    params = dict(PARAMS if params is None else params)
    w_tr = None if weights is None else weights[y_tr].astype(np.float32)
    w_va = None if weights is None else weights[y_va].astype(np.float32)

    dtrain = lgb.Dataset(tr_part[feats], label=y_tr, weight=w_tr, categorical_feature=cat_use)
    dvalid = lgb.Dataset(va_part[feats], label=y_va, weight=w_va, reference=dtrain,
                         categorical_feature=cat_use)
    t0 = time.time()
    model = lgb.train(
        params, dtrain, num_boost_round=n_rounds, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(50)],
    )
    print(f"  selesai: {model.best_iteration} iterasi terbaik  [{time.time() - t0:.1f}s]")
    return model


def tune_multipliers(p, y_true, n_class: int = len(CLASSES), n_iter: int = 10,
                     grid=None, verbose: bool = True):
    """Coordinate ascent pengali probabilitas kelas untuk memaksimalkan macro F1.

    Implementasi hemat memori: perkalian probabilitas dilakukan float32 dan
    memakai buffer yang dipakai ulang (bukan alokasi baru tiap evaluasi), supaya
    pencarian 26 titik x 7 kelas tetap muat di RAM kecil.
    """
    from sklearn.metrics import f1_score

    grid = np.linspace(0.5, 3.0, 26) if grid is None else grid
    p = np.asarray(p, dtype=np.float32)
    buf = np.empty_like(p)
    w = np.ones(n_class, dtype=np.float32)

    def score(weights) -> float:
        np.multiply(p, weights, out=buf)
        return float(f1_score(y_true, buf.argmax(1), average="macro"))

    best = score(w)
    for it in range(n_iter):
        improved = False
        for k in range(n_class):
            cur = w[k]
            for g in grid:
                w[k] = g
                s = score(w)
                if s > best + 1e-6:
                    best, cur, improved = s, g, True
            w[k] = cur
        if verbose:
            print(f"  iterasi {it + 1}: macro F1 = {best:.4f}")
        if not improved:
            break
    del buf
    return w, best


# ------------------------------------------------------- artefak & inferensi
def save_bundle(model, feats, cat_use, cat_categories, te_maps, multipliers,
                y_tr, out_dir=PROCESSED_DIR, name: str = "lgbm_final") -> dict:
    """Simpan model + seluruh encoder ke ``data/processed`` untuk inferensi ulang."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{name}.txt"
    model.save_model(str(model_path))

    bundle = {
        "classes": list(CLASSES),
        "features": list(feats),
        "cat_cols": list(cat_use),
        "cat_categories": {c: cat_categories[c] for c in cat_use},
        "te_maps": te_maps,
        "te_cols": TE_COLS,
        "prior": (np.bincount(y_tr, minlength=len(CLASSES)) / len(y_tr)).astype(np.float32),
        "multipliers": np.asarray(multipliers, dtype=np.float64),
        "best_iteration": int(model.best_iteration),
    }
    with open(out_dir / "preprocess.pkl", "wb") as fh:
        pickle.dump(bundle, fh, protocol=4)
    return {"model_path": model_path, "bundle": bundle}


def load_bundle(model_dir=PROCESSED_DIR, name: str = "lgbm_final"):
    """Muat model + encoder tersimpan (tanpa menyentuh data train)."""
    model_dir = Path(model_dir)
    booster = lgb.Booster(model_file=str(model_dir / f"{name}.txt"))
    with open(model_dir / "preprocess.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    return booster, bundle


def predict_from_bundle(model_dir=PROCESSED_DIR, out_path=None, use_cache: bool = True,
                        sample_csv=SAMPLE_SUBMISSION) -> pd.DataFrame:
    """Inferensi tanpa retraining: bangun fitur test dari CSV mentah lalu prediksi.

    Semua transformasi memakai kamus tersimpan (``preprocess.pkl``) sehingga hasilnya
    identik dengan prediksi di dalam notebook.
    """
    booster, b = load_bundle(model_dir)
    feats, classes = b["features"], b["classes"]

    cache = Path(model_dir) / "feat_test.parquet"
    if use_cache and cache.exists():
        feat = pd.read_parquet(cache)
    else:                                  # bangun ulang dari CSV mentah test/
        feat = build_features("test").to_pandas()

    df = feat if isinstance(feat, pd.DataFrame) else feat.to_pandas()
    df = df.copy()
    test_ids = df.pop("shipment_id").to_numpy()

    # 1) kategorikal -> kode memakai kamus tersimpan (nilai tak dikenal -> 0)
    for c in b["cat_cols"]:
        mapping = {v: i + 1 for i, v in enumerate(b["cat_categories"][c])}
        df[c] = df[c].map(mapping).fillna(0).astype(np.int32)

    # 2) target encoding memakai tabel tersimpan
    for c in b["te_cols"]:
        raw = df.pop(c).to_numpy() if c in df.columns else None
        for k in range(len(classes)):
            fill = float(b["prior"][k])
            df[f"te_{c}_{k}"] = (
                pd.Series(raw).map(b["te_maps"][c][k]).fillna(fill).to_numpy().astype(np.float32)
                if raw is not None else np.float32(fill)
            )

    # 3) prediksi + pengali kelas (tidak ada pelatihan)
    proba = booster.predict(df[feats], num_iteration=b.get("best_iteration") or 0)
    pred = (proba * np.asarray(b["multipliers"])).argmax(1)

    sub = pd.DataFrame({"shipment_id": test_ids,
                        "status_penanganan": [classes[i] for i in pred]})
    out_path = Path(out_path) if out_path else Path(model_dir) / "submission_inference.csv"
    sample = pd.read_csv(sample_csv)
    sub = sample[["shipment_id"]].merge(sub, on="shipment_id", how="left")
    sub.to_csv(out_path, index=False)
    print(f"submission (inferensi) ditulis: {out_path}  baris={len(sub)}")
    return sub
