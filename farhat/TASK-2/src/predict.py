"""Tahap 6: inferensi & penulisan submission.

Probe menghasilkan probabilitas; penugasan akhir diseimbangkan dengan Sinkhorn
(marginal tiap kelas = 1/6) karena metrik Macro-F1 memberi bobot sama per kelas.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
from sklearn.metrics import f1_score

from src import config as C
from src.features import sinkhorn
from src.utils import get_logger, load_json

log = get_logger("predict")


def probs_from_bundle(bundle: dict, X: np.ndarray) -> np.ndarray:
    def aligned(model) -> np.ndarray:
        p = model.predict_proba(X)
        out = np.zeros((len(X), C.N_CLASSES), dtype=np.float64)
        for j, cls in enumerate(model.classes_):
            out[:, int(cls)] = p[:, j]
        return out

    p1, p2 = aligned(bundle["lr"]), aligned(bundle["knn"])
    return 0.7 * p1 + 0.3 * p2


def balanced_labels(probs: np.ndarray, tau: float = 0.2, iters: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Seimbangkan marginal prediksi: tiap kelas memperoleh ~1/K dari sampel."""
    n, k = probs.shape
    p = sinkhorn(np.log(probs + 1e-12) / max(tau, 1e-3), r=1.0 / n, c=1.0 / k, iters=iters)
    return p.argmax(1), p


def write_submission(ids: list[str], labels: np.ndarray, path) -> None:
    assert len(ids) == len(labels)
    # Aturan tidak mensyaratkan jumlah minimum per kelas.
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["Id", "Phenomenon"])
        w.writerows((pid, C.CLASSES[int(c)]) for pid, c in zip(ids, labels))


def validate_submission(path) -> dict:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    sample = list(csv.DictReader(open(C.SAMPLE_SUBMISSION, encoding="utf-8")))
    ids = [r["Id"] for r in rows]
    assert len(rows) == len(sample) == 3180, "jumlah baris bukan 3180"
    assert list(rows[0]) == ["Id", "Phenomenon"], "header salah"
    assert len(ids) == len(set(ids)), "Id duplikat"
    assert set(ids) == {r["Id"] for r in sample}, "kumpulan Id tidak sama dengan sample_submission"
    assert all(r["Phenomenon"] in C.CLASSES for r in rows), "label di luar daftar"
    counts = {c: sum(r["Phenomenon"] == c for r in rows) for c in C.CLASSES}
    return {"rows": len(rows), "class_counts": counts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", default="main")
    ap.add_argument("--probe", default="main")
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--tag", default="final")
    args = ap.parse_args()

    import joblib
    import pandas as pd

    d = np.load(C.FEATURE_DIR / f"clusters_{args.clusters}.npz", allow_pickle=True)
    ids = [str(x) for x in d["ids"]]
    X = d["X"]
    bundle = joblib.load(C.MODELS / f"probe_{args.probe}.joblib")
    probs = probs_from_bundle(bundle, X)
    labels = probs.argmax(1)
    if not args.no_balance:
        labels, _ = balanced_labels(probs, tau=args.tau)

    # urutan Id mengikuti sample_submission (bukan urutan folder)
    order = [r["Id"] for r in csv.DictReader(open(C.SAMPLE_SUBMISSION, encoding="utf-8"))]
    pos = {pid: i for i, pid in enumerate(ids)}
    labels_ordered = np.array([labels[pos[pid]] for pid in order])
    out = C.SUBMISSIONS / f"submission_{args.tag}.csv"
    write_submission(order, labels_ordered, out)
    report = validate_submission(out)
    mapping = load_json(C.MODELS / "cluster_mapping.json")["cluster_to_class"]
    pseudo = np.array([C.CLASS_TO_IDX[mapping[str(c)]] for c in d["labels"]])
    report["agreement_with_clusters"] = float((labels == pseudo).mean())
    report["macro_f1_vs_clusters"] = float(f1_score(pseudo, labels, average="macro"))
    log.info("submission: %s -> %s", out, report)


if __name__ == "__main__":
    main()
