"""Validasi akhir artefak & kepatuhan format (tanpa akses ground truth)."""
from __future__ import annotations

import json

from src import config as C
from src.predict import validate_submission
from src.utils import get_logger

log = get_logger("validate")


def main() -> None:
    checks = {}
    for p in (C.MANIFEST, C.CHANGE_STATS, C.FEATURE_DIR / "clusters_main.npz",
              C.MODELS / "cluster_mapping.json", C.MODELS / "feature_state_main.joblib",
              C.MODELS / "probe_main.joblib", C.SUBMISSIONS / "submission_final.csv",
              C.NOTEBOOKS / "Datavidia_Task2_Notebook.ipynb"):
        checks[str(p.relative_to(C.ROOT))] = p.exists() and p.stat().st_size > 0
    report = {"artifacts": checks}
    if (C.SUBMISSIONS / "submission_final.csv").exists():
        report["submission"] = validate_submission(C.SUBMISSIONS / "submission_final.csv")
    if (C.MODELS / "cluster_mapping.json").exists():
        m = json.loads((C.MODELS / "cluster_mapping.json").read_text(encoding="utf-8"))
        report["mapping"] = m
    print(json.dumps(report, indent=2, ensure_ascii=False))
    assert all(checks.values()), f"artefak kurang: {[k for k, v in checks.items() if not v]}"


if __name__ == "__main__":
    main()
