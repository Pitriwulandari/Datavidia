"""Buat notebook reproduksi (inference tanpa latih ulang) di notebooks/.

Notebook memuat kode yang menghasilkan submission: mulai dari citra mentah,
encoder beku (bobot publik), transform PCA tersimpan, probe tersimpan, hingga
penulisan berkas CSV. Tidak ada label/GT yang digunakan.
"""
from __future__ import annotations

import json

from src import config as C


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip().splitlines(True)}


def build_notebook(encoders: list[str], size: int, tag: str,
                   probe_tag: str | None = None, out_tag: str = "final") -> dict:
    enc = ", ".join(f'"{e}"' for e in encoders)
    probe_tag = probe_tag or tag
    cells = [
        md(f"""
# Datavidia 11.0 — Task 2: Klasifikasi Fenomena dari Pasangan Citra

Notebook ini mereproduksi submission secara **inference-only**: bobot encoder
pure-vision publik (DINOv2 ViT-S/14 melalui timm), state normalisasi fitur, dan
probe klasifikasi dimuat dari folder `models/`. Tidak ada pelatihan ulang, tidak
ada label/ground truth, tidak ada prompt teks/VLM, dan tidak ada data eksternal
berlabel yang dipakai pada pelatihan.

Alur: manifest & thumbnail → statistik perubahan → embedding encoder beku →
fitur tersimpan → probe → Sinkhorn balance → `submissions/submission_{tag}.csv`.

## Referensi Model Pretrained

**DINOv2 (vit_small_patch14_dinov2.lvd142m):**
- Paper: Oquab, M., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193. https://arxiv.org/abs/2304.07193
- Weights: https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m
- Original: https://github.com/facebookresearch/dinov2
- Method: Self-supervised learning on LVD-142M dataset, pure-vision (no text prompts)

**MobileNetV3 (mobilenetv3_large_100.ra_in1k):**
- Paper: Howard, A., et al. (2019). Searching for MobileNetV3. arXiv:1905.02244. https://arxiv.org/abs/1905.02244
- Weights: https://huggingface.co/timm/mobilenetv3_large_100.ra_in1k
- Pretrained: ImageNet-1k (supervised)

**timm library:**
- Wightman, R. (2019). PyTorch Image Models. https://github.com/huggingface/pytorch-image-models DOI: 10.5281/zenodo.4414861

Semua model menerima citra saja tanpa prompt teks, sesuai FAQ Task-2 no. 2.
"""),
        code(f"""
import os, sys, csv
from pathlib import Path
import numpy as np
import torch

ROOT = Path.cwd().resolve()
if not (ROOT / 'src').exists() and (ROOT.parent / 'src').exists():
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src import config as C
from src.prepare import build_manifest, build_thumbs, build_change_stats
from src.features import build_features
from src.predict import probs_from_bundle, balanced_labels, write_submission, validate_submission

torch.set_num_threads(C.N_JOBS)
ENC = [{enc}]
SIZE = {size}
STATE_TAG = "{tag}"
PROBE_TAG = "{probe_tag}"
OUT_TAG = "{out_tag}"
print('ROOT =', ROOT, '| encoders =', ENC, '| size =', SIZE)
"""),
        md("## 1. Data: hanya pasangan citra (tanpa label)"),
        code("""
build_manifest()      # validasi 3.180 Id vs sample_submission.csv
build_thumbs()        # cache 256px agar inference lebih cepat
build_change_stats()  # 31 statistik perubahan piksel
"""),
        md("## 2. Inference encoder beku (pure-vision, tanpa prompt)\n\nBobot publik diunduh otomatis oleh `timm`; tidak ada proses training pada tahap ini."),
        code("""
from src.embed import run as run_embed
for enc in ENC:
    out = C.FEATURE_DIR / f'emb_{enc}_{SIZE}.npz'
    if not out.exists():
        run_embed(enc, SIZE)
    print('embedding:', out)
"""),
        md("## 3. Pemetaan klaster -> kelas (bukti visual, tanpa GT)"),
        code("""
import json
mapping = json.loads((C.MODELS / 'cluster_mapping.json').read_text(encoding='utf-8'))
for c, name in sorted(mapping['cluster_to_class'].items()):
    print(f'klaster {c} -> {name} | {mapping.get("evidence", {}).get(c, "lihat cache/figures")}')
"""),
        md("## 4. Transform fitur + probe tersimpan (tanpa latih ulang)"),
        code("""
import joblib
feat = build_features(ENC, state_tag=STATE_TAG)
bundle = joblib.load(C.MODELS / f'probe_{PROBE_TAG}.joblib')
probs = probs_from_bundle(bundle, feat['X'])
labels, balanced = balanced_labels(probs)
print('distribusi prediksi:', np.bincount(labels, minlength=6))
"""),
        md("## 5. Tulis & validasi submission"),
        code(f"""
order = [r['Id'] for r in csv.DictReader(open(C.SAMPLE_SUBMISSION, encoding='utf-8'))]
pos = {{pid: i for i, pid in enumerate(feat['ids'])}}
ordered = np.array([labels[pos[p]] for p in order])
out = C.SUBMISSIONS / f'submission_{{OUT_TAG}}.csv'
write_submission(order, ordered, out)
print(out)
print(validate_submission(out))
"""),
    ]
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    enc = ["dinov2_s"]
    size = 196
    tag = "main"
    state = C.MODELS / "notebook_config.json"
    if state.exists():
        cfg = json.loads(state.read_text(encoding="utf-8"))
        enc, size, tag = cfg["encoders"], cfg["size"], cfg["tag"]
        probe_tag = cfg.get("probe_tag", tag)
        out_tag = cfg.get("out_tag", "final")
    else:
        probe_tag, out_tag = tag, "final"
    out = C.NOTEBOOKS / "Datavidia_Task2_Notebook.ipynb"
    out.write_text(json.dumps(build_notebook(enc, size, tag, probe_tag, out_tag),
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print("notebook:", out)


if __name__ == "__main__":
    main()
