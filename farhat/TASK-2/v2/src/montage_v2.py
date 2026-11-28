"""Task-2 v2 Part 3: montage visual per komunitas untuk audit manual.

Render grid 12x12 (144 sampel per komunitas, stratified). Output ke cache/figures/montage_v2_C*.jpg
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src import config as C
from src.utils import get_logger

log = get_logger("montage_v2")
FIG_DIR = C.FIGURE_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Earthquake", "Fire", "Hurricane", "Tsunami", "Volcano", "Explosion"]
GRID = 12
CELL = 128
PAD = 2


def center_crop_70(im: Image.Image) -> Image.Image:
    w, h = im.size
    mw, mh = int(0.15 * w), int(0.15 * h)
    return im.crop((mw, mh, w - mw, h - mh))


def _one(pid: str) -> tuple[np.ndarray, np.ndarray]:
    folder = C.PHENOMENA_DIR / pid
    with Image.open(folder / "pre.jpg") as a, Image.open(folder / "post.jpg") as b:
        c1 = center_crop_70(a.convert("RGB")).resize((CELL, CELL), Image.Resampling.BILINEAR)
        c2 = center_crop_70(b.convert("RGB")).resize((CELL, CELL), Image.Resampling.BILINEAR)
    return np.asarray(c1), np.asarray(c2)


def render_montage(cluster_id: int, labels: np.ndarray, ids: list[str], max_per: int = 144) -> Path:
    idx = np.where(labels == cluster_id)[0]
    if len(idx) == 0:
        return None
    sel = np.random.choice(idx, min(len(idx), max_per), replace=False)
    n = len(sel)
    rows = int(np.ceil(n / GRID))
    H = rows * CELL + (rows + 1) * PAD
    W = GRID * CELL + (GRID + 1) * PAD
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 240

    for i, s in enumerate(sel):
        r, c = divmod(i, GRID)
        y = PAD + r * (CELL + PAD)
        x = PAD + c * (CELL + PAD)
        pid = ids[s]
        pre, post = _one(pid)
        # gabung pre|post horizontal
        pair = np.hstack([pre, post])
        pair = cv2.resize(pair, (CELL, CELL))
        canvas[y:y + CELL, x:x + CELL] = pair

    out = FIG_DIR / f"montage_v2_C{cluster_id:02d}.jpg"
    cv2.imwrite(str(out), canvas)
    log.info("montage C%02d: %d sampel -> %s", cluster_id, n, out.name)
    return out


def main():
    cl = np.load(C.FEATURE_DIR / "clusters_v2.npz")
    labels = cl["labels"]
    ids = pd.read_csv(C.SAMPLE_SUBMISSION)["Id"].tolist()
    n_com = labels.max() + 1
    for c in range(n_com):
        render_montage(c, labels, ids)
    log.info("selesai: %d montage di %s", n_com, FIG_DIR)


if __name__ == "__main__":
    main()