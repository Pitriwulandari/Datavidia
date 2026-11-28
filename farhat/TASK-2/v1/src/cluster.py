"""Tahap 4: diagnosis klaster (montage prototipe & profil statistik).

Keluaran dipakai untuk menafsirkan arti tiap klaster (pemetaan klaster -> nama
fenomena) berdasarkan bukti visual pada pasangan pre/post dan profil perubahan.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src import config as C
from src.prepare import load_pair
from src.utils import get_logger, load_json

log = get_logger("cluster")


def _font(size: int):
    for p in ("C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def representative_indices(X: np.ndarray, labels: np.ndarray, c: int, n: int = 16,
                           random_n: int = 4, seed: int = C.SEED) -> list[int]:
    idx = np.where(labels == c)[0]
    center = X[idx].mean(0)
    d = 1.0 - (X[idx] @ center) / (np.linalg.norm(X[idx], axis=1) * np.linalg.norm(center) + 1e-9)
    nearest = idx[np.argsort(d)[:n]]
    rng = np.random.default_rng(seed + c)
    extra = rng.choice(idx, size=min(random_n, len(idx)), replace=False)
    return list(dict.fromkeys([int(x) for x in nearest] + [int(x) for x in extra]))


def _cell(pid: str, size: int = 168) -> Image.Image:
    pre, post = load_pair(pid)
    h = size + 34
    cell = Image.new("RGB", (size * 2 + 6, h), "white")
    cell.paste(Image.fromarray((pre * 255).astype(np.uint8), "RGB").resize((size, size)), (0, 0))
    cell.paste(Image.fromarray((post * 255).astype(np.uint8), "RGB").resize((size, size)), (size + 6, 0))
    dr = ImageDraw.Draw(cell)
    dr.text((4, size + 3), f"{pid} pre | post", fill="black", font=_font(12))
    return cell


def make_montage(ids: list[str], X: np.ndarray, labels: np.ndarray, c: int,
                 n: int = 16, random_n: int = 4, cols: int = 4) -> Image.Image:
    reps = representative_indices(X, labels, c, n=n, random_n=random_n)
    cells = [_cell(ids[i]) for i in reps]
    w, h = cells[0].size
    rows = int(np.ceil(len(cells) / cols))
    canvas = Image.new("RGB", (cols * (w + 6) + 6, rows * (h + 6) + 44), "white")
    dr = ImageDraw.Draw(canvas)
    dr.text((10, 8), f"CLUSTER {c}  (n={int((labels == c).sum())})  "
                     f"-- nearest-to-centroid + random", fill="black", font=_font(18))
    for j, cell in enumerate(cells):
        x = 6 + (j % cols) * (w + 6)
        y = 44 + (j // cols) * (h + 6)
        canvas.paste(cell, (x, y))
    return canvas


def make_atlas(ids: list[str], X: np.ndarray, labels: np.ndarray, per_cluster: int = 8) -> Image.Image:
    rows = []
    for c in sorted(np.unique(labels)):
        reps = representative_indices(X, labels, c, n=per_cluster, random_n=0)
        cells = [_cell(ids[i], size=112) for i in reps]
        w, h = cells[0].size
        strip = Image.new("RGB", (len(cells) * (w + 4) + 130, h + 6), "white")
        d = ImageDraw.Draw(strip)
        d.text((4, 8), f"C{c}", fill="black", font=_font(20))
        for j, cell in enumerate(cells):
            strip.paste(cell, (126 + j * (w + 4), 3))
        rows.append(strip)
    W = max(r.width for r in rows)
    H = sum(r.height + 4 for r in rows)
    atlas = Image.new("RGB", (W, H), "white")
    y = 0
    for r in rows:
        atlas.paste(r, (0, y))
        y += r.height + 4
    return atlas


def cluster_profile(ids: list[str], labels: np.ndarray, path=None) -> pd.DataFrame:
    df = pd.read_csv(C.CHANGE_STATS).set_index("Id").loc[ids].copy()
    df["cluster"] = labels
    prof = df.groupby("cluster").mean()
    if path:
        prof.to_csv(path, float_format="%.5f")
    return prof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", default="main")
    ap.add_argument("--per-cluster", type=int, default=16)
    args = ap.parse_args()

    d = np.load(C.FEATURE_DIR / f"clusters_{args.clusters}.npz", allow_pickle=True)
    ids = [str(x) for x in d["ids"]]
    X, labels = d["X"], d["labels"]
    k = int(d["k"])

    prof = cluster_profile(ids, labels, C.DATA_PROCESSED / "cluster_profile.csv")
    log.info("profil klaster (mean):\n%s", prof.to_string(float_format=lambda x: f"{x:.3f}"))
    for c in range(k):
        out = C.FIGURE_DIR / f"cluster_{c}_prototypes.jpg"
        make_montage(ids, X, labels, c, n=args.per_cluster).save(out, quality=90)
        log.info("montage %s", out)
    atlas = C.FIGURE_DIR / "cluster_atlas.jpg"
    make_atlas(ids, X, labels).save(atlas, quality=90)
    log.info("atlas %s", atlas)


if __name__ == "__main__":
    main()
