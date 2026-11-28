"""Tahap 0-1: inventarisasi data, cache gambar, dan statistik perubahan.

Tiga keluaran:
  data/processed/manifest.csv      -> daftar 3.180 pasangan + metadata berkas
  cache/thumbs/<id>_{pre,post}.jpg -> cache 256px (mempercepat semua tahap berikutnya)
  data/processed/change_stats.csv  -> fitur perubahan piksel (interpretable, tanpa label)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from src import config as C
from src.utils import Timer, get_logger

log = get_logger("prepare")

# ---------------------------------------------------------------- manifest


def list_ids() -> list[str]:
    ids = sorted(p.name for p in C.PHENOMENA_DIR.iterdir() if p.is_dir())
    return ids


def build_manifest() -> Path:
    ids = list_ids()
    sub_ids = [r["Id"] for r in csv.DictReader(open(C.SAMPLE_SUBMISSION, encoding="utf-8-sig"))]
    assert len(ids) == len(sub_ids), f"jumlah folder {len(ids)} != submission {len(sub_ids)}"
    assert set(ids) == set(sub_ids), "Id folder tidak sama dengan sample_submission"

    rows = []
    for i in ids:
        pre = C.PHENOMENA_DIR / i / "pre.jpg"
        post = C.PHENOMENA_DIR / i / "post.jpg"
        if not (pre.exists() and post.exists()):
            raise FileNotFoundError(f"pasangan tidak lengkap: {i}")
        with Image.open(pre) as im:
            w, h = im.size
        rows.append({
            "Id": i,
            "pre_path": str(pre.relative_to(C.ROOT)).replace("\\", "/"),
            "post_path": str(post.relative_to(C.ROOT)).replace("\\", "/"),
            "width": w,
            "height": h,
            "pre_bytes": pre.stat().st_size,
            "post_bytes": post.stat().st_size,
        })
    C.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(C.MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log.info("manifest: %d pasangan -> %s", len(rows), C.MANIFEST)
    return C.MANIFEST


# ---------------------------------------------------------------- thumbnails


def thumb_path(pid: str, kind: str) -> Path:
    return C.THUMB_DIR / f"{pid}_{kind}.jpg"


def build_thumbs(ids: list[str] | None = None) -> None:
    ids = ids or list_ids()
    done = 0
    for pid in ids:
        for kind in ("pre", "post"):
            out = thumb_path(pid, kind)
            if out.exists():
                continue
            src = C.PHENOMENA_DIR / pid / f"{kind}.jpg"
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail((C.THUMB_SIZE, C.THUMB_SIZE), Image.BICUBIC)
                if im.size != (C.THUMB_SIZE, C.THUMB_SIZE):
                    im = im.resize((C.THUMB_SIZE, C.THUMB_SIZE), Image.BICUBIC)
                im.save(out, quality=C.JPEG_QUALITY, subsampling=0)
            done += 1
    log.info("thumbnail siap: %d berkas baru (total %d)", done, len(list(C.THUMB_DIR.glob('*.jpg'))))


def load_pair(pid: str) -> tuple[np.ndarray, np.ndarray]:
    """Kembalikan pasangan gambar 256px sebagai float32 [0,1], shape (256,256,3)."""
    out = []
    for kind in ("pre", "post"):
        p = thumb_path(pid, kind)
        if not p.exists():  # fallback bila cache belum dibuat
            with Image.open(C.PHENOMENA_DIR / pid / f"{kind}.jpg") as im:
                im = im.convert("RGB").resize((C.THUMB_SIZE, C.THUMB_SIZE), Image.BICUBIC)
                arr = np.asarray(im, dtype=np.float32) / 255.0
        else:
            with Image.open(p) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        out.append(arr)
    return out[0], out[1]


# ---------------------------------------------------------------- change stats

STAT_NAMES = [
    # magnitudo perubahan
    "diff_abs", "diff_abs_r", "diff_abs_g", "diff_abs_b", "frac_changed", "corr_gray",
    # arah perubahan per kanal
    "diff_signed_r", "diff_signed_g", "diff_signed_b",
    # kecerahan / saturasi
    "val_pre", "val_post", "val_delta", "sat_pre", "sat_post", "sat_delta",
    # indeks warna (vegetasi, api, air, abu)
    "green_pre", "green_post", "green_delta",
    "red_pre", "red_post", "red_delta",
    "blue_pre", "blue_post", "blue_delta",
    "gray_pre", "gray_post", "gray_delta",
    # area gelap (bekas terbakar) & kerapatan tepi (kerusakan struktur)
    "dark_pre", "dark_post", "dark_delta",
    "edge_pre", "edge_post", "edge_delta",
]


def _gray(a: np.ndarray) -> np.ndarray:
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def _edge_density(g: np.ndarray) -> float:
    gx = np.abs(np.diff(g, axis=1))[:-1, :]
    gy = np.abs(np.diff(g, axis=0))[:, :-1]
    mag = np.hypot(gx, gy)
    return float((mag > 0.08).mean())


def pair_stats(pre: np.ndarray, post: np.ndarray) -> dict[str, float]:
    r, g, b = 0, 1, 2
    gp, gq = _gray(pre), _gray(post)
    d = post - pre
    ad = np.abs(d)

    val_p = pre.max(axis=2)
    val_q = post.max(axis=2)
    sat_p = pre.max(axis=2) - pre.min(axis=2)
    sat_q = post.max(axis=2) - post.min(axis=2)

    def idx(a, ch):
        return a[..., ch] - a.mean(axis=2)

    feats = {
        "diff_abs": float(ad.mean()),
        "diff_abs_r": float(ad[..., r].mean()),
        "diff_abs_g": float(ad[..., g].mean()),
        "diff_abs_b": float(ad[..., b].mean()),
        "frac_changed": float((ad.max(axis=2) > 0.12).mean()),
        "corr_gray": float(np.corrcoef(gp.ravel(), gq.ravel())[0, 1]),
        "diff_signed_r": float(d[..., r].mean()),
        "diff_signed_g": float(d[..., g].mean()),
        "diff_signed_b": float(d[..., b].mean()),
        "val_pre": float(val_p.mean()), "val_post": float(val_q.mean()),
        "val_delta": float(val_q.mean() - val_p.mean()),
        "sat_pre": float(sat_p.mean()), "sat_post": float(sat_q.mean()),
        "sat_delta": float(sat_q.mean() - sat_p.mean()),
        "green_pre": float(idx(pre, g).mean()), "green_post": float(idx(post, g).mean()),
        "green_delta": float(idx(post, g).mean() - idx(pre, g).mean()),
        "red_pre": float(idx(pre, r).mean()), "red_post": float(idx(post, r).mean()),
        "red_delta": float(idx(post, r).mean() - idx(pre, r).mean()),
        "blue_pre": float(idx(pre, b).mean()), "blue_post": float(idx(post, b).mean()),
        "blue_delta": float(idx(post, b).mean() - idx(pre, b).mean()),
        "gray_pre": float((1.0 - sat_p).mean()), "gray_post": float((1.0 - sat_q).mean()),
        "gray_delta": float((1.0 - sat_q).mean() - (1.0 - sat_p).mean()),
        "dark_pre": float((val_p < 0.25).mean()), "dark_post": float((val_q < 0.25).mean()),
        "dark_delta": float((val_q < 0.25).mean() - (val_p < 0.25).mean()),
        "edge_pre": _edge_density(gp), "edge_post": _edge_density(gq),
        "edge_delta": _edge_density(gq) - _edge_density(gp),
    }
    return feats


def build_change_stats(ids: list[str] | None = None) -> Path:
    ids = ids or list_ids()
    rows = []
    for pid in ids:
        pre, post = load_pair(pid)
        row = {"Id": pid}
        row.update(pair_stats(pre, post))
        rows.append(row)
    with open(C.CHANGE_STATS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log.info("change stats: %d baris x %d fitur -> %s", len(rows), len(STAT_NAMES), C.CHANGE_STATS)
    return C.CHANGE_STATS


def main() -> None:
    ap = argparse.ArgumentParser(description="Bangun manifest, thumbnail, dan statistik perubahan")
    ap.add_argument("--skip-stats", action="store_true")
    args = ap.parse_args()
    with Timer("manifest", log):
        build_manifest()
    with Timer("thumbnail 256px", log):
        build_thumbs()
    if not args.skip_stats:
        with Timer("statistik perubahan", log):
            build_change_stats()


if __name__ == "__main__":
    main()
