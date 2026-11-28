"""Task-2 v2 Part 1: fitur fisik DE-AUGMENTED (crop 70%, white-balance, gray+CLAHE).

Mengatasi cacat v1-v3: data diaugmentasi rotasi/tint/grayscale/awan, sehingga
fitur mentah kacau. Modul ini menghasilkan blok fisik bersih yang digabung
dengan embedding deep yang sudah ada (dinov2_s, dinov3_s).

Output: cache/features/phys_v2.npz -> {features, feature_names, audit_flags}
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import skew

from src import config as C
from src.utils import get_logger

log = get_logger("features_v2")
OUT = C.FEATURE_DIR / "phys_v2.npz"


def center_crop_70(im: Image.Image) -> Image.Image:
    w, h = im.size
    mw, mh = int(0.15 * w), int(0.15 * h)
    return im.crop((mw, mh, w - mw, h - mh))


def to_wb_rgb(im: Image.Image) -> np.ndarray:
    """Z-score per channel pada piksel valid -> mean 128 std 48 (buang tint)."""
    a = np.asarray(im, dtype=np.float32)
    valid = (a.max(axis=2) > 8.0) & (a.min(axis=2) < 225.0)
    if valid.mean() < 0.10:
        valid = np.ones(a.shape[:2], dtype=bool)
    out = np.empty_like(a)
    for c in range(3):
        vals = a[:, :, c][valid]
        out[:, :, c] = np.clip((a[:, :, c] - vals.mean()) / (vals.std() + 1e-5) * 48.0 + 128.0, 0, 255)
    return out


def to_gray_clahe(im: Image.Image) -> np.ndarray:
    g = np.asarray(im.convert("L"), dtype=np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def _rot_flag(im: Image.Image) -> float:
    a = np.asarray(im.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32)
    corners = np.concatenate([a[:4, :4].reshape(-1, 3), a[:4, -4:].reshape(-1, 3),
                              a[-4:, :4].reshape(-1, 3), a[-4:, -4:].reshape(-1, 3)])
    return float((corners.max(axis=1) < 10.0).mean() > 0.25)


def _side_stats(crop: Image.Image) -> tuple[np.ndarray, dict]:
    """Statistik pada sisi (pre/post) setelah de-augmentasi."""
    a = np.asarray(crop.resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32)
    ch_diff = float(np.abs(a[:, :, 0] - a[:, :, 1]).mean() + np.abs(a[:, :, 1] - a[:, :, 2]).mean())
    is_gray = 1.0 if ch_diff < 2.5 else 0.0
    blk = a.max(axis=2) <= 8.0
    cld = a.min(axis=2) >= 225.0
    valid = (~blk) & (~cld)
    if valid.mean() < 0.05:
        valid = np.ones(a.shape[:2], dtype=bool)

    lum = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    lv = lum[valid]
    l_mean, l_std = float(lv.mean()), float(lv.std())
    p10, p50, p90 = [float(x) for x in np.percentile(lv, [10, 50, 90])]

    g = to_gray_clahe(crop.resize((256, 256), Image.Resampling.BILINEAR)).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)[valid]
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)[valid]
    canny = (cv2.Canny(g.astype(np.uint8), 60, 150) > 0)[valid]
    g_blk = g.reshape(32, 8, 32, 8)
    v_blk = valid.reshape(32, 8, 32, 8).mean(axis=(1, 3)) > 0.5
    bv = g_blk.var(axis=(1, 3))[v_blk] if v_blk.any() else g_blk.var(axis=(1, 3)).ravel()

    wb = to_wb_rgb(crop.resize((256, 256), Image.Resampling.BILINEAR))
    exg = (2.0 * wb[:, :, 1] - wb[:, :, 0] - wb[:, :, 2])[valid]
    hsv = cv2.cvtColor(wb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[:, :, 1][valid] / 255.0
    val = hsv[:, :, 2][valid] / 255.0

    d = {
        "valid_frac": float(valid.mean()), "cloud_frac": float(cld.mean()), "blk_frac": float(blk.mean()),
        "is_gray": is_gray,
        "lum_mean": l_mean, "lum_std": l_std, "lum_p10": p10, "lum_p50": p50, "lum_p90": p90,
        "dark_frac": float((lv < 55).mean()), "vdark_frac": float((lv < 35).mean()),
        "bright_frac": float((lv > 195).mean()),
        "sob_mean": float(grad.mean()), "sob_std": float(grad.std()), "lap_var": float(lap.var()),
        "canny_dens": float(canny.mean()), "bvar_mean": float(bv.mean()), "bvar_std": float(bv.std()),
        "exg_std": float(exg.std()), "exg_skew": float(skew(exg)) if exg.std() > 1e-4 else 0.0,
        "sat_mean": float(sat.mean()), "sat_std": float(sat.std()),
        "low_sat": float((sat < 0.15).mean()), "dark_low_sat": float(((sat < 0.15) & (val < 0.30)).mean()),
    }
    return g, d


def feature_names() -> list[str]:
    names = ["valid_frac_pre", "valid_frac_post", "cloud_frac_pre", "cloud_frac_post",
             "blk_frac_pre", "blk_frac_post", "is_gray_pre", "is_gray_post"]
    lum = ["lum_mean", "lum_std", "lum_p10", "lum_p50", "lum_p90", "dark_frac", "vdark_frac", "bright_frac"]
    tex = ["sob_mean", "sob_std", "lap_var", "canny_dens", "bvar_mean", "bvar_std"]
    col = ["exg_std", "exg_skew", "sat_mean", "sat_std", "low_sat", "dark_low_sat"]
    for k in lum:
        names += [f"{k}_pre", f"{k}_post", f"{k}_delta"]
    for k in tex:
        names += [f"{k}_pre", f"{k}_post", f"{k}_delta", f"{k}_ratio"]
    for k in col:
        names += [f"{k}_pre", f"{k}_post", f"{k}_delta"]
    return names


def _one(pid: str) -> tuple[np.ndarray, np.ndarray]:
    folder = C.PHENOMENA_DIR / pid
    with Image.open(folder / "pre.jpg") as a, Image.open(folder / "post.jpg") as b:
        r1, r2 = a.convert("RGB"), b.convert("RGB")
        rot1, rot2 = _rot_flag(r1), _rot_flag(r2)
        c1, c2 = center_crop_70(r1), center_crop_70(r2)
    g1, s1 = _side_stats(c1)
    g2, s2 = _side_stats(c2)
    either_gray = s1["is_gray"] > 0.5 or s2["is_gray"] > 0.5

    vec = [s1[k] for k in ("valid_frac", "cloud_frac", "blk_frac", "is_gray")] + \
          [s2[k] for k in ("valid_frac", "cloud_frac", "blk_frac", "is_gray")]
    for k in ("lum_mean", "lum_std", "lum_p10", "lum_p50", "lum_p90", "dark_frac", "vdark_frac", "bright_frac"):
        vec += [s1[k], s2[k], s2[k] - s1[k]]
    for k in ("sob_mean", "sob_std", "lap_var", "canny_dens", "bvar_mean", "bvar_std"):
        vec += [s1[k], s2[k], s2[k] - s1[k], (s2[k] + 1e-4) / (s1[k] + 1e-4)]
    for k in ("exg_std", "exg_skew", "sat_mean", "sat_std", "low_sat", "dark_low_sat"):
        vec += [s1[k], s2[k], 0.0 if either_gray else s2[k] - s1[k]]
    return np.asarray(vec, np.float32), np.asarray([rot1, rot2, s1["is_gray"], s2["is_gray"]], np.float32)


def build(force: bool = False) -> Path:
    if OUT.exists() and not force:
        d = np.load(OUT)
        if d["features"].shape[0] == 3180:
            log.info("checkpoint ada: %s", OUT)
            return OUT
    ids = pd.read_csv(C.SAMPLE_SUBMISSION)["Id"].tolist()
    t0 = time.time()
    rows, auds = [], []
    with ThreadPoolExecutor(max_workers=2) as ex:
        for i, (v, a) in enumerate(ex.map(_one, ids, chunksize=16)):
            rows.append(v); auds.append(a)
            if (i + 1) % 640 == 0:
                log.info("phys %d/%2d (%.0fs)", i + 1, len(ids), time.time() - t0)
    mat = np.nan_to_num(np.vstack(rows).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    aud = np.vstack(auds).astype(np.float32)
    np.savez_compressed(OUT, features=mat, feature_names=np.asarray(feature_names()), audit_flags=aud)
    log.info("tersimpan %s shape=%s (%.0fs)", OUT, mat.shape, time.time() - t0)
    return OUT


if __name__ == "__main__":
    build()
