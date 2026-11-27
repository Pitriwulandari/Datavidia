"""Task-2 v5 Part 1: De-Tinted True-Color Aligned Extraction, 2D-FFT Sensor/Stripe Physics,
and Aligned Visual Change-Map DINOv2 Features.

100% Pure-Vision & Compliant (FAQ Task-2 no. 2):
  - No text/point/box prompts, no VLM/LLM/CLIP/SAM, no external labeled data.
  - Deterministic SEED = 42, local hub model loading, checkpointed per stage, RAM-safe.
  - Reuses 4-DOF spatial alignment from ghani/task2/cache/v4/alignment_v4.npz.
"""
from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "ghani" / "task2" / "src" / "v4"))
from align_extract_v4 import (  # noqa: E402
    _local_3x3_chamfer_pair,
    l2norm,
    valid_mask_grid,
    warp_post_to_pre,
)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(12, os.cpu_count() or 12))

DATA_DIR = ROOT / "Task-2"
FEN_DIR = DATA_DIR / "Fenomena"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"
C4_DIR = ROOT / "ghani" / "task2" / "cache" / "v4"
CACHE_DIR = ROOT / "ghani" / "task2" / "cache" / "v5"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def detint_clean_rgb(im_rgb: Image.Image, fill_neutral_border: bool = True) -> Image.Image:
    """Invert synthetic multiplicative RGB channel tints while preserving natural disaster colors.

    - Estimates per-channel gray-world gain strictly on valid non-black, non-cloud pixels.
    - Smoothly preserves bright white clouds (min(R,G,B) > 205) so clouds stay neutral white.
    - Optionally fills rotated black border triangles with neutral gray (124) for ViT inputs.
    """
    a = np.asarray(im_rgb, dtype=np.float32)
    nonblk = a.max(axis=2) > 12.0
    valid = nonblk & (a.min(axis=2) < 225.0)
    if valid.sum() >= 100:
        px = a[valid]
        is_gray = (
            float(np.abs(px[:, 0] - px[:, 1]).mean()) < 2.0
            and float(np.abs(px[:, 1] - px[:, 2]).mean()) < 2.0
        )
        if not is_gray:
            ch_mean = px.mean(axis=0)
            gain = np.clip(float(ch_mean.mean()) / (ch_mean + 1e-5), 0.55, 1.85)
            out = np.clip(a * gain[None, None, :], 0.0, 255.0)
            cmin = a.min(axis=2)
            cw = np.clip((cmin - 205.0) / 30.0, 0.0, 1.0)[..., None]
            lum_orig = a.mean(axis=2, keepdims=True)
            out = out * (1.0 - cw) + lum_orig * cw
        else:
            out = a.copy()
    else:
        out = a.copy()

    if fill_neutral_border:
        out[~nonblk] = 124.0
    else:
        out[~nonblk] = 0.0
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def _fft_sensor_and_stripe_features(g_u8: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Compute rotation-invariant 12-bin radial 2D FFT log-power spectrum + angular stripe peak."""
    h, w = g_u8.shape
    sz = 256
    g = cv2.resize(g_u8, (sz, sz), interpolation=cv2.INTER_AREA).astype(np.float32)
    m = cv2.resize(valid_mask.astype(np.float32), (sz, sz), interpolation=cv2.INTER_NEAREST) > 0.5
    if m.sum() < 256:
        return np.zeros(12, dtype=np.float32), 0.0, 0.0

    mu = float(g[m].mean())
    sd = float(g[m].std()) + 1e-5
    g_norm = np.where(m, (g - mu) / sd, 0.0)
    win = np.outer(np.hanning(sz), np.hanning(sz)).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(g_norm * win))
    P = np.log1p(np.abs(F) ** 2).astype(np.float32)

    cy, cx = sz // 2, sz // 2
    yy, xx = np.ogrid[:sz, :sz]
    dy = (yy - cy).astype(np.float32)
    dx = (xx - cx).astype(np.float32)
    rr = np.hypot(dy, dx)
    ang = (np.degrees(np.arctan2(dy, dx)) % 180.0).astype(np.float32)

    # 12 log-spaced radial bins in [2, 120]
    edges = np.geomspace(2.0, 120.0, 13)
    rad_bins = np.zeros(12, dtype=np.float32)
    for b in range(12):
        rm = (rr >= edges[b]) & (rr < edges[b + 1])
        rad_bins[b] = float(P[rm].mean()) if rm.any() else 0.0
    rad_norm = rad_bins / (rad_bins.sum() + 1e-6)

    # Rotation-invariant stripe detector: max vs median angular power across 36 wedges (5 deg each)
    ann = (rr >= 14.0) & (rr <= 105.0)
    wedge_vals = []
    for k in range(36):
        wm = ann & (ang >= k * 5.0) & (ang < (k + 1) * 5.0)
        if wm.any():
            wedge_vals.append(float(P[wm].mean()))
    if wedge_vals:
        wv = np.array(wedge_vals, dtype=np.float32)
        med = float(np.median(wv))
        iqr = float(np.percentile(wv, 75) - np.percentile(wv, 25)) + 1e-4
        stripe_peak = float(wv.max() - med)
        stripe_z = float((wv.max() - med) / iqr)
    else:
        stripe_peak, stripe_z = 0.0, 0.0
    return rad_norm, stripe_peak, stripe_z


def _compute_detint_phys_single(args):
    sid, ang, dy, dx = args
    sz = 384
    folder = FEN_DIR / sid
    with Image.open(folder / "pre.jpg") as im1, Image.open(folder / "post.jpg") as im2:
        r1 = im1.convert("RGB").resize((sz, sz), Image.Resampling.BILINEAR)
        r2 = warp_post_to_pre(im2.convert("RGB"), float(ang), int(dy), int(dx), sz)

    raw1 = np.asarray(r1, dtype=np.float32)
    raw2 = np.asarray(r2, dtype=np.float32)
    m1 = (raw1.max(axis=2) > 12.0) & (raw1.min(axis=2) < 225.0)
    m2 = (raw2.max(axis=2) > 12.0) & (raw2.min(axis=2) < 225.0)
    ov = m1 & m2
    if ov.sum() < 400:
        ov = (raw1.max(axis=2) > 12.0) & (raw2.max(axis=2) > 12.0)
    if ov.sum() < 200:
        ov = np.ones((sz, sz), dtype=bool)

    d1 = np.asarray(detint_clean_rgb(r1, fill_neutral_border=False), dtype=np.float32)
    d2 = np.asarray(detint_clean_rgb(r2, fill_neutral_border=False), dtype=np.float32)

    g1_u8 = cv2.cvtColor(d1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    g2_u8 = cv2.cvtColor(d2.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    g1 = g1_u8.astype(np.float32)
    g2 = g2_u8.astype(np.float32)

    # Check if either image is originally grayscale
    px1 = raw1[ov]
    px2 = raw2[ov]
    gray1 = float(np.abs(px1[:, 0] - px1[:, 1]).mean()) < 2.5 and float(np.abs(px1[:, 1] - px1[:, 2]).mean()) < 2.5
    gray2 = float(np.abs(px2[:, 0] - px2[:, 1]).mean()) < 2.5 and float(np.abs(px2[:, 1] - px2[:, 2]).mean()) < 2.5
    either_gray = float(gray1 or gray2)

    # HSV saturation & Excess Green on de-tinted RGB
    hsv1 = cv2.cvtColor(d1.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv2 = cv2.cvtColor(d2.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    sat1 = hsv1[:, :, 1] / 255.0
    sat2 = hsv2[:, :, 1] / 255.0

    exg1 = (2.0 * d1[:, :, 1] - d1[:, :, 0] - d1[:, :, 2]) / (d1.sum(axis=2) + 1.0)
    exg2 = (2.0 * d2[:, :, 1] - d2[:, :, 0] - d2[:, :, 2]) / (d2.sum(axis=2) + 1.0)
    rg1 = (d1[:, :, 0] - d1[:, :, 1]) / (d1[:, :, 0] + d1[:, :, 1] + 1.0)
    rg2 = (d2[:, :, 0] - d2[:, :, 1]) / (d2[:, :, 0] + d2[:, :, 1] + 1.0)
    br1 = (d1[:, :, 2] - d1[:, :, 0]) / (d1[:, :, 2] + d1[:, :, 0] + 1.0)
    br2 = (d2[:, :, 2] - d2[:, :, 0]) / (d2[:, :, 2] + d2[:, :, 0] + 1.0)

    # Texture / edges
    sob1 = np.hypot(cv2.Sobel(g1, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g1, cv2.CV_32F, 0, 1, ksize=3))
    sob2 = np.hypot(cv2.Sobel(g2, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g2, cv2.CV_32F, 0, 1, ksize=3))
    lap1 = np.abs(cv2.Laplacian(g1, cv2.CV_32F, ksize=3))
    lap2 = np.abs(cv2.Laplacian(g2, cv2.CV_32F, ksize=3))
    can1 = (cv2.Canny(g1_u8, 60, 150) > 0).astype(np.float32)
    can2 = (cv2.Canny(g2_u8, 60, 150) > 0).astype(np.float32)

    # Local SSIM change map on 11x11 window
    k_sz = (11, 11)
    mu1 = cv2.GaussianBlur(g1, k_sz, 2.0)
    mu2 = cv2.GaussianBlur(g2, k_sz, 2.0)
    sig1_sq = np.maximum(0.0, cv2.GaussianBlur(g1 * g1, k_sz, 2.0) - mu1 * mu1)
    sig2_sq = np.maximum(0.0, cv2.GaussianBlur(g2 * g2, k_sz, 2.0) - mu2 * mu2)
    sig12 = cv2.GaussianBlur(g1 * g2, k_sz, 2.0) - mu1 * mu2
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sig12 + c2)) / ((mu1 * mu1 + mu2 * mu2 + c1) * (sig1_sq + sig2_sq + c2) + 1e-6)
    chg = np.clip(1.0 - ssim, 0.0, 2.0)

    chg_ov = chg[ov]
    q50, q80 = float(np.percentile(chg_ov, 50)), float(np.percentile(chg_ov, 80))
    hi = ov & (chg >= q80)
    lo = ov & (chg <= q50)
    if hi.sum() < 20 or lo.sum() < 20:
        hi = ov
        lo = ov

    # DiD features on de-tinted color & structure
    did_vals = []
    for k_name, a1, a2 in [
        ("lum", g1, g2),
        ("sat", sat1, sat2),
        ("exg", exg1, exg2),
        ("rg", rg1, rg2),
        ("br", br1, br2),
        ("lap", lap1, lap2),
        ("can", can1, can2),
    ]:
        if either_gray > 0.5 and k_name in ("sat", "exg", "rg", "br"):
            did_vals.extend([0.0, 0.0, 0.0])
        else:
            m_pre = float(a1[ov].mean())
            m_post = float(a2[ov].mean())
            d_hi = float((a2[hi] - a1[hi]).mean())
            d_lo = float((a2[lo] - a1[lo]).mean())
            did_vals.extend([m_post - m_pre, d_hi - d_lo, float(a2[hi].mean())])

    # Explicit disaster physics indicators on de-tinted overlap
    if either_gray > 0.5:
        charcoal_gain = float(((g2[ov] < 65.0).mean()) - ((g1[ov] < 65.0).mean()))
        hollow_ash_gain = float(((g2[ov] > 120.0).mean() * can2[ov].mean()) - ((g1[ov] > 120.0).mean() * can1[ov].mean()))
        volc_ash_veil = 0.0
        neutral_gray_gain = 0.0
        exg_veg_drop = 0.0
    else:
        ch1 = ((g1 < 65.0) & (sat1 < 0.16))[ov].mean()
        ch2 = ((g2 < 65.0) & (sat2 < 0.16))[ov].mean()
        charcoal_gain = float(ch2 - ch1)
        ha1 = ((g1 > 115.0) & (sat1 < 0.14))[ov].mean() * float(can1[ov].mean())
        ha2 = ((g2 > 115.0) & (sat2 < 0.14))[ov].mean() * float(can2[ov].mean())
        hollow_ash_gain = float(ha2 - ha1)
        ng1 = ((sat1 < 0.12) & (g1 >= 75.0) & (g1 <= 195.0))[ov].mean()
        ng2 = ((sat2 < 0.12) & (g2 >= 75.0) & (g2 <= 195.0))[ov].mean()
        neutral_gray_gain = float(ng2 - ng1)
        lap_lr = float(np.log((lap2[ov].var() + 1.0) / (lap1[ov].var() + 1.0)))
        volc_ash_veil = float(max(0.0, -(float(sat2[ov].mean()) - float(sat1[ov].mean()))) * max(0.0, -lap_lr))
        exg_veg_drop = float(((exg1 > 0.06) & (exg2 < 0.02))[ov].mean())

    # Coherent newly darkened lava/flood component
    dark_new = (ov & ((g2 - g1) < -25.0) & (g2 < 75.0)).astype(np.uint8)
    n_cc, _, stats_cc, _ = cv2.connectedComponentsWithStats(dark_new, connectivity=8)
    lava_dark_cc = float(stats_cc[1:, cv2.CC_STAT_AREA].max() / (ov.sum() + 1.0)) if n_cc > 1 else 0.0

    # Land-only radiative blast metrics (excludes smooth ocean/water pixels!)
    land_mask = ov & ((sob1 >= 14.0) | (sob2 >= 14.0))
    if land_mask.sum() >= 250:
        lc = chg[land_mask]
        lp50, lp90, lp95, lp99 = [float(x) for x in np.percentile(lc, [50, 90, 95, 99])]
        land_p95_p50 = lp95 / (lp50 + 0.05)
        land_p99_p50 = lp99 / (lp50 + 0.05)
        ly, lx = np.where(land_mask)
        top10 = lc >= lp90
        if top10.sum() >= 10:
            cy, cx = float(ly[top10].mean()), float(lx[top10].mean())
            land_spread = float(np.sqrt(ly[top10].var() + lx[top10].var()) / sz)
            dists = np.hypot(ly.astype(np.float32) - cy, lx.astype(np.float32) - cx)
            land_rad_corr = float(np.corrcoef(dists, lc)[0, 1]) if dists.std() > 1e-3 and lc.std() > 1e-4 else 0.0
            if not np.isfinite(land_rad_corr):
                land_rad_corr = 0.0
        else:
            land_spread, land_rad_corr = 0.45, 0.0
        land_frac = float(land_mask.sum() / (ov.sum() + 1.0))
    else:
        land_p95_p50, land_p99_p50, land_spread, land_rad_corr, land_frac = 1.0, 1.0, 0.45, 0.0, 0.0

    # 2D FFT Sensor/GSD & Rotation-Invariant Sunda Stripe Detector
    rad1, sp1, sz1 = _fft_sensor_and_stripe_features(g1_u8, ov)
    rad2, sp2, sz2 = _fft_sensor_and_stripe_features(g2_u8, ov)
    rad_diff_6 = np.array([
        float(rad2[0:2].sum() - rad1[0:2].sum()),
        float(rad2[2:4].sum() - rad1[2:4].sum()),
        float(rad2[4:6].sum() - rad1[4:6].sum()),
        float(rad2[6:8].sum() - rad1[6:8].sum()),
        float(rad2[8:10].sum() - rad1[8:10].sum()),
        float(rad2[10:12].sum() - rad1[10:12].sum()),
    ], dtype=np.float32)

    vec = np.concatenate([
        np.array(did_vals, dtype=np.float32),
        np.array([
            charcoal_gain, hollow_ash_gain, neutral_gray_gain, volc_ash_veil, exg_veg_drop, lava_dark_cc,
            land_p95_p50, land_p99_p50, land_spread, land_rad_corr, land_frac,
            sp1, sp2, sp2 - sp1, sz1, sz2, sz2 - sz1,
        ], dtype=np.float32),
        rad1, rad2, rad_diff_6,
    ])
    return vec, either_gray


def get_detint_phys_names() -> list[str]:
    names = []
    for k in ["lum", "sat", "exg", "rg", "br", "lap", "can"]:
        names.extend([f"dt_delta_{k}", f"dt_did_{k}", f"dt_hi_post_{k}"])
    names.extend([
        "charcoal_gain", "hollow_ash_gain", "neutral_gray_gain", "volc_ash_veil", "exg_veg_drop", "lava_dark_cc",
        "land_p95_p50", "land_p99_p50", "land_spread", "land_rad_corr", "land_frac",
        "stripe_peak_pre", "stripe_peak_post", "stripe_peak_delta",
        "stripe_z_pre", "stripe_z_post", "stripe_z_delta",
    ])
    names.extend([f"fft_pre_b{i}" for i in range(12)])
    names.extend([f"fft_post_b{i}" for i in range(12)])
    names.extend([f"fft_diff_g{i}" for i in range(6)])
    return names


def stage_5a1_detint_phys(sample_ids: list[str], aln) -> Path:
    out_path = CACHE_DIR / "detint_phys_v5.npz"
    names = get_detint_phys_names()
    N = len(sample_ids)
    if out_path.exists():
        d = np.load(out_path)
        if d["features"].shape == (N, len(names)):
            print(f"[Stage 5A-1] Checkpoint exists: {out_path} {d['features'].shape}", flush=True)
            return out_path

    t0 = time.time()
    print(f"[Stage 5A-1] Extracting {len(names)} de-tinted 384px physics + 2D-FFT features for {N} pairs...", flush=True)
    args_list = [
        (sample_ids[i], aln["best_ang"][i], aln["best_dy"][i], aln["best_dx"][i])
        for i in range(N)
    ]
    rows, gflags = [], []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for idx, (vec, gf) in enumerate(ex.map(_compute_detint_phys_single, args_list, chunksize=32)):
            rows.append(vec)
            gflags.append(gf)
            if (idx + 1) % 640 == 0 or (idx + 1) == N:
                print(f"  [5A-1 Phys+FFT] {idx+1}/{N} in {time.time()-t0:.1f}s", flush=True)

    mat = np.nan_to_num(np.vstack(rows).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    np.savez_compressed(
        out_path,
        features=mat,
        feature_names=np.array(names),
        either_gray=np.array(gflags, dtype=np.float32),
    )
    print(f"[Stage 5A-1] Saved {out_path} shape={mat.shape} in {time.time()-t0:.1f}s", flush=True)
    return out_path


def stage_5a2_detint_vitb(sample_ids: list[str], aln) -> Path:
    out_path = CACHE_DIR / "detint_vitb_224.npz"
    N, img_size, grid_size, dim, batch_size = len(sample_ids), 224, 16, 768, 24
    if out_path.exists():
        d = np.load(out_path)
        if d["pmean_post"].shape == (N, dim):
            print(f"[Stage 5A-2] Checkpoint exists: {out_path} {d['pmean_post'].shape}", flush=True)
            return out_path

    t0 = time.time()
    print(f"[Stage 5A-2] Loading dinov2_vitb14_reg (local hub) for De-Tinted Aligned RGB @ {img_size}px...", flush=True)
    hub_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    model = torch.hub.load(str(hub_dir), "dinov2_vitb14_reg", source="local", pretrained=True).eval()

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def _load_pair(idx: int):
        sid = sample_ids[idx]
        folder = FEN_DIR / sid
        with Image.open(folder / "pre.jpg") as im1, Image.open(folder / "post.jpg") as im2:
            r1 = im1.convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
            r2 = warp_post_to_pre(
                im2.convert("RGB"), float(aln["best_ang"][idx]), int(aln["best_dy"][idx]), int(aln["best_dx"][idx]), img_size
            )
        m1 = valid_mask_grid(r1, grid_size)
        m2 = valid_mask_grid(r2, grid_size)
        d1 = detint_clean_rgb(r1, fill_neutral_border=True)
        d2 = detint_clean_rgb(r2, fill_neutral_border=True)
        return tf(d1), tf(d2), m1, m2

    pmean_pre = np.zeros((N, dim), dtype=np.float32)
    pmean_post = np.zeros((N, dim), dtype=np.float32)
    aligned_delta_emb = np.zeros((N, dim), dtype=np.float32)
    topchg_pre = np.zeros((N, dim), dtype=np.float32)
    topchg_post = np.zeros((N, dim), dtype=np.float32)
    topchg_delta = np.zeros((N, dim), dtype=np.float32)
    loc_stats = np.zeros((N, 12), dtype=np.float32)

    with ThreadPoolExecutor(max_workers=4) as ex, torch.inference_mode():
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            bsz = end - i
            loaded = list(ex.map(_load_pair, range(i, end)))
            t1_b = torch.stack([x[0] for x in loaded])
            t2_b = torch.stack([x[1] for x in loaded])

            p1 = l2norm(model.forward_features(t1_b)["x_norm_patchtokens"].numpy().astype(np.float32))
            p2 = l2norm(model.forward_features(t2_b)["x_norm_patchtokens"].numpy().astype(np.float32))
            p1 = p1.reshape(bsz, grid_size, grid_size, dim)
            p2 = p2.reshape(bsz, grid_size, grid_size, dim)

            for j in range(bsz):
                idx = i + j
                st, tc_pre, tc_post, tc_del, p1_ov, p2_ov = _local_3x3_chamfer_pair(
                    p1[j], loaded[j][2], p2[j], loaded[j][3]
                )
                pmean_pre[idx] = l2norm(p1_ov.mean(axis=0), axis=0)
                pmean_post[idx] = l2norm(p2_ov.mean(axis=0), axis=0)
                aligned_delta_emb[idx] = l2norm((p2_ov - p1_ov).mean(axis=0), axis=0)
                topchg_pre[idx] = tc_pre
                topchg_post[idx] = tc_post
                topchg_delta[idx] = tc_del
                loc_stats[idx] = st

            if end % 360 == 0 or end == N:
                el = time.time() - t0
                eta = el / end * (N - end) / 60.0
                print(f"  [5A-2 DeTint ViT-B] {end}/{N} in {el:.1f}s (ETA {eta:.1f} min)", flush=True)

    np.savez_compressed(
        out_path,
        pmean_pre=pmean_pre,
        pmean_post=pmean_post,
        aligned_delta_emb=aligned_delta_emb,
        topchg_pre=topchg_pre,
        topchg_post=topchg_post,
        topchg_delta=topchg_delta,
        loc_stats=loc_stats,
    )
    print(f"[Stage 5A-2] Saved {out_path} in {(time.time()-t0)/60:.1f} min", flush=True)
    del model
    gc.collect()
    return out_path


def _make_changemap_image(r1: Image.Image, r2: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """Build 3-channel Aligned Visual Change-Map [|dLum|, signed_dLum, |dSat|+|dEdge|] on valid overlap."""
    raw1 = np.asarray(r1, dtype=np.float32)
    raw2 = np.asarray(r2, dtype=np.float32)
    ov = (raw1.max(axis=2) > 12.0) & (raw2.max(axis=2) > 12.0) & (raw1.min(axis=2) < 225.0) & (raw2.min(axis=2) < 225.0)

    d1 = np.asarray(detint_clean_rgb(r1, fill_neutral_border=False), dtype=np.float32)
    d2 = np.asarray(detint_clean_rgb(r2, fill_neutral_border=False), dtype=np.float32)
    g1 = cv2.cvtColor(d1.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(d2.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    hsv1 = cv2.cvtColor(d1.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)[:, :, 1] / 255.0
    hsv2 = cv2.cvtColor(d2.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)[:, :, 1] / 255.0
    sob1 = np.hypot(cv2.Sobel(g1, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g1, cv2.CV_32F, 0, 1, ksize=3))
    sob2 = np.hypot(cv2.Sobel(g2, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g2, cv2.CV_32F, 0, 1, ksize=3))

    dl = g2 - g1
    ch0 = np.clip(np.abs(dl) * 2.5, 0.0, 255.0)
    ch1 = np.clip(128.0 + dl * 1.5, 0.0, 255.0)
    ch2 = np.clip(np.abs(hsv2 - hsv1) * 200.0 + np.abs(sob2 - sob1) * 1.2, 0.0, 255.0)

    cmap = np.stack([ch0, ch1, ch2], axis=-1)
    cmap[~ov] = np.array([0.0, 128.0, 0.0], dtype=np.float32)
    m_grid = valid_mask_grid(r1, 16) * valid_mask_grid(r2, 16)
    return Image.fromarray(cmap.astype(np.uint8), mode="RGB"), m_grid


def stage_5a3_changemap_vits(sample_ids: list[str], aln) -> Path:
    out_path = CACHE_DIR / "changemap_vits_224.npz"
    N, img_size, grid_size, dim, batch_size = len(sample_ids), 224, 16, 384, 32
    if out_path.exists():
        d = np.load(out_path)
        if d["emb"].shape == (N, dim * 2):
            print(f"[Stage 5A-3] Checkpoint exists: {out_path} {d['emb'].shape}", flush=True)
            return out_path

    t0 = time.time()
    print(f"[Stage 5A-3] Loading dinov2_vits14 (local hub) for Aligned Visual Change-Map @ {img_size}px...", flush=True)
    hub_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    model = torch.hub.load(str(hub_dir), "dinov2_vits14", source="local", pretrained=True).eval()

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def _load_cm(idx: int):
        sid = sample_ids[idx]
        folder = FEN_DIR / sid
        with Image.open(folder / "pre.jpg") as im1, Image.open(folder / "post.jpg") as im2:
            r1 = im1.convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
            r2 = warp_post_to_pre(
                im2.convert("RGB"), float(aln["best_ang"][idx]), int(aln["best_dy"][idx]), int(aln["best_dx"][idx]), img_size
            )
        cm_im, mg = _make_changemap_image(r1, r2)
        return tf(cm_im), mg

    emb = np.zeros((N, dim * 2), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=4) as ex, torch.inference_mode():
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            bsz = end - i
            loaded = list(ex.map(_load_cm, range(i, end)))
            tb = torch.stack([x[0] for x in loaded])
            feats = model.forward_features(tb)
            cls_t = l2norm(feats["x_norm_clstoken"].numpy().astype(np.float32))
            pt_t = l2norm(feats["x_norm_patchtokens"].numpy().astype(np.float32)).reshape(bsz, grid_size, grid_size, dim)

            for j in range(bsz):
                idx = i + j
                mb = loaded[j][1] > 0.5
                if mb.sum() < 8:
                    mb = np.ones((grid_size, grid_size), dtype=bool)
                pm = l2norm(pt_t[j][mb].mean(axis=0), axis=0)
                emb[idx] = np.concatenate([cls_t[j], pm])

            if end % 640 == 0 or end == N:
                el = time.time() - t0
                eta = el / end * (N - end) / 60.0
                print(f"  [5A-3 ChangeMap ViT-S] {end}/{N} in {el:.1f}s (ETA {eta:.1f} min)", flush=True)

    np.savez_compressed(out_path, emb=emb)
    print(f"[Stage 5A-3] Saved {out_path} in {(time.time()-t0)/60:.1f} min", flush=True)
    del model
    gc.collect()
    return out_path


def main():
    t_all = time.time()
    sample_ids = json.load(open(C4_DIR / "sample_ids.json", "r", encoding="utf-8"))
    ref_ids = pd.read_csv(SAMPLE_SUB)["Id"].tolist()
    assert len(sample_ids) == 3180 and sample_ids == ref_ids
    with open(CACHE_DIR / "sample_ids.json", "w", encoding="utf-8") as f:
        json.dump(sample_ids, f)

    aln = np.load(C4_DIR / "alignment_v4.npz")
    stage_5a1_detint_phys(sample_ids, aln)
    gc.collect()
    stage_5a2_detint_vitb(sample_ids, aln)
    gc.collect()
    stage_5a3_changemap_vits(sample_ids, aln)
    gc.collect()

    spec = {
        "seed": SEED,
        "n_samples": len(sample_ids),
        "de_tinting": "Masked per-channel gray-world gain inversion [0.55, 1.85] with smooth cloud preservation & neutral border fill",
        "alignment_source": "ghani/task2/cache/v4/alignment_v4.npz (4-DOF rigid FFT alignment)",
        "models": [
            {
                "name": "dinov2_vitb14_reg",
                "source": "https://github.com/facebookresearch/dinov2 (hub: dinov2_vitb14_reg)",
                "license": "Apache-2.0",
                "prompting": "None (pure-vision pixels only)",
                "usage": "De-tinted aligned RGB patch embeddings & local 3x3 chamfer change",
            },
            {
                "name": "dinov2_vits14",
                "source": "https://github.com/facebookresearch/dinov2 (hub: dinov2_vits14)",
                "license": "Apache-2.0",
                "prompting": "None (pure-vision pixels only)",
                "usage": "Aligned 3-channel visual change-map spatial geometry embeddings",
            },
        ],
    }
    with open(CACHE_DIR / "pretrained_spec_v5.json", "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    print(f"\nALL v5 PART 1 EXTRACTION STAGES DONE in {(time.time()-t_all)/60:.1f} min!", flush=True)


if __name__ == "__main__":
    main()
