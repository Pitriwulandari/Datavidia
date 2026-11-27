"""Task-2 v4 Part 1: 4-Canonical DINOv2 Rotational+FFT Alignment & High-Res Feature Extraction.

100% Pure-Vision & Compliant (FAQ Task-2 no. 2):
  - No text/point/box prompts, no VLM/LLM/CLIP/SAM, no external labeled data.
  - Deterministic SEED = 42, local hub model loading, checkpointed per stage to conserve RAM.
"""
from __future__ import annotations

import gc
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.special import softmax
from torchvision import transforms

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(12, os.cpu_count() or 12))

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "Task-2"
FEN_DIR = DATA_DIR / "Fenomena"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"
CACHE_DIR = ROOT / "ghani" / "task2" / "cache" / "v4"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def l2norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def to_gray_clahe_rgb(im_rgb: Image.Image) -> Image.Image:
    """Luminance CLAHE stacked to 3 channels: immune to color tint and paired grayscale bias."""
    g = np.asarray(im_rgb.convert("L"), dtype=np.uint8)
    g_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    return Image.fromarray(np.stack([g_eq, g_eq, g_eq], axis=-1), mode="RGB")


def valid_mask_grid(im_rgb: Image.Image, grid_size: int) -> np.ndarray:
    """Compute (grid_size, grid_size) float32 valid mask excluding black triangles & bright white clouds."""
    sz = grid_size * 14
    arr = np.asarray(im_rgb.resize((sz, sz), Image.Resampling.BILINEAR), dtype=np.float32)
    blk = arr.reshape(grid_size, 14, grid_size, 14, 3)
    p_max = blk.max(axis=4).mean(axis=(1, 3))
    p_min = blk.min(axis=4).mean(axis=(1, 3))
    mask = (p_max > 12.0) & (p_min < 220.0)
    if mask.sum() < 8:
        mask = np.ones((grid_size, grid_size), dtype=bool)
    return mask.astype(np.float32)


def warp_post_to_pre(im_post_rgb: Image.Image, ang_deg: float, dy_g16: int, dx_g16: int, out_size: int) -> Image.Image:
    """Rotate post CCW by ang_deg and shift by (dx_g16, dy_g16) in 16-grid units onto out_size black canvas."""
    im_r = im_post_rgb.resize((out_size, out_size), Image.Resampling.BILINEAR)
    im_rot = im_r.rotate(float(ang_deg), resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    dx_px = int(round(dx_g16 * out_size / 16.0))
    dy_px = int(round(dy_g16 * out_size / 16.0))
    if dx_px == 0 and dy_px == 0:
        return im_rot
    canvas = Image.new("RGB", (out_size, out_size), (0, 0, 0))
    canvas.paste(im_rot, (dx_px, dy_px))
    return canvas


def rotate_grid_torch(x: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    """Rotate (B, C, H, W) tensor CCW by angles_deg (shape (B,)) matching PIL.Image.rotate and np.rot90(k=1)."""
    b, c, h, w = x.shape
    rad = torch.deg2rad(angles_deg.to(dtype=torch.float32))
    cos_a = torch.cos(rad)
    sin_a = torch.sin(rad)
    theta = torch.zeros((b, 2, 3), dtype=torch.float32, device=x.device)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    grid = F.affine_grid(theta, (b, c, h, w), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def test_rotation_sign() -> None:
    """Mandatory unit test verifying rotate_grid_torch(+90) and warp_post_to_pre(+90) match np.rot90(k=1)."""
    arr = np.zeros((16, 16), dtype=np.float32)
    arr[2, 12] = 1.0
    ref_rot90 = np.rot90(arr, k=1)
    ref_yx = np.unravel_index(int(np.argmax(ref_rot90)), (16, 16))

    xt = torch.from_numpy(arr[None, None, :, :])
    yt = rotate_grid_torch(xt, torch.tensor([90.0]))[0, 0].numpy()
    got_yx = np.unravel_index(int(np.argmax(yt)), (16, 16))
    assert got_yx == ref_yx, f"rotate_grid_torch(+90) mismatch: {got_yx} vs np.rot90 {ref_yx}"

    rgb_arr = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb_arr[8:12, 48:52] = 255
    pil_im = Image.fromarray(rgb_arr, mode="RGB")
    warped = np.asarray(warp_post_to_pre(pil_im, 90.0, 0, 0, 64))[:, :, 0]
    ref_pil = np.rot90(rgb_arr[:, :, 0], k=1)
    w_yx = np.unravel_index(int(np.argmax(warped)), (64, 64))
    r_yx = np.unravel_index(int(np.argmax(ref_pil)), (64, 64))
    assert abs(w_yx[0] - r_yx[0]) <= 1 and abs(w_yx[1] - r_yx[1]) <= 1, f"warp_post_to_pre mismatch: {w_yx} vs {r_yx}"
    print("[Unit Test] rotate_grid_torch(+90) & warp_post_to_pre(+90) match np.rot90(k=1)!", flush=True)


def _load_5view_align_pair(sid: str, tf224):
    folder = FEN_DIR / sid
    with Image.open(folder / "pre.jpg") as im1, Image.open(folder / "post.jpg") as im2:
        r1 = im1.convert("RGB")
        r2 = im2.convert("RGB")
        m1 = valid_mask_grid(r1, 16)
        m2_0 = valid_mask_grid(r2, 16)
        g1 = to_gray_clahe_rgb(r1)
        g2 = to_gray_clahe_rgb(r2)
        t1 = tf224(g1)
        t2_0 = tf224(g2)
        t2_90 = tf224(g2.rotate(90))
        t2_180 = tf224(g2.rotate(180))
        t2_270 = tf224(g2.rotate(270))
    return t1, t2_0, t2_90, t2_180, t2_270, m1, m2_0


def _score_angles_batched(
    F1: torch.Tensor,
    Fm1: torch.Tensor,
    p2_views: dict[int, torch.Tensor],
    m2_views: dict[int, torch.Tensor],
    ang_list: list[int],
    mask_sh: torch.Tensor,
):
    """Evaluate all angles in ang_list using nearest 90-deg canonical view + batched rotate_grid_torch + FFT."""
    by_canon: dict[int, list[tuple[int, float]]] = {0: [], 90: [], 180: [], 270: []}
    for ang in ang_list:
        ang_mod = ang % 360
        k_rot = (round(ang_mod / 90.0) * 90) % 360
        delta = ((ang_mod - k_rot + 180) % 360) - 180
        by_canon[int(k_rot)].append((ang_mod, float(delta)))

    all_angs = []
    all_scores = []
    all_dys = []
    all_dxs = []
    all_ovf = []

    for k_rot, items in by_canon.items():
        if not items:
            continue
        deltas = torch.tensor([it[1] for it in items], dtype=torch.float32)
        k_cnt = len(items)
        p2_rep = p2_views[k_rot].unsqueeze(0).expand(k_cnt, -1, -1, -1)
        m2_rep = m2_views[k_rot].unsqueeze(0).expand(k_cnt, -1, -1, -1)
        if any(d != 0.0 for _, d in items):
            p2r = rotate_grid_torch(p2_rep, deltas)
            m2r = (rotate_grid_torch(m2_rep, deltas) > 0.5).to(torch.float32)
        else:
            p2r = p2_rep
            m2r = m2_rep

        F2 = torch.fft.rfft2(p2r * m2r, dim=(-2, -1))
        cc = torch.fft.irfft2((F1.unsqueeze(0) * torch.conj(F2)).sum(dim=1), s=(16, 16), dim=(-2, -1))
        Fm2 = torch.fft.rfft2(m2r[:, 0], dim=(-2, -1))
        ov = torch.fft.irfft2(Fm1.unsqueeze(0) * torch.conj(Fm2), s=(16, 16), dim=(-2, -1))

        for idx_k, (ang_mod, _) in enumerate(items):
            ov_k = ov[idx_k]
            valid_sh = mask_sh & (ov_k >= 48.0)
            if not valid_sh.any():
                valid_sh = mask_sh & (ov_k >= 24.0)
            if not valid_sh.any():
                all_angs.append(ang_mod)
                all_scores.append(-1.0)
                all_dys.append(0)
                all_dxs.append(0)
                all_ovf.append(0.0)
                continue
            norm_cc = torch.where(valid_sh, cc[idx_k] / (ov_k + 1e-6), torch.tensor(-1.0))
            flat_idx = int(torch.argmax(norm_cc).item())
            iy, ix = divmod(flat_idx, 16)
            sc = float(norm_cc[iy, ix].item())
            ov_cells = float(ov_k[iy, ix].item())
            dy = iy if iy <= 8 else iy - 16
            dx = ix if ix <= 8 else ix - 16
            all_angs.append(ang_mod)
            all_scores.append(sc)
            all_dys.append(dy)
            all_dxs.append(dx)
            all_ovf.append(ov_cells / 256.0)

    best_i = int(np.argmax(all_scores))
    return (
        all_angs[best_i],
        all_dys[best_i],
        all_dxs[best_i],
        all_scores[best_i],
        all_ovf[best_i],
        all_scores,
    )


def _render_align_audit(sample_ids: list[str], best_ang, best_dy, best_dx, best_sim, z_peak):
    id2i = {s: i for i, s in enumerate(sample_ids)}
    picks = []
    for pref in ["08c7a86c", "dec05312", "1bba29f7", "d3a1e358", "038f53e4", "f4965d34", "3d3b0398", "6062a3a7"]:
        m = [s for s in sample_ids if s.startswith(pref)]
        if m:
            picks.append((pref, id2i[m[0]]))
    # Add 2 lowest z_peak pairs
    low_ord = np.argsort(z_peak)
    for ii in low_ord[:2]:
        picks.append((f"lowZ_{sample_ids[int(ii)][:6]}", int(ii)))

    T = 128
    can = Image.new("RGB", (115 + 3 * (T + 6), len(picks) * (T + 14)), "white")
    dr = ImageDraw.Draw(can)
    for r, (lbl, ii) in enumerate(picks):
        sid = sample_ids[ii]
        with Image.open(FEN_DIR / sid / "pre.jpg") as im1, Image.open(FEN_DIR / sid / "post.jpg") as im2:
            r1 = im1.convert("RGB").resize((T, T), Image.Resampling.BILINEAR)
            r2_raw = im2.convert("RGB").resize((T, T), Image.Resampling.BILINEAR)
            r2_aln = warp_post_to_pre(im2.convert("RGB"), float(best_ang[ii]), int(best_dy[ii]), int(best_dx[ii]), T)
        y0 = r * (T + 14) + 2
        dr.text((4, y0 + 22), f"{lbl}\na={int(best_ang[ii])}d\nsh=({int(best_dy[ii]):+d},{int(best_dx[ii]):+d})\nz={z_peak[ii]:.2f}", fill="black")
        for c_idx, cim in enumerate([r1, r2_raw, r2_aln]):
            can.paste(cim, (115 + c_idx * (T + 6), y0))
    out_p = CACHE_DIR / "align_audit_v4.jpg"
    can.save(out_p, quality=90)
    print(f"[Stage 4A] Saved alignment visual audit: {out_p}", flush=True)


def stage_4a_align(sample_ids: list[str]) -> Path:
    out_path = CACHE_DIR / "alignment_v4.npz"
    N = len(sample_ids)
    if out_path.exists():
        d = np.load(out_path)
        if len(d["best_ang"]) == N:
            print(f"[Stage 4A] Checkpoint exists: {out_path} (N={N})", flush=True)
            return out_path

    test_rotation_sign()
    t0 = time.time()
    print("[Stage 4A] Loading dinov2_vits14 (local hub) for 4-canonical rotational+FFT alignment...", flush=True)
    hub_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    model = torch.hub.load(str(hub_dir), "dinov2_vits14", source="local", pretrained=True).eval()

    tf224 = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    best_ang = np.zeros(N, dtype=np.float32)
    best_dy = np.zeros(N, dtype=np.int32)
    best_dx = np.zeros(N, dtype=np.int32)
    best_sim = np.zeros(N, dtype=np.float32)
    z_peak = np.zeros(N, dtype=np.float32)
    ov_frac = np.zeros(N, dtype=np.float32)

    mask_sh = torch.zeros((16, 16), dtype=torch.bool)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            mask_sh[dy % 16, dx % 16] = True
    coarse_angs = list(range(0, 360, 10))

    BATCH = 16
    with ThreadPoolExecutor(max_workers=4) as ex, torch.inference_mode():
        for i in range(0, N, BATCH):
            b_ids = sample_ids[i:i + BATCH]
            bsz = len(b_ids)
            loaded = list(ex.map(lambda s: _load_5view_align_pair(s, tf224), b_ids))
            t_all = torch.stack(
                [x[0] for x in loaded]
                + [x[1] for x in loaded]
                + [x[2] for x in loaded]
                + [x[3] for x in loaded]
                + [x[4] for x in loaded]
            )
            ptok = model.forward_features(t_all)["x_norm_patchtokens"]
            ptok = F.normalize(ptok, p=2, dim=-1)
            # Reshape to (5, bsz, 16, 16, 384) -> (5, bsz, 384, 16, 16)
            ptok = ptok.view(5, bsz, 16, 16, 384).permute(0, 1, 4, 2, 3).contiguous()

            for j in range(bsz):
                idx = i + j
                p1 = ptok[0, j]  # (384, 16, 16)
                m1 = torch.from_numpy(loaded[j][5]).unsqueeze(0)  # (1, 16, 16)
                m2_0 = loaded[j][6]
                m2_views = {
                    0: torch.from_numpy(m2_0.copy()).unsqueeze(0),
                    90: torch.from_numpy(np.rot90(m2_0, 1).copy()).unsqueeze(0),
                    180: torch.from_numpy(np.rot90(m2_0, 2).copy()).unsqueeze(0),
                    270: torch.from_numpy(np.rot90(m2_0, 3).copy()).unsqueeze(0),
                }
                p2_views = {0: ptok[1, j], 90: ptok[2, j], 180: ptok[3, j], 270: ptok[4, j]}

                F1 = torch.fft.rfft2(p1 * m1, dim=(-2, -1))
                Fm1 = torch.fft.rfft2(m1[0], dim=(-2, -1))

                ang_c, _, _, sc_c, _, sc_list = _score_angles_batched(F1, Fm1, p2_views, m2_views, coarse_angs, mask_sh)
                z_val = (sc_c - float(np.median(sc_list))) / (float(np.std(sc_list)) + 1e-6)

                fine_angs = [(ang_c + d) % 360 for d in range(-8, 9, 2)]
                ang_f, dy_f, dx_f, sc_f, ov_f, _ = _score_angles_batched(F1, Fm1, p2_views, m2_views, fine_angs, mask_sh)

                best_ang[idx] = float(ang_f)
                best_dy[idx] = int(dy_f)
                best_dx[idx] = int(dx_f)
                best_sim[idx] = float(sc_f)
                z_peak[idx] = float(z_val)
                ov_frac[idx] = float(ov_f)

            done = min(i + BATCH, N)
            if done % 320 == 0 or done == N:
                el = time.time() - t0
                eta = el / done * (N - done) / 60.0
                print(
                    f"  [4A Align] {done}/{N} in {el:.1f}s (ETA {eta:.1f} min) | "
                    f"z>2.0={(z_peak[:done] > 2.0).mean()*100:.1f}% | med_sim={np.median(best_sim[:done]):.3f}",
                    flush=True,
                )

    np.savez_compressed(
        out_path,
        best_ang=best_ang,
        best_dy=best_dy,
        best_dx=best_dx,
        best_sim=best_sim,
        z_peak=z_peak,
        ov_frac=ov_frac,
    )
    _render_align_audit(sample_ids, best_ang, best_dy, best_dx, best_sim, z_peak)
    print(f"[Stage 4A] Saved {out_path} in {(time.time()-t0)/60:.1f} min", flush=True)
    del model
    gc.collect()
    return out_path


def _local_ssim_change_map(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """Fast box-filter 11x11 local SSIM change map (1 - SSIM) on uint8 gray images."""
    f1 = g1.astype(np.float32)
    f2 = g2.astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu1 = cv2.blur(f1, (11, 11))
    mu2 = cv2.blur(f2, (11, 11))
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    s1_sq = np.maximum(0.0, cv2.blur(f1 * f1, (11, 11)) - mu1_sq)
    s2_sq = np.maximum(0.0, cv2.blur(f2 * f2, (11, 11)) - mu2_sq)
    s12 = cv2.blur(f1 * f2, (11, 11)) - mu1_mu2
    ssim = ((2.0 * mu1_mu2 + c1) * (2.0 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1_sq + s2_sq + c2) + 1e-6)
    return np.clip(1.0 - ssim, 0.0, 2.0)


def _compute_aligned_phys_single(args):
    sid, ang, dy, dx, aln_sim, aln_z = args
    folder = FEN_DIR / sid
    with Image.open(folder / "pre.jpg") as im1, Image.open(folder / "post.jpg") as im2:
        p1 = im1.convert("RGB").resize((384, 384), Image.Resampling.BILINEAR)
        p2 = warp_post_to_pre(im2.convert("RGB"), float(ang), int(dy), int(dx), 384)

    a1 = np.asarray(p1, dtype=np.float32)
    a2 = np.asarray(p2, dtype=np.float32)

    v1 = (a1.max(axis=2) > 10.0) & (a1.min(axis=2) < 225.0)
    v2 = (a2.max(axis=2) > 10.0) & (a2.min(axis=2) < 225.0)
    ov = v1 & v2
    if ov.mean() < 0.05:
        ov = (a1.max(axis=2) > 10.0) & (a2.max(axis=2) > 10.0)
    if ov.mean() < 0.02:
        ov = np.ones((384, 384), dtype=bool)
    ov_f = float(ov.mean())

    g1_raw = np.asarray(p1.convert("L"), dtype=np.uint8)
    g2_raw = np.asarray(p2.convert("L"), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g1 = clahe.apply(g1_raw)
    g2 = clahe.apply(g2_raw)

    chg_map = _local_ssim_change_map(g1, g2)
    chg_ov = chg_map[ov]
    s_mean = float(chg_ov.mean())
    s_std = float(chg_ov.std())
    p50, p75, p85, p90, p95 = [float(x) for x in np.percentile(chg_ov, [50, 75, 85, 90, 95])]
    s_max = float(chg_ov.max())
    f40 = float((chg_ov > 0.40).mean())
    f60 = float((chg_ov > 0.60).mean())

    # Spatial Damage Geometry
    top_mask = ov & (chg_map >= p85)
    ys, xs = np.where(top_mask)
    if len(ys) >= 4:
        cy, cx = float(ys.mean()), float(xs.mean())
        spread = float(np.sqrt(ys.var() + xs.var()) / 384.0)
    else:
        cy, cx = 192.0, 192.0
        spread = 0.40

    oy, ox = np.where(ov)
    dist_ov = np.hypot(oy.astype(np.float32) - cy, ox.astype(np.float32) - cx)
    if dist_ov.std() > 1e-4 and chg_ov.std() > 1e-4:
        rad_corr = float(np.corrcoef(dist_ov, chg_ov)[0, 1])
    else:
        rad_corr = 0.0
    if not np.isfinite(rad_corr):
        rad_corr = 0.0

    pk_to_p50 = float(p95 / (p50 + 0.05))
    pk_minus_p50 = float(p95 - p50)

    top_96 = cv2.resize(top_mask.astype(np.uint8), (96, 96), interpolation=cv2.INTER_NEAREST)
    n_cc, _, stats_cc, _ = cv2.connectedComponentsWithStats(top_96, connectivity=8)
    if n_cc > 1 and top_96.sum() > 0:
        largest_cc = float(stats_cc[1:, cv2.CC_STAT_AREA].max() / float(top_96.sum()))
    else:
        largest_cc = 0.0

    # Half asymmetry (left vs right, top vs bottom)
    m_l = ov[:, :192]
    m_r = ov[:, 192:]
    m_t = ov[:192, :]
    m_b = ov[192:, :]
    lr_asym = abs(float(chg_map[:, :192][m_l].mean()) - float(chg_map[:, 192:][m_r].mean())) if (m_l.any() and m_r.any()) else 0.0
    tb_asym = abs(float(chg_map[:192, :][m_t].mean()) - float(chg_map[192:, :][m_b].mean())) if (m_t.any() and m_b.any()) else 0.0
    half_asym = float(max(lr_asym, tb_asym))

    # Local 9x9 std for water inundation and contrast
    g1f = g1.astype(np.float32)
    g2f = g2.astype(np.float32)
    lstd1 = np.sqrt(np.maximum(0.0, cv2.blur(g1f * g1f, (9, 9)) - cv2.blur(g1f, (9, 9)) ** 2))
    lstd2 = np.sqrt(np.maximum(0.0, cv2.blur(g2f * g2f, (9, 9)) - cv2.blur(g2f, (9, 9)) ** 2))

    lum1 = 0.299 * a1[:, :, 0] + 0.587 * a1[:, :, 1] + 0.114 * a1[:, :, 2]
    lum2 = 0.299 * a2[:, :, 0] + 0.587 * a2[:, :, 1] + 0.114 * a2[:, :, 2]
    p25_lum2 = float(np.percentile(lum2[ov], 25))
    water_inund = float(((lstd2 < 8.0) & (lum2 < p25_lum2) & (lstd1 > 12.0))[ov].mean())

    # DiD Within-Image Relative Contrast
    hi = ov & (chg_map >= p75)
    lo = ov & (chg_map <= p50)
    if not hi.any():
        hi = ov
    if not lo.any():
        lo = ov

    is_gray1 = (np.abs(a1[:, :, 0] - a1[:, :, 1])[ov].mean() + np.abs(a1[:, :, 1] - a1[:, :, 2])[ov].mean()) < 2.5
    is_gray2 = (np.abs(a2[:, :, 0] - a2[:, :, 1])[ov].mean() + np.abs(a2[:, :, 1] - a2[:, :, 2])[ov].mean()) < 2.5
    either_gray = is_gray1 or is_gray2

    sat1 = cv2.cvtColor(a1.astype(np.uint8), cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32) / 255.0
    sat2 = cv2.cvtColor(a2.astype(np.uint8), cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32) / 255.0
    lap1_abs = np.abs(cv2.Laplacian(g1f, cv2.CV_32F, ksize=3))
    lap2_abs = np.abs(cv2.Laplacian(g2f, cv2.CV_32F, ksize=3))
    can1 = (cv2.Canny(g1, 60, 150) > 0).astype(np.float32)
    can2 = (cv2.Canny(g2, 60, 150) > 0).astype(np.float32)

    def _gw_exg(arr, mask):
        gw = np.empty_like(arr)
        for c in range(3):
            mu = float(arr[:, :, c][mask].mean()) + 1.0
            gw[:, :, c] = arr[:, :, c] / mu * 128.0
        return (2.0 * gw[:, :, 1] - gw[:, :, 0] - gw[:, :, 2]) / (gw[:, :, 0] + gw[:, :, 1] + gw[:, :, 2] + 1.0)

    exg1 = _gw_exg(a1, ov)
    exg2 = _gw_exg(a2, ov)

    did_feats = []
    for k_name, m1_arr, m2_arr in [
        ("lum", lum1, lum2),
        ("sat", sat1, sat2),
        ("lap", lap1_abs, lap2_abs),
        ("can", can1, can2),
        ("lstd", lstd1, lstd2),
        ("exg", exg1, exg2),
    ]:
        if either_gray and k_name in ("sat", "exg"):
            did_feats.extend([0.0, 0.0])
        else:
            d1 = float(m1_arr[hi].mean() - m1_arr[lo].mean())
            d2 = float(m2_arr[hi].mean() - m2_arr[lo].mean())
            did_feats.extend([d2 - d1, d2])

    # Overlap global texture/contrast ratios & deltas (16)
    lap_var_pre = float(cv2.Laplacian(g1f, cv2.CV_32F, ksize=3)[ov].var())
    lap_var_post = float(cv2.Laplacian(g2f, cv2.CV_32F, ksize=3)[ov].var())
    lap_log_ratio = float(np.log((lap_var_post + 1.0) / (lap_var_pre + 1.0)))

    can_pre = float(can1[ov].mean())
    can_post = float(can2[ov].mean())
    can_delta = can_post - can_pre

    sob1 = np.hypot(cv2.Sobel(g1f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g1f, cv2.CV_32F, 0, 1, ksize=3))
    sob2 = np.hypot(cv2.Sobel(g2f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g2f, cv2.CV_32F, 0, 1, ksize=3))
    sob_pre = float(sob1[ov].mean())
    sob_post = float(sob2[ov].mean())
    sob_log_ratio = float(np.log((sob_post + 1.0) / (sob_pre + 1.0)))

    lum_std_pre = float(lum1[ov].std())
    lum_std_post = float(lum2[ov].std())
    p20_post = float(np.percentile(lum2[ov], 20))
    dark_rel_gain = float((lum2[hi] < p20_post).mean() - (lum2[lo] < p20_post).mean())

    vec = [
        s_mean, s_std, p50, p75, p90, p95, s_max, f40, f60,
        spread, rad_corr, pk_to_p50, pk_minus_p50, largest_cc, half_asym, water_inund,
        *did_feats,
        ov_f, float(aln_sim), float(aln_z), float(aln_z > 2.0),
        lap_var_pre, lap_var_post, lap_log_ratio,
        can_pre, can_post, can_delta,
        sob_pre, sob_post, sob_log_ratio,
        lum_std_pre, lum_std_post, dark_rel_gain,
    ]
    return np.array(vec, dtype=np.float32), float(either_gray)


def get_aligned_phys_names() -> list[str]:
    names = [
        "ssim_chg_mean", "ssim_chg_std", "ssim_chg_p50", "ssim_chg_p75", "ssim_chg_p90",
        "ssim_chg_p95", "ssim_chg_max", "ssim_chg_frac40", "ssim_chg_frac60",
        "epicenter_spread", "radial_decay_corr", "peak_to_p50_ssim", "peak_minus_p50_ssim",
        "largest_cc_frac", "half_asym_max", "water_inund_frac",
    ]
    for k in ["lum", "sat", "lap", "can", "lstd", "exg"]:
        names.extend([f"did_{k}", f"post_hilo_{k}"])
    names.extend([
        "ov_frac", "align_sim", "align_z_peak", "align_reliable",
        "lap_var_pre", "lap_var_post", "lap_log_ratio",
        "canny_pre", "canny_post", "canny_delta",
        "sob_pre", "sob_post", "sob_log_ratio",
        "lum_std_pre", "lum_std_post", "dark_rel_gain",
    ])
    return names


def stage_4b_aligned_phys(sample_ids: list[str], aln) -> Path:
    out_path = CACHE_DIR / "aligned_phys_v4.npz"
    N = len(sample_ids)
    names = get_aligned_phys_names()
    if out_path.exists():
        d = np.load(out_path)
        if d["features"].shape == (N, len(names)):
            print(f"[Stage 4B] Checkpoint exists: {out_path} {d['features'].shape}", flush=True)
            return out_path

    t0 = time.time()
    print(f"[Stage 4B] Extracting {len(names)} aligned physical/DiD features for {N} pairs...", flush=True)
    args_list = [
        (sample_ids[i], aln["best_ang"][i], aln["best_dy"][i], aln["best_dx"][i], aln["best_sim"][i], aln["z_peak"][i])
        for i in range(N)
    ]
    rows = []
    gray_flags = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for idx, (vec, gf) in enumerate(ex.map(_compute_aligned_phys_single, args_list, chunksize=32)):
            rows.append(vec)
            gray_flags.append(gf)
            if (idx + 1) % 640 == 0 or (idx + 1) == N:
                print(f"  [4B Phys] {idx+1}/{N} in {time.time()-t0:.1f}s", flush=True)

    mat = np.nan_to_num(np.vstack(rows).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    np.savez_compressed(out_path, features=mat, feature_names=np.array(names), either_gray=np.array(gray_flags, dtype=np.float32))
    print(f"[Stage 4B] Saved {out_path} shape={mat.shape} in {time.time()-t0:.1f}s", flush=True)
    return out_path


def _local_3x3_chamfer_pair(p1_grid: np.ndarray, m1_grid: np.ndarray, p2_grid: np.ndarray, m2_grid: np.ndarray):
    """Compute local 3x3 aligned patch-chamfer stats (12-d) and topchg pre/post/delta embeddings."""
    g = p1_grid.shape[0]
    m1b = m1_grid > 0.5
    m2b = m2_grid > 0.5
    ov = m1b & m2b
    if ov.sum() < 8:
        ov = np.ones((g, g), dtype=bool)
        m1b = np.ones((g, g), dtype=bool)

    loc_chg = np.zeros((g, g), dtype=np.float32)
    oy, ox = np.where(ov)
    for y, x in zip(oy, ox):
        y0, y1 = max(0, y - 1), min(g, y + 2)
        x0, x1 = max(0, x - 1), min(g, x + 2)
        win_m = m1b[y0:y1, x0:x1]
        if win_m.any():
            win_p = p1_grid[y0:y1, x0:x1][win_m]
            max_cos = float((win_p @ p2_grid[y, x]).max())
        else:
            max_cos = float(p1_grid[y, x] @ p2_grid[y, x])
        loc_chg[y, x] = np.clip(1.0 - max_cos, 0.0, 2.0)

    chg_vals = loc_chg[ov]
    p50, p75, p85, p90, p95 = [float(x) for x in np.percentile(chg_vals, [50, 75, 85, 90, 95])]
    top_sel = chg_vals >= p85
    if top_sel.sum() >= 2:
        pts_y = oy[top_sel].astype(np.float32)
        pts_x = ox[top_sel].astype(np.float32)
        cy, cx = float(pts_y.mean()), float(pts_x.mean())
        spread = float(np.sqrt(pts_y.var() + pts_x.var()) / float(g))
    else:
        cy, cx = g / 2.0, g / 2.0
        spread = 0.45

    dists = np.hypot(oy.astype(np.float32) - cy, ox.astype(np.float32) - cx)
    if dists.std() > 1e-4 and chg_vals.std() > 1e-4:
        rad_corr = float(np.corrcoef(dists, chg_vals)[0, 1])
    else:
        rad_corr = 0.0
    if not np.isfinite(rad_corr):
        rad_corr = 0.0

    stats = np.array([
        float(chg_vals.mean()), float(chg_vals.std()),
        p50, p75, p90, p95, float(chg_vals.max()),
        float((chg_vals > 0.30).mean()), float((chg_vals > 0.45).mean()),
        spread, rad_corr, float(p95 - p50),
    ], dtype=np.float32)

    w = softmax(10.0 * chg_vals).astype(np.float32)[:, None]
    p1_ov = p1_grid[ov]
    p2_ov = p2_grid[ov]
    topchg_pre = l2norm((w * p1_ov).sum(axis=0), axis=0).astype(np.float32)
    topchg_post = l2norm((w * p2_ov).sum(axis=0), axis=0).astype(np.float32)
    topchg_delta = l2norm(topchg_post - topchg_pre, axis=0).astype(np.float32)
    return stats, topchg_pre, topchg_post, topchg_delta, p1_ov, p2_ov


def _extract_aligned_dino_stage(
    stage_tag: str,
    model_name: str,
    out_name: str,
    img_size: int,
    grid_size: int,
    dim: int,
    batch_size: int,
    sample_ids: list[str],
    aln,
) -> Path:
    out_path = CACHE_DIR / out_name
    N = len(sample_ids)
    if out_path.exists():
        d = np.load(out_path)
        if d["pmean_post"].shape == (N, dim):
            print(f"[{stage_tag}] Checkpoint exists: {out_path} {d['pmean_post'].shape}", flush=True)
            return out_path

    t0 = time.time()
    print(f"[{stage_tag}] Loading {model_name} (local hub) @ {img_size}x{img_size} ({grid_size}x{grid_size} grid)...", flush=True)
    hub_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    model = torch.hub.load(str(hub_dir), model_name, source="local", pretrained=True).eval()

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
            r2 = warp_post_to_pre(im2.convert("RGB"), float(aln["best_ang"][idx]), int(aln["best_dy"][idx]), int(aln["best_dx"][idx]), img_size)
        m1 = valid_mask_grid(r1, grid_size)
        m2 = valid_mask_grid(r2, grid_size)
        return tf(to_gray_clahe_rgb(r1)), tf(to_gray_clahe_rgb(r2)), m1, m2

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

            if end % 320 == 0 or end == N:
                el = time.time() - t0
                eta = el / end * (N - end) / 60.0
                print(f"  [{stage_tag}] {end}/{N} in {el:.1f}s (ETA {eta:.1f} min)", flush=True)

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
    print(f"[{stage_tag}] Saved {out_path} in {(time.time()-t0)/60:.1f} min", flush=True)
    del model
    gc.collect()
    return out_path


def main():
    t_all = time.time()
    sample_ids = pd.read_csv(SAMPLE_SUB)["Id"].tolist()
    assert len(sample_ids) == 3180
    with open(CACHE_DIR / "sample_ids.json", "w", encoding="utf-8") as f:
        json.dump(sample_ids, f)

    spec = {
        "seed": SEED,
        "n_samples": len(sample_ids),
        "alignment": "4-canonical DINOv2 ViT-S/14 Gray-CLAHE coarse(10deg)-to-fine(2deg) FFT rotational+shift alignment (zero center crop)",
        "models": [
            {
                "name": "dinov2_vits14",
                "source": "https://github.com/facebookresearch/dinov2 (hub: dinov2_vits14)",
                "paper": "Oquab et al. (2023) DINOv2: Learning Robust Visual Features without Supervision",
                "license": "Apache-2.0",
                "prompting": "None (pure-vision pixels only)",
                "input_sizes": [224, 336],
            },
            {
                "name": "dinov2_vitb14_reg",
                "source": "https://github.com/facebookresearch/dinov2 (hub: dinov2_vitb14_reg)",
                "paper": "Darcet et al. (2023) Vision Transformers Need Registers; Oquab et al. (2023) DINOv2",
                "license": "Apache-2.0",
                "prompting": "None (pure-vision pixels only)",
                "input_sizes": [224],
            },
        ],
    }
    with open(CACHE_DIR / "pretrained_spec_v4.json", "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    aln_path = stage_4a_align(sample_ids)
    aln = np.load(aln_path)
    stage_4b_aligned_phys(sample_ids, aln)
    _extract_aligned_dino_stage(
        "Stage 4C ViT-S/14 336", "dinov2_vits14", "aligned_vits_336.npz", 336, 24, 384, 16, sample_ids, aln
    )
    _extract_aligned_dino_stage(
        "Stage 4D ViT-B/14-reg 224", "dinov2_vitb14_reg", "aligned_vitb_224.npz", 224, 16, 768, 24, sample_ids, aln
    )

    print(f"\nALL v4 PART 1 EXTRACTION STAGES DONE in {(time.time()-t_all)/60:.1f} min!", flush=True)


if __name__ == "__main__":
    main()
