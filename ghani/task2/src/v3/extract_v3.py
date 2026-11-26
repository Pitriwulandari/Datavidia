"""Task-2 v3 Feature Extraction: Pure-Vision Multi-Backbone + De-Augmentation + Alignment-Free Patch-Chamfer.

100% Compliant with FAQ Task-2 no. 2:
  - Pure-vision pretrained backbones only (no text prompts, no VLM/CLIP/SAM, no external labeled data).
  - Deterministic SEED = 42, checkpointed per stage to conserve RAM (~3.1 GB avail).
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
import torchvision.models as tv
from PIL import Image
from scipy.special import softmax
from scipy.stats import skew
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
CACHE_DIR = ROOT / "ghani" / "task2" / "cache" / "v3"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

IMAGENET_TF = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def center_crop_70(im_rgb: Image.Image) -> Image.Image:
    """Crop 15% margin on all 4 sides (70% center square) to remove rotation black triangles."""
    w, h = im_rgb.size
    mw, mh = int(0.15 * w), int(0.15 * h)
    return im_rgb.crop((mw, mh, w - mw, h - mh))


def to_wb_rgb(im_crop: Image.Image) -> Image.Image:
    """Per-channel robust standardization inside valid non-black, non-cloud pixels to remove tint."""
    arr = np.asarray(im_crop, dtype=np.float32)
    valid = (arr.max(axis=2) > 8.0) & (arr.min(axis=2) < 225.0)
    if valid.mean() < 0.10:
        valid = np.ones(arr.shape[:2], dtype=bool)
    out = np.empty_like(arr)
    for c in range(3):
        vals = arr[:, :, c][valid]
        mu = float(vals.mean())
        sd = float(vals.std()) + 1e-5
        out[:, :, c] = np.clip((arr[:, :, c] - mu) / sd * 48.0 + 128.0, 0.0, 255.0)
    return Image.fromarray(out.astype(np.uint8), mode="RGB")


def to_gray_clahe_rgb(im_crop: Image.Image) -> Image.Image:
    """Luminance CLAHE stacked to 3 channels: immune to color tint and paired grayscale bias."""
    g = np.asarray(im_crop.convert("L"), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_eq = clahe.apply(g)
    return Image.fromarray(np.stack([g_eq, g_eq, g_eq], axis=-1), mode="RGB")


def patch_valid_mask_16x16(im_crop: Image.Image) -> np.ndarray:
    """Compute (16, 16) bool mask excluding black no-data patches and bright white clouds."""
    im224 = im_crop.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.asarray(im224, dtype=np.float32)
    blk = arr.reshape(16, 14, 16, 14, 3)
    p_max = blk.max(axis=4).mean(axis=(1, 3))
    p_min = blk.min(axis=4).mean(axis=(1, 3))
    mask = (p_max > 12.0) & (p_min < 220.0)
    if mask.sum() < 16:
        mask = np.ones((16, 16), dtype=bool)
    return mask


def l2norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def _compute_single_phys(sid: str) -> tuple[np.ndarray, np.ndarray]:
    folder = FEN_DIR / sid
    with Image.open(folder / "pre.jpg") as im1, Image.open(folder / "post.jpg") as im2:
        r1 = im1.convert("RGB")
        r2 = im2.convert("RGB")
        # Raw rotation-corner audit flag (on 64x64 full image, NOT used as clustering feature)
        def _rot_flag(im_full: Image.Image) -> float:
            a64 = np.asarray(im_full.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32)
            corners = np.concatenate([
                a64[:4, :4].reshape(-1, 3), a64[:4, -4:].reshape(-1, 3),
                a64[-4:, :4].reshape(-1, 3), a64[-4:, -4:].reshape(-1, 3),
            ])
            return float((corners.max(axis=1) < 10.0).mean() > 0.25)
        rot_pre = _rot_flag(r1)
        rot_post = _rot_flag(r2)
        c_pre = center_crop_70(r1).resize((256, 256), Image.Resampling.BILINEAR)
        c_post = center_crop_70(r2).resize((256, 256), Image.Resampling.BILINEAR)

    a_pre = np.asarray(c_pre, dtype=np.float32)
    a_post = np.asarray(c_post, dtype=np.float32)

    def _side_stats(a: np.ndarray, im_c: Image.Image):
        ch_diff = float(np.abs(a[:, :, 0] - a[:, :, 1]).mean() + np.abs(a[:, :, 1] - a[:, :, 2]).mean())
        is_gray = 1.0 if ch_diff < 2.5 else 0.0
        blk = (a.max(axis=2) <= 8.0)
        cld = (a.min(axis=2) >= 225.0)
        valid = (~blk) & (~cld)
        if valid.mean() < 0.05:
            valid = np.ones(a.shape[:2], dtype=bool)

        lum = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
        lv = lum[valid]
        l_mean = float(lv.mean())
        l_std = float(lv.std())
        l_p10, l_p50, l_p90 = [float(x) for x in np.percentile(lv, [10, 50, 90])]
        f_dark = float((lv < 55.0).mean())
        f_vdark = float((lv < 35.0).mean())
        f_bright = float((lv > 195.0).mean())

        # Texture & structure on Gray-CLAHE
        g_clahe = np.asarray(to_gray_clahe_rgb(im_c).convert("L"), dtype=np.float32)
        gx = cv2.Sobel(g_clahe, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g_clahe, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.hypot(gx, gy)[valid]
        sob_mean = float(grad.mean())
        sob_std = float(grad.std())

        lap = cv2.Laplacian(g_clahe, cv2.CV_32F, ksize=3)[valid]
        lap_var = float(lap.var())

        edges = cv2.Canny(g_clahe.astype(np.uint8), 60, 150) > 0
        canny_dens = float(edges[valid].mean())

        # Local 8x8 block luminance variance
        g_blk = g_clahe.reshape(32, 8, 32, 8)
        v_blk = valid.reshape(32, 8, 32, 8).mean(axis=(1, 3)) > 0.5
        b_vars = g_blk.var(axis=(1, 3))
        if v_blk.any():
            bv = b_vars[v_blk]
        else:
            bv = b_vars.ravel()
        bvar_mean = float(bv.mean())
        bvar_std = float(bv.std())

        # Relative color contrast on white-balanced RGB
        wb = np.asarray(to_wb_rgb(im_c), dtype=np.float32)
        exg = (2.0 * wb[:, :, 1] - wb[:, :, 0] - wb[:, :, 2])[valid]
        exg_std = float(exg.std())
        exg_skew = float(skew(exg)) if exg_std > 1e-4 else 0.0

        hsv = cv2.cvtColor(wb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        sat = hsv[:, :, 1][valid] / 255.0
        val = hsv[:, :, 2][valid] / 255.0
        sat_mean = float(sat.mean())
        sat_std = float(sat.std())
        low_sat = float((sat < 0.15).mean())
        dark_low_sat = float(((sat < 0.15) & (val < 0.30)).mean())

        return {
            "valid_frac": float(valid.mean()),
            "cloud_frac": float(cld.mean()),
            "blk_frac": float(blk.mean()),
            "is_gray": is_gray,
            "lum_mean": l_mean, "lum_std": l_std,
            "lum_p10": l_p10, "lum_p50": l_p50, "lum_p90": l_p90,
            "dark_frac": f_dark, "vdark_frac": f_vdark, "bright_frac": f_bright,
            "sob_mean": sob_mean, "sob_std": sob_std,
            "lap_var": lap_var, "canny_dens": canny_dens,
            "bvar_mean": bvar_mean, "bvar_std": bvar_std,
            "exg_std": exg_std, "exg_skew": exg_skew,
            "sat_mean": sat_mean, "sat_std": sat_std,
            "low_sat": low_sat, "dark_low_sat": dark_low_sat,
        }

    sp = _side_stats(a_pre, c_pre)
    sq = _side_stats(a_post, c_post)
    either_gray = (sp["is_gray"] > 0.5) or (sq["is_gray"] > 0.5)

    vec = [
        sp["valid_frac"], sq["valid_frac"],
        sp["cloud_frac"], sq["cloud_frac"],
        sp["blk_frac"], sq["blk_frac"],
        sp["is_gray"], sq["is_gray"],
    ]
    lum_keys = ["lum_mean", "lum_std", "lum_p10", "lum_p50", "lum_p90", "dark_frac", "vdark_frac", "bright_frac"]
    for k in lum_keys:
        vec.extend([sp[k], sq[k], sq[k] - sp[k]])

    tex_keys = ["sob_mean", "sob_std", "lap_var", "canny_dens", "bvar_mean", "bvar_std"]
    for k in tex_keys:
        vec.extend([sp[k], sq[k], sq[k] - sp[k], (sq[k] + 1e-4) / (sp[k] + 1e-4)])

    col_keys = ["exg_std", "exg_skew", "sat_mean", "sat_std", "low_sat", "dark_low_sat"]
    for k in col_keys:
        delta = 0.0 if either_gray else (sq[k] - sp[k])
        vec.extend([sp[k], sq[k], delta])

    return np.array(vec, dtype=np.float32), np.array([rot_pre, rot_post, sp["is_gray"], sq["is_gray"]], dtype=np.float32)


def get_phys_feature_names() -> list[str]:
    names = [
        "valid_frac_pre", "valid_frac_post",
        "cloud_frac_pre", "cloud_frac_post",
        "blk_frac_pre", "blk_frac_post",
        "is_gray_pre", "is_gray_post",
    ]
    for k in ["lum_mean", "lum_std", "lum_p10", "lum_p50", "lum_p90", "dark_frac", "vdark_frac", "bright_frac"]:
        names.extend([f"{k}_pre", f"{k}_post", f"{k}_delta"])
    for k in ["sob_mean", "sob_std", "lap_var", "canny_dens", "bvar_mean", "bvar_std"]:
        names.extend([f"{k}_pre", f"{k}_post", f"{k}_delta", f"{k}_ratio"])
    for k in ["exg_std", "exg_skew", "sat_mean", "sat_std", "low_sat", "dark_low_sat"]:
        names.extend([f"{k}_pre", f"{k}_post", f"{k}_delta"])
    return names


def stage_3a_phys(sample_ids: list[str]) -> Path:
    out_path = CACHE_DIR / "phys_v3.npz"
    if out_path.exists():
        d = np.load(out_path)
        if d["features"].shape[0] == len(sample_ids):
            print(f"[Stage 3A] Checkpoint exists: {out_path} {d['features'].shape}", flush=True)
            return out_path

    t0 = time.time()
    print(f"[Stage 3A] Extracting physical & texture stats for {len(sample_ids)} pairs...", flush=True)
    rows = []
    audits = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for idx, (vec, aud) in enumerate(ex.map(_compute_single_phys, sample_ids, chunksize=32)):
            rows.append(vec)
            audits.append(aud)
            if (idx + 1) % 640 == 0 or (idx + 1) == len(sample_ids):
                print(f"  [3A] {idx+1}/{len(sample_ids)} in {time.time()-t0:.1f}s", flush=True)
    mat = np.vstack(rows).astype(np.float32)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    aud_mat = np.vstack(audits).astype(np.float32)
    names = np.array(get_phys_feature_names())
    np.savez_compressed(out_path, features=mat, feature_names=names, audit_flags=aud_mat)
    print(f"[Stage 3A] Saved {out_path} shape={mat.shape} in {time.time()-t0:.1f}s", flush=True)
    return out_path


def _chamfer_pair_stats(ptok_pre: np.ndarray, m_pre_flat: np.ndarray,
                        ptok_post: np.ndarray, m_post_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute 9 alignment-free Chamfer change scalars + soft change-weighted post patch embedding."""
    p_valid = ptok_pre[m_pre_flat]
    q_valid = ptok_post[m_post_flat]
    sim_mat = q_valid @ p_valid.T  # (Nq, Np)
    chg = np.clip(1.0 - sim_mat.max(axis=1), 0.0, 2.0)  # (Nq,)

    p50, p75, p90 = np.percentile(chg, [50, 75, 90])
    # Spatial concentration on 16x16 grid for top 15% changed valid patches
    valid_yx = np.argwhere(m_post_flat.reshape(16, 16))
    thr = np.percentile(chg, 85)
    top_sel = chg >= thr
    if top_sel.sum() >= 2:
        pts = valid_yx[top_sel].astype(np.float32)
        spread = float(np.sqrt(pts[:, 0].var() + pts[:, 1].var()) / 16.0)
    else:
        spread = 0.45

    stats = np.array([
        float(chg.mean()), float(chg.std()),
        float(p50), float(p75), float(p90), float(chg.max()),
        float((chg > 0.35).mean()), float((chg > 0.45).mean()),
        spread,
    ], dtype=np.float32)

    w = softmax(10.0 * chg).astype(np.float32)[:, None]
    chg_emb = (w * q_valid).sum(axis=0)
    chg_emb = l2norm(chg_emb, axis=0).astype(np.float32)
    return stats, chg_emb


def _load_crop_wb_pair(sid: str):
    folder = FEN_DIR / sid
    with Image.open(folder / "pre.jpg") as im1, Image.open(folder / "post.jpg") as im2:
        c1 = center_crop_70(im1.convert("RGB"))
        c2 = center_crop_70(im2.convert("RGB"))
    m1 = patch_valid_mask_16x16(c1).ravel()
    m2 = patch_valid_mask_16x16(c2).ravel()
    t1 = IMAGENET_TF(to_wb_rgb(c1))
    t2 = IMAGENET_TF(to_wb_rgb(c2))
    return t1, t2, m1, m2


def stage_3b_vitb_wb(sample_ids: list[str]) -> Path:
    out_path = CACHE_DIR / "vitb_wb.npz"
    if out_path.exists():
        d = np.load(out_path)
        if d["cls_post"].shape == (len(sample_ids), 768):
            print(f"[Stage 3B] Checkpoint exists: {out_path} {d['cls_post'].shape}", flush=True)
            return out_path

    t0 = time.time()
    print("[Stage 3B] Loading dinov2_vitb14_reg (local hub)...", flush=True)
    hub_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    model = torch.hub.load(str(hub_dir), "dinov2_vitb14_reg", source="local", pretrained=True).eval()

    N = len(sample_ids)
    cls_pre = np.zeros((N, 768), dtype=np.float32)
    cls_post = np.zeros((N, 768), dtype=np.float32)
    pmean_pre = np.zeros((N, 768), dtype=np.float32)
    pmean_post = np.zeros((N, 768), dtype=np.float32)
    changed_emb = np.zeros((N, 768), dtype=np.float32)
    chamfer_stats = np.zeros((N, 9), dtype=np.float32)

    BATCH = 32
    with ThreadPoolExecutor(max_workers=4) as ex, torch.inference_mode():
        for i in range(0, N, BATCH):
            b_ids = sample_ids[i:i + BATCH]
            loaded = list(ex.map(_load_crop_wb_pair, b_ids))
            t1_b = torch.stack([x[0] for x in loaded])
            t2_b = torch.stack([x[1] for x in loaded])
            m1_b = [x[2] for x in loaded]
            m2_b = [x[3] for x in loaded]

            out1 = model.forward_features(t1_b)
            out2 = model.forward_features(t2_b)

            c1 = l2norm(out1["x_norm_clstoken"].numpy().astype(np.float32))
            c2 = l2norm(out2["x_norm_clstoken"].numpy().astype(np.float32))
            p1 = l2norm(out1["x_norm_patchtokens"].numpy().astype(np.float32))
            p2 = l2norm(out2["x_norm_patchtokens"].numpy().astype(np.float32))

            bsz = len(b_ids)
            cls_pre[i:i + bsz] = c1
            cls_post[i:i + bsz] = c2
            for j in range(bsz):
                idx = i + j
                pmean_pre[idx] = l2norm(p1[j][m1_b[j]].mean(axis=0), axis=0)
                pmean_post[idx] = l2norm(p2[j][m2_b[j]].mean(axis=0), axis=0)
                st, ce = _chamfer_pair_stats(p1[j], m1_b[j], p2[j], m2_b[j])
                chamfer_stats[idx] = st
                changed_emb[idx] = ce

            done = min(i + BATCH, N)
            if done % 320 == 0 or done == N:
                el = time.time() - t0
                eta = el / done * (N - done) / 60.0
                print(f"  [3B ViT-B/14-reg WB] {done}/{N} in {el:.1f}s (ETA {eta:.1f} min)", flush=True)

    np.savez_compressed(
        out_path,
        cls_pre=cls_pre, cls_post=cls_post,
        pmean_pre=pmean_pre, pmean_post=pmean_post,
        changed_emb=changed_emb, chamfer_stats=chamfer_stats,
    )
    print(f"[Stage 3B] Saved {out_path} in {(time.time()-t0)/60:.1f} min", flush=True)
    del model
    gc.collect()
    return out_path


def _load_crop_gray_pair(sid: str):
    folder = FEN_DIR / sid
    with Image.open(folder / "pre.jpg") as im1, Image.open(folder / "post.jpg") as im2:
        c1 = center_crop_70(im1.convert("RGB"))
        c2 = center_crop_70(im2.convert("RGB"))
    g1 = to_gray_clahe_rgb(c1)
    g2 = to_gray_clahe_rgb(c2)
    return IMAGENET_TF(g1), IMAGENET_TF(g2)


def stage_3c_cnxb_gray(sample_ids: list[str]) -> Path:
    out_path = CACHE_DIR / "cnxb_gray.npz"
    if out_path.exists():
        d = np.load(out_path)
        if d["post"].shape == (len(sample_ids), 1024):
            print(f"[Stage 3C] Checkpoint exists: {out_path} {d['post'].shape}", flush=True)
            return out_path

    t0 = time.time()
    print("[Stage 3C] Loading convnext_base (IMAGENET1K_V1)...", flush=True)
    model = tv.convnext_base(weights=tv.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    model.classifier[2] = torch.nn.Identity()
    model.eval()

    N = len(sample_ids)
    pre_arr = np.zeros((N, 1024), dtype=np.float32)
    post_arr = np.zeros((N, 1024), dtype=np.float32)

    BATCH = 32
    with ThreadPoolExecutor(max_workers=4) as ex, torch.inference_mode():
        for i in range(0, N, BATCH):
            b_ids = sample_ids[i:i + BATCH]
            loaded = list(ex.map(_load_crop_gray_pair, b_ids))
            t1_b = torch.stack([x[0] for x in loaded])
            t2_b = torch.stack([x[1] for x in loaded])

            e1 = l2norm(model(t1_b).numpy().astype(np.float32))
            e2 = l2norm(model(t2_b).numpy().astype(np.float32))
            bsz = len(b_ids)
            pre_arr[i:i + bsz] = e1
            post_arr[i:i + bsz] = e2

            done = min(i + BATCH, N)
            if done % 320 == 0 or done == N:
                el = time.time() - t0
                eta = el / done * (N - done) / 60.0
                print(f"  [3C ConvNeXt-Base Gray] {done}/{N} in {el:.1f}s (ETA {eta:.1f} min)", flush=True)

    np.savez_compressed(out_path, pre=pre_arr, post=post_arr)
    print(f"[Stage 3C] Saved {out_path} in {(time.time()-t0)/60:.1f} min", flush=True)
    del model
    gc.collect()
    return out_path


def _load_crop_gray_rot2_pair(sid: str):
    folder = FEN_DIR / sid
    with Image.open(folder / "pre.jpg") as im1, Image.open(folder / "post.jpg") as im2:
        c1 = center_crop_70(im1.convert("RGB"))
        c2 = center_crop_70(im2.convert("RGB"))
    m1 = patch_valid_mask_16x16(c1)
    m2 = patch_valid_mask_16x16(c2)
    m1_0 = m1.ravel()
    m1_90 = np.rot90(m1, k=1).ravel()  # matches PIL.Image.rotate(90) CCW
    m2_0 = m2.ravel()
    m2_90 = np.rot90(m2, k=1).ravel()

    g1 = to_gray_clahe_rgb(c1)
    g2 = to_gray_clahe_rgb(c2)
    return (
        IMAGENET_TF(g1), IMAGENET_TF(g1.rotate(90)),
        IMAGENET_TF(g2), IMAGENET_TF(g2.rotate(90)),
        m1_0, m1_90, m2_0, m2_90,
    )


def stage_3d_vits_gray_rot2(sample_ids: list[str]) -> Path:
    out_path = CACHE_DIR / "vits_gray_rot2.npz"
    if out_path.exists():
        d = np.load(out_path)
        if d["cls_post"].shape == (len(sample_ids), 384):
            print(f"[Stage 3D] Checkpoint exists: {out_path} {d['cls_post'].shape}", flush=True)
            return out_path

    t0 = time.time()
    print("[Stage 3D] Loading dinov2_vits14 (local hub) for 2-rot Gray-CLAHE + Multi-Rot Chamfer...", flush=True)
    hub_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    model = torch.hub.load(str(hub_dir), "dinov2_vits14", source="local", pretrained=True).eval()

    N = len(sample_ids)
    cls_pre = np.zeros((N, 384), dtype=np.float32)
    cls_post = np.zeros((N, 384), dtype=np.float32)
    pmean_pre = np.zeros((N, 384), dtype=np.float32)
    pmean_post = np.zeros((N, 384), dtype=np.float32)
    changed_emb = np.zeros((N, 384), dtype=np.float32)
    chamfer_stats = np.zeros((N, 9), dtype=np.float32)

    BATCH = 16  # 16 pairs * 4 views = 64 images/batch
    with ThreadPoolExecutor(max_workers=4) as ex, torch.inference_mode():
        for i in range(0, N, BATCH):
            b_ids = sample_ids[i:i + BATCH]
            bsz = len(b_ids)
            loaded = list(ex.map(_load_crop_gray_rot2_pair, b_ids))
            t_all = torch.stack(
                [x[0] for x in loaded] + [x[1] for x in loaded] +
                [x[2] for x in loaded] + [x[3] for x in loaded]
            )
            out = model.forward_features(t_all)
            c_all = l2norm(out["x_norm_clstoken"].numpy().astype(np.float32))
            p_all = l2norm(out["x_norm_patchtokens"].numpy().astype(np.float32))

            c1_0, c1_90 = c_all[:bsz], c_all[bsz:2*bsz]
            c2_0, c2_90 = c_all[2*bsz:3*bsz], c_all[3*bsz:]
            p1_0, p1_90 = p_all[:bsz], p_all[bsz:2*bsz]
            p2_0, p2_90 = p_all[2*bsz:3*bsz], p_all[3*bsz:]

            cls_pre[i:i + bsz] = l2norm(0.5 * (c1_0 + c1_90))
            cls_post[i:i + bsz] = l2norm(0.5 * (c2_0 + c2_90))

            for j in range(bsz):
                idx = i + j
                m1_0, m1_90, m2_0, m2_90 = loaded[j][4], loaded[j][5], loaded[j][6], loaded[j][7]
                pm1 = 0.5 * (l2norm(p1_0[j][m1_0].mean(axis=0), axis=0) +
                             l2norm(p1_90[j][m1_90].mean(axis=0), axis=0))
                pm2 = 0.5 * (l2norm(p2_0[j][m2_0].mean(axis=0), axis=0) +
                             l2norm(p2_90[j][m2_90].mean(axis=0), axis=0))
                pmean_pre[idx] = l2norm(pm1, axis=0)
                pmean_post[idx] = l2norm(pm2, axis=0)

                # Multi-rotation Chamfer: post(0) valid patches vs pre(0) U pre(90) valid patches
                pre_pool = np.vstack([p1_0[j][m1_0], p1_90[j][m1_90]])
                all_true = np.ones(len(pre_pool), dtype=bool)
                st, ce = _chamfer_pair_stats(pre_pool, all_true, p2_0[j], m2_0)
                chamfer_stats[idx] = st
                changed_emb[idx] = ce

            done = min(i + BATCH, N)
            if done % 320 == 0 or done == N:
                el = time.time() - t0
                eta = el / done * (N - done) / 60.0
                print(f"  [3D ViT-S/14 Gray-Rot2] {done}/{N} in {el:.1f}s (ETA {eta:.1f} min)", flush=True)

    np.savez_compressed(
        out_path,
        cls_pre=cls_pre, cls_post=cls_post,
        pmean_pre=pmean_pre, pmean_post=pmean_post,
        changed_emb=changed_emb, chamfer_stats=chamfer_stats,
    )
    print(f"[Stage 3D] Saved {out_path} in {(time.time()-t0)/60:.1f} min", flush=True)
    del model
    gc.collect()
    return out_path


def verify_all(sample_ids: list[str]):
    print("\n=== VERIFICATION & DE-AUGMENTATION CHECK ===", flush=True)
    phys = np.load(CACHE_DIR / "phys_v3.npz")
    vitb = np.load(CACHE_DIR / "vitb_wb.npz")
    cnxb = np.load(CACHE_DIR / "cnxb_gray.npz")
    vits = np.load(CACHE_DIR / "vits_gray_rot2.npz")

    names = list(phys["feature_names"])
    pf = phys["features"]
    is_gray = (pf[:, names.index("is_gray_pre")] > 0.5) & (pf[:, names.index("is_gray_post")] > 0.5)
    g_idx = np.where(is_gray)[0]
    c_idx = np.where(~is_gray)[0]

    for tag, d, k_pre, k_post in [
        ("ViT-B/14-reg WB (pmean)", vitb, "pmean_pre", "pmean_post"),
        ("ViT-B/14-reg WB (cls)",   vitb, "cls_pre",   "cls_post"),
        ("ConvNeXt-Base Gray",      cnxb, "pre",       "post"),
        ("ViT-S/14 Gray-Rot2 (pm)", vits, "pmean_pre", "pmean_post"),
    ]:
        P, Q = d[k_pre], d[k_post]
        assert P.shape[0] == len(sample_ids) and Q.shape[0] == len(sample_ids)
        assert np.isfinite(P).all() and np.isfinite(Q).all()
        same_cos = float((P * Q).sum(axis=1).mean())
        gg = float((Q[g_idx] @ Q[g_idx].T)[~np.eye(len(g_idx), dtype=bool)].mean())
        gc_val = float((Q[g_idx] @ Q[c_idx].T).mean())
        print(f"  {tag:26s} | shape={Q.shape} | mean_cos(pre,post)={same_cos:.3f} | gray_bias(gg-gc)={gg - gc_val:+.4f}", flush=True)


def main():
    t_all = time.time()
    sample_df = pd.read_csv(SAMPLE_SUB)
    sample_ids = sample_df["Id"].tolist()
    assert len(sample_ids) == 3180

    with open(CACHE_DIR / "sample_ids.json", "w", encoding="utf-8") as f:
        json.dump(sample_ids, f)

    spec = {
        "seed": SEED,
        "n_samples": len(sample_ids),
        "preprocessing": {
            "center_crop_70": "15% margin cropped on all 4 sides to eliminate random-rotation black triangles",
            "wb_rgb": "per-channel z-score inside non-black (>8) and non-cloud (<225) mask mapped to mean=128, std=48",
            "gray_clahe": "grayscale luminance + CLAHE(clipLimit=2.0, tile=(8,8)) stacked to 3 channels",
        },
        "models": [
            {
                "name": "dinov2_vitb14_reg",
                "source": "https://github.com/facebookresearch/dinov2 (hub: dinov2_vitb14_reg)",
                "paper": "Darcet et al. (2023) Vision Transformers Need Registers; Oquab et al. (2023) DINOv2",
                "license": "Apache-2.0",
                "prompting": "None (pure-vision pixels only)",
                "input_size": 224,
            },
            {
                "name": "convnext_base (IMAGENET1K_V1)",
                "source": "torchvision.models.convnext_base (ConvNeXt_Base_Weights.IMAGENET1K_V1)",
                "paper": "Liu et al. (2022) A ConvNet for the 2020s",
                "license": "MIT / BSD-3-Clause (torchvision)",
                "prompting": "None (pure-vision pixels only)",
                "input_size": 224,
            },
            {
                "name": "dinov2_vits14",
                "source": "https://github.com/facebookresearch/dinov2 (hub: dinov2_vits14)",
                "paper": "Oquab et al. (2023) DINOv2: Learning Robust Visual Features without Supervision",
                "license": "Apache-2.0",
                "prompting": "None (pure-vision pixels only)",
                "input_size": 224,
            },
        ],
    }
    with open(CACHE_DIR / "pretrained_spec_v3.json", "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    stage_3a_phys(sample_ids)
    stage_3b_vitb_wb(sample_ids)
    stage_3c_cnxb_gray(sample_ids)
    stage_3d_vits_gray_rot2(sample_ids)
    verify_all(sample_ids)
    print(f"\nALL EXTRACTION STAGES DONE in {(time.time()-t_all)/60:.1f} min!", flush=True)


if __name__ == "__main__":
    main()
