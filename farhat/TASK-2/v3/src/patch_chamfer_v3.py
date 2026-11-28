"""Task-2 v3: Patch-Chamfer Alignment-Free Local Change Detection.

Adapted from Ghani v4 align_extract_v4.py with key modifications:
1. No rotation/shift alignment (assume pre/post roughly aligned)
2. Patch-level chamfer distance for localized change detection
3. Top-change weighted embedding for disaster-specific features

Target: Capture local disasters (Fire, Earthquake, Volcano, Explosion, Tsunami)
that global embeddings miss.
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
from PIL import Image
from scipy.special import softmax
from torchvision import transforms

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 8))

ROOT = Path(__file__).resolve().parents[1]
FEN_DIR = ROOT / "data" / "raw" / "Fenomena"
SAMPLE_SUB = ROOT / "data" / "raw" / "sample_submission.csv"
CACHE_DIR = ROOT / "cache" / "features"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def l2norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def to_gray_clahe_rgb(im_rgb: Image.Image) -> Image.Image:
    """Luminance CLAHE stacked to 3 channels: immune to color tint."""
    g = np.asarray(im_rgb.convert("L"), dtype=np.uint8)
    g_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    return Image.fromarray(np.stack([g_eq, g_eq, g_eq], axis=-1), mode="RGB")


def valid_mask_grid(im_rgb: Image.Image, grid_size: int) -> np.ndarray:
    """Compute (grid_size, grid_size) valid mask excluding black triangles & bright clouds."""
    sz = grid_size * 14
    arr = np.asarray(im_rgb.resize((sz, sz), Image.Resampling.BILINEAR), dtype=np.float32)
    blk = arr.reshape(grid_size, 14, grid_size, 14, 3)
    p_max = blk.max(axis=4).mean(axis=(1, 3))
    p_min = blk.min(axis=4).mean(axis=(1, 3))
    mask = (p_max > 12.0) & (p_min < 220.0)
    if mask.sum() < 8:
        mask = np.ones((grid_size, grid_size), dtype=bool)
    return mask.astype(np.float32)


def _local_3x3_chamfer_pair(
    p1_grid: np.ndarray, m1_grid: np.ndarray, 
    p2_grid: np.ndarray, m2_grid: np.ndarray
) -> tuple:
    """Compute local 3x3 patch-chamfer stats and top-change embeddings.
    
    This is the KEY innovation from Ghani: instead of global image embedding,
    compute per-patch cosine distance, then weight embeddings by change score.
    
    Args:
        p1_grid: (G, G, D) pre patch embeddings
        m1_grid: (G, G) pre valid mask
        p2_grid: (G, G, D) post patch embeddings  
        m2_grid: (G, G) post valid mask
    
    Returns:
        stats: (12,) chamfer statistics
        topchg_pre: (D,) top-change weighted pre embedding
        topchg_post: (D,) top-change weighted post embedding
        topchg_delta: (D,) top-change weighted delta embedding
    """
    g = p1_grid.shape[0]
    dim = p1_grid.shape[2]
    m1b = m1_grid > 0.5
    m2b = m2_grid > 0.5
    ov = m1b & m2b
    
    if ov.sum() < 8:
        ov = np.ones((g, g), dtype=bool)
        m1b = np.ones((g, g), dtype=bool)
    
    loc_chg = np.zeros((g, g), dtype=np.float32)
    oy, ox = np.where(ov)
    
    # For each valid patch in post, find max cosine similarity in 3x3 neighbourhood in pre
    for y, x in zip(oy, ox):
        y0, y1 = max(0, y - 1), min(g, y + 2)
        x0, x1 = max(0, x - 1), min(g, x + 2)
        win_m = m1b[y0:y1, x0:x1]
        if win_m.any():
            win_p = p1_grid[y0:y1, x0:x1][win_m]  # (K, D)
            # Cosine similarity: higher = more similar = less change
            max_cos = float((win_p @ p2_grid[y, x]).max())
        else:
            max_cos = float(p1_grid[y, x] @ p2_grid[y, x])
        loc_chg[y, x] = np.clip(1.0 - max_cos, 0.0, 2.0)  # chamfer distance
    
    chg_vals = loc_chg[ov]
    p50, p75, p85, p90, p95 = [float(x) for x in np.percentile(chg_vals, [50, 75, 85, 90, 95])]
    
    # Spatial stats of high-change patches
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
    
    # Weighted average of patch embeddings by change score
    # Higher change = higher weight → focuses on disaster-affected areas
    w = softmax(10.0 * chg_vals).astype(np.float32)[:, None]  # (K, 1)
    p1_ov = p1_grid[ov]  # (K, D)
    p2_ov = p2_grid[ov]  # (K, D)
    topchg_pre = l2norm((w * p1_ov).sum(axis=0), axis=0).astype(np.float32)
    topchg_post = l2norm((w * p2_ov).sum(axis=0), axis=0).astype(np.float32)
    topchg_delta = l2norm(topchg_post - topchg_pre, axis=0).astype(np.float32)
    
    return stats, topchg_pre, topchg_post, topchg_delta


def extract_patch_chamfer_v3(
    sample_ids: list[str],
    model_name: str = "dinov2_vits14",
    img_size: int = 336,
    batch_size: int = 16,
) -> Path:
    """Extract patch-chamfer features using DINOv2.
    
    Output NPZ contains:
        - chamfer_stats: (N, 12) local change statistics
        - topchg_pre: (N, D) top-change weighted pre embeddings
        - topchg_post: (N, D) top-change weighted post embeddings
        - topchg_delta: (N, D) top-change weighted delta embeddings
        - pmean_pre: (N, D) global mean pre embeddings
        - pmean_post: (N, D) global mean post embeddings
    """
    grid_size = img_size // 14  # DINOv2 patch size = 14
    dim_map = {"dinov2_vits14": 384, "dinov2_vitb14_reg": 768}
    dim = dim_map.get(model_name, 384)
    
    out_name = f"patch_chamfer_v3_{model_name.replace('/', '_')}_{img_size}.npz"
    out_path = CACHE_DIR / out_name
    N = len(sample_ids)
    
    if out_path.exists():
        d = np.load(out_path)
        if d["chamfer_stats"].shape == (N, 12):
            print(f"[v3] Checkpoint exists: {out_path}", flush=True)
            return out_path
    
    t0 = time.time()
    print(f"[v3] Loading {model_name} @ {img_size}x{img_size} (grid={grid_size}x{grid_size})...", flush=True)
    
    # Load from local hub
    hub_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if not hub_dir.exists():
        # Fallback to online hub
        model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True).eval()
    else:
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
            r2 = im2.convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
        m1 = valid_mask_grid(r1, grid_size)
        m2 = valid_mask_grid(r2, grid_size)
        # Use gray CLAHE for color invariance
        return tf(to_gray_clahe_rgb(r1)), tf(to_gray_clahe_rgb(r2)), m1, m2
    
    chamfer_stats = np.zeros((N, 12), dtype=np.float32)
    topchg_pre = np.zeros((N, dim), dtype=np.float32)
    topchg_post = np.zeros((N, dim), dtype=np.float32)
    topchg_delta = np.zeros((N, dim), dtype=np.float32)
    pmean_pre = np.zeros((N, dim), dtype=np.float32)
    pmean_post = np.zeros((N, dim), dtype=np.float32)
    
    with ThreadPoolExecutor(max_workers=4) as ex, torch.inference_mode():
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            bsz = end - i
            loaded = list(ex.map(_load_pair, range(i, end)))
            t1_b = torch.stack([x[0] for x in loaded])
            t2_b = torch.stack([x[1] for x in loaded])
            
            # Extract patch tokens
            p1 = l2norm(model.forward_features(t1_b)["x_norm_patchtokens"].numpy().astype(np.float32))
            p2 = l2norm(model.forward_features(t2_b)["x_norm_patchtokens"].numpy().astype(np.float32))
            p1 = p1.reshape(bsz, grid_size, grid_size, dim)
            p2 = p2.reshape(bsz, grid_size, grid_size, dim)
            
            for j in range(bsz):
                idx = i + j
                st, tc_pre, tc_post, tc_del = _local_3x3_chamfer_pair(
                    p1[j], loaded[j][2], p2[j], loaded[j][3]
                )
                chamfer_stats[idx] = st
                topchg_pre[idx] = tc_pre
                topchg_post[idx] = tc_post
                topchg_delta[idx] = tc_del
                
                # Global mean for comparison
                ov = (loaded[j][2] > 0.5) & (loaded[j][3] > 0.5)
                if ov.sum() < 8:
                    ov = np.ones((grid_size, grid_size), dtype=bool)
                p1_ov = p1[j][ov]
                p2_ov = p2[j][ov]
                pmean_pre[idx] = l2norm(p1_ov.mean(axis=0), axis=0)
                pmean_post[idx] = l2norm(p2_ov.mean(axis=0), axis=0)
            
            if end % 320 == 0 or end == N:
                el = time.time() - t0
                eta = el / end * (N - end) / 60.0
                print(f"  [v3] {end}/{N} in {el:.1f}s (ETA {eta:.1f} min)", flush=True)
    
    np.savez_compressed(
        out_path,
        chamfer_stats=chamfer_stats,
        topchg_pre=topchg_pre,
        topchg_post=topchg_post,
        topchg_delta=topchg_delta,
        pmean_pre=pmean_pre,
        pmean_post=pmean_post,
    )
    print(f"[v3] Saved {out_path} in {(time.time()-t0)/60:.1f} min", flush=True)
    
    del model
    gc.collect()
    return out_path


def main():
    t_all = time.time()
    
    # Load sample IDs
    sample_sub = pd.read_csv(SAMPLE_SUB)
    sample_ids = sample_sub["Id"].tolist()
    assert len(sample_ids) == 3180
    print(f"[v3] N={len(sample_ids)} samples", flush=True)
    
    # Save sample IDs for reference
    with open(CACHE_DIR / "sample_ids_v3.json", "w") as f:
        json.dump(sample_ids, f)
    
    # Extract with DINOv2-S/14 @ 336 (24x24 grid)
    extract_patch_chamfer_v3(sample_ids, "dinov2_vits14", img_size=336, batch_size=16)
    
    print(f"\n[v3] ALL DONE in {(time.time()-t_all)/60:.1f} min!", flush=True)


if __name__ == "__main__":
    main()
