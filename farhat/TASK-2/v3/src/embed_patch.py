"""Patch-level embedding extraction for localized damage detection."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import timm
from PIL import Image
from tqdm import tqdm

from src import config as C
from src.prepare import list_ids
from src.utils import get_logger, l2norm, set_seed

log = get_logger("embed_patch")

# Use histmatched thumbs if available, else regular thumbs
THUMB_DIR = C.CACHE / "thumbs_histmatched"
if not THUMB_DIR.exists():
    THUMB_DIR = C.CACHE / "thumbs"

PATCH_SIZE = 64  # patch size in thumbnail pixels (256x256 -> 16 patches per side)
STRIDE = 64      # non-overlapping patches

ENCODERS = {
    "dinov2_s": ("vit_small_patch14_dinov2.lvd142m", 384, 224),
    "dinov3_s": ("vit_small_patch16_dinov3.lvd1689m", 384, 224),
    "mae_b": ("vit_base_patch16_224.mae", 768, 224),
    "mobilenetv3_l": ("mobilenetv3_large_100.ra_in1k", 1280, 224),
}

def build_encoder(key: str, size: int):
    model_name, out_dim, _ = ENCODERS[key]
    kwargs = dict(pretrained=True, num_classes=0)
    if "vit" in model_name:
        kwargs.update(img_size=size)
    model = timm.create_model(model_name, **kwargs).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    cfg["input_size"] = (3, size, size)
    tf = timm.data.create_transform(**cfg, is_training=False)
    log.info("encoder=%s model=%s input=%dpx params=%.1fM",
             key, model_name, size, sum(p.numel() for p in model.parameters()) / 1e6)
    return model, tf, model_name, out_dim

def extract_patches(img: Image.Image, patch_size: int, stride: int) -> list[Image.Image]:
    """Extract patches from image."""
    w, h = img.size
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
    return patches

@torch.inference_mode()
def embed_patches(model, tf, patches: list[Image.Image], batch: int = 64) -> np.ndarray:
    feats = []
    for i in range(0, len(patches), batch):
        chunk = patches[i:i + batch]
        x = torch.stack([tf(p.convert("RGB")) for p in chunk])
        y = model(x)
        feats.append(y.float().numpy())
    return np.concatenate(feats, axis=0) if feats else np.empty((0,))

def run(encoder: str, limit: int | None = None) -> Path:
    model_name, out_dim, size = ENCODERS[encoder]
    set_seed(C.SEED)
    torch.set_num_threads(C.N_JOBS)
    ids = list_ids()
    if limit:
        ids = ids[:limit]
    
    missing = [i for i in ids if not (THUMB_DIR / f"{i}_pre.jpg").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} thumbnail belum ada")
    
    model, tf, name, _ = build_encoder(encoder, size)
    out = C.FEATURE_DIR / f"emb_{encoder}_{size}_patch_{PATCH_SIZE}.npz"
    
    all_pre_patches = []
    all_post_patches = []
    patch_coords = []  # (pid, patch_idx)
    
    t0 = time.time()
    for pid in tqdm(ids, desc=f"Patch extract {encoder}"):
        with Image.open(THUMB_DIR / f"{pid}_pre.jpg") as pre_img:
            pre_patches = extract_patches(pre_img, PATCH_SIZE, STRIDE)
        with Image.open(THUMB_DIR / f"{pid}_post.jpg") as post_img:
            post_patches = extract_patches(post_img, PATCH_SIZE, STRIDE)
        
        all_pre_patches.extend(pre_patches)
        all_post_patches.extend(post_patches)
        patch_coords.extend([(pid, i) for i in range(len(pre_patches))])
    
    log.info(f"Extracted {len(all_pre_patches)} patches from {len(ids)} images")
    
    # Embed pre patches
    pre_feats = embed_patches(model, tf, all_pre_patches, batch=C.BATCH_SIZE)
    pre_feats = l2norm(pre_feats.astype(np.float32))
    
    # Embed post patches
    post_feats = embed_patches(model, tf, all_post_patches, batch=C.BATCH_SIZE)
    post_feats = l2norm(post_feats.astype(np.float32))
    
    # Compute patch-wise change (post - pre)
    change_feats = l2norm((post_feats - pre_feats).astype(np.float32))
    
    np.savez_compressed(out,
        ids=np.array([c[0] for c in patch_coords]),
        patch_idx=np.array([c[1] for c in patch_coords]),
        pre=pre_feats,
        post=post_feats,
        change=change_feats,
        model=name,
        size=size,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        thumb_size=256
    )
    log.info("Patch embedding %s: pre=%s post=%s change=%s -> %s (%.1f min)",
             encoder, pre_feats.shape, post_feats.shape, change_feats.shape, out, (time.time() - t0) / 60)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS.keys()), help="Encoder names")
    ap.add_argument("--limit", type=int, default=None, help="Limit samples")
    ap.add_argument("--patch-size", type=int, default=64, help="Patch size in thumb pixels")
    ap.add_argument("--stride", type=int, default=64, help="Stride in thumb pixels")
    args = ap.parse_args()
    
    global PATCH_SIZE, STRIDE
    PATCH_SIZE = args.patch_size
    STRIDE = args.stride
    
    for enc in args.encoders:
        run(enc, args.limit)

if __name__ == "__main__":
    main()