"""Tahap 2: ekstraksi embedding dengan encoder pure-vision beku.

Encoder yang dipakai (bobot publik, tanpa prompt teks / tanpa input interaktif):
  * dinov2_s -> timm: vit_small_patch14_dinov2.lvd142m (DINOv2 ViT-S/14, self-supervised)
  * effb0    -> timm: mobilenetv3_large_100.ra_in1k (ImageNet-1k, supervised)

Keluaran: cache/features/emb_<encoder>_<size>.npz  (ids, pre, post)
"""
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
from src.prepare import list_ids, thumb_path
from src.utils import get_logger, l2norm, set_seed

log = get_logger("embed")

# nama model per kunci encoder (config.ENCODERS memuat default; ini menimpa bila perlu)
MODEL_ZOO = {
    "dinov2_s": "vit_small_patch14_dinov2.lvd142m",
    "effb0": "mobilenetv3_large_100.ra_in1k",
}


def build_encoder(key: str, size: int):
    name = MODEL_ZOO.get(key, C.ENCODERS[key])
    kwargs = dict(pretrained=True, num_classes=0)
    if "vit" in name:
        kwargs.update(img_size=size)
    model = timm.create_model(name, **kwargs).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    cfg["input_size"] = (3, size, size)
    tf = timm.data.create_transform(**cfg, is_training=False)
    log.info("encoder=%s model=%s input=%dpx params=%.1fM",
             key, name, size, sum(p.numel() for p in model.parameters()) / 1e6)
    return model, tf, name


def _load(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


@torch.inference_mode()
def embed_paths(model, tf, paths: list[Path], size: int, batch: int = C.BATCH_SIZE) -> np.ndarray:
    feats = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        x = torch.stack([tf(_load(p)) for p in chunk])
        y = model(x)
        feats.append(y.float().numpy())
    return np.concatenate(feats, axis=0)


def run(encoder: str, size: int, limit: int | None = None, tag: str | None = None) -> Path:
    set_seed(C.SEED)
    torch.set_num_threads(C.N_JOBS)
    ids = list_ids()
    if limit:
        ids = ids[:limit]
    missing = [i for i in ids if not thumb_path(i, "pre").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} thumbnail belum ada; jalankan `python -m src.prepare`")

    model, tf, name = build_encoder(encoder, size)
    out = C.FEATURE_DIR / f"emb_{tag or encoder}_{size}.npz"
    paths = [thumb_path(pid, kind) for pid in ids for kind in ("pre", "post")]
    t0 = time.time()
    feats = embed_paths(model, tf, paths, size, batch=C.BATCH_SIZE)
    pre_a = l2norm(feats[0::2].astype(np.float32))
    post_a = l2norm(feats[1::2].astype(np.float32))
    np.savez_compressed(out, ids=np.array(ids), pre=pre_a, post=post_a, model=name, size=size)
    log.info("embedding %s: %s -> %s (%.1f min)", encoder, pre_a.shape, out, (time.time() - t0) / 60)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinov2_s", choices=list(MODEL_ZOO))
    ap.add_argument("--size", type=int, default=196)
    ap.add_argument("--limit", type=int, default=None, help="batasi jumlah sampel (untuk uji cepat)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    run(args.encoder, args.size, args.limit, args.tag)


if __name__ == "__main__":
    main()
