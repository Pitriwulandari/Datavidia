"""Task-2 v7 Part 1: Satellite-Native Pure-Vision Feature Extraction via SatlasPretrain
(Aerial_SwinB_SI & Sentinel2_SwinB_SI_RGB, AllenAI, Apache-2.0 / ODC-BY).

100% Pure-Vision & Compliant (FAQ Task-2 no. 2):
  - Pretrained strictly on high-resolution NAIP aerial (0.5-2 m/px) and Sentinel-2 satellite imagery.
  - Zero text/prompts/VLM/CLIP/SAM, zero external disaster labels, 100% offline local checkpoint loading.
  - Deterministic SEED = 42, checkpointed per backbone, RAM-safe.
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

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.special import softmax
from satlaspretrain_models.model import Model
from satlaspretrain_models.utils import SatlasPretrain_weights

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "ghani" / "task2" / "src" / "v4"))
sys.path.insert(0, str(ROOT / "ghani" / "task2" / "src" / "v5"))
from align_extract_v4 import l2norm, warp_post_to_pre  # noqa: E402
from extract_v5 import detint_clean_rgb  # noqa: E402

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(12, os.cpu_count() or 12))

FEN_DIR = ROOT / "Task-2" / "Fenomena"
SAMPLE_SUB = ROOT / "Task-2" / "sample_submission.csv"
C4_DIR = ROOT / "ghani" / "task2" / "cache" / "v4"
C7_DIR = ROOT / "ghani" / "task2" / "cache" / "v7"
W_DIR = C7_DIR / "weights"
C7_DIR.mkdir(parents=True, exist_ok=True)


def _valid_mask_hw(im_rgb: Image.Image, grid_hw: int) -> np.ndarray:
    """Compute (grid_hw, grid_hw) float32 valid mask excluding black rotated borders & white clouds."""
    sz = grid_hw * 16
    arr = np.asarray(im_rgb.resize((sz, sz), Image.Resampling.BILINEAR), dtype=np.float32)
    blk = arr.reshape(grid_hw, 16, grid_hw, 16, 3)
    p_max = blk.max(axis=4).mean(axis=(1, 3))
    p_min = blk.min(axis=4).mean(axis=(1, 3))
    mask = (p_max > 12.0) & (p_min < 220.0)
    if mask.sum() < 4:
        mask = np.ones((grid_hw, grid_hw), dtype=bool)
    return mask.astype(np.float32)


def _pil_to_unit_tensor(im_rgb: Image.Image) -> torch.Tensor:
    """SatlasPretrain official RGB normalization: 8-bit [0..255] -> float32 [0..1], CHW."""
    arr = np.asarray(im_rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def extract_satlas_backbone(
    model_id: str,
    out_name: str,
    sample_ids: list[str],
    aln,
    img_size: int = 256,
    batch_size: int = 24,
) -> Path:
    out_path = C7_DIR / out_name
    N = len(sample_ids)
    if out_path.exists():
        d = np.load(out_path)
        if d["f3_post"].shape == (N, 1024):
            print(f"[{model_id}] Checkpoint exists: {out_path} {d['f3_post'].shape}", flush=True)
            return out_path

    pth = W_DIR / f"{model_id}.pth"
    assert pth.exists(), f"Missing cached weight file: {pth}"
    info = SatlasPretrain_weights[model_id]

    t0 = time.time()
    print(f"[{model_id}] Loading offline weights from {pth}...", flush=True)
    weights = torch.load(pth, map_location="cpu", weights_only=False)
    model = Model(info["num_channels"], info["multi_image"], info["backbone"], fpn=False, weights=weights).eval()
    del weights
    gc.collect()

    bang, bdy, bdx = aln["best_ang"], aln["best_dy"], aln["best_dx"]

    def _load_pair(idx: int):
        sid = sample_ids[idx]
        folder = FEN_DIR / sid
        with Image.open(folder / "pre.jpg") as i1, Image.open(folder / "post.jpg") as i2:
            r1 = i1.convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
            r2 = warp_post_to_pre(i2.convert("RGB"), float(bang[idx]), int(bdy[idx]), int(bdx[idx]), img_size)
        m1_16 = _valid_mask_hw(r1, 16)
        m2_16 = _valid_mask_hw(r2, 16)
        m1_8 = (m1_16.reshape(8, 2, 8, 2).mean(axis=(1, 3)) > 0.25).astype(np.float32)
        m2_8 = (m2_16.reshape(8, 2, 8, 2).mean(axis=(1, 3)) > 0.25).astype(np.float32)
        d1 = detint_clean_rgb(r1, fill_neutral_border=True)
        d2 = detint_clean_rgb(r2, fill_neutral_border=True)
        return _pil_to_unit_tensor(d1), _pil_to_unit_tensor(d2), m1_16 * m2_16, m1_8 * m2_8

    f3_pre = np.zeros((N, 1024), dtype=np.float32)
    f3_post = np.zeros((N, 1024), dtype=np.float32)
    f3_delta = np.zeros((N, 1024), dtype=np.float32)
    f2_pre = np.zeros((N, 512), dtype=np.float32)
    f2_post = np.zeros((N, 512), dtype=np.float32)
    f2_topchg_post = np.zeros((N, 512), dtype=np.float32)
    f2_topchg_delta = np.zeros((N, 512), dtype=np.float32)
    loc_stats = np.zeros((N, 8), dtype=np.float32)

    with ThreadPoolExecutor(max_workers=4) as ex, torch.inference_mode():
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            bsz = end - i
            loaded = list(ex.map(_load_pair, range(i, end)))
            t1 = torch.stack([x[0] for x in loaded])
            t2 = torch.stack([x[1] for x in loaded])

            outs1 = model(t1)
            outs2 = model(t2)
            # SwinB multi-scale outputs: [-2] is (B, 512, 16, 16), [-1] is (B, 1024, 8, 8)
            s2_1 = outs1[-2].numpy().transpose(0, 2, 3, 1).astype(np.float32)
            s2_2 = outs2[-2].numpy().transpose(0, 2, 3, 1).astype(np.float32)
            s3_1 = outs1[-1].numpy().transpose(0, 2, 3, 1).astype(np.float32)
            s3_2 = outs2[-1].numpy().transpose(0, 2, 3, 1).astype(np.float32)

            for j in range(bsz):
                idx = i + j
                ov16 = loaded[j][2] > 0.5
                ov8 = loaded[j][3] > 0.5
                if ov16.sum() < 8:
                    ov16 = np.ones((16, 16), dtype=bool)
                if ov8.sum() < 4:
                    ov8 = np.ones((8, 8), dtype=bool)

                # Stage-4 (1024-d, 8x8)
                p3_1 = l2norm(s3_1[j][ov8], axis=-1)
                p3_2 = l2norm(s3_2[j][ov8], axis=-1)
                f3_pre[idx] = l2norm(p3_1.mean(axis=0), axis=0)
                f3_post[idx] = l2norm(p3_2.mean(axis=0), axis=0)
                f3_delta[idx] = l2norm((p3_2 - p3_1).mean(axis=0), axis=0)

                # Stage-3 (512-d, 16x16 fine spatial grid)
                p2_1 = l2norm(s2_1[j][ov16], axis=-1)
                p2_2 = l2norm(s2_2[j][ov16], axis=-1)
                f2_pre[idx] = l2norm(p2_1.mean(axis=0), axis=0)
                f2_post[idx] = l2norm(p2_2.mean(axis=0), axis=0)

                chg = np.clip(1.0 - (p2_1 * p2_2).sum(axis=-1), 0.0, 2.0)
                w = softmax(8.0 * chg).astype(np.float32)[:, None]
                tc1 = l2norm((w * p2_1).sum(axis=0), axis=0)
                tc2 = l2norm((w * p2_2).sum(axis=0), axis=0)
                f2_topchg_post[idx] = tc2
                f2_topchg_delta[idx] = l2norm(tc2 - tc1, axis=0)

                p50, p90, p95 = [float(x) for x in np.percentile(chg, [50, 90, 95])]
                loc_stats[idx] = np.array([
                    float(chg.mean()), float(chg.std()), p50, p90, p95, float(chg.max()),
                    float((chg > 0.25).mean()), float((chg > 0.40).mean()),
                ], dtype=np.float32)

            if end % 360 == 0 or end == N:
                el = time.time() - t0
                eta = el / end * (N - end) / 60.0
                print(f"  [{model_id}] {end}/{N} in {el:.1f}s (ETA {eta:.1f} min)", flush=True)

    np.savez_compressed(
        out_path,
        f3_pre=f3_pre,
        f3_post=f3_post,
        f3_delta=f3_delta,
        f2_pre=f2_pre,
        f2_post=f2_post,
        f2_topchg_post=f2_topchg_post,
        f2_topchg_delta=f2_topchg_delta,
        loc_stats=loc_stats,
    )
    print(f"[{model_id}] Saved {out_path} in {(time.time()-t0)/60:.1f} min", flush=True)
    del model
    gc.collect()
    return out_path


def main():
    t_all = time.time()
    sample_ids = json.load(open(C4_DIR / "sample_ids.json", "r", encoding="utf-8"))
    ref_ids = pd.read_csv(SAMPLE_SUB)["Id"].tolist()
    assert len(sample_ids) == 3180 and sample_ids == ref_ids
    with open(C7_DIR / "sample_ids.json", "w", encoding="utf-8") as f:
        json.dump(sample_ids, f)

    aln = np.load(C4_DIR / "alignment_v4.npz")
    extract_satlas_backbone("Aerial_SwinB_SI", "satlas_aerial_swinb.npz", sample_ids, aln)
    if (W_DIR / "Sentinel2_SwinB_SI_RGB.pth").exists():
        extract_satlas_backbone("Sentinel2_SwinB_SI_RGB", "satlas_s2_swinb.npz", sample_ids, aln)

    spec = {
        "seed": SEED,
        "n_samples": len(sample_ids),
        "models": [
            {
                "name": "SatlasPretrain Aerial_SwinB_SI",
                "source": "https://github.com/allenai/satlaspretrain_models (HF: allenai/satlas-pretrain/aerial_swinb_si.pth)",
                "paper": "Bastani et al. (ICCV 2023) SatlasPretrain: A Large-Scale Dataset for Remote Sensing Image Understanding",
                "code_license": "Apache-2.0",
                "weights_license": "ODC-BY",
                "prompting": "None (pure-vision high-res aerial RGB only, normalized [0,1])",
            },
            {
                "name": "SatlasPretrain Sentinel2_SwinB_SI_RGB",
                "source": "https://github.com/allenai/satlaspretrain_models (HF: allenai/satlas-pretrain/sentinel2_swinb_si_rgb.pth)",
                "paper": "Bastani et al. (ICCV 2023) SatlasPretrain",
                "code_license": "Apache-2.0",
                "weights_license": "ODC-BY",
                "prompting": "None (pure-vision global satellite RGB only, normalized [0,1])",
            },
        ],
    }
    with open(C7_DIR / "pretrained_spec_v7.json", "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    print(f"\nALL v7 SATLAS EXTRACTION DONE in {(time.time()-t_all)/60:.1f} min!", flush=True)


if __name__ == "__main__":
    main()
