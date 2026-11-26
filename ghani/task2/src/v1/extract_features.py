"""Feature extraction script for Task-2 (World Phenomena).
Extracts physical, spectral, color, texture, and spatial concentration change metrics between pre.jpg and post.jpg.
100% Pure-Vision, No Prompts, No Text, No External Data.
"""
import os
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image

CLASSES = ['Earthquake', 'Fire', 'Hurricane', 'Tsunami', 'Volcano', 'Explosion']

def rgb_to_hsv(rgb):
    """Vectorized RGB [0, 255] to HSV [0, 1]."""
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    deltac = maxc - minc
    
    s = np.zeros_like(v)
    mask = maxc > 0
    s[mask] = deltac[mask] / maxc[mask]
    
    h = np.zeros_like(v)
    # where deltac == 0, h is 0
    mask_r = (maxc == r) & (deltac > 0)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / deltac[mask_r]) % 6
    mask_g = (maxc == g) & (deltac > 0)
    h[mask_g] = ((b[mask_g] - r[mask_g]) / deltac[mask_g]) + 2
    mask_b = (maxc == b) & (deltac > 0)
    h[mask_b] = ((r[mask_b] - g[mask_b]) / deltac[mask_b]) + 4
    h = (h / 6.0) % 1.0
    return h, s, v

def compute_pair_features(folder_path):
    folder = Path(folder_path)
    sample_id = folder.name
    pre_path = folder / 'pre.jpg'
    post_path = folder / 'post.jpg'
    
    if not (pre_path.exists() and post_path.exists()):
        return None
    
    with Image.open(pre_path) as im_pre:
        im_pre_res = im_pre.resize((256, 256), Image.Resampling.BILINEAR)
        pre_arr = np.array(im_pre_res, dtype=np.float32)
        
    with Image.open(post_path) as im_post:
        im_post_res = im_post.resize((256, 256), Image.Resampling.BILINEAR)
        post_arr = np.array(im_post_res, dtype=np.float32)
        
    feats = {'Id': sample_id}
    
    # 1. Basic RGB Stats & Mean Diff
    diff_arr = np.abs(post_arr - pre_arr)
    feats['mean_diff_all'] = float(diff_arr.mean())
    feats['mean_diff_r'] = float(diff_arr[:, :, 0].mean())
    feats['mean_diff_g'] = float(diff_arr[:, :, 1].mean())
    feats['mean_diff_b'] = float(diff_arr[:, :, 2].mean())
    
    pre_mean = pre_arr.mean(axis=(0, 1))
    post_mean = post_arr.mean(axis=(0, 1))
    feats['pre_mean_r'] = float(pre_mean[0])
    feats['pre_mean_g'] = float(pre_mean[1])
    feats['pre_mean_b'] = float(pre_mean[2])
    feats['post_mean_r'] = float(post_mean[0])
    feats['post_mean_g'] = float(post_mean[1])
    feats['post_mean_b'] = float(post_mean[2])
    feats['delta_r'] = float(post_mean[0] - pre_mean[0])
    feats['delta_g'] = float(post_mean[1] - pre_mean[1])
    feats['delta_b'] = float(post_mean[2] - pre_mean[2])
    
    # 2. Vegetation & Greenness Index (ExG = 2*G - R - B)
    exg_pre = 2 * pre_arr[:, :, 1] - pre_arr[:, :, 0] - pre_arr[:, :, 2]
    exg_post = 2 * post_arr[:, :, 1] - post_arr[:, :, 0] - post_arr[:, :, 2]
    feats['mean_exg_pre'] = float(exg_pre.mean())
    feats['mean_exg_post'] = float(exg_post.mean())
    feats['delta_exg'] = float(exg_post.mean() - exg_pre.mean())
    # Defoliation fraction: pixels where green dropped significantly
    feats['defoliation_ratio'] = float(np.mean((exg_pre > 20) & (exg_post < 5)))
    
    # 3. HSV Space: Hue, Saturation, Value
    h_pre, s_pre, v_pre = rgb_to_hsv(pre_arr)
    h_post, s_post, v_post = rgb_to_hsv(post_arr)
    
    feats['mean_val_pre'] = float(v_pre.mean())
    feats['mean_val_post'] = float(v_post.mean())
    feats['delta_val'] = float(v_post.mean() - v_pre.mean())
    
    feats['mean_sat_pre'] = float(s_pre.mean())
    feats['mean_sat_post'] = float(s_post.mean())
    feats['delta_sat'] = float(s_post.mean() - s_pre.mean())
    
    # 4. Fire / Burn Scars Indicator
    # Burn scars are very dark (v_post < 0.25) where it wasn't dark before (v_pre >= 0.25)
    burn_mask = (v_post < 0.25) & (v_pre >= 0.25)
    feats['burn_scar_ratio'] = float(np.mean(burn_mask))
    
    # 5. Volcano Ash Indicator
    # Ash leads to extreme desaturation across wide areas (s_post < 0.12) with moderate-to-high brightness
    ash_mask = (s_post < 0.12) & (s_pre >= 0.18) & (v_post > 0.3)
    feats['ash_ratio'] = float(np.mean(ash_mask))
    
    # 6. Tsunami / Water Inundation Indicator
    # Water has high blue relative to red: (B - R) / (B + R + 1e-4) or low brightness with blue tint
    water_idx_pre = (pre_arr[:, :, 2] - pre_arr[:, :, 0]) / (pre_arr[:, :, 2] + pre_arr[:, :, 0] + 1e-4)
    water_idx_post = (post_arr[:, :, 2] - post_arr[:, :, 0]) / (post_arr[:, :, 2] + post_arr[:, :, 0] + 1e-4)
    feats['water_idx_pre'] = float(water_idx_pre.mean())
    feats['water_idx_post'] = float(water_idx_post.mean())
    feats['delta_water_idx'] = float(water_idx_post.mean() - water_idx_pre.mean())
    # Inundation ratio: pixels shifting from dry (water_idx < 0) to wet/water (water_idx > 0.05)
    feats['inundation_ratio'] = float(np.mean((water_idx_pre < 0.0) & (water_idx_post > 0.05)))
    
    # 7. Spatial Concentration & Epicenter Blast (Explosion vs Distributed Disasters)
    # Compute change intensity map (grayscale diff)
    gray_diff = diff_arr.mean(axis=2) # 256x256
    thresh = np.percentile(gray_diff, 90)
    top_mask = gray_diff >= max(thresh, 15.0)
    
    if np.sum(top_mask) > 10:
        ys, xs = np.where(top_mask)
        weights = gray_diff[top_mask]
        total_w = np.sum(weights) + 1e-6
        cy = np.sum(ys * weights) / total_w
        cx = np.sum(xs * weights) / total_w
        
        # Spatial dispersion (spread radius sigma in pixels normalized to [0, 1])
        spread_radius = np.sqrt(np.sum(weights * ((ys - cy)**2 + (xs - cx)**2)) / total_w) / 256.0
        # Peak to average ratio of change
        peak_to_mean = float(gray_diff.max() / (gray_diff.mean() + 1e-4))
    else:
        spread_radius = 0.5
        peak_to_mean = 1.0
        
    feats['spread_radius'] = float(spread_radius)
    feats['peak_to_mean_diff'] = float(peak_to_mean)
    
    # 8. High-Frequency Edge Destruction (Earthquake structural damage)
    # Simple Sobel-like gradient approximation
    gy_pre, gx_pre = np.gradient(pre_arr.mean(axis=2))
    gy_post, gx_post = np.gradient(post_arr.mean(axis=2))
    edge_mag_pre = np.sqrt(gx_pre**2 + gy_pre**2)
    edge_mag_post = np.sqrt(gx_post**2 + gy_post**2)
    
    feats['mean_edge_pre'] = float(edge_mag_pre.mean())
    feats['mean_edge_post'] = float(edge_mag_post.mean())
    feats['delta_edge'] = float(edge_mag_post.mean() - edge_mag_pre.mean())
    feats['edge_loss_ratio'] = float((edge_mag_pre.mean() - edge_mag_post.mean()) / (edge_mag_pre.mean() + 1e-4))
    
    return feats

if __name__ == '__main__':
    import pandas as pd
    t0 = time.time()
    fen_dir = Path('Task-2/Fenomena')
    folders = sorted([f for f in fen_dir.iterdir() if f.is_dir()])
    print(f'Starting parallel extraction for {len(folders)} folders...')
    
    out_dir = Path('ghani/task2/cache/v1')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'physical_features.csv'
    
    # Process with ProcessPoolExecutor
    results = []
    # Using max_workers=6 to avoid pegging all CPU cores completely
    with ProcessPoolExecutor(max_workers=6) as executor:
        for res in executor.map(compute_pair_features, folders, chunksize=20):
            if res:
                results.append(res)
                if len(results) % 500 == 0:
                    print(f'  Done {len(results)}/{len(folders)} in {time.time() - t0:.1f}s...', flush=True)
                    
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f'SUCCESS! Extracted {len(df)} samples in {time.time() - t0:.1f}s. Saved to: {out_csv}')

