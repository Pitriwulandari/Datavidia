"""DINOv2 + Domain-Physical Fusion Classifier for Task-2 (World Phenomena).
Fuses DINOv2 ViT-S/14 deep visual representations with calibrated spectral/spatial physical change signatures.
100% Pure-Vision, No Prompts, No Text, No External Data.
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import zscore

CLASSES = ['Earthquake', 'Fire', 'Hurricane', 'Tsunami', 'Volcano', 'Explosion']

def generate_v2_submission():
    t0 = time.time()
    cache_v2 = Path('ghani/task2/cache/v2')
    cache_v1 = Path('ghani/task2/cache/v1')
    out_csv = Path('ghani/submissions/task2/submission_v2_ghani.csv')
    
    # 1. Load DINOv2 Features
    feats_pre = np.load(cache_v2 / 'feats_pre.npy') # (3180, 384)
    feats_post = np.load(cache_v2 / 'feats_post.npy') # (3180, 384)
    with open(cache_v2 / 'sample_ids.json', 'r') as f:
        sample_ids = json.load(f)
        
    phy_df = pd.read_csv(cache_v1 / 'physical_features.csv')
    phy_df = phy_df.set_index('Id').loc[sample_ids].reset_index()
    N = len(sample_ids)
    print(f'Loaded DINOv2 features & physical features for {N} samples')
    
    # 2. Compute Physical Heuristic Scores
    num_cols = [c for c in phy_df.columns if c != 'Id']
    phy_df[num_cols] = phy_df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    z_burn = zscore(phy_df['burn_scar_ratio'].values)
    z_ash = zscore(phy_df['ash_ratio'].values)
    z_inundation = zscore(phy_df['inundation_ratio'].values)
    z_water_delta = zscore(phy_df['delta_water_idx'].values)
    z_exg_delta = zscore(phy_df['delta_exg'].values)
    z_defol = zscore(phy_df['defoliation_ratio'].values)
    z_peak_mean = zscore(phy_df['peak_to_mean_diff'].values)
    z_spread = zscore(phy_df['spread_radius'].values)
    z_edge_loss = zscore(phy_df['edge_loss_ratio'].values)
    z_sat_delta = zscore(phy_df['delta_sat'].values)
    z_val_delta = zscore(phy_df['delta_val'].values)

    s_phy = np.zeros((N, 6), dtype=np.float32)
    # Fire
    s_phy[:, 1] = 3.0 * z_burn - 2.0 * z_exg_delta - 1.5 * z_val_delta + 1.0 * z_defol + 2.0 * (phy_df['burn_scar_ratio'] > np.percentile(phy_df['burn_scar_ratio'], 90))
    # Volcano
    s_phy[:, 4] = 3.5 * z_ash - 2.5 * z_sat_delta + 1.0 * z_spread + 2.5 * (phy_df['ash_ratio'] > np.percentile(phy_df['ash_ratio'], 90))
    # Tsunami
    s_phy[:, 3] = 3.0 * z_inundation + 2.5 * z_water_delta - 1.0 * z_sat_delta + 2.5 * (phy_df['inundation_ratio'] > np.percentile(phy_df['inundation_ratio'], 90))
    # Explosion
    s_phy[:, 5] = 3.0 * z_peak_mean - 2.5 * z_spread - 1.0 * (z_burn > 2.0) - 1.0 * (z_ash > 2.0) + 2.5 * (phy_df['peak_to_mean_diff'] > np.percentile(phy_df['peak_to_mean_diff'], 95))
    # Hurricane
    s_phy[:, 2] = 3.0 * z_defol - 1.5 * z_exg_delta + 1.5 * z_spread + 1.0 * z_edge_loss - 2.0 * (z_burn > 1.5) - 2.0 * (z_ash > 1.5)
    # Earthquake
    s_phy[:, 0] = 2.5 * z_edge_loss + 1.0 * z_spread - 2.0 * (z_burn > 1.0) - 2.0 * (z_ash > 1.0) - 2.0 * (z_inundation > 1.0) - 1.5 * np.abs(z_exg_delta) + 2.0
    z_s_phy = zscore(s_phy, axis=0)
    
    # 3. Compute DINOv2 Deep Prototype Similarities
    norm_post = feats_post / (np.linalg.norm(feats_post, axis=1, keepdims=True) + 1e-7)
    norm_pre = feats_pre / (np.linalg.norm(feats_pre, axis=1, keepdims=True) + 1e-7)
    diff_feats = norm_post - norm_pre
    norm_diff = diff_feats / (np.linalg.norm(diff_feats, axis=1, keepdims=True) + 1e-7)
    X_vis = np.hstack([norm_post, norm_diff])
    X_vis = X_vis / (np.linalg.norm(X_vis, axis=1, keepdims=True) + 1e-7)

    K_ANCHORS = 40
    anchors = {}
    anchors['Fire'] = (phy_df['burn_scar_ratio'].rank(ascending=False) + (-phy_df['delta_exg']).rank(ascending=False)).nsmallest(K_ANCHORS).index.values
    anchors['Volcano'] = (phy_df['ash_ratio'].rank(ascending=False) + (-phy_df['delta_sat']).rank(ascending=False)).nsmallest(K_ANCHORS).index.values
    anchors['Tsunami'] = (phy_df['inundation_ratio'].rank(ascending=False) + phy_df['delta_water_idx'].rank(ascending=False)).nsmallest(K_ANCHORS).index.values
    anchors['Explosion'] = (phy_df['peak_to_mean_diff'].rank(ascending=False) + phy_df['spread_radius'].rank(ascending=True)).nsmallest(K_ANCHORS).index.values
    anchors['Hurricane'] = (phy_df['defoliation_ratio'].rank(ascending=False) - phy_df['burn_scar_ratio'].rank(ascending=False) - phy_df['ash_ratio'].rank(ascending=False)).nsmallest(K_ANCHORS).index.values
    anchors['Earthquake'] = (phy_df['edge_loss_ratio'].rank(ascending=False) - phy_df['burn_scar_ratio'].rank(ascending=False) - phy_df['ash_ratio'].rank(ascending=False) - phy_df['inundation_ratio'].rank(ascending=False)).nsmallest(K_ANCHORS).index.values

    prototypes = np.zeros((len(CLASSES), X_vis.shape[1]), dtype=np.float32)
    for c_idx, c_name in enumerate(CLASSES):
        proto = X_vis[anchors[c_name]].mean(axis=0)
        prototypes[c_idx] = proto / np.linalg.norm(proto)

    sims = np.dot(X_vis, prototypes.T)
    z_sims = zscore(sims, axis=0)

    # 4. Optimal Multi-Modal Fusion (50% DINOv2 Deep Visual + 50% Domain Physical)
    ALPHA = 0.50
    fused_scores = ALPHA * z_s_phy + (1.0 - ALPHA) * z_sims
    
    # 5. Class Assignment & Prior Multiplier
    class_priors = np.array([1.05, 1.00, 0.95, 0.95, 0.95, 0.95], dtype=np.float32)
    final_scores = fused_scores * class_priors
    pred_idx = np.argmax(final_scores, axis=1)
    
    sub_df = pd.DataFrame({
        'Id': sample_ids,
        'Phenomenon': [CLASSES[i] for i in pred_idx]
    })
    
    print('Distribution of predictions in v2 (DINOv2 + Domain Physics Fusion):')
    dist = sub_df['Phenomenon'].value_counts()
    for cls in CLASSES:
        cnt = dist.get(cls, 0)
        print(f'  {cls:12s}: {cnt:5d} ({cnt/N*100:5.1f}%)')
        
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(out_csv, index=False)
    print(f'SUCCESS! Saved submission v2 to: {out_csv} in {time.time() - t0:.2f}s')
    return sub_df

if __name__ == '__main__':
    generate_v2_submission()
