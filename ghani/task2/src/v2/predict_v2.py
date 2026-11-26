"""DINOv2 Prototype & Label Propagation Classifier for Task-2 (World Phenomena).
Fuses DINOv2 self-supervised visual features with physical anchors via iterative prototype refinement.
100% Pure-Vision Foundation Model, No Prompts, No Text, No External Data.
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import softmax

CLASSES = ['Earthquake', 'Fire', 'Hurricane', 'Tsunami', 'Volcano', 'Explosion']

def build_v2_predictions():
    t0 = time.time()
    cache_v2 = Path('ghani/task2/cache/v2')
    cache_v1 = Path('ghani/task2/cache/v1')
    out_csv = Path('submissions/task2/submission_v2_ghani.csv')
    
    feats_pre = np.load(cache_v2 / 'feats_pre.npy') # (3180, 384)
    feats_post = np.load(cache_v2 / 'feats_post.npy') # (3180, 384)
    with open(cache_v2 / 'sample_ids.json', 'r') as f:
        sample_ids = json.load(f)
        
    phy_df = pd.read_csv(cache_v1 / 'physical_features.csv')
    # Align phy_df by sample_ids
    phy_df = phy_df.set_index('Id').loc[sample_ids].reset_index()
    
    N = len(sample_ids)
    print(f'Loaded DINOv2 features for {N} samples')
    
    # 1. Construct representations
    # L2 normalize post embeddings and diff embeddings
    norm_post = feats_post / (np.linalg.norm(feats_post, axis=1, keepdims=True) + 1e-7)
    norm_pre = feats_pre / (np.linalg.norm(feats_pre, axis=1, keepdims=True) + 1e-7)
    diff_feats = norm_post - norm_pre
    norm_diff = diff_feats / (np.linalg.norm(diff_feats, axis=1, keepdims=True) + 1e-7)
    
    # Combined visual feature: [norm_post, norm_diff]
    X_vis = np.hstack([norm_post, norm_diff]) # 384 + 384 = 768
    X_vis = X_vis / (np.linalg.norm(X_vis, axis=1, keepdims=True) + 1e-7)
    
    # 2. Extract High-Confidence Anchor Indices for each class (Top 25 purest samples)
    K_ANCHORS = 25
    anchors = {}
    
    # Fire: extreme burn scar & negative exg
    fire_rank = phy_df['burn_scar_ratio'].rank(ascending=False) + (-phy_df['delta_exg']).rank(ascending=False)
    anchors['Fire'] = fire_rank.nsmallest(K_ANCHORS).index.values
    
    # Volcano: extreme ash ratio & negative delta_sat
    volc_rank = phy_df['ash_ratio'].rank(ascending=False) + (-phy_df['delta_sat']).rank(ascending=False)
    anchors['Volcano'] = volc_rank.nsmallest(K_ANCHORS).index.values
    
    # Tsunami: extreme inundation & water index increase
    tsun_rank = phy_df['inundation_ratio'].rank(ascending=False) + phy_df['delta_water_idx'].rank(ascending=False)
    anchors['Tsunami'] = tsun_rank.nsmallest(K_ANCHORS).index.values
    
    # Explosion: extreme peak to mean & smallest spread radius
    expl_rank = phy_df['peak_to_mean_diff'].rank(ascending=False) + phy_df['spread_radius'].rank(ascending=True)
    anchors['Explosion'] = expl_rank.nsmallest(K_ANCHORS).index.values
    
    # Hurricane: high defoliation & large spread without burn/ash
    hurr_score = (
        phy_df['defoliation_ratio'].rank(ascending=False) 
        + phy_df['spread_radius'].rank(ascending=False)
        - 2.0 * phy_df['burn_scar_ratio'].rank(ascending=False)
        - 2.0 * phy_df['ash_ratio'].rank(ascending=False)
    )
    anchors['Hurricane'] = hurr_score.nsmallest(K_ANCHORS).index.values
    
    # Earthquake: high edge loss & large spread without burn/ash/water
    eq_score = (
        phy_df['edge_loss_ratio'].rank(ascending=False)
        - 2.0 * phy_df['burn_scar_ratio'].rank(ascending=False)
        - 2.0 * phy_df['ash_ratio'].rank(ascending=False)
        - 2.0 * phy_df['inundation_ratio'].rank(ascending=False)
    )
    anchors['Earthquake'] = eq_score.nsmallest(K_ANCHORS).index.values
    
    print('Anchors computed successfully:')
    for c in CLASSES:
        print(f'  {c:12s}: {len(anchors[c])} anchors')
        
    # 3. Compute initial prototypes from anchors
    prototypes = np.zeros((len(CLASSES), X_vis.shape[1]), dtype=np.float32)
    for c_idx, c_name in enumerate(CLASSES):
        p_idx = anchors[c_name]
        proto = X_vis[p_idx].mean(axis=0)
        prototypes[c_idx] = proto / (np.linalg.norm(proto) + 1e-7)
        
    # 4. Iterative Soft EM Label Propagation (5 iterations)
    # Temperature parameter for cosine similarity
    tau = 12.0
    probs = None
    for it in range(8):
        sims = np.dot(X_vis, prototypes.T) # (N, 6)
        probs = softmax(sims * tau, axis=1) # (N, 6)
        
        # Update prototypes with soft cluster assignments
        for c_idx in range(len(CLASSES)):
            weights = probs[:, c_idx:c_idx+1]
            new_proto = np.sum(weights * X_vis, axis=0)
            prototypes[c_idx] = new_proto / (np.linalg.norm(new_proto) + 1e-7)
            
    # 5. Prior-Calibrated Decision
    # Add prior log-multipliers so Earthquake is around 20-22% and classes are well-balanced
    class_priors = np.array([0.21, 0.18, 0.12, 0.16, 0.18, 0.15], dtype=np.float32)
    calibrated_probs = probs * class_priors
    pred_idx = np.argmax(calibrated_probs, axis=1)
    
    df_pred = pd.DataFrame({
        'Id': sample_ids,
        'Phenomenon': [CLASSES[i] for i in pred_idx]
    })
    
    print('\nDistribution of predictions in v2 (DINOv2 + Refined Prototypes):')
    dist = df_pred['Phenomenon'].value_counts()
    for cls in CLASSES:
        cnt = dist.get(cls, 0)
        print(f'  {cls:12s}: {cnt:5d} ({cnt/N*100:5.1f}%)')
        
    # Save submission
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(out_csv, index=False)
    print(f'SUCCESS! Saved submission v2 to: {out_csv} in {time.time() - t0:.1f}s')
    return df_pred

if __name__ == '__main__':
    build_v2_predictions()
