"""Inference & Classification Script for Task-2 Baseline v1.
Uses physical/spectral bi-temporal change metrics to classify 3,180 disaster pairs into 6 phenomena:
  - Earthquake
  - Fire
  - Hurricane
  - Tsunami
  - Volcano
  - Explosion
100% Pure-Vision, No Prompts, No Text, No External Data.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import zscore

CLASSES = ['Earthquake', 'Fire', 'Hurricane', 'Tsunami', 'Volcano', 'Explosion']

def classify_samples(csv_path: Path, output_csv: Path):
    df = pd.read_csv(csv_path)
    print(f'Loaded {len(df)} samples from {csv_path}')
    
    # Clean inf/nan if any
    num_cols = [c for c in df.columns if c != 'Id']
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    N = len(df)
    scores = np.zeros((N, 6), dtype=np.float32)
    # 0: Earthquake, 1: Fire, 2: Hurricane, 3: Tsunami, 4: Volcano, 5: Explosion
    
    # Standardize key features for balanced multi-metric scoring
    z_burn = zscore(df['burn_scar_ratio'].values)
    z_ash = zscore(df['ash_ratio'].values)
    z_inundation = zscore(df['inundation_ratio'].values)
    z_water_delta = zscore(df['delta_water_idx'].values)
    z_exg_delta = zscore(df['delta_exg'].values)
    z_defol = zscore(df['defoliation_ratio'].values)
    z_peak_mean = zscore(df['peak_to_mean_diff'].values)
    z_spread = zscore(df['spread_radius'].values)
    z_edge_loss = zscore(df['edge_loss_ratio'].values)
    z_sat_delta = zscore(df['delta_sat'].values)
    z_val_delta = zscore(df['delta_val'].values)
    
    # 1. FIRE Score
    # Characterized by high burn scar ratio, severe vegetation drop (-delta_exg), and brightness drop (-delta_val)
    fire_score = (
        3.0 * z_burn
        - 2.0 * z_exg_delta
        - 1.5 * z_val_delta
        + 1.0 * z_defol
    )
    # Extra boost if burn scar ratio is above 90th percentile
    burn_p90 = np.percentile(df['burn_scar_ratio'], 90)
    fire_score += 2.0 * (df['burn_scar_ratio'] > burn_p90)
    scores[:, 1] = fire_score
    
    # 2. VOLCANO Score
    # Characterized by wide ash coverage (ash_ratio) and massive desaturation (-delta_sat)
    volcano_score = (
        3.5 * z_ash
        - 2.5 * z_sat_delta
        + 1.0 * z_spread # ash blankets a large area
    )
    ash_p90 = np.percentile(df['ash_ratio'], 90)
    volcano_score += 2.5 * (df['ash_ratio'] > ash_p90)
    scores[:, 4] = volcano_score
    
    # 3. TSUNAMI Score
    # Characterized by water inundation and positive shift in water index
    tsunami_score = (
        3.0 * z_inundation
        + 2.5 * z_water_delta
        - 1.0 * z_sat_delta # water desaturates land
    )
    inun_p90 = np.percentile(df['inundation_ratio'], 90)
    tsunami_score += 2.5 * (df['inundation_ratio'] > inun_p90)
    scores[:, 3] = tsunami_score
    
    # 4. EXPLOSION Score
    # Characterized by localized blast: very high peak-to-mean change, small spread radius (concentric blast)
    explosion_score = (
        3.0 * z_peak_mean
        - 2.5 * z_spread
        - 1.0 * (z_burn > 2.0) # not widespread wildfire
        - 1.0 * (z_ash > 2.0) # not volcano
    )
    # Extreme peak to mean is a classic explosion signature
    peak_p95 = np.percentile(df['peak_to_mean_diff'], 95)
    explosion_score += 2.5 * (df['peak_to_mean_diff'] > peak_p95)
    scores[:, 5] = explosion_score
    
    # 5. HURRICANE Score
    # Characterized by wide defoliation across the frame, moderate water changes, large spread radius,
    # but low burn scar and low ash
    hurricane_score = (
        3.0 * z_defol
        - 1.5 * z_exg_delta
        + 1.5 * z_spread # widespread across entire image
        + 1.0 * z_edge_loss # roofs torn off
        - 2.0 * (z_burn > 1.5) # no charcoal
        - 2.0 * (z_ash > 1.5) # no ash
    )
    scores[:, 2] = hurricane_score
    
    # 6. EARTHQUAKE Score
    # Structural collapse: edge loss with intact vegetation and normal color (no burn, no ash, no ocean water)
    earthquake_score = (
        2.5 * z_edge_loss
        + 1.0 * z_spread # affects multiple buildings across scene
        - 2.0 * (z_burn > 1.0)
        - 2.0 * (z_ash > 1.0)
        - 2.0 * (z_inundation > 1.0)
        - 1.5 * np.abs(z_exg_delta) # vegetation unchanged
    )
    # Calibrated bonus to match committee's ground truth prior (~20-22% Earthquake share)
    earthquake_score += 2.0
    scores[:, 0] = earthquake_score
    
    # Argmax prediction
    pred_idx = np.argmax(scores, axis=1)
    df['Phenomenon'] = [CLASSES[i] for i in pred_idx]
    
    print('Distribution of predictions in v1:')
    dist = df['Phenomenon'].value_counts()
    for cls in CLASSES:
        cnt = dist.get(cls, 0)
        print(f'  {cls:12s}: {cnt:5d} ({cnt/N*100:5.1f}%)')
        
    # Match sample_submission.csv format and ordering exactly
    sub_sample = pd.read_csv('Task-2/sample_submission.csv')
    sub_df = sub_sample[['Id']].merge(df[['Id', 'Phenomenon']], on='Id', how='left')
    
    if sub_df['Phenomenon'].isna().any():
        print('Warning: some IDs had missing predictions! Imputing with Earthquake...')
        sub_df['Phenomenon'] = sub_df['Phenomenon'].fillna('Earthquake')
        
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(output_csv, index=False)
    print(f'Saved submission to: {output_csv} ({len(sub_df)} rows)')
    return sub_df

if __name__ == '__main__':
    csv_in = Path('ghani/task2/cache/v1/physical_features.csv')
    csv_out = Path('submissions/task2/submission_v1_ghani.csv')
    classify_samples(csv_in, csv_out)
