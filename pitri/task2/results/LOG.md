# Task-2 Experiment Log — Pitri Wulandari

## EDA (2026-11-28)
- Dataset: 3,180 pairs (`pre.jpg`, `post.jpg`), 1024x1024x3 each.
- Missing/corrupt images: 0; duplicate IDs: 0; low-variance (`std < 5`): 0.
- Saved audit summary to `pitri/task2/cache/eda_summary.csv`.

## v1 Baseline Domain-Physical Heuristic + v3 Anchor (2026-11-28)
- Seed: 42; Output: `pitri/submissions/task2/submission_v1_pitri.csv`
- Distribution: Hurricane=900, Earthquake=772, Volcano=589, Fire=543, Tsunami=351, Explosion=25

## v2 Spatial Grid 4x4 + Sobel Edge + Color/Spectral Features (2026-11-28)
- Seed: 42; Output: `pitri/submissions/task2/submission_v2_pitri.csv`
- Distribution: Hurricane=890, Earthquake=772, Volcano=585, Fire=545, Tsunami=360, Explosion=28

## v2 Spatial Grid 4x4 + Sobel Edge + Color/Spectral Features (2026-11-28)
- Seed: 42; Output: `pitri/submissions/task2/submission_v2_pitri.csv`
- Distribution: Hurricane=890, Earthquake=772, Volcano=585, Fire=545, Tsunami=360, Explosion=28

## v3 Fast CPU Physical Color-Spatial + GMM + 3-Family Ensemble (2026-11-28)
- Seed: 42; Output: `pitri/submissions/task2/submission_v3_pitri.csv`
- Distribution: Earthquake=1296, Hurricane=884, Fire=376, Volcano=339, Tsunami=233, Explosion=52

## v4 Calibrated Prior + Physical Color-Spatial + GMM + 3-Family Ensemble (2026-11-28)
- Seed: 42; Output: `pitri/submissions/task2/submission_v4_pitri.csv`
- Distribution: Earthquake=1204, Hurricane=865, Fire=404, Volcano=384, Tsunami=268, Explosion=55

## v5 All-in-One Physical + Spatial + GMM(8) + KMeans(16) + 3-Family Ensemble (2026-11-28)
- Seed: 42; Output: `pitri/submissions/task2/submission_v5_pitri.csv`
- Distribution: Earthquake=1326, Hurricane=855, Fire=366, Volcano=362, Tsunami=222, Explosion=49

## v6 Change-Focus Interaction + GMM(8) + 3-Family Ensemble (2026-11-28)
- Seed: 42; Output: `pitri/submissions/task2/submission_v6_pitri.csv`
- Distribution: Earthquake=1230, Hurricane=921, Volcano=390, Fire=337, Tsunami=248, Explosion=54

## Merged Available Pitri Submissions + Physical Tie-Break (2026-11-28)
- Sources: ['pitri/submissions/task2/submission_v4_pitri.csv', 'pitri/submissions/task2/submission_v6_pitri.csv', 'pitri/submissions/task2/submission_v3_pitri.csv']; Output: `pitri/submissions/task2/submission_merged_available_pitri.csv`
- Distribution: Earthquake=1243, Hurricane=867, Fire=387, Volcano=376, Tsunami=253, Explosion=54

## v9 Consensus Self-Training Ensemble (2026-11-28)
- Sources: ['submission_merged_available_pitri.csv', 'submission_v4_pitri.csv', 'submission_v6_pitri.csv']; High-conf: 2502/3180; Output: `pitri/submissions/task2/submission_v9_pitri.csv`
- Distribution: Hurricane=861, Earthquake=1271, Volcano=386, Fire=359, Tsunami=250, Explosion=53

## v10 CPU CatBoost Balanced + LGBM + MLP (2026-11-28)
- Train samples: 3180/3180; Output: `/content/Datavidia/pitri/submissions/task2/submission_v10_pitri.csv`
- Distribution: Hurricane=871, Earthquake=1201, Volcano=392, Fire=391, Tsunami=270, Explosion=55
