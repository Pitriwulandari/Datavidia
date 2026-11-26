"""Master Orchestrator for v8 End-to-End Pipeline:
  1. Stage-1 5-Fold LightGBM (`run_stage1_v8.py`)
  2. Scale-Normalized Joint GLMM + Multi-Scale Bayesian TE (`build_glmm_v8.py`)
  3. Stage-2 5-Fold Tree Zoo: LGB_res, LGB_full, LGB_et, XGBoost (`train_stage2_trees_v8.py`)
  4. Stage-2 5-Fold Neural Zoo: TabularResNetV8, RankGaussResNetV8 (`train_stage2_nn_v8.py`)
  5. Super-Ensemble + L2 Meta-Stacker + Macro-F1 Optimizer (`ensemble_v8.py`)
  6. Zero-Retraining Bit-for-Bit Inference Verification (`infer_v8.py --rebuild-features`)

Every step & fold caches its outputs so if interrupted, re-running resumes instantly!
"""
import subprocess
import sys
import time

STEPS = [
    ("Stage-1 5-Fold LightGBM",             [sys.executable, "-u", "-W", "ignore", "ghani/task1/src/v8/run_stage1_v8.py"]),
    ("Joint GLMM + Smoothed TE",            [sys.executable, "-u", "-W", "ignore", "ghani/task1/src/v8/build_glmm_v8.py"]),
    ("Stage-2 5-Fold Tree Models",          [sys.executable, "-u", "-W", "ignore", "ghani/task1/src/v8/train_stage2_trees_v8.py"]),
    ("Stage-2 5-Fold Neural ResNets",       [sys.executable, "-u", "-W", "ignore", "ghani/task1/src/v8/train_stage2_nn_v8.py"]),
    ("v8 Super-Ensemble & Submission",      [sys.executable, "-u", "-W", "ignore", "ghani/task1/src/v8/ensemble_v8.py"]),
    ("Zero-Retrain Raw-CSV Verification",   [sys.executable, "-u", "-W", "ignore", "ghani/task1/src/v8/infer_v8.py", "--rebuild-features", "--output", "submissions/task1/submission_v8_inferred_ghani.csv", "--verify-against", "submissions/task1/submission_v8_ghani.csv"]),
]

t0 = time.time()
for idx, (title, cmd) in enumerate(STEPS, 1):
    ts = time.time()
    print(f"\n{'=' * 72}\n[{idx}/{len(STEPS)}] STARTING: {title} (elapsed={(ts - t0)/60:.1f} min)\n{'=' * 72}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"ERROR: Step {idx} ({title}) exited with code {rc}!", flush=True)
        sys.exit(rc)
    print(f"[{idx}/{len(STEPS)}] COMPLETED: {title} in {(time.time() - ts)/60:.1f} min (total={(time.time() - t0)/60:.1f} min)", flush=True)

print(f"\nALL v8 PIPELINE STEPS COMPLETED SUCCESSFULLY IN {(time.time() - t0)/60:.1f} MINUTES!", flush=True)
