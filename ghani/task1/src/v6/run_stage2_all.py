"""Final Stage-2 Runner (resumes cleanly with LGB_res, LGB_full, XGBoost, ResNet-v1 all cached!):
  1. Fast Ensemble Checkpoint 4 (LGB_res + LGB_full + XGBoost + ResNet + GLMM)
  2. Stage-2 RankGauss-ResNet + Top-3 SWA (5-Fold, ~12m/fold) + Checkpoint 5
  3. Stage-2 Fast CatBoost (Symmetric/Oblivious Trees, 5-Fold) + FINAL 6-MODEL SUPER-ENSEMBLE!
"""
import os
import subprocess
import sys
import time

T0 = time.time()


def run_step(label, cmd, env_extra=None):
    print(f"\n[{time.time()-T0:7.1f}s] >>> STARTING: {label}", flush=True)
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    rc = subprocess.call(cmd, env=env)
    print(f"[{time.time()-T0:7.1f}s] <<< FINISHED: {label} (rc={rc})", flush=True)
    if rc != 0:
        raise SystemExit(f"Step failed: {label} (rc={rc})")


def main():
    py = sys.executable
    steps = [
        ("Fast Ensemble Checkpoint 4 (+LGB_full)", [py, "-u", "-W", "ignore", "ghani/task1/src/ensemble_final.py"],        {"TAG": "v6_ensemble"}),
        ("Stage-2 RankGauss-ResNet (5-Fold)",      [py, "-u", "-W", "ignore", "ghani/task1/src/train_stage2_nn2.py"],      None),
        ("Fast Ensemble Checkpoint 5 (+RGResNet)", [py, "-u", "-W", "ignore", "ghani/task1/src/ensemble_final.py"],        {"TAG": "v6_ensemble"}),
        ("Stage-2 Fast CatBoost (5-Fold)",         [py, "-u", "-W", "ignore", "ghani/task1/src/train_stage2_catboost.py"], None),
        ("FINAL 6-MODEL SUPER-ENSEMBLE",           [py, "-u", "-W", "ignore", "ghani/task1/src/ensemble_final.py"],        {"TAG": "v6_ensemble"}),
    ]
    for label, cmd, env_extra in steps:
        run_step(label, cmd, env_extra)


if __name__ == "__main__":
    main()
