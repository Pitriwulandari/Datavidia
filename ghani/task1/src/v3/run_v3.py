"""Runner v3: fitur v2 + OOF Target Encoding + Count Encoding (subsample 450k)."""
import os, subprocess, sys
env = dict(os.environ, TAG="v3", TE="1", SUB_N="450000", LR="0.10", ROUNDS="400")
sys.exit(subprocess.call([sys.executable, "ghani/task1/src/v2/train_v2.py"], env=env))
