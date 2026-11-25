"""Runner v4: fitur v2 + OOF Target Encoding pada seluruh data penuh (1.05M train, lr=0.08)."""
import os, subprocess, sys
env = dict(os.environ, TAG="v4", TE="1", LR="0.08", ROUNDS="500")
sys.exit(subprocess.call([sys.executable, "ghani/src/v2/train_v2.py"], env=env))
