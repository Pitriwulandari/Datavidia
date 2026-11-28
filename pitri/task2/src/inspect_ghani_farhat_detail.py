import os
import glob
import json

print("=== 1. LOG TERBARU GHANI ===")
ghani_log = "ghani/task2/results/LOG.md"
if os.path.exists(ghani_log):
    with open(ghani_log, "r", encoding="utf-8", errors="ignore") as f:
        print(f.read()[-1500:])
else:
    print("Tidak ada ghani/task2/results/LOG.md")

print("\n=== 2. KONFIG & MAPPING FARHAT v3 ===")
for p in [
    "farhat/README.md",
    "farhat/TASK-2/v3/models/model_config_v3.json",
    "farhat/TASK-2/v3/models/cluster_mapping_v3.json",
]:
    if os.path.exists(p):
        print(f"\n--- {p} ---")
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            print(f.read()[:1500])

print("\n=== 3. CARI FILE .CSV DI SELURUH REPO ===")
csvs = sorted(glob.glob("**/*.csv", recursive=True))
print(csvs if csvs else "Belum ada file .csv yang ter-commit di repo.")
