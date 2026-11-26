"""Helper unduh & ekstrak dataset Task-2 Datavidia 11.0 (World Phenomena).

Pastikan sudah:
  1. Klik 'Join Competition' & 'I Understand and Accept' di https://www.kaggle.com/competitions/task-2-datavidia-11-0/rules
  2. Login Kaggle API (`kagglehub.login()` atau `~/.kaggle/kaggle.json`), ATAU taruh file zip hasil download browser
     ke `data/archive/task-2-datavidia-11-0.zip` lalu jalankan skrip ini.

Penggunaan (dari root repo):
  python ghani/task2/src/common/download_task2.py
"""
import os
import shutil
import zipfile
from pathlib import Path

TARGET_DIR = Path("Task-2-Datavidia-11.0")
ZIP_PATH = Path("data/archive/task-2-datavidia-11.0.zip")


def main():
    if (TARGET_DIR / "sample_submission.csv").exists():
        print(f"Dataset sudah siap di: {TARGET_DIR.resolve()}")
        return

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists():
        print(f"Mengekstrak {ZIP_PATH} ke {TARGET_DIR} ...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(TARGET_DIR)
        print(f"Selesai diekstrak ke: {TARGET_DIR.resolve()}")
        return

    print("Mencoba mengunduh via kagglehub.competition_download('task-2-datavidia-11-0') ...")
    import kagglehub

    dl_path = Path(kagglehub.competition_download("task-2-datavidia-11-0"))
    print("Downloaded cache path:", dl_path)
    if dl_path.is_file() and dl_path.suffix == ".zip":
        with zipfile.ZipFile(dl_path, "r") as zf:
            zf.extractall(TARGET_DIR)
    elif dl_path.is_dir():
        for item in dl_path.iterdir():
            dst = TARGET_DIR / item.name
            if not dst.exists():
                if item.is_dir():
                    shutil.copytree(item, dst)
                else:
                    shutil.copy2(item, dst)
    print(f"Dataset Task-2 siap di: {TARGET_DIR.resolve()}")


if __name__ == "__main__":
    main()
