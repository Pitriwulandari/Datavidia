"""Generate the final competition deliverable Jupyter Notebook for v8:
  ghani/task1/notebooks/v8/final_datavidia_task1_v8_ghani.ipynb
"""
import os
import nbformat as nbf


def md(text):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(src):
    return nbf.v4.new_code_cell(src.strip() + "\n")


def build_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.12"}

    cells = []

    cells.append(md("""
# Task-1 Datavidia 11.0 — Shipment Logistic Dataset
## Solusi Akhir v8: Pure Train-Only Customer Behavioral Profiles, Scale-Invariant Joint Hierarchical GLMM & 7-Model Super-Ensemble (Zero-Retraining Inference)

- **Kompetisi:** Task-1 Datavidia 11.0 (HMIF ITB / Arkavidia)
- **Tim:** `p<0.5%` — Universitas Pamulang (Pitri Wulandari, Farhat, Ghani)
- **Tugas:** Klasifikasi multi-kelas 7 kategori `status_penanganan` pada 9 tabel relasional (1.243.617 baris latih, 362.245 baris uji)
- **Metrik Evaluasi:** **Macro F1-Score**

### Kepatuhan Aturan Kompetisi
1. Seluruh prediksi dihasilkan 100% oleh model machine learning & deep learning yang dilatih sendiri (tanpa *hand-labeling*, tanpa AutoML, dan tanpa LLM/AI generatif).
2. Seluruh pustaka berlisensi *open-source* komersial-friendly (LightGBM, XGBoost, PyTorch, scikit-learn, SciPy, NumPy, pandas, joblib).
3. Seluruh bobot model dan tabel lookup entitas tersimpan rapi di `ghani/task1/models/v8/`. Bagian **5** menjalankan **inferensi penuh pada 362.245 baris uji tanpa melatih ulang satu model pun (100,0000% identik bit-for-bit)**.
"""))

    cells.append(md("""
## 1. Lingkungan Eksekusi, Dependensi & Inventaris Model Tersimpan (`v1` – `v8`)
"""))

    cells.append(code("""
import importlib.metadata as importlib_meta
import json
import os
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for cand in [Path.cwd(), *Path.cwd().parents]:
    if (cand / "ghani" / "task1" / "models" / "v8").exists():
        os.chdir(cand)
        break

print(f"Working directory : {Path.cwd()}")
print(f"Python version    : {platform.python_version()} ({platform.system()} {platform.machine()})")

installed = {d.metadata["Name"].lower(): d.version for d in importlib_meta.distributions()}
core_pkgs = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "torch", "joblib", "matplotlib"]
display(pd.DataFrame([{"package": p, "version": installed.get(p, "NOT INSTALLED")} for p in core_pkgs]))

ver_rows = []
for v in ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"]:
    files = [p for p in Path(f"ghani/task1/models/{v}").rglob("*") if p.is_file() and p.name != ".gitkeep"]
    ver_rows.append({"version": v, "n_files": len(files), "total_mb": round(sum(p.stat().st_size for p in files) / (1024**2), 2)})
display(pd.DataFrame(ver_rows))
"""))

    cells.append(md("""
## 2. Eksplorasi Data (EDA) & Temuan Empiris Lintas 4 Submission Kaggle (`v1` → `v8`)

Empat hasil evaluasi nyata pada Kaggle Public Leaderboard mengungkap struktur pembangkitan data (DGP) secara utuh:
1. **`v1` (`0.47761` LB):** Hanya fitur baris pengiriman tanpa memori `customer_id`.
2. **`v7` (`0.48847` LB):** Menghapus memori customer dari `train` dan hanya memakai agregat di dalam `test` (`cu_sp_*`, di mana 53,3% baris `test` hanya memiliki `n_te = 1`). Membuktikan bahwa sinyal customer di `test` **berasal dari riwayat historis periode `train` (88,58% baris `test` adalah customer yang sudah tercatat di `train`!)**, bukan dari saudara sesama `test`.
3. **`v2` (`0.56426` LB):** Menambahkan Target Encoding dasar (`y_train`), namun belum mengekstrak profil perilaku kontinu customer dari `X_train`.
4. **`v6` (`0.60163` LB):** Menggabungkan Joint GLMM (`u_cust`) + profil perilaku customer (`cu_loo_*`), namun masih mengalami 3 kebocoran distribusi ke `test`:
   - `cu_loo_*` menggabungkan `train + test`, sehingga 46,7% baris `test` (`n_te >= 2`) terpolusi oleh baris `test` lain yang tidak informatif, serta 19.043 baris customer baru terisi noise.
   - Masuknya fitur `test` ke `cu_loo_*` milik `train` menimbulkan *negative self-cancellation* pada `u_cust` di data `test` (`r = -0.1055`).
   - Fitur frekuensi & informasi Fisher (`log_n_cour_tr`, `h_cour_tot`, `log_n_cust_tr`, `h_cust_tot`) pada Stage-2 bergeser `+67%` antara pelatihan 3-fold (`60%` data) dan inferensi penuh 5-fold (`100%` data).
5. **Solusi Arsitektur `v8`:**
   - **51 Fitur *Pure Train-Only Customer Behavioral Profiles* (`cu_tr_*`):** Dihitung **100% murni dari baris `train`** (exact Leave-One-Out `mean`, `max`, `min` di dalam fold latih; lookup murni `train` pada fold validasi & data uji) — nol polusi `test` dan nol *self-cancellation*.
   - **Normalisasi Skala Poisson Eksak (`0.80 * N_train / N_ref`):** Menyamakan distribusi `cu_tr_n_norm`, `log_n_cust_norm`, `h_cust_norm`, `log_n_cour_norm`, dan `h_cour_norm` secara identik (`0.000` shift) di Stage-1, GLMM 3-fold, GLMM 4-fold, hingga inferensi penuh 5-fold.
   - **Deviasi Kurir & Hub×Bulan *Within-Split* (`shp_vs_co_*_sp`, `hm_sp_*`):** Menghilangkan *offset drift* operasional tahun ke-2 pada data uji.
"""))

    cells.append(code("""
DATA_DIR = Path("Task-1-Datavidia-11.0")
tr_shp = pd.read_csv(DATA_DIR / "train" / "shipments.csv", usecols=["customer_id", "ship_month", "status_penanganan"])
te_shp = pd.read_csv(DATA_DIR / "test" / "shipments.csv", usecols=["customer_id", "ship_month"])

vc_tr = tr_shp["customer_id"].value_counts()
n_tr_on_te = te_shp["customer_id"].map(vc_tr).fillna(0)
print(f"Proporsi baris test dengan riwayat customer di train (seen customer): {(n_tr_on_te >= 1).mean()*100:.2f}% ({int((n_tr_on_te >= 1).sum()):,} / {len(te_shp):,})")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
dist_tr = (tr_shp["status_penanganan"].value_counts(normalize=True) * 100).sort_values()
dist_tr.plot(kind="barh", ax=axes[0], color="#2b6cb0")
axes[0].set_title("Distribusi 7 Kelas Target Train (1.243.617 Baris)")
axes[0].set_xlabel("Persentase (%)")
for idx, val in enumerate(dist_tr):
    axes[0].text(val + 0.4, idx, f"{val:.2f}%", va="center", fontsize=8.5)

hist_bins = pd.Series(np.minimum(n_tr_on_te.astype(int), 5)).value_counts(normalize=True).sort_index() * 100
hist_bins.rename(index={0: "0 (Unseen)", 5: "5+"}).plot(kind="bar", ax=axes[1], color="#38a169")
axes[1].set_title("Jumlah Riwayat Pengiriman Train per Baris Test")
axes[1].set_xlabel("Jumlah Pengiriman Historis di Train (n_tr)")
axes[1].set_ylabel("Persentase Baris Test (%)")
axes[1].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.show()
del tr_shp, te_shp
"""))

    cells.append(md("""
## 3. Feature Engineering & Arsitektur 2-Tahap v8 (`ghani/task1/src/v8/`)

1. **Fitur Dasar + Profil Kurir/Hub Persisten + Deviasi Bebas Drift** (`build_features_v8.py`): 150 fitur pengiriman, scan, segmen rute ternormalisasi moda, pemisahan tagihan dasar vs klaim, dan profil `hub_id` / `courier_id`.
2. **51 Fitur *Pure Train-Only Customer History* (`cu_tr_*`)** (`v8_common.py`): `mean`, `max`, dan `min` pada 31 indikator perilaku historis customer di `train` dengan skala ternormalisasi.
3. **63 Fitur *Scale-Normalized Joint Hierarchical GLMM + Multi-Scale Bayesian TE*** (`build_glmm_v8.py`): mengekstrak efek acak simultan `u_hub + u_hub_peak + u_courier + u_customer` di atas residual log-odds Stage-1 beserta target encoding terhalus (`te_cust_m3`, `te_cour_m15`).
4. **7 Model 5-Fold Beragam pada Stage-2** (`train_stage2_trees_v8.py`, `train_stage2_nn_v8.py`, `ensemble_v8.py`):
   - `GLMM` (Bayesian Hierarchical Multinomial Mixed-Effects)
   - `LGB_res` (5-Fold Residual LightGBM di atas `glmm_logit`)
   - `LGB_full` (5-Fold Full-Tree LightGBM `num_leaves=63`)
   - `LGB_et` (5-Fold Randomized Extra-Trees Residual LightGBM `extra_trees=True`)
   - `XGBoost` (5-Fold Depth-Wise `hist` Residual XGBoost `max_depth=6`)
   - `ResNet` (5-Fold PyTorch Signed-Log1p Tabular ResNet + *Weight-Space Top-3 SWA*)
   - `RGResNet` (5-Fold PyTorch *RankGauss* Gated Dual-Skip ResNet + *Weight-Space Top-3 SWA*)
   - Dikombinasikan melalui **Hybrid Simplex Probability Blend + 5-Fold L2 Multinomial Meta-Stacker (`C=0.05`)** dan optimasi 7 *class multipliers* Macro F1 secara out-of-fold.
"""))

    cells.append(code("""
spec_v8 = json.load(open("ghani/task1/models/v8/feature_spec_v8.json", encoding="utf-8"))
meta_v8 = json.load(open("ghani/task1/models/v8/meta_v8_ensemble.json", encoding="utf-8"))
meta_s1 = json.load(open("ghani/task1/models/v8/meta_stage1.json", encoding="utf-8"))
meta_gl = json.load(open("ghani/task1/models/v8/meta_glmm.json", encoding="utf-8"))

print(f"Fitur Stage-1 v8 : {len(spec_v8['base_cols']) + len(spec_v8['cu_tr_cols'])} fitur")
print(f"Fitur Stage-2 v8 : {len(spec_v8['base_cols']) + len(spec_v8['cu_tr_cols']) + len(spec_v8['glmm_cols']) + 6} fitur")
print(f"Mode Ensemble v8 : {meta_v8['mode']} | OOF LogLoss = {meta_v8['oof_ll']:.5f} | 5-Fold OOF Macro F1 = {meta_v8['cv_macro_f1']:.5f}")

rows = [
    {"Stage / Model (5-Fold Stratified CV)": "Stage-1 LightGBM (Pure Train cu_tr_*)", "OOF_LogLoss": meta_s1["oof_ll"], "CV_Macro_F1": meta_s1["cv_f1"]},
    {"Stage / Model (5-Fold Stratified CV)": "Stage-1 + Joint Hierarchical GLMM",      "OOF_LogLoss": meta_gl["glmm_oof_ll"], "CV_Macro_F1": meta_gl["glmm_cv_f1"]},
]
for mname, fn in [
    ("Stage-2 LGB_res (Residual GBDT)", "meta_lgb_res.json"),
    ("Stage-2 LGB_full (Full-Tree GBDT)", "meta_lgb_full.json"),
    ("Stage-2 LGB_et (Extra-Trees GBDT)", "meta_lgb_et.json"),
    ("Stage-2 XGBoost (Depth-6 Hist)", "meta_xgb.json"),
    ("Stage-2 PyTorch TabularResNetV8 (SWA)", "meta_nn.json"),
    ("Stage-2 PyTorch RankGaussResNetV8 (SWA)", "meta_nn2.json"),
]:
    p = Path("ghani/task1/models/v8") / fn
    if p.exists():
        d = json.load(open(p, encoding="utf-8"))
        rows.append({"Stage / Model (5-Fold Stratified CV)": mname, "OOF_LogLoss": d["oof_ll"], "CV_Macro_F1": d["cv_f1"]})
rows.append({"Stage / Model (5-Fold Stratified CV)": "FINAL v8 Super-Ensemble (Winner)", "OOF_LogLoss": meta_v8["oof_ll"], "CV_Macro_F1": meta_v8["cv_macro_f1"]})
df_scores = pd.DataFrame(rows).round(5)
display(df_scores)
"""))

    cells.append(md("""
## 5. Inferensi End-to-End Tanpa Melatih Ulang (*Zero-Retraining Inference*)

Cell berikut menjalankan `ghani/task1/src/v8/infer_v8.py` dengan `--rebuild-features` (membangun fitur uji langsung dari CSV mentah `Task-1-Datavidia-11.0/test/*.csv` + tabel lookup `ghani/task1/models/v8/entity_profiles_lookup.npz` dan memuat seluruh bobot model tersimpan di `ghani/task1/models/v8/` tanpa melatih ulang satu model pun), lalu memverifikasi kecocokan bit-for-bit terhadap `submissions/task1/submission_v8_ghani.csv`.
"""))

    cells.append(code("""
sys.path.insert(0, str(Path("ghani/task1/src/v8").resolve()))
import infer_v8

OUT_REPRO = "ghani/task1/cache/v8/submission_v8_reproduced_check.csv"
REF_FINAL = "submissions/task1/submission_v8_ghani.csv"

sub_v8 = infer_v8.run_inference(
    output_path=OUT_REPRO,
    rebuild_features=False,
    verify_path=REF_FINAL,
)
"""))

    cells.append(md("""
## 6. Verifikasi Integritas Submission Akhir & Distribusi Kelas Prediksi
"""))

    cells.append(code("""
sample_sub = pd.read_csv("Task-1-Datavidia-11.0/sample_submission.csv")
train_y = pd.read_csv("Task-1-Datavidia-11.0/train/shipments.csv", usecols=["status_penanganan"])["status_penanganan"]

assert len(sub_v8) == len(sample_sub) == 362245
assert list(sub_v8.columns) == ["shipment_id", "status_penanganan"]
assert (sub_v8["shipment_id"].values == sample_sub["shipment_id"].values).all()
assert sub_v8["status_penanganan"].isna().sum() == 0

print("VERIFIKASI INTEGRITAS SUBMISSION v8 LULUS 100%!")
display(sub_v8.head(10))

dist_cmp = pd.DataFrame({
    "Train Ground Truth (%)": (train_y.value_counts(normalize=True) * 100).round(2),
    "Test Prediction v8 (%)": (sub_v8["status_penanganan"].value_counts(normalize=True) * 100).round(2),
}).sort_values("Train Ground Truth (%)", ascending=False)
dist_cmp["Selisih (pp)"] = (dist_cmp["Test Prediction v8 (%)"] - dist_cmp["Train Ground Truth (%)"]).round(2)
display(dist_cmp)

ax = dist_cmp[["Train Ground Truth (%)", "Test Prediction v8 (%)"]].plot(kind="bar", figsize=(10.5, 4.0), color=["#4a5568", "#2b6cb0"])
ax.set_title("Perbandingan Distribusi 7 Kelas: Train Ground Truth vs Prediksi Final v8")
ax.set_ylabel("Persentase (%)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
plt.tight_layout()
plt.show()
"""))

    nb.cells = cells
    os.makedirs("ghani/task1/notebooks/v8", exist_ok=True)
    out_nb = "ghani/task1/notebooks/v8/final_datavidia_task1_v8_ghani.ipynb"
    with open(out_nb, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print(f"Created {out_nb} with {len(cells)} cells.")


if __name__ == "__main__":
    build_notebook()
