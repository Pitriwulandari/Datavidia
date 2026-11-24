"""Pemeriksaan sebelum submit (Task-1 Datavidia 11.0).

A. Format file vs sample_submission
B. Kesejajaran baris: prediksi dikaitkan ke fitur test lewat shipment_id (BUKAN urutan baris),
   lalu bandingkan pola prediksi vs pola label asli di train (mis. address_quality=1 -> konfirmasi_alamat).
C. Kesepakatan antar-versi submission & distribusi kelas vs train
D. Apakah test mirip kondisi validasi? cakupan riwayat pelanggan, blok fitur GLMM (60% / 80% / 100% data), kepercayaan prediksi
E. Adversarial validation train vs test (AUC ~0.5 = tidak ada pergeseran)
"""
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

T0 = time.time()
ROOT = "Task-1-Datavidia-11.0"


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def head(t):
    print("\n" + "=" * 8 + " " + t + " " + "=" * 8, flush=True)


SUB = os.environ.get("SUB", "submissions/submission_v6_ensemble_ghani.csv")

# ---------------- A. format ----------------
head("A. FORMAT")
sub = pd.read_csv(SUB)
smp = pd.read_csv(f"{ROOT}/sample_submission.csv")
classes = sorted(pd.read_csv(f"{ROOT}/train/shipments.csv", usecols=["status_penanganan"])["status_penanganan"].unique())
print("file:", SUB)
print("kolom sama:", list(sub.columns) == list(smp.columns), "| baris:", len(sub), "vs", len(smp))
print("urutan id sama dengan sample:", bool((sub.shipment_id.values == smp.shipment_id.values).all()))
print("id unik:", sub.shipment_id.is_unique, "| NaN:", int(sub.status_penanganan.isna().sum()))
print("label valid semua:", set(sub.status_penanganan.unique()).issubset(set(classes)))

# ---------------- muat fitur ----------------
tr = pd.read_pickle("ghani/cache/feat_v5_train.pkl")
te = pd.read_pickle("ghani/cache/feat_v5_test.pkl")
log(f"fitur dimuat: train {tr.shape}, test {te.shape}")

# ---------------- B. kesejajaran ----------------
head("B. KESEJAJARAN BARIS (join lewat shipment_id)")
m = te[["shipment_id", "cu_address_quality", "co_courier_score", "hu_automation_level", "inv_n" if "inv_n" in te.columns else "scan_n"]].merge(
    sub, on="shipment_id", how="left")
print("test tanpa prediksi setelah join:", int(m.status_penanganan.isna().sum()))
print("urutan fitur test == urutan file submission:", bool((te.shipment_id.values == sub.shipment_id.values).all()))


def rate_table(df, col, cls, label_col):
    return df.groupby(col, observed=True)[label_col].apply(lambda s: (s == cls).mean()).round(3)


for col, cls in [("cu_address_quality", "konfirmasi_alamat"), ("co_courier_score", "gagal_antar_awal"),
                 ("co_courier_score", "rusak_perjalanan"), ("hu_automation_level", "klaim_hilang")]:
    a = rate_table(tr, col, cls, "status_penanganan")
    b = rate_table(m, col, cls, "status_penanganan")
    out = pd.DataFrame({"train_label": a, "test_pred": b})
    print(f"\nproporsi '{cls}' menurut {col}:")
    print(out.to_string())

# ---------------- C. antar-versi ----------------
head("C. DISTRIBUSI & KESEPAKATAN ANTAR-VERSI")
trd = tr["status_penanganan"].value_counts(normalize=True).round(4)
ted = sub["status_penanganan"].value_counts(normalize=True).round(4)
print(pd.DataFrame({"train_label": trd, "v6_test_pred": ted}).to_string())
for other in ["submission_v5_ghani.csv", "submission_v4_ghani.csv"]:
    p = f"submissions/{other}"
    if os.path.exists(p):
        o = pd.read_csv(p)
        print(f"kesepakatan v6 vs {other}: {(o.status_penanganan.values == sub.status_penanganan.values).mean() * 100:.1f}%")

# ---------------- D. test vs kondisi validasi ----------------
head("D. TEST vs KONDISI VALIDASI")
folds = np.load("ghani/cache/folds_v5.npy")
c_tr = tr["customer_id_code"].to_numpy()
c_te = te["customer_id_code"].to_numpy()
n_code = int(max(c_tr.max(), c_te.max())) + 1
cnt_all = np.bincount(c_tr, minlength=n_code)
seen_te = (cnt_all[c_te] > 0).mean()
mean_te = cnt_all[c_te].mean()
seen_va, mean_va = [], []
for f in range(5):
    cnt_f = np.bincount(c_tr[folds != f], minlength=n_code)
    cv_ = c_tr[folds == f]
    seen_va.append((cnt_f[cv_] > 0).mean())
    mean_va.append(cnt_f[cv_].mean())
print(f"baris dgn >=1 riwayat pelanggan di data berlabel: test={seen_te * 100:.1f}% | validasi (rata2 5 fold)={np.mean(seen_va) * 100:.1f}%")
print(f"rata-rata jumlah baris berlabel milik pelanggan yang sama: test={mean_te:.2f} | validasi={np.mean(mean_va):.2f}")

oof_blk = np.load("ghani/cache/stage2/glmm_oof_block.npy")
te_blk = np.load("ghani/cache/stage2/glmm_test_block.npy")
inner = np.vstack([np.load(f"ghani/cache/stage2/inner_hold{h}_pred{p}.npy") for h in range(5) for p in range(5) if h != p][:5])
names = {42: "glmm_max_p", 43: "glmm_entropy", 45: "log_n_cust_tr", 47: "log_n_cour_tr"}
rows = []
for i, nme in names.items():
    rows.append({"fitur": nme, "inner(60% data)": inner[:, i].mean(), "outer-val(80%)": oof_blk[:, i].mean(), "test(100%)": te_blk[:, i].mean()})
print(pd.DataFrame(rows).round(4).to_string(index=False))

for nm, a_oof, a_te in [("LGB_res", "lgb_res_oof_logits", "lgb_res_test_logits"), ("ResNet", "nn_oof_logits", "nn_test_logits")]:
    po = np.load(f"ghani/cache/stage2/{a_oof}.npy")
    pt = np.load(f"ghani/cache/stage2/{a_te}.npy")

    def sm(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    print(f"{nm}: rata-rata prob kelas teratas OOF={sm(po).max(1).mean():.4f} | test={sm(pt).max(1).mean():.4f}")

# ---------------- E. adversarial validation ----------------
head("E. ADVERSARIAL VALIDATION (train vs test)")
drop = {"shipment_id", "status_penanganan", "customer_id", "courier_id", "hub_id", "package_id",
        "customer_id_code", "courier_id_code", "hub_id_code", "package_id_code"}
feats = [c for c in te.columns if c not in drop and c in tr.columns]
cats = [c for c in feats if str(tr[c].dtype) == "category"]
rng = np.random.RandomState(0)
n = 150_000
Xa = pd.concat([tr[feats].iloc[rng.choice(len(tr), n, replace=False)], te[feats].iloc[rng.choice(len(te), n, replace=False)]], ignore_index=True)
ya = np.r_[np.zeros(n), np.ones(n)]
idx = rng.permutation(2 * n)
cut = int(0.7 * len(idx))
tr_i, va_i = idx[:cut], idx[cut:]
d = lgb.Dataset(Xa.iloc[tr_i], ya[tr_i], categorical_feature=cats)
mdl = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=200, feature_fraction=0.7,
                     verbose=-1, n_jobs=os.cpu_count()), d, num_boost_round=150)
auc = roc_auc_score(ya[va_i], mdl.predict(Xa.iloc[va_i]))
print(f"AUC train-vs-test = {auc:.4f}  (0.50 = tidak ada pergeseran; >0.55 patut dicurigai)")
imp = pd.Series(mdl.feature_importance("gain"), index=feats).sort_values(ascending=False)
print("fitur pembeda teratas (gain %):", (imp / imp.sum() * 100).round(1).head(6).to_dict())
log("selesai")
