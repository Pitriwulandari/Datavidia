"""Tindak lanjut adversarial validation: apakah pergeseran train-vs-test terpusat di beberapa fitur turunan saja?"""
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

T0 = time.time()
tr = pd.read_pickle("ghani/task1/cache/v5/feat_v5_train.pkl")
te = pd.read_pickle("ghani/task1/cache/v5/feat_v5_test.pkl")
drop = {"shipment_id", "status_penanganan", "customer_id", "courier_id", "hub_id", "package_id",
        "customer_id_code", "courier_id_code", "hub_id_code", "package_id_code"}
feats = [c for c in te.columns if c not in drop and c in tr.columns]

print("=== Sebaran fitur pembeda teratas: train vs test ===")
for c in ["shp_vs_co_scan_q", "shp_vs_co_delay", "shp_vs_co_lgap", "hm_p_n_ratio", "cu_loo_delay_res_co"]:
    a, b = tr[c].astype("float64"), te[c].astype("float64")
    print(f"{c:22s} train mean={a.mean():+.4f} std={a.std():.4f} na={a.isna().mean():.3f} | test mean={b.mean():+.4f} std={b.std():.4f} na={b.isna().mean():.3f}")

mon_tr = tr["ship_month"].value_counts(normalize=True).sort_index()
mon_te = te["ship_month"].value_counts(normalize=True).sort_index()
print("\nbulan pengiriman (%): train vs test")
print(pd.DataFrame({"train": (mon_tr * 100).round(2), "test": (mon_te * 100).round(2)}).T.to_string())
print(f"porsi bulan puncak (3,10,11,12): train={tr['is_peak_month'].mean() * 100:.2f}% | test={te['is_peak_month'].mean() * 100:.2f}%")

rng = np.random.RandomState(1)
n = 150_000
a = tr[feats].iloc[rng.choice(len(tr), n, replace=False)]
b = te[feats].iloc[rng.choice(len(te), n, replace=False)]
X = pd.concat([a, b], ignore_index=True)
y = np.r_[np.zeros(n), np.ones(n)]
idx = rng.permutation(2 * n)
cut = int(0.7 * len(idx))
cats = [c for c in feats if str(tr[c].dtype) == "category"]


def auc_without(drop_cols):
    cols = [c for c in feats if c not in drop_cols]
    cc = [c for c in cats if c in cols]
    d = lgb.Dataset(X[cols].iloc[idx[:cut]], y[idx[:cut]], categorical_feature=cc)
    m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=200,
                       feature_fraction=0.7, verbose=-1, n_jobs=os.cpu_count()), d, num_boost_round=150)
    return roc_auc_score(y[idx[cut:]], m.predict(X[cols].iloc[idx[cut:]])), m, cols


auc0, m0, cols0 = auc_without(set())
imp = pd.Series(m0.feature_importance("gain"), index=cols0).sort_values(ascending=False)
top6 = set(imp.head(6).index)
auc1, _, _ = auc_without(top6)
shp_vs = {c for c in feats if c.startswith("shp_vs_") or c.startswith("co_vs_hu")}
auc2, _, _ = auc_without(shp_vs | top6)
auc3, _, _ = auc_without({"ship_month", "is_peak_month"})
print(f"\nAUC semua fitur: {auc0:.4f}")
print(f"AUC tanpa 6 fitur teratas {sorted(top6)}: {auc1:.4f}")
print(f"AUC tanpa semua fitur shp_vs_*/co_vs_hu_* dan 6 teratas: {auc2:.4f}")
print(f"AUC tanpa ship_month/is_peak_month saja: {auc3:.4f}")
print(f"selesai {time.time() - T0:.0f}s")
