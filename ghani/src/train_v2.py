"""Training v2 dari cache fitur: LightGBM + tuning bobot kelas (Macro F1).

Jalankan (setelah features_v2.py train & test selesai):
  python ghani/src/train_v2.py                 # env: SUB_N (default seluruh data), LR, ROUNDS, TAG
Output: ghani/models/lgbm_{TAG}.txt, ghani/models/meta_{TAG}.json, submissions/submission_{TAG}_ghani.csv, ghani/cache/proba_{TAG}_*.npy
"""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import baseline_lgbm as b

TAG = os.environ.get("TAG", "v5")
SUB_N = int(os.environ.get("SUB_N", 10**9))
LR = float(os.environ.get("LR", 0.1))
ROUNDS = int(os.environ.get("ROUNDS", 400))

tr = pd.read_pickle("ghani/cache/features_train.pkl")
te = pd.read_pickle("ghani/cache/features_test.pkl")
b.log(f"loaded {tr.shape} {te.shape}")

classes = sorted(tr["status_penanganan"].unique())
y = tr["status_penanganan"].map({c: i for i, c in enumerate(classes)}).values

if os.environ.get("TE", "1") == "1":
    from sklearn.model_selection import StratifiedKFold

    K = len(classes)
    tr["route"] = tr["kota_asal"].astype(str) + ">" + tr["kota_tujuan"].astype(str)
    te["route"] = te["kota_asal"].astype(str) + ">" + te["kota_tujuan"].astype(str)
    tr["cu_route"] = tr["customer_id"].astype(str) + "|" + tr["goods_category"].astype(str)
    te["cu_route"] = te["customer_id"].astype(str) + "|" + te["goods_category"].astype(str)
    ents = ["customer_id", "courier_id", "hub_id", "package_id", "promo_id", "route", "cu_route"]
    prior = np.bincount(y, minlength=K) / len(y)
    A = 20.0
    Y1 = np.eye(K, dtype="float32")[y]
    skf = StratifiedKFold(5, shuffle=True, random_state=7)
    folds = list(skf.split(np.zeros(len(y)), y))
    for e in ents:
        allv = pd.concat([tr[e], te[e]]).astype(str)
        codes, _ = pd.factorize(allv)
        ctr, cte = codes[: len(tr)], codes[len(tr):]
        ncode = codes.max() + 1
        cnt_all = np.bincount(codes, minlength=ncode)
        tr[f"cnt_{e}"] = cnt_all[ctr].astype("float32")
        te[f"cnt_{e}"] = cnt_all[cte].astype("float32")
        oof = np.zeros((len(tr), K), dtype="float32")
        for a_idx, b_idx in folds:
            s = np.zeros((ncode, K), dtype="float32")
            np.add.at(s, ctr[a_idx], Y1[a_idx])
            n = np.bincount(ctr[a_idx], minlength=ncode).astype("float32")[:, None]
            oof[b_idx] = ((s + A * prior) / (n + A))[ctr[b_idx]]
        s = np.zeros((ncode, K), dtype="float32")
        np.add.at(s, ctr, Y1)
        n = np.bincount(ctr, minlength=ncode).astype("float32")[:, None]
        tet = ((s + A * prior) / (n + A))[cte]
        for k in range(K):
            tr[f"te_{e}_{k}"] = oof[:, k]
            te[f"te_{e}_{k}"] = tet[:, k]
        b.log(f"TE {e}: card={ncode} mean_cnt={cnt_all.mean():.1f}")
    tr = tr.drop(columns=["route", "cu_route"])
    te = te.drop(columns=["route", "cu_route"])
drop = ["shipment_id", "status_penanganan", "customer_id", "courier_id", "package_id", "promo_id", "hub_id"]
feats = [c for c in tr.columns if c not in drop]
cat_cols = [c for c in feats if tr[c].dtype == object]
for c in cat_cols:
    cats = pd.Categorical(pd.concat([tr[c], te[c]]).astype(str)).categories
    tr[c] = pd.Categorical(tr[c].astype(str), categories=cats)
    te[c] = pd.Categorical(te[c].astype(str), categories=cats)

idx = np.arange(len(tr))
if SUB_N < len(tr):
    idx, _ = train_test_split(idx, train_size=SUB_N, stratify=y, random_state=1)
itr, iva = train_test_split(idx, test_size=0.15, stratify=y[idx], random_state=42)
Xtr, Xva = tr.iloc[itr][feats], tr.iloc[iva][feats]
ytr, yva = y[itr], y[iva]

params = dict(objective="multiclass", num_class=len(classes), learning_rate=LR, num_leaves=63, min_data_in_leaf=100,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_bin=63,
              cat_smooth=20, verbose=-1, num_threads=os.cpu_count())
dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat_cols, free_raw_data=False)
dva = lgb.Dataset(Xva, yva, reference=dtr, categorical_feature=cat_cols)
m = lgb.train(params, dtr, num_boost_round=ROUNDS, valid_sets=[dva],
              callbacks=[lgb.early_stopping(30), lgb.log_evaluation(20)])
b.log(f"best_iter {m.best_iteration}")
m.save_model(f"ghani/models/lgbm_{TAG}.txt", num_iteration=m.best_iteration)

imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
b.log("top gain: " + ", ".join(f"{k}={v:.0f}" for k, v in imp.head(15).items()))
imp.to_csv(f"ghani/cache/importance_{TAG}.csv")

pva = m.predict(Xva, num_iteration=m.best_iteration)
base = f1_score(yva, pva.argmax(1), average="macro")
b.log(f"valid macro F1 argmax: {base:.5f}")
w, best = np.ones(len(classes)), base
for _ in range(3):
    for k in range(len(classes)):
        for mult in [0.6, 0.8, 0.9, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0]:
            w2 = w.copy()
            w2[k] *= mult
            sc = f1_score(yva, (pva * w2).argmax(1), average="macro")
            if sc > best + 1e-5:
                best, w = sc, w2
b.log(f"valid macro F1 tuned: {best:.5f} w={np.round(w, 3).tolist()}")

pte = m.predict(te[feats], num_iteration=m.best_iteration)
np.save(f"ghani/cache/proba_{TAG}_test.npy", pte.astype("float32"))
np.save(f"ghani/cache/proba_{TAG}_valid.npy", pva.astype("float32"))
np.save(f"ghani/cache/yva_{TAG}.npy", yva)
sub = pd.DataFrame({"shipment_id": te["shipment_id"], "status_penanganan": [classes[i] for i in (pte * w).argmax(1)]})
sub.to_csv(f"submissions/submission_{TAG}_ghani.csv", index=False)
json.dump({"classes": classes, "weights": w.tolist(), "features": feats, "valid_macro_f1": best,
           "best_iter": m.best_iteration}, open(f"ghani/models/meta_{TAG}.json", "w"))
b.log("done " + str(sub["status_penanganan"].value_counts(normalize=True).round(3).to_dict()))

