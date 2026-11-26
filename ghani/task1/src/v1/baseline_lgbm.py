"""Baseline cepat Task-1 Datavidia 11.0: agregasi tabel relasional (hemat RAM, chunked) + LightGBM + tuning bobot kelas (Macro F1).

Jalankan:  python ghani/task1/src/baseline_lgbm.py        (env SUB_N=450000 untuk ukuran subsample training)
Output  :  ghani/task1/models/v1/lgbm_v1.txt, ghani/task1/models/v1/meta_v1.json, submissions/task1/submission_v1_ghani.csv
"""
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = "Task-1-Datavidia-11.0"
CHUNK = 1_500_000
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def chunked(path, fn):
    """Agregasi per-chunk lalu gabungkan: *max -> max, *_min -> min, lainnya -> sum."""
    parts = [fn(ch) for ch in pd.read_csv(path, chunksize=CHUNK)]
    d = pd.concat(parts)
    agg = {}
    for c in d.columns:
        agg[c] = "max" if ("_max" in c or "gapmax" in c) else ("min" if c.endswith("_min") else "sum")
    out = d.groupby(level=0).agg(agg)
    for c in out.columns:
        if agg[c] == "sum":
            out[c] = out[c].fillna(0)
    return out.astype("float32")


def _scan(d):
    g = d.groupby("shipment_id")
    out = g.agg(scan_n=("event_id", "count"), scan_gap_sum=("gap_hours", "sum"), scan_gap_max=("gap_hours", "max"),
                scan_q_sum=("scan_quality", "sum"), scan_q_min=("scan_quality", "min"), scan_manual_sum=("is_manual", "sum"))
    cnt = pd.crosstab(d["shipment_id"], d["event_type"]).add_prefix("scan_cnt_")
    gap = d.pivot_table(index="shipment_id", columns="event_type", values="gap_hours", aggfunc="max").add_prefix("scan_gapmax_")
    dev = pd.crosstab(d["shipment_id"], d["device_type"]).add_prefix("scan_dev_")
    return out.join(cnt).join(gap).join(dev)


def agg_scan(path):
    o = chunked(path, _scan)
    o["scan_gap_mean"] = o["scan_gap_sum"] / o["scan_n"]
    o["scan_q_mean"] = o["scan_q_sum"] / o["scan_n"]
    o["scan_manual_mean"] = o["scan_manual_sum"] / o["scan_n"]
    return o


def _route(d):
    g = d.groupby("shipment_id")
    out = g.agg(seg_n=("segment_id", "count"), seg_dist_sum=("jarak_segmen_km", "sum"), seg_dist_max=("jarak_segmen_km", "max"),
                seg_dur_sum=("durasi_jam", "sum"), seg_dur_max=("durasi_jam", "max"), seg_cost_sum=("biaya_segmen", "sum"))
    return out.join(pd.crosstab(d["shipment_id"], d["moda"]).add_prefix("seg_moda_"))


def agg_route(path):
    o = chunked(path, _route)
    o["seg_speed"] = o["seg_dist_sum"] / (o["seg_dur_sum"] + 1e-6)
    return o


def _inv(d):
    d = d[d["shipment_id"].notna()]
    g = d.groupby("shipment_id")
    out = g.agg(inv_n=("invoice_id", "count"), inv_amount_sum=("amount", "sum"), inv_discount_sum=("discount", "sum"),
                inv_tax_sum=("tax", "sum"), inv_days_max=("days_to_pay", "max"), inv_penalty_sum=("is_penalty", "sum"))
    for col, pre in [("payment_status", "inv_ps_"), ("invoice_type", "inv_type_"), ("payment_method", "inv_pm_")]:
        out = out.join(pd.crosstab(d["shipment_id"], d[col]).add_prefix(pre))
    return out


def agg_inv(path):
    return chunked(path, _inv)


def build(split):
    p = f"{ROOT}/{split}"
    s = pd.read_csv(f"{p}/shipments.csv")
    num = s.select_dtypes("float64").columns
    s[num] = s[num].astype("float32")
    log(f"{split}: shipments {s.shape}")
    for name, fn in [("scan_events", agg_scan), ("route_segments", agg_route), ("invoices", agg_inv)]:
        a = fn(f"{p}/{name}.csv")
        s = s.merge(a, left_on="shipment_id", right_index=True, how="left")
        log(f"{split}: merged {name} -> {s.shape}")
    cu = pd.read_csv(f"{p}/customers.csv").add_prefix("cu_").rename(columns={"cu_customer_id": "customer_id"})
    co = pd.read_csv(f"{p}/couriers.csv").add_prefix("co_").rename(columns={"co_courier_id": "courier_id"})
    hu = pd.read_csv(f"{p}/hubs.csv").add_prefix("hu_").rename(columns={"hu_hub_id": "hub_id"})
    sp = pd.read_csv(f"{p}/service_packages.csv").add_prefix("sp_").rename(columns={"sp_package_id": "package_id"})
    s = (s.merge(cu, on="customer_id", how="left")
          .merge(co, on="courier_id", how="left")
          .merge(hu, on="hub_id", how="left")
          .merge(sp, on="package_id", how="left"))
    num = s.select_dtypes("float64").columns
    s[num] = s[num].astype("float32")
    s["vol_ratio"] = s["berat_volumetrik"] / (s["weight_kg"] + 1e-6)
    s["value_per_kg"] = s["declared_value"] / (s["weight_kg"] + 1e-6)
    s["est_vs_dur"] = s["seg_dur_sum"] / (s["sp_estimasi_hari"] * 24 + 1e-6)
    return s


def main():
    os.makedirs("ghani/task1/models", exist_ok=True)
    os.makedirs("submissions", exist_ok=True)
    tr = build("train")
    te = build("test")

    classes = sorted(tr["status_penanganan"].unique())
    cmap = {c: i for i, c in enumerate(classes)}
    y = tr["status_penanganan"].map(cmap).values
    drop = ["shipment_id", "status_penanganan", "customer_id", "courier_id", "package_id", "promo_id", "hub_id"]
    feats = [c for c in tr.columns if c not in drop]
    cat_cols = [c for c in feats if tr[c].dtype == object]
    for c in cat_cols:
        cats = pd.Categorical(pd.concat([tr[c], te[c]]).astype(str)).categories
        tr[c] = pd.Categorical(tr[c].astype(str), categories=cats)
        te[c] = pd.Categorical(te[c].astype(str), categories=cats)
    log(f"features: {len(feats)} (cat {len(cat_cols)})")

    sub_n = int(os.environ.get("SUB_N", 450000))
    idx_all = np.arange(len(tr))
    if sub_n < len(tr):
        idx_all, _ = train_test_split(idx_all, train_size=sub_n, stratify=y, random_state=1)
    idx_tr, idx_va = train_test_split(idx_all, test_size=0.2, stratify=y[idx_all], random_state=42)
    Xtr, Xva = tr.iloc[idx_tr][feats], tr.iloc[idx_va][feats]
    ytr, yva = y[idx_tr], y[idx_va]

    params = dict(objective="multiclass", num_class=len(classes), learning_rate=0.15, num_leaves=63,
                  min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=5.0, max_bin=127, verbose=-1, num_threads=os.cpu_count())
    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat_cols, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, reference=dtr, categorical_feature=cat_cols)
    model = lgb.train(params, dtr, num_boost_round=300, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(25), lgb.log_evaluation(25)])
    log(f"best_iter {model.best_iteration}")
    model.save_model("ghani/task1/models/v1/lgbm_v1.txt", num_iteration=model.best_iteration)

    pva = model.predict(Xva, num_iteration=model.best_iteration)
    base = f1_score(yva, pva.argmax(1), average="macro")
    log(f"valid macro F1 (argmax): {base:.5f}")

    w = np.ones(len(classes))
    best = base
    for _ in range(3):
        for k in range(len(classes)):
            for m in [0.6, 0.8, 0.9, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0]:
                w2 = w.copy()
                w2[k] = w[k] * m
                sc = f1_score(yva, (pva * w2).argmax(1), average="macro")
                if sc > best + 1e-5:
                    best, w = sc, w2
    log(f"valid macro F1 (tuned weights): {best:.5f} weights={np.round(w, 3).tolist()}")

    pte = model.predict(te[feats], num_iteration=model.best_iteration)
    pred = (pte * w).argmax(1)
    sub = pd.DataFrame({"shipment_id": te["shipment_id"], "status_penanganan": [classes[i] for i in pred]})
    sub.to_csv("submissions/task1/submission_v1_ghani.csv", index=False)
    json.dump({"classes": classes, "weights": w.tolist(), "features": feats, "cat_cols": cat_cols,
               "valid_macro_f1": best, "best_iter": model.best_iteration}, open("ghani/task1/models/v1/meta_v1.json", "w"))
    log("done. " + str(sub["status_penanganan"].value_counts(normalize=True).round(3).to_dict()))


if __name__ == "__main__":
    main()

