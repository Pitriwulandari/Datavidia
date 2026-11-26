"""Fitur v2: semua fitur baseline + fitur urutan scan (event pertama/terakhir, bigram transisi, perubahan hub, dst).
Hasil di-cache ke ghani/task1/cache/features_{split}.pkl supaya eksperimen tidak mengulang agregasi 9 menit.

Jalankan:  python ghani/task1/src/features_v2.py train   |   python ghani/task1/src/features_v2.py test
"""
import os
import sys

import numpy as np
import pandas as pd

import os, sys
for _sub in ['common', 'v1', 'v2', 'v5', 'v6', 'v7']:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)
import baseline_lgbm as b

EV = {"pickup": 0, "sortasi": 1, "upaya_antar": 2, "selesai": 3, "transit": 4, "keluar_hub": 5}
CH = 1_000_000


def _seq(d):
    d = d.reset_index(drop=True)
    code = d["event_type"].map(EV).astype("int8")
    sid = d["shipment_id"]
    same = sid.eq(sid.shift())
    prev = code.shift().where(same, -1).fillna(-1).astype("int8")
    hub = pd.Series(pd.factorize(d["hub_id"])[0])
    hub_chg = (hub.ne(hub.shift()) & same).astype("int8")
    pos = d.groupby("shipment_id").cumcount()
    n = d.groupby("shipment_id")["event_type"].transform("size")
    out = pd.DataFrame(index=pd.Index(sid.unique(), name="shipment_id"))
    out["seq_first"] = code[pos == 0].values
    last = pos == n - 1
    out["seq_last"] = pd.Series(code[last].values, index=sid[last].values)
    out["seq_last_gap"] = pd.Series(d["gap_hours"][last].values, index=sid[last].values)
    out["seq_last_q"] = pd.Series(d["scan_quality"][last].values, index=sid[last].values)
    out["seq_last_manual"] = pd.Series(d["is_manual"][last].values, index=sid[last].values)
    sl = pos == n - 2
    out["seq_last2"] = pd.Series(code[sl].values, index=sid[sl].values)
    out["seq_hub_chg"] = hub_chg.groupby(sid.values).sum()
    out["seq_lowq"] = (d["scan_quality"] <= 2).groupby(sid.values).sum()
    out["seq_gap_after_up"] = d["gap_hours"][code == 2].groupby(sid[code == 2].values).mean()
    out["seq_gap_sel"] = d["gap_hours"][code == 3].groupby(sid[code == 3].values).mean()
    out["seq_gap_std"] = d["gap_hours"].groupby(sid.values).std()
    out["seq_manual_last_half"] = d["is_manual"][pos >= n // 2].groupby(sid[pos >= n // 2].values).sum()
    v = prev >= 0
    big = (prev[v] * 10 + code[v]).astype(int)
    bg = pd.crosstab(sid[v].values, big.values).add_prefix("bg_")
    bg.index.name = "shipment_id"
    return out.join(bg).astype("float32")


def seq_agg(path):
    parts, carry = [], None
    for ch in pd.read_csv(path, chunksize=CH, usecols=["shipment_id", "hub_id", "event_type", "gap_hours", "scan_quality", "is_manual"]):
        if carry is not None:
            ch = pd.concat([carry, ch], ignore_index=True)
        m = ch["shipment_id"].eq(ch["shipment_id"].iloc[-1])
        carry, ch = ch[m], ch[~m]
        if len(ch):
            parts.append(_seq(ch))
    parts.append(_seq(carry))
    o = pd.concat(parts)
    bgc = [c for c in o.columns if c.startswith("bg_")]
    o[bgc] = o[bgc].fillna(0)
    return o


if __name__ == "__main__":
    split = sys.argv[1]
    os.makedirs("ghani/task1/cache", exist_ok=True)
    s = b.build(split)
    q = seq_agg(f"{b.ROOT}/{split}/scan_events.csv")
    b.log(f"{split}: seq feats {q.shape}")
    s = s.merge(q, left_on="shipment_id", right_index=True, how="left")
    s["seq_gap_up_ratio"] = s["seq_gap_after_up"] / (s["scan_gap_mean"] + 1e-6)
    s["scan_per_seg"] = s["scan_n"] / (s["seg_n"] + 1e-6)
    s.to_pickle(f"ghani/task1/cache/features_{split}.pkl")
    b.log(f"{split}: saved {s.shape}")

