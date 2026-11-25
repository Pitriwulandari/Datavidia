"""Feature Engineering v5 (Task-1 Datavidia 11.0):
- Cepat & hemat RAM (vectorized numpy/pandas, chunked dengan carry-over per shipment_id).
- Membuang ~130 kolom noise yang terbukti 0 sinyal (kota_asal/tujuan, bigram, device_type, promo, distraktor).
- Menambahkan sinyal baru yang terverifikasi:
  1. Moda-normalized route delay ratio & excess duration (route_segments.csv)
  2. Base invoice vs Klaim invoice terpisah + tax_rate & amount/declared_value (invoices.csv)
  3. Distribusi lengkap scan_quality (q1..q5, std, pct_le2), log(gap_hours) mean/std, upaya_antar ratio, has_selesai (scan_events.csv)
  4. Step-function domain flags (is_peak_month, is_night_ship, is_outside_hub_hours, has_catatan)
  5. Unsupervised Entity Profiles across ALL 1.6M train+test rows (100% leak-free):
     - Courier profile (~67.5 rows/courier) + Courier-vs-Hub + Shipment-vs-Courier deviations
     - Hub profile (~2455 rows/hub) + Hub x Month congestion & delay profiles
     - Customer Leave-One-Out (LOO) behavioral & courier-residualized profiles

Jalankan dari root repo:
  python ghani/src/build_features_v5.py
Output:
  ghani/cache/v5/feat_v5_train.pkl
  ghani/cache/v5/feat_v5_test.pkl
"""
import gc
import os
import time

import numpy as np
import pandas as pd

ROOT = "Task-1-Datavidia-11.0"
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def agg_scan_fast(path):
    """Vectorized chunked aggregation of scan_events.csv (contiguous by shipment_id)."""
    ev_map = {"pickup": 0, "sortasi": 1, "upaya_antar": 2, "selesai": 3, "transit": 4, "keluar_hub": 5}
    usecols = ["shipment_id", "event_type", "gap_hours", "scan_quality", "is_manual"]
    parts = []
    carry = None
    for ch in pd.read_csv(path, chunksize=1_500_000, usecols=usecols):
        if carry is not None:
            ch = pd.concat([carry, ch], ignore_index=True)
        last_sid = ch["shipment_id"].iloc[-1]
        m = ch["shipment_id"].eq(last_sid)
        carry, ch = ch[m], ch[~m]
        if len(ch):
            parts.append(_agg_scan_chunk(ch, ev_map))
    if carry is not None and len(carry):
        parts.append(_agg_scan_chunk(carry, ev_map))
    out = pd.concat(parts)
    # In case a shipment_id appeared non-contiguously across chunks (safety check)
    if out.index.has_duplicates:
        log("WARNING: non-contiguous shipment_id in scan_events, re-aggregating duplicates")
        out = out.groupby(level=0).first()
    return out


def _agg_scan_chunk(ch, ev_map):
    codes, uniques = pd.factorize(ch["shipment_id"], sort=False)
    n_u = len(uniques)
    cnt = np.bincount(codes, minlength=n_u).astype(np.float32)

    gap = ch["gap_hours"].to_numpy(dtype=np.float64)
    lgap = np.log1p(np.maximum(gap, 0.0))
    q = ch["scan_quality"].to_numpy(dtype=np.float64)
    man = ch["is_manual"].to_numpy(dtype=np.float64)
    ev = ch["event_type"].map(ev_map).to_numpy(dtype=np.int8)

    gap_sum = np.bincount(codes, weights=gap, minlength=n_u)
    gap_sq = np.bincount(codes, weights=gap * gap, minlength=n_u)
    lgap_sum = np.bincount(codes, weights=lgap, minlength=n_u)
    lgap_sq = np.bincount(codes, weights=lgap * lgap, minlength=n_u)

    gap_max = np.full(n_u, -np.inf, dtype=np.float64)
    np.maximum.at(gap_max, codes, gap)
    gap_min = np.full(n_u, np.inf, dtype=np.float64)
    np.minimum.at(gap_min, codes, gap)

    q_sum = np.bincount(codes, weights=q, minlength=n_u)
    q_sq = np.bincount(codes, weights=q * q, minlength=n_u)
    q_min = np.full(n_u, 99.0, dtype=np.float64)
    np.minimum.at(q_min, codes, q)
    q_max = np.full(n_u, -99.0, dtype=np.float64)
    np.maximum.at(q_max, codes, q)

    q1 = np.bincount(codes, weights=(q == 1).astype(np.float64), minlength=n_u)
    q2 = np.bincount(codes, weights=(q == 2).astype(np.float64), minlength=n_u)
    q3 = np.bincount(codes, weights=(q == 3).astype(np.float64), minlength=n_u)
    q4 = np.bincount(codes, weights=(q == 4).astype(np.float64), minlength=n_u)
    q5 = np.bincount(codes, weights=(q == 5).astype(np.float64), minlength=n_u)
    man_sum = np.bincount(codes, weights=man, minlength=n_u)

    cnt_upaya = np.bincount(codes, weights=(ev == 2).astype(np.float64), minlength=n_u)
    cnt_selesai = np.bincount(codes, weights=(ev == 3).astype(np.float64), minlength=n_u)

    gap_mean = gap_sum / cnt
    gap_std = np.sqrt(np.maximum(gap_sq / cnt - gap_mean * gap_mean, 0.0))
    lgap_mean = lgap_sum / cnt
    lgap_std = np.sqrt(np.maximum(lgap_sq / cnt - lgap_mean * lgap_mean, 0.0))
    q_mean = q_sum / cnt
    q_std = np.sqrt(np.maximum(q_sq / cnt - q_mean * q_mean, 0.0))

    df = pd.DataFrame(
        {
            "scan_n": cnt,
            "scan_gap_mean": gap_mean.astype(np.float32),
            "scan_gap_std": gap_std.astype(np.float32),
            "scan_gap_max": gap_max.astype(np.float32),
            "scan_gap_min": gap_min.astype(np.float32),
            "scan_lgap_mean": lgap_mean.astype(np.float32),
            "scan_lgap_std": lgap_std.astype(np.float32),
            "scan_q_mean": q_mean.astype(np.float32),
            "scan_q_std": q_std.astype(np.float32),
            "scan_q_min": q_min.astype(np.float32),
            "scan_q_max": q_max.astype(np.float32),
            "scan_q_pct_1": (q1 / cnt).astype(np.float32),
            "scan_q_pct_2": (q2 / cnt).astype(np.float32),
            "scan_q_pct_le2": ((q1 + q2) / cnt).astype(np.float32),
            "scan_q_pct_3": (q3 / cnt).astype(np.float32),
            "scan_q_pct_4": (q4 / cnt).astype(np.float32),
            "scan_q_pct_5": (q5 / cnt).astype(np.float32),
            "scan_manual_sum": man_sum.astype(np.float32),
            "scan_manual_mean": (man_sum / cnt).astype(np.float32),
            "scan_cnt_upaya": cnt_upaya.astype(np.float32),
            "scan_pct_upaya": (cnt_upaya / cnt).astype(np.float32),
            "scan_has_selesai": (cnt_selesai > 0).astype(np.float32),
            "scan_middle_cnt": (cnt - cnt_upaya - cnt_selesai).astype(np.float32),
        },
        index=pd.Index(uniques, name="shipment_id"),
    )
    return df


def agg_route_fast(path, moda_ref_speed=None):
    """Vectorized route_segments aggregation with MODA-NORMALIZED delay ratios."""
    d = pd.read_csv(path, usecols=["shipment_id", "moda", "jarak_segmen_km", "durasi_jam"])
    if moda_ref_speed is None:
        valid = d["durasi_jam"].notna() & (d["durasi_jam"] > 0)
        spd = d.loc[valid, "jarak_segmen_km"] / d.loc[valid, "durasi_jam"]
        moda_ref_speed = spd.groupby(d.loc[valid, "moda"]).median().to_dict()
        log(f"moda_ref_speed (train median km/h): { {k: round(v, 2) for k, v in moda_ref_speed.items()} }")

    ref_v = d["moda"].map(moda_ref_speed).to_numpy(dtype=np.float64)
    dist = d["jarak_segmen_km"].to_numpy(dtype=np.float64)
    dur = d["durasi_jam"].to_numpy(dtype=np.float64)
    dur_ok = ~np.isnan(dur)

    exp_dur = dist / ref_v
    delay_ratio = np.where(dur_ok, dur / np.maximum(exp_dur, 1e-6), np.nan)
    log_delay = np.where(dur_ok, np.log(np.maximum(delay_ratio, 1e-4)), np.nan)

    codes, uniques = pd.factorize(d["shipment_id"], sort=False)
    n_u = len(uniques)
    seg_n = np.bincount(codes, minlength=n_u).astype(np.float32)
    ok_cnt = np.bincount(codes, weights=dur_ok.astype(np.float64), minlength=n_u)
    na_cnt = seg_n - ok_cnt

    dur_sum = np.bincount(codes, weights=np.where(dur_ok, dur, 0.0), minlength=n_u)
    exp_dur_ok_sum = np.bincount(codes, weights=np.where(dur_ok, exp_dur, 0.0), minlength=n_u)
    exp_dur_all_sum = np.bincount(codes, weights=exp_dur, minlength=n_u)

    dur_max = np.full(n_u, -np.inf, dtype=np.float64)
    np.maximum.at(dur_max, codes[dur_ok], dur[dur_ok])
    dur_max = np.where(ok_cnt > 0, dur_max, np.nan)

    dr_sum = np.bincount(codes, weights=np.where(dur_ok, delay_ratio, 0.0), minlength=n_u)
    ldr_sum = np.bincount(codes, weights=np.where(dur_ok, log_delay, 0.0), minlength=n_u)
    dr_max = np.full(n_u, -np.inf, dtype=np.float64)
    np.maximum.at(dr_max, codes[dur_ok], delay_ratio[dur_ok])
    dr_max = np.where(ok_cnt > 0, dr_max, np.nan)
    dr_min = np.full(n_u, np.inf, dtype=np.float64)
    np.minimum.at(dr_min, codes[dur_ok], delay_ratio[dur_ok])
    dr_min = np.where(ok_cnt > 0, dr_min, np.nan)

    moda = d["moda"].to_numpy()
    m_darat = np.bincount(codes, weights=(moda == "darat").astype(np.float64), minlength=n_u)
    m_kereta = np.bincount(codes, weights=(moda == "kereta").astype(np.float64), minlength=n_u)
    m_laut = np.bincount(codes, weights=(moda == "laut").astype(np.float64), minlength=n_u)
    m_udara = np.bincount(codes, weights=(moda == "udara").astype(np.float64), minlength=n_u)
    m_nuniq = ((m_darat > 0).astype(np.int8) + (m_kereta > 0).astype(np.int8) +
               (m_laut > 0).astype(np.int8) + (m_udara > 0).astype(np.int8)).astype(np.float32)

    tot_delay = np.where(ok_cnt > 0, dur_sum / np.maximum(exp_dur_ok_sum, 1e-6), np.nan)
    dur_sum_clean = np.where(ok_cnt > 0, dur_sum, np.nan)
    dur_excess = np.where(ok_cnt > 0, dur_sum - exp_dur_ok_sum, np.nan)

    out = pd.DataFrame(
        {
            "seg_n": seg_n,
            "seg_dur_sum": dur_sum_clean.astype(np.float32),
            "seg_dur_max": dur_max.astype(np.float32),
            "seg_dur_mean": np.where(ok_cnt > 0, dur_sum / np.maximum(ok_cnt, 1), np.nan).astype(np.float32),
            "seg_dur_na_cnt": na_cnt.astype(np.float32),
            "seg_exp_dur_sum": exp_dur_all_sum.astype(np.float32),
            "seg_total_delay_ratio": tot_delay.astype(np.float32),
            "seg_log_total_delay": np.log(np.maximum(tot_delay, 1e-4)).astype(np.float32),
            "seg_delay_mean": np.where(ok_cnt > 0, dr_sum / np.maximum(ok_cnt, 1), np.nan).astype(np.float32),
            "seg_log_delay_mean": np.where(ok_cnt > 0, ldr_sum / np.maximum(ok_cnt, 1), np.nan).astype(np.float32),
            "seg_delay_max": dr_max.astype(np.float32),
            "seg_delay_min": dr_min.astype(np.float32),
            "seg_dur_excess": dur_excess.astype(np.float32),
            "seg_moda_darat": m_darat.astype(np.float32),
            "seg_moda_kereta": m_kereta.astype(np.float32),
            "seg_moda_laut": m_laut.astype(np.float32),
            "seg_moda_udara": m_udara.astype(np.float32),
            "seg_moda_udara_pct": (m_udara / seg_n).astype(np.float32),
            "seg_moda_laut_pct": (m_laut / seg_n).astype(np.float32),
            "seg_moda_kereta_pct": (m_kereta / seg_n).astype(np.float32),
            "seg_moda_nunique": m_nuniq,
        },
        index=pd.Index(uniques, name="shipment_id"),
    )
    return out, moda_ref_speed


def agg_inv_fast(path):
    """Separate base invoice (pengiriman/langganan) from klaim invoice."""
    usecols = ["shipment_id", "invoice_type", "amount", "discount", "tax", "payment_status", "issued_month", "days_to_pay", "is_penalty"]
    d = pd.read_csv(path, usecols=usecols)
    ps_map = {"lunas": 0, "tertunda": 1, "gagal": 2}
    d["ps_code"] = d["payment_status"].map(ps_map).astype(np.int8)

    base = d[d["invoice_type"] != "klaim"].drop_duplicates("shipment_id", keep="first").set_index("shipment_id")
    kl = d[d["invoice_type"] == "klaim"].drop_duplicates("shipment_id", keep="first").set_index("shipment_id")

    out = pd.DataFrame(index=base.index)
    out["base_inv_is_langganan"] = (base["invoice_type"] == "langganan").astype(np.float32)
    out["base_inv_amount"] = base["amount"].astype(np.float32)
    out["base_inv_log_amount"] = np.log1p(np.maximum(base["amount"].to_numpy(dtype=np.float32), 0.0))
    out["base_inv_discount"] = base["discount"].astype(np.float32)
    out["base_inv_tax"] = base["tax"].astype(np.float32)
    out["base_inv_tax_rate"] = (base["tax"] / np.maximum(base["amount"], 1.0)).astype(np.float32)
    out["base_inv_days"] = base["days_to_pay"].astype(np.float32)
    out["base_inv_ps"] = base["ps_code"].astype(np.float32)
    out["base_inv_penalty"] = base["is_penalty"].astype(np.float32)

    kl_sub = pd.DataFrame(
        {
            "has_klaim_inv": np.ones(len(kl), dtype=np.float32),
            "klaim_inv_amount": kl["amount"].astype(np.float32),
            "klaim_inv_days": kl["days_to_pay"].astype(np.float32),
            "klaim_inv_ps": kl["ps_code"].astype(np.float32),
            "klaim_inv_penalty": kl["is_penalty"].astype(np.float32),
            "klaim_issued_month": kl["issued_month"].astype(np.float32),
        },
        index=kl.index,
    )
    out = out.join(kl_sub, how="left")
    out["has_klaim_inv"] = out["has_klaim_inv"].fillna(0.0).astype(np.float32)
    out["klaim_vs_base_amt"] = (out["klaim_inv_amount"] / np.maximum(out["base_inv_amount"], 1.0)).astype(np.float32)
    out["klaim_days_diff"] = (out["klaim_inv_days"] - out["base_inv_days"]).astype(np.float32)
    out["inv_days_max"] = np.fmax(out["base_inv_days"], out["klaim_inv_days"]).astype(np.float32)
    out["inv_days_mean"] = np.where(
        out["has_klaim_inv"] > 0, 0.5 * (out["base_inv_days"] + out["klaim_inv_days"]), out["base_inv_days"]
    ).astype(np.float32)
    out["inv_any_penalty"] = np.fmax(out["base_inv_penalty"], out["klaim_inv_penalty"].fillna(0.0)).astype(np.float32)
    out["inv_any_tertunda"] = ((out["base_inv_ps"] == 1) | (out["klaim_inv_ps"] == 1)).astype(np.float32)
    out["inv_total_amount"] = (out["base_inv_amount"] + out["klaim_inv_amount"].fillna(0.0)).astype(np.float32)
    return out


def build_split(split, moda_ref_speed=None):
    p = f"{ROOT}/{split}"
    keep_shp = [
        "shipment_id", "customer_id", "courier_id", "hub_id", "package_id",
        "declared_value", "distance_km", "ship_month", "ship_hour", "is_cod",
        "order_to_ship_hours", "prioritas_pengiriman", "jumlah_catatan", "flag_alamat_baru",
        "weight_kg", "volume_l", "qty_items", "coupon_count",
    ]
    if split == "train":
        keep_shp.append("status_penanganan")
    s = pd.read_csv(f"{p}/shipments.csv", usecols=keep_shp)
    log(f"{split}: loaded shipments {s.shape}")

    # Step-function & log transforms on shipments
    s["log_declared_value"] = np.log1p(np.maximum(s["declared_value"].to_numpy(dtype=np.float32), 0.0))
    s["log_distance_km"] = np.log1p(np.maximum(s["distance_km"].to_numpy(dtype=np.float32), 0.0))
    s["is_peak_month"] = s["ship_month"].isin([3, 10, 11, 12]).astype(np.float32)
    s["is_night_ship"] = ((s["ship_hour"] < 6) | (s["ship_hour"] >= 20)).astype(np.float32)
    s["has_catatan"] = (s["jumlah_catatan"] > 0).astype(np.float32)
    s["volume_l_isna"] = s["volume_l"].isna().astype(np.float32)
    s["value_per_kg"] = (s["declared_value"] / (s["weight_kg"] + 1e-6)).astype(np.float32)
    s["log_value_per_kg"] = np.log1p(np.maximum(s["value_per_kg"].to_numpy(dtype=np.float32), 0.0))

    # Merge scan, route, invoices
    sc = agg_scan_fast(f"{p}/scan_events.csv")
    log(f"{split}: scan_events agg {sc.shape}")
    s = s.merge(sc, left_on="shipment_id", right_index=True, how="left")
    del sc
    gc.collect()

    rs, moda_ref_speed = agg_route_fast(f"{p}/route_segments.csv", moda_ref_speed)
    log(f"{split}: route_segments agg {rs.shape}")
    s = s.merge(rs, left_on="shipment_id", right_index=True, how="left")
    del rs
    gc.collect()

    iv = agg_inv_fast(f"{p}/invoices.csv")
    log(f"{split}: invoices agg {iv.shape}")
    s = s.merge(iv, left_on="shipment_id", right_index=True, how="left")
    del iv
    gc.collect()

    # Cross-table derived features
    s["base_amt_per_dec"] = (s["base_inv_amount"] / (s["declared_value"] + 1e-6)).astype(np.float32)
    s["base_amt_per_log_dec"] = (s["base_inv_amount"] / (s["log_declared_value"] + 1e-6)).astype(np.float32)
    s["klaim_mdiff"] = (s["klaim_issued_month"] - s["ship_month"]).astype(np.float32)
    s.drop(columns=["klaim_issued_month"], inplace=True)
    s["scan_per_seg"] = (s["scan_n"] / (s["seg_n"] + 1e-6)).astype(np.float32)
    s["total_time_proxy"] = (s["seg_dur_sum"].fillna(0.0) + s["scan_gap_mean"] * s["scan_n"]).astype(np.float32)

    # Master tables (signal columns only)
    cu = pd.read_csv(
        f"{p}/customers.csv",
        usecols=["customer_id", "address_quality", "complaint_index", "credit_score", "preferred_payment", "segment", "loyalty_tier"],
    ).add_prefix("cu_").rename(columns={"cu_customer_id": "customer_id"})
    cu["cu_credit_score_isna"] = cu["cu_credit_score"].isna().astype(np.float32)

    co = pd.read_csv(
        f"{p}/couriers.csv",
        usecols=["courier_id", "courier_score", "experience_years", "vehicle_type", "contract_type", "shift_code", "training_level"],
    ).add_prefix("co_").rename(columns={"co_courier_id": "courier_id"})
    co["co_courier_score_isna"] = co["co_courier_score"].isna().astype(np.float32)

    hu = pd.read_csv(
        f"{p}/hubs.csv",
        usecols=["hub_id", "region", "hub_type", "automation_level", "capacity_index", "density_index", "luas_gudang_m2", "jam_buka", "jam_tutup"],
    ).add_prefix("hu_").rename(columns={"hu_hub_id": "hub_id"})
    hu["hu_open_hours"] = (hu["hu_jam_tutup"] - hu["hu_jam_buka"]).astype(np.float32)

    sp = pd.read_csv(
        f"{p}/service_packages.csv",
        usecols=["package_id", "speed_class", "handling_class", "fragile_score", "estimasi_hari", "cost_multiplier", "biaya_dasar", "max_weight_kg"],
    ).add_prefix("sp_").rename(columns={"sp_package_id": "package_id"})

    s = (
        s.merge(cu, on="customer_id", how="left")
        .merge(co, on="courier_id", how="left")
        .merge(hu, on="hub_id", how="left")
        .merge(sp, on="package_id", how="left")
    )

    s["is_outside_hub_hours"] = ((s["ship_hour"] < s["hu_jam_buka"]) | (s["ship_hour"] > s["hu_jam_tutup"])).astype(np.float32)
    s["est_vs_dur"] = (s["seg_dur_sum"] / (s["sp_estimasi_hari"] * 24.0 + 1e-6)).astype(np.float32)
    s["hu_dens_x_cap"] = (s["hu_density_index"] * s["hu_capacity_index"]).astype(np.float32)
    s["cu_bad_addr_x_comp"] = ((6.0 - s["cu_address_quality"]) * s["cu_complaint_index"]).astype(np.float32)
    s["co_bad_x_dist"] = ((6.0 - s["co_courier_score"].fillna(3.0)) * s["log_distance_km"]).astype(np.float32)
    s["dec_x_low_auto"] = (s["log_declared_value"] * (6.0 - s["hu_automation_level"])).astype(np.float32)

    # Downcast float64/int64
    for c in s.columns:
        if s[c].dtype == np.float64:
            s[c] = s[c].astype(np.float32)
        elif s[c].dtype == np.int64:
            s[c] = s[c].astype(np.int32)

    log(f"{split}: base features ready {s.shape}")
    return s, moda_ref_speed


def add_unsupervised_entity_profiles(tr, te):
    """Compute leak-free unsupervised entity behavioral profiles across ALL 1.6M train+test rows."""
    n_tr = len(tr)
    key_cols = ["customer_id", "courier_id", "hub_id", "ship_month"]
    prof_src = [
        "scan_gap_mean", "scan_lgap_mean", "scan_q_mean", "scan_q_pct_le2", "scan_manual_mean",
        "seg_total_delay_ratio", "seg_log_delay_mean", "seg_dur_sum",
        "inv_days_max", "base_inv_amount", "has_klaim_inv", "inv_any_penalty",
        "scan_cnt_upaya", "scan_has_selesai", "jumlah_catatan", "has_catatan",
        "flag_alamat_baru", "log_declared_value", "distance_km", "is_cod", "is_night_ship",
    ]
    all_df = pd.concat([tr[key_cols + prof_src], te[key_cols + prof_src]], ignore_index=True)
    all_df["no_selesai"] = (1.0 - all_df["scan_has_selesai"]).astype(np.float32)
    log(f"entity profiles on combined {all_df.shape}")

    def _group_mean_and_cnt(df, grp_col, cols, prefix, loo=False):
        codes, uniques = pd.factorize(df[grp_col], sort=False)
        n_u = len(uniques)
        cnt_all = np.bincount(codes, minlength=n_u).astype(np.float32)
        res = {f"{prefix}_n": cnt_all[codes]}
        for col in cols:
            x = df[col].to_numpy(dtype=np.float64)
            ok = ~np.isnan(x)
            s_ok = np.bincount(codes, weights=np.where(ok, x, 0.0), minlength=n_u)
            c_ok = np.bincount(codes, weights=ok.astype(np.float64), minlength=n_u)
            if loo:
                num = s_ok[codes] - np.where(ok, x, 0.0)
                den = c_ok[codes] - ok.astype(np.float64)
                val = np.where(den > 0, num / np.maximum(den, 1.0), np.nan)
            else:
                val = np.where(c_ok[codes] > 0, s_ok[codes] / np.maximum(c_ok[codes], 1.0), np.nan)
            res[f"{prefix}_{col}"] = val.astype(np.float32)
        return res

    # 1. Hub profiles (~2455 rows/hub)
    hu_cols = [
        "scan_lgap_mean", "scan_gap_mean", "seg_log_delay_mean", "seg_total_delay_ratio", "seg_dur_sum",
        "scan_q_mean", "scan_manual_mean", "inv_days_max", "has_klaim_inv", "log_declared_value", "base_inv_amount", "inv_any_penalty",
    ]
    hu_p = _group_mean_and_cnt(all_df, "hub_id", hu_cols, "hu_p", loo=False)

    # 2. Courier profiles (~67.5 rows/courier — LOO so zero self-correlation!)
    co_cols = [
        "scan_lgap_mean", "scan_gap_mean", "scan_q_mean", "scan_q_pct_le2", "scan_manual_mean",
        "seg_log_delay_mean", "seg_total_delay_ratio", "seg_dur_sum",
        "inv_days_max", "has_klaim_inv", "scan_cnt_upaya", "no_selesai", "inv_any_penalty", "log_declared_value",
    ]
    co_p = _group_mean_and_cnt(all_df, "courier_id", co_cols, "co_p", loo=True)

    # 3. Hub x Month profiles (~205 rows/group — LOO!)
    all_df["hub_month"] = all_df["hub_id"].astype(str) + "_" + all_df["ship_month"].astype(str)
    hm_cols = ["scan_lgap_mean", "seg_log_delay_mean", "inv_days_max"]
    hm_p = _group_mean_and_cnt(all_df, "hub_month", hm_cols, "hm_p", loo=True)
    hm_p["hm_p_n_ratio"] = (hm_p["hm_p_n"] / (hu_p["hu_p_n"] / 12.0 + 1e-6)).astype(np.float32)
    hm_p["hm_lgap_vs_hu"] = (hm_p["hm_p_scan_lgap_mean"] - hu_p["hu_p_scan_lgap_mean"]).astype(np.float32)
    hm_p["hm_delay_vs_hu"] = (hm_p["hm_p_seg_log_delay_mean"] - hu_p["hu_p_seg_log_delay_mean"]).astype(np.float32)

    # 4. Courier vs Hub & Shipment vs Courier deviations!
    dev = {
        "co_vs_hu_lgap": (co_p["co_p_scan_lgap_mean"] - hu_p["hu_p_scan_lgap_mean"]).astype(np.float32),
        "co_vs_hu_delay": (co_p["co_p_seg_log_delay_mean"] - hu_p["hu_p_seg_log_delay_mean"]).astype(np.float32),
        "co_vs_hu_scan_q": (co_p["co_p_scan_q_mean"] - hu_p["hu_p_scan_q_mean"]).astype(np.float32),
        "shp_vs_co_lgap": (all_df["scan_lgap_mean"].to_numpy(np.float32) - co_p["co_p_scan_lgap_mean"]).astype(np.float32),
        "shp_vs_co_scan_q": (all_df["scan_q_mean"].to_numpy(np.float32) - co_p["co_p_scan_q_mean"]).astype(np.float32),
        "shp_vs_co_delay": (all_df["seg_log_delay_mean"].to_numpy(np.float32) - co_p["co_p_seg_log_delay_mean"]).astype(np.float32),
    }

    # 5. Customer LOO profiles (raw + courier-residualized!)
    all_df["scan_q_res_co"] = dev["shp_vs_co_scan_q"]
    all_df["lgap_res_co"] = dev["shp_vs_co_lgap"]
    all_df["delay_res_co"] = dev["shp_vs_co_delay"]
    cu_cols = [
        "jumlah_catatan", "has_catatan", "scan_cnt_upaya", "no_selesai", "has_klaim_inv",
        "inv_days_max", "base_inv_amount", "scan_q_mean", "scan_q_res_co",
        "scan_manual_mean", "lgap_res_co", "delay_res_co",
        "distance_km", "flag_alamat_baru", "is_cod", "is_night_ship", "log_declared_value",
    ]
    cu_p = _group_mean_and_cnt(all_df, "customer_id", cu_cols, "cu_loo", loo=True)

    # Combine all new profile columns cleanly
    new_cols = {}
    for d in (hu_p, co_p, hm_p, dev, cu_p):
        new_cols.update(d)
    prof_df = pd.DataFrame(new_cols)
    log(f"built {prof_df.shape[1]} unsupervised entity profile features")

    tr = pd.concat([tr.reset_index(drop=True), prof_df.iloc[:n_tr].reset_index(drop=True)], axis=1)
    te = pd.concat([te.reset_index(drop=True), prof_df.iloc[n_tr:].reset_index(drop=True)], axis=1)
    return tr, te


def main():
    os.makedirs("ghani/cache", exist_ok=True)
    tr, moda_ref = build_split("train", None)
    te, _ = build_split("test", moda_ref)
    tr, te = add_unsupervised_entity_profiles(tr, te)

    # Convert low-cardinality string columns to category (consistent across train/test)
    id_cols = {"shipment_id", "status_penanganan", "customer_id", "courier_id", "hub_id", "package_id"}
    for c in tr.columns:
        if c not in id_cols and tr[c].dtype == object:
            cats = pd.Categorical(pd.concat([tr[c], te[c]]).astype(str)).categories
            tr[c] = pd.Categorical(tr[c].astype(str), categories=cats)
            te[c] = pd.Categorical(te[c].astype(str), categories=cats)

    # Also integer-encode entity IDs compactly (huge RAM & speed win for CV / GLMM / TE!)
    for eid in ["customer_id", "courier_id", "hub_id", "package_id"]:
        all_v = pd.concat([tr[eid], te[eid]]).astype(str)
        codes, _ = pd.factorize(all_v, sort=False)
        tr[eid + "_code"] = codes[: len(tr)].astype(np.int32)
        te[eid + "_code"] = codes[len(tr):].astype(np.int32)

    tr.to_pickle("ghani/cache/v5/feat_v5_train.pkl")
    te.to_pickle("ghani/cache/v5/feat_v5_test.pkl")
    log(f"SAVED feat_v5_train.pkl {tr.shape} ({os.path.getsize('ghani/cache/v5/feat_v5_train.pkl')/1e6:.1f} MB)")
    log(f"SAVED feat_v5_test.pkl  {te.shape} ({os.path.getsize('ghani/cache/v5/feat_v5_test.pkl')/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
