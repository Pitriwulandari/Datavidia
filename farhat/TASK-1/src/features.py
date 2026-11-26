"""Transformasi data: feature engineering, encoding kategorikal, target encoding.

Semua agregasi tabel anak (scan_events, route_segments, invoices) dilakukan dengan
polars agar muat di RAM 8 GB / 2 core; hasilnya disimpan ke ``data/processed``.
"""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import StratifiedKFold

from .data_loader import CLASSES, PROCESSED_DIR, path_of

# Fitur dengan kontribusi gain <= 0,01% pada uji importance (eksperimen 400 ribu
# baris + validasi penuh) dibuang agar matriks fitur 1,24 juta baris muat di RAM.
DROP_COLS = [
    "prm_month_offset", "pk_speed", "moda_darat_n", "ship_dow", "hub_area",
    "moda_laut_n", "cus_kota_tipe", "pickup_type", "man_sum", "co_contract",
    "prm_disc", "qty_items", "co_shift", "ev_sortasi_n", "dev_pda_n",
    "prioritas_pengiriman", "pk_est_days", "sc_n_dev", "co_train", "cus_notif",
    "prm_start", "co_capacity", "ev_pickup_n", "dev_gerbang_otomatis_n",
    "km_gap_route", "moda_kereta_n", "hub_region", "pk_max_w", "ev_transit_n",
    "moda_kereta_sh", "channel", "inv_disc_na_sh", "pk_handling", "prm_end",
    "ev_keluar_hub_n", "dev_manual_n", "cus_segment", "cus_same_region", "hub_hours",
    "pk_ins_def", "cus_verified", "cus_loyalty", "coupon_count", "cus_business",
    "cus_tagihan", "km_ratio_route", "hub_type", "co_vehicle", "dest_same_region",
    "insurance_opt", "pay_problem_n", "fragility_flag", "inv_langganan_n",
    "hub_open", "hub_close", "inv_pen_sum", "co_hazmat", "pay_problem_sh", "pk_cod",
    "is_promo", "pay_tertunda_n", "gift_wrap", "is_cod", "ev_selesai_n", "same_city",
    "prm_type", "inv_pengiriman_n", "no_selesai", "pay_gagal_n", "courier_at_origin",
    "first_hub_is_origin", "sc_n_hub", "rs_n_hub", "volume_isna",
    "last_hub_is_origin",
]

TE_COLS = ["courier_id", "hub_id", "customer_id"]
N_CLASS = len(CLASSES)
SMOOTH = 50

EVENT_TYPES = ["pickup", "upaya_antar", "selesai", "keluar_hub", "sortasi", "transit"]
DEVICE_TYPES = ["scanner", "pda", "manual", "gerbang_otomatis"]


# ------------------------------------------------------------------ agregasi anak
def scan_feats(split: str) -> pl.DataFrame:
    """Agregasi scan_events per shipment: volume scan, kualitas, jeda waktu, jenis event."""
    sc = pl.read_csv(path_of(split, "scan_events.csv"))
    aggs = [
        pl.len().alias("sc_n"),
        pl.col("hub_id").n_unique().alias("sc_n_hub"),
        pl.col("device_type").n_unique().alias("sc_n_dev"),
        pl.col("gap_hours").mean().alias("gap_mean"),
        pl.col("gap_hours").std().alias("gap_std"),
        pl.col("gap_hours").max().alias("gap_max"),
        pl.col("gap_hours").min().alias("gap_min"),
        pl.col("gap_hours").sum().alias("gap_sum"),
        pl.col("gap_hours").quantile(0.9).alias("gap_q90"),
        pl.col("scan_quality").mean().alias("q_mean"),
        pl.col("scan_quality").std().alias("q_std"),
        pl.col("scan_quality").min().alias("q_min"),
        pl.col("scan_quality").max().alias("q_max"),
        pl.col("is_manual").mean().alias("man_mean"),
        pl.col("is_manual").sum().alias("man_sum"),
    ]
    for e in EVENT_TYPES:
        aggs.append((pl.col("event_type") == e).sum().alias(f"ev_{e}_n"))
    for d in DEVICE_TYPES:
        aggs.append((pl.col("device_type") == d).sum().alias(f"dev_{d}_n"))
    aggs += [
        pl.col("scan_quality").filter(pl.col("event_type") == "upaya_antar").mean().alias("q_upaya"),
        pl.col("gap_hours").filter(pl.col("event_type") == "upaya_antar").mean().alias("gap_upaya"),
        pl.col("gap_hours").filter(pl.col("event_type") == "selesai").mean().alias("gap_selesai"),
    ]
    f = sc.group_by("shipment_id").agg(aggs)
    for e in EVENT_TYPES:
        f = f.with_columns((pl.col(f"ev_{e}_n") / pl.col("sc_n")).alias(f"ev_{e}_sh"))
    for d in DEVICE_TYPES:
        f = f.with_columns((pl.col(f"dev_{d}_n") / pl.col("sc_n")).alias(f"dev_{d}_sh"))
    f = f.with_columns((pl.col("gap_max") / (pl.col("gap_mean") + 1e-6)).alias("gap_max_over_mean"))
    del sc
    gc.collect()
    return f


def route_feats(split: str) -> pl.DataFrame:
    """Agregasi route_segments per shipment: jumlah segmen, jarak, durasi, biaya, moda."""
    rs = pl.read_csv(path_of(split, "route_segments.csv"))
    modas = ["darat", "udara", "laut", "kereta"]
    aggs = [
        pl.len().alias("rs_n"),
        pl.col("segmen_hub_id").n_unique().alias("rs_n_hub"),
        pl.col("jarak_segmen_km").sum().alias("rs_km_sum"),
        pl.col("jarak_segmen_km").mean().alias("rs_km_mean"),
        pl.col("jarak_segmen_km").max().alias("rs_km_max"),
        pl.col("durasi_jam").sum().alias("rs_dur_sum"),
        pl.col("durasi_jam").mean().alias("rs_dur_mean"),
        pl.col("durasi_jam").max().alias("rs_dur_max"),
        pl.col("durasi_jam").std().alias("rs_dur_std"),
        pl.col("durasi_jam").is_null().mean().alias("rs_dur_na_sh"),
        pl.col("biaya_segmen").sum().alias("rs_cost_sum"),
        pl.col("biaya_segmen").mean().alias("rs_cost_mean"),
        pl.col("biaya_segmen").max().alias("rs_cost_max"),
        pl.col("urutan").max().alias("rs_urutan_max"),
    ]
    for m in modas:
        aggs.append((pl.col("moda") == m).sum().alias(f"moda_{m}_n"))
    aggs += [
        pl.col("moda").sort_by("urutan").first().alias("moda_first"),
        pl.col("moda").sort_by("urutan").last().alias("moda_last"),
        pl.col("jarak_segmen_km").sort_by("urutan").first().alias("rs_km_first"),
        (pl.col("durasi_jam").sum() / (pl.col("jarak_segmen_km").sum() + 1e-6)).alias("dur_per_km"),
        (pl.col("biaya_segmen").sum() / (pl.col("jarak_segmen_km").sum() + 1e-6)).alias("cost_per_km"),
        pl.col("segmen_hub_id").sort_by("urutan").first().alias("first_hub"),
        pl.col("segmen_hub_id").sort_by("urutan").last().alias("last_hub"),
    ]
    f = rs.group_by("shipment_id").agg(aggs)
    for m in modas:
        f = f.with_columns((pl.col(f"moda_{m}_n") / pl.col("rs_n")).alias(f"moda_{m}_sh"))
    del rs
    gc.collect()
    return f


def inv_feats(split: str) -> pl.DataFrame:
    """Agregasi invoices per shipment: jenis tagihan, nominal, diskon, status bayar."""
    iv = pl.read_csv(path_of(split, "invoices.csv"))
    aggs = [
        pl.len().alias("inv_n"),
        pl.col("invoice_type").n_unique().alias("inv_n_type"),
        pl.col("amount").sum().alias("inv_amt_sum"),
        pl.col("amount").mean().alias("inv_amt_mean"),
        pl.col("amount").max().alias("inv_amt_max"),
        pl.col("discount").sum().alias("inv_disc_sum"),
        pl.col("discount").mean().alias("inv_disc_mean"),
        pl.col("discount").is_null().mean().alias("inv_disc_na_sh"),
        pl.col("tax").sum().alias("inv_tax_sum"),
        pl.col("days_to_pay").mean().alias("inv_d2p_mean"),
        pl.col("days_to_pay").max().alias("inv_d2p_max"),
        pl.col("is_penalty").sum().alias("inv_pen_sum"),
        pl.col("issued_month").mean().alias("inv_month_mean"),
    ]
    for t in ["klaim", "langganan", "pengiriman"]:
        aggs.append((pl.col("invoice_type") == t).sum().alias(f"inv_{t}_n"))
    for s in ["lunas", "tertunda", "gagal"]:
        aggs.append((pl.col("payment_status") == s).sum().alias(f"pay_{s}_n"))
    f = iv.group_by("shipment_id").agg(aggs)
    f = f.with_columns([
        (pl.col("inv_amt_sum") / (pl.col("inv_n") + 1e-6)).alias("inv_amt_per_inv"),
        (pl.col("inv_disc_sum") / (pl.col("inv_amt_sum") + 1e-6)).alias("inv_disc_rate"),
        (pl.col("pay_tertunda_n") + pl.col("pay_gagal_n")).alias("pay_problem_n"),
    ])
    f = f.with_columns((pl.col("pay_problem_n") / pl.col("inv_n")).alias("pay_problem_sh"))
    del iv
    gc.collect()
    return f


# ---------------------------------------------------------------- fitur gabungan
def build_features(split: str) -> pl.DataFrame:
    """Bangun seluruh fitur level-shipment untuk ``split`` dan simpan ke parquet."""
    sh = pl.read_csv(path_of(split, "shipments.csv"))
    sh = sh.with_columns([
        (pl.col("weight_kg") * 1000 / (pl.col("distance_km") + 1)).alias("w_per_km"),
        (pl.col("declared_value") / (pl.col("weight_kg") + 1e-6)).alias("value_per_kg"),
        (pl.col("berat_volumetrik") / (pl.col("weight_kg") + 1e-6)).alias("vol_w_ratio"),
        (pl.col("volume_l") / (pl.col("weight_kg") + 1e-6)).alias("vol_per_kg"),
        (pl.col("declared_value") / (pl.col("volume_l") + 1e-6)).alias("value_per_l"),
        (pl.col("distance_km") / (pl.col("order_to_ship_hours") + 1)).alias("km_per_hour"),
        pl.col("declared_value").log1p().alias("log_value"),
        pl.col("volume_l").is_null().cast(pl.Int8).alias("volume_isna"),
        (pl.col("ship_hour") % 24).alias("hour_mod"),
    ])

    cu = pl.read_csv(path_of(split, "customers.csv")).rename({
        "region": "cus_region", "segment": "cus_segment", "tenure_months": "cus_tenure",
        "credit_score": "cus_credit", "is_business": "cus_business", "preferred_payment": "cus_pay",
        "loyalty_tier": "cus_loyalty", "address_quality": "cus_addr_q", "complaint_index": "cus_complaint",
        "is_verified": "cus_verified", "kota_tipe": "cus_kota_tipe", "metode_notifikasi": "cus_notif",
        "tagihan_langganan": "cus_tagihan"})
    sh = sh.join(cu, on="customer_id", how="left")
    del cu
    gc.collect()

    co = pl.read_csv(path_of(split, "couriers.csv")).rename({
        "hub_id": "co_hub", "vehicle_type": "co_vehicle", "experience_years": "co_exp",
        "contract_type": "co_contract", "courier_score": "co_score", "shift_code": "co_shift",
        "training_level": "co_train", "jarak_domisili_hub_km": "co_home_km",
        "kapasitas_muatan_kg": "co_capacity", "sertifikasi_hazardous": "co_hazmat"})
    sh = sh.join(co, on="courier_id", how="left")
    del co
    gc.collect()

    hu = pl.read_csv(path_of(split, "hubs.csv")).rename({
        "region": "hub_region", "hub_type": "hub_type", "automation_level": "hub_auto",
        "capacity_index": "hub_cap", "density_index": "hub_density", "luas_gudang_m2": "hub_area",
        "jam_buka": "hub_open", "jam_tutup": "hub_close"})
    sh = sh.join(hu, on="hub_id", how="left")
    sh = sh.with_columns((pl.col("hub_close") - pl.col("hub_open")).alias("hub_hours"))
    del hu
    gc.collect()

    pr = pl.read_csv(path_of(split, "promotions.csv")).rename({
        "nama_promo": "prm_name", "tipe_promo": "prm_type", "diskon_persen": "prm_disc",
        "bulan_mulai": "prm_start", "bulan_akhir": "prm_end"})
    sh = sh.join(pr, on="promo_id", how="left")
    sh = sh.with_columns((pl.col("ship_month") - pl.col("prm_start")).alias("prm_month_offset"))
    del pr
    gc.collect()

    pk = pl.read_csv(path_of(split, "service_packages.csv")).rename({
        "speed_class": "pk_speed", "handling_class": "pk_handling", "insurance_default": "pk_ins_def",
        "cod_allowed": "pk_cod", "cost_multiplier": "pk_cost_mult", "max_weight_kg": "pk_max_w",
        "fragile_score": "pk_fragile", "estimasi_hari": "pk_est_days", "biaya_dasar": "pk_base_cost"})
    sh = sh.join(pk, on="package_id", how="left")
    sh = sh.with_columns((pl.col("weight_kg") / (pl.col("pk_max_w") + 1e-6)).alias("pk_weight_fill"))
    del pk
    gc.collect()

    sh = sh.join(scan_feats(split), on="shipment_id", how="left")
    sh = sh.join(route_feats(split), on="shipment_id", how="left")
    sh = sh.join(inv_feats(split), on="shipment_id", how="left")

    sh = sh.with_columns([
        (pl.col("rs_km_sum") - pl.col("distance_km")).alias("km_gap_route"),
        (pl.col("rs_km_sum") / (pl.col("distance_km") + 1e-6)).alias("km_ratio_route"),
        (pl.col("inv_amt_sum") / (pl.col("declared_value") + 1e-6)).alias("amt_per_value"),
        (pl.col("sc_n") / (pl.col("rs_n") + 1e-6)).alias("scans_per_seg"),
        (pl.col("ev_upaya_antar_n").cast(pl.Float32) / (pl.col("ev_selesai_n") + 1)).alias("upaya_per_selesai"),
        (pl.col("ev_selesai_n") == 0).cast(pl.Int8).alias("no_selesai"),
        (pl.col("last_hub") == pl.col("hub_id")).cast(pl.Int8).alias("last_hub_is_origin"),
        (pl.col("first_hub") == pl.col("hub_id")).cast(pl.Int8).alias("first_hub_is_origin"),
        (pl.col("co_hub") == pl.col("hub_id")).cast(pl.Int8).alias("courier_at_origin"),
        (pl.col("hub_region") == pl.col("dest_region")).cast(pl.Int8).alias("dest_same_region"),
        (pl.col("cus_region") == pl.col("dest_region")).cast(pl.Int8).alias("cus_same_region"),
        (pl.col("kota_asal") == pl.col("kota_tujuan")).cast(pl.Int8).alias("same_city"),
    ])
    sh = sh.with_columns((pl.col("rs_dur_sum") - pl.col("pk_est_days") * 24).alias("dur_vs_estimate"))

    out = PROCESSED_DIR / f"feat_{split}.parquet"
    sh.write_parquet(out)
    print(f"{split:5s} -> {sh.shape}  disimpan ke {out.relative_to(PROCESSED_DIR.parents[0])}")
    return sh


def load_features(split: str) -> pd.DataFrame:
    """Muat fitur hasil engineering dari ``data/processed`` (bangun bila belum ada)."""
    path = PROCESSED_DIR / f"feat_{split}.parquet"
    if not path.exists():
        build_features(split)
    return pd.read_parquet(path)


# ---------------------------------------------------------------------- encoding
def encode_categoricals(tr: pd.DataFrame, te: pd.DataFrame):
    """Ubah kolom kategorikal menjadi kode integer 1..n memakai kamus gabungan train+test.

    Penggabungan hanya memakai nilai fitur (tanpa target) sehingga tidak ada leakage.
    """
    cat_cols = [c for c in tr.columns if not pd.api.types.is_numeric_dtype(tr[c])]
    cat_categories = {}
    for c in cat_cols:
        all_vals = pd.concat([tr[c], te[c]], ignore_index=True)
        codes, uniques = pd.factorize(all_vals, sort=True)
        cat_categories[c] = uniques.to_numpy()
        tr[c] = (codes[: len(tr)] + 1).astype(np.int32)
        te[c] = (codes[len(tr):] + 1).astype(np.int32)

    num_cols = [c for c in tr.columns if c not in cat_cols]
    tr[num_cols] = tr[num_cols].astype(np.float32)
    te[num_cols] = te[num_cols].astype(np.float32)
    print(f"{len(cat_cols)} kolom kategorikal, {len(num_cols)} kolom numerik")
    return tr, te, cat_cols, cat_categories


def add_te_columns(df_tr, df_va, df_te, ids_tr, ids_va, ids_te, y_tr, cols=None):
    """Tambahkan kolom target encoding out-of-fold ``te_<id>_<k>``.

    Nilai untuk train dihitung out-of-fold (5-fold) agar tidak leakage; validasi dan
    test memakai tabel yang dihitung dari seluruh data train.
    """
    cols = TE_COLS if cols is None else cols
    te_maps = {}
    prior = np.bincount(y_tr, minlength=N_CLASS) / len(y_tr)

    for c in cols:
        c_tr, c_va, c_te = ids_tr[c], ids_va[c], ids_te[c]
        codes, uniq = pd.factorize(pd.Series(np.concatenate([c_tr, c_va, c_te])), sort=True)
        a, b = len(c_tr), len(c_va)
        c_tr, c_va, c_te = codes[:a], codes[a:a + b], codes[a + b:]

        oof = np.zeros((a, N_CLASS), dtype=np.float32)
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        for idx_tr, idx_va in skf.split(c_tr, y_tr):
            for k in range(N_CLASS):
                g = (pd.DataFrame({"c": c_tr[idx_tr], "t": (y_tr[idx_tr] == k).astype(np.float32)})
                     .groupby("c")["t"].agg(["sum", "count"]))
                m = (g["sum"] + prior[k] * SMOOTH) / (g["count"] + SMOOTH)
                oof[idx_va, k] = pd.Series(c_tr[idx_va]).map(m).fillna(prior[k]).to_numpy()

        maps = {}
        for k in range(N_CLASS):
            g = (pd.DataFrame({"c": c_tr, "t": (y_tr == k).astype(np.float32)})
                 .groupby("c")["t"].agg(["sum", "count"]))
            maps[k] = (g["sum"] + prior[k] * SMOOTH) / (g["count"] + SMOOTH)

        for k in range(N_CLASS):
            name = f"te_{c}_{k}"
            df_tr[name] = oof[:, k]
            df_va[name] = pd.Series(c_va).map(maps[k]).fillna(prior[k]).to_numpy().astype(np.float32)
            df_te[name] = pd.Series(c_te).map(maps[k]).fillna(prior[k]).to_numpy().astype(np.float32)

        # tabel TE berkunci nilai ID asli (untuk inferensi tanpa retraining); indeks
        # groupby berupa kode, jadi diterjemahkan lewat uniq agar tidak salah pasang
        te_maps[c] = {k: {uniq[i]: v for i, v in zip(m.index.to_numpy(), m.to_numpy())}
                      for k, m in maps.items()}
        print(f"  target encoding selesai: {c} ({len(uniq):,} nilai unik)")
    return te_maps
