"""Task-2 v7 Part 2: Satellite-Native SatlasPretrain + DINOv2 Mean-Centered Fusion Benchmark
& High-Res Micro-Clustering Audit.

Benchmarks SatlasPretrain Aerial_SwinB_SI standalone & fused with v3+v4+v5 caches,
builds the upgraded X_v7_master graph, runs Louvain micro-clustering, prints full
st_pre / st_post / st_del + prototype diagnostics (anti-plow-artifact & anti-sugarcoating),
and renders RANDOM-sample De-Tinted Aligned montages for 6-Class Pure-Core SSL selection.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "ghani" / "task2" / "src" / "v4"))
sys.path.insert(0, str(ROOT / "ghani" / "task2" / "src" / "v5"))
sys.path.insert(0, str(ROOT / "ghani" / "task2" / "src" / "v6"))
from align_extract_v4 import l2norm, warp_post_to_pre  # noqa: E402
from extract_v5 import detint_clean_rgb  # noqa: E402
from cluster_audit_v6 import center_l2  # noqa: E402

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

CLASSES = ["Earthquake", "Fire", "Hurricane", "Tsunami", "Volcano", "Explosion"]
FEN_DIR = ROOT / "Task-2" / "Fenomena"
C3_DIR = ROOT / "ghani" / "task2" / "cache" / "v3"
C4_DIR = ROOT / "ghani" / "task2" / "cache" / "v4"
C5_DIR = ROOT / "ghani" / "task2" / "cache" / "v5"
C6_DIR = ROOT / "ghani" / "task2" / "cache" / "v6"
C7_DIR = ROOT / "ghani" / "task2" / "cache" / "v7"


def build_v7_master_matrix(center_state: dict | None = None, sat_weight: float = 1.35):
    """Build Satellite-Native (SatlasPretrain Aerial_SwinB_SI) + DINOv2 + ConvNeXt + Physics matrix."""
    vb3 = np.load(C3_DIR / "vitb_wb.npz")
    vs3 = np.load(C3_DIR / "vits_gray_rot2.npz")
    cx3 = np.load(C3_DIR / "cnxb_gray.npz")
    p3 = np.load(C3_DIR / "phys_v3.npz", allow_pickle=True)

    vs4 = np.load(C4_DIR / "aligned_vits_336.npz")
    vb4 = np.load(C4_DIR / "aligned_vitb_224.npz")
    ap4 = np.load(C4_DIR / "aligned_phys_v4.npz", allow_pickle=True)

    vb5 = np.load(C5_DIR / "detint_vitb_224.npz")
    cm5 = np.load(C5_DIR / "changemap_vits_224.npz")
    p5 = np.load(C5_DIR / "detint_phys_v5.npz", allow_pickle=True)

    sat7 = np.load(C7_DIR / "satlas_aerial_swinb.npz")

    raw_views = [
        # Satellite-Native SatlasPretrain Aerial_SwinB_SI (0.5-2 m/px aerial domain!)
        ("sat_f3_pre", 1.05 * sat_weight, sat7["f3_pre"]),
        ("sat_f3_post", 1.30 * sat_weight, sat7["f3_post"]),
        ("sat_f3_diff", 0.70 * sat_weight, sat7["f3_post"] - sat7["f3_pre"]),
        ("sat_f2_pre", 0.90 * sat_weight, sat7["f2_pre"]),
        ("sat_f2_post", 1.15 * sat_weight, sat7["f2_post"]),
        ("sat_f2_diff", 0.55 * sat_weight, sat7["f2_post"] - sat7["f2_pre"]),
        # DINOv2 + ConvNeXt clean multi-view (from v3+v4+v5)
        ("vb5_pre", 1.00, vb5["pmean_pre"]),
        ("vb5_post", 1.35, vb5["pmean_post"]),
        ("vb5_diff", 0.75, vb5["pmean_post"] - vb5["pmean_pre"]),
        ("vs4_pre", 1.00, vs4["pmean_pre"]),
        ("vs4_post", 1.35, vs4["pmean_post"]),
        ("vb4_pre", 0.90, vb4["pmean_pre"]),
        ("vb4_post", 1.20, vb4["pmean_post"]),
        ("vb3_cpre", 0.85, vb3["cls_pre"]),
        ("vb3_cpost", 1.05, vb3["cls_post"]),
        ("vb3_ppre", 0.85, vb3["pmean_pre"]),
        ("vb3_ppost", 1.05, vb3["pmean_post"]),
        ("vs3_ppre", 0.75, vs3["pmean_pre"]),
        ("vs3_ppost", 0.95, vs3["pmean_post"]),
        ("cx3_pre", 0.85, cx3["pre"]),
        ("cx3_post", 0.95, cx3["post"]),
        ("cm5", 0.40, cm5["emb"]),
    ]

    pf5 = np.nan_to_num(p5["features"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    apf = np.nan_to_num(ap4["features"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    new_state = {} if center_state is None else center_state
    blocks = []
    for key, wt, arr in raw_views:
        mv = None if center_state is None else center_state[key]
        c_arr, mv_out = center_l2(arr, mv)
        if center_state is None:
            new_state[key] = mv_out
        blocks.append(wt * c_arr)

    if center_state is None:
        sc5 = StandardScaler().fit(pf5)
        sc4 = StandardScaler().fit(apf)
        new_state["sc5"] = sc5
        new_state["sc4"] = sc4
    else:
        sc5 = center_state["sc5"]
        sc4 = center_state["sc4"]

    pf5_n = l2norm(np.clip(sc5.transform(pf5), -4.0, 4.0).astype(np.float32))
    ap4_n = l2norm(np.clip(sc4.transform(apf), -4.0, 4.0).astype(np.float32))
    blocks.append(0.60 * pf5_n)
    blocks.append(0.50 * ap4_n)

    X_master = l2norm(np.hstack(blocks).astype(np.float32))
    return X_master, new_state, p5, ap4, p3, sat7


def run_benchmark_satlas(sample_ids: list[str], sat7):
    """Honest LOO kNN benchmark comparing SatlasPretrain alone vs v6 vs v7 fused on 60 visual anchors."""
    id2i = {s: i for i, s in enumerate(sample_ids)}
    anchors = {
        "Tsunami":    ["6062a3a7", "7e8c19a", "f079ccd", "1fce27c", "380b5d8", "9d8231e", "fb0eba1", "4857cc6", "f3a9842", "b70b27b", "43e578d"],
        "Volcano":    ["eb0a14a", "3d3b0398", "9de751e", "46611e7", "d9e17cb", "5a5f496", "47cfaf0", "67052b1", "d69a2ad", "4288963"],
        "Fire":       ["f4965d34", "abb57b0", "11da794", "41e98f8", "31795bc", "1dff71b", "e1b956d", "bfcc30b", "2f1d841", "75aa976"],
        "Hurricane":  ["541c2e1", "f6cefe8", "2c438ca", "36d1091", "12b9452", "674da12", "46d3cea", "0bc54e8", "1e8c587", "76c97c0"],
        "Earthquake": ["dec05312", "038f53e4", "961fbe1", "d4967d5", "08c7a86c", "74419ba", "bd88fe7", "73d2b1f", "6e76e5d", "80a01d9"],
        "Explosion":  ["d3a1e358", "c963993", "99cb5f0", "73ed7a2", "6726106", "2b0fcc2", "55dc286", "e4a86ed", "b61f21d"],
    }
    a_idx, a_y = [], []
    for cls, prefs in anchors.items():
        for pf in prefs:
            m = [s for s in sample_ids if s.startswith(pf)]
            if m:
                a_idx.append(id2i[m[0]])
                a_y.append(CLASSES.index(cls))
    a_idx = np.array(a_idx)
    a_y = np.array(a_y)

    def _eval(name, X):
        Xn = l2norm(X.astype(np.float32))
        A = Xn[a_idx]
        S = A @ A.T
        np.fill_diagonal(S, -1e9)
        p1 = a_y[np.argmax(S, axis=1)]
        p3, p5_p = [], []
        for i in range(len(a_idx)):
            for k, out in [(3, p3), (5, p5_p)]:
                tk = np.argsort(-S[i])[:k]
                sc = np.zeros(6)
                for j in tk:
                    sc[a_y[j]] += max(0.0, S[i, j])
                out.append(np.argmax(sc))
        pc = f1_score(a_y, p3, average=None)
        ps = " ".join(f"{CLASSES[k][:4]}:{pc[k]:.2f}" for k in range(6))
        print(f"  {name:30s} | 1NN={f1_score(a_y,p1,average='macro'):.3f} 3NN={f1_score(a_y,p3,average='macro'):.3f} 5NN={f1_score(a_y,p5_p,average='macro'):.3f} | {ps}", flush=True)

    print("\n=== HONEST SATLASPRETRAIN SEPARABILITY BENCHMARK (60 Multi-Event Anchors) ===", flush=True)
    sat_f3 = np.hstack([center_l2(sat7["f3_pre"])[0], 1.2 * center_l2(sat7["f3_post"])[0]])
    sat_f2 = np.hstack([center_l2(sat7["f2_pre"])[0], 1.2 * center_l2(sat7["f2_post"])[0]])
    sat_both = np.hstack([sat_f3, 0.85 * sat_f2, 0.6 * center_l2(sat7["f3_post"] - sat7["f3_pre"])[0]])
    _eval("Satlas Aerial f3 (8x8) only", sat_f3)
    _eval("Satlas Aerial f2 (16x16) only", sat_f2)
    _eval("Satlas Aerial f3+f2+diff only", sat_both)
    for sw in [0.0, 0.8, 1.35, 2.0]:
        Xm, *_ = build_v7_master_matrix(None, sat_weight=sw)
        _eval(f"v7 Fused (sat_weight={sw:.2f})", Xm)


def run_louvain_v7(X: np.ndarray, k_nn: int = 18):
    N = len(X)
    nn = NearestNeighbors(n_neighbors=k_nn + 1, metric="cosine", n_jobs=-1).fit(X)
    dists, nbrs = nn.kneighbors(X)
    sims = np.clip(1.0 - dists[:, 1:], 0.0, 1.0)
    nbrs = nbrs[:, 1:]

    nbr_sets = [set(nbrs[i].tolist()) for i in range(N)]
    edge_map: dict[tuple[int, int], float] = {}
    for i in range(N):
        for r in range(k_nn):
            j = int(nbrs[i, r])
            if i == j:
                continue
            u, v = (i, j) if i < j else (j, i)
            w = float(sims[i, r]) ** 3
            if i in nbr_sets[j]:
                w *= 1.40
            if (u, v) not in edge_map or w > edge_map[(u, v)]:
                edge_map[(u, v)] = w

    g = ig.Graph(n=N, edges=list(edge_map.keys()), directed=False)
    g.es["weight"] = list(edge_map.values())

    best = None
    for res in [4.5, 5.0, 5.5, 6.0, 6.5]:
        random.seed(SEED)
        part = g.community_multilevel(weights="weight", resolution=res)
        raw = np.array(part.membership, dtype=np.int32)
        vals, cnts = np.unique(raw, return_counts=True)
        n_maj = int((cnts >= 14).sum())
        tiny_n = int(cnts[cnts < 14].sum())
        print(f"  Louvain res={res:.1f}: {n_maj} major (>=14), tiny={tiny_n}, mod={part.modularity:.4f}", flush=True)
        score = abs(n_maj - 54) + (0.0 if 44 <= n_maj <= 66 else 30.0)
        if best is None or score < best[0]:
            best = (score, res, raw, float(part.modularity))

    _, res, raw, mod = best
    vals, cnts = np.unique(raw, return_counts=True)
    maj_order = vals[cnts >= 14]
    maj_order = maj_order[np.argsort(-cnts[cnts >= 14])]
    n_maj = len(maj_order)

    cents = np.vstack([l2norm(X[raw == c].mean(axis=0, keepdims=True))[0] for c in maj_order]).astype(np.float32)
    lab = np.full(N, -1, dtype=np.int32)
    for new_c, old_c in enumerate(maj_order):
        lab[raw == old_c] = new_c
    tiny_mask = lab < 0
    if tiny_mask.any():
        lab[tiny_mask] = np.argmax(X[tiny_mask] @ cents.T, axis=1)
    cents = np.vstack([l2norm(X[lab == c].mean(axis=0, keepdims=True))[0] for c in range(n_maj)]).astype(np.float32)
    print(f"Selected Louvain res={res:.1f}: {n_maj} micro-clusters (tiny={int(tiny_mask.sum())}), mod={mod:.4f}", flush=True)
    return lab, cents, res, mod


def main():
    t0 = time.time()
    sample_ids = json.load(open(C7_DIR / "sample_ids.json", "r", encoding="utf-8"))
    aln = np.load(C4_DIR / "alignment_v4.npz")
    bang, bdy, bdx, zp = aln["best_ang"], aln["best_dy"], aln["best_dx"], aln["z_peak"]
    cl3 = np.load(C3_DIR / "clusters_v3_clean.npz")["lab_clean"]
    beirut_v3 = (cl3 == 22)
    s4 = pd.read_csv(ROOT / "ghani" / "submissions" / "task2" / "submission_v4_ghani.csv").set_index("Id").loc[sample_ids]["Phenomenon"].values
    s6 = pd.read_csv(ROOT / "ghani" / "submissions" / "task2" / "submission_v6_ghani.csv").set_index("Id").loc[sample_ids]["Phenomenon"].values

    X_master, _, p5, ap4, p3, sat7 = build_v7_master_matrix(None, sat_weight=1.35)
    run_benchmark_satlas(sample_ids, sat7)

    print(f"\nBuilding v7 Louvain graph on X_master shape={X_master.shape}...", flush=True)
    lab, cents, res, mod = run_louvain_v7(X_master, k_nn=18)
    n_c = len(cents)

    np.savez_compressed(
        C7_DIR / "clusters_v7_micro.npz",
        ids=np.array(sample_ids),
        lab_micro=lab,
        cents=cents,
        res=np.float32(res),
        mod=np.float32(mod),
    )

    pf5 = p5["features"]; p5m = {k: i for i, k in enumerate(list(p5["feature_names"]))}
    apf = ap4["features"]; apm = {k: i for i, k in enumerate(list(ap4["feature_names"]))}
    audit3 = p3["audit_flags"]; p3m = {k: i for i, k in enumerate(list(p3["feature_names"]))}
    both_gray = (audit3[:, 2] > 0.5) & (audit3[:, 3] > 0.5)
    cld_post = p3["features"][:, p3m["cloud_frac_post"]]

    c_beir = l2norm(X_master[beirut_v3].mean(axis=0, keepdims=True))[0]
    b_sim = X_master @ c_beir

    rows = []
    for c in range(n_c):
        m = (lab == c)
        vc4 = pd.Series(s4[m]).value_counts()
        vc6 = pd.Series(s6[m]).value_counts()
        rows.append({
            "mc": c,
            "n": int(m.sum()),
            "pct_gray": round(float(both_gray[m].mean() * 100.0), 1),
            "cld_post": round(float(cld_post[m].mean() * 100.0), 2),
            "st_pre": round(float(pf5[m, p5m["stripe_z_pre"]].mean()), 2),
            "st_pst": round(float(pf5[m, p5m["stripe_z_post"]].mean()), 2),
            "st_del": round(float(pf5[m, p5m["stripe_z_delta"]].mean()), 2),
            "charc": round(float(pf5[m, p5m["charcoal_gain"]].mean()), 3),
            "hol_ash": round(float(pf5[m, p5m["hollow_ash_gain"]].mean()), 4),
            "volc_ash": round(float(pf5[m, p5m["volc_ash_veil"]].mean()), 4),
            "h_asym": round(float(apf[m, apm["half_asym_max"]].mean()), 4),
            "wat_inund": round(float(apf[m, apm["water_inund_frac"]].mean()), 4),
            "lap_lr": round(float(apf[m, apm["lap_log_ratio"]].mean()), 3),
            "b_v3c22": int(beirut_v3[m].sum()),
            "b_sim": round(float(b_sim[m].mean()), 3),
            "v4_top": f"{vc4.index[0][:4]}({vc4.iloc[0]/m.sum()*100:.0f}%)",
            "v6_top": f"{vc6.index[0][:4]}({vc6.iloc[0]/m.sum()*100:.0f}%)",
        })
    df_prof = pd.DataFrame(rows)
    df_prof.to_csv(C7_DIR / "micro_profiles_v7.csv", index=False)

    print("\n=== v7 LEAKAGE & BEIRUT CHECK ===", flush=True)
    print(f"Max pct_gray: {df_prof['pct_gray'].max():.1f}% (>25% cnt={(df_prof['pct_gray']>25).sum()})", flush=True)
    print(f"Max cld_post: {df_prof['cld_post'].max():.2f}% (>8% cnt={(df_prof['cld_post']>8).sum()})", flush=True)
    print("Clusters w/ v3_C22 Beirut:", df_prof[df_prof["b_v3c22"] > 0][["mc", "n", "b_v3c22", "b_sim", "v6_top"]].to_dict("records"), flush=True)
    print("Top 6 TRUE post-only Sunda scanline clusters (by st_del):", df_prof.nlargest(6, "st_del")[["mc", "n", "st_pre", "st_pst", "st_del", "v6_top"]].to_dict("records"), flush=True)

    rng = np.random.RandomState(SEED)
    T = 104
    per_page = 10
    n_pages = (n_c + per_page - 1) // per_page
    for pg in range(n_pages):
        cs = list(range(pg * per_page, min(n_c, (pg + 1) * per_page)))
        can = Image.new("RGB", (148 + 6 * (2 * T + 6), len(cs) * (T + 16)), "white")
        dr = ImageDraw.Draw(can)
        for r, c in enumerate(cs):
            idx_c = np.where(lab == c)[0]
            picks = rng.choice(idx_c, size=min(6, len(idx_c)), replace=False)
            rd = rows[c]
            lbl = (
                f"S{c:02d} n={rd['n']} B={rd['b_v3c22']}\n"
                f"6:{rd['v6_top']}\n"
                f"4:{rd['v4_top']}\n"
                f"sD={rd['st_del']:+.1f} ch={rd['charc']:+.2f}\n"
                f"lR={rd['lap_lr']:+.2f} ha={rd['h_asym']:.3f}"
            )
            dr.text((3, r * (T + 16) + 8), lbl, fill="black")
            for j, ii in enumerate(picks):
                sid = sample_ids[ii]
                with Image.open(FEN_DIR / sid / "pre.jpg") as i1, Image.open(FEN_DIR / sid / "post.jpg") as i2:
                    r1 = detint_clean_rgb(i1.convert("RGB").resize((T, T), Image.Resampling.BILINEAR), False)
                    r2 = detint_clean_rgb(warp_post_to_pre(i2.convert("RGB"), float(bang[ii]), int(bdy[ii]), int(bdx[ii]), T), False)
                x0 = 148 + j * (2 * T + 6)
                y0 = r * (T + 16) + 2
                can.paste(r1, (x0, y0))
                can.paste(r2, (x0 + T + 2, y0))
                bt = "*" if beirut_v3[ii] else ""
                dr.text((x0 + 2, y0 + T + 1), f"{bt}{sid[:7]} z={zp[ii]:.1f}", fill="black")
        out_pg = C7_DIR / f"montage_v7_micro_p{pg}.jpg"
        can.save(out_pg, quality=86)
        print(f"Saved {out_pg}", flush=True)

    print(f"DONE cluster_audit_v7 in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
