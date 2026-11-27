"""Task-2 v5 Part 2: Change-Boosted De-Tinted High-Res Micro-Clustering & Random-Sample Audit.

Builds a Change-Boosted De-Tinted + Gray-Aligned + Visual-ChangeMap + 2D-FFT Event-Mechanism graph,
sweeps Louvain resolution to obtain 55-75 crisp micro-communities, audits leakage & physics,
and renders RANDOM-sample (not centroid-only!) de-tinted aligned montages for honest visual mapping.
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "ghani" / "task2" / "src" / "v4"))
sys.path.insert(0, str(ROOT / "ghani" / "task2" / "src" / "v5"))
from align_extract_v4 import l2norm, warp_post_to_pre  # noqa: E402
from extract_v5 import detint_clean_rgb  # noqa: E402

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

FEN_DIR = ROOT / "Task-2" / "Fenomena"
C3_DIR = ROOT / "ghani" / "task2" / "cache" / "v3"
C4_DIR = ROOT / "ghani" / "task2" / "cache" / "v4"
C5_DIR = ROOT / "ghani" / "task2" / "cache" / "v5"


def build_v5_graph_matrix():
    """Assemble Change-Boosted De-Tinted + Gray-Aligned + ChangeMap + 2D-FFT matrix."""
    p5 = np.load(C5_DIR / "detint_phys_v5.npz", allow_pickle=True)
    vb5 = np.load(C5_DIR / "detint_vitb_224.npz")
    cm5 = np.load(C5_DIR / "changemap_vits_224.npz")
    vs4 = np.load(C4_DIR / "aligned_vits_336.npz")
    vb4 = np.load(C4_DIR / "aligned_vitb_224.npz")
    ap4 = np.load(C4_DIR / "aligned_phys_v4.npz", allow_pickle=True)
    p3 = np.load(C3_DIR / "phys_v3.npz", allow_pickle=True)

    # Clean scaled physics (drop raw gray/cloud leakage cols, clip outliers)
    pf5 = np.nan_to_num(p5["features"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pf5_sc = np.clip(StandardScaler().fit_transform(pf5), -4.0, 4.0).astype(np.float32)
    pf5_n = l2norm(pf5_sc)

    ap4_f = np.nan_to_num(ap4["features"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ap4_n = l2norm(np.clip(StandardScaler().fit_transform(ap4_f), -4.0, 4.0).astype(np.float32))

    # Change-boosted multi-view fusion:
    # Emphasize De-Tinted post/delta + Visual ChangeMap + Gray-CLAHE high-res delta + 2D-FFT physics
    blocks = [
        0.85 * l2norm(vb5["pmean_pre"]),
        1.15 * l2norm(vb5["pmean_post"]),
        1.25 * l2norm(vb5["aligned_delta_emb"]),
        1.00 * l2norm(vb5["topchg_post"]),
        1.10 * l2norm(vb5["topchg_delta"]),
        0.95 * l2norm(cm5["emb"]),
        0.75 * l2norm(vs4["pmean_pre"]),
        0.85 * l2norm(vs4["pmean_post"]),
        0.95 * l2norm(vs4["aligned_delta_emb"]),
        0.85 * l2norm(vs4["topchg_delta"]),
        0.65 * l2norm(vb4["aligned_delta_emb"]),
        0.65 * l2norm(vb4["topchg_delta"]),
        0.75 * pf5_n,
        0.45 * ap4_n,
    ]
    X_graph = l2norm(np.hstack(blocks).astype(np.float32))
    return X_graph, p5, ap4, p3, vb5


def run_louvain_micro(X: np.ndarray, k_nn: int = 18):
    """Build power-weighted cosine kNN graph (sim^3) and sweep Louvain to 55-75 major communities."""
    N = len(X)
    nn = NearestNeighbors(n_neighbors=k_nn + 1, metric="cosine", n_jobs=-1).fit(X)
    dists, nbrs = nn.kneighbors(X)
    sims = np.clip(1.0 - dists[:, 1:], 0.0, 1.0)
    nbrs = nbrs[:, 1:]

    # Mutual boost + cubic edge sharpening to break weak cross-biome bridges
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
                w *= 1.35
            if (u, v) not in edge_map or w > edge_map[(u, v)]:
                edge_map[(u, v)] = w

    edges = list(edge_map.keys())
    weights = [edge_map[e] for e in edges]
    g = ig.Graph(n=N, edges=edges, directed=False)
    g.es["weight"] = weights

    best = None
    for res in [4.5, 5.0, 5.5, 6.0, 6.5, 7.0]:
        random.seed(SEED)
        part = g.community_multilevel(weights="weight", resolution=res)
        raw = np.array(part.membership, dtype=np.int32)
        vals, cnts = np.unique(raw, return_counts=True)
        maj = vals[cnts >= 12]
        n_maj = len(maj)
        tiny_n = int(cnts[cnts < 12].sum())
        print(f"  Louvain res={res:.1f}: {n_maj} major (>=12), tiny={tiny_n}, mod={part.modularity:.4f}", flush=True)
        score = abs(n_maj - 64) + (0.0 if 52 <= n_maj <= 78 else 30.0)
        if best is None or score < best[0]:
            best = (score, res, raw, float(part.modularity))

    _, res, raw, mod = best
    vals, cnts = np.unique(raw, return_counts=True)
    maj_order = vals[cnts >= 12]
    maj_order = maj_order[np.argsort(-cnts[cnts >= 12])]
    n_maj = len(maj_order)

    cents = np.vstack([l2norm(X[raw == c].mean(axis=0, keepdims=True))[0] for c in maj_order]).astype(np.float32)
    lab = np.full(N, -1, dtype=np.int32)
    for new_c, old_c in enumerate(maj_order):
        lab[raw == old_c] = new_c
    tiny_mask = lab < 0
    if tiny_mask.any():
        lab[tiny_mask] = np.argmax(X[tiny_mask] @ cents.T, axis=1)
    # Recompute clean centroids
    cents = np.vstack([l2norm(X[lab == c].mean(axis=0, keepdims=True))[0] for c in range(n_maj)]).astype(np.float32)
    print(f"Selected Louvain res={res:.1f}: {n_maj} micro-clusters (assigned {int(tiny_mask.sum())} tiny samples), mod={mod:.4f}", flush=True)
    return lab, cents, res, mod


def main():
    t0 = time.time()
    sample_ids = json.load(open(C5_DIR / "sample_ids.json", "r", encoding="utf-8"))
    aln = np.load(C4_DIR / "alignment_v4.npz")
    bang, bdy, bdx, zp = aln["best_ang"], aln["best_dy"], aln["best_dx"], aln["z_peak"]

    cl3 = np.load(C3_DIR / "clusters_v3_clean.npz")["lab_clean"]
    beirut_v3 = (cl3 == 22)
    s4 = pd.read_csv(ROOT / "ghani" / "submissions" / "task2" / "submission_v4_ghani.csv").set_index("Id").loc[sample_ids]["Phenomenon"].values

    X_graph, p5, ap4, p3, vb5 = build_v5_graph_matrix()
    lab, cents, res, mod = run_louvain_micro(X_graph, k_nn=18)
    n_c = len(cents)

    np.savez_compressed(
        C5_DIR / "clusters_v5_micro.npz",
        ids=np.array(sample_ids),
        lab_micro=lab,
        cents=cents,
        res=np.float32(res),
        mod=np.float32(mod),
    )

    # Profile every micro-cluster
    pf5 = p5["features"]
    p5_map = {k: i for i, k in enumerate(list(p5["feature_names"]))}
    apf = ap4["features"]
    ap_map = {k: i for i, k in enumerate(list(ap4["feature_names"]))}
    audit3 = p3["audit_flags"]
    p3_map = {k: i for i, k in enumerate(list(p3["feature_names"]))}
    both_gray = (audit3[:, 2] > 0.5) & (audit3[:, 3] > 0.5)
    cld_post = p3["features"][:, p3_map["cloud_frac_post"]]

    # Beirut similarity across all samples in X_graph
    c_beirut = l2norm(X_graph[beirut_v3].mean(axis=0, keepdims=True))[0]
    beirut_sim = X_graph @ c_beirut

    rows = []
    for c in range(n_c):
        m = (lab == c)
        vc4 = pd.Series(s4[m]).value_counts()
        top4_cls = str(vc4.index[0])
        top4_pct = round(float(vc4.iloc[0] / m.sum() * 100.0), 1)
        rows.append({
            "mc": c,
            "n": int(m.sum()),
            "pct_gray": round(float(both_gray[m].mean() * 100.0), 1),
            "cld_post": round(float(cld_post[m].mean() * 100.0), 2),
            "stripe_z": round(float(pf5[m, p5_map["stripe_z_post"]].mean()), 2),
            "charcoal": round(float(pf5[m, p5_map["charcoal_gain"]].mean()), 3),
            "hol_ash": round(float(pf5[m, p5_map["hollow_ash_gain"]].mean()), 4),
            "volc_ash": round(float(pf5[m, p5_map["volc_ash_veil"]].mean()), 4),
            "ngray_g": round(float(pf5[m, p5_map["neutral_gray_gain"]].mean()), 3),
            "exg_drop": round(float(pf5[m, p5_map["exg_veg_drop"]].mean()), 3),
            "wat_inund": round(float(apf[m, ap_map["water_inund_frac"]].mean()), 4),
            "h_asym": round(float(apf[m, ap_map["half_asym_max"]].mean()), 4),
            "lap_lr": round(float(apf[m, ap_map["lap_log_ratio"]].mean()), 3),
            "can_d": round(float(apf[m, ap_map["canny_delta"]].mean()), 4),
            "land_p95": round(float(pf5[m, p5_map["land_p95_p50"]].mean()), 2),
            "b_v3c22": int(beirut_v3[m].sum()),
            "b_sim": round(float(beirut_sim[m].mean()), 3),
            "v4_top": top4_cls,
            "v4_pct": top4_pct,
        })
    df_prof = pd.DataFrame(rows)
    df_prof.to_csv(C5_DIR / "micro_profiles_v5.csv", index=False)
    print("\n=== LEAKAGE & SUMMARY CHECK ===", flush=True)
    print(f"Max pct_gray among major micro-clusters: {df_prof['pct_gray'].max():.1f}% (>25% count: {(df_prof['pct_gray']>25).sum()})", flush=True)
    print(f"Max cld_post among major micro-clusters: {df_prof['cld_post'].max():.2f}% (>8% count: {(df_prof['cld_post']>8).sum()})", flush=True)
    print(f"Clusters containing v3_C22 Beirut: {df_prof[df_prof['b_v3c22']>0][['mc','n','b_v3c22','b_sim','v4_top']].to_dict('records')}", flush=True)
    print(f"Top 8 stripe_z clusters: {df_prof.nlargest(8, 'stripe_z')[['mc','n','stripe_z','v4_top']].to_dict('records')}", flush=True)
    print(f"Top 8 charcoal/hollow_ash clusters: {df_prof.nlargest(8, 'charcoal')[['mc','n','charcoal','hol_ash','v4_top']].to_dict('records')}", flush=True)

    # Render De-Tinted Aligned RANDOM-Sample Audit Montages (10 micro-clusters per page, 6 RANDOM samples each!)
    rng = np.random.RandomState(SEED)
    T = 104
    per_page = 10
    n_pages = (n_c + per_page - 1) // per_page
    for pg in range(n_pages):
        cs = list(range(pg * per_page, min(n_c, (pg + 1) * per_page)))
        can = Image.new("RGB", (135 + 6 * (2 * T + 6), len(cs) * (T + 16)), "white")
        dr = ImageDraw.Draw(can)
        for r, c in enumerate(cs):
            idx_c = np.where(lab == c)[0]
            picks = rng.choice(idx_c, size=min(6, len(idx_c)), replace=False)
            row_d = rows[c]
            lbl = (
                f"M{c:02d} n={row_d['n']}\n"
                f"v4:{row_d['v4_top'][:4]}({row_d['v4_pct']:.0f}%)\n"
                f"st={row_d['stripe_z']:.1f} ch={row_d['charcoal']:+.2f}\n"
                f"lR={row_d['lap_lr']:+.2f} B={row_d['b_v3c22']}"
            )
            dr.text((3, r * (T + 16) + 12), lbl, fill="black")
            for j, ii in enumerate(picks):
                sid = sample_ids[ii]
                with Image.open(FEN_DIR / sid / "pre.jpg") as i1, Image.open(FEN_DIR / sid / "post.jpg") as i2:
                    r1 = detint_clean_rgb(i1.convert("RGB").resize((T, T), Image.Resampling.BILINEAR), fill_neutral_border=False)
                    r2 = detint_clean_rgb(
                        warp_post_to_pre(i2.convert("RGB"), float(bang[ii]), int(bdy[ii]), int(bdx[ii]), T),
                        fill_neutral_border=False,
                    )
                x0 = 135 + j * (2 * T + 6)
                y0 = r * (T + 16) + 2
                can.paste(r1, (x0, y0))
                can.paste(r2, (x0 + T + 2, y0))
                bt = "*" if beirut_v3[ii] else ""
                dr.text((x0 + 2, y0 + T + 1), f"{bt}{sid[:7]} z={zp[ii]:.1f}", fill="black")
        out_pg = C5_DIR / f"montage_v5_micro_p{pg}.jpg"
        can.save(out_pg, quality=86)
        print(f"Saved {out_pg}", flush=True)

    # Render dedicated Explosion / Beirut scan montage (top 8 micro-clusters by b_sim + top 2 by land_p95)
    top_expl_mcs = list(df_prof.nlargest(8, "b_sim")["mc"].values)
    for mc_x in df_prof.nlargest(4, "land_p95")["mc"].values:
        if mc_x not in top_expl_mcs:
            top_expl_mcs.append(int(mc_x))
    top_expl_mcs = top_expl_mcs[:10]
    can_e = Image.new("RGB", (135 + 6 * (2 * T + 6), len(top_expl_mcs) * (T + 16)), "white")
    dr_e = ImageDraw.Draw(can_e)
    for r, c in enumerate(top_expl_mcs):
        idx_c = np.where(lab == c)[0]
        # Pick 6 random deterministic samples from this cluster
        picks = rng.choice(idx_c, size=min(6, len(idx_c)), replace=False)
        row_d = rows[c]
        dr_e.text(
            (3, r * (T + 16) + 12),
            f"M{c:02d} n={row_d['n']}\nB22={row_d['b_v3c22']} sim={row_d['b_sim']:.2f}\np95={row_d['land_p95']:.2f}\nv4:{row_d['v4_top'][:5]}",
            fill="black",
        )
        for j, ii in enumerate(picks):
            sid = sample_ids[ii]
            with Image.open(FEN_DIR / sid / "pre.jpg") as i1, Image.open(FEN_DIR / sid / "post.jpg") as i2:
                r1 = detint_clean_rgb(i1.convert("RGB").resize((T, T), Image.Resampling.BILINEAR), fill_neutral_border=False)
                r2 = detint_clean_rgb(
                    warp_post_to_pre(i2.convert("RGB"), float(bang[ii]), int(bdy[ii]), int(bdx[ii]), T),
                    fill_neutral_border=False,
                )
            x0 = 135 + j * (2 * T + 6)
            y0 = r * (T + 16) + 2
            can_e.paste(r1, (x0, y0))
            can_e.paste(r2, (x0 + T + 2, y0))
            bt = "B:" if beirut_v3[ii] else ""
            dr_e.text((x0 + 2, y0 + T + 1), f"{bt}{sid[:7]} bs={beirut_sim[ii]:.2f}", fill="black")
    out_ex = C5_DIR / "v5_beirut_explosion_scan.jpg"
    can_e.save(out_ex, quality=88)
    print(f"Saved {out_ex} in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
