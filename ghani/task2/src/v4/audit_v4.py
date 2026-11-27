"""Task-2 v4 Part 1 Diagnostic Audit: Alignment Quality, Disentangled Clustering & Radial Blast Check."""
from __future__ import annotations

import json
import random
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[4]
FEN_DIR = ROOT / "Task-2" / "Fenomena"
CACHE_V4 = ROOT / "ghani" / "task2" / "cache" / "v4"
CACHE_V3 = ROOT / "ghani" / "task2" / "cache" / "v3"


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def zscore_l2(x: np.ndarray) -> np.ndarray:
    sc = StandardScaler().fit_transform(x)
    sc = np.clip(sc, -8.0, 8.0).astype(np.float32)
    return l2norm(sc)


def warp_post_to_pre(im_post_rgb: Image.Image, ang_deg: float, dy_g16: int, dx_g16: int, out_size: int) -> Image.Image:
    im_r = im_post_rgb.resize((out_size, out_size), Image.Resampling.BILINEAR)
    im_rot = im_r.rotate(float(ang_deg), resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    dx_px = int(round(dx_g16 * out_size / 16.0))
    dy_px = int(round(dy_g16 * out_size / 16.0))
    if dx_px == 0 and dy_px == 0:
        return im_rot
    canvas = Image.new("RGB", (out_size, out_size), (0, 0, 0))
    canvas.paste(im_rot, (dx_px, dy_px))
    return canvas


def check_gray_bias(tag: str, emb: np.ndarray, both_gray: np.ndarray) -> float:
    g_idx = np.where(both_gray)[0]
    c_idx = np.where(~both_gray)[0]
    gg = float((emb[g_idx] @ emb[g_idx].T)[~np.eye(len(g_idx), dtype=bool)].mean())
    gc_val = float((emb[g_idx] @ emb[c_idx].T).mean())
    bias = gg - gc_val
    print(f"  {tag:28s} | gray_bias={bias:+.4f}", flush=True)
    return bias


def sweep_louvain(name: str, X: np.ndarray, both_gray: np.ndarray, cld_post: np.ndarray):
    nn = NearestNeighbors(n_neighbors=16, metric="cosine").fit(X)
    dists, idxs = nn.kneighbors(X)
    edges, weights = [], []
    for i in range(len(X)):
        for j, d in zip(idxs[i, 1:], dists[i, 1:]):
            if i < j:
                edges.append((i, int(j)))
                weights.append(max(0.0, 1.0 - float(d)))
    g = ig.Graph(n=len(X), edges=edges, directed=False)

    print(f"\n=== LOUVAIN SWEEP: {name} ===", flush=True)
    out = {}
    for res in [0.8, 1.2, 1.8, 2.4]:
        random.seed(SEED)
        part = g.community_multilevel(weights=weights, resolution=res)
        lab = np.array(part.membership, dtype=int)
        maj = sorted([c for c in set(lab) if (lab == c).sum() >= 15], key=lambda c: -(lab == c).sum())
        max_g = max(float(both_gray[lab == c].mean()) for c in maj) if maj else 1.0
        max_c = max(float(cld_post[lab == c].mean()) for c in maj) if maj else 1.0
        sil = float(silhouette_score(X, lab, metric="cosine")) if len(set(lab)) > 1 else 0.0
        sizes = [int((lab == c).sum()) for c in maj]
        print(
            f"  res={res:.1f} | maj={len(maj):2d} | mod={part.modularity:.3f} | sil={sil:+.4f} | "
            f"max_gray={max_g*100:4.1f}% | max_cld={max_c*100:4.2f}% | top8={sizes[:8]}",
            flush=True,
        )
        out[res] = (lab, maj, float(part.modularity), sil)
    return out


def main():
    with open(CACHE_V4 / "sample_ids.json", "r", encoding="utf-8") as f:
        sample_ids = json.load(f)

    aln = np.load(CACHE_V4 / "alignment_v4.npz")
    aphys = np.load(CACHE_V4 / "aligned_phys_v4.npz")
    vits = np.load(CACHE_V4 / "aligned_vits_336.npz")
    vitb = np.load(CACHE_V4 / "aligned_vitb_224.npz")

    # Cloud & gray audit flags from v3 phys cache (same sample_ids order)
    phys_v3 = np.load(CACHE_V3 / "phys_v3.npz")
    pnames3 = list(phys_v3["feature_names"])
    pf3 = phys_v3["features"]
    cld_pre = pf3[:, pnames3.index("cloud_frac_pre")]
    cld_post = pf3[:, pnames3.index("cloud_frac_post")]
    audit3 = phys_v3["audit_flags"]
    both_gray = (audit3[:, 2] > 0.5) & (audit3[:, 3] > 0.5)

    for arr_name, mat in [
        ("aligned_phys", aphys["features"]),
        ("vits_336_pmean", vits["pmean_post"]),
        ("vitb_224_pmean", vitb["pmean_post"]),
    ]:
        assert np.isfinite(mat).all(), f"NaN/Inf detected in {arr_name}!"

    print("=== 1. ALIGNMENT QUALITY SUMMARY ===", flush=True)
    z = aln["z_peak"]
    print(f"  z_peak > 2.0 : {(z > 2.0).mean()*100:.1f}% ({int((z > 2.0).sum())}/3180)", flush=True)
    print(f"  z_peak > 2.5 : {(z > 2.5).mean()*100:.1f}% ({int((z > 2.5).sum())}/3180)", flush=True)
    print(f"  median best_sim={np.median(aln['best_sim']):.3f}, median ov_frac={np.median(aln['ov_frac']):.3f}", flush=True)

    print("\n=== 2. GRAYSCALE LEAKAGE CHECK ===", flush=True)
    check_gray_bias("vits_336 pmean_post", vits["pmean_post"], both_gray)
    check_gray_bias("vits_336 aligned_delta", vits["aligned_delta_emb"], both_gray)
    check_gray_bias("vitb_224 pmean_post", vitb["pmean_post"], both_gray)
    check_gray_bias("vitb_224 aligned_delta", vitb["aligned_delta_emb"], both_gray)

    wp = np.clip(1.0 - 6.0 * cld_pre, 0.05, 1.0)[:, None]
    wq = np.clip(1.0 - 6.0 * cld_post, 0.05, 1.0)[:, None]
    wc = np.minimum(wp, wq)
    w_aln = np.clip((aln["z_peak"] - 1.0) / 1.5, 0.25, 1.0)[:, None]
    w_dmg = wc * w_aln

    ap_names = list(aphys["feature_names"])
    ap_f = aphys["features"]
    ap_map = {n: i for i, n in enumerate(ap_names)}

    phys_block = zscore_l2(ap_f)
    loc_block = zscore_l2(np.hstack([vits["loc_stats"], vitb["loc_stats"]]))

    X_dmg = l2norm(np.hstack([
        l2norm(vits["aligned_delta_emb"]) * w_dmg * 1.15,
        l2norm(vits["topchg_delta"]) * w_dmg * 1.10,
        l2norm(vits["topchg_post"]) * wq * 0.70,
        l2norm(vitb["aligned_delta_emb"]) * w_dmg * 1.05,
        l2norm(vitb["topchg_delta"]) * w_dmg * 1.00,
        l2norm(vitb["topchg_post"]) * wq * 0.65,
        phys_block * 0.55,
        loc_block * 0.50,
    ]).astype(np.float32))

    X_scene = l2norm(np.hstack([
        l2norm(0.85 * wp * l2norm(vits["pmean_pre"]) + 1.15 * wq * l2norm(vits["pmean_post"])) * 1.15,
        l2norm(0.85 * wp * l2norm(vitb["pmean_pre"]) + 1.15 * wq * l2norm(vitb["pmean_post"])) * 1.05,
    ]).astype(np.float32))

    X_fused = l2norm(np.hstack([
        X_scene * 1.00,
        X_dmg * 0.90,
    ]).astype(np.float32))

    _ = sweep_louvain("X_v4_dmg (Pure Aligned Damage)", X_dmg, both_gray, cld_post)
    fused_res = sweep_louvain("X_v4_fused (55% Scene + 45% Aligned Damage)", X_fused, both_gray, cld_post)

    # Pick resolution producing ~18-26 communities
    best_res = min(fused_res.keys(), key=lambda r: abs(len(fused_res[r][1]) - 22))
    lab_f, maj_f, mod_f, sil_f = fused_res[best_res]

    # Top-24 Aligned Radial Blast Candidates
    clear = (cld_pre < 0.03) & (cld_post < 0.03) & (aln["z_peak"] > 2.0)
    rad_c = ap_f[:, ap_map["radial_decay_corr"]]
    pk_ssim = ap_f[:, ap_map["peak_minus_p50_ssim"]]
    sp_ssim = ap_f[:, ap_map["epicenter_spread"]]
    loc_vits = vits["loc_stats"]
    # loc_stats: 0=mean,2=p50,5=p95,9=spread,10=rad_corr,11=p95-p50
    blast_sc = (
        (-rad_c) * 1.5
        + (-loc_vits[:, 10]) * 1.2
        + pk_ssim * 1.2
        + loc_vits[:, 11] * 1.2
        - sp_ssim * 1.0
        - loc_vits[:, 9] * 0.8
    )
    blast_sc[~clear] = -99.0
    top24 = np.argsort(-blast_sc)[:24]
    print("\n=== 3. TOP-24 RADIAL BLAST CLUSTER DISTRIBUTION ===", flush=True)
    print("Fused clusters:", pd.Series(lab_f[top24]).value_counts().to_dict(), flush=True)

    T = 120
    can_b = Image.new("RGB", (95 + 6 * (2 * T + 6), 4 * (T + 16)), "white")
    dr_b = ImageDraw.Draw(can_b)
    for r in range(4):
        dr_b.text((4, r * (T + 16) + 35), f"Blast\n#{r*6+1}-{(r+1)*6}", fill="black")
        for j in range(6):
            ii = int(top24[r * 6 + j])
            sid = sample_ids[ii]
            with Image.open(FEN_DIR / sid / "pre.jpg") as im1, Image.open(FEN_DIR / sid / "post.jpg") as im2:
                r1 = im1.convert("RGB").resize((T, T), Image.Resampling.BILINEAR)
                r2 = warp_post_to_pre(im2.convert("RGB"), float(aln["best_ang"][ii]), int(aln["best_dy"][ii]), int(aln["best_dx"][ii]), T)
            x0 = 95 + j * (2 * T + 6)
            y0 = r * (T + 16) + 2
            can_b.paste(r1, (x0, y0))
            can_b.paste(r2, (x0 + T + 2, y0))
            dr_b.text((x0 + 2, y0 + T + 1), f"C{lab_f[ii]}:{sid[:7]} rc={rad_c[ii]:+.2f}", fill="black")
    can_b.save(CACHE_V4 / "v4_top24_aligned_blast.jpg", quality=90)

    # Render montages for fused communities (9 per page)
    clear_c = (cld_pre < 0.04) & (cld_post < 0.04)
    chunks = [maj_f[i:i + 9] for i in range(0, len(maj_f), 9)]
    chg_s = loc_vits[:, 4]  # p90
    for pi, ch_list in enumerate(chunks):
        can = Image.new("RGB", (105 + 6 * (2 * 96 + 6), len(ch_list) * (96 + 15)), "white")
        dr = ImageDraw.Draw(can)
        for r, c in enumerate(ch_list):
            idx = np.where(lab_f == c)[0]
            cidx = idx[clear_c[idx]]
            use_idx = cidx if len(cidx) >= 6 else idx
            cent = l2norm(X_fused[idx].mean(axis=0, keepdims=True))[0]
            top_c = [int(x) for x in use_idx[np.argsort(-(X_fused[use_idx] @ cent))][:3]]
            top_d = [int(x) for x in use_idx[np.argsort(-chg_s[use_idx])] if int(x) not in top_c][:3]
            picks = top_c + top_d
            dr.text((4, r * 111 + 18), f"C{c}\nn={len(idx)}\ng={both_gray[idx].mean()*100:.0f}%", fill="black")
            for j, ii in enumerate(picks):
                sid = sample_ids[ii]
                with Image.open(FEN_DIR / sid / "pre.jpg") as im1, Image.open(FEN_DIR / sid / "post.jpg") as im2:
                    r1 = im1.convert("RGB").resize((96, 96), Image.Resampling.BILINEAR)
                    r2 = warp_post_to_pre(im2.convert("RGB"), float(aln["best_ang"][ii]), int(aln["best_dy"][ii]), int(aln["best_dx"][ii]), 96)
                x0 = 105 + j * (2 * 96 + 6)
                y0 = r * 111 + 2
                can.paste(r1, (x0, y0))
                can.paste(r2, (x0 + 98, y0))
                tag = "C" if j < 3 else "D"
                dr.text((x0 + 2, y0 + 97), f"{tag}:{sid[:7]}:{chg_s[ii]:.2f}", fill="black")
        out_m = CACHE_V4 / f"montage_v4_fused_p{pi}.jpg"
        can.save(out_m, quality=88)
        print(f"Saved {out_m.name}", flush=True)

    np.savez_compressed(
        CACHE_V4 / "v4_audit_clusters.npz",
        X_dmg=X_dmg,
        X_scene=X_scene,
        X_fused=X_fused,
        lab_fused=lab_f,
        best_res=np.array([best_res]),
    )
    print(f"\nDONE v4 Audit! Selected res={best_res} ({len(maj_f)} major comms, mod={mod_f:.3f}, sil={sil_f:+.4f})", flush=True)


if __name__ == "__main__":
    main()
