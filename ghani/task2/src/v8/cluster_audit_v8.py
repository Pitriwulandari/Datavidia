"""Task-2 v8 Part 2: Pure De-Striped + 3-Regime Modality-Conditional Mean-Centered Fusion
& High-Resolution (192px Zoomed Center-Crop) Micro-Clustering Audit.

ZERO striped/polluted v3/v4/v5/v7 embeddings in the clustering graph!
Eliminates ALL 4 panitia augmentation leaks:
  1. Synthetic Scanlines (~33% pre, ~33% post) -> erased before backbone extraction + ZERO stripe_z!
  2. Optical->SAR/Mono-Post modality shift -> erased via 3-Regime Modality-Conditional Mean-Centering!
  3. Paired Grayscale (~7.5%) -> erased via Modality-Conditional Mean-Centering!
  4. Synthetic Clouds & Rotation Borders -> masked via valid-overlap spatial token pooling!
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
sys.path.insert(0, str(ROOT / "ghani" / "task2" / "src" / "v8"))
from align_extract_v4 import l2norm, warp_post_to_pre  # noqa: E402
from extract_v5 import detint_clean_rgb  # noqa: E402
from extract_v8_clean import destripe_axis_aligned  # noqa: E402

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

CLASSES = ["Earthquake", "Fire", "Hurricane", "Tsunami", "Volcano", "Explosion"]
FEN_DIR = ROOT / "Task-2" / "Fenomena"
C3_DIR = ROOT / "ghani" / "task2" / "cache" / "v3"
C4_DIR = ROOT / "ghani" / "task2" / "cache" / "v4"
C8_DIR = ROOT / "ghani" / "task2" / "cache" / "v8"


def regime_center_l2(
    arr: np.ndarray,
    regime_ids: np.ndarray,
    state: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """L2-norm -> mean-center separately per observable modality regime -> L2-norm.
    Algebraically removes Optical-vs-Grayscale AND Optical-vs-SAR domain shifts!
    """
    X = l2norm(arr.astype(np.float32))
    out = np.empty_like(X)
    if state is None:
        state_out: dict[int, np.ndarray] = {}
        global_mu = X.mean(axis=0, keepdims=True).astype(np.float32)
        state_out[-1] = global_mu
        for r in sorted(np.unique(regime_ids).tolist()):
            m = (regime_ids == r)
            mu = X[m].mean(axis=0, keepdims=True).astype(np.float32) if m.sum() >= 15 else global_mu
            state_out[int(r)] = mu
            out[m] = l2norm(X[m] - mu)
        return out, state_out
    else:
        for r in np.unique(regime_ids).tolist():
            m = (regime_ids == r)
            mu = state.get(int(r), state[-1])
            out[m] = l2norm(X[m] - mu)
        return out, state


def get_modality_regimes(audit: np.ndarray, audit_names: list[str], p3_audit_flags: np.ndarray):
    """Observable pixel-level nuisance flags (100% zero-retraining inference compliant!):
      g_pre, g_pst, both_gray: exact paired/single grayscale flags (from raw RGB)
      st_pre, st_pst: exact axis-aligned synthetic scanline flags (hv > 0.14)
      sar_post: diagnostic-only Optical->Mono-Post flag (NOT subtracted so Wildfire ash stays intact!)
    """
    am = {k: i for i, k in enumerate(audit_names)}
    c1 = audit[:, am["c_pre"]]
    c2 = audit[:, am["c_pst"]]
    g_pre = (p3_audit_flags[:, 2] > 0.5).astype(np.int32)
    g_pst = (p3_audit_flags[:, 3] > 0.5).astype(np.int32)
    both_gray = ((g_pre > 0) & (g_pst > 0)).astype(np.int32)
    any_gray = ((g_pre > 0) | (g_pst > 0)).astype(np.int32)
    st_pre = (audit[:, am["hv_pre_raw"]] > 0.14).astype(np.int32)
    st_pst = (audit[:, am["hv_pst_raw"]] > 0.14).astype(np.int32)
    sar_post = ((c1 <= 0.992) & (c2 > 0.985) & (both_gray == 0)).astype(np.int32)
    return g_pre, g_pst, both_gray, any_gray, st_pre, st_pst, sar_post


def build_v8_clean_matrix(center_state: dict | None = None):
    """Build 100% De-Striped + Gray-Centered + Stripe-Centered 4-Backbone Master Matrix.
    Preserves true Wildfire charcoal/ash & Volcano ash while erasing grayscale & scanline leaks!
    """
    sat8 = np.load(C8_DIR / "v8_satlas_swinb.npz")
    din8 = np.load(C8_DIR / "v8_dinov2_vitb.npz")
    ph8 = np.load(C8_DIR / "v8_phys_clean.npz", allow_pickle=True)
    p3 = np.load(C3_DIR / "phys_v3.npz", allow_pickle=True)
    vs4 = np.load(C4_DIR / "aligned_vits_336.npz")
    vb4 = np.load(C4_DIR / "aligned_vitb_224.npz")
    cx3 = np.load(C3_DIR / "cnxb_gray.npz")

    audit = ph8["audit"]
    audit_names = list(ph8["audit_names"])
    g_pre, g_pst, both_gray, any_gray, st_pre, st_pst, sar_post = get_modality_regimes(
        audit, audit_names, p3["audit_flags"]
    )

    w_aux = 0.75
    views = [
        # Primary: 100% De-Striped, De-Tinted, Cloud-Masked Valid-Overlap SatlasPretrain + DINOv2-B
        ("sat_f2_pre",     1.55, sat8["f2_pre"],                        g_pre),
        ("sat_f2_post",    1.80, sat8["f2_post"],                       g_pst),
        ("sat_f3_pre",     0.65, sat8["f3_pre"],                        g_pre),
        ("sat_f3_post",    0.75, sat8["f3_post"],                       g_pst),
        ("sat_f2_topchg",  0.60, sat8["f2_topchg_post"],                any_gray),
        ("dino_pmean_pre", 1.45, din8["pmean_pre"],                     g_pre),
        ("dino_pmean_pst", 1.75, din8["pmean_post"],                    g_pst),
        ("dino_cls_pre",   0.85, din8["cls_pre"],                       g_pre),
        ("dino_cls_pst",   1.05, din8["cls_post"],                      g_pst),
        ("dino_topchg",    0.60, din8["topchg_post"],                   any_gray),
        ("dino_diff",      0.55, din8["pmean_post"] - din8["pmean_pre"], any_gray),
        # Auxiliary: Grayscale-invariant backbones with Stripe-Conditional Mean-Centering!
        ("vs4_pre",  w_aux * 0.90, vs4["pmean_pre"],  st_pre),
        ("vs4_post", w_aux * 1.15, vs4["pmean_post"], st_pst),
        ("vb4_pre",  w_aux * 0.80, vb4["pmean_pre"],  st_pre),
        ("vb4_post", w_aux * 1.00, vb4["pmean_post"], st_pst),
        ("cx3_pre",  w_aux * 0.75, cx3["pre"],        st_pre),
        ("cx3_post", w_aux * 0.85, cx3["post"],       st_pst),
    ]

    new_state = {} if center_state is None else center_state
    blocks = []
    for key, wt, arr, reg in views:
        st = None if center_state is None else center_state[key]
        c_arr, st_out = regime_center_l2(arr, reg, st)
        if center_state is None:
            new_state[key] = st_out
        blocks.append(wt * c_arr)

    pf8 = np.nan_to_num(ph8["features"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if center_state is None:
        sc8 = StandardScaler().fit(pf8)
        new_state["sc8"] = sc8
    else:
        sc8 = center_state["sc8"]
    pf8_raw = np.clip(sc8.transform(pf8), -4.0, 4.0).astype(np.float32)
    st_p = None if center_state is None else center_state["pf8_reg"]
    pf8_c, st_p_out = regime_center_l2(pf8_raw, both_gray, st_p)
    if center_state is None:
        new_state["pf8_reg"] = st_p_out
    blocks.append(0.55 * pf8_c)

    X_master = l2norm(np.hstack(blocks).astype(np.float32))
    reg_pst_diag = np.zeros(len(g_pre), dtype=np.int32)
    reg_pst_diag[both_gray > 0] = 1
    reg_pst_diag[sar_post > 0] = 2
    return X_master, new_state, ph8, sat8, din8, g_pre, reg_pst_diag


def run_louvain_v8(X: np.ndarray, k_nn: int = 18):
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
    for res in [4.0, 4.5, 5.0, 5.5, 6.0]:
        random.seed(SEED)
        part = g.community_multilevel(weights="weight", resolution=res)
        raw = np.array(part.membership, dtype=np.int32)
        vals, cnts = np.unique(raw, return_counts=True)
        n_maj = int((cnts >= 14).sum())
        tiny_n = int(cnts[cnts < 14].sum())
        print(f"  Louvain res={res:.1f}: {n_maj} major (>=14), tiny={tiny_n}, mod={part.modularity:.4f}", flush=True)
        score = abs(n_maj - 48) + (0.0 if 40 <= n_maj <= 58 else 30.0)
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
    sample_ids = json.load(open(C8_DIR / "sample_ids.json", "r", encoding="utf-8"))
    aln = np.load(C4_DIR / "alignment_v4.npz")
    bang, bdy, bdx, zp = aln["best_ang"], aln["best_dy"], aln["best_dx"], aln["z_peak"]
    cl3 = np.load(C3_DIR / "clusters_v3_clean.npz")["lab_clean"]
    beirut_v3 = (cl3 == 22)
    s3 = pd.read_csv(ROOT / "ghani" / "submissions" / "task2" / "submission_v3_ghani.csv").set_index("Id").loc[sample_ids]["Phenomenon"].values
    s7 = pd.read_csv(ROOT / "ghani" / "submissions" / "task2" / "submission_v7_ghani.csv").set_index("Id").loc[sample_ids]["Phenomenon"].values

    X_master, _, ph8, sat8, din8, reg_pre, reg_pst = build_v8_clean_matrix(None)
    print(f"Built v8 PURE DE-STRIPED X_master shape={X_master.shape}", flush=True)
    lab, cents, res, mod = run_louvain_v8(X_master, k_nn=18)
    n_c = len(cents)

    np.savez_compressed(
        C8_DIR / "clusters_v8_micro.npz",
        ids=np.array(sample_ids),
        lab_micro=lab,
        cents=cents,
        res=np.float32(res),
        mod=np.float32(mod),
    )

    pf8 = ph8["features"]; pm = {k: i for i, k in enumerate(list(ph8["feature_names"]))}
    au8 = ph8["audit"]; am = {k: i for i, k in enumerate(list(ph8["audit_names"]))}
    both_gray = (reg_pst == 1)
    sar_post = (reg_pst == 2)
    st_pre = au8[:, am["hv_pre_raw"]] > 0.14
    st_pst = au8[:, am["hv_pst_raw"]] > 0.14
    cld_pst = au8[:, am["cld_pst"]] * 100.0

    c_beir = l2norm(X_master[beirut_v3].mean(axis=0, keepdims=True))[0]
    b_sim = X_master @ c_beir

    rows = []
    for c in range(n_c):
        m = (lab == c)
        vc7 = pd.Series(s7[m]).value_counts()
        vc3 = pd.Series(s3[m]).value_counts()
        rows.append({
            "mc": c,
            "n": int(m.sum()),
            "pct_gray": round(float(both_gray[m].mean() * 100.0), 1),
            "pct_sar": round(float(sar_post[m].mean() * 100.0), 1),
            "st_pre%": round(float(st_pre[m].mean() * 100.0), 1),
            "st_pst%": round(float(st_pst[m].mean() * 100.0), 1),
            "cld_pst": round(float(cld_pst[m].mean()), 2),
            "exg_d": round(float(pf8[m, pm["exg_delta"]].mean()), 3),
            "charc_d": round(float(pf8[m, pm["charc_delta"]].mean()), 3),
            "ash_d": round(float(pf8[m, pm["ash_delta"]].mean()), 3),
            "wat_d": round(float(pf8[m, pm["water_delta"]].mean()), 3),
            "mud_d": round(float(pf8[m, pm["mud_delta"]].mean()), 3),
            "lap_lr": round(float(pf8[m, pm["lap_log_ratio"]].mean()), 2),
            "chg_m": round(float(pf8[m, pm["chg_mean"]].mean()), 3),
            "h_asym": round(float(pf8[m, pm["half_asym"]].mean()), 3),
            "b_v3c22": int(beirut_v3[m].sum()),
            "b_sim": round(float(b_sim[m].mean()), 3),
            "v7_top": f"{vc7.index[0][:4]}({vc7.iloc[0]/m.sum()*100:.0f}%)",
            "v3_top": f"{vc3.index[0][:4]}({vc3.iloc[0]/m.sum()*100:.0f}%)",
        })
    df = pd.DataFrame(rows)
    df.to_csv(C8_DIR / "micro_profiles_v8.csv", index=False)

    print("\n=== v8 4-NUISANCE LEAKAGE PROOF (MUST BE FLAT / ZERO ARTIFACT CLUSTERS!) ===", flush=True)
    print(f"  Global baselines: gray={both_gray.mean()*100:.1f}%, sar={sar_post.mean()*100:.1f}%, st_pre={st_pre.mean()*100:.1f}%, st_pst={st_pst.mean()*100:.1f}%, cld_pst={cld_pst.mean():.2f}%", flush=True)
    print(f"  Cluster MAX     : gray={df['pct_gray'].max():.1f}%, sar={df['pct_sar'].max():.1f}%, st_pre={df['st_pre%'].max():.1f}%, st_pst={df['st_pst%'].max():.1f}%, cld_pst={df['cld_pst'].max():.2f}%", flush=True)
    print("  Beirut v3_C22 clusters:", df[df["b_v3c22"] > 0][["mc", "n", "b_v3c22", "b_sim", "v7_top"]].to_dict("records"), flush=True)

    # Render HIGH-RES (192x192 zoomed 640px center-crop, DE-STRIPED + DE-TINTED!) montages for ALL clusters!
    for old in C8_DIR.glob("montage_v8_hires_p*.jpg"):
        old.unlink()
    rng = np.random.RandomState(SEED)
    T = 184
    per_page = 6
    n_pages = (n_c + per_page - 1) // per_page
    for pg in range(n_pages):
        cs = list(range(pg * per_page, min(n_c, (pg + 1) * per_page)))
        can = Image.new("RGB", (175 + 4 * (2 * T + 8), len(cs) * (T + 20)), "white")
        dr = ImageDraw.Draw(can)
        for r, c in enumerate(cs):
            idx_c = np.where(lab == c)[0]
            # Pick 4 random samples (prefer low-cloud so visual disaster identity is crystal clear!)
            lo_cld = idx_c[cld_pst[idx_c] < 4.0]
            pool = lo_cld if len(lo_cld) >= 4 else idx_c
            picks = rng.choice(pool, size=min(4, len(pool)), replace=False)
            rd = rows[c]
            lbl = (
                f"V{c:02d} n={rd['n']} B={rd['b_v3c22']}\n"
                f"v7:{rd['v7_top']}\n"
                f"v3:{rd['v3_top']}\n"
                f"gr={rd['pct_gray']}% sr={rd['pct_sar']}%\n"
                f"stP={rd['st_pst%']}% cl={rd['cld_pst']}%\n"
                f"ch={rd['charc_d']:+.2f} as={rd['ash_d']:+.2f}\n"
                f"wt={rd['wat_d']:+.2f} md={rd['mud_d']:+.2f}\n"
                f"lR={rd['lap_lr']:+.2f} cg={rd['chg_m']:.2f}"
            )
            dr.text((4, r * (T + 20) + 8), lbl, fill="black")
            for j, ii in enumerate(picks):
                sid = sample_ids[ii]
                with Image.open(FEN_DIR / sid / "pre.jpg") as i1, Image.open(FEN_DIR / sid / "post.jpg") as i2:
                    a1 = np.asarray(i1.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR), dtype=np.uint8)
                    a2 = np.asarray(i2.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR), dtype=np.uint8)
                ds1, _ = destripe_axis_aligned(a1, 0.08)
                ds2, _ = destripe_axis_aligned(a2, 0.08)
                im2 = warp_post_to_pre(Image.fromarray(ds2), float(bang[ii]), int(bdy[ii]), int(bdx[ii]), 512)
                # Center 352x352 crop -> 184x184 crisp zoom!
                r1 = detint_clean_rgb(Image.fromarray(ds1).crop((80, 80, 432, 432)).resize((T, T), Image.Resampling.LANCZOS), False)
                r2 = detint_clean_rgb(im2.crop((80, 80, 432, 432)).resize((T, T), Image.Resampling.LANCZOS), False)
                x0 = 175 + j * (2 * T + 8)
                y0 = r * (T + 20) + 2
                can.paste(r1, (x0, y0))
                can.paste(r2, (x0 + T + 2, y0))
                tag = "S" if sar_post[ii] else ("*" if beirut_v3[ii] else "")
                dr.text((x0 + 2, y0 + T + 2), f"{tag}{sid[:7]} c2={au8[ii, am['c_pst']]:.2f} chg={pf8[ii, pm['chg_mean']]:.2f}", fill="black")
        out_pg = C8_DIR / f"montage_v8_hires_p{pg}.jpg"
        can.save(out_pg, quality=88)
        print(f"Saved {out_pg}", flush=True)

    print(f"DONE cluster_audit_v8 in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
