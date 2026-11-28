"""Download and cache SatlasPretrain pure-vision aerial & satellite Swin-v2-Base weights locally (Apache-2.0 / ODC-BY)."""
from pathlib import Path
import requests
import torch
from satlaspretrain_models.model import Model
from satlaspretrain_models.utils import SatlasPretrain_weights

ROOT = Path(__file__).resolve().parents[4]
W_DIR = ROOT / "ghani" / "task2" / "cache" / "v7" / "weights"
W_DIR.mkdir(parents=True, exist_ok=True)

dummy = torch.zeros((1, 3, 256, 256), dtype=torch.float32)

for model_id in ["Aerial_SwinB_SI", "Sentinel2_SwinB_SI_RGB"]:
    info = SatlasPretrain_weights[model_id]
    out_p = W_DIR / f"{model_id}.pth"
    if not out_p.exists() or out_p.stat().st_size < 10_000_000:
        url = info["url"]
        print(f"Downloading {model_id} from {url}...", flush=True)
        with requests.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(out_p, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print(f"  Saved {out_p} ({out_p.stat().st_size / 1e6:.1f} MB)", flush=True)
    else:
        print(f"  Cached: {out_p} ({out_p.stat().st_size / 1e6:.1f} MB)", flush=True)

    weights = torch.load(out_p, map_location="cpu", weights_only=False)
    m = Model(info["num_channels"], info["multi_image"], info["backbone"], fpn=False, weights=weights).eval()
    with torch.inference_mode():
        outs = m(dummy)
    shapes = [tuple(o.shape) for o in outs] if isinstance(outs, (list, tuple)) else [tuple(outs.shape)]
    print(f"  Verified offline load {model_id} -> feature map shapes: {shapes}", flush=True)
    del m, weights

print("DONE downloading & verifying SatlasPretrain weights!", flush=True)
