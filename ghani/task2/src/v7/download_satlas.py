"""Download and cache SatlasPretrain pure-vision aerial & satellite Swin-v2-Base weights (Apache-2.0 / ODC-BY)."""
import torch
import satlaspretrain_models

wm = satlaspretrain_models.Weights()
dummy = torch.zeros((1, 3, 256, 256), dtype=torch.float32)

for model_id in ["Aerial_SwinB_SI", "Sentinel2_SwinB_SI_RGB"]:
    print(f"Fetching {model_id}...", flush=True)
    m = wm.get_pretrained_model(model_id, fpn=False).eval()
    with torch.inference_mode():
        outs = m(dummy)
    shapes = [tuple(o.shape) for o in outs] if isinstance(outs, (list, tuple)) else [tuple(outs.shape)]
    print(f"  OK {model_id} -> output shapes: {shapes}", flush=True)
    del m
print("DONE caching SatlasPretrain weights!", flush=True)
