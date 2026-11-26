"""Extract DINOv2 visual embeddings for Task-2 (World Phenomena).
Pure-Vision Image Foundation Model (ViT-S/14), 100% compliant with competition rules.
Extracts:
  - feats_pre.npy (3180, 384)
  - feats_post.npy (3180, 384)
  - sample_ids.json (3180,)
"""
import os
import json
import time
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def extract_all():
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading DINOv2 vits14 on {device}...')
    
    # Load model from torch hub
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.to(device)
    model.eval()
    
    fen_dir = Path('Task-2/Fenomena')
    sub_sample = Path('Task-2/sample_submission.csv')
    
    import pandas as pd
    sample_df = pd.read_csv(sub_sample)
    sample_ids = sample_df['Id'].tolist()
    print(f'Found {len(sample_ids)} samples in sample_submission.csv')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    out_dir = Path('ghani/task2/cache/v2')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    BATCH_SIZE = 32
    feats_pre = []
    feats_post = []
    
    print('Starting batched inference...')
    with torch.no_grad():
        for i in range(0, len(sample_ids), BATCH_SIZE):
            batch_ids = sample_ids[i:i + BATCH_SIZE]
            pre_tensors = []
            post_tensors = []
            
            for sid in batch_ids:
                folder = fen_dir / sid
                with Image.open(folder / 'pre.jpg') as im:
                    pre_tensors.append(transform(im.convert('RGB')))
                with Image.open(folder / 'post.jpg') as im:
                    post_tensors.append(transform(im.convert('RGB')))
                    
            pre_batch = torch.stack(pre_tensors).to(device)
            post_batch = torch.stack(post_tensors).to(device)
            
            emb_pre = model(pre_batch).cpu().numpy()
            emb_post = model(post_batch).cpu().numpy()
            
            feats_pre.append(emb_pre)
            feats_post.append(emb_post)
            
            if (i + BATCH_SIZE) % 160 == 0 or (i + BATCH_SIZE) >= len(sample_ids):
                print(f'  Done {min(i + BATCH_SIZE, len(sample_ids))}/{len(sample_ids)} in {time.time() - t0:.1f}s...', flush=True)
                
    feats_pre = np.vstack(feats_pre)
    feats_post = np.vstack(feats_post)
    
    np.save(out_dir / 'feats_pre.npy', feats_pre)
    np.save(out_dir / 'feats_post.npy', feats_post)
    with open(out_dir / 'sample_ids.json', 'w') as f:
        json.dump(sample_ids, f)
        
    print(f'SUCCESS! Saved feats_pre shape {feats_pre.shape} and feats_post shape {feats_post.shape} in {time.time() - t0:.1f}s')

if __name__ == '__main__':
    extract_all()
