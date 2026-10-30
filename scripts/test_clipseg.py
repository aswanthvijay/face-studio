import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
from core.clothing_engine import get_clipseg, parse_clothing_labels
import torch

img = Image.open('inputs/src_f6ee97d9-c.png').convert('RGB')
proc, model = get_clipseg()
prompts = ['dress, saree, clothing', 'embroidery border, lace, sleeve', 'clothes, fabric']
inputs = proc(text=prompts, images=[img]*len(prompts), padding=True, return_tensors='pt')
with torch.no_grad():
    logits = model(**inputs).logits.sigmoid().cpu().numpy()

for i, p in enumerate(prompts):
    cov = (logits[i] > 0.25).mean()
    print(f'CLIPSeg for "{p}": coverage {cov*100:.1f}%, max prob {logits[i].max():.3f}')
