import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image
import torch
from core.clothing_engine import parse_clothing_labels, get_clipseg, detect_face_landmarks

img_path = Path("inputs/src_f6ee97d9-c.png")
img = Image.open(img_path).convert("RGB")
w, h = img.size

# 1. SegFormer
labels = parse_clothing_labels(img)
garment_ids = (4, 5, 6, 7, 8, 17) # Upper, Skirt, Pants, Dress, Belt, Scarf
seg_garment = np.isin(labels, garment_ids).astype(np.uint8)

# 2. CLIPSeg with specialized fashion prompts
proc, model = get_clipseg()
queries = ["saree", "dress, gown, clothing, garment", "fabric border, sleeve, embroidery"]
inputs = proc(text=queries, images=[img]*len(queries), padding=True, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits.sigmoid().cpu().numpy()

# Combine CLIPSeg predictions
clip_combined = np.zeros((h, w), dtype=np.float32)
for i in range(len(queries)):
    resized = cv2.resize(logits[i], (w, h), interpolation=cv2.INTER_LINEAR)
    clip_combined = np.maximum(clip_combined, resized)

clip_mask = (clip_combined > 0.28).astype(np.uint8)

# 3. Ensemble: Union of SegFormer and CLIPSeg
ensemble_raw = np.maximum(seg_garment, clip_mask)

# Close small gaps in fabric
ensemble_closed = cv2.morphologyEx(
    ensemble_raw,
    cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
)

# Protect face, hair, and hands
faces = detect_face_landmarks(img)
skin = (labels == 11)
face_protect = np.zeros((h, w), dtype=bool)
if faces:
    y_coords, x_coords = np.ogrid[:h, :w]
    for f in faces:
        fx, fy, fw, fh = f["box"]
        x_min = max(0, int(fx - fw * 0.25))
        x_max = min(w, int(fx + fw * 1.25))
        chin_y = f["chin_y"]
        face_protect |= (skin & (y_coords <= chin_y) & (x_coords >= x_min) & (x_coords <= x_max))
    face_protect |= (skin & (y_coords <= h * 0.35))
else:
    face_protect = skin

hair = (labels == 2)
hair_eroded = cv2.erode(hair.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)

# Subtract face protect from ensemble
ensemble_clean = ensemble_closed.copy()
ensemble_clean[face_protect] = 0
ensemble_clean[hair_eroded] = 0

# Dilate slightly to absorb seams
ensemble_dilated = cv2.dilate(ensemble_clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
ensemble_dilated[face_protect] = 0
ensemble_dilated[hair_eroded] = 0

# Smooth Gaussian feather
feathered = cv2.GaussianBlur(ensemble_dilated.astype(np.float32) * 255.0, (0, 0), sigmaX=3.5, sigmaY=3.5)
final_mask = np.clip(feathered, 0, 255).astype(np.uint8)

# Create visual comparison overlay
overlay = img.copy()
color_layer = np.zeros((h, w, 3), dtype=np.uint8)
color_layer[:] = [14, 165, 233]  # Sky Cyan
alpha = (final_mask.astype(float) / 255.0 * 0.45)[:, :, None]
img_arr = np.asarray(overlay).astype(float)
blended = (img_arr * (1 - alpha) + color_layer * alpha).clip(0, 255).astype(np.uint8)
preview_ensemble = Image.fromarray(blended)

out_path = Path("outputs/diagnostic_tests/test_ensemble_f6ee97d9.png")
preview_ensemble.save(out_path)

print(f"SegFormer only coverage: {seg_garment.mean()*100:.1f}%")
print(f"CLIPSeg only coverage:   {clip_mask.mean()*100:.1f}%")
print(f"Ensemble final coverage: {(final_mask > 127).mean()*100:.1f}%")
print(f"Saved visual comparison to {out_path}")
