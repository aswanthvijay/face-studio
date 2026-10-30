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
# 4: Upper, 5: Skirt, 6: Pants, 7: Dress, 8: Belt, 16: Bag, 17: Scarf/Dupatta
garment_ids = (4, 5, 6, 7, 8, 16, 17)
seg_garment = np.isin(labels, garment_ids).astype(np.uint8)

# 2. CLIPSeg Multi-Query Grounding
proc, model = get_clipseg()
queries = [
    "saree, dress, gown, clothing, garment",
    "embroidery border, lace, sleeve, fabric trim, pallu"
]
inputs = proc(text=queries, images=[img] * len(queries), padding=True, return_tensors="pt")
device = next(model.parameters()).device
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.inference_mode():
    logits = model(**inputs).logits.sigmoid().cpu().numpy()

clip_combined = np.zeros((h, w), dtype=np.float32)
for i in range(len(queries)):
    resized = cv2.resize(logits[i], (w, h), interpolation=cv2.INTER_LINEAR)
    clip_combined = np.maximum(clip_combined, resized)

clip_garment = (clip_combined > 0.25).astype(np.uint8)

# 3. Dual-Engine Fusion
garment_raw = np.maximum(seg_garment, clip_garment)

# 4. Fill gaps and connect borders
garment_closed = cv2.morphologyEx(
    garment_raw,
    cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
)

# 5. Smart Anatomical Protection:
faces = detect_face_landmarks(img)
y_coords, x_coords = np.ogrid[:h, :w]

# A. Face protection: face skin above chin line
face_protect = np.zeros((h, w), dtype=bool)
if faces:
    for f in faces:
        fx, fy, fw, fh = f["box"]
        x_min = max(0, int(fx - fw * 0.2))
        x_max = min(w, int(fx + fw * 1.2))
        chin_y = f["chin_y"]
        face_protect |= ((labels == 11) & (y_coords <= chin_y) & (x_coords >= x_min) & (x_coords <= x_max))
    # Top 30% skin safety fallback
    face_protect |= ((labels == 11) & (y_coords <= h * 0.30))
else:
    face_protect = (labels == 11) & (y_coords <= h * 0.40)

# B. Hands / bare arms protection: protect arms ONLY where not part of the garment
bare_arms = np.isin(labels, (14, 15)) & (~garment_closed.astype(bool))
bare_arms_eroded = cv2.erode(bare_arms.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)

# C. Legs / feet protection
legs_feet = np.isin(labels, (9, 10, 12, 13)) & (~garment_closed.astype(bool))

# D. Hair: protect hair above chin line; allow garment dilation under lower hair strands
hair = (labels == 2)
hair_upper = hair & (y_coords <= h * 0.35)
hair_lower_eroded = cv2.erode((hair & (~hair_upper)).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
hair_protect = hair_upper | hair_lower_eroded

# Accessories (hat, sunglasses)
head_acc = np.isin(labels, (1, 3))

# Final protection mask
protect_mask = face_protect | bare_arms_eroded | legs_feet | hair_protect | head_acc

# Apply dilation to garment but respect protection
dilated = cv2.dilate(garment_closed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
dilated[protect_mask] = 0

# Filter out disconnected background noise
min_comp_area = max(150, int(w * h * 0.0015))
num_labels, comp_labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
clean_mask = np.zeros_like(dilated)
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= min_comp_area:
        clean_mask[comp_labels == i] = 1

if not clean_mask.any():
    clean_mask = dilated

# Feathering
mask_blur = cv2.GaussianBlur(clean_mask.astype(np.float32) * 255.0, (0, 0), sigmaX=3.5, sigmaY=3.5)
final_mask = np.clip(mask_blur, 0, 255).astype(np.uint8)

# Visual overlay
overlay = img.copy()
color_layer = np.zeros((h, w, 3), dtype=np.uint8)
color_layer[:] = [14, 165, 233]  # Sky Cyan
alpha = (final_mask.astype(float) / 255.0 * 0.45)[:, :, None]
img_arr = np.asarray(overlay).astype(float)
blended = (img_arr * (1 - alpha) + color_layer * alpha).clip(0, 255).astype(np.uint8)
preview_fusion = Image.fromarray(blended)

out_vis = Path("outputs/diagnostic_tests/fixed_f6ee97d9.png")
preview_fusion.save(out_vis)

cov = float((final_mask > 127).mean())
print(f"Fixed Dual-Engine Mask Coverage: {cov*100:.1f}%")
print(f"Saved visual overlay to: {out_vis}")
