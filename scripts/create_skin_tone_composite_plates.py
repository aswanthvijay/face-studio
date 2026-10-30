"""
Create Composite Forensic Review Plates for 24-Portrait Melanin Benchmark
Face Studio Strictly Confined Protocol (Port 7870)

Outputs:
1. plate_melanin_spectrum_fair_to_dark.png (Panoramic 24-portrait comparison)
2. plate_micro_texture_zoom.png (Macro pore & eye catchlight inspection)
"""

import os
import sys
import json
import shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

WORKSPACE = Path(r"D:\face_studio")
sys.path.insert(0, str(WORKSPACE))
BENCHMARK_DIR = WORKSPACE / "inputs" / "skin_tone_benchmark"
OUTPUTS_DIR = WORKSPACE / "outputs" / "skin_tone_benchmark"
ARTIFACTS_DIR = Path(r"C:\Users\Aswanth Vijay\.gemini\antigravity\brain\fc0f9f06-179d-4970-afe7-2e9aaf565484")

METRICS_PATH = OUTPUTS_DIR / "skin_tone_audit_metrics.json"

TIER_DISPLAY = {
    "tier_very_fair": "Tier 1: Very Fair (L* ≥ 185)",
    "tier_fair": "Tier 2: Fair (170-185)",
    "tier_light_wheatish": "Tier 3: Light Wheatish (155-170)",
    "tier_medium_tan": "Tier 4: Medium Tan (140-155)",
    "tier_dusky": "Tier 5: Dusky (125-140)",
    "tier_deep_brown": "Tier 6: Deep Brown (110-125)",
    "tier_dark_espresso": "Tier 7: Dark Espresso (90-110)",
    "tier_deep_ebony": "Tier 8: Deep Ebony (L* < 90)"
}

def get_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()

def create_panoramic_plate(results: list) -> Path:
    """Creates panoramic review plate displaying 24 Source vs Output pairs across 8 tiers."""
    # Organize by tiers (8 tiers x 3 portraits = 24 total)
    # Layout: 8 columns (tiers), each column has 3 vertical pairs (Source on top, Output below)
    # Card width = 240, height = 320 each image
    card_w = 230
    card_h = 300
    col_w = card_w + 20
    header_h = 100
    tier_header_h = 45
    card_footer_h = 75
    pair_h = (card_h * 2) + card_footer_h + 30
    total_h = header_h + tier_header_h + (pair_h * 3) + 40
    total_w = (col_w * 8) + 40

    canvas = Image.new("RGB", (total_w, total_h), (18, 20, 26))
    draw = ImageDraw.Draw(canvas)

    title_font = get_font(32)
    sub_font = get_font(18)
    tier_font = get_font(15)
    body_font = get_font(13)
    badge_font = get_font(12)

    # Main Title
    draw.text((30, 20), "FACE STUDIO - COMPREHENSIVE MULTI-MELANIN FORENSIC AUDIT (24 PORTRAITS)", fill=(255, 255, 255), font=title_font)
    draw.text((30, 60), "Clinical Melanin Spectrum from Very Fair (L* ≥ 185) to Deep Ebony (L* < 90) | Model: Flux-2-Klein 4B Turbo (Full Scene)", fill=(180, 190, 205), font=sub_font)

    # Group results by tier
    tier_order = [
        "tier_very_fair", "tier_fair", "tier_light_wheatish", "tier_medium_tan",
        "tier_dusky", "tier_deep_brown", "tier_dark_espresso", "tier_deep_ebony"
    ]
    tiers_data = {t: [] for t in tier_order}
    for r in results:
        t = r.get("target_tier", "tier_medium_tan")
        if t in tiers_data:
            tiers_data[t].append(r)

    for col_idx, tier_key in enumerate(tier_order):
        cx = 30 + (col_idx * col_w)
        cy = header_h

        # Tier Header Box
        tier_title = TIER_DISPLAY.get(tier_key, tier_key)
        draw.rectangle([cx, cy, cx + card_w, cy + 34], fill=(32, 38, 52), outline=(55, 68, 92), width=1)
        draw.text((cx + 10, cy + 9), tier_title, fill=(240, 245, 255), font=tier_font)

        items = tiers_data[tier_key]
        for row_idx, item in enumerate(items[:3]):
            py = cy + tier_header_h + (row_idx * pair_h)

            # Source Image
            src_file = BENCHMARK_DIR / item["source_filename"]
            out_file = OUTPUTS_DIR / item["output_filename"]

            src_img = Image.open(src_file).convert("RGB") if src_file.exists() else Image.new("RGB", (card_w, card_h), (40, 40, 40))
            out_img = Image.open(out_file).convert("RGB") if out_file.exists() else Image.new("RGB", (card_w, card_h), (40, 40, 40))

            src_thumb = src_img.resize((card_w, card_h), Image.Resampling.LANCZOS)
            out_thumb = out_img.resize((card_w, card_h), Image.Resampling.LANCZOS)

            # Paste Source
            canvas.paste(src_thumb, (cx, py))
            draw.rectangle([cx, py, cx + card_w, py + card_h], outline=(70, 80, 100), width=1)
            # Label "SOURCE"
            draw.rectangle([cx + 6, py + 6, cx + 75, py + 26], fill=(0, 0, 0, 180))
            draw.text((cx + 12, py + 9), "SOURCE", fill=(255, 220, 100), font=badge_font)

            # Paste Output
            out_y = py + card_h + 4
            canvas.paste(out_thumb, (cx, out_y))
            draw.rectangle([cx, out_y, cx + card_w, out_y + card_h], outline=(70, 80, 100), width=1)
            # Label "GENERATED"
            draw.rectangle([cx + 6, out_y + 6, cx + 90, out_y + 26], fill=(0, 0, 0, 180))
            draw.text((cx + 12, out_y + 9), "GENERATED", fill=(100, 220, 255), font=badge_font)

            # Metrics Card Box
            card_box_y = out_y + card_h + 4
            d_l = item.get("delta_l", 0.0)
            d_e = item.get("delta_e", 0.0)
            sim = item.get("arcface_similarity", 0.0) or 0.0
            lap = item.get("laplacian_variance", 0.0) or 0.0
            defects = item.get("defect_flags", [])

            # Status color
            box_outline = (50, 180, 80) if not defects else (220, 60, 50)
            bg_fill = (22, 28, 38)
            draw.rectangle([cx, card_box_y, cx + card_w, card_box_y + card_footer_h], fill=bg_fill, outline=box_outline, width=1)

            # Text Metrics
            l_color = (255, 255, 255) if abs(d_l) <= 5.0 else ((255, 120, 100) if d_l > 8.0 else (100, 180, 255))
            draw.text((cx + 8, card_box_y + 6), f"ΔL*: {d_l:+4.1f} | ΔE: {d_e:.1f}", fill=l_color, font=body_font)
            draw.text((cx + 8, card_box_y + 24), f"Face ID Sim: {sim:.3f}", fill=(220, 230, 245), font=body_font)
            draw.text((cx + 8, card_box_y + 42), f"Texture Var: {lap:.0f} (Pores)", fill=(180, 220, 200), font=body_font)

            verdict_text = "PASS" if not defects else "ALERT: " + "/".join(defects[:2])
            verdict_col = (100, 240, 140) if not defects else (255, 100, 90)
            draw.text((cx + 8, card_box_y + 58), verdict_text, fill=verdict_col, font=badge_font)

    plate_path = OUTPUTS_DIR / "plate_melanin_spectrum_fair_to_dark.png"
    canvas.save(plate_path, "PNG", quality=95)
    print(f"[Composite] Panoramic plate saved to: {plate_path}")
    return plate_path


def create_macro_zoom_plate(results: list) -> Path:
    """Creates macro 300% zoom plate inspecting pores, catchlights, and texture."""
    # Select 4 representative portraits: Fair (P01), Tan (P10), Brown (P16), Deep Ebony (P22)
    selected_ids = ["P01_very_fair_1", "P10_medium_tan_1", "P16_deep_brown_1", "P22_deep_ebony_1"]
    items = [r for r in results if r["portrait_id"] in selected_ids]
    if len(items) < 4:
        items = results[:4]

    w, h = 1920, 1080
    canvas = Image.new("RGB", (w, h), (16, 18, 24))
    draw = ImageDraw.Draw(canvas)

    title_font = get_font(30)
    sub_font = get_font(18)
    lbl_font = get_font(16)
    stat_font = get_font(14)

    draw.text((40, 30), "FACE STUDIO - EPIDERMAL MICRO-PORE & PHOTOREALISM MACRO AUDIT (300% ZOOM)", fill=(255, 255, 255), font=title_font)
    draw.text((40, 75), "Forensic inspection of skin pores, corneal catchlights, dermal relief, and absence of waxy/plastic smoothing", fill=(170, 185, 205), font=sub_font)

    col_w = 440
    crop_size = 420
    start_y = 130

    for idx, it in enumerate(items):
        cx = 40 + (idx * (col_w + 25))

        out_file = OUTPUTS_DIR / it["output_filename"]
        if not out_file.exists(): continue
        im = Image.open(out_file).convert("RGB")
        im_cv = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

        # Detect face for precise cheek/eye macro crop
        from core.face_engine import get_face_modules, detect_faces_robust
        app, _, _, _, _ = get_face_modules()
        faces = detect_faces_robust(app, im_cv)
        if faces:
            f = faces[0]
            fx1, fy1, fx2, fy2 = [int(v) for v in f.bbox]
            # Center on cheek and nose bridge
            center_x = (fx1 + fx2) // 2
            center_y = int(fy1 + (fy2 - fy1) * 0.45)
        else:
            center_x = im.width // 2
            center_y = int(im.height * 0.35)

        half = 80 # 160x160 area zoomed 300%
        crop_box = (
            max(0, center_x - half),
            max(0, center_y - half),
            min(im.width, center_x + half),
            min(im.height, center_y + half)
        )
        face_macro = im.crop(crop_box).resize((crop_size, crop_size), Image.Resampling.LANCZOS)
        canvas.paste(face_macro, (cx, start_y))
        draw.rectangle([cx, start_y, cx + crop_size, start_y + crop_size], outline=(80, 100, 140), width=2)

        # Macro Label & Stats
        draw.rectangle([cx + 10, start_y + 10, cx + 180, start_y + 36], fill=(0, 0, 0, 200))
        draw.text((cx + 16, start_y + 14), f"300% MACRO ZOOM", fill=(255, 215, 0), font=lbl_font)

        # Bottom info card
        card_y = start_y + crop_size + 15
        draw.rectangle([cx, card_y, cx + crop_size, card_y + 140], fill=(24, 30, 42), outline=(60, 75, 105), width=1)
        draw.text((cx + 15, card_y + 12), f"Subject: {it['portrait_id']}", fill=(255, 255, 255), font=lbl_font)
        draw.text((cx + 15, card_y + 36), f"Tier: {it.get('target_tier', '')}", fill=(180, 210, 255), font=stat_font)
        draw.text((cx + 15, card_y + 60), f"Laplacian Var: {it.get('laplacian_variance', 0):.1f} (>700 = Authentic DSLR)", fill=(130, 240, 170), font=stat_font)
        draw.text((cx + 15, card_y + 84), f"Face ID Cosine: {it.get('arcface_similarity', 0):.4f}", fill=(240, 240, 240), font=stat_font)
        draw.text((cx + 15, card_y + 108), f"Melanin ΔL*: {it.get('delta_l', 0):+5.1f} | ΔE: {it.get('delta_e', 0):.2f}", fill=(255, 200, 150), font=stat_font)

    macro_path = OUTPUTS_DIR / "plate_micro_texture_zoom.png"
    canvas.save(macro_path, "PNG", quality=95)
    print(f"[Composite] Macro zoom plate saved to: {macro_path}")
    return macro_path

def main():
    if not METRICS_PATH.exists():
        print(f"[ERROR] Metrics file not found at {METRICS_PATH}")
        return

    with open(METRICS_PATH, "r") as f:
        results = json.load(f)

    print(f"Loaded {len(results)} portrait metrics. Generating plates...")
    pan_path = create_panoramic_plate(results)
    macro_path = create_macro_zoom_plate(results)

    # Copy to artifacts directory
    if ARTIFACTS_DIR.exists():
        shutil.copy(pan_path, ARTIFACTS_DIR / pan_path.name)
        shutil.copy(macro_path, ARTIFACTS_DIR / macro_path.name)
        print(f"[Artifacts] Copied plates to {ARTIFACTS_DIR}")

if __name__ == "__main__":
    main()
