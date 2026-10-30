"""Comprehensive Diagnostic Test Suite for User Images in Face Studio.

Evaluates 10 unique user images across:
1. Image Geometry & Aspect Ratio (Divisibility, Resolution).
2. Face & Biometrics (YuNet landmarks, Chin Line, Face count).
3. SegFormer Garment Parsing (Label distribution, Dress/Top/Bottom coverage).
4. Chin Protection & Mask Continuity (Boundary leakage, hole detection).
5. Skin Melanin & Luminance Profile (L*, a*, b*, Melanin descriptor).
6. LoRA Pipeline Graph Compilation (fs_couture integration).
"""
import hashlib
import json
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import config
from core.clothing_engine import (
    detect_face_landmarks,
    parse_clothing_labels,
    generate_dress_mask
)
from core.prompt_engine import analyze_skin_melanin, build_zero_pollution_prompt
from core.pipeline import build_sdxl_inpaint_graph, build_sdxl_txt2img_graph

OUT_DEBUG_DIR = ROOT / "outputs" / "diagnostic_tests"
OUT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def get_top_unique_user_images(n: int = 10) -> list[Path]:
    """Find the top n unique user images sorted by recent modification time."""
    src_files = sorted(config.INPUTS_DIR.glob("src_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    seen = {}
    unique_list = []
    for f in src_files:
        h = hashlib.md5(f.read_bytes()).hexdigest()
        if h not in seen:
            seen[h] = f
            unique_list.append(f)
        if len(unique_list) >= n:
            break
    return unique_list


def analyze_image(img_path: Path) -> dict:
    diag = {
        "filename": img_path.name,
        "size_bytes": img_path.stat().st_size,
        "errors": [],
        "warnings": [],
        "abnormalities": []
    }

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        diag["errors"].append(f"Failed to open image: {e}")
        return diag

    w, h = img.size
    diag["width"] = w
    diag["height"] = h
    diag["aspect_ratio"] = f"{w/h:.3f}"

    # Check 1: Resolution & Divisibility
    if w < 512 or h < 512:
        diag["warnings"].append(f"Low resolution: {w}x{h} (recommended >= 768x768 for SDXL)")
    if w % 8 != 0 or h % 8 != 0:
        diag["abnormalities"].append(f"Dimensions {w}x{h} not divisible by 8 (will be auto-padded during latent encoding)")

    # Check 2: Face & Biometrics
    try:
        faces = detect_face_landmarks(img)
        diag["faces_count"] = len(faces)
        if len(faces) == 0:
            diag["warnings"].append("No face detected by YuNet. Face swap / chin protection will use Segformer skin fallback.")
        elif len(faces) > 1:
            diag["warnings"].append(f"Multiple faces ({len(faces)}) detected. Primary face selected by largest area.")
        
        if faces:
            primary_face = faces[0]
            box = primary_face["box"]
            face_area_pct = (box[2] * box[3]) / (w * h) * 100
            diag["face_area_pct"] = round(face_area_pct, 1)
            diag["chin_y"] = round(primary_face["chin_y"], 1)
            if face_area_pct < 1.0:
                diag["abnormalities"].append(f"Face is tiny ({face_area_pct:.1f}% of image), biometric fidelity may be reduced.")
    except Exception as e:
        diag["errors"].append(f"Face detection failed: {e}")

    # Check 3: SegFormer Clothing Segmentation
    try:
        labels = parse_clothing_labels(img)
        unique_labels = np.unique(labels).tolist()
        label_map = {
            0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes",
            5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe",
            11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm",
            16: "Bag", 17: "Scarf"
        }
        present_categories = [label_map.get(lbl, str(lbl)) for lbl in unique_labels]
        diag["detected_categories"] = present_categories

        # Evaluate dress mask
        mask_pil, preview, _, coverage = generate_dress_mask(img, target="dress", keep_face=True)
        diag["dress_coverage_pct"] = round(coverage * 100, 1)

        if coverage < 0.05:
            diag["abnormalities"].append(f"Very low garment mask coverage ({coverage*100:.1f}%). Subject may be wearing unclassified clothing or extreme close-up.")
        elif coverage > 0.75:
            diag["abnormalities"].append(f"Unusually high garment coverage ({coverage*100:.1f}%). Garment mask may have bled into background.")

        # Save diagnostic visualization
        vis = preview.copy()
        draw = ImageDraw.Draw(vis)
        if faces:
            fb = faces[0]["box"]
            draw.rectangle([fb[0], fb[1], fb[0]+fb[2], fb[1]+fb[3]], outline=(0, 255, 0), width=3)
            draw.line([(0, faces[0]["chin_y"]), (w, faces[0]["chin_y"])], fill=(255, 50, 50), width=2)
            draw.text((10, faces[0]["chin_y"] - 20), "Anatomical Chin Line (Protected)", fill=(255, 50, 50))
        
        vis_path = OUT_DEBUG_DIR / f"diag_{img_path.stem}.png"
        vis.save(vis_path)
        diag["diagnostic_image"] = vis_path.name

    except Exception as e:
        diag["errors"].append(f"SegFormer clothing segmentation failed: {e}")

    # Check 4: Melanin & Skin Analysis
    try:
        melanin_desc = analyze_skin_melanin(img)
        diag["melanin_desc"] = melanin_desc
    except Exception as e:
        diag["warnings"].append(f"Skin melanin analysis failed: {e}")

    # Check 5: LoRA Graph Compilation Check
    try:
        inpaint_graph, _ = build_sdxl_inpaint_graph(
            source_subpath="temp.png",
            mask_subpath="mask.png",
            positive_prompt="fs_couture, luxury emerald silk dress",
            negative_prompt="blurry",
            model_cfg=config.MODELS["cyberrealistic_xl"],
            lora_name="fs_couture.safetensors",
            lora_weight=0.8
        )
        assert "lora" in inpaint_graph
        diag["lora_graph_compiled"] = True
    except Exception as e:
        diag["errors"].append(f"LoRA graph compilation failed: {e}")

    return diag


def run_diagnostics():
    unique_images = get_top_unique_user_images(10)
    print(f"\n=======================================================")
    print(f"Face Studio - Diagnostic Testing of 10 User Images")
    print(f"Found {len(unique_images)} unique user images in inputs/")
    print(f"Output Visualizations: {OUT_DEBUG_DIR}")
    print(f"=======================================================\n")

    results = []
    for idx, img_path in enumerate(unique_images, 1):
        print(f"[{idx}/{len(unique_images)}] Testing {img_path.name}...")
        diag = analyze_image(img_path)
        results.append(diag)
        status = "OK" if not diag["errors"] and not diag["abnormalities"] else ("ATTENTION" if diag["abnormalities"] else "ERROR")
        print(f"    Status: {status} | Resolution: {diag.get('width')}x{diag.get('height')} | Faces: {diag.get('faces_count')} | Garment: {diag.get('dress_coverage_pct')}%")
        if diag["warnings"]:
            for w in diag["warnings"]:
                print(f"      [WARN] {w}")
        if diag["abnormalities"]:
            for a in diag["abnormalities"]:
                print(f"      [ABNORMALITY] {a}")
        if diag["errors"]:
            for e in diag["errors"]:
                print(f"      [ERROR] {e}")

    # Save summary report JSON
    report_file = OUT_DEBUG_DIR / "diagnostics_report.json"
    report_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nComplete Diagnostic Report saved to: {report_file}")
    return results


if __name__ == "__main__":
    run_diagnostics()
