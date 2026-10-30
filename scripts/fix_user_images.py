"""Batch Normalizer & Fixer for User Images in Face Studio.

Fixes:
1. EXIF orientation issues from mobile camera uploads.
2. Odd / non-divisible dimensions (snaps to multiples of 16).
3. Low-resolution images (upscales images < 768px using high-fidelity Lanczos).
4. Re-generates clean clothing segmentation mask previews.
"""
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import config
from core.clothing_engine import normalize_input_image, generate_dress_mask
from scripts.test_user_images import get_top_unique_user_images


def fix_user_images():
    all_images = sorted(config.INPUTS_DIR.glob("src_*.png"))
    print("\n=======================================================")
    print(f"Fixing & Normalizing ALL {len(all_images)} User Images in inputs/")
    print("=======================================================\n")

    fixed_count = 0
    for idx, img_path in enumerate(all_images, 1):
        with Image.open(img_path) as raw_img:
            orig_w, orig_h = raw_img.size

        # Run normalizer
        with Image.open(img_path) as raw_img:
            normalized_img = normalize_input_image(raw_img, min_dim=768, max_dim=1920, stride=16)

        new_w, new_h = normalized_img.size
        changed = (orig_w, orig_h) != (new_w, new_h)

        if changed:
            # Save normalized image
            normalized_img.save(img_path, "PNG")
            print(f"[{idx}/10] FIXED {img_path.name}: {orig_w}x{orig_h} -> {new_w}x{new_h}")
            fixed_count += 1
        else:
            print(f"[{idx}/10] ALREADY OPTIMAL {img_path.name}: {orig_w}x{orig_h}")

        # Re-generate preview overlay if image was changed or preview is missing
        stem = img_path.stem.replace("src_", "")
        preview_path = config.INPUTS_DIR / f"preview_{stem}.png"
        if changed or not preview_path.exists():
            mask_pil, preview_overlay, faces, coverage = generate_dress_mask(
                normalized_img, target="dress", keep_face=True
            )
            preview_overlay.save(preview_path, "PNG")

    print(f"\nCompleted normalization. {fixed_count} images resized/upscaled and aligned to stride 16.\n")


if __name__ == "__main__":
    fix_user_images()
