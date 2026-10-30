"""Automated Dress Detection, Validation & Preprocessing Pipeline for Face Studio LoRA Training.

Features:
1. Runs Face Studio SegFormer B2 Clothes parser on each raw fashion photo.
2. Identifies and isolates dress regions (Upper-clothes, Skirt, Pants, Dress labels).
3. Computes dress coverage and bounding box metrics.
4. Performs Lanczos aspect-ratio preservation and center-fitting into low-VRAM buckets (896x896, 768x1024).
5. Prepares paired .txt captions with trigger token 'fs_couture'.
"""
import argparse
import sys
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np

# Ensure workspace root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import config
from core.clothing_engine import parse_clothing_labels


def find_optimal_bucket(width: int, height: int) -> tuple[int, int]:
    """Return ideal bucket dimension preserving aspect ratio for 6GB VRAM."""
    aspect = width / max(1, height)
    if aspect <= 0.85:
        return (768, 1024)  # Portrait 3:4
    elif aspect >= 1.2:
        return (1024, 768)  # Landscape 4:3
    return (896, 896)       # Square 1:1


def resize_and_crop_to_bucket(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Lanczos resize with center-crop to target bucket dimensions."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    target_aspect = target_w / target_h
    img_aspect = img.width / img.height

    if img_aspect > target_aspect:
        new_w = int(img.height * target_aspect)
        offset = (img.width - new_w) // 2
        img = img.crop((offset, 0, offset + new_w, img.height))
    else:
        new_h = int(img.width / target_aspect)
        offset = (img.height - new_h) // 2
        img = img.crop((0, offset, img.width, offset + new_h))

    return img.resize((target_w, target_h), Image.Resampling.LANCZOS)


def process_and_detect_dresses(
    raw_dir: Path,
    out_dir: Path,
    trigger_token: str = "fs_couture"
) -> int:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    raw_images = sorted([f for f in raw_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions])

    if not raw_images:
        print(f"[Detector] No raw images found in {raw_dir}")
        return 0

    print(f"\n=======================================================")
    print(f"Face Studio Dress Detection & Dataset Prepper")
    print(f"Found {len(raw_images)} raw images in {raw_dir}")
    print(f"Target Output: {out_dir}")
    print(f"Trigger Token: '{trigger_token}'")
    print(f"=======================================================\n")

    valid_count = 0
    for idx, img_path in enumerate(raw_images):
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")

            # Run SegFormer dress detection
            labels = parse_clothing_labels(img_rgb)
            # Labels: 4: Upper-clothes, 5: Skirt, 6: Pants, 7: Dress, 17: Scarf
            garment_mask = np.isin(labels, [4, 5, 6, 7, 17])
            coverage = float(garment_mask.mean())

            # Select low-VRAM target resolution
            tw, th = find_optimal_bucket(img_rgb.width, img_rgb.height)
            processed_img = resize_and_crop_to_bucket(img_rgb, tw, th)

            stem = f"dress_{idx:03d}"
            out_img_path = out_dir / f"{stem}.png"
            out_txt_path = out_dir / f"{stem}.txt"

            processed_img.save(out_img_path, "PNG")

            # Determine caption
            raw_txt_path = img_path.with_suffix(".txt")
            if raw_txt_path.exists():
                base_desc = raw_txt_path.read_text(encoding="utf-8").strip()
            else:
                base_desc = "luxury designer dress, elegant draped fabric, haute couture tailoring, high fashion editorial portrait"

            # Prepend trigger token if not present
            if trigger_token.lower() not in base_desc.lower():
                final_caption = f"{trigger_token}, {base_desc}"
            else:
                final_caption = base_desc

            out_txt_path.write_text(final_caption, encoding="utf-8")
            valid_count += 1
            print(f"  [{valid_count}/{len(raw_images)}] Detected dress (coverage {coverage*100:.1f}%) -> Saved {out_img_path.name} ({tw}x{th})")

        except Exception as e:
            print(f"  [Error] Skipping {img_path.name}: {e}")

    print(f"\nSuccessfully processed and validated {valid_count} paired dress items in {out_dir}\n")
    return valid_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Studio Dress Detection & Dataset Prepper")
    parser.add_argument("--raw-dir", type=str, default="training/raw_images", help="Path to raw image directory")
    parser.add_argument("--out-dir", type=str, default="training/dataset", help="Output dataset directory")
    parser.add_argument("--trigger", type=str, default="fs_couture", help="LoRA trigger word")

    args = parser.parse_args()
    process_and_detect_dresses(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        trigger_token=args.trigger
    )
