"""Dataset Preprocessing and Preparation Pipeline for Face Studio Custom LoRA Training.

Handles:
1. Aspect-ratio bucketing and high-quality Lanczos resizing for SDXL (1024x1024, 896x896, 832x1024).
2. Paired .txt caption generation with custom trigger tokens (e.g., 'fs_couture, luxury emerald silk gown').
3. Validation and cleaning of image datasets.
4. Starter sample dataset generator for immediate dry-run training tests.
"""
import argparse
import os
import shutil
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

# SDXL native resolution buckets
BUCKETS = [
    (1024, 1024),  # 1:1
    (832, 1024),   # 4:5
    (768, 1024),   # 3:4
    (896, 896),    # 1:1 low-vram
    (1024, 768),   # 4:3
    (1024, 832)    # 5:4
]


def find_best_bucket(width: int, height: int, low_vram: bool = True) -> tuple:
    """Find optimal target bucket preserving aspect ratio."""
    if low_vram:
        # 896x896 or 768x1024 is optimal for 6GB RTX 3060
        aspect = width / max(1, height)
        if aspect <= 0.85:
            return (768, 1024)
        elif aspect >= 1.2:
            return (1024, 768)
        return (896, 896)

    aspect = width / max(1, height)
    best_bucket = BUCKETS[0]
    min_diff = float("inf")
    for bw, bh in BUCKETS:
        b_aspect = bw / bh
        diff = abs(aspect - b_aspect)
        if diff < min_diff:
            min_diff = diff
            best_bucket = (bw, bh)
    return best_bucket


def preprocess_image(img: Image.Image, target_size: tuple) -> Image.Image:
    """Center-crop and resize image to exact bucket dimensions."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    target_w, target_h = target_size
    target_aspect = target_w / target_h
    img_aspect = img.width / img.height

    if img_aspect > target_aspect:
        # Image is wider: crop width
        new_w = int(img.height * target_aspect)
        offset = (img.width - new_w) // 2
        img = img.crop((offset, 0, offset + new_w, img.height))
    else:
        # Image is taller: crop height
        new_h = int(img.width / target_aspect)
        offset = (img.height - new_h) // 2
        img = img.crop((0, offset, img.width, offset + new_h))

    return img.resize((target_w, target_h), Image.Resampling.LANCZOS)


def prepare_dataset(
    raw_dir: Path,
    out_dir: Path,
    trigger_word: str = "fs_couture",
    default_caption: str = "luxury haute couture designer gown, intricate embroidery, rich fabric drape, editorial portrait",
    low_vram: bool = True
) -> int:
    """Processes images from raw_dir into out_dir with paired caption files."""
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".png", ".jpg", ".jpeg", ".webp"}
    raw_files = [f for f in raw_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions]

    if not raw_files:
        print(f"[Dataset] No images found in {raw_dir}")
        return 0

    print(f"[Dataset] Processing {len(raw_files)} images into {out_dir} (trigger: '{trigger_word}')...")

    count = 0
    for idx, file_path in enumerate(raw_files):
        try:
            with Image.open(file_path) as img:
                target_size = find_best_bucket(img.width, img.height, low_vram=low_vram)
                processed = preprocess_image(img, target_size)

                stem = f"fashion_{idx:03d}"
                out_img_path = out_dir / f"{stem}.png"
                out_txt_path = out_dir / f"{stem}.txt"

                processed.save(out_img_path, "PNG")

                # Check if existing caption exists in raw directory
                existing_txt = file_path.with_suffix(".txt")
                if existing_txt.exists():
                    user_caption = existing_txt.read_text(encoding="utf-8").strip()
                    if trigger_word.lower() not in user_caption.lower():
                        full_caption = f"{trigger_word}, {user_caption}"
                    else:
                        full_caption = user_caption
                else:
                    full_caption = f"{trigger_word}, {default_caption}"

                out_txt_path.write_text(full_caption, encoding="utf-8")
                count += 1
                print(f"  [{count}/{len(raw_files)}] Saved {out_img_path.name} ({target_size[0]}x{target_size[1]})")
        except Exception as e:
            print(f"  [Warning] Skipping {file_path.name}: {e}")

    print(f"[Dataset] Done. Successfully prepared {count} paired items in {out_dir}")
    return count


def create_starter_dataset(out_dir: Path, trigger_word: str = "fs_couture") -> Path:
    """Generates a clean starter dataset with diverse couture fashion styling for instant dry runs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    styles = [
        ("emerald_velvet", (10, 60, 35), "emerald green velvet gown, intricate golden thread floral embroidery, luxury high fashion styling"),
        ("royal_navy_suit", (20, 30, 60), "royal navy bespoke tailored blazer with silk satin lapels, high-fashion editorial drape"),
        ("crimson_silk", (90, 15, 25), "crimson red fluid silk gown, delicate metallic lace detailing, haute couture gala attire"),
        ("champagne_tweed", (75, 65, 50), "champagne ivory Parisian tweed jacket, pearl and gold buttons, sophisticated tailored silhouette"),
        ("noir_chiffon", (25, 25, 30), "midnight noir pleated chiffon evening gown, translucent layered fabric, avant-garde couture")
    ]

    print(f"[Starter Dataset] Creating 5 starter fashion concept images in {out_dir}...")
    for idx, (name, color, desc) in enumerate(styles):
        img = Image.new("RGB", (896, 896), color=color)
        draw = ImageDraw.Draw(img)

        # Procedural luxury fabric folds/shading for texture
        for i in range(0, 896, 32):
            shade = int(color[0] * (0.8 + 0.4 * (i % 64) / 64))
            draw.line([(0, i), (896, i + 64)], fill=(shade, color[1], color[2]), width=8)

        draw.rectangle([100, 100, 796, 796], outline=(220, 190, 120), width=6)
        draw.text((120, 120), f"Face Studio Starter Concept:\n{name.replace('_', ' ').title()}", fill=(255, 255, 255))

        stem = f"starter_{idx:03d}_{name}"
        img.save(out_dir / f"{stem}.png", "PNG")
        (out_dir / f"{stem}.txt").write_text(f"{trigger_word}, {desc}", encoding="utf-8")
        print(f"  Created starter concept: {stem}")

    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Studio LoRA Dataset Preparation")
    parser.add_argument("--input-dir", type=str, default="", help="Path to raw image folder")
    parser.add_argument("--output-dir", type=str, default="training/dataset", help="Path to prepared output dataset folder")
    parser.add_argument("--trigger", type=str, default="fs_couture", help="LoRA trigger word")
    parser.add_argument("--starter-sample", action="store_true", help="Generate 5 starter sample images for immediate dry-run testing")
    parser.add_argument("--low-vram", action="store_true", default=True, help="Target 896x896 buckets for 6GB VRAM")

    args = parser.parse_args()

    if args.starter_sample or not args.input_dir:
        create_starter_dataset(Path(args.output_dir), trigger_word=args.trigger)
    else:
        prepare_dataset(
            raw_dir=Path(args.input_dir),
            out_dir=Path(args.output_dir),
            trigger_word=args.trigger,
            low_vram=args.low_vram
        )
