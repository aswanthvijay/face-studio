"""Download 30 curated high-resolution dress photos for Face Studio LoRA training."""
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "training" / "raw_images"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 30 curated Unsplash fashion dress photo IDs
DRESS_ITEMS = [
    ("1515372039744-b8f02a3ae446", "crimson_silk_evening_gown"),
    ("1539109136881-3be0616acf4b", "beige_couture_draped_dress"),
    ("1566174053879-31528523f8ae", "evening_gown_studio_portrait"),
    ("1509631179647-0177331693ae", "high_fashion_couture_gown"),
    ("1496747611176-843222e1e57c", "white_lace_boho_dress"),
    ("1518895949257-7621c3c786d7", "chic_floral_print_sundress"),
    ("1572804013309-59a88b7e92f1", "classic_black_cocktail_dress"),
    ("1485230895905-ec40ba36b9bc", "tailored_blazer_dress_silhouette"),
    ("1529139574466-a303027c1d8b", "editorial_fashion_portrait_dress"),
    ("1550614000-4895a10e1bfd", "yellow_summer_chiffon_dress"),
    ("1551803091-e20673f15770", "green_silk_pleated_maxi_dress"),
    ("1515886657613-9f3515b0c78f", "runway_street_couture_dress"),
    ("1574015974293-817f0ebebb74", "rose_pink_satin_evening_gown"),
    ("1544441893-675973e31985", "blue_patterned_couture_dress"),
    ("1585487000160-6ebcfceb0d03", "midnight_navy_formal_dress"),
    ("1502716119720-b23a93e5fe1b", "white_bridal_couture_gown"),
    ("1520591799415-82a3a5ec7bbd", "red_carpet_draped_dress"),
    ("1583496661160-fb5886a0aaaa", "black_velvet_cocktail_dress"),
    ("1594938298603-c8148c4dae35", "emerald_satin_slit_gown"),
    ("1607083206869-4c7672e72a8a", "black_lace_formal_evening_gown"),
    ("1612423284934-2850a4ea6b0f", "minimalist_tailored_linen_dress"),
    ("1617059063772-34532796cdb5", "vibrant_orange_silk_slip_dress"),
    ("1618244972963-dbee1a7edc95", "sparkling_metallic_gala_gown"),
    ("1621786032758-111a43a8542c", "emerald_green_velvet_gown"),
    ("1512436991641-6745cdb1723f", "ivory_ruffled_couture_dress"),
    ("1525507119028-ed4c629a60a3", "off_shoulder_romantic_gown"),
    ("1534528741775-53994a69daeb", "burgundy_velvet_portrait_dress"),
    ("1554412933-514a83d2f3c8", "lilac_pastel_evening_gown"),
    ("1568252542512-9fe8fe9c87bb", "modern_cutout_party_dress"),
    ("1576995853123-5a10305d93c0", "gold_shimmering_festive_gown")
]


def download_dataset():
    print(f"\n=======================================================")
    print(f"Downloading 30 Curated Dress Images for LoRA Training")
    print(f"Destination: {RAW_DIR}")
    print(f"=======================================================\n")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    success_count = 0
    for idx, (photo_id, tag_desc) in enumerate(DRESS_ITEMS):
        filename = f"dress_{idx:02d}_{tag_desc}.jpg"
        target_path = RAW_DIR / filename
        txt_path = RAW_DIR / f"dress_{idx:02d}_{tag_desc}.txt"

        if target_path.exists() and target_path.stat().st_size > 50000:
            print(f"[{idx+1}/30] Already downloaded: {filename}")
            success_count += 1
            continue

        url = f"https://images.unsplash.com/photo-{photo_id}?auto=format&fit=crop&w=1024&q=85"
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=15) as resp:
                data = resp.read()

            with open(target_path, "wb") as f:
                f.write(data)

            # Validate PIL image
            with Image.open(target_path) as img:
                w, h = img.size

            # Create paired descriptive tag file
            caption = f"{tag_desc.replace('_', ' ')}, elegant fabric texture, haute couture tailoring, luxury dress design, editorial studio lighting"
            txt_path.write_text(caption, encoding="utf-8")

            print(f"[{idx+1}/30] Saved {filename} ({w}x{h}, {len(data)//1024} KB)")
            success_count += 1
            time.sleep(0.3)
        except Exception as e:
            print(f"[{idx+1}/30] Failed downloading {photo_id}: {e}")

    print(f"\nDataset download complete: {success_count}/{len(DRESS_ITEMS)} images verified in {RAW_DIR}\n")
    return success_count


if __name__ == "__main__":
    download_dataset()
