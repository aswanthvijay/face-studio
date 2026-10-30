"""
Curate Skin Tone Benchmark Dataset (24 Diverse Human Portraits across 8 Melanin Tiers)
Face Studio Strictly Confined Protocol (Port 7870)
"""

import os
import sys
import shutil
import json
import numpy as np
from PIL import Image

sys.path.insert(0, r"D:\face_studio")
from core.skin_engine import extract_universal_skin_profile, UNIVERSAL_MELANIN_TIERS

BENCHMARK_DIR = r"D:\face_studio\inputs\skin_tone_benchmark"
ARCHIVE_DIR = r"D:\Image AI L model\engine-input\FaceStudio"
os.makedirs(BENCHMARK_DIR, exist_ok=True)

TARGET_CONFIG = [
    # Tier 1: Very Fair (L* >= 185)
    {"id": "P01_very_fair_1", "tier": "tier_very_fair", "src": "src_0296ab9e.png", "source_dir": ARCHIVE_DIR},
    {"id": "P02_very_fair_2", "tier": "tier_very_fair", "src": "src_02b8aa15.png", "source_dir": ARCHIVE_DIR},
    {"id": "P03_very_fair_3", "tier": "tier_very_fair", "src": "src_0436b708.png", "source_dir": ARCHIVE_DIR},

    # Tier 2: Fair (170 <= L* < 185)
    {"id": "P04_fair_1", "tier": "tier_fair", "src": "src_0a11429f.png", "source_dir": ARCHIVE_DIR},
    {"id": "P05_fair_2", "tier": "tier_fair", "src": "src_0af5a377.png", "source_dir": ARCHIVE_DIR},
    {"id": "P06_fair_3", "tier": "tier_fair", "src": "src_0b4cf05e.png", "source_dir": ARCHIVE_DIR},

    # Tier 3: Light Wheatish (155 <= L* < 170)
    {"id": "P07_light_wheatish_1", "tier": "tier_light_wheatish", "src": "src_dc2f9ffa-8.png", "source_dir": r"D:\face_studio\inputs"},
    {"id": "P08_light_wheatish_2", "tier": "tier_light_wheatish", "src": "src_02949993.png", "source_dir": ARCHIVE_DIR},
    {"id": "P09_light_wheatish_3", "tier": "tier_light_wheatish", "src": "src_00415dc8.png", "source_dir": ARCHIVE_DIR},

    # Tier 4: Medium Tan (140 <= L* < 155)
    {"id": "P10_medium_tan_1", "tier": "tier_medium_tan", "src": "src_09ad1bab.png", "source_dir": ARCHIVE_DIR},
    {"id": "P11_medium_tan_2", "tier": "tier_medium_tan", "src": "src_16389e44.png", "source_dir": ARCHIVE_DIR},
    {"id": "P12_medium_tan_3", "tier": "tier_medium_tan", "src": "src_17d0d5a9.png", "source_dir": ARCHIVE_DIR},

    # Tier 5: Dusky (125 <= L* < 140)
    {"id": "P13_dusky_1", "tier": "tier_dusky", "src": "src_4bfa1540.png", "source_dir": ARCHIVE_DIR},
    {"id": "P14_dusky_2", "tier": "tier_dusky", "src": "src_30945fe9.png", "source_dir": ARCHIVE_DIR},
    {"id": "P15_dusky_3", "tier": "tier_dusky", "src": "src_60a179dd.png", "source_dir": ARCHIVE_DIR},

    # Tier 6: Deep Brown (110 <= L* < 125)
    {"id": "P16_deep_brown_1", "tier": "tier_deep_brown", "src": "src_1fbce836.png", "source_dir": ARCHIVE_DIR},
    {"id": "P17_deep_brown_2", "tier": "tier_deep_brown", "src": "src_284bdd4b.png", "source_dir": ARCHIVE_DIR},
    {"id": "P18_deep_brown_3", "tier": "tier_deep_brown", "src": "src_4ccd24a9.png", "source_dir": ARCHIVE_DIR},

    # Tier 7: Dark Espresso (90 <= L* < 110)
    {"id": "P19_dark_espresso_1", "tier": "tier_dark_espresso", "src": "src_69e42822.png", "source_dir": ARCHIVE_DIR},
    {"id": "P20_dark_espresso_2", "tier": "tier_dark_espresso", "src": "src_9a666caa.png", "source_dir": ARCHIVE_DIR},
    {"id": "P21_dark_espresso_3", "tier": "tier_dark_espresso", "src": "src_2fee696d.png", "source_dir": ARCHIVE_DIR},

    # Tier 8: Deep Ebony (L* <= 90 / Deepest South Sudanese Melanin)
    {"id": "P22_deep_ebony_1", "tier": "tier_deep_ebony", "src": "src_b4cd9f11.png", "source_dir": ARCHIVE_DIR},
    {"id": "P23_deep_ebony_2", "tier": "tier_deep_ebony", "src": "src_778e9828.png", "source_dir": ARCHIVE_DIR},
    {"id": "P24_deep_ebony_3", "tier": "tier_deep_ebony", "src": "downloaded_Anok_Yai_at_Paco_Rabanne_Spring_Summer_2020.jpg", "source_dir": BENCHMARK_DIR}
]

def main():
    print("=" * 70)
    print("CURATING 24-PORTRAIT BALANCED MULTI-MELANIN BENCHMARK DATASET")
    print("=" * 70)

    dataset_manifest = []

    for entry in TARGET_CONFIG:
        src_path = os.path.join(entry["source_dir"], entry["src"])
        if not os.path.exists(src_path):
            src_path = os.path.join(ARCHIVE_DIR, entry["src"])
        
        if not os.path.exists(src_path):
            print(f"[FAIL] Missing source file for {entry['id']}: {entry['src']}")
            continue

        im = Image.open(src_path).convert("RGB")
        prof = extract_universal_skin_profile(im)

        dest_filename = f"{entry['id']}_{entry['src']}"
        if not dest_filename.lower().endswith(".png"):
            dest_filename = f"{entry['id']}_{os.path.splitext(entry['src'])[0]}.png"
        dest_path = os.path.join(BENCHMARK_DIR, dest_filename)
        im.save(dest_path, "PNG")

        manifest_item = {
            "portrait_id": entry["id"],
            "filename": dest_filename,
            "filepath": dest_path,
            "target_tier": entry["tier"],
            "measured_luminance_l": prof["luminance_l"],
            "measured_chroma_a": prof["chroma_a"],
            "measured_chroma_b": prof["chroma_b"],
            "measured_tier_id": prof["tier_id"],
            "measured_tier_name": prof["tier_name"],
            "undertone": prof["undertone"],
            "positive_tokens": prof["positive_tokens"],
            "anti_conflict_tokens": prof["anti_conflict_tokens"],
            "image_size": list(im.size)
        }
        dataset_manifest.append(manifest_item)
        print(f"[{entry['id']:<19}] L*={prof['luminance_l']:>5.1f} | {prof['tier_id']:<19} | {prof['undertone']:<11} | {entry['src']}")

    manifest_json_path = os.path.join(BENCHMARK_DIR, "skin_tone_manifest.json")
    with open(manifest_json_path, "w") as jf:
        json.dump(dataset_manifest, jf, indent=2)

    print("\n" + "=" * 70)
    print(f"Successfully curated and registered {len(dataset_manifest)}/24 benchmark portraits!")
    print(f"Directory: {BENCHMARK_DIR}")
    print(f"Manifest:  {manifest_json_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
