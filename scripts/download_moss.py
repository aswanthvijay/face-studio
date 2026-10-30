"""Automated High-Speed Downloader for OpenMOSS MOSS-Image 5B Models and Companions.

Downloads:
  1. moss_image_color_depth_control.safetensors (Official Color/Depth Control LoRA, ~313 MB)
  2. moss_image_sft_int8_convrot.safetensors (MOSS-Image 5B SFT INT8 Single-Stream DiT, ~5.41 GB)
  3. qwen3.5_9b_int8_convrot.safetensors (Native Qwen3.5 9B Text Encoder, ~9.75 GB, optional)

Usage:
  python scripts/download_moss.py [--lora] [--dit] [--text-encoder] [--all]
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

DIFFUSION_DIR = Path(r"D:\Image AI\StabilityMatrix-win-x64\Data\Models\DiffusionModels")
LORA_DIR = Path(r"D:\Image AI\StabilityMatrix-win-x64\Data\Models\Lora")
TEXT_ENCODER_DIR = Path(r"D:\Image AI\StabilityMatrix-win-x64\Data\Models\TextEncoders")

DIFFUSION_DIR.mkdir(parents=True, exist_ok=True)
LORA_DIR.mkdir(parents=True, exist_ok=True)
TEXT_ENCODER_DIR.mkdir(parents=True, exist_ok=True)

MOSS_ASSETS = {
    "lora": {
        "name": "moss_image_color_depth_control.safetensors",
        "dir": LORA_DIR,
        "size_expected": 313063368,
        "url": "https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/loras/moss_image_color_depth_control.safetensors",
        "desc": "MOSS-Image Official Color & Depth Control LoRA (~313 MB)"
    },
    "dit": {
        "name": "moss_image_sft_int8_convrot.safetensors",
        "dir": DIFFUSION_DIR,
        "size_expected": 5413698958,
        "url": "https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/diffusion_models/moss_image_sft_int8_convrot.safetensors",
        "desc": "MOSS-Image 5B SFT INT8 Single-Stream DiT (~5.41 GB)"
    },
    "text_encoder": {
        "name": "qwen3.5_9b_int8_convrot.safetensors",
        "dir": TEXT_ENCODER_DIR,
        "size_expected": 9748121532,
        "url": "https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/text_encoders/qwen3.5_9b_int8_convrot.safetensors",
        "desc": "MOSS-Image Qwen3.5 9B INT8 Text Encoder (~9.75 GB)"
    }
}

def download_file(asset_key: str):
    asset = MOSS_ASSETS[asset_key]
    dest = asset["dir"] / asset["name"]
    expected_bytes = asset["size_expected"]

    print(f"\n[{asset_key.upper()}] Checking {asset['desc']}...")
    if dest.exists():
        actual_bytes = dest.stat().st_size
        if actual_bytes >= expected_bytes * 0.98:
            print(f"  [OK] Already downloaded: {dest.name} ({actual_bytes / (1024**3):.2f} GB)")
            return True
        else:
            print(f"  [Resuming] Partial file found ({actual_bytes / (1024**2):.1f} MB / {expected_bytes / (1024**2):.1f} MB). Resuming...")
    else:
        print(f"  [Downloading] Starting new download to {dest}...")

    cmd = [
        "curl.exe",
        "-L",
        "-C", "-",
        "--retry", "5",
        "--retry-delay", "3",
        "-o", str(dest),
        asset["url"]
    ]

    try:
        ret = subprocess.run(cmd)
        if ret.returncode == 0:
            final_bytes = dest.stat().st_size
            print(f"  [DONE] Successfully downloaded {asset['name']} ({final_bytes / (1024**3):.2f} GB)")
            return True
        else:
            print(f"  [ERROR] curl failed with exit code {ret.returncode}")
            return False
    except Exception as e:
        print(f"  [ERROR] Download exception: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download MOSS-Image 5B models and mods.")
    parser.add_argument("--lora", action="store_true", help="Download official Color & Depth Control LoRA (~313 MB)")
    parser.add_argument("--dit", action="store_true", help="Download 5B SFT INT8 DiT (~5.41 GB)")
    parser.add_argument("--text-encoder", action="store_true", help="Download Qwen3.5 9B Text Encoder (~9.75 GB)")
    parser.add_argument("--all", action="store_true", help="Download all MOSS-Image assets")
    args = parser.parse_args()

    print("======================================================================")
    print("      Face Studio - OpenMOSS MOSS-Image 5B Asset Downloader")
    print("======================================================================")

    to_download = []
    if args.all:
        to_download = ["lora", "dit", "text_encoder"]
    elif args.lora or args.dit or args.text_encoder:
        if args.lora: to_download.append("lora")
        if args.dit: to_download.append("dit")
        if args.text_encoder: to_download.append("text_encoder")
    else:
        # Default: download lora and dit
        to_download = ["lora", "dit"]

    print(f"Selected targets: {', '.join(to_download)}")
    for target in to_download:
        download_file(target)

    print("\n[MOSS-Image Downloader] Finished processing.")

if __name__ == "__main__":
    main()
