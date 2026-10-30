"""Download helper for additional XL models with resume support."""
import os
import subprocess
import sys
from pathlib import Path

TARGET_DIR = Path(r"D:\Image AI\StabilityMatrix-win-x64\Data\Models\StableDiffusion")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        "name": "epicrealismXL_v7FinalDestination.safetensors",
        "url": "https://huggingface.co/misri/epicrealismXL_v7FinalDestination/resolve/main/epicrealismXL_v7FinalDestination.safetensors"
    },
    {
        "name": "CyberRealisticXLPlay_V7.0_FP16.safetensors",
        "url": "https://huggingface.co/cyberdelia/CyberRealisticXL/resolve/main/CyberRealisticXLPlay_V7.0_FP16.safetensors"
    }
]

def download_models():
    print("======================================================================")
    print("            Face Studio - XL Model Checkpoint Downloader")
    print("======================================================================")
    for item in MODELS:
        dest = TARGET_DIR / item["name"]
        if dest.exists() and dest.stat().st_size > 2 * 1024 * 1024 * 1024:
            print(f"[OK] Already exists ({dest.stat().st_size / (1024*1024*1024):.2f} GB): {dest.name}")
            continue

        print(f"\n[Downloading] {item['name']}...")
        print(f"Destination: {dest}")
        # Use curl with resume support and progress bar
        cmd = [
            "curl.exe",
            "-L",
            "-C", "-",
            "-o", str(dest),
            item["url"]
        ]
        ret = subprocess.run(cmd)
        if ret.returncode == 0:
            print(f"[Done] Successfully downloaded {item['name']}")
        else:
            print(f"[Warning] curl exited with code {ret.returncode} for {item['name']}")

if __name__ == "__main__":
    download_models()
