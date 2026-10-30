"""Verification script for Full Scene Generation Fixes:
1. DEF-01 Fix Verification: SDXL realvis_xl full_body framing ratio & silhouette
2. DEF-04 Fix Verification: MOSS-Image 18-step latency & visual crispness
"""

import sys
import os
import json
import time
import urllib.request
import urllib.error
import numpy as np
import cv2
from pathlib import Path

WORKSPACE = Path(r"D:\face_studio")
sys.path.append(str(WORKSPACE))

from core.config import INPUTS_DIR, OUTPUTS_DIR
from core.face_engine import get_face_modules, detect_faces_robust

VERIFY_DIR = OUTPUTS_DIR / "verification_full_scene_fixes"
VERIFY_DIR.mkdir(parents=True, exist_ok=True)

SERVER_URL = "http://127.0.0.1:7870/api/change-dress"
SOURCE_IMAGE_ID = "0994defb"
SOURCE_PATH = INPUTS_DIR / f"src_{SOURCE_IMAGE_ID}.png"

TEST_CASES = [
    {
        "id": "verify_sdxl_full_body_framing",
        "model": "realvis_xl",
        "prompt": "a stunning high fashion model posing in Milan fashion week wearing a luxury velvet burgundy tailored blazer with gold trim, standing full body head to toe, pulled back wide camera framing, complete dress silhouette and stylish footwear visible, centered composition, relaxed natural expression, razor-sharp focus, photorealistic skin micro-pores, delicate fine lines, 35mm wide lens photography, crystal clear 8k uhd",
        "slogan": "",
        "framing": "full_body",
        "aspect_ratio": "3:4"
    },
    {
        "id": "verify_moss_image_speed_and_type",
        "model": "moss_image",
        "prompt": "a fashion model standing in a rainy Tokyo street at night illuminated by vibrant neon signs wearing an oversized black streetwear hoodie with crisp white text, authentic cloth folds, photorealistic skin texture, dramatic cinematic backlight reflections on wet asphalt, razor-sharp focus",
        "slogan": "VOGUE 2026",
        "framing": "full_body",
        "aspect_ratio": "3:4"
    }
]

def main():
    print("=" * 70)
    print("   VERIFYING FULL SCENE REMEDIATIONS ON FACE STUDIO (PORT 7870)")
    print("=" * 70)

    modules = get_face_modules()
    app = modules[0]
    src_bgr = cv2.imread(str(SOURCE_PATH))

    results = []

    for tc in TEST_CASES:
        print(f"\n[Test] Running: {tc['id']} (Model: {tc['model']})")
        payload = {
            "image_id": SOURCE_IMAGE_ID,
            "mode": "full_scene",
            "model_id": tc["model"],
            "model": tc["model"],
            "prompt": tc["prompt"],
            "negative_prompt": "blurry, deformed, cartoon, low quality, bad anatomy",
            "fidelity": 0.85,
            "enable_face_id": True,
            "framing": tc["framing"],
            "aspect_ratio": tc["aspect_ratio"],
            "slogan": tc["slogan"],
            "preserve_identity": True,
            "face_similarity_boost": True
        }

        req = urllib.request.Request(
            SERVER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - t0
        except Exception as e:
            print(f"  [ERROR] Request failed: {e}")
            continue

        if not data.get("success"):
            print(f"  [ERROR] Server reported failure: {data.get('error')}")
            continue

        rel_path = data.get("result_url") or data.get("output_url") or data.get("output_image")
        out_path = WORKSPACE / rel_path.lstrip("/").replace("/", os.sep)
        out_bgr = cv2.imread(str(out_path))

        h, w = out_bgr.shape[:2]
        faces = detect_faces_robust(app, out_bgr)
        face_ratio = None
        arc_sim = None
        if faces:
            f = faces[0]
            fh = f.bbox[3] - f.bbox[1]
            face_ratio = round(fh / h, 4)
            src_faces = detect_faces_robust(app, src_bgr)
            if src_faces:
                emb_src = src_faces[0].embedding
                emb_out = f.embedding
                arc_sim = round(float(np.dot(emb_src, emb_out) / (np.linalg.norm(emb_src) * np.linalg.norm(emb_out))), 4)

        dest_file = VERIFY_DIR / f"{tc['id']}_{tc['model']}.png"
        cv2.imwrite(str(dest_file), out_bgr)

        res = {
            "test_id": tc["id"],
            "model": tc["model"],
            "elapsed_s": round(elapsed, 2),
            "dimensions": f"{w}x{h}",
            "faces_detected": len(faces) if faces else 0,
            "face_to_height_ratio": face_ratio,
            "arcface_similarity": arc_sim,
            "output_saved": str(dest_file)
        }
        results.append(res)
        print(f"  -> Elapsed: {res['elapsed_s']}s | Dimensions: {res['dimensions']} | Face Height Ratio: {res['face_to_height_ratio']} | ArcFace: {res['arcface_similarity']}")

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY:")
    print(json.dumps(results, indent=2))
    print("=" * 70)

if __name__ == "__main__":
    main()
