"""Targeted Verification Script for Forensic Quality Fixes:
1. DEF-01: CyberRealistic XL with Black Hoodie (Head-Body Delta E high-collar immunity)
2. DEF-02: RealVisXL with Sunset Tropical Beach (Adaptive fill lighting & ArcFace likeness)
3. DEF-05: Flux-Klein with Editorial Framing (Medium fashion shot framing check)
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
from core.face_engine import get_face_modules, detect_faces_robust, extract_face_skin_profile

VERIFY_DIR = OUTPUTS_DIR / "verification_remediations"
VERIFY_DIR.mkdir(parents=True, exist_ok=True)

SERVER_URL = "http://127.0.0.1:7870/api/change-dress"
SOURCE_IMAGE_ID = "0994defb"
SOURCE_PATH = INPUTS_DIR / f"src_{SOURCE_IMAGE_ID}.png"

TESTS = [
    {
        "id": "verify_def01_collar_immunity",
        "model": "cyberrealistic_xl",
        "prompt": "a fashion model standing in a rainy Tokyo street at night illuminated by vibrant neon signs wearing an oversized black streetwear hoodie with crisp white text, authentic cloth folds, photorealistic skin texture, dramatic cinematic backlight reflections on wet asphalt, razor-sharp focus",
        "slogan": "VOGUE 2026",
        "framing": "full_body",
        "aspect_ratio": "3:4"
    },
    {
        "id": "verify_def02_sunset_fill_light",
        "model": "realvis_xl",
        "prompt": "a beautiful fashion model standing full body head to toe on a pristine tropical beach during golden hour sunset, wearing an elegant flowing white summer maxi dress, gentle sea breeze moving hair, warm golden sunlight casting soft authentic shadows, standing barefoot on wet sand reflecting amber sky, 35mm wide landscape portrait photography, photorealistic skin texture, fine pores",
        "slogan": "",
        "framing": "full_body",
        "aspect_ratio": "3:4"
    },
    {
        "id": "verify_def05_dit_editorial_medium",
        "model": "flux_klein",
        "prompt": "a stunning high fashion model posing in Milan fashion week wearing a luxury velvet burgundy tailored blazer with gold trim, centered composition, relaxed natural expression, razor-sharp focus, photorealistic skin micro-pores, delicate fine lines, 50mm lens photography, crystal clear 8k uhd",
        "slogan": "",
        "framing": "editorial",
        "aspect_ratio": "3:4"
    }
]

def main():
    print("=" * 75)
    print("   EMPIRICAL VERIFICATION OF REMEDIATIONS (DEF-01, DEF-02, DEF-05)")
    print("=" * 75)

    modules = get_face_modules()
    app = modules[0]
    src_bgr = cv2.imread(str(SOURCE_PATH))

    results = []

    for tc in TESTS:
        print(f"\n[Running Test] {tc['id']} (Model: {tc['model']})")
        payload = {
            "image_id": SOURCE_IMAGE_ID,
            "mode": "full_scene",
            "model_id": tc["model"],
            "model": tc["model"],
            "prompt": tc["prompt"],
            "slogan": tc["slogan"],
            "framing": tc["framing"],
            "aspect_ratio": tc["aspect_ratio"],
            "enable_face_id": True,
            "face_id_weight": 0.65,
            "fidelity": 0.85,
            "enable_pores": True,
            "enable_grain": True,
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

        rel_path = data.get("result_url") or data.get("output_url")
        out_path = WORKSPACE / rel_path.lstrip("/").replace("/", os.sep)
        out_bgr = cv2.imread(str(out_path))

        h, w = out_bgr.shape[:2]
        src_faces = detect_faces_robust(app, src_bgr)
        out_faces = detect_faces_robust(app, out_bgr)

        face_ratio = None
        arc_sim = None
        face_dE = None
        hb_dE = None

        if out_faces and src_faces:
            of = out_faces[0]
            sf = src_faces[0]
            fh = of.bbox[3] - of.bbox[1]
            fw = of.bbox[2] - of.bbox[0]
            face_ratio = round(float(fh / h), 4)

            emb_src = sf.embedding
            emb_out = of.embedding
            arc_sim = round(float(np.dot(emb_src, emb_out) / (np.linalg.norm(emb_src) * np.linalg.norm(emb_out))), 4)

            src_mean, _ = extract_face_skin_profile(src_bgr, sf.bbox)
            out_mean, _ = extract_face_skin_profile(out_bgr, of.bbox)
            face_dE = round(float(np.linalg.norm(src_mean - out_mean)), 2)

            # Neck sample
            ny1 = min(h - 10, int(of.bbox[3]) + 4)
            ny2 = min(h, int(of.bbox[3]) + int(fh * 0.35))
            nx1 = max(0, int(of.bbox[0]) + int(fw * 0.20))
            nx2 = min(w, int(of.bbox[2]) - int(fw * 0.20))
            if ny2 > ny1 and nx2 > nx1:
                neck_crop = out_bgr[ny1:ny2, nx1:nx2]
                ycrcb_n = cv2.cvtColor(neck_crop, cv2.COLOR_BGR2YCrCb)
                gray_n = cv2.cvtColor(neck_crop, cv2.COLOR_BGR2GRAY)
                skin_n = (ycrcb_n[:,:,1] >= 133) & (ycrcb_n[:,:,1] <= 175) & (ycrcb_n[:,:,2] >= 77) & (ycrcb_n[:,:,2] <= 130) & (gray_n >= 50)
                if np.count_nonzero(skin_n) >= 50:
                    lab_n = cv2.cvtColor(neck_crop, cv2.COLOR_BGR2LAB)
                    neck_mean = np.mean(lab_n[skin_n], axis=0)
                    hb_dE = round(float(np.linalg.norm(out_mean - neck_mean)), 2)

        dest_file = VERIFY_DIR / f"{tc['id']}_{tc['model']}.png"
        cv2.imwrite(str(dest_file), out_bgr)

        res = {
            "test_id": tc["id"],
            "model": tc["model"],
            "elapsed_s": round(elapsed, 2),
            "dimensions": f"{w}x{h}",
            "arcface_similarity": arc_sim,
            "face_skin_delta_e": face_dE,
            "head_body_delta_e": hb_dE,
            "face_height_ratio": face_ratio,
            "output_saved": str(dest_file)
        }
        results.append(res)
        print(f"  -> Elapsed: {res['elapsed_s']}s | ArcFace: {res['arcface_similarity']} | Face dE: {res['face_skin_delta_e']} | HB dE: {res['head_body_delta_e']} | Face Ratio: {res['face_height_ratio']}")

    print("\n" + "=" * 75)
    print("REMEDIATION VERIFICATION RESULTS:")
    for r in results:
        print(f"- {r['test_id']} ({r['model']}): ArcFace={r['arcface_similarity']}, Face dE={r['face_skin_delta_e']}, HB dE={r['head_body_delta_e']}, Face Ratio={r['face_height_ratio']}")
    print("=" * 75)

if __name__ == "__main__":
    main()
