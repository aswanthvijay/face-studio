"""
Verification Script: Post-Fix Melanin Calibration & Tone Harmony Test
Tests previously bleached portraits to verify remediation of Defect #1 and #2.
"""

import os
import sys
import json
import time
import base64
import urllib.request
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

WORKSPACE = Path(r"D:\face_studio")
sys.path.insert(0, str(WORKSPACE))

from core.face_engine import get_face_modules, detect_faces_robust, extract_face_skin_profile
from core.skin_engine import extract_universal_skin_profile

BENCHMARK_DIR = WORKSPACE / "inputs" / "skin_tone_benchmark"
OUTPUTS_DIR = WORKSPACE / "outputs" / "skin_tone_benchmark"
SERVER_URL = "http://127.0.0.1:7870/api/change-dress"

# Test 3 representative portraits spanning Dusky, Deep Brown, and Deep Ebony
TEST_CASES = [
    {"id": "P14_dusky_2", "expected_l": 132.0, "prior_delta_l": 59.0},
    {"id": "P18_deep_brown_3", "expected_l": 122.0, "prior_delta_l": 24.0},
    {"id": "P22_deep_ebony_1", "expected_l": 88.0, "prior_delta_l": 26.0}
]

def main():
    print("=" * 70)
    print("VERIFYING POST-FIX SKIN TONE CALIBRATION & MELANIN PRESERVATION")
    print("=" * 70)

    app, _, _, _, _ = get_face_modules()

    with open(BENCHMARK_DIR / "skin_tone_manifest.json") as f:
        manifest = {m["portrait_id"]: m for m in json.load(f)}

    results = []
    for tc in TEST_CASES:
        pid = tc["id"]
        m = manifest[pid]
        src_path = Path(m["filepath"])
        src_pil = Image.open(src_path).convert("RGB")
        src_bgr = cv2.cvtColor(np.array(src_pil), cv2.COLOR_BGR2RGB)

        with open(src_path, "rb") as imf:
            b64_data = base64.b64encode(imf.read()).decode("utf-8")

        payload = {
            "image": b64_data,
            "mode": "full_scene",
            "model_id": "flux_klein",
            "prompt": "a stunning elegant luxury designer gown, cinematic studio lighting, professional fashion editorial portrait, 85mm lens, natural skin micro-pores",
            "framing": "editorial",
            "enable_face_id": True,
            "face_id_weight": 0.90,
            "enable_pores": True,
            "enable_grain": True,
            "seed": 999
        }

        req = urllib.request.Request(
            SERVER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        print(f"\n[Testing {pid}] Source L*={m['measured_luminance_l']} (Prior ΔL* was +{tc['prior_delta_l']})...")
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        elapsed = round(time.time() - t0, 2)

        out_url = data["output_url"]
        out_path = WORKSPACE / "outputs" / Path(out_url).name
        out_pil = Image.open(out_path).convert("RGB")
        out_bgr = cv2.cvtColor(np.array(out_pil), cv2.COLOR_BGR2RGB)

        # Profile output
        out_prof = extract_universal_skin_profile(out_pil)
        new_l = out_prof["luminance_l"]
        new_dl = round(new_l - m["measured_luminance_l"], 1)

        # ArcFace
        sf = detect_faces_robust(app, src_bgr)
        of = detect_faces_robust(app, out_bgr)
        sim = 0.0
        if sf and of:
            sim = round(float(np.dot(sf[0].embedding, of[0].embedding) / (np.linalg.norm(sf[0].embedding) * np.linalg.norm(of[0].embedding))), 4)

        # Save verification image
        verify_name = f"verified_fix_{pid}.png"
        out_pil.save(OUTPUTS_DIR / verify_name)

        print(f"  -> Generated in {elapsed}s | Out L*={new_l:.1f} | New ΔL*={new_dl:+5.1f} (Was +{tc['prior_delta_l']}) | ArcFace={sim:.4f}")
        results.append({
            "portrait_id": pid,
            "src_l": m["measured_luminance_l"],
            "new_l": new_l,
            "prior_delta_l": tc["prior_delta_l"],
            "new_delta_l": new_dl,
            "arcface": sim,
            "output_file": verify_name
        })

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY:")
    for r in results:
        print(f"  {r['portrait_id']:<19}: Prior ΔL*=+{r['prior_delta_l']:<4} -> New ΔL*={r['new_delta_l']:+5.1f} | ArcFace={r['arcface']:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
