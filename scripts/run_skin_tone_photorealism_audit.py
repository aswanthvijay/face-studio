"""
Comprehensive Multi-Melanin Skin Tone & Photorealism Audit Runner
Face Studio Strictly Confined Protocol (Port 7870)

Audits 24 diverse human portraits across all 8 Melanin Tiers:
- Tier 1: Very Fair (L* >= 185)
- Tier 2: Fair (170 <= L* < 185)
- Tier 3: Light Wheatish (155 <= L* < 170)
- Tier 4: Medium Tan (140 <= L* < 155)
- Tier 5: Dusky (125 <= L* < 140)
- Tier 6: Deep Brown (110 <= L* < 125)
- Tier 7: Dark Espresso (90 <= L* < 110)
- Tier 8: Deep Ebony (L* <= 90 / Deep Melanin)

Measures 7 Forensic Quality Dimensions:
1. Delta L* (Whitewashing > +8 vs Darkening < -8)
2. Perceptual Skin Delta E (Melanin fidelity)
3. Delta ITA (Individual Typology Angle shift)
4. Biometric Face ID Cosine Similarity (ArcFace)
5. Landmark Geometry Displacement (Face Originality)
6. Laplacian Texture Energy (Micro-Pores vs Waxy/Plastic)
7. Head-to-Body Skin Harmony Delta E
"""

import os
import sys
import json
import time
import math
import base64
import urllib.request
import urllib.error
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

WORKSPACE = Path(r"D:\face_studio")
sys.path.insert(0, str(WORKSPACE))

from core.face_engine import get_face_modules, detect_faces_robust, extract_face_skin_profile
from core.skin_engine import extract_universal_skin_profile

BENCHMARK_DIR = WORKSPACE / "inputs" / "skin_tone_benchmark"
OUTPUTS_DIR = WORKSPACE / "outputs" / "skin_tone_benchmark"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

SERVER_URL = "http://127.0.0.1:7870/api/change-dress"
SERVER_BASE = "http://127.0.0.1:7870"
PRIMARY_MODEL = "flux_klein"

STANDARD_PROMPT = (
    "a stunning elegant luxury designer gown, cinematic studio lighting, "
    "professional fashion editorial portrait, 85mm lens, natural skin micro-pores, "
    "delicate fine skin texture, sharp clear limbal rings, razor-sharp focus"
)

def compute_ita(l_star_cv: float, b_star_cv: float) -> float:
    """Computes Individual Typology Angle (ITA) from OpenCV LAB values."""
    # Convert OpenCV LAB to standard CIE-LAB: L* in [0..100], b* in [-128..127]
    l_std = (l_star_cv / 255.0) * 100.0
    b_std = b_star_cv - 128.0
    if abs(b_std) < 1e-4:
        return 90.0 if l_std >= 50.0 else -90.0
    rad = math.atan2(l_std - 50.0, b_std)
    return math.degrees(rad)

def extract_neck_skin_lab(bgr_img: np.ndarray, face_bbox: list) -> np.ndarray:
    """Extracts valid neck skin mean LAB below face box."""
    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    h, w = bgr_img.shape[:2]
    fw, fh = x2 - x1, y2 - y1
    ny1 = min(h - 10, y2 + 4)
    ny2 = min(h, y2 + int(fh * 0.40))
    nx1 = max(0, x1 + int(fw * 0.20))
    nx2 = min(w, x2 - int(fw * 0.20))
    if ny2 <= ny1 or nx2 <= nx1:
        return None
    neck_crop = bgr_img[ny1:ny2, nx1:nx2]
    if neck_crop.size == 0:
        return None
    ycrcb = cv2.cvtColor(neck_crop, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    gray = cv2.cvtColor(neck_crop, cv2.COLOR_BGR2GRAY)
    valid = (cr >= 130) & (cr <= 178) & (cb >= 75) & (cb <= 132) & (gray >= 40) & (gray <= 235)
    if np.count_nonzero(valid) < 50:
        return None
    lab = cv2.cvtColor(neck_crop, cv2.COLOR_BGR2LAB)
    return np.mean(lab[valid], axis=0)

def main():
    print("=" * 80)
    print("STARTING 24-PORTRAIT COMPREHENSIVE MULTI-MELANIN BENCHMARK")
    print(f"Target Server: {SERVER_URL}")
    print(f"Primary Benchmark Model: {PRIMARY_MODEL}")
    print("=" * 80)

    # 1. Load Face Modules for Forensic Likeness & Landmark Extraction
    print("[Audit] Initializing InsightFace modules...")
    app, _, _, _, _ = get_face_modules()

    manifest_path = BENCHMARK_DIR / "skin_tone_manifest.json"
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found at {manifest_path}. Run curate_skin_tone_dataset.py first.")
        return

    with open(manifest_path, "r") as f:
        portraits = json.load(f)

    print(f"[Audit] Loaded {len(portraits)} benchmark portraits from manifest.\n")

    audit_results = []
    total_start = time.time()

    for idx, p in enumerate(portraits, 1):
        pid = p["portrait_id"]
        img_path = Path(p["filepath"])
        target_tier = p["target_tier"]
        src_l = p["measured_luminance_l"]

        print(f"\n--- [{idx:02d}/24] Processing {pid} ({p['measured_tier_name']}, Src L*={src_l:.1f}) ---")

        if not img_path.exists():
            print(f"[ERROR] Image file missing: {img_path}")
            continue

        # Read Source Image
        src_pil = Image.open(img_path).convert("RGB")
        src_bgr = cv2.cvtColor(np.array(src_pil), cv2.COLOR_RGB2BGR)

        # Base64 encode for API upload
        with open(img_path, "rb") as imf:
            b64_data = base64.b64encode(imf.read()).decode("utf-8")

        payload = {
            "image": b64_data,
            "mode": "full_scene",
            "model_id": PRIMARY_MODEL,
            "prompt": STANDARD_PROMPT,
            "framing": "editorial",
            "aspect_ratio": "original",
            "enable_face_id": True,
            "face_id_weight": 0.90,
            "enable_pores": True,
            "enable_grain": True,
            "seed": 42 + idx # Deterministic for audit reproducibility
        }

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            SERVER_URL,
            data=req_data,
            headers={"Content-Type": "application/json"}
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[FAIL] Server generation failed for {pid}: {e}")
            continue

        latency = round(time.time() - t0, 2)
        output_url = resp_json.get("output_url", "")
        print(f"[Audit] Generation completed in {latency}s: {output_url}")

        # Download or read generated output image
        out_filename = Path(output_url).name
        local_output_path = WORKSPACE / "outputs" / out_filename
        if not local_output_path.exists():
            # Try fetching from server
            full_out_url = f"{SERVER_BASE}{output_url}"
            with urllib.request.urlopen(full_out_url) as r:
                with open(local_output_path, "wb") as of:
                    of.write(r.read())

        out_pil = Image.open(local_output_path).convert("RGB")
        out_bgr = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)

        # Save benchmark copy
        benchmark_out_name = f"benchmark_out_{pid}_{PRIMARY_MODEL}.png"
        benchmark_out_path = OUTPUTS_DIR / benchmark_out_name
        out_pil.save(benchmark_out_path, "PNG")

        # =================================================================
        # FORENSIC QUALITY COMPUTATION
        # =================================================================
        src_faces = detect_faces_robust(app, src_bgr)
        out_faces = detect_faces_robust(app, out_bgr)
        if src_faces:
            src_faces = sorted(src_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        if out_faces:
            out_faces = sorted(out_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)

        # 1. Measured Output Skin Profile
        out_prof = extract_universal_skin_profile(out_pil)
        out_l = out_prof["luminance_l"]
        out_a = out_prof["chroma_a"]
        out_b = out_prof["chroma_b"]
        delta_l = round(out_l - src_l, 1)

        src_a = p["measured_chroma_a"]
        src_b = p["measured_chroma_b"]
        delta_e = round(float(math.sqrt((out_l - src_l)**2 + (out_a - src_a)**2 + (out_b - src_b)**2)), 2)

        ita_src = round(compute_ita(src_l, src_b), 1)
        ita_out = round(compute_ita(out_l, out_b), 1)
        delta_ita = round(ita_out - ita_src, 1)

        # Whitewash / Darken defect detection
        whitewashed = bool(delta_l > 8.0)
        overdarkened = bool(delta_l < -8.0)

        # 2. Biometric Likeness (ArcFace)
        arcface_sim = None
        landmark_disp = None
        face_detected = len(out_faces) > 0

        if src_faces and out_faces:
            sf = sorted(src_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[0]
            of = sorted(out_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[0]
            emb_s = sf.embedding
            emb_o = of.embedding
            sim = float(np.dot(emb_s, emb_o) / (np.linalg.norm(emb_s) * np.linalg.norm(emb_o)))
            arcface_sim = round(sim, 4)

            # Landmark displacement
            if hasattr(sf, "kps") and hasattr(of, "kps"):
                sk = sf.kps
                ok = of.kps
                sx1, sy1, sx2, sy2 = [int(v) for v in sf.bbox]
                sw, sh = max(1, sx2 - sx1), max(1, sy2 - sy1)
                ox1, oy1, ox2, oy2 = [int(v) for v in of.bbox]
                ow, oh = max(1, ox2 - ox1), max(1, oy2 - oy1)
                sk_norm = (sk - [sx1, sy1]) / [sw, sh]
                ok_norm = (ok - [ox1, oy1]) / [ow, oh]
                disp = float(np.mean(np.linalg.norm(sk_norm - ok_norm, axis=1)))
                landmark_disp = round(disp, 4)

        # 3. Epidermal Texture Energy (Laplacian Variance)
        laplacian_var = None
        if out_faces:
            of = sorted(out_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[0]
            ox1, oy1, ox2, oy2 = [int(v) for v in of.bbox]
            ox1, oy1 = max(0, ox1), max(0, oy1)
            ox2, oy2 = min(out_bgr.shape[1], ox2), min(out_bgr.shape[0], oy2)
            face_crop = out_bgr[oy1:oy2, ox1:ox2]
            if face_crop.size > 0:
                face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                laplacian_var = round(float(cv2.Laplacian(face_gray, cv2.CV_64F).var()), 1)

        # 4. Head-to-Body Skin Harmony Delta E
        head_body_de = None
        if out_faces:
            of = sorted(out_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[0]
            neck_lab = extract_neck_skin_lab(out_bgr, of.bbox)
            if neck_lab is not None:
                face_lab = np.array([out_l, out_a, out_b], dtype=np.float32)
                hb_de = float(np.linalg.norm(face_lab - neck_lab))
                head_body_de = round(hb_de, 2)

        # Defect classification
        defect_flags = []
        if whitewashed: defect_flags.append("WHITEWASHED")
        if overdarkened: defect_flags.append("OVERDARKENED")
        if arcface_sim is not None and arcface_sim < 0.65: defect_flags.append("LOW_FACE_LIKENESS")
        if laplacian_var is not None and laplacian_var < 350: defect_flags.append("WAXY_PLASTIC_SKIN")
        if head_body_de is not None and head_body_de > 15.0: defect_flags.append("TWO_TONE_HEAD_BODY")

        status_str = "PASS" if not defect_flags else "DEFECT: " + ", ".join(defect_flags)
        print(f"  [Metrics] Delta L*={delta_l:+5.1f} | Delta E={delta_e:5.2f} | ArcFace={arcface_sim or 0:.4f} | LapVar={laplacian_var or 0:.1f} | HB dE={head_body_de or 0:.1f}")
        print(f"  [Verdict] {status_str}")

        result_item = {
            "portrait_id": pid,
            "target_tier": target_tier,
            "source_filename": p["filename"],
            "output_filename": benchmark_out_name,
            "source_luminance_l": src_l,
            "output_luminance_l": out_l,
            "delta_l": delta_l,
            "whitewashed": whitewashed,
            "overdarkened": overdarkened,
            "source_chroma_a": src_a,
            "output_chroma_a": out_a,
            "source_chroma_b": src_b,
            "output_chroma_b": out_b,
            "delta_e": delta_e,
            "ita_source": ita_src,
            "ita_output": ita_out,
            "delta_ita": delta_ita,
            "arcface_similarity": arcface_sim,
            "landmark_displacement": landmark_disp,
            "laplacian_variance": laplacian_var,
            "head_body_delta_e": head_body_de,
            "face_detected": face_detected,
            "defect_flags": defect_flags,
            "latency_seconds": latency
        }
        audit_results.append(result_item)

        # Incremental save after each portrait
        metrics_json_path = OUTPUTS_DIR / "skin_tone_audit_metrics.json"
        with open(metrics_json_path, "w") as jf:
            json.dump(audit_results, jf, indent=2)

    total_time = round(time.time() - total_start, 2)
    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETED: {len(audit_results)}/24 Portraits Executed in {total_time}s ({total_time/60:.1f} mins)")
    print(f"Metrics saved to: {OUTPUTS_DIR / 'skin_tone_audit_metrics.json'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
