"""Deep Forensic Full Scene Quality Audit Test Harness for Face Studio.

Audits all 7 production models across 3 diverse fashion scenarios:
1. luxury_editorial (Editorial medium shot, studio lighting, velvet blazer)
2. streetwear_neon (Full body, dark neon rain, oversized hoodie + "VOGUE 2026" typography)
3. tropical_golden_hour (Full body, warm sunset lighting, flowing white maxi dress)

Measures 8 core quality dimensions:
1. Skin Tone Matching: Face-to-Face CIELAB Delta E (melanin preservation)
2. Head-Body Skin Harmony: Face-to-Neck CIELAB Delta E (two-tone prevention)
3. Body Skin Melanin Sync: Source Face-to-Body Skin CIELAB Delta E
4. Face ID / Biometric Likeness: ArcFace 512-d Cosine Similarity
5. Face Originality & Geometry: 5-Point Landmark Geometry Displacement
6. Face Naturalness & Pores: Laplacian Texture Variance (anti-plastic/waxy)
7. Body Realism & Framing: Face-to-Canvas Height Ratio (anti-bust-bias)
8. Head-Body Seam Coherence: Neck Seam Boundary Gradient Smoothness
"""

import sys
import os
import json
import time
import urllib.request
import urllib.error
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

WORKSPACE = Path(r"D:\face_studio")
sys.path.append(str(WORKSPACE))

from core.config import INPUTS_DIR, OUTPUTS_DIR
from core.face_engine import get_face_modules, detect_faces_robust, extract_face_skin_profile

AUDIT_DIR = OUTPUTS_DIR / "forensic_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

SERVER_URL = "http://127.0.0.1:7870/api/change-dress"
SOURCE_IMAGE_ID = "0994defb"
SOURCE_PATH = INPUTS_DIR / f"src_{SOURCE_IMAGE_ID}.png"

MODELS = [
    "cyberrealistic_xl",
    "epicrealism_xl",
    "realvis_xl",
    "flux_klein",
    "qwen_image_21",
    "z_image_turbo",
    "moss_image"
]

SCENARIOS = [
    {
        "id": "luxury_editorial",
        "title": "Scenario 1: High-Fashion Luxury Editorial (Studio Lighting)",
        "prompt": "a stunning high fashion model posing in Milan fashion week wearing a luxury velvet burgundy tailored blazer with gold trim, pulled back wide camera framing, complete dress silhouette and stylish footwear visible, centered composition, relaxed natural expression, razor-sharp focus, photorealistic skin micro-pores, delicate fine lines, 35mm wide lens photography, crystal clear 8k uhd",
        "slogan": "",
        "framing": "editorial",
        "aspect_ratio": "3:4"
    },
    {
        "id": "streetwear_neon",
        "title": "Scenario 2: Streetwear & Typography (Tokyo Neon Night Rain)",
        "prompt": "a fashion model standing in a rainy Tokyo street at night illuminated by vibrant neon signs wearing an oversized black streetwear hoodie with crisp white text, authentic cloth folds, photorealistic skin texture, dramatic cinematic backlight reflections on wet asphalt, razor-sharp focus",
        "slogan": "VOGUE 2026",
        "framing": "full_body",
        "aspect_ratio": "3:4"
    },
    {
        "id": "tropical_golden_hour",
        "title": "Scenario 3: Tropical Beach Golden Hour (Warm Natural Sunset)",
        "prompt": "a beautiful fashion model standing full body head to toe on a pristine tropical beach during golden hour sunset, wearing an elegant flowing white summer maxi dress, gentle sea breeze moving hair, warm golden sunlight casting soft authentic shadows, standing barefoot on wet sand reflecting amber sky, 35mm wide landscape portrait photography, photorealistic skin texture, fine pores",
        "slogan": "",
        "framing": "full_body",
        "aspect_ratio": "3:4"
    }
]


# =====================================================================
# METRIC EXTRACTION SUITE
# =====================================================================

def extract_valid_skin_lab(bgr_crop: np.ndarray) -> np.ndarray:
    """Extracts mean CIE-LAB of validated human skin pixels."""
    if bgr_crop.size == 0:
        return None
    ycrcb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    # Human skin cluster in YCrCb with luminance protection (rejects dark fabric L < 50)
    skin_mask = (cr >= 133) & (cr <= 175) & (cb >= 77) & (cb <= 130) & (gray >= 50) & (gray <= 240)
    if np.count_nonzero(skin_mask) < 50:
        return None
    lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB)
    return np.mean(lab[skin_mask], axis=0)


def compute_metrics(app, src_bgr: np.ndarray, out_bgr: np.ndarray):
    """Computes all 8 forensic quality metrics between source and output."""
    m = {
        "face_detected": False,
        "arcface_similarity": None,
        "face_skin_delta_e": None,
        "head_body_delta_e": None,
        "body_skin_delta_e": None,
        "landmark_displacement": None,
        "face_laplacian_var": None,
        "face_height_ratio": None,
        "neck_seam_smoothness": None,
        "edge_sharpness": None
    }

    try:
        h_out, w_out = out_bgr.shape[:2]
        gray_out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY)
        m["edge_sharpness"] = round(float(np.mean(np.abs(cv2.Sobel(gray_out, cv2.CV_64F, 1, 1)))), 2)

        src_faces = detect_faces_robust(app, src_bgr)
        out_faces = detect_faces_robust(app, out_bgr)

        if not out_faces:
            return m

        m["face_detected"] = True
        out_f = out_faces[0]
        ox1, oy1, ox2, oy2 = [int(v) for v in out_f.bbox]
        of_w, of_h = ox2 - ox1, oy2 - oy1
        m["face_height_ratio"] = round(float(of_h / h_out), 4)

        # 1. Face Naturalness / Texture Sharpness (Laplacian Variance)
        cx1, cy1 = max(0, ox1), max(0, oy1)
        cx2, cy2 = min(w_out, ox2), min(h_out, oy2)
        face_crop = out_bgr[cy1:cy2, cx1:cx2]
        if face_crop.size > 0:
            face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            m["face_laplacian_var"] = round(float(cv2.Laplacian(face_gray, cv2.CV_64F).var()), 2)

        # 2. Biometric Likeness (ArcFace Cosine Similarity)
        if src_faces:
            src_f = src_faces[0]
            emb_s = src_f.embedding
            emb_o = out_f.embedding
            sim = float(np.dot(emb_s, emb_o) / (np.linalg.norm(emb_s) * np.linalg.norm(emb_o)))
            m["arcface_similarity"] = round(sim, 4)

            # 3. Landmark Geometry Displacement (Face Originality)
            if hasattr(src_f, "kps") and hasattr(out_f, "kps"):
                sk = src_f.kps
                ok = out_f.kps
                # Normalize landmarks to [0..1] within each face's bounding box
                sx1, sy1, sx2, sy2 = [int(v) for v in src_f.bbox]
                sw, sh = max(1, sx2 - sx1), max(1, sy2 - sy1)
                sk_norm = (sk - [sx1, sy1]) / [sw, sh]
                ok_norm = (ok - [ox1, oy1]) / [max(1, of_w), max(1, of_h)]
                disp = float(np.mean(np.linalg.norm(sk_norm - ok_norm, axis=1)))
                m["landmark_displacement"] = round(disp, 4)

            # 4. Skin Tone Matching: Source Face vs Output Face (CIELAB Delta E)
            src_mean, _ = extract_face_skin_profile(src_bgr, src_f.bbox)
            out_mean, _ = extract_face_skin_profile(out_bgr, out_f.bbox)
            face_dE = float(np.linalg.norm(src_mean - out_mean))
            m["face_skin_delta_e"] = round(face_dE, 2)

        # 5. Head-Body Skin Harmony: Output Face vs Output Neck (CIELAB Delta E)
        ny1 = min(h_out - 10, oy2 + 4)
        ny2 = min(h_out, oy2 + int(of_h * 0.35))
        nx1 = max(0, ox1 + int(of_w * 0.20))
        nx2 = min(w_out, ox2 - int(of_w * 0.20))
        if ny2 > ny1 and nx2 > nx1:
            neck_crop = out_bgr[ny1:ny2, nx1:nx2]
            neck_lab = extract_valid_skin_lab(neck_crop)
            if neck_lab is not None and "out_mean" in locals():
                hb_dE = float(np.linalg.norm(out_mean - neck_lab))
                m["head_body_delta_e"] = round(hb_dE, 2)

        # 6. Exposed Body Skin Tone: Source Face vs Output Lower Body Skin
        by1 = min(h_out - 20, oy2 + int(of_h * 0.5))
        by2 = min(h_out, oy2 + int(of_h * 1.8))
        bx1 = max(0, ox1 - int(of_w * 0.5))
        bx2 = min(w_out, ox2 + int(of_w * 0.5))
        if by2 > by1 and bx2 > bx1:
            body_crop = out_bgr[by1:by2, bx1:bx2]
            body_lab = extract_valid_skin_lab(body_crop)
            if body_lab is not None and "src_mean" in locals():
                b_dE = float(np.linalg.norm(src_mean - body_lab))
                m["body_skin_delta_e"] = round(b_dE, 2)

        # 7. Head-Body Seam Coherence: Neck Seam Boundary Gradient Smoothness
        # Samples a 16px high horizontal slice centered on the chin/jawline boundary
        sy1 = max(0, oy2 - 8)
        sy2 = min(h_out, oy2 + 8)
        sx1 = max(0, ox1 + int(of_w * 0.15))
        sx2 = min(w_out, ox2 - int(of_w * 0.15))
        if sy2 > sy1 and sx2 > sx1:
            seam_slice = gray_out[sy1:sy2, sx1:sx2]
            sobel_y = np.abs(cv2.Sobel(seam_slice, cv2.CV_64F, 0, 1, ksize=3))
            max_gradient = float(np.max(sobel_y)) if sobel_y.size > 0 else 0.0
            # A harsh artificial pasted edge produces gradient > 120; natural blend is < 60
            smoothness_score = round(max(0.0, min(1.0, 1.0 - (max_gradient / 150.0))), 3)
            m["neck_seam_smoothness"] = smoothness_score

    except Exception as e:
        print(f"  [Metric Computation Exception] {e}")

    return m


# =====================================================================
# MAIN AUDIT RUNNER
# =====================================================================

def run_deep_audit():
    print("=" * 80)
    print("      FACE STUDIO - COMPREHENSIVE 7-MODEL FULL SCENE FORENSIC AUDIT")
    print("=" * 80)
    print(f"Server Target: {SERVER_URL}")
    print(f"Source Image:  {SOURCE_PATH.name}")

    if not SOURCE_PATH.exists():
        print(f"[FATAL] Source reference image not found at {SOURCE_PATH}")
        return

    src_bgr = cv2.imread(str(SOURCE_PATH))
    h_src, w_src = src_bgr.shape[:2]
    print(f"Source Dimensions: {w_src}x{h_src}")

    print("\nLoading InsightFace for forensic verification...")
    app, _, _, _, _ = get_face_modules()
    src_faces = detect_faces_robust(app, src_bgr)
    if not src_faces:
        print("[FATAL] Could not detect face in source image.")
        return
    print(f"[OK] Source Face Locked: Score={src_faces[0].det_score:.3f}, Age={int(src_faces[0].age)}")

    audit_results = {}
    total_runs = len(SCENARIOS) * len(MODELS)
    current_run = 0

    for sc in SCENARIOS:
        sc_id = sc["id"]
        print("\n" + "#" * 80)
        print(f"STARTING SCENARIO: {sc['title']}")
        print("#" * 80)

        audit_results[sc_id] = {
            "title": sc["title"],
            "prompt": sc["prompt"],
            "slogan": sc["slogan"],
            "framing": sc["framing"],
            "aspect_ratio": sc["aspect_ratio"],
            "models": {}
        }

        for model_id in MODELS:
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] Testing Model: {model_id} in {sc_id}...")

            payload = {
                "image_id": SOURCE_IMAGE_ID,
                "mode": "full_scene",
                "model_id": model_id,
                "model": model_id,
                "prompt": sc["prompt"],
                "slogan": sc["slogan"],
                "framing": sc["framing"],
                "aspect_ratio": sc["aspect_ratio"],
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
                with urllib.request.urlopen(req, timeout=360) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                elapsed = time.time() - t0

                if not resp_data.get("success"):
                    err_msg = resp_data.get("error", "Unknown server error")
                    print(f"  [SERVER FAILURE] {err_msg}")
                    audit_results[sc_id]["models"][model_id] = {
                        "status": "failed",
                        "error": err_msg,
                        "elapsed": round(elapsed, 2)
                    }
                    continue

                out_rel = resp_data.get("result_url") or resp_data.get("output_url")
                out_path = WORKSPACE / out_rel.lstrip("/").replace("/", os.sep)

                if not out_path.exists():
                    print(f"  [FILE ERROR] Output image not found: {out_path}")
                    audit_results[sc_id]["models"][model_id] = {
                        "status": "missing_file",
                        "elapsed": round(elapsed, 2)
                    }
                    continue

                out_bgr = cv2.imread(str(out_path))
                out_h, out_w = out_bgr.shape[:2]

                # Save copy in forensic audit directory
                save_filename = f"{sc_id}_{model_id}_{out_path.name}"
                dest_file = AUDIT_DIR / save_filename
                cv2.imwrite(str(dest_file), out_bgr)

                # Compute complete forensic metric suite
                metrics = compute_metrics(app, src_bgr, out_bgr)

                res_entry = {
                    "status": "success",
                    "elapsed": round(elapsed, 2),
                    "resolution": f"{out_w}x{out_h}",
                    "filename": save_filename,
                    "metrics": metrics
                }
                audit_results[sc_id]["models"][model_id] = res_entry

                print(f"  [SUCCESS] Elapsed: {res_entry['elapsed']}s | Dim: {res_entry['resolution']}")
                print(f"    - ArcFace Sim:        {metrics['arcface_similarity']}")
                print(f"    - Face Skin dE:       {metrics['face_skin_delta_e']}")
                print(f"    - Head-Body Skin dE:  {metrics['head_body_delta_e']}")
                print(f"    - Body Skin dE:       {metrics['body_skin_delta_e']}")
                print(f"    - Face Height Ratio:  {metrics['face_height_ratio']}")
                print(f"    - Landmark Disp:      {metrics['landmark_displacement']}")
                print(f"    - Face Laplacian:     {metrics['face_laplacian_var']}")
                print(f"    - Seam Smoothness:    {metrics['neck_seam_smoothness']}")

            except urllib.error.HTTPError as he:
                err_body = he.read().decode("utf-8")
                print(f"  [HTTP ERROR {he.code}] {err_body}")
                audit_results[sc_id]["models"][model_id] = {
                    "status": "http_error",
                    "code": he.code,
                    "error": err_body
                }
            except Exception as ex:
                print(f"  [EXCEPTION] {ex}")
                audit_results[sc_id]["models"][model_id] = {
                    "status": "exception",
                    "error": str(ex)
                }

    # Save complete JSON metrics
    metrics_path = AUDIT_DIR / "deep_forensic_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\n[OK] Saved forensic metrics JSON to: {metrics_path}")

    # Generate visual composite plates for human review
    build_composite_plates(audit_results, src_bgr)


def build_composite_plates(results, src_bgr):
    """Builds side-by-side composite review plates for each scenario."""
    print("\nGenerating Side-by-Side Review Plates...")
    for sc_id, sc_data in results.items():
        models_data = sc_data["models"]
        successful_models = [m for m in MODELS if m in models_data and models_data[m].get("status") == "success"]
        if not successful_models:
            continue

        thumb_w, thumb_h = 320, 420
        total_cols = len(successful_models) + 1
        plate_w = total_cols * thumb_w + (total_cols + 1) * 12
        plate_h = thumb_h + 90

        plate = Image.new("RGB", (plate_w, plate_h), (15, 23, 42))
        draw = ImageDraw.Draw(plate)

        # Draw Source Reference
        src_pil = Image.fromarray(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        plate.paste(src_pil, (12, 50))
        draw.text((15, 14), "SOURCE REFERENCE", fill=(255, 255, 255))
        draw.text((15, 30), "Ground Truth Identity", fill=(148, 163, 184))
        draw.text((15, thumb_h + 55), "Melanin Baseline", fill=(148, 163, 184))

        col_idx = 1
        for m in successful_models:
            m_info = models_data[m]
            img_path = AUDIT_DIR / m_info["filename"]
            if img_path.exists():
                m_img = Image.open(img_path).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                x_pos = 12 + col_idx * (thumb_w + 12)
                plate.paste(m_img, (x_pos, 50))

                # Labels
                metrics = m_info.get("metrics", {})
                sim = metrics.get("arcface_similarity", "N/A")
                el = m_info.get("elapsed", 0)
                face_dE = metrics.get("face_skin_delta_e", "N/A")
                hb_dE = metrics.get("head_body_delta_e", "N/A")
                f_ratio = metrics.get("face_height_ratio", "N/A")

                # Color coding
                sim_color = (74, 222, 128) if isinstance(sim, (int, float)) and sim >= 0.70 else (251, 146, 60)

                draw.text((x_pos + 4, 12), f"{m.upper()}", fill=(245, 158, 11))
                draw.text((x_pos + 4, 30), f"Time: {el}s | ArcFace: {sim}", fill=sim_color)
                draw.text((x_pos + 4, thumb_h + 55), f"Face dE: {face_dE} | Head-Body dE: {hb_dE}", fill=(203, 213, 225))
                draw.text((x_pos + 4, thumb_h + 70), f"Face Ratio: {f_ratio}", fill=(148, 163, 184))

                col_idx += 1

        plate_out = AUDIT_DIR / f"forensic_composite_{sc_id}.png"
        plate.save(plate_out)
        print(f"[OK] Saved Plate: {plate_out.name} ({plate.width}x{plate.height})")


if __name__ == "__main__":
    run_deep_audit()
