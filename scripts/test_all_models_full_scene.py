"""Comprehensive Full Scene Benchmark Test Harness for Face Studio.

Systematically tests all 7 production models in Full Scene Generation mode:
1. cyberrealistic_xl
2. epicrealism_xl
3. realvis_xl
4. flux_klein
5. qwen_image_21
6. z_image_turbo
7. moss_image

Measures:
- Latency (seconds)
- Biometric likeness (ArcFace Cosine Similarity)
- Skin melanin delta (LAB Delta E)
- Face detection confidence
- Output dimensions and visual integrity
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
from core.face_engine import get_face_modules, detect_faces_robust

BENCHMARK_DIR = OUTPUTS_DIR / "full_scene_benchmark"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

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
        "title": "Scenario 1: High-Fashion Luxury Editorial (Velvet Blazer)",
        "prompt": "a stunning high fashion model posing in Milan fashion week wearing a luxury velvet burgundy tailored blazer with gold trim, pulled back wide camera framing, complete dress silhouette and stylish footwear visible, centered composition, relaxed natural expression, razor-sharp focus, photorealistic skin micro-pores, delicate fine lines, 35mm wide lens photography, crystal clear 8k uhd",
        "slogan": "",
        "framing": "editorial",
        "aspect_ratio": "3:4"
    },
    {
        "id": "streetwear_typography",
        "title": "Scenario 2: Streetwear & Typography (VOGUE 2026)",
        "prompt": "a fashion model standing in a rainy Tokyo street at night illuminated by vibrant neon signs wearing an oversized black streetwear hoodie with crisp white text, authentic cloth folds, photorealistic skin texture, dramatic cinematic backlight reflections on wet asphalt, razor-sharp focus",
        "slogan": "VOGUE 2026",
        "framing": "full_body",
        "aspect_ratio": "3:4"
    }
]

def compute_arcface_similarity(app, src_bgr, out_bgr):
    try:
        src_faces = detect_faces_robust(app, src_bgr)
        out_faces = detect_faces_robust(app, out_bgr)
        if not src_faces or not out_faces:
            return None, (len(src_faces) > 0, len(out_faces) > 0)
        emb_src = src_faces[0].embedding
        emb_out = out_faces[0].embedding
        sim = float(np.dot(emb_src, emb_out) / (np.linalg.norm(emb_src) * np.linalg.norm(emb_out)))
        return round(sim, 4), (True, True)
    except Exception as e:
        print(f"  [Metric Error] ArcFace comparison: {e}")
        return None, (False, False)

def compute_skin_delta_e(app, out_bgr):
    """Computes Delta E between face and upper body/neck in LAB space."""
    try:
        faces = detect_faces_robust(app, out_bgr)
        if not faces:
            return None
        f = faces[0]
        fx1, fy1, fx2, fy2 = [int(v) for v in f.bbox]
        # Sample center of face
        cx, cy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
        r = max(4, (fx2 - fx1) // 8)
        face_roi = out_bgr[cy-r:cy+r, cx-r:cx+r]
        if face_roi.size == 0:
            return None
        face_lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        face_mean = np.mean(face_lab, axis=(0, 1))

        # Sample below chin (neck region)
        ny1 = min(out_bgr.shape[0] - 10, fy2 + 10)
        ny2 = min(out_bgr.shape[0], fy2 + 40)
        neck_roi = out_bgr[ny1:ny2, cx-r:cx+r]
        if neck_roi.size == 0:
            return None
        neck_lab = cv2.cvtColor(neck_roi, cv2.COLOR_BGR2LAB)
        neck_mean = np.mean(neck_lab, axis=(0, 1))

        delta_e = float(np.linalg.norm(face_mean - neck_mean))
        return round(delta_e, 2)
    except Exception as e:
        print(f"  [Metric Error] Skin Delta E: {e}")
        return None

def run_test():
    print("=" * 75)
    print("      FACE STUDIO - ALL 7 MODELS FULL SCENE BENCHMARK HARNESS")
    print("=" * 75)

    if not SOURCE_PATH.exists():
        print(f"Error: Source image {SOURCE_PATH} not found.")
        return

    src_bgr = cv2.imread(str(SOURCE_PATH))
    print(f"Source Reference: {SOURCE_PATH.name} ({src_bgr.shape[1]}x{src_bgr.shape[0]})")

    print("\nInitializing InsightFace for metric verification...")
    app, _, _, _, _ = get_face_modules()

    src_faces = detect_faces_robust(app, src_bgr)
    if not src_faces:
        print("Error: Could not detect face in source image.")
        return
    print(f"Source Face Detected: Conf={src_faces[0].det_score:.3f}, Age={int(src_faces[0].age)}")

    all_results = {}

    for scenario in SCENARIOS:
        sc_id = scenario["id"]
        print("\n" + "#" * 75)
        print(f"RUNNING: {scenario['title']}")
        print("#" * 75)
        all_results[sc_id] = {
            "title": scenario["title"],
            "prompt": scenario["prompt"],
            "slogan": scenario["slogan"],
            "models": {}
        }

        for model_id in MODELS:
            print(f"\n---> Benchmarking Model: {model_id} in {sc_id}...")
            payload = {
                "image_id": SOURCE_IMAGE_ID,
                "mode": "full_scene",
                "model_id": model_id,
                "prompt": scenario["prompt"],
                "slogan": scenario["slogan"],
                "framing": scenario["framing"],
                "aspect_ratio": scenario["aspect_ratio"],
                "enable_face_id": True,
                "face_id_weight": 0.65,
                "seed": 42
            }

            if model_id == "moss_image":
                payload["lora_id"] = "moss_image_color_depth_control"
                payload["lora_weight"] = 0.70

            t0 = time.time()
            req = urllib.request.Request(
                SERVER_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )

            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    elapsed = round(time.time() - t0, 2)
                    output_url = res.get("output_url", "")
                    filename = Path(output_url).name
                    local_out_path = OUTPUTS_DIR / filename
                    dest_bench_path = BENCHMARK_DIR / f"{sc_id}_{model_id}_{filename}"

                    if local_out_path.exists():
                        # Copy to benchmark dir
                        import shutil
                        shutil.copy2(str(local_out_path), str(dest_bench_path))

                        out_bgr = cv2.imread(str(local_out_path))
                        h, w = out_bgr.shape[:2]

                        # Compute ArcFace Similarity
                        sim, (src_found, out_found) = compute_arcface_similarity(app, src_bgr, out_bgr)
                        delta_e = compute_skin_delta_e(app, out_bgr)

                        model_metrics = {
                            "status": "success",
                            "elapsed": elapsed,
                            "pipeline_elapsed": res.get("elapsed", 0),
                            "resolution": f"{w}x{h}",
                            "filename": dest_bench_path.name,
                            "face_swapped": res.get("face_swapped", False),
                            "arcface_similarity": sim,
                            "skin_delta_e": delta_e,
                            "face_detected": out_found
                        }
                        all_results[sc_id]["models"][model_id] = model_metrics
                        print(f"  [SUCCESS] {model_id}: {elapsed:.1f}s | Res: {w}x{h} | ArcFace: {sim} | Delta_E: {delta_e}")
                    else:
                        print(f"  [FAIL] Output file {local_out_path} missing.")
                        all_results[sc_id]["models"][model_id] = {
                            "status": "missing_file",
                            "elapsed": elapsed,
                            "error": "Generated image file not found"
                        }
            except urllib.error.HTTPError as he:
                err_body = he.read().decode("utf-8")
                print(f"  [HTTP ERROR {he.code}] {err_body}")
                all_results[sc_id]["models"][model_id] = {
                    "status": "http_error",
                    "code": he.code,
                    "error": err_body
                }
            except Exception as e:
                print(f"  [EXCEPTION] {e}")
                all_results[sc_id]["models"][model_id] = {
                    "status": "exception",
                    "error": str(e)
                }

    # Save metrics JSON
    metrics_path = BENCHMARK_DIR / "benchmark_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Done] Saved full benchmark metrics to {metrics_path}")

    # Build Side-by-Side Comparison Composite Plates
    build_composite_plates(all_results, src_bgr)

def build_composite_plates(results, src_bgr):
    print("\nGenerating Side-by-Side Composite Plates...")
    for sc_id, sc_data in results.items():
        models_data = sc_data["models"]
        successful_models = [m for m in MODELS if m in models_data and models_data[m].get("status") == "success"]
        if not successful_models:
            continue

        thumb_w, thumb_h = 320, 420
        # 1 slot for source + N models
        total_cols = len(successful_models) + 1
        plate_w = total_cols * thumb_w + (total_cols + 1) * 10
        plate_h = thumb_h + 80

        plate = Image.new("RGB", (plate_w, plate_h), (18, 24, 38))
        draw = ImageDraw.Draw(plate)

        # Draw Source Reference
        src_pil = Image.fromarray(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        plate.paste(src_pil, (10, 50))
        draw.text((15, 20), "SOURCE REFERENCE", fill=(255, 255, 255))
        draw.text((15, thumb_h + 55), "100% Identity Ground Truth", fill=(148, 163, 184))

        col_idx = 1
        for m in successful_models:
            m_info = models_data[m]
            img_path = BENCHMARK_DIR / m_info["filename"]
            if img_path.exists():
                m_img = Image.open(img_path).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                x_pos = 10 + col_idx * (thumb_w + 10)
                plate.paste(m_img, (x_pos, 50))

                # Labels
                sim = m_info.get("arcface_similarity", "N/A")
                el = m_info.get("elapsed", 0)
                draw.text((x_pos + 5, 12), f"{m.upper()}", fill=(245, 158, 11))
                draw.text((x_pos + 5, 30), f"Time: {el}s | Sim: {sim}", fill=(203, 213, 225))
                delta_e = m_info.get("skin_delta_e", "N/A")
                draw.text((x_pos + 5, thumb_h + 55), f"Skin dE: {delta_e}", fill=(148, 163, 184))
                col_idx += 1

        plate_out = BENCHMARK_DIR / f"composite_plate_{sc_id}.png"
        plate.save(plate_out)
        print(f"Saved composite plate: {plate_out.name} ({plate.width}x{plate.height})")

if __name__ == "__main__":
    run_test()
