"""Face Studio & Photovance AI - Web & Mobile REST API Server."""
import base64
import email
import io
import json
import mimetypes
import os
import sys
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import config
from core.clothing_engine import generate_dress_mask
from core.pipeline import run_face_studio_pipeline, ensure_backend
from core.automatic_outfit import OutfitRejected, rejection_response
from urllib.parse import urlsplit, parse_qs

# Session history (bounded in-memory thread-safe FIFO buffer)
HISTORY = deque(maxlen=50)
HISTORY_LOCK = threading.Lock()

# Concurrency semaphore for heavy GPU inference jobs (prevents CUDA OOM on single GPU)
GPU_SEMAPHORE = threading.Semaphore(1)
OUTFIT_PROGRESS = {}
PROGRESS_LOCK = threading.Lock()
BACKEND_STATUS_LOCK = threading.Lock()
BACKEND_CONNECTED = False
BACKEND_STATUS_POLL_SECONDS = 3


def start_backend_status_monitor():
    """Refresh backend health off the request path so status reads stay fast."""
    def monitor_loop():
        global BACKEND_CONNECTED
        while True:
            try:
                response = requests.get(f"{config.BACKEND_URL}/system_stats", timeout=1.0)
                connected = response.status_code == 200
            except requests.RequestException:
                connected = False
            with BACKEND_STATUS_LOCK:
                BACKEND_CONNECTED = connected
            time.sleep(BACKEND_STATUS_POLL_SECONDS)

    threading.Thread(target=monitor_loop, daemon=True).start()


def set_outfit_progress(request_id, message):
    if not request_id:
        return
    with PROGRESS_LOCK:
        now = time.monotonic()
        for key in list(OUTFIT_PROGRESS):
            if now - OUTFIT_PROGRESS[key]['updated'] > 1800:
                del OUTFIT_PROGRESS[key]
        if request_id not in OUTFIT_PROGRESS and len(OUTFIT_PROGRESS) >= 256:
            del OUTFIT_PROGRESS[next(iter(OUTFIT_PROGRESS))]
        OUTFIT_PROGRESS[request_id] = {'stage': str(message), 'updated': now}


def safe_path(base_dir: Path, rel_path: str) -> Path | None:
    """Validate path and prevent directory traversal (e.g. ../../)."""
    try:
        clean = rel_path.lstrip("/\\")
        target = (base_dir / clean).resolve()
        if target.is_relative_to(base_dir.resolve()) and target.is_file():
            return target
    except Exception:
        pass
    return None


def extract_request_image(data: dict) -> tuple[Image.Image, str]:
    """Helper to extract PIL Image and URL from request payload (image_id or base64 data)."""
    raw_b64 = data.get("data") or data.get("image")
    image_id = data.get("image_id")
    if raw_b64:
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(raw_b64)))).convert("RGB")
        img_id = str(uuid.uuid4())[:10]
        src_path = config.INPUTS_DIR / f"src_{img_id}.png"
        img.save(src_path, "PNG")
        return img, f"/inputs/{src_path.name}"
    elif image_id:
        src_path = config.INPUTS_DIR / f"src_{image_id}.png"
        if not src_path.exists():
            if image_id == "sample_model" and (config.SAMPLES_DIR / "sample_model.png").exists():
                src_path = config.SAMPLES_DIR / "sample_model.png"
            else:
                raise ValueError("Image ID not found. Please upload again.")
        img = ImageOps.exif_transpose(Image.open(src_path)).convert("RGB")
        return img, f"/inputs/{src_path.name}" if "inputs" in str(src_path) else "/samples/sample_model.png"
    else:
        sample_path = config.SAMPLES_DIR / "sample_model.png"
        if sample_path.exists():
            return ImageOps.exif_transpose(Image.open(sample_path)).convert("RGB"), "/samples/sample_model.png"
        raise ValueError("No input image specified.")


def decode_uploaded_image(image_data: bytes) -> Image.Image:
        """Decode supported raster uploads and apply EXIF orientation before processing."""
        try:
            with Image.open(io.BytesIO(image_data)) as decoded:
                if decoded.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError(
                        f"Unsupported image format: {decoded.format or 'unknown'}. "
                        "Use a JPEG, PNG, or WebP image; camera RAW files must be exported first."
                    )
                decoded.load()
                return ImageOps.exif_transpose(decoded).convert("RGB")
        except UnidentifiedImageError as error:
            raise ValueError(
                "The file is not a readable JPEG, PNG, or WebP image. "
                "Camera RAW files must be exported to one of these formats first."
            ) from error
        except OSError as error:
            raise ValueError(
                "The image could not be decoded. Use a valid JPEG, PNG, or WebP file."
            ) from error


def start_cache_cleaner():
    """Background daemon to prune ephemeral inputs/outputs older than 24 hours."""
    def cleaner_loop():
        while True:
            try:
                now = time.time()
                folders = (
                    config.INPUTS_DIR,
                    config.OUTPUTS_DIR,
                    config.IMAGE_AI_ROOT / "engine-input" / "FaceStudio"
                )
                for folder in folders:
                    if folder.exists():
                        for f in folder.glob("*"):
                            if f.is_file() and (now - f.stat().st_mtime) > 86400:
                                try:
                                    f.unlink()
                                except Exception:
                                    pass
            except Exception:
                pass
            time.sleep(3600)

    t = threading.Thread(target=cleaner_loop, daemon=True)
    t.start()


class FaceStudioHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS for local Web UI, Android Emulator, LAN Mobile devices, and Cloudflare Tunnels
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def send_json(self, status_code: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == '/api/outfit-progress':
            request_id = parse_qs(urlsplit(self.path).query).get('id', [''])[0]
            with PROGRESS_LOCK:
                state = OUTFIT_PROGRESS.get(request_id)
                payload = {'stage': state['stage']} if state else {'stage': 'Waiting for generation...'}
            return self.send_json(200, payload)

        # Fast Non-blocking API Status & Health Endpoint
        if path in ("/api/status", "/health", "/api/health"):
            with BACKEND_STATUS_LOCK:
                backend_ok = BACKEND_CONNECTED

            models_dict = {}
            models_list = []
            for k, v in config.MODELS.items():
                m_info = {
                    "id": k,
                    "name": v.get("name", k),
                    "badge": v.get("badge", ""),
                    "desc": v.get("desc", ""),
                    "type": v.get("type", "sdxl")
                }
                models_dict[k] = m_info
                models_list.append(m_info)

            return self.send_json(200, {
                "ready": backend_ok,
                "status": "ready" if backend_ok else "engine_starting",
                "backend_connected": backend_ok,
                "automatic_outfit_repair": config.AUTOMATIC_OUTFIT_REPAIR,
                "port": config.PORT,
                "models": models_dict,
                "models_list": models_list,
                "loras": config.get_available_loras(),
                "default_model": "cyberrealistic_xl",
                "default_negative_prompt": config.DEFAULT_NEGATIVE_PROMPT,
                "modes": [
                    {
                        "id": "in_place_swap",
                        "name": "In-Place Outfit Swap",
                        "desc": "100% original body, pose, face & background preserved; in-paints only the dress."
                    },
                    {
                        "id": "full_scene",
                        "name": "Full Scene Generation",
                        "desc": "Generates a brand new scene and outfit from prompt with 1:1 biometric identity lock."
                    }
                ],
                "history": list(HISTORY)[-12:]
            })

        if path == "/api/loras":
            return self.send_json(200, {
                "loras": config.get_available_loras(),
                "default_lora": None
            })

        if path == "/api/sample":
            sample_file = config.SAMPLES_DIR / "sample_model.png"
            w, h = 914, 1785
            if sample_file.exists():
                try:
                    with Image.open(sample_file) as simg:
                        w, h = simg.size
                except Exception:
                    pass
            return self.send_json(200, {
                "image_id": "sample_model",
                "image_url": "/samples/sample_model.png",
                "width": w,
                "height": h
            })

        if path == "/api/history":
            with HISTORY_LOCK:
                history_items = list(HISTORY)[-20:]
            return self.send_json(200, history_items)

        # Static File Routing
        if path in ("", "/"):
            return self.serve_file(config.WORKSPACE_ROOT / "static" / "index.html", "text/html; charset=utf-8")
        if path in ("/privacy", "/privacy.html"):
            return self.serve_file(config.WORKSPACE_ROOT / "privacy.html", "text/html; charset=utf-8")

        # Protected Static Assets (Path Traversal Prevention)
        if path.startswith("/static/"):
            target = safe_path(config.WORKSPACE_ROOT / "static", path[len("/static/"):])
            if target: return self.serve_file(target)
        elif path.startswith("/outputs/"):
            target = safe_path(config.OUTPUTS_DIR, path[len("/outputs/"):])
            if target: return self.serve_file(target)
        elif path.startswith("/inputs/"):
            target = safe_path(config.INPUTS_DIR, path[len("/inputs/"):])
            if target: return self.serve_file(target)
        elif path.startswith("/samples/"):
            target = safe_path(config.SAMPLES_DIR, path[len("/samples/"):])
            if target: return self.serve_file(target)

        # Fallback to static folder
        target = safe_path(config.WORKSPACE_ROOT / "static", path.lstrip("/"))
        if target:
            return self.serve_file(target)

        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"404 Not Found")

    def do_POST(self):
        path = self.path.split("?")[0]
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 60 * 1024 * 1024:
            return self.send_json(413, {"error": "Payload exceeds 60MB limit"})

        body = self.rfile.read(content_length)

        # /api/upload
        if path == "/api/upload":
            try:
                image_data = None
                content_type = self.headers.get("Content-Type", "")

                if "application/json" in content_type:
                    data = json.loads(body.decode("utf-8"))
                    raw_b64 = data.get("data") or data.get("image") or ""
                    if "," in raw_b64:
                        raw_b64 = raw_b64.split(",", 1)[1]
                    if raw_b64:
                        image_data = base64.b64decode(raw_b64, validate=True)
                elif "multipart/form-data" in content_type:
                    msg = email.message_from_bytes(b"Content-Type: " + content_type.encode("utf-8") + b"\r\n\r\n" + body)
                    payload = msg.get_payload()
                    if isinstance(payload, list):
                        for part in payload:
                            if hasattr(part, "get_filename") and part.get_filename():
                                image_data = part.get_payload(decode=True)
                                break
                    elif hasattr(msg, "get_payload") and hasattr(msg, "get_filename") and msg.get_filename():
                        image_data = msg.get_payload(decode=True)

                if not image_data:
                    return self.send_json(400, {"error": "No valid image provided in upload"})

                try:
                    img = decode_uploaded_image(image_data)
                except ValueError as error:
                    return self.send_json(415, {"error": str(error)})
                img_id = str(uuid.uuid4())[:10]
                src_filename = f"src_{img_id}.png"
                src_path = config.INPUTS_DIR / src_filename
                img.save(src_path, "PNG")

                # Generate clothing segmentation preview
                mask_pil, preview_overlay, faces, coverage = generate_dress_mask(
                    img, target="dress", keep_face=True
                )
                preview_filename = f"preview_{img_id}.png"
                preview_path = config.INPUTS_DIR / preview_filename
                preview_overlay.save(preview_path, "PNG")

                return self.send_json(200, {
                    "image_id": img_id,
                    "image_url": f"/inputs/{src_filename}",
                    "source_url": f"/inputs/{src_filename}",
                    "preview_url": f"/inputs/{preview_filename}",
                    "width": img.width,
                    "height": img.height,
                    "coverage": coverage,
                    "faces_detected": len(faces),
                    "warnings": (
                        ["Multiple people detected; the largest detected person is used for face matching."]
                        if len(faces) > 1 else []
                    )
                })
            except Exception as e:
                return self.send_json(500, {"error": f"Upload processing failed: {str(e)}"})

        # /api/change-dress
        if path == "/api/change-dress":
            try:
                data = json.loads(body.decode("utf-8"))
                image_id = data.get("image_id")
                prompt = data.get("prompt", "luxurious designer dress, elegant fabric")
                target = data.get("target", "dress")
                model_id = data.get("model_id") or data.get("model_key") or data.get("model") or "cyberrealistic_xl"
                if model_id == "flux":
                    model_id = "flux_klein"
                seed = data.get("seed")
                if seed is not None:
                    try:
                        seed = int(seed)
                    except Exception:
                        seed = None
                lora_id = data.get("lora_id")
                lora_weight = float(data.get("lora_weight", 0.75))
                mode_raw = data.get("mode", "in_place_swap")
                mode = "in_place_swap" if mode_raw in ("in_place", "in_place_swap", "wardrobe") else "full_scene"
                aspect_ratio = data.get("aspect_ratio", "original")
                framing = data.get("framing", "editorial")
                variations = int(data.get("variations", 1))
                garment_id = data.get("garment_id")

                # Extract source image
                source_image, source_url = extract_request_image(data)

                enable_face_id = bool(data.get("enable_face_id", True))
                face_id_weight = float(data.get("face_id_weight", 0.80))

                # If Virtual Wardrobe garment was uploaded, incorporate context into prompt
                if garment_id:
                    garment_file = config.INPUTS_DIR / f"src_{garment_id}.png"
                    if garment_file.exists():
                        prompt = f"{prompt}, exact match with reference garment drape and styling"

                # Incorporate MOSS-Image Typography & Slogan Mod if provided
                slogan = (data.get("slogan") or data.get("typography") or "").strip()
                if slogan and model_id == "moss_image":
                    try:
                        from core.tools_engine import render_moss_typography
                        prompt = render_moss_typography(prompt, slogan)
                    except Exception as e:
                        print(f"[Server] MOSS typography formatting note: {e}")

                # Run inference pipeline inside GPU semaphore queue
                progress_id = str(data.get('request_id') or '')
                if len(progress_id) > 64 or not all(c.isalnum() or c in '-_' for c in progress_id):
                    progress_id = ''
                set_outfit_progress(progress_id, 'Queued for local generation...')
                start_time = time.monotonic()
                with GPU_SEMAPHORE:
                    result = run_face_studio_pipeline(
                        source_image=source_image,
                        prompt=prompt,
                        target_garment=target,
                        model_id=model_id,
                        mode=mode,
                        aspect_ratio=aspect_ratio,
                        framing=framing,
                        variations=variations,
                        seed=seed,
                        lora_id=lora_id,
                        lora_weight=lora_weight,
                        enable_face_id=enable_face_id,
                        face_id_weight=face_id_weight,
                        progress_cb=lambda message: set_outfit_progress(progress_id, message)
                    )
                elapsed = round(time.monotonic() - start_time, 2)

                output_url = f"/outputs/{result['filename']}"
                results_list = result.get("results", [output_url])

                # Update in-memory history (bounded FIFO buffer)
                entry = {
                    "output_url": output_url,
                    "source_url": source_url,
                    "prompt": prompt,
                    "model_name": result["model_name"],
                    "lora_id": lora_id,
                    "lora_weight": lora_weight,
                    "mode": mode,
                    "elapsed": elapsed,
                    "timestamp": time.time()
                }
                with HISTORY_LOCK:
                    HISTORY.append(entry)

                return self.send_json(200, {
                    "success": True,
                    "status": "success",
                    "result_url": output_url,
                    "output_url": output_url,
                    "source_url": source_url,
                    "preview_url": source_url,
                    "model_name": result["model_name"],
                    "lora_id": lora_id,
                    "lora_weight": lora_weight,
                    "mode": mode,
                    "elapsed": elapsed,
                    "elapsed_seconds": elapsed,
                    "coverage": result.get("coverage", 0),
                    "face_swapped": result.get("face_swapped", False),
                    "results": results_list,
                    "requested_variations": result.get("requested_variations", 1),
                    "generated_variations": result.get("generated_variations", len(results_list)),
                    "warnings": result.get("warnings", []),
                    "quality": result.get("quality", {}),
                    "quality_results": result.get("quality_results", [])
                })
            except OutfitRejected as rejected:
                set_outfit_progress(locals().get('progress_id', ''), 'Rejected: original source retained.')
                return self.send_json(422, rejection_response(rejected, source_url))
            except ValueError as ve:
                return self.send_json(400, {"error": str(ve)})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return self.send_json(500, {"error": f"Outfit generation failed: {str(e)}"})

        # /api/restore
        if path == "/api/restore":
            try:
                data = json.loads(body.decode("utf-8"))
                source_image, source_url = extract_request_image(data)
                fidelity = float(data.get("fidelity", 0.85))
                start_time = time.monotonic()
                from core.tools_engine import restore_photo
                with GPU_SEMAPHORE:
                    restored = restore_photo(source_image, fidelity=fidelity)
                elapsed = round(time.monotonic() - start_time, 2)
                out_id = str(uuid.uuid4())[:10]
                out_filename = f"restore_{out_id}.png"
                out_path = config.OUTPUTS_DIR / out_filename
                restored.save(out_path, "PNG")
                return self.send_json(200, {
                    "success": True,
                    "status": "success",
                    "output_url": f"/outputs/{out_filename}",
                    "result_url": f"/outputs/{out_filename}",
                    "source_url": source_url,
                    "elapsed": elapsed
                })
            except Exception as e:
                import traceback; traceback.print_exc()
                return self.send_json(500, {"error": f"Photo restoration failed: {str(e)}"})

        # /api/remove-bg
        if path == "/api/remove-bg":
            try:
                data = json.loads(body.decode("utf-8"))
                source_image, source_url = extract_request_image(data)
                start_time = time.monotonic()
                from core.tools_engine import remove_background
                rembg_img = remove_background(source_image)
                elapsed = round(time.monotonic() - start_time, 2)
                out_id = str(uuid.uuid4())[:10]
                out_filename = f"rembg_{out_id}.png"
                out_path = config.OUTPUTS_DIR / out_filename
                rembg_img.save(out_path, "PNG")
                return self.send_json(200, {
                    "success": True,
                    "status": "success",
                    "output_url": f"/outputs/{out_filename}",
                    "result_url": f"/outputs/{out_filename}",
                    "source_url": source_url,
                    "elapsed": elapsed
                })
            except Exception as e:
                import traceback; traceback.print_exc()
                return self.send_json(500, {"error": f"Background removal failed: {str(e)}"})

        # /api/erase-object
        if path == "/api/erase-object":
            try:
                data = json.loads(body.decode("utf-8"))
                source_image, source_url = extract_request_image(data)
                mask_b64 = data.get("mask") or data.get("mask_data")
                if not mask_b64:
                    return self.send_json(400, {"error": "No eraser brush mask provided."})
                if "," in mask_b64:
                    mask_b64 = mask_b64.split(",", 1)[1]
                mask_img = Image.open(io.BytesIO(base64.b64decode(mask_b64)))
                start_time = time.monotonic()
                from core.tools_engine import erase_object
                erased = erase_object(source_image, mask_img)
                elapsed = round(time.monotonic() - start_time, 2)
                out_id = str(uuid.uuid4())[:10]
                out_filename = f"erase_{out_id}.png"
                out_path = config.OUTPUTS_DIR / out_filename
                erased.save(out_path, "PNG")
                return self.send_json(200, {
                    "success": True,
                    "status": "success",
                    "output_url": f"/outputs/{out_filename}",
                    "result_url": f"/outputs/{out_filename}",
                    "source_url": source_url,
                    "elapsed": elapsed
                })
            except Exception as e:
                import traceback; traceback.print_exc()
                return self.send_json(500, {"error": f"Object erasing failed: {str(e)}"})

        # /api/upscale
        if path == "/api/upscale":
            try:
                data = json.loads(body.decode("utf-8"))
                source_image, source_url = extract_request_image(data)
                model_name = data.get("model_name", "RealESRGAN_x4plus.pth")
                start_time = time.monotonic()
                from core.tools_engine import upscale_4k
                with GPU_SEMAPHORE:
                    upscaled = upscale_4k(source_image, model_name=model_name)
                elapsed = round(time.monotonic() - start_time, 2)
                out_id = str(uuid.uuid4())[:10]
                out_filename = f"upscale_{out_id}.png"
                out_path = config.OUTPUTS_DIR / out_filename
                upscaled.save(out_path, "PNG")
                return self.send_json(200, {
                    "success": True,
                    "status": "success",
                    "output_url": f"/outputs/{out_filename}",
                    "result_url": f"/outputs/{out_filename}",
                    "source_url": source_url,
                    "width": upscaled.width,
                    "height": upscaled.height,
                    "elapsed": elapsed
                })
            except Exception as e:
                import traceback; traceback.print_exc()
                return self.send_json(500, {"error": f"4K upscaling failed: {str(e)}"})

        # /api/deblur
        if path == "/api/deblur":
            try:
                data = json.loads(body.decode("utf-8"))
                source_image, source_url = extract_request_image(data)
                sharpness = float(data.get("sharpness", 1.35))
                start_time = time.monotonic()
                from core.tools_engine import deblur_image
                with GPU_SEMAPHORE:
                    deblurred = deblur_image(source_image, sharpness_factor=sharpness)
                elapsed = round(time.monotonic() - start_time, 2)
                out_id = str(uuid.uuid4())[:10]
                out_filename = f"deblur_{out_id}.png"
                out_path = config.OUTPUTS_DIR / out_filename
                deblurred.save(out_path, "PNG")
                return self.send_json(200, {
                    "success": True,
                    "status": "success",
                    "output_url": f"/outputs/{out_filename}",
                    "result_url": f"/outputs/{out_filename}",
                    "source_url": source_url,
                    "elapsed": elapsed
                })
            except Exception as e:
                import traceback; traceback.print_exc()
                return self.send_json(500, {"error": f"Deblur processing failed: {str(e)}"})

        self.send_json(404, {"error": "Unknown API endpoint"})

    def serve_file(self, filepath: Path, content_type: str = None):
        if not filepath.exists() or not filepath.is_file():
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")
            return

        if not content_type:
            content_type, _ = mimetypes.guess_type(str(filepath))
            if not content_type:
                content_type = "application/octet-stream"

        try:
            content = filepath.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode("utf-8"))


def main():
    global BACKEND_CONNECTED
    print("======================================================================")
    print("            Face Studio & Photovance AI Server")
    print("======================================================================")
    print(f"Host: http://127.0.0.1:{config.PORT}")
    print(f"ComfyUI Backend: {config.BACKEND_URL}")
    print("Connecting to local inference engine...")
    try:
        ensure_backend()
        print(f"[OK] Local ComfyUI GPU backend connected at {config.BACKEND_URL}.")
        with BACKEND_STATUS_LOCK:
            BACKEND_CONNECTED = True
    except Exception as e:
        print(f"[Warning] Backend initialization error: {e}")

    start_backend_status_monitor()
    start_cache_cleaner()
    print("[OK] Ephemeral cache cleaner active (24-hour TTL).")

    ThreadingHTTPServer.allow_reuse_address = True
    while True:
        server = None
        try:
            server = ThreadingHTTPServer((config.HOST, config.PORT), FaceStudioHandler)
            print(f"\n[Running] Face Studio Web Studio is active at http://{config.HOST}:{config.PORT}")
            print("Press Ctrl+C to stop.\n")
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
            if server:
                try: server.server_close()
                except Exception: pass
            break
        except BaseException as e:
            print(f"[Server Warning] Recovering from error: {e}")
            if server:
                try: server.server_close()
                except Exception: pass
            time.sleep(2)


if __name__ == "__main__":
    main()
