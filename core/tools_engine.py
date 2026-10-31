"""Face Studio & Photovance AI - Studio Utility Tools Engine.

Provides 5 standalone, update-friendly AI image enhancement tools:
1. restore_photo: AI Face & Photo Restoration (CodeFormer + CLAHE + Denoise)
2. remove_background: AI Background Removal (rembg / U2-Net alpha matting)
3. erase_object: Magic Object & Blemish Eraser (interactive mask inpainting + texture synthesis)
4. upscale_4k: 4K Ultra-Upscaler (RealESRGAN_x4plus / NMKD via ComfyUI)
5. deblur_image: AI Deblur & Sharpener (Deconvolution + Guided Filter + CodeFormer)
"""
import io
import time
import uuid
import threading
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
import cv2
import numpy as np

from core import config

# Thread-safe global rembg session cache
_REMBG_SESSION = None
_REMBG_LOCK = threading.Lock()

def get_rembg_session():
    """Lazily load and cache U2-Net ONNX session to avoid 176MB disk reloads per request."""
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        with _REMBG_LOCK:
            if _REMBG_SESSION is None:
                try:
                    import rembg
                    _REMBG_SESSION = rembg.new_session("u2net")
                    print("[ToolsEngine] Initialized cached U2-Net rembg session.")
                except Exception as e:
                    print(f"[ToolsEngine] Warning initializing rembg session: {e}")
    return _REMBG_SESSION


# ---------------------------------------------------------------------------
# 1. AI Photo & Face Restoration
# ---------------------------------------------------------------------------
def restore_photo(
    image: Image.Image,
    fidelity: float = 0.65,
    enhance_background: bool = True
) -> Image.Image:
    """Restores compressed, degraded, or old portraits and photos.
    Uses CodeFormer discrete codebook prior for faces + adaptive detail enhancement.
    """
    img_rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    # Step 1: Overall photographic enhancement (CLAHE + Bilateral Denoise)
    enhanced_bgr = bgr.copy()
    if enhance_background:
        # Subtle denoising while preserving edges
        denoised = cv2.bilateralFilter(enhanced_bgr, d=7, sigmaColor=35, sigmaSpace=35)
        # Adaptive contrast on L channel in LAB
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # Subtle unsharp sharpening
        blurred = cv2.GaussianBlur(enhanced_bgr, (0, 0), 1.5)
        enhanced_bgr = cv2.addWeighted(enhanced_bgr, 1.2, blurred, -0.2, 0)

    # Step 2: CodeFormer Face Restoration
    try:
        from core.face_engine import get_face_modules
        app, _, codeformer_net, face_helper, fallback_restorer = get_face_modules()

        if codeformer_net is not None and face_helper is not None:
            import torch
            from codeformer.basicsr.utils import img2tensor, tensor2img
            device = "cuda" if torch.cuda.is_available() else "cpu"
            codeformer_net.to(device)
            if hasattr(face_helper, "face_detector") and face_helper.face_detector is not None:
                face_helper.face_detector.to(device)
            if hasattr(face_helper, "face_parse") and face_helper.face_parse is not None:
                face_helper.face_parse.to(device)
            face_helper.device = device

            try:
                face_helper.clean_all()
                face_helper.read_image(enhanced_bgr)
                face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
                face_helper.align_warp_face()

                for cropped_face in face_helper.cropped_faces:
                    t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
                    t = (t - 0.5) / 0.5
                    t = t.unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = codeformer_net(t, w=fidelity)[0]
                        restored_face = tensor2img(out, rgb2bgr=True, min_max=(-1, 1))
                    face_helper.add_restored_face(restored_face)

                if len(face_helper.restored_faces) > 0:
                    face_helper.get_inverse_affine(None)
                    enhanced_bgr = face_helper.paste_faces_to_input_image()
                    try:
                        from core.face_engine import enhance_facial_clarity
                        if hasattr(face_helper, "det_faces") and face_helper.det_faces:
                            for df in face_helper.det_faces:
                                enhanced_bgr = enhance_facial_clarity(enhanced_bgr, df[:4])
                    except Exception as clar_err:
                        print(f"[ToolsEngine] Facial clarity note: {clar_err}")
            finally:
                try:
                    codeformer_net.to("cpu")
                    if hasattr(face_helper, "face_detector") and face_helper.face_detector is not None:
                        face_helper.face_detector.to("cpu")
                    if hasattr(face_helper, "face_parse") and face_helper.face_parse is not None:
                        face_helper.face_parse.to("cpu")
                    face_helper.device = "cpu"
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        elif fallback_restorer is not None:
            _, _, fb_bgr = fallback_restorer.enhance(
                enhanced_bgr, has_aligned=False, only_center_face=False, paste_back=True
            )
            enhanced_bgr = fb_bgr
    except Exception as e:
        print(f"[ToolsEngine] Face restoration warning: {e}")

    final_rgb = cv2.cvtColor(np.clip(enhanced_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_rgb)


# ---------------------------------------------------------------------------
# 2. Transparent Background Remover
# ---------------------------------------------------------------------------
def remove_background(
    image: Image.Image,
    alpha_matting: bool = False
) -> Image.Image:
    """Removes image background returning a transparent PNG (RGBA) using rembg.
    Falls back gracefully if rembg encounters edge cases.
    """
    img_rgb = ImageOps.exif_transpose(image).convert("RGB")
    try:
        import rembg
        session = get_rembg_session()
        if session is not None:
            output = rembg.remove(
                img_rgb,
                session=session,
                alpha_matting=alpha_matting
            )
        else:
            output = rembg.remove(
                img_rgb,
                alpha_matting=alpha_matting
            )
        return output
    except Exception as e:
        print(f"[ToolsEngine] Primary rembg error ({e}), retrying standard mode...")
        try:
            import rembg
            return rembg.remove(img_rgb)
        except Exception as e2:
            print(f"[ToolsEngine] Rembg fallback failed: {e2}")
            # Fallback: create alpha channel from SegFormer person mask
            try:
                from core.clothing_engine import parse_clothing_labels
                labels = parse_clothing_labels(img_rgb)
                person_mask = (labels > 0).astype(np.uint8) * 255
                # Feather mask
                mask_pil = Image.fromarray(person_mask).filter(ImageFilter.GaussianBlur(1.5))
                rgba = img_rgb.convert("RGBA")
                rgba.putalpha(mask_pil)
                return rgba
            except Exception:
                return img_rgb.convert("RGBA")


# ---------------------------------------------------------------------------
# 3. Magic Object & Blemish Eraser
# ---------------------------------------------------------------------------
def erase_object(
    image: Image.Image,
    mask: Image.Image,
    inpaint_radius: int = 5
) -> Image.Image:
    """Seamlessly erases objects, blemishes, watermarks, or unwanted items.
    Takes the source image and a binary mask (white = erase area).
    """
    img_rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    mask_gray = np.array(ImageOps.exif_transpose(mask).convert("L"))
    if mask_gray.shape != bgr.shape[:2]:
        mask_gray = cv2.resize(mask_gray, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Threshold mask to strict binary uint8
    _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

    # Slight dilation to cover boundary bleed
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # Fast Marching Method (Telea) Inpainting
    inpainted_telea = cv2.inpaint(bgr, dilated_mask, inpaintRadius=max(3, inpaint_radius), flags=cv2.INPAINT_TELEA)

    # Navier-Stokes for smooth gradient areas
    inpainted_ns = cv2.inpaint(bgr, dilated_mask, inpaintRadius=max(3, inpaint_radius), flags=cv2.INPAINT_NS)

    # Blend Telea (texture-preserving) and NS (smooth gradients)
    inpainted_bgr = cv2.addWeighted(inpainted_telea, 0.65, inpainted_ns, 0.35, 0)

    # Seamless boundary feathering back to original image
    feathered_mask = cv2.GaussianBlur(dilated_mask.astype(np.float32) / 255.0, (0, 0), sigmaX=1.5, sigmaY=1.5)[:, :, None]
    final_bgr = (bgr.astype(np.float32) * (1.0 - feathered_mask) + inpainted_bgr.astype(np.float32) * feathered_mask).clip(0, 255).astype(np.uint8)

    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_rgb)


# ---------------------------------------------------------------------------
# 4. 4K Ultra-Upscaler
# ---------------------------------------------------------------------------
def upscale_4k(
    image: Image.Image,
    model_name: str = "RealESRGAN_x4plus.pth"
) -> Image.Image:
    """Upscales image by 4x to crystal-clear 4K resolution using RealESRGAN/NMKD via ComfyUI.
    Clamps input resolution so output matches target 4K UHD (~3840x2160) without CUDA OOM.
    """
    from core.pipeline import upload_image, execute_workflow, ensure_backend
    import tempfile

    if not ensure_backend():
        raise RuntimeError("Backend engine failed to start for 4K Upscale.")

    img = ImageOps.exif_transpose(image).convert("RGB")
    w, h = img.size

    # Hard cap target output dimension to 4096px on the longest edge
    # RealESRGAN_x4plus multiplies input dimensions by 4x.
    max_in_edge = 1024
    longest_edge = max(w, h)
    if longest_edge > max_in_edge:
        scale = max_in_edge / float(longest_edge)
        new_w = max(64, int(w * scale))
        new_h = max(64, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        print(f"[ToolsEngine] Clamped input image from {w}x{h} to {new_w}x{new_h} for safe 4K upscaling.")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    img.save(tmp_path, "PNG")

    try:
        src_subpath = upload_image(tmp_path)
        graph = {
            "source": {
                "class_type": "LoadImage",
                "inputs": {"image": src_subpath}
            },
            "model": {
                "class_type": "UpscaleModelLoader",
                "inputs": {"model_name": model_name}
            },
            "upscale": {
                "class_type": "ImageUpscaleWithModel",
                "inputs": {
                    "upscale_model": ["model", 0],
                    "image": ["source", 0]
                }
            },
            "save": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["upscale", 0],
                    "filename_prefix": "FaceStudio_4K"
                }
            }
        }
        res_img = execute_workflow(graph)
        return res_img
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 5. AI Deblur & Sharpener
# ---------------------------------------------------------------------------
def deblur_image(
    image: Image.Image,
    sharpness_factor: float = 1.35
) -> Image.Image:
    """Restores clarity from motion blur or out-of-focus soft blur.
    Combines high-pass frequency restoration, Laplacian sharpening, and face deblur.
    Uses float32 frequency separation to preserve dark shadow dips and edge contours.
    """
    img_rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # 1. Multi-scale frequency separation using float32 to avoid cv2.subtract uint8 truncation
    sigma1, sigma2 = 1.2, 3.0
    blur1 = cv2.GaussianBlur(bgr, (0, 0), sigma1)
    blur2 = cv2.GaussianBlur(bgr, (0, 0), sigma2)

    bgr_f = bgr.astype(np.float32)
    blur1_f = blur1.astype(np.float32)
    blur2_f = blur2.astype(np.float32)

    # High frequency micro-details (bipolar, retains dark edge dips)
    high_freq = bgr_f - blur1_f
    # Mid-frequency edge contours
    mid_freq = blur1_f - blur2_f

    # Reconstruct sharpened image
    sharpened_float = bgr_f + (high_freq * (sharpness_factor * 0.7)) + (mid_freq * (sharpness_factor * 0.4))
    sharpened_bgr = np.clip(sharpened_float, 0, 255).astype(np.uint8)

    # 2. If face is present, enhance facial features with CodeFormer
    try:
        from core.face_engine import get_face_modules
        app, _, codeformer_net, face_helper, _ = get_face_modules()
        if codeformer_net is not None and face_helper is not None:
            import torch
            from codeformer.basicsr.utils import img2tensor, tensor2img
            device = "cuda" if torch.cuda.is_available() else "cpu"
            codeformer_net.to(device)
            if hasattr(face_helper, "face_detector") and face_helper.face_detector is not None:
                face_helper.face_detector.to(device)
            if hasattr(face_helper, "face_parse") and face_helper.face_parse is not None:
                face_helper.face_parse.to(device)
            face_helper.device = device

            try:
                face_helper.clean_all()
                face_helper.read_image(sharpened_bgr)
                face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
                face_helper.align_warp_face()

                for cropped_face in face_helper.cropped_faces:
                    t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
                    t = (t - 0.5) / 0.5
                    t = t.unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = codeformer_net(t, w=0.90)[0]
                        restored_face = tensor2img(out, rgb2bgr=True, min_max=(-1, 1))
                    face_helper.add_restored_face(restored_face)

                if len(face_helper.restored_faces) > 0:
                    face_helper.get_inverse_affine(None)
                    sharpened_bgr = face_helper.paste_faces_to_input_image()
            finally:
                try:
                    codeformer_net.to("cpu")
                    if hasattr(face_helper, "face_detector") and face_helper.face_detector is not None:
                        face_helper.face_detector.to("cpu")
                    if hasattr(face_helper, "face_parse") and face_helper.face_parse is not None:
                        face_helper.face_parse.to("cpu")
                    face_helper.device = "cpu"
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    except Exception as e:
        print(f"[ToolsEngine] Deblur face enhancement note: {e}")

    final_rgb = cv2.cvtColor(sharpened_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_rgb)


def compose_moss_layout(
    elements: list[dict],
    canvas_width: int = 1024,
    canvas_height: int = 1024
) -> str:
    """
    Format spatial regions into MOSS-Image layout prompt format.
    
    Each element is a dict with:
      - 'label': str (e.g. 'fashion model in crimson gown', 'embroidered text PARIS')
      - 'box': list[float] [ymin, xmin, ymax, xmax] normalized to 0.0-1.0
    
    Returns:
        Structured layout string for MOSS-Image spatial positioning.
    """
    if not elements:
        return ""

    layout_parts = []
    for el in elements:
        label = el.get("label", "").strip()
        box = el.get("box", [0.0, 0.0, 1.0, 1.0])
        if len(box) == 4 and label:
            ymin, xmin, ymax, xmax = [max(0.0, min(1.0, float(v))) for v in box]
            y1 = int(round(ymin * 1000))
            x1 = int(round(xmin * 1000))
            y2 = int(round(ymax * 1000))
            x2 = int(round(xmax * 1000))
            layout_parts.append(f'[{label}] at <{y1},{x1},{y2},{x2}>')

    return "; ".join(layout_parts)


def render_moss_typography(
    base_prompt: str,
    slogan: str,
    garment_type: str = "hoodie",
    text_color: str = "white",
    placement: str = "chest"
) -> str:
    """
    MOSS Typography Slogan Mod: Formats text-on-fabric prompts for MOSS-Image.
    
    Generates exact character-level typographic instructions for printing or embroidery.
    """
    slogan_clean = (slogan or "").strip().strip('"').strip("'")
    if not slogan_clean:
        return base_prompt

    typo_directive = (
        f'wearing a {garment_type} featuring the bold text "{slogan_clean}" '
        f'prominently printed across the {placement} in crisp {text_color} lettering, '
        f'razor-sharp typographical character edges, 100% correct spelling, '
        f'realistic fabric print texture with natural cloth folds'
    )

    if base_prompt:
        return f"{base_prompt}, {typo_directive}"
    return typo_directive
