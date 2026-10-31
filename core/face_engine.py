"""Face identity transfer, scene-aware tone calibration and semantic hair recovery."""
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageOps
from core.skin_engine import feather_region

# Ensure distutils compatibility shim for Python 3.12 + CodeFormer
if "distutils" not in sys.modules:
    try:
        import setuptools._distutils as distutils
        import setuptools._distutils.version as distutils_version
        distutils.version = distutils_version
        sys.modules["distutils"] = distutils
        sys.modules["distutils.version"] = distutils_version
    except Exception:
        pass

import threading

from core import config

# Suppress OpenCV FaceDetectorYN diagnostic logging
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

_FACE_APP = None
_SWAPPER = None
_CODEFORMER_NET = None
_FACE_HELPER = None
_FALLBACK_RESTORER = None
_MODULE_LOCK = threading.Lock()
_DETECTOR_LOCK = threading.Lock()
_CODEFORMER_LOCK = threading.Lock()


def get_face_modules():
    """Lazily load and cache face detector, swapper, CodeFormer prior, and face helper with thread lock."""
    global _FACE_APP, _SWAPPER, _CODEFORMER_NET, _FACE_HELPER, _FALLBACK_RESTORER

    if _FACE_APP is None:
        with _MODULE_LOCK:
            if _FACE_APP is None:
                import insightface
                from insightface.app import FaceAnalysis

                models_dir = config.INSIGHTFACE_DIR

                # 1. Buffalo_l Detector & 512-d Identity Extractor
                _FACE_APP = FaceAnalysis(name="buffalo_l", root=str(models_dir.parent), providers=["CPUExecutionProvider"])
                _FACE_APP.prepare(ctx_id=0, det_size=(640, 640))

                # 2. Inswapper 128
                swapper_path = models_dir / "inswapper_128.onnx"
                if not swapper_path.exists():
                    raise FileNotFoundError(f"Missing {swapper_path}")
                _SWAPPER = insightface.model_zoo.get_model(str(swapper_path), download=False, providers=["CPUExecutionProvider"])

                # 3. CodeFormer discrete codebook prior network (stored on CPU until hydrated)
                codeformer_path = models_dir / "codeformer.pth"
                if codeformer_path.exists():
                    try:
                        from codeformer.basicsr.utils.registry import ARCH_REGISTRY
                        import codeformer.basicsr.archs
                        from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper

                        arch = ARCH_REGISTRY.get("CodeFormer")
                        net = arch(
                            dim_embd=512,
                            codebook_size=1024,
                            n_head=8,
                            n_layers=9,
                            connect_list=["32", "64", "128", "256"]
                        ).to("cpu")

                        ckpt = torch.load(str(codeformer_path), map_location="cpu")
                        weights = ckpt["params_ema"] if "params_ema" in ckpt else ckpt
                        net.load_state_dict(weights)
                        net.eval()
                        _CODEFORMER_NET = net

                        _FACE_HELPER = FaceRestoreHelper(
                            upscale_factor=1,
                            face_size=512,
                            crop_ratio=(1, 1),
                            det_model="retinaface_resnet50",
                            save_ext="png",
                            use_parse=True,
                            device="cpu"
                        )
                        print("[FaceEngine] CodeFormer Discrete Codebook Prior loaded on CPU (On-Demand CUDA Hydration active)")
                    except Exception as e:
                        print(f"[FaceEngine] Failed to load CodeFormer ({e}), falling back to GFPGAN")

                # 4. Fallback restorer (GFPGAN)
                gfpgan_path = models_dir / "GFPGANv1.4.pth"
                if gfpgan_path.exists() and _CODEFORMER_NET is None:
                    try:
                        from gfpgan import GFPGANer
                        _FALLBACK_RESTORER = GFPGANer(
                            model_path=str(gfpgan_path),
                            upscale=1,
                            arch="clean",
                            channel_multiplier=2,
                            device="cpu"
                        )
                        print("[FaceEngine] GFPGAN fallback loaded on CPU")
                    except Exception as e:
                        print(f"[FaceEngine] GFPGAN load warning: {e}")

    return _FACE_APP, _SWAPPER, _CODEFORMER_NET, _FACE_HELPER, _FALLBACK_RESTORER



def reinject_micro_pores(
    base_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    bbox: list,
    src_bgr: np.ndarray = None,
    src_bbox: list = None,
    src_kps: np.ndarray = None,
    tgt_kps: np.ndarray = None,
    pore_strength: float = 1.0,
    body_var: float = None
) -> np.ndarray:
    """
    Multi-Scale Frequency-Separated DSLR Epidermal Pore Recovery (Bug B2):
    Extracts authentic micro-pores and dermal relief exclusively from ref_bgr (the pose-aligned render).
    Computes scalar target pore variance strictly from src_bgr inner facial skin without 2D affine warping,
    preventing reference hair strands and earrings from being embossed as vertical cheek scars.
    """
    try:
        t_bx1, t_by1, t_bx2, t_by2 = [int(x) for x in bbox]
        h_img, w_img = base_bgr.shape[:2]
        t_bx1, t_by1 = max(0, t_bx1), max(0, t_by1)
        t_bx2, t_by2 = min(w_img, t_bx2), min(h_img, t_by2)
        if t_bx2 <= t_bx1 or t_by2 <= t_by1:
            return base_bgr

        # 1. Compute target variance from src_bgr inside its inner facial ellipse
        target_var = 120.0
        if src_bgr is not None and src_bbox is not None:
            try:
                sx1, sy1, sx2, sy2 = [int(v) for v in src_bbox]
                sx1, sy1 = max(0, sx1), max(0, sy1)
                sx2, sy2 = min(src_bgr.shape[1], sx2), min(src_bgr.shape[0], sy2)
                src_crop = src_bgr[sy1:sy2, sx1:sx2]
                if src_crop.size > 0:
                    sh, sw = src_crop.shape[:2]
                    src_face_gray = cv2.cvtColor(src_crop, cv2.COLOR_BGR2GRAY)
                    src_yc = cv2.cvtColor(src_crop, cv2.COLOR_BGR2YCrCb)
                    inner_oval = np.zeros((sh, sw), dtype=np.uint8)
                    cv2.ellipse(inner_oval, (sw // 2, sh // 2), (max(1, int(sw * 0.28)), max(1, int(sh * 0.30))), 0, 0, 360, 1, -1)
                    skin_m = (inner_oval > 0) & (src_yc[:, :, 1] >= 133) & (src_yc[:, :, 1] <= 175) & (src_yc[:, :, 2] >= 77) & (src_yc[:, :, 2] <= 130) & (src_face_gray >= 45) & (src_face_gray <= 235)
                    lap_s = cv2.Laplacian(src_face_gray, cv2.CV_64F)
                    raw_var = float(np.var(lap_s[skin_m])) if np.count_nonzero(skin_m) >= 50 else float(lap_s.var())
                    target_var = float(np.clip(raw_var, 35.0, 240.0))
            except Exception:
                pass
        elif body_var is not None and body_var > 10.0:
            target_var = float(np.clip(body_var, 35.0, 240.0))

        # 2. Extract spatial high-frequency pore structure exclusively from pose-aligned ref_bgr
        ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_l = ref_lab[:, :, 0]
        blur_1_2 = cv2.GaussianBlur(ref_l, (0, 0), 1.2)
        blur_3_6 = cv2.GaussianBlur(ref_l, (0, 0), 3.6)
        detail_band = blur_1_2 - blur_3_6
        detail_band = detail_band - np.mean(detail_band, axis=(0, 1), keepdims=True)

        ref_gray = cv2.cvtColor(ref_bgr[t_by1:t_by2, t_bx1:t_bx2], cv2.COLOR_BGR2GRAY) if t_by2 > t_by1 and t_bx2 > t_bx1 else None
        ref_var = float(cv2.Laplacian(ref_gray, cv2.CV_64F).var()) if ref_gray is not None and ref_gray.size > 0 else 100.0
        ref_var = max(ref_var, 1.0)
        gain = float(np.clip(np.sqrt(target_var / ref_var), 0.70, 1.45))
        pore_detail = detail_band * gain

        # 3. Anatomical polygon face mask down to mandibular jawline / chin fade zone
        bw, bh = t_bx2 - t_bx1, t_by2 - t_by1
        face_mask = np.zeros((h_img, w_img), dtype=np.float32)
        neck_bottom = min(h_img, int(t_by2 + bh * 0.12))
        pts = np.array([
            [t_bx1 + bw * 0.15, t_by1],
            [t_bx2 - bw * 0.15, t_by1],
            [t_bx2 + bw * 0.05, t_by1 + bh * 0.35],
            [t_bx2, t_by1 + bh * 0.70],
            [t_bx2 + int(bw * 0.05), neck_bottom],
            [t_bx1 - int(bw * 0.05), neck_bottom],
            [t_bx1, t_by1 + bh * 0.70],
            [t_bx1 - bw * 0.05, t_by1 + bh * 0.35],
        ], dtype=np.int32)
        cv2.fillPoly(face_mask, [pts], 1.0)
        face_mask = cv2.GaussianBlur(face_mask, (21, 21), 6.0)

        # 4. YCrCb skin chrominance segmentation (face and neck)
        ycrcb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2YCrCb)
        skin_cr_cb = cv2.inRange(ycrcb, np.array([0, 125, 70]), np.array([255, 185, 140])).astype(np.float32) / 255.0

        # 5. Exclude dark regions (pupils, lashes, nostrils, brows)
        gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
        non_dark = cv2.inRange(gray, 40, 240).astype(np.float32) / 255.0

        # 6. Semantic eye orbits exclusion
        eye_mask = np.zeros((h_img, w_img), dtype=np.float32)
        if tgt_kps is not None and len(tgt_kps) >= 5:
            eye_dist = max(10.0, float(np.linalg.norm(tgt_kps[0] - tgt_kps[1])))
            cv2.ellipse(eye_mask, (int(tgt_kps[0][0]), int(tgt_kps[0][1])), (int(eye_dist * 0.28), int(eye_dist * 0.20)), 0, 0, 360, 1.0, -1)
            cv2.ellipse(eye_mask, (int(tgt_kps[1][0]), int(tgt_kps[1][1])), (int(eye_dist * 0.28), int(eye_dist * 0.20)), 0, 0, 360, 1.0, -1)
        else:
            cv2.ellipse(eye_mask, (int(t_bx1 + bw * 0.32), int(t_by1 + bh * 0.40)), (int(bw * 0.16), int(bh * 0.12)), 0, 0, 360, 1.0, -1)
            cv2.ellipse(eye_mask, (int(t_bx1 + bw * 0.68), int(t_by1 + bh * 0.40)), (int(bw * 0.16), int(bh * 0.12)), 0, 0, 360, 1.0, -1)
        eye_mask = cv2.GaussianBlur(eye_mask, (15, 15), 5.0)
        non_eye = 1.0 - eye_mask

        # 7. Semantic lip vermilion exclusion
        lip_mask = np.zeros((h_img, w_img), dtype=np.float32)
        if tgt_kps is not None and len(tgt_kps) >= 5:
            mouth_center = (tgt_kps[3] + tgt_kps[4]) / 2.0
            mouth_w = max(10.0, float(np.linalg.norm(tgt_kps[3] - tgt_kps[4])))
            cv2.ellipse(lip_mask, (int(mouth_center[0]), int(mouth_center[1])), (int(mouth_w * 0.60), int(mouth_w * 0.32)), 0, 0, 360, 1.0, -1)
        else:
            lip_y_mid = int(t_by1 + bh * 0.77)
            lip_x_mid = (t_bx1 + t_bx2) // 2
            cv2.ellipse(lip_mask, (lip_x_mid, lip_y_mid), (int(bw * 0.22), int(bh * 0.11)), 0, 0, 360, 1.0, -1)
        lip_mask = cv2.GaussianBlur(lip_mask, (15, 15), 5.0)
        non_lip = 1.0 - lip_mask

        # 8. Final combined skin mask with smooth feathering (tgt_skin_gate)
        final_skin_mask = face_mask * skin_cr_cb * non_dark * non_lip * non_eye
        final_skin_mask = cv2.GaussianBlur(final_skin_mask, (11, 11), 3.0)

        # 9. Re-inject multi-band epidermal pore detail into Luminance (L*) channel
        base_lab = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        base_lab[:, :, 0] = np.clip(base_lab[:, :, 0] + (pore_detail * pore_strength * final_skin_mask * 0.85), 0, 255)
        return cv2.cvtColor(base_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    except Exception as e:
        print(f"[FaceEngine] Micro-pore reinjection note: {e}")
        return base_bgr


def inject_photographic_grain(img_bgr: np.ndarray, intensity: float = 0.007) -> np.ndarray:
    """Injects subtle monochromatic 35mm film emulsion grain to break digital sensor sheen."""
    h, w = img_bgr.shape[:2]
    noise = np.random.normal(0, intensity * 255, (h, w, 1)).astype(np.float32)
    return np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def enhance_facial_clarity(restored_bgr: np.ndarray, face_bbox: list, kps: np.ndarray = None, strength: float = 1.0) -> np.ndarray:
    """Enhance ocular, brow, nasal, and lip definition with adaptive multi-scale micro-contrast."""
    try:
        h, w = restored_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in face_bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)

        # 1. Build landmark focus mask (eyes, brows, nose, lips)
        clarity_mask = np.zeros((h, w), dtype=np.float32)

        if kps is not None and len(kps) >= 5:
            # 5-point facial keypoints: 0=left eye, 1=right eye, 2=nose tip, 3=mouth left, 4=mouth right
            eye_dist = max(10.0, float(np.linalg.norm(kps[0] - kps[1])))
            r_eye_x = int(eye_dist * 0.28)
            r_eye_y = int(eye_dist * 0.22)
            cv2.ellipse(clarity_mask, (int(kps[0][0]), int(kps[0][1])), (r_eye_x, r_eye_y), 0, 0, 360, 1.0, -1)
            cv2.ellipse(clarity_mask, (int(kps[1][0]), int(kps[1][1])), (r_eye_x, r_eye_y), 0, 0, 360, 1.0, -1)

            # Nose tip & ridge
            cv2.ellipse(clarity_mask, (int(kps[2][0]), int(kps[2][1])), (int(eye_dist * 0.20), int(eye_dist * 0.25)), 0, 0, 360, 0.85, -1)

            # Mouth
            mouth_center = (kps[3] + kps[4]) / 2.0
            mouth_w = max(10.0, float(np.linalg.norm(kps[3] - kps[4])))
            cv2.ellipse(clarity_mask, (int(mouth_center[0]), int(mouth_center[1])), (int(mouth_w * 0.65), int(mouth_w * 0.35)), 0, 0, 360, 1.0, -1)
        else:
            # Geometric fallback from bounding box
            cv2.ellipse(clarity_mask, (int(x1 + bw * 0.31), int(y1 + bh * 0.38)),
                        (int(bw * 0.17), int(bh * 0.13)), 0, 0, 360, 1.0, -1)
            cv2.ellipse(clarity_mask, (int(x1 + bw * 0.69), int(y1 + bh * 0.38)),
                        (int(bw * 0.17), int(bh * 0.13)), 0, 0, 360, 1.0, -1)

            cv2.ellipse(clarity_mask, (int(x1 + bw * 0.50), int(y1 + bh * 0.55)),
                        (int(bw * 0.13), int(bh * 0.15)), 0, 0, 360, 0.85, -1)

            cv2.ellipse(clarity_mask, (int(x1 + bw * 0.50), int(y1 + bh * 0.77)),
                        (int(bw * 0.22), int(bh * 0.11)), 0, 0, 360, 1.0, -1)

        clarity_mask = cv2.GaussianBlur(clarity_mask, (21, 21), 5.0)[:, :, np.newaxis]

        # 2. Multi-scale unsharp masking for micro-detail without haloing
        bgr_f = restored_bgr.astype(np.float32)
        blur1 = cv2.GaussianBlur(bgr_f, (0, 0), 0.8)
        high1 = bgr_f - blur1

        blur2 = cv2.GaussianBlur(bgr_f, (0, 0), 2.0)
        high2 = blur1 - blur2

        # Contrast-adaptive gain: boost fine detail while preventing clipping/harshness
        detail_boost = (high1 * 0.28 + high2 * 0.16) * clarity_mask * float(strength)
        detail_boost = np.clip(detail_boost, -20.0, 20.0)
        enhanced = np.clip(bgr_f + detail_boost, 0, 255).astype(np.uint8)
        return enhanced
    except Exception as e:
        print(f"[FaceEngine] Facial clarity enhancement note: {e}")
        return restored_bgr


def _match_low_freq_cheek_boundary(
    warped_bgr: np.ndarray,
    target_bgr: np.ndarray,
    hull_mask: np.ndarray,
    bbox=None,
) -> np.ndarray:
    """Align low-frequency LAB luminance/chrominance along the outer cheek transition ring."""
    h, w = target_bgr.shape[:2]
    ring = (hull_mask > 0.02) & (hull_mask < 0.85)
    if np.count_nonzero(ring) < 60:
        return warped_bgr
    if bbox is not None:
        bw = max(1.0, float(bbox[2] - bbox[0]))
        bh = max(1.0, float(bbox[3] - bbox[1]))
        sigma = max(9.0, 0.08 * min(bw, bh))
    else:
        sigma = 15.0
    ksize = int(sigma * 3) | 1
    w_lab = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    t_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    valid_w = (np.any(warped_bgr > 2, axis=2)).astype(np.float32)
    valid_blur = cv2.GaussianBlur(valid_w, (ksize, ksize), sigma) + 1e-6
    out_lab = w_lab.copy()
    edge_weight = np.clip(1.0 - (hull_mask / 0.85), 0.0, 1.0) * (hull_mask > 0.005).astype(np.float32)
    edge_weight = cv2.GaussianBlur(edge_weight, (21, 21), 5.0)
    for c, strength in ((0, 0.65), (1, 0.85), (2, 0.85)):
        w_low = cv2.GaussianBlur(w_lab[:, :, c] * valid_w, (ksize, ksize), sigma) / valid_blur
        t_low = cv2.GaussianBlur(t_lab[:, :, c], (ksize, ksize), sigma)
        delta = (t_low - w_low) * edge_weight * strength
        out_lab[:, :, c] = np.clip(out_lab[:, :, c] + delta, 0.0, 255.0)
    return cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def warp_back_128(
    bgr_fake: np.ndarray,
    M: np.ndarray,
    target_bgr: np.ndarray,
    hull_mask: np.ndarray = None,
    bbox: list = None,
) -> np.ndarray:
    """
    Warps 128x128 face back onto canvas with elliptical margin-14 crop oval,
    smooth feathering, and low-frequency cheek boundary matching (Bug B1).
    """
    h_tgt, w_tgt = target_bgr.shape[:2]
    IM = cv2.invertAffineTransform(M)
    
    warped_bgr = cv2.warpAffine(
        bgr_fake,
        IM,
        (w_tgt, h_tgt),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    margin = 14
    crop_oval = np.zeros((128, 128), dtype=np.float32)
    center = (64, 64)
    axes = (64 - margin, 64 - margin)
    cv2.ellipse(crop_oval, center, axes, 0, 0, 360, 1.0, -1)
    crop_oval = cv2.GaussianBlur(crop_oval, (21, 21), 6.0)
    
    valid_crop_mask = cv2.warpAffine(
        crop_oval,
        IM,
        (w_tgt, h_tgt),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0
    )
    if valid_crop_mask.ndim == 2:
        valid_crop_mask = valid_crop_mask[:, :, np.newaxis]
        
    if hull_mask is not None:
        if hull_mask.ndim == 2:
            hull_mask = hull_mask[:, :, np.newaxis]
        h_norm = hull_mask.astype(np.float32)
        if h_norm.max() > 1.0:
            h_norm = h_norm / 255.0
        active_mask = h_norm * valid_crop_mask
    else:
        active_mask = valid_crop_mask
        
    warped_bgr = _match_low_freq_cheek_boundary(
        warped_bgr, target_bgr, active_mask[:, :, 0] if active_mask.ndim == 3 else active_mask, bbox
    )
    
    blended = warped_bgr.astype(np.float32) * active_mask + target_bgr.astype(np.float32) * (1.0 - active_mask)
    return np.clip(blended, 0, 255).astype(np.uint8)


def detect_faces_robust(app_or_bgr, img_bgr: np.ndarray = None) -> list:
    """Thread-safe multi-scale, multi-threshold face detector for difficult lighting or angles (Bug B7)."""
    if img_bgr is None:
        bgr_image = app_or_bgr
        app = get_face_app()
    else:
        app = app_or_bgr if app_or_bgr is not None else get_face_app()
        bgr_image = img_bgr

    if app is None or bgr_image is None or getattr(bgr_image, "size", 0) == 0:
        return []

    with _DETECTOR_LOCK:
        faces = app.get(bgr_image)
        if faces:
            return faces

        det_model = getattr(app, "det_model", None)
        if det_model is None:
            return faces or []

        orig_det_thresh = getattr(det_model, "det_thresh", 0.5)
        orig_det_size = getattr(det_model, "det_size", (640, 640))
        try:
            det_model.det_size = (1024, 1024)
            faces = app.get(bgr_image)
            if faces:
                return faces

            det_model.det_thresh = 0.35
            faces = app.get(bgr_image)
            if faces:
                return faces

            det_model.det_size = (512, 512)
            faces = app.get(bgr_image)
            return faces or []
        finally:
            try:
                det_model.det_thresh = orig_det_thresh
                det_model.det_size = orig_det_size
            except Exception:
                pass

def extract_face_skin_profile(face_bgr: np.ndarray, face_bbox: list) -> tuple[np.ndarray, np.ndarray]:
    """Extracts reference skin distribution (median L*, mean a*, b*, and std in CIE-LAB) trimmed to [15, 85] percentiles."""
    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    h_img, w_img = face_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    fw, fh = x2 - x1, y2 - y1

    if fw <= 0 or fh <= 0:
        return np.array([128.0, 128.0, 128.0], dtype=np.float32), np.array([10.0, 5.0, 5.0], dtype=np.float32)

    # 3-zone sampling coordinates avoiding eyes, brows, nostrils, and mouth
    zones = [
        (int(x1 + fw * 0.30), int(y1 + fh * 0.12), int(x1 + fw * 0.70), int(y1 + fh * 0.28)),
        (int(x1 + fw * 0.15), int(y1 + fh * 0.50), int(x1 + fw * 0.38), int(y1 + fh * 0.72)),
        (int(x1 + fw * 0.62), int(y1 + fh * 0.50), int(x1 + fw * 0.85), int(y1 + fh * 0.72)),
    ]

    sampled_bgr_list = []
    for zx1, zy1, zx2, zy2 in zones:
        zx1, zy1 = max(0, zx1), max(0, zy1)
        zx2, zy2 = min(w_img, zx2), min(h_img, zy2)
        if zx2 > zx1 and zy2 > zy1:
            crop = face_bgr[zy1:zy2, zx1:zx2]
            if crop.size > 0:
                sampled_bgr_list.append(crop.reshape(-1, 3))

    if sampled_bgr_list:
        sampled_bgr = np.concatenate(sampled_bgr_list, axis=0)
    else:
        sampled_bgr = face_bgr[y1:y2, x1:x2].reshape(-1, 3)

    if sampled_bgr.size == 0:
        return np.array([128.0, 128.0, 128.0], dtype=np.float32), np.array([10.0, 5.0, 5.0], dtype=np.float32)

    bgr_1d = sampled_bgr.reshape(-1, 1, 3)
    ycrcb = cv2.cvtColor(bgr_1d, cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
    gray = cv2.cvtColor(bgr_1d, cv2.COLOR_BGR2GRAY).reshape(-1)

    cr = ycrcb[:, 1]
    cb = ycrcb[:, 2]
    # Filter valid facial skin pixels using YCrCb (Cr in [133, 175], Cb in [77, 129])
    skin_chroma = (cr >= 133) & (cr <= 175) & (cb >= 77) & (cb <= 129)

    shadow_floor = max(12, int(np.percentile(gray, 10))) if len(gray) > 10 else 12
    valid_mask = skin_chroma & (gray >= shadow_floor) & (gray <= 235)

    if np.count_nonzero(valid_mask) < 50:
        valid_mask = (gray >= shadow_floor) & (gray <= 235)

    if np.count_nonzero(valid_mask) == 0:
        valid_mask = np.ones(len(gray), dtype=bool)

    lab_pixels = cv2.cvtColor(bgr_1d, cv2.COLOR_BGR2LAB).reshape(-1, 3)[valid_mask]

    # Trim outliers outside the [15, 85] percentile range on L, a, and b to reject specular glare and deep shadows
    if len(lab_pixels) > 10:
        l15, l85 = np.percentile(lab_pixels[:, 0], [15, 85])
        a15, a85 = np.percentile(lab_pixels[:, 1], [15, 85])
        b15, b85 = np.percentile(lab_pixels[:, 2], [15, 85])
        trim_mask = (
            (lab_pixels[:, 0] >= l15) & (lab_pixels[:, 0] <= l85) &
            (lab_pixels[:, 1] >= a15) & (lab_pixels[:, 1] <= a85) &
            (lab_pixels[:, 2] >= b15) & (lab_pixels[:, 2] <= b85)
        )
        if np.count_nonzero(trim_mask) >= 10:
            lab_pixels = lab_pixels[trim_mask]

    med_l = float(np.median(lab_pixels[:, 0]))
    if len(lab_pixels) >= 20:
        a_low, a_high = np.percentile(lab_pixels[:, 1], [15, 85])
        b_low, b_high = np.percentile(lab_pixels[:, 2], [15, 85])
        a_trim = lab_pixels[:, 1][(lab_pixels[:, 1] >= a_low) & (lab_pixels[:, 1] <= a_high)]
        b_trim = lab_pixels[:, 2][(lab_pixels[:, 2] >= b_low) & (lab_pixels[:, 2] <= b_high)]
        mean_a = float(np.mean(a_trim)) if len(a_trim) > 0 else float(np.mean(lab_pixels[:, 1]))
        mean_b = float(np.mean(b_trim)) if len(b_trim) > 0 else float(np.mean(lab_pixels[:, 2]))
    else:
        mean_a = float(np.mean(lab_pixels[:, 1]))
        mean_b = float(np.mean(lab_pixels[:, 2]))
    ref_mean = np.array([med_l, mean_a, mean_b], dtype=np.float32)

    std_l = float(np.std(lab_pixels[:, 0]))
    std_a = float(np.std(lab_pixels[:, 1]))
    std_b = float(np.std(lab_pixels[:, 2]))
    ref_std = np.array([std_l, std_a, std_b], dtype=np.float32) + 1e-6

    return ref_mean, ref_std


def match_skin_tone_lab(source_img_bgr: np.ndarray, target_img_bgr: np.ndarray, source_mask: np.ndarray = None, target_mask: np.ndarray = None) -> np.ndarray:
    """
    Transfers the Lab color distribution (mean and std) from the input source skin 
    to the generated target skin (body/face) to prevent skin lightening bias.
    """
    source_lab = cv2.cvtColor(source_img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Compute stats only on valid skin mask pixels if masks are provided
    src_pixels = source_lab[source_mask > 0] if (source_mask is not None and np.any(source_mask > 0)) else source_lab.reshape(-1, 3)
    tgt_pixels = target_lab[target_mask > 0] if (target_mask is not None and np.any(target_mask > 0)) else target_lab.reshape(-1, 3)

    src_mean, src_std = np.mean(src_pixels, axis=0), np.std(src_pixels, axis=0) + 1e-6
    tgt_mean, tgt_std = np.mean(tgt_pixels, axis=0), np.std(tgt_pixels, axis=0) + 1e-6

    # Apply Reinhard color transfer in Lab space
    result_lab = target_lab.copy()
    for c in range(3):
        # Moderate the L-channel (luminance) std scaling slightly to preserve studio lighting shadows
        scale = (src_std[c] / tgt_std[c]) * (0.75 if c == 0 else 1.0)
        result_lab[:, :, c] = (result_lab[:, :, c] - tgt_mean[c]) * scale + src_mean[c]

    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    matched_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    # Blend back using a feathered target mask if provided
    if target_mask is not None and np.any(target_mask > 0):
        feathered_mask = cv2.GaussianBlur(target_mask.astype(np.float32) / 255.0, (31, 31), 0)[..., None]
        return (matched_bgr * feathered_mask + target_img_bgr * (1.0 - feathered_mask)).astype(np.uint8)
    
    return matched_bgr


def get_clean_face_hull_mask(image_shape: tuple, landmarks_2d: np.ndarray, erode_ratio: float = 0.08) -> np.ndarray:
    """
    Creates an inward-eroded, smoothly feathered convex hull mask from facial landmarks
    to completely eliminate square bounding-box lines and grid artifacts on cheek/jawline.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(landmarks_2d.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)

    face_width = float(np.max(landmarks_2d[:, 0]) - np.min(landmarks_2d[:, 0]))
    erode_size = max(5, int(face_width * erode_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
    mask = cv2.erode(mask, kernel, iterations=1)

    blur_size = (erode_size * 2 + 1) | 1
    feathered_mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (blur_size, blur_size), 0)
    return feathered_mask[:, :, np.newaxis]


def seamless_face_blend(swapped_face_bgr: np.ndarray, body_scene_bgr: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
    """
    Seamlessly blends swapped face into body scene eliminating hard jawline edges
    and cheek/ear boundary artifacts while preserving authentic skin tones.
    """
    # If face_mask is already a normalized float feather mask [0, 1]
    if face_mask.dtype in (np.float32, np.float64):
        feathered = face_mask if face_mask.ndim == 3 else face_mask[:, :, np.newaxis]
        feathered = np.clip(feathered, 0.0, 1.0)
    else:
        # Erode and smooth the mask to avoid boundary artifacts near the ears/jaw
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        clean_mask = cv2.erode(face_mask.astype(np.uint8), kernel, iterations=1)
        feathered = cv2.GaussianBlur(clean_mask.astype(np.float32) / 255.0, (25, 25), 0)[:, :, np.newaxis]

    # Subtle monochromatic grain on luminance to prevent digital plastic sheen
    noise = np.random.normal(0, 1.2, (swapped_face_bgr.shape[0], swapped_face_bgr.shape[1], 1)).astype(np.float32)
    textured_face = np.clip(swapped_face_bgr.astype(np.float32) + noise * feathered, 0, 255)

    return (textured_face * feathered + body_scene_bgr.astype(np.float32) * (1.0 - feathered)).clip(0, 255).astype(np.uint8)


def calibrate_swapped_face_tone(
    swapped_bgr: np.ndarray,
    src_bgr: np.ndarray,
    tgt_bbox: list,
    src_bbox: list,
    ref_bgr: np.ndarray = None,
    src_kps: np.ndarray = None,
    tgt_kps: np.ndarray = None,
    tgt_lmks: np.ndarray = None
) -> np.ndarray:
    """Calibrate swapped face melanin and undertones with wide feathered ellipse & continuous face_skin_gate (Bug B5)."""
    try:
        ref_mean, ref_std = extract_face_skin_profile(src_bgr, src_bbox)
        tgt_mean, tgt_std = extract_face_skin_profile(swapped_bgr, tgt_bbox)

        src_l = float(ref_mean[0])
        src_a = float(ref_mean[1])
        src_b = float(ref_mean[2])

        tgt_l = float(tgt_mean[0])
        tgt_a = float(tgt_mean[1])
        tgt_b = float(tgt_mean[2])

        x1, y1, x2, y2 = [int(v) for v in tgt_bbox]
        h, w = swapped_bgr.shape[:2]
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)

        # Wide feathered ellipse (rx = 0.66 * bw, ry = 0.62 * bh) so ears are calibrated alongside face
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        rx = int(0.66 * bw)
        ry = int(0.62 * bh)

        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
        ksize = max(15, (int(bw * 0.12) // 2) * 2 + 1)
        sigma = max(5.0, float(bw * 0.045))
        mask = cv2.GaussianBlur(mask, (ksize, ksize), sigma)

        # Smooth per-pixel YCrCb/HSV face_skin_gate (Cr in [132, 178], Cb in [75, 132], L >= 30)
        ycrcb = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        lum = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
        face_skin_gate = ((cr >= 132) & (cr <= 178) & (cb >= 75) & (cb <= 132) & (lum >= 30)).astype(np.float32)
        face_skin_gate = cv2.GaussianBlur(face_skin_gate, (15, 15), 0)
        mask = (mask * face_skin_gate)[:, :, np.newaxis]

        swapped_lab = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        calibrated_lab = swapped_lab.copy()

        # Exact ground-truth melanin chrominance deltas
        da = src_a - tgt_a
        db = src_b - tgt_b

        # Scene-lighting anchored luminance calibration
        if ref_bgr is not None:
            scene_mean, _ = extract_face_skin_profile(ref_bgr, tgt_bbox)
            scene_l = float(scene_mean[0])
            if ref_bgr is swapped_bgr or (ref_bgr.shape == swapped_bgr.shape and np.array_equal(ref_bgr, swapped_bgr)):
                dL = src_l - tgt_l
            else:
                ambient_delta = np.clip(scene_l - 138.0, -20.0, 20.0) * 0.10
                target_l = np.clip(src_l + ambient_delta, 25.0, 245.0)
                dL = (target_l - tgt_l) * 0.95
        else:
            dL = float(np.clip(src_l - tgt_l, -20.0, 20.0))

        # Specular Highlight & Deep Shadow Protection
        curr_l = swapped_lab[:, :, 0]
        chroma = np.sqrt((swapped_lab[:, :, 1] - 128.0)**2 + (swapped_lab[:, :, 2] - 128.0)**2)
        specular_mask = np.clip((curr_l - 195.0) / 45.0, 0.0, 1.0) * np.clip((24.0 - chroma) / 14.0, 0.0, 1.0)
        specular_attenuation = np.clip((245.0 - curr_l) / 50.0, 0.15, 1.0) * (1.0 - 0.75 * specular_mask)
        shadow_attenuation = np.clip((curr_l - 32.0) / 40.0, 0.20, 1.0)
        tone_weight = specular_attenuation * shadow_attenuation

        # Soft chrominance standard deviation scaling
        scale_a = float(np.clip(ref_std[1] / max(1.0, tgt_std[1]), 0.85, 1.15))
        scale_b = float(np.clip(ref_std[2] / max(1.0, tgt_std[2]), 0.85, 1.15))

        calibrated_lab[:, :, 0] += dL * mask[:, :, 0] * specular_attenuation
        calibrated_lab[:, :, 1] += (da + (scale_a - 1.0) * (calibrated_lab[:, :, 1] - tgt_a)) * mask[:, :, 0] * tone_weight
        calibrated_lab[:, :, 2] += (db + (scale_b - 1.0) * (calibrated_lab[:, :, 2] - tgt_b)) * mask[:, :, 0] * tone_weight

        calibrated_bgr = cv2.cvtColor(np.clip(calibrated_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        result = (calibrated_bgr.astype(np.float32) * mask + swapped_bgr.astype(np.float32) * (1.0 - mask)).clip(0, 255).astype(np.uint8)
        return result
    except Exception as e:
        print(f"[FaceEngine] Face tone calibration note: {e}")
        return swapped_bgr


def restore_scene_hair(restored_bgr: np.ndarray, scene_bgr: np.ndarray, face_bbox: list) -> np.ndarray:
    """Restore generated foreground hair without treating dark skin as hair (Bug B4)."""
    try:
        from core.clothing_engine import parse_clothing_labels
        scene = Image.fromarray(cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2RGB))
        labels = parse_clothing_labels(scene)
    except Exception as error:
        print(f"[FaceEngine] Hair recovery skipped without segmentation: {error}")
        return restored_bgr

    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    h_img, w_img = restored_bgr.shape[:2]

    # Inner facial ellipse: rx = 0.36 * bw, ry = 0.38 * bh centered at (cx, cy + 0.04 * bh)
    inner_face_oval = np.zeros((h_img, w_img), dtype=bool)
    cv2.ellipse(
        inner_face_oval.view(np.uint8),
        (int(cx), int(cy + 0.04 * bh)),
        (int(0.36 * bw), int(0.38 * bh)),
        0, 0, 360, 1, -1
    )

    ycrcb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    is_skin = inner_face_oval & (cr >= 130) & (cr <= 180) & (cb >= 75) & (cb <= 135)
    is_face_seg = (labels == 11)
    hair_mask = (labels == 2) & ~(is_skin | is_face_seg)

    alpha = feather_region(hair_mask, max(2.0, min(8.0, bw * 0.02)))[:, :, None]
    blended = restored_bgr.astype(np.float32) * (1.0 - alpha) + scene_bgr.astype(np.float32) * alpha
    return np.rint(np.clip(blended, 0, 255)).astype(np.uint8)


def apply_face_lock(
    source_image: Image.Image,
    target_image: Image.Image,
    fidelity_weight: float = 0.45,
    enable_pores: bool = False,
    enable_grain: bool = False,
    return_status: bool = False,
    src_faces: list = None
):
    """Transfer identity, restore the face and retain scene lighting and semantic hair (Bugs B1, B3)."""
    app, swapper, codeformer_net, face_helper, fallback_restorer = get_face_modules()

    # Orientation safe conversion
    src_rgb = np.array(ImageOps.exif_transpose(source_image).convert("RGB"))
    src_bgr = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR)

    tgt_rgb = np.array(ImageOps.exif_transpose(target_image).convert("RGB"))
    tgt_bgr = cv2.cvtColor(tgt_rgb, cv2.COLOR_RGB2BGR)

    if src_faces is None:
        src_faces = detect_faces_robust(app, src_bgr)
    tgt_faces = detect_faces_robust(app, tgt_bgr)

    if not src_faces or not tgt_faces:
        return (target_image, False) if return_status else target_image

    src_face = sorted(src_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[0]
    tgt_face = sorted(tgt_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[0]

    # Native direct swap without artificial head-roll patch rotation (Bug B3)
    tgt_face_for_swap = tgt_face
    tgt_bgr_for_swap = tgt_bgr.copy()

    fx1, fy1, fx2, fy2 = [int(v) for v in tgt_face_for_swap.bbox]
    fh_w, fh_h = fx2 - fx1, fy2 - fy1

    # Step 1: Transfer Biometric Facial Identity using Inswapper with crop-safe boundary warp (Bug B1)
    bgr_fake, M = swapper.get(tgt_bgr_for_swap.copy(), tgt_face_for_swap, src_face, paste_back=False)

    lmks = getattr(tgt_face_for_swap, 'landmark_2d_106', None)
    if lmks is None and hasattr(tgt_face_for_swap, 'landmark_3d_68') and tgt_face_for_swap.landmark_3d_68 is not None:
        lmks = tgt_face_for_swap.landmark_3d_68[:, :2]
    if lmks is None:
        lmks = getattr(tgt_face_for_swap, 'kps', None)

    if lmks is not None and len(lmks) >= 5:
        face_blend_mask = get_clean_face_hull_mask(tgt_bgr_for_swap.shape, lmks, erode_ratio=0.08)
    else:
        face_blend_mask = None

    swapped_bgr = warp_back_128(
        bgr_fake=bgr_fake,
        M=M,
        target_bgr=tgt_bgr_for_swap,
        hull_mask=face_blend_mask,
        bbox=tgt_face_for_swap.bbox
    )

    # Step 2: CodeFormer Discrete Codebook Prior with balanced fidelity
    restored_bgr = swapped_bgr
    is_small_face = fh_h < 150
    effective_fidelity = min(0.50, max(0.20, float(fidelity_weight)))
    if is_small_face:
        effective_fidelity = min(0.50, effective_fidelity + 0.05)

    if codeformer_net is not None and face_helper is not None:
        with _CODEFORMER_LOCK:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                from torchvision.transforms.functional import normalize
                from codeformer.basicsr.utils import img2tensor, tensor2img

                codeformer_net.to(device)
                if hasattr(face_helper, "face_detector") and face_helper.face_detector is not None:
                    face_helper.face_detector.to(device)
                if hasattr(face_helper, "face_parse") and face_helper.face_parse is not None:
                    face_helper.face_parse.to(device)
                face_helper.device = device

                face_helper.clean_all()
                face_helper.read_image(swapped_bgr)
                face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
                face_helper.align_warp_face()

                if face_helper.cropped_faces:
                    for cropped_face in face_helper.cropped_faces:
                        t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
                        normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                        t = t.unsqueeze(0).to(device)
                        with torch.no_grad():
                            out = codeformer_net(t, w=effective_fidelity)[0]
                            restored_face = tensor2img(out, rgb2bgr=True, min_max=(-1, 1))
                        face_helper.add_restored_face(restored_face)

                    face_helper.get_inverse_affine(None)
                    restored_bgr = face_helper.paste_faces_to_input_image()

                    # Seamless face blend with landmark convex hull feathering
                    if face_blend_mask is None:
                        if lmks is not None and len(lmks) >= 5:
                            face_blend_mask = get_clean_face_hull_mask(restored_bgr.shape, lmks, erode_ratio=0.08)
                        else:
                            tx1, ty1, tx2, ty2 = [int(v) for v in tgt_face_for_swap.bbox]
                            tw_f = max(1, tx2 - tx1)
                            th_f = max(1, ty2 - ty1)
                            face_blend_mask = np.zeros((restored_bgr.shape[0], restored_bgr.shape[1]), dtype=np.uint8)
                            fcx = int((tx1 + tx2) / 2)
                            fcy = int((ty1 + ty2) / 2)
                            cv2.ellipse(face_blend_mask, (fcx, fcy), (int(tw_f * 0.48), int(th_f * 0.52)), 0, 0, 360, 255, -1)

                    restored_bgr = seamless_face_blend(
                        swapped_face_bgr=restored_bgr,
                        body_scene_bgr=tgt_bgr_for_swap,
                        face_mask=face_blend_mask
                    )
                else:
                    restored_bgr = swapped_bgr
            except Exception as e:
                print(f"[FaceEngine] CodeFormer restoration warning: {e}")
                restored_bgr = swapped_bgr
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
        try:
            _, _, fb_bgr = fallback_restorer.enhance(
                swapped_bgr,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )
            restored_bgr = fb_bgr
        except Exception as e:
            print(f"[FaceEngine] GFPGAN fallback warning: {e}")
            restored_bgr = swapped_bgr

    # Step 3: Definitive Ground-Truth Tone Calibration
    tgt_lmks_tone = getattr(tgt_face_for_swap, 'landmark_2d_106', None)
    if tgt_lmks_tone is None and hasattr(tgt_face_for_swap, 'landmark_3d_68') and tgt_face_for_swap.landmark_3d_68 is not None:
        tgt_lmks_tone = tgt_face_for_swap.landmark_3d_68[:, :2]
    restored_bgr = calibrate_swapped_face_tone(
        swapped_bgr=restored_bgr,
        src_bgr=src_bgr,
        tgt_bbox=tgt_face_for_swap.bbox,
        src_bbox=src_face.bbox,
        ref_bgr=tgt_bgr_for_swap,
        src_kps=getattr(src_face, "kps", None),
        tgt_kps=getattr(tgt_face_for_swap, "kps", None),
        tgt_lmks=tgt_lmks_tone
    )

    # Step 3.5: Dual-Source DSLR micro-pores & epidermal texture recovery on tone-stabilized face
    if enable_pores:
        try:
            x1_t, y1_t, x2_t, y2_t = [int(v) for v in tgt_face_for_swap.bbox]
            bw_t, bh_t = x2_t - x1_t, y2_t - y1_t
            h_curr, w_curr = tgt_bgr_for_swap.shape[:2]
            ny1_b = min(h_curr - 10, y2_t)
            ny2_b = min(h_curr, int(y2_t + bh_t * 0.45))
            nx1_b = max(0, int(x1_t + bw_t * 0.15))
            nx2_b = min(w_curr, int(x2_t - bw_t * 0.15))
            body_var = None
            if ny2_b > ny1_b + 10 and nx2_b > nx1_b + 10:
                neck_c = tgt_bgr_for_swap[ny1_b:ny2_b, nx1_b:nx2_b]
                if neck_c.size > 0:
                    ycrcb_n = cv2.cvtColor(neck_c, cv2.COLOR_BGR2YCrCb)
                    gray_n = cv2.cvtColor(neck_c, cv2.COLOR_BGR2GRAY)
                    skin_n = (ycrcb_n[:, :, 1] >= 133) & (ycrcb_n[:, :, 1] <= 175) & (ycrcb_n[:, :, 2] >= 77) & (ycrcb_n[:, :, 2] <= 130) & (gray_n >= 50)
                    if np.count_nonzero(skin_n) >= 150:
                        lap_neck = cv2.Laplacian(gray_n, cv2.CV_64F)
                        body_var = float(np.var(lap_neck[skin_n]))

            restored_bgr = reinject_micro_pores(
                base_bgr=restored_bgr,
                ref_bgr=tgt_bgr_for_swap,
                bbox=tgt_face_for_swap.bbox,
                src_bgr=src_bgr,
                src_bbox=src_face.bbox,
                src_kps=getattr(src_face, "kps", None),
                tgt_kps=getattr(tgt_face_for_swap, "kps", None),
                pore_strength=1.0,
                body_var=body_var
            )
        except Exception as e:
            print(f"[FaceEngine] Micro-texture enhancement warning: {e}")

    # Restore semantic foreground hair
    restored_bgr = restore_scene_hair(restored_bgr, tgt_bgr, tgt_face.bbox)

    # Step 3.8: Targeted Sensory Landmark Clarity
    restored_bgr = enhance_facial_clarity(restored_bgr, tgt_face.bbox, kps=getattr(tgt_face, "kps", None))

    # Step 4: Photographic 35mm Micro-Grain
    if enable_grain:
        try:
            restored_bgr = inject_photographic_grain(restored_bgr, intensity=0.005)
        except Exception as e:
            print(f"[FaceEngine] Film grain warning: {e}")

    final_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
    res_img = Image.fromarray(final_rgb)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (res_img, True) if return_status else res_img

