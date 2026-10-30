"""Garment masks with semantic foreground, face-contour and hand protection.

Mask estimates require visual review when changing garment silhouette or coverage.
"""
import os
import re
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    CLIPSegProcessor,
    CLIPSegForImageSegmentation
)
from pathlib import Path

from core import config
from core.skin_engine import feather_region
from core.inpaint_regions import face_contour


def normalize_input_image(image: Image.Image, max_dim: int = 2048) -> Image.Image:
    """Normalize input image orientation, color space (RGB), and bounds."""
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        image = image.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.LANCZOS)
    return image

_clothing_proc = None
_clothing_model = None
_clip_proc = None
_clip_model = None


def get_clothing_parser():
    """Lazily load Segformer B2 clothing parser from local assets."""
    global _clothing_proc, _clothing_model
    if _clothing_proc is None or _clothing_model is None:
        _clothing_proc = SegformerImageProcessor.from_pretrained(
            str(config.CLOTHING_PARSER_DIR),
            local_files_only=True
        )
        _clothing_model = SegformerForSemanticSegmentation.from_pretrained(
            str(config.CLOTHING_PARSER_DIR),
            local_files_only=True,
            use_safetensors=True
        ).eval()
    return _clothing_proc, _clothing_model


def get_clipseg():
    """Lazily load CLIPSeg zero-shot segmentation model from local assets."""
    global _clip_proc, _clip_model
    if not config.CLIPSEG_DIR.is_dir():
        raise FileNotFoundError(f"Optional CLIPSeg assets unavailable at {config.CLIPSEG_DIR}; using SegFormer")
    if _clip_proc is None or _clip_model is None:
        _clip_proc = CLIPSegProcessor.from_pretrained(
            str(config.CLIPSEG_DIR),
            local_files_only=True
        )
        _clip_model = CLIPSegForImageSegmentation.from_pretrained(
            str(config.CLIPSEG_DIR),
            local_files_only=True,
            use_safetensors=True
        ).eval()
    return _clip_proc, _clip_model


def detect_face_landmarks(image: Image.Image) -> list:
    """Detect face landmarks using YuNet ONNX detector or InsightFace fallback for chin line protection."""
    if config.FACE_DETECTOR_PATH.exists():
        try:
            scale = min(1.0, 1280 / max(image.size))
            preview = image.resize(
                (max(32, round(image.width * scale)), max(32, round(image.height * scale))),
                Image.Resampling.LANCZOS
            )
            pixels = cv2.cvtColor(np.asarray(preview.convert("RGB")), cv2.COLOR_RGB2BGR)
            detector = cv2.FaceDetectorYN.create(str(config.FACE_DETECTOR_PATH), "", preview.size, 0.55, 0.3, 1000)
            _, faces = detector.detect(pixels)
            if faces is not None:
                results = []
                for face in faces:
                    x, y, w, h = face[:4]
                    if w < 12 or h < 12:
                        continue
                    unscaled = [float(v / scale) for v in face[:14]]
                    eye_y = (unscaled[5] + unscaled[7]) / 2
                    mouth_y = (unscaled[11] + unscaled[13]) / 2
                    chin_y = mouth_y + (mouth_y - eye_y) * 0.65
                    results.append({
                        "box": (unscaled[0], unscaled[1], unscaled[2], unscaled[3]),
                        "chin_y": chin_y,
                        "eye_y": eye_y,
                        "mouth_y": mouth_y
                    })
                if results:
                    results.sort(key=lambda f: f["box"][2] * f["box"][3], reverse=True)
                    return results
        except Exception as e:
            print(f"[ClothingEngine] YuNet detector note: {e}")

    # Fallback to InsightFace when YuNet ONNX is missing or encounters issues
    try:
        from core.face_engine import get_face_modules, detect_faces_robust
        app, _, _, _, _ = get_face_modules()
        rgb = np.asarray(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = detect_faces_robust(app, bgr)
        if faces:
            results = []
            for face in faces:
                x1, y1, x2, y2 = face.bbox
                w, h = x2 - x1, y2 - y1
                kps = face.kps
                eye_y = (kps[0][1] + kps[1][1]) / 2.0
                mouth_y = (kps[3][1] + kps[4][1]) / 2.0
                chin_y = mouth_y + (mouth_y - eye_y) * 0.65
                results.append({
                    "box": (float(x1), float(y1), float(w), float(h)),
                    "chin_y": float(chin_y),
                    "eye_y": float(eye_y),
                    "mouth_y": float(mouth_y)
                })
            results.sort(key=lambda f: f["box"][2] * f["box"][3], reverse=True)
            return results
    except Exception as e:
        print(f"[ClothingEngine] Landmark detection fallback note: {e}")

    return []


def parse_clothing_labels(image: Image.Image) -> np.ndarray:
    """Classify pixels into clothing categories using SegFormer with memory-efficient resolution-capped interpolation."""
    if hasattr(image, "_cached_clothing_labels") and getattr(image, "_cached_clothing_labels") is not None:
        cached = image._cached_clothing_labels
        if cached.shape[:2] == (image.height, image.width):
            return cached.copy()
        return cv2.resize(cached, (image.width, image.height), interpolation=cv2.INTER_NEAREST)

    processor, model = get_clothing_parser()
    orig_w, orig_h = image.width, image.height

    # Cap interpolation target to max 1024 to slash 18-channel float32 logits RAM from 432MB to 56MB
    max_dim = max(orig_w, orig_h)
    if max_dim > 1024:
        scale = 1024.0 / max_dim
        interp_w = max(64, int(round(orig_w * scale)))
        interp_h = max(64, int(round(orig_h * scale)))
    else:
        interp_w, interp_h = orig_w, orig_h

    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
        logits = torch.nn.functional.interpolate(
            logits, size=(interp_h, interp_w), mode="bilinear", align_corners=False
        )
        labels_downscaled = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        del logits

    if (interp_w, interp_h) != (orig_w, orig_h):
        labels = cv2.resize(labels_downscaled, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    else:
        labels = labels_downscaled

    try:
        image._cached_clothing_labels = labels
    except Exception:
        pass

    return labels


def segment_with_clipseg(image: Image.Image, query: str = "dress, clothing") -> np.ndarray:
    """Zero-shot segmentation fallback with CLIPSeg and on-demand cleanup."""
    try:
        proc, model = get_clipseg()
        inputs = proc(text=[query], images=[image], padding=True, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**inputs).logits.sigmoid().numpy()
        clip_2d = (logits[0] > 0.35).astype(np.uint8) * 255
        del logits, inputs
        resized = np.array(Image.fromarray(clip_2d).resize(image.size, Image.Resampling.BILINEAR))
        return (resized > 127).astype(np.uint8)
    except Exception:
        return np.zeros((image.height, image.width), dtype=np.uint8)


def refine_foreground_hair(image: Image.Image, labels: np.ndarray, chin_y: float) -> np.ndarray:
    """Remove confidently clothing-colored pixels from hair below the head."""
    hair = labels == 2
    yy = np.arange(labels.shape[0])[:, None]
    head_hair = hair & (yy < chin_y)
    garments = np.isin(labels, (4, 5, 6, 7, 8, 17))
    if np.count_nonzero(head_hair) < 100 or np.count_nonzero(garments) < 100:
        return hair
    lab = cv2.cvtColor(np.array(image.convert("RGB"), dtype=np.float32)/255.0, cv2.COLOR_RGB2LAB)
    def palette(region):
        points = lab[region][::max(1, int(np.count_nonzero(region)//2000))]
        centers = [np.median(points, axis=0)]
        for _ in range(3):
            distances = np.min(np.sum((points[:, None] - np.array(centers))**2, axis=2), axis=1)
            centers.append(points[np.argmax(distances)])
        return np.asarray(centers)
    hair_palette, cloth_palette = palette(head_hair), palette(garments)
    candidates = hair & (yy >= chin_y)
    points = lab[candidates]
    hair_distance = np.sqrt(np.min(np.sum((points[:, None]-hair_palette)**2, axis=2), axis=1))
    cloth_distance = np.sqrt(np.min(np.sum((points[:, None]-cloth_palette)**2, axis=2), axis=1))
    refined = hair.copy()
    refined[candidates] = ~(cloth_distance + 10.0 < hair_distance)
    return refined


def generate_fused_dress_mask(
    image: Image.Image,
    target: str = "dress",
    prompt: str = None,
    keep_face: bool = True,
    expand_pixels: int = 14,
    feather_sigma: float = 3.0,
    return_flux_image: bool = False
) -> tuple:
    """Return a garment alpha, overlay, detected faces and mask coverage."""
    img_rgb = ImageOps.exif_transpose(image).convert("RGB")
    orig_w, orig_h = img_rgb.size

    # Cap working resolution to 2048px to prevent 4K/6K mobile sensor latency spikes
    max_dim = max(orig_w, orig_h)
    if max_dim > 2048:
        scale = 2048.0 / max_dim
        work_w = int(round(orig_w * scale))
        work_h = int(round(orig_h * scale))
        img_work = img_rgb.resize((work_w, work_h), Image.Resampling.LANCZOS)
    else:
        scale = 1.0
        work_w, work_h = orig_w, orig_h
        img_work = img_rgb

    w, h = work_w, work_h

    # Bidirectional garment coverage analysis from prompt
    prompt_lower = (prompt or "").lower()

    # Negative guards for sleeveless / bare arms
    is_explicitly_sleeveless = bool(re.search(
        r"\b(sleeveless|strapless|off[- ]shoulder|tank\b|tank top|camisole|tube top|spaghetti strap|halter|bare arm|bare arms|bare shoulder|bare shoulders)\b",
        prompt_lower
    ))

    # Sleeve keyword matching with word boundaries
    has_sleeve_kw = bool(re.search(
        r"\b(long[- ]sleeve[s]?|full[- ]sleeve[s]?|quarter[- ]sleeve[s]?|three[- ]quarter|jacket|blazer|cardigan|coat|suit|robe|kimono)\b",
        prompt_lower
    )) or (bool(re.search(r"\b(sleeves|sleeve)\b", prompt_lower)) and not is_explicitly_sleeveless)

    wants_sleeves = has_sleeve_kw and not is_explicitly_sleeveless

    # High collar detection
    wants_high_collar = bool(re.search(
        r"\b(turtleneck|high collar|mandarin collar|choker|polo neck|mock neck)\b",
        prompt_lower
    ))

    # Leg coverage detection (pants, long skirts, ballgowns)
    wants_leg_coverage = bool(re.search(
        r"\b(pants|trousers|jeans|leggings|sweatpants|tights|maxi|floor[- ]length|long gown|ballgown)\b",
        prompt_lower
    )) and (target not in ("upper", "top", "shirt"))

    # 1. SegFormer Base
    if hasattr(image, "_cached_clothing_labels") and getattr(image, "_cached_clothing_labels") is not None:
        cached = image._cached_clothing_labels
        if cached.shape[:2] == (work_h, work_w):
            labels = cached.copy()
        else:
            labels = cv2.resize(cached, (work_w, work_h), interpolation=cv2.INTER_NEAREST)
    else:
        labels = parse_clothing_labels(img_work)
        if hasattr(image, "_cached_clothing_labels") is False or image._cached_clothing_labels is None:
            try:
                if labels.shape[:2] != (orig_h, orig_w):
                    image._cached_clothing_labels = cv2.resize(labels, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                else:
                    image._cached_clothing_labels = labels.copy()
            except Exception:
                pass

    # ATR SegFormer label indices:
    # 0: Background, 1: Hat, 2: Hair, 3: Sunglasses, 4: Upper-clothes, 5: Skirt,
    # 6: Pants, 7: Dress, 8: Belt, 9: Left-shoe, 10: Right-shoe, 11: Face,
    # 12: Left-leg, 13: Right-leg, 14: Left-arm, 15: Right-arm, 16: Bag, 17: Scarf

    if target in ("upper", "top", "shirt"):
        garment_ids = (4, 17)
        queries = ["shirt", "top", "blouse", "collar", "neckline", "jacket", "vest", "sweater"]
        forbidden_labels = (5, 6, 7, 8, 9, 10, 12, 13)  # Strictly isolate lower garments & legs & shoes
    elif target in ("lower", "bottom", "pants", "skirt"):
        garment_ids = (5, 6, 8)
        queries = ["pants", "skirt", "trousers", "shorts", "jeans"]
        forbidden_labels = (4, 11, 14, 15, 17)  # Strictly isolate upper garments & arms & face
    else:  # dress, all, outfit
        garment_ids = (4, 5, 6, 7, 8, 17)
        queries = ["dress", "shirt", "blouse", "collar", "strap", "neckline", "gown", "skirt", "outfit", "sleeves", "ruffles"]
        forbidden_labels = (9, 10)  # Always protect shoes

    if wants_sleeves and "sleeves" not in queries:
        queries.append("sleeves")

    garment_base = np.isin(labels, garment_ids)
    forbidden_mask = np.isin(labels, forbidden_labels)

    # 2. CLIPSeg Zero-Shot Detection (without generic 'clothing' / 'fabric' to prevent bedsheet/sofa leaks)
    clip_mask_up = np.zeros((h, w), dtype=bool)
    clip_confident = np.zeros((h, w), dtype=bool)
    try:
        proc, model = get_clipseg()
        inputs = proc(text=queries, images=[img_work] * len(queries), padding=True, return_tensors="pt")
        with torch.no_grad():
            clip_probs = model(**inputs).logits.sigmoid().max(dim=0)[0].numpy()
        clip_probs_up = cv2.resize(clip_probs, (w, h), interpolation=cv2.INTER_LINEAR)
        clip_confident = clip_probs_up > 0.20
        clip_mask_up = clip_probs_up > 0.22
    except Exception as e:
        print(f"[ClothingEngine] CLIPSeg zero-shot warning: {e}")

    # 3. Face & Head Protection via YuNet Landmarks and SegFormer Face Segment
    faces_detected = detect_face_landmarks(img_work) if keep_face else []
    face_protect = np.zeros((h, w), dtype=bool)
    chin_y = 0.0

    seg_face = (labels == 11)

    if faces_detected and keep_face:
        f = faces_detected[0]
        bx, by, bw, bh = f["box"]
        chin_y = f["chin_y"]
        fcx = bx + bw / 2.0
        fcy = (by + chin_y) / 2.0
        frx = int(bw * 0.52)
        fry = int((chin_y - by) * 0.58)
        face_ellipse = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(face_ellipse, (int(fcx), int(fcy)), (frx, fry), 0, 0, 360, 255, -1)
        face_protect = (face_ellipse > 0)
        # Protect head, cheeks, chin envelope down to chin_y + (0 if wants_high_collar else 2)
        face_protect |= (np.arange(h)[:, None] <= (chin_y + (0 if wants_high_collar else 2))) & (np.abs(np.arange(w)[None, :] - fcx) <= (bw * 0.55))
        # Protect SegFormer face above chin_y + 4 strictly within the facial column (prevents mislabeled shoulders from being shielded)
        face_protect |= seg_face & (np.arange(h)[:, None] <= (chin_y + 4)) & (np.abs(np.arange(w)[None, :] - fcx) <= (bw * 0.65))
        # Confident clothing detections below chin are never suppressed as face
        face_protect &= ~((np.arange(h)[:, None] > chin_y) & clip_confident)
        try:
            contour = face_contour(img_work, f)
            if contour is not None:
                # Landmarks refine the jaw, not forehead/head protection. Their
                # hull starts near the eyebrows and cannot own the whole head.
                face_protect = contour | (face_protect & (np.arange(h)[:, None] <= f["mouth_y"]))
        except Exception as error:
            print(f"[ClothingEngine] Face contour unavailable: {error}")
    elif keep_face:
        face_y_indices = np.where(seg_face)[0]
        if len(face_y_indices) > 50:
            chin_y = float(np.percentile(face_y_indices, 88))
            face_protect = seg_face & (np.arange(h)[:, None] <= (chin_y + 2))
        else:
            face_protect = seg_face

    # Strict face safety barrier dilated by 7px
    face_safety_barrier = cv2.dilate(face_protect.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0

    # 4. Hair Protection: Protect all hair strands (labels == 2)
    hair = refine_foreground_hair(img_work, labels, chin_y) if chin_y > 0 else (labels == 2)
    hair_core = cv2.erode(hair.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0

    # 5. Arm Skin Core Protection via Adaptive Distance Transform
    arm_skin = np.isin(labels, (14, 15))
    mask_warnings = []
    if np.any(arm_skin):
        if not wants_sleeves:
            dist_map_arm = cv2.distanceTransform(arm_skin.astype(np.uint8), cv2.DIST_L2, 5)
            max_dist_arm = dist_map_arm.max() if dist_map_arm.size > 0 else 0
            adaptive_radius_arm = max(1, min(4, int(max_dist_arm * 0.12)))
            arm_core = dist_map_arm > adaptive_radius_arm
            
            # Exempt upper deltoid/shoulder joint (top 35% of each arm component) from arm_core
            # to allow full enveloping of puffed sleeves, straps, and armhole ruffles
            num_arm_comps, arm_comp_lbls, arm_stats, _ = cv2.connectedComponentsWithStats(arm_skin.astype(np.uint8))
            for aci in range(1, num_arm_comps):
                top_y = arm_stats[aci, cv2.CC_STAT_TOP]
                h_comp = arm_stats[aci, cv2.CC_STAT_HEIGHT]
                if h_comp > 40:
                    shoulder_cutoff = top_y + int(h_comp * 0.35)
                    arm_core &= ~((arm_comp_lbls == aci) & (np.arange(h)[:, None] < shoulder_cutoff))
        else:
            from core.hand_regions import hand_protection
            try:
                arm_core, detected_hands = hand_protection(img_work, labels)
            except Exception as error:
                print(f"[ClothingEngine] Hand landmarks unavailable: {error}")
                arm_core = np.zeros((h, w), dtype=bool)
                detected_hands = []
            num_arm_comps, arm_comp_lbls, arm_stats, _ = cv2.connectedComponentsWithStats(arm_skin.astype(np.uint8))
            for aci in range(1, num_arm_comps):
                component = arm_comp_lbls == aci
                if arm_stats[aci, cv2.CC_STAT_AREA] > 80 and not np.any(component & arm_core):
                    # Unknown hand placement must not be guessed from a vertical
                    # arm fraction (crossed/bent arms invalidate that assumption).
                    arm_core |= component
                    if not mask_warnings:
                        mask_warnings.append("A hand boundary could not be located; that exposed arm was preserved. Review sleeve coverage.")
    else:
        arm_core = np.zeros((h, w), dtype=bool)

    # 6. Leg Skin Core Protection via Adaptive Distance Transform
    leg_skin = np.isin(labels, (12, 13))
    if np.any(leg_skin) and not wants_leg_coverage:
        dist_map_leg = cv2.distanceTransform(leg_skin.astype(np.uint8), cv2.DIST_L2, 5)
        max_dist_leg = dist_map_leg.max() if dist_map_leg.size > 0 else 0
        adaptive_radius_leg = max(1, min(4, int(max_dist_leg * 0.10)))
        leg_core = (dist_map_leg > adaptive_radius_leg)
    else:
        leg_core = np.zeros((h, w), dtype=bool)

    # 7. Shoe / Footwear Protection
    shoes = np.isin(labels, (9, 10))
    if np.any(shoes):
        shoe_core = cv2.erode(shoes.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0 if shoes.sum() > 100 else shoes
    else:
        shoe_core = np.zeros((h, w), dtype=bool)

    # 8. Person Torso Silhouette Envelope (strictly clamps background bedsheet/sofa leaks)
    person_pixels = np.isin(labels, (1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17)) | clip_confident
    person_envelope = cv2.dilate(person_pixels.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) > 0

    # 9. Background Protection: Strictly eliminate bedsheets, sofas, and wall clutter
    bg = (labels == 0)
    bg_shield = cv2.erode(bg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    seam_band = cv2.dilate(garment_base.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*max(1, int(expand_pixels))+1,)*2)) > 0
    effective_bg_shield = bg_shield & ~clip_confident & ~seam_band

    # 10. Candidate fusion with multi-target isolation
    candidate = (garment_base | clip_mask_up)
    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    if wants_high_collar and faces_detected:
        bx, by, bw, bh = faces_detected[0]["box"]
        neck_skin = seg_face & ~face_safety_barrier & (yy > by + bh * 0.55) & (yy < chin_y + bh * 1.1)
        neck_skin &= np.abs(xx - (bx + bw / 2.0)) < bw * 1.20
        candidate |= neck_skin

    # A dress must also replace bare midriff between the original top and skirt.
    # Keep deliberately requested cropped/cutout outfits free to expose the waist.
    wants_bare_midriff = bool(re.search(r"\b(crop(?:ped)?[- ]top|midriff|cut[- ]out|two[- ]piece)\b", prompt_lower))
    if target in ("dress", "all", "outfit") and not wants_bare_midriff:
        first_garment_y = np.where(garment_base, yy, h).min(axis=0)
        last_garment_y = np.where(garment_base, yy, -1).max(axis=0)
        candidate |= seg_face & (yy > first_garment_y) & (yy < last_garment_y)
    if wants_sleeves:
        candidate |= (arm_skin & ~arm_core)
    else:
        candidate &= ~arm_skin

    if wants_leg_coverage:
        candidate |= leg_skin
    else:
        candidate &= ~leg_core

    # Strictly suppress opposite/forbidden categories
    candidate &= ~forbidden_mask

    # Shield against background, arm/leg cores, face, hair, and shoes
    candidate = candidate & ~effective_bg_shield & ~arm_core & ~leg_core & ~shoe_core & ~face_protect & ~hair_core & person_envelope

    # 11. Connected Component Filter: Only retain clothing attached to the person's body
    num_labels, comp_labels, stats, centroids = cv2.connectedComponentsWithStats(candidate.astype(np.uint8))
    person_garment = np.zeros((h, w), dtype=bool)

    if faces_detected:
        bx, by, bw, bh = faces_detected[0]["box"]
        min_x = max(0, int(bx - 1.2 * bw))
        max_x = min(w, int(bx + 2.2 * bw))
    else:
        min_x, max_x = int(w * 0.10), int(w * 0.90)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 80:
            continue
        comp_x = stats[i, cv2.CC_STAT_LEFT]
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        if not (comp_x + comp_w < min_x or comp_x > max_x):
            person_garment |= (comp_labels == i)

    # 12. Seam expansion dilation
    kernel_size = 2*max(1, int(expand_pixels)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(person_garment.astype(np.uint8) * 255, kernel)

    # Strict post-dilation clipping: zero face leak, zero hair destruction, zero leg swallowing, zero shoe clipping, zero category bleed
    dilated[face_protect] = 0
    dilated[face_safety_barrier & (np.arange(h)[:, None] <= (chin_y + 2 if chin_y > 0 else h))] = 0
    dilated[hair_core] = 0
    dilated[arm_core] = 0
    dilated[leg_core] = 0
    dilated[shoe_core] = 0
    dilated[forbidden_mask] = 0
    dilated[~person_envelope] = 0
    dilated[effective_bg_shield] = 0

    # 13. Guided alpha feathering with 1px inner erosion to eliminate old fabric color halos
    feathered = feather_region(dilated > 0, max(1.0, feather_sigma*1.5))
    final_mask = np.rint(feathered * 255).astype(np.uint8)

    # Final zero-leak clamps
    final_mask[face_protect] = 0
    final_mask[face_safety_barrier & (np.arange(h)[:, None] <= (chin_y + 2 if chin_y > 0 else h))] = 0
    final_mask[hair_core] = 0
    final_mask[arm_core] = 0
    final_mask[leg_core] = 0
    final_mask[shoe_core] = 0
    final_mask[forbidden_mask] = 0
    final_mask[~person_envelope] = 0
    final_mask[effective_bg_shield] = 0

    # 14. Upsample to original image dimensions if multi-scale resolution scaling was applied
    if scale != 1.0:
        final_mask_full = cv2.resize(final_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        faces_returned = []
        for f in faces_detected:
            fb = f["box"]
            faces_returned.append({
                "box": (fb[0] / scale, fb[1] / scale, fb[2] / scale, fb[3] / scale),
                "chin_y": f["chin_y"] / scale,
                "eye_y": f["eye_y"] / scale,
                "mouth_y": f["mouth_y"] / scale,
            })
    else:
        final_mask_full = final_mask
        faces_returned = faces_detected

    mask_pil = Image.fromarray(final_mask_full)
    mask_pil.quality_warnings = mask_warnings
    coverage = float((final_mask_full > 127).mean())

    # Visual overlay preview (Cyan highlight on masked clothing area)
    overlay = np.array(img_rgb).copy()
    color_layer = np.zeros_like(overlay)
    color_layer[:] = [14, 165, 233]  # Sky Cyan
    alpha = (final_mask_full.astype(float) / 255.0 * 0.45)[:, :, None]
    blended = (overlay * (1 - alpha) + color_layer * alpha).clip(0, 255).astype(np.uint8)
    preview_overlay = Image.fromarray(blended)

    print(f"[ClothingEngine] Fused dress detection completed. Coverage: {coverage:.4f}, Chin_Y: {chin_y / scale:.1f}")

    if return_flux_image:
        return mask_pil, preview_overlay, faces_returned, coverage, preview_overlay
    return mask_pil, preview_overlay, faces_returned, coverage


def generate_dress_mask(
    image: Image.Image,
    target: str = "dress",
    prompt: str = None,
    keep_face: bool = True,
    expand_pixels: int = 12,
    feather_sigma: float = 3.5
) -> tuple[Image.Image, Image.Image, list, float]:
    """Compatibility wrapper calling generate_fused_dress_mask."""
    return generate_fused_dress_mask(
        image=image,
        target=target,
        prompt=prompt,
        keep_face=keep_face,
        expand_pixels=expand_pixels,
        feather_sigma=feather_sigma,
        return_flux_image=False
    )



def clothing_mask_prompt(target: str = "dress") -> str:
    """Generate Stability Matrix ImageLabs visual segmentation instruction prompt."""
    garments = {
        "dress": "entire dress, including bodice, straps, skirt, hem, all trailing fabric and train",
        "upper": "upper-body garment, including sleeves, collar, cuffs and hem",
        "top": "upper-body garment, including sleeves, collar, cuffs and hem",
        "shirt": "upper-body garment, including sleeves, collar, cuffs and hem",
        "lower": "lower-body garment, including waistband, legs or skirt, hem and trailing fabric",
        "bottom": "lower-body garment, including waistband, legs or skirt, hem and trailing fabric",
        "pants": "lower-body garment, including waistband, legs or skirt, hem and trailing fabric",
        "skirt": "lower-body garment, including waistband, legs or skirt, hem and trailing fabric",
        "all": "entire outfit, including all garments, sleeves, straps, hems and trailing fabric",
        "outfit": "entire outfit, including all garments, sleeves, straps, hems and trailing fabric",
    }
    garment = garments.get(target.lower(), garments["dress"])
    return (
        "Convert this exact image into a black and white clothing segmentation mask. "
        f"Fill the {garment}, with pure white. "
        "Everything else must be pure black: face, hair, neck, exposed skin, arms, hands, legs, accessories and background. "
        "Keep the exact original outlines, position, proportions and image framing. Flat binary silhouette only, no shading, no details, no new clothing."
    )


def build_klein_mask_graph(
    source_subpath: str,
    target: str = "dress",
    width: int = 768,
    height: int = 1024,
    seed: int = 123
) -> dict:
    """Construct ComfyUI prompt graph for Klein Flux 2 4B visual segmentation mask synthesis."""
    mask_prompt = clothing_mask_prompt(target)
    return {
        "model": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "flux-2-klein-4b-fp8.safetensors",
                "weight_dtype": "default"
            }
        },
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3-4b-heretic_fp8_e4m3fn.safetensors",
                "type": "flux2",
                "device": "default"
            }
        },
        "vae": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "flux2-vae.safetensors"
            }
        },
        "source": {
            "class_type": "LoadImage",
            "inputs": {"image": source_subpath}
        },
        "resize": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["source", 0],
                "upscale_method": "nearest-exact",
                "megapixels": 1.0,
                "resolution_steps": 16
            }
        },
        "reference": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["resize", 0],
                "vae": ["vae", 0]
            }
        },
        "positive_text": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["clip", 0],
                "text": mask_prompt
            }
        },
        "negative_text": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["clip", 0],
                "text": ""
            }
        },
        "positive_cond": {
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["positive_text", 0],
                "latent": ["reference", 0]
            }
        },
        "negative_cond": {
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["negative_text", 0],
                "latent": ["reference", 0]
            }
        },
        "guider": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["model", 0],
                "positive": ["positive_cond", 0],
                "negative": ["negative_cond", 0],
                "cfg": 1.0
            }
        },
        "latent": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        "schedule": {
            "class_type": "Flux2Scheduler",
            "inputs": {
                "steps": 4,
                "width": width,
                "height": height
            }
        },
        "sampler": {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": "euler"
            }
        },
        "noise": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed}
        },
        "sample": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["noise", 0],
                "guider": ["guider", 0],
                "sampler": ["sampler", 0],
                "sigmas": ["schedule", 0],
                "latent_image": ["latent", 0]
            }
        },
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["sample", 0],
                "vae": ["vae", 0]
            }
        },
        "save": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["decode", 0],
                "filename_prefix": "Klein_Dress_Mask"
            }
        }
    }


def merge_clothing_proposal(
    base_mask: Image.Image,
    proposal: Image.Image,
    protected: np.ndarray
) -> tuple[Image.Image, dict]:
    """Klein proposes coverage; the SegFormer parser anchors alignment and protects skin.

    Exact Stability Matrix ImageLabs algorithm:
    1. Reject obvious failed renders (binary_fraction < 0.90, recall < 0.65, precision < 0.55).
    2. Constrain additions to existing garment vicinity via connected component anchoring (> 5% base overlap).
    3. Limit dilation drift to 12% of image dimension.
    4. Enforce strict skin & face protection.
    """
    base = np.asarray(base_mask) > 127
    rgb = np.asarray(proposal.convert("RGB").resize(base_mask.size, Image.Resampling.BILINEAR)).astype(np.int16)
    gray = rgb.mean(axis=2)
    binary_fraction = float((((gray < 32) | (gray > 223)) & ((rgb.max(axis=2) - rgb.min(axis=2)) < 32)).mean())
    selected = (gray > 127).astype(np.uint8)
    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    selected[protected] = False
    overlap = int((selected & base).sum())
    recall = overlap / max(1, int(base.sum()))
    precision = overlap / max(1, int(selected.sum()))
    report = {
        "binary_fraction": round(binary_fraction, 4),
        "parser_recall": round(recall, 4),
        "parser_precision": round(precision, 4)
    }
    if binary_fraction < 0.90 or recall < 0.65 or precision < 0.55 or base.sum() < 16:
        return base_mask, dict(report, accepted=False, reason="Mask did not agree sufficiently with the original garment")

    # Only keep components anchored to actual clothing, then limit boundary drift
    count, components = cv2.connectedComponents(selected.astype(np.uint8))
    anchored = np.zeros(base.shape, dtype=bool)
    for index in range(1, count):
        component = (components == index)
        if (component & base).sum() >= max(8, int(component.sum() * 0.05)):
            anchored |= component

    radius = max(1, round(min(base_mask.size) * 0.12))
    nearby = cv2.dilate(
        base.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    ).astype(bool)
    combined = base | (anchored & nearby)
    combined[protected] = False
    return Image.fromarray(combined.astype(np.uint8) * 255), dict(report, accepted=True, added_pixels=int((combined & ~base).sum()))


def detect_dress_mask_flux(
    image: Image.Image,
    prompt: str = "",
    target: str = "dress",
    keep_face: bool = True,
    feather_sigma: float = 3.0,
    progress_cb=None,
    return_flux_image: bool = False,
    seed: int = 123
) -> tuple:
    """High-precision dress detection for Klein Flux 2 and in-place outfit swap workflows.
    
    Combines SegFormer B2 clothing parser, CLIPSeg zero-shot text conditioning,
    vivid trim detection, YuNet facial protection, and arm/background separation.
    
    Returns:
        (mask_pil, preview_overlay, faces, coverage[, proposal_img])
    """
    return generate_fused_dress_mask(
        image=image,
        target=target,
        prompt=prompt,
        keep_face=keep_face,
        expand_pixels=14,
        feather_sigma=feather_sigma,
        return_flux_image=return_flux_image
    )

