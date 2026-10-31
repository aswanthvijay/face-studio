import cv2
import numpy as np
from PIL import Image, ImageOps

UNIVERSAL_MELANIN_TIERS = [
    {
        "id": "tier_very_fair",
        "name": "Pale Porcelain / Very Fair",
        "min_l": 185.0,
        "max_l": 255.0,
        "positive_tokens": "(extremely fair porcelain skin tone, pale natural ivory complexion, soft translucent undertones:1.25)",
        "anti_conflict_tokens": ["dark skin", "bronze skin", "tan skin", "dusky skin", "sunburnt skin", "brown skin"],
        "desc": "Nordic, Very Fair North Indian, East Asian porcelain skin."
    },
    {
        "id": "tier_fair",
        "name": "Fair Ivory / Light Wheatish",
        "min_l": 170.0,
        "max_l": 185.0,
        "positive_tokens": "(fair natural ivory skin tone, light smooth wheatish complexion, natural skin glow:1.25)",
        "anti_conflict_tokens": ["dark skin", "deep bronze skin", "heavy tan", "dusky skin"],
        "desc": "Fair European, East Asian, North Indian fair wheatish."
    },
    {
        "id": "tier_light_wheatish",
        "name": "Light Wheatish / Olive",
        "min_l": 155.0,
        "max_l": 170.0,
        "positive_tokens": "(light wheatish skin tone, warm olive complexion, golden undertones:1.25)",
        "anti_conflict_tokens": ["pale white skin", "bleached skin", "dark bronze skin", "sunburnt skin"],
        "desc": "Mediterranean, Middle Eastern, Wheatish North/Central Indian."
    },
    {
        "id": "tier_medium_tan",
        "name": "Medium Warm Honey / Sunkissed Tan",
        "min_l": 140.0,
        "max_l": 155.0,
        "positive_tokens": "(medium warm honey skin tone, sunkissed tan complexion, rich golden undertones:1.25)",
        "anti_conflict_tokens": ["pale white skin", "bleached skin", "chalky skin", "porcelain skin"],
        "desc": "Latin, Southeast Asian, Central/South Asian warm honey."
    },
    {
        "id": "tier_dusky",
        "name": "Warm Dusky / Golden Brown",
        "min_l": 125.0,
        "max_l": 140.0,
        "positive_tokens": "(natural warm dusky South Indian skin tone, golden brown melanin-rich complexion, authentic warm undertones:1.30)",
        "anti_conflict_tokens": ["pale skin", "white skin", "bleached skin", "chalky porcelain", "doll face"],
        "desc": "South Indian, Afro-Latino, Brown skin."
    },
    {
        "id": "tier_deep_brown",
        "name": "Deep Warm Brown / Chestnut",
        "min_l": 110.0,
        "max_l": 125.0,
        "positive_tokens": "(deep warm brown skin tone, rich chestnut melanin complexion, warm bronze undertones:1.30)",
        "anti_conflict_tokens": ["pale skin", "white skin", "fair skin", "chalky face"],
        "desc": "Deep South Indian, Afro-Caribbean, Brown."
    },
    {
        "id": "tier_dark_espresso",
        "name": "Dark Espresso / Rich African Melanin",
        "min_l": 95.0,
        "max_l": 110.0,
        "positive_tokens": "(rich dark espresso African skin tone, deep melanin complexion, radiant smooth dark skin, natural epidermal depth:1.35)",
        "anti_conflict_tokens": ["pale skin", "white skin", "fair skin", "ivory skin", "chalky skin"],
        "desc": "African, Sub-Saharan, Melanin-Rich."
    },
    {
        "id": "tier_deep_ebony",
        "name": "Deep Ebony / Dark African Melanin",
        "min_l": 0.0,
        "max_l": 95.0,
        "positive_tokens": "(deep ebony dark skin tone, authentic dark African complexion, glowing natural skin depth, rich melanin:1.35)",
        "anti_conflict_tokens": ["pale skin", "white skin", "fair skin", "ivory skin", "chalky skin"],
        "desc": "Deep Ebony African melanin."
    }
]

def extract_universal_skin_profile(
    image: Image.Image,
    face_bbox: list = None
) -> dict:
    """
    Universally extracts the authentic melanin profile, Fitzpatrick scale,
    luminance, and undertones from a user portrait with glare and shadow rejection.
    
    Works reliably across ALL racial backgrounds without bias.
    """
    img_rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    h, w = img_bgr.shape[:2]

    detected_faces = []
    # Automatic face detection if face_bbox is not passed
    if face_bbox is None:
        try:
            from core.clothing_engine import detect_face_landmarks
            detected_faces = detect_face_landmarks(image)
            if detected_faces:
                box = detected_faces[0]["box"]
                face_bbox = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        except Exception:
            pass

    if face_bbox is None:
        try:
            from core.face_engine import get_face_modules, detect_faces_robust
            app, _, _, _, _ = get_face_modules()
            faces = detect_faces_robust(app, img_bgr)
            if faces:
                face_bbox = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[0].bbox
        except Exception:
            pass

    roi_lab_list = []
    roi_hsv_list = []
    roi_ycrcb_list = []

    if detected_faces or face_bbox is not None:
        if detected_faces:
            f = detected_faces[0]
            bx, by, bw, bh = f["box"]
            eye_y = f.get("eye_y", by + bh * 0.40)
            mouth_y = f.get("mouth_y", by + bh * 0.70)
        else:
            x1, y1, x2, y2 = [float(v) for v in face_bbox[:4]]
            bx, by = x1, y1
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            eye_y = by + bh * 0.40
            mouth_y = by + bh * 0.70
        chin_y = f.get("chin_y", by + bh * 0.88) if detected_faces else by + bh * 0.88
        cx = int(bx + bw / 2.0)

        # 1. Forehead region (strictly excludes hairline and eyebrows)
        fh_y1 = max(0, int(by + (eye_y - by) * 0.20))
        fh_y2 = max(0, int(by + (eye_y - by) * 0.80))
        fh_x1 = max(0, int(cx - bw * 0.25))
        fh_x2 = min(w, int(cx + bw * 0.25))
        fh_y1, fh_y2 = max(0, fh_y1), min(h, fh_y2)
        fh_x1, fh_x2 = max(0, fh_x1), min(w, fh_x2)
        if fh_y2 > fh_y1 and fh_x2 > fh_x1:
            roi_lab_list.append(img_lab[fh_y1:fh_y2, fh_x1:fh_x2].reshape(-1, 3))
            roi_hsv_list.append(img_hsv[fh_y1:fh_y2, fh_x1:fh_x2].reshape(-1, 3))
            roi_ycrcb_list.append(img_ycrcb[fh_y1:fh_y2, fh_x1:fh_x2].reshape(-1, 3))

        # 2. Left and Right Cheeks (strictly excludes nose bridge, nostrils, and mouth)
        chk_y1 = max(0, int(eye_y + (mouth_y - eye_y) * 0.15))
        chk_y2 = min(h, int(mouth_y - (mouth_y - eye_y) * 0.15))
        if chk_y2 > chk_y1:
            # Left cheek
            lc_x1 = max(0, int(cx - bw * 0.42))
            lc_x2 = min(w, int(cx - bw * 0.12))
            chk_y1, chk_y2 = max(0, chk_y1), min(h, chk_y2)
            if lc_x2 > lc_x1:
                roi_lab_list.append(img_lab[chk_y1:chk_y2, lc_x1:lc_x2].reshape(-1, 3))
                roi_hsv_list.append(img_hsv[chk_y1:chk_y2, lc_x1:lc_x2].reshape(-1, 3))
                roi_ycrcb_list.append(img_ycrcb[chk_y1:chk_y2, lc_x1:lc_x2].reshape(-1, 3))
            # Right cheek
            rc_x1 = max(0, int(cx + bw * 0.12))
            rc_x2 = min(w, int(cx + bw * 0.42))
            if rc_x2 > rc_x1:
                roi_lab_list.append(img_lab[chk_y1:chk_y2, rc_x1:rc_x2].reshape(-1, 3))
                roi_hsv_list.append(img_hsv[chk_y1:chk_y2, rc_x1:rc_x2].reshape(-1, 3))
                roi_ycrcb_list.append(img_ycrcb[chk_y1:chk_y2, rc_x1:rc_x2].reshape(-1, 3))

        # Note: Avoid sampling below chin_y to prevent contamination from collars, white shirts, necklaces, and clothing.

    if roi_lab_list:
        roi_lab_flat = np.concatenate(roi_lab_list, axis=0)
        roi_hsv_flat = np.concatenate(roi_hsv_list, axis=0)
        roi_ycrcb_flat = np.concatenate(roi_ycrcb_list, axis=0)
    elif face_bbox is not None and len(face_bbox) >= 4:
        x1, y1, x2, y2 = [int(v) for v in face_bbox]
        fw, fh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        fx1 = max(0, cx - int(fw * 0.30))
        fx2 = min(w, cx + int(fw * 0.30))
        fy1 = max(0, cy - int(fh * 0.30))
        fy2 = min(h, cy + int(fh * 0.25))
        roi_lab_flat = img_lab[fy1:fy2, fx1:fx2].reshape(-1, 3)
        roi_hsv_flat = img_hsv[fy1:fy2, fx1:fx2].reshape(-1, 3)
        roi_ycrcb_flat = img_ycrcb[fy1:fy2, fx1:fx2].reshape(-1, 3)
    else:
        # Fallback to upper-center portrait zone if all face detectors failed
        roi_lab_flat = img_lab[int(h*0.15):int(h*0.45), int(w*0.35):int(w*0.65)].reshape(-1, 3)
        roi_hsv_flat = img_hsv[int(h*0.15):int(h*0.45), int(w*0.35):int(w*0.65)].reshape(-1, 3)
        roi_ycrcb_flat = img_ycrcb[int(h*0.15):int(h*0.45), int(w*0.35):int(w*0.65)].reshape(-1, 3)

    if roi_lab_flat.size == 0:
        roi_lab_flat = img_lab.reshape(-1, 3)
        roi_hsv_flat = img_hsv.reshape(-1, 3)
        roi_ycrcb_flat = img_ycrcb.reshape(-1, 3)

    # Multi-condition Glare, Shadow, and Garment/Hair Filter:
    l_chan = roi_lab_flat[:, 0]
    sat_chan = roi_hsv_flat[:, 1]
    val_chan = roi_hsv_flat[:, 2]
    cr_chan = roi_ycrcb_flat[:, 1]
    cb_chan = roi_ycrcb_flat[:, 2]

    # Skin chrominance constraints:
    # Human skin across all world ethnicities strictly clusters in Cr in [130..178] and Cb in [75..132] in YCrCb space.
    skin_chroma = (cr_chan >= 130) & (cr_chan <= 178) & (cb_chan >= 75) & (cb_chan <= 132)
    
    # Shadow and glare rejection: exclude deep shadows below the 15th percentile or L < 20..38
    l_min_thresh = max(20.0, min(38.0, float(np.percentile(l_chan, 15)) if len(l_chan) > 10 else 38.0))
    valid_mask = (l_chan >= l_min_thresh) & (l_chan <= 235) & (val_chan >= 20) & (sat_chan >= 10) & skin_chroma

    if np.count_nonzero(valid_mask) > 50:
        filtered_l = l_chan[valid_mask]
        filtered_a = roi_lab_flat[:, 1][valid_mask]
        filtered_b = roi_lab_flat[:, 2][valid_mask]
    else:
        # Relax chrominance slightly if illumination is non-standard
        valid_mask_relaxed = (l_chan >= 20) & (l_chan <= 235) & (val_chan >= 15) & (sat_chan >= 8)
        if np.count_nonzero(valid_mask_relaxed) > 50:
            filtered_l = l_chan[valid_mask_relaxed]
            filtered_a = roi_lab_flat[:, 1][valid_mask_relaxed]
            filtered_b = roi_lab_flat[:, 2][valid_mask_relaxed]
        else:
            filtered_l = l_chan.flatten()
            filtered_a = roi_lab_flat[:, 1].flatten()
            filtered_b = roi_lab_flat[:, 2].flatten()

    median_l = float(np.median(filtered_l))
    mean_a = float(np.mean(filtered_a))
    mean_b = float(np.mean(filtered_b))

    # Determine Undertone:
    # b > 140 is Golden / Warm
    # b < 132 or a > b is Rosy / Cool
    # Otherwise Neutral
    if mean_b > 140.0:
        undertone = "warm golden"
    elif mean_b < 132.0 or mean_a > mean_b:
        undertone = "cool rosy"
    else:
        undertone = "neutral"

    # Match Universal Melanin Tier
    matched_tier = UNIVERSAL_MELANIN_TIERS[4] # Default to medium/dusky
    for tier in UNIVERSAL_MELANIN_TIERS:
        if tier["min_l"] <= median_l < tier["max_l"] or (tier["id"] == "tier_deep_ebony" and median_l < tier["max_l"] and median_l >= 0.0):
            matched_tier = tier
            break

    return {
        "luminance_l": round(median_l, 1),
        "chroma_a": round(mean_a, 1),
        "chroma_b": round(mean_b, 1),
        "tier_id": matched_tier["id"],
        "tier_name": matched_tier["name"],
        "undertone": undertone,
        "positive_tokens": matched_tier["positive_tokens"],
        "anti_conflict_tokens": matched_tier["anti_conflict_tokens"],
        "desc": matched_tier["desc"],
        "lab_mean": np.array([median_l, mean_a, mean_b], dtype=np.float32),
        "lab_std": np.array([float(np.std(filtered_l)), float(np.std(filtered_a)), float(np.std(filtered_b))], dtype=np.float32)
    }


def generate_dynamic_skin_negative(skin_profile: dict) -> list:
    """Generate dynamic negative prompt tokens that strictly protect authentic skin color."""
    if not skin_profile:
        return []
    anti_tokens = skin_profile.get("anti_conflict_tokens", [])
    # Format with moderate negative weight
    return [f"({tok}:1.4)" for tok in anti_tokens]


def feather_region(mask: np.ndarray, radius: float) -> np.ndarray:
    """Fade inside a region; opacity is zero at its boundary and outside it."""
    padded = np.pad(mask.astype(np.uint8), 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    ramp = np.clip((distance - 1.0) / max(1.0, radius), 0.0, 1.0)
    return ramp * ramp * (3.0 - 2.0 * ramp)


def harmonize_unified_canvas_skin(
    canvas_image: Image.Image,
    face_bbox: list = None,
    src_skin_profile: dict = None,
    blend_strength: float = 0.55,
    return_info: bool = False,
    clothing_labels: np.ndarray = None,
    edit_mask: Image.Image = None
):
    """Gently align body chroma to the face while preserving scene luminance.

    Semantic skin support and inward feathering protect clothing, hair and
    background. A single bounded chroma offset preserves directional shading;
    anatomical rectangles and per-limb exposure corrections are not used.
    """
    img_rgb = np.array(canvas_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    if face_bbox is None:
        try:
            from core.face_engine import get_face_modules, detect_faces_robust
            app, _, _, _, _ = get_face_modules()
            faces = detect_faces_robust(app, img_bgr)
            if faces:
                face_bbox = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).bbox
        except Exception as error:
            print(f"[SkinEngine] Face detection unavailable: {error}")
    if face_bbox is None:
        return (canvas_image, {}) if return_info else canvas_image

    source_lab = None
    if src_skin_profile is not None:
        profile_lab = src_skin_profile.get("lab_mean")
        if profile_lab is not None:
            profile_lab = np.asarray(profile_lab, dtype=np.float32).reshape(-1)
            if profile_lab.size == 3 and np.isfinite(profile_lab).all():
                source_lab = profile_lab

    try:
        if clothing_labels is None:
            from core.clothing_engine import parse_clothing_labels
            labels = parse_clothing_labels(canvas_image)
        else:
            labels = clothing_labels
        if labels.shape[:2] != (h, w):
            labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
        if source_lab is not None:
            target = source_lab
        else:
            from core.face_engine import extract_face_skin_profile
            target, _ = extract_face_skin_profile(img_bgr, face_bbox)
    except Exception as error:
        print(f"[SkinEngine] Skin correction skipped without reliable segmentation: {error}")
        return (canvas_image, {}) if return_info else canvas_image

    x1, y1, x2, y2 = [int(value) for value in face_bbox]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    yy, xx = np.indices((h, w), dtype=np.float32)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(face_mask, ((x1+x2)//2, (y1+y2)//2),
                (max(1, int(bw*0.51)), max(1, int(bh*0.53))), 0, 0, 360, 1, -1)
    face_protection = feather_region(face_mask, max(3.0, bw*0.04))
    jaw_ramp = np.clip((yy - (y2 - bh*0.15)) / max(1.0, bh*0.35), 0.0, 1.0)
    jaw_ramp = jaw_ramp * jaw_ramp * (3.0 - 2.0*jaw_ramp)
    semantic_skin = np.isin(labels, (11, 12, 13, 14, 15))
    body_skin_mask = semantic_skin & (jaw_ramp > 0) & (face_protection < 1)
    edit_alpha = None
    if edit_mask is not None:
        edit_alpha = np.array(edit_mask.convert("L").resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        body_skin_mask &= edit_alpha > 0
    if np.count_nonzero(body_skin_mask) < 80:
        return (canvas_image, {}) if return_info else canvas_image

    lab = cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    sample = body_skin_mask & (lightness > 15) & (lightness < 85)
    if np.count_nonzero(sample) < 80:
        return (canvas_image, {}) if return_info else canvas_image
    body_ab = np.median(lab[:, :, 1:][sample], axis=0)
    target_ab = np.asarray(target[1:], dtype=np.float32) - 128.0
    # Allow full authentic melanin chroma alignment (up to ±20.0) across body skin
    # without artificial clamping that leaves arms/neck pale and disconnected from face
    chroma_delta = np.clip(target_ab - body_ab, -20.0, 20.0)
    alpha_mask = feather_region(body_skin_mask, max(4.0, min(24.0, bw*0.07)))
    alpha_mask *= jaw_ramp * (1.0 - face_protection)
    shadow_weight = np.clip((lightness - 4.0) / 12.0, 0.0, 1.0)
    highlight_weight = np.clip((98.0 - lightness) / 18.0, 0.0, 1.0)
    alpha_mask *= shadow_weight * highlight_weight * np.clip(blend_strength, 0.0, 1.0)
    if edit_alpha is not None:
        alpha_mask *= edit_alpha
    lab_out = lab.copy()
    lab_out[:, :, 1:] += chroma_delta * alpha_mask[:, :, None]
    adjusted = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR) * 255.0
    result = img_bgr.copy()
    support = alpha_mask > 0
    result[support] = np.rint(np.clip(adjusted[support], 0, 255)).astype(np.uint8)
    output = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    output._cached_clothing_labels = labels
    if return_info:
        encoded_lab = lab_out.copy()
        encoded_lab[:, :, 0] *= 2.55
        encoded_lab[:, :, 1:] += 128.0
        info = {
            "target_lab": target.tolist(), "active_zones": [{
                "name": "body_chroma", "dl": 0.0,
                "da": float(chroma_delta[0]), "db": float(chroma_delta[1]),
                "pixels": int(np.count_nonzero(body_skin_mask)), "mask": body_skin_mask,
                "orig_med_l": float(np.median(lightness[sample])*2.55),
                "orig_mean_a": float(body_ab[0]+128), "orig_mean_b": float(body_ab[1]+128)}],
            "body_skin_mask": body_skin_mask, "is_garment": np.isin(labels, (1,4,5,6,7,8,9,10,16,17)),
            "alpha_mask": alpha_mask, "garment_fade": semantic_skin.astype(np.float32), "lab_out": encoded_lab,
            "blended_bgr": result, "img_bgr": img_bgr, "labels": labels,
            "yy": yy, "xx": xx, "face_bbox": list(face_bbox),
        }
        return output, info
    return output
