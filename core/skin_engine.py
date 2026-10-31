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
        "desc": "Very light observed facial color."
    },
    {
        "id": "tier_fair",
        "name": "Fair Ivory / Light Wheatish",
        "min_l": 170.0,
        "max_l": 185.0,
        "positive_tokens": "(fair natural ivory skin tone, light smooth wheatish complexion, natural skin glow:1.25)",
        "anti_conflict_tokens": ["dark skin", "deep bronze skin", "heavy tan", "dusky skin"],
        "desc": "Light observed facial color."
    },
    {
        "id": "tier_light_wheatish",
        "name": "Light Wheatish / Olive",
        "min_l": 155.0,
        "max_l": 170.0,
        "positive_tokens": "(light wheatish skin tone, natural warm olive complexion, soft diffuse skin scattering:1.25)",
        "anti_conflict_tokens": ["pale white skin", "bleached skin", "dark bronze skin", "sunburnt skin"],
        "desc": "Light to medium observed facial color."
    },
    {
        "id": "tier_medium_tan",
        "name": "Medium Warm Honey / Sunkissed Tan",
        "min_l": 140.0,
        "max_l": 155.0,
        "positive_tokens": "(medium warm honey skin tone, authentic sunkissed tan complexion, natural warm undertones, soft diffuse skin scattering:1.25)",
        "anti_conflict_tokens": ["pale white skin", "bleached skin", "chalky skin", "porcelain skin"],
        "desc": "Medium observed facial color."
    },
    {
        "id": "tier_dusky",
        "name": "Warm Dusky / Golden Brown",
        "min_l": 125.0,
        "max_l": 140.0,
        "positive_tokens": "(natural warm dusky South Indian skin tone, authentic melanin-rich complexion, natural warm undertones, soft diffuse skin scattering:1.30)",
        "anti_conflict_tokens": ["pale skin", "white skin", "bleached skin", "chalky porcelain", "doll face", "metallic skin", "orange skin", "deep bronze skin", "copper skin"],
        "desc": "Medium brown observed facial color."
    },
    {
        "id": "tier_deep_brown",
        "name": "Deep Warm Brown / Chestnut",
        "min_l": 110.0,
        "max_l": 125.0,
        "positive_tokens": "(deep warm brown skin tone, rich chestnut melanin complexion, authentic warm undertones, soft diffuse skin scattering:1.30)",
        "anti_conflict_tokens": ["pale skin", "white skin", "fair skin", "chalky face"],
        "desc": "Deep brown observed facial color."
    },
    {
        "id": "tier_dark_espresso",
        "name": "Dark Espresso / Dark Brown",
        "min_l": 95.0,
        "max_l": 110.0,
        "positive_tokens": "(rich dark espresso skin tone, deep melanin complexion, radiant smooth dark skin, natural epidermal depth:1.35)",
        "anti_conflict_tokens": ["pale skin", "white skin", "fair skin", "ivory skin", "chalky skin"],
        "desc": "Dark brown observed facial color."
    },
    {
        "id": "tier_deep_ebony",
        "name": "Deep Ebony / Very Dark Brown",
        "min_l": 0.0,
        "max_l": 95.0,
        "positive_tokens": "(deep ebony skin tone, natural dark complexion, glowing natural skin depth, rich melanin:1.35)",
        "anti_conflict_tokens": ["pale skin", "white skin", "fair skin", "ivory skin", "chalky skin"],
        "desc": "Very dark brown observed facial color."
    }
]

def extract_universal_skin_profile(
    image: Image.Image,
    face_bbox: list = None
) -> dict:
    """
    Estimate observed face color for prompt conditioning.

    These heuristic color bins are not a measurement of melanin, ethnicity or
    Fitzpatrick phototype. Lighting, makeup and camera processing affect them.
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

    if detected_faces and len(detected_faces) > 0:
        f = detected_faces[0]
        bx, by, bw, bh = f["box"]
        eye_y = f.get("eye_y", by + bh * 0.40)
        mouth_y = f.get("mouth_y", by + bh * 0.70)
        chin_y = f.get("chin_y", by + bh * 0.88)
        cx = int(bx + bw / 2.0)

        # 1. Forehead region (strictly excludes hairline and eyebrows)
        fh_y1 = max(0, int(by + (eye_y - by) * 0.20))
        fh_y2 = max(0, int(by + (eye_y - by) * 0.80))
        fh_x1 = max(0, int(cx - bw * 0.25))
        fh_x2 = min(w, int(cx + bw * 0.25))
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
            lc_x2 = max(0, int(cx - bw * 0.12))
            if lc_x2 > lc_x1:
                roi_lab_list.append(img_lab[chk_y1:chk_y2, lc_x1:lc_x2].reshape(-1, 3))
                roi_hsv_list.append(img_hsv[chk_y1:chk_y2, lc_x1:lc_x2].reshape(-1, 3))
                roi_ycrcb_list.append(img_ycrcb[chk_y1:chk_y2, lc_x1:lc_x2].reshape(-1, 3))
            # Right cheek
            rc_x1 = min(w, int(cx + bw * 0.12))
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
    # Heuristic chroma filter; unusual lighting and complexions can fall outside it.
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
        undertone = "warm natural"
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
    edit_mask: Image.Image = None,
    align_luminance: bool = False
):
    """Gently align body chroma to the face with lateral ear coverage and continuous soft skin gating (Bugs B5, B6)."""
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

    try:
        if clothing_labels is None:
            from core.clothing_engine import parse_clothing_labels
            labels = parse_clothing_labels(canvas_image)
        else:
            labels = clothing_labels
        if labels.shape[:2] != (h, w):
            labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
        from core.face_engine import extract_face_skin_profile
        target, target_std = extract_face_skin_profile(img_bgr, face_bbox)
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
    is_limb_skin = np.isin(labels, (12, 13, 14, 15))
    is_torso_skin = (labels == 11) & (jaw_ramp > 0)

    # Lateral ear skin zone (Bug B5): y in [y1, y2 + 0.08*bh], x within x1 - 0.28*bw .. x2 + 0.28*bw excluding face
    lateral_ear_zone = (yy >= y1) & (yy <= y2 + 0.08 * bh) & (
        ((xx >= x1 - 0.28 * bw) & (xx < x1 - 0.02 * bw)) |
        ((xx > x2 + 0.02 * bw) & (xx <= x2 + 0.28 * bw))
    )
    ycrcb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    cr_img, cb_img = ycrcb_img[:, :, 1], ycrcb_img[:, :, 2]
    ear_chroma = (cr_img >= 132) & (cr_img <= 178) & (cb_img >= 75) & (cb_img <= 132)
    lateral_ear_skin = lateral_ear_zone & ear_chroma

    semantic_skin = is_limb_skin | is_torso_skin | lateral_ear_skin
    body_skin_mask = semantic_skin & (face_protection < 1)
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
    body_ab = np.mean(lab[:, :, 1:][sample], axis=0) if align_luminance else np.median(lab[:, :, 1:][sample], axis=0)
    target_ab = np.asarray(target[1:], dtype=np.float32) - 128.0

    chroma_delta = np.clip(target_ab - body_ab, -25.0, 25.0)

    body_std = np.std(lab[:, :, 1:][sample], axis=0)
    ref_std_ab = np.asarray(target_std[1:], dtype=np.float32) if target_std is not None and len(target_std) >= 3 else np.array([6.0, 6.0], dtype=np.float32)
    sigma_ratio = np.clip(ref_std_ab / np.maximum(body_std, 1.0), 0.75, 1.0)

    # Continuous per-pixel soft_skin_gate (Cr in [132, 178], Cb in [75, 132], S in [12, 210], L >= 28)
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat_img = hsv_img[:, :, 1]
    soft_skin_gate = (
        (cr_img >= 132) & (cr_img <= 178) &
        (cb_img >= 75) & (cb_img <= 132) &
        (sat_img >= 12) & (sat_img <= 210) &
        (lightness >= 11.0)
    ).astype(np.float32)
    soft_skin_gate = cv2.GaussianBlur(soft_skin_gate, (15, 15), 0)

    # Highlight desaturation lowered to 0.15 (Bug B6)
    glare_mask = np.clip((lightness - 72.0) / 18.0, 0.0, 1.0)
    glare_damping = 1.0 - 0.15 * glare_mask * (body_skin_mask.astype(np.float32))

    alpha_mask = feather_region(body_skin_mask, max(3.0, min(12.0, bw*0.035)))
    torso_ramp = np.where(labels == 11, jaw_ramp, 1.0).astype(np.float32)
    alpha_mask *= torso_ramp * (1.0 - face_protection)
    shadow_weight = np.clip((lightness - 2.0) / 10.0, 0.25, 1.0)
    highlight_weight = np.clip((100.0 - lightness) / 12.0, 0.40, 1.0)
    alpha_mask *= shadow_weight * highlight_weight * np.clip(blend_strength, 0.0, 1.0) * soft_skin_gate
    if edit_alpha is not None:
        alpha_mask *= edit_alpha

    lab_out = lab.copy()
    delta_l = 0.0
    if align_luminance:
        target_l = float(target[0]) * (100.0 / 255.0)
        body_l = float(np.mean(lightness[sample]))
        delta_l = np.clip(target_l - body_l, -30.0, 30.0)
        eff_weight = float(np.mean(alpha_mask[sample])) if np.count_nonzero(sample) > 0 else 0.5
        l_scale = np.clip(1.0 / max(0.20, eff_weight), 1.0, 3.0)
        l_specular_rolloff = np.clip((96.0 - lightness) / 22.0, 0.20, 1.0)
        shadow_fill = np.clip((48.0 - lightness) / 20.0, 0.0, 1.0) * 7.5
        lab_out[:, :, 0] += (delta_l + shadow_fill) * np.clip(alpha_mask * l_scale, 0.0, 1.0) * l_specular_rolloff
        lab_out[:, :, 0] = np.clip(lab_out[:, :, 0], 0.0, 100.0)

    eff_weight = float(np.mean(alpha_mask[sample])) if np.count_nonzero(sample) > 0 else 0.5
    chroma_scale = np.clip(1.0 / max(0.25, eff_weight), 1.0, 2.5) if align_luminance else 1.0
    delta_term = chroma_delta * np.clip(alpha_mask * chroma_scale, 0.0, 1.0)[:, :, None]
    variance_term = (sigma_ratio - 1.0) * (lab[:, :, 1:] - body_ab) * alpha_mask[:, :, None]
    lab_out[:, :, 1:] += delta_term + variance_term
    lab_out[:, :, 1:] *= glare_damping[:, :, None]

    if align_luminance:
        clamp_weight = (alpha_mask * soft_skin_gate)
        max_allowed_b = float(target_ab[1]) + 2.5
        excess_b = np.maximum(0.0, lab_out[:, :, 2] - max_allowed_b)
        lab_out[:, :, 2] -= excess_b * clamp_weight
        max_allowed_a = float(target_ab[0]) + 3.0
        excess_a = np.maximum(0.0, lab_out[:, :, 1] - max_allowed_a)
        lab_out[:, :, 1] -= excess_a * clamp_weight
        min_allowed_b = float(target_ab[1]) - 5.0
        deficit_b = np.maximum(0.0, min_allowed_b - lab_out[:, :, 2])
        lab_out[:, :, 2] += deficit_b * clamp_weight
        min_allowed_a = float(target_ab[0]) - 6.0
        deficit_a = np.maximum(0.0, min_allowed_a - lab_out[:, :, 1])
        lab_out[:, :, 1] += deficit_a * clamp_weight
        dermal_micro_grain = np.random.normal(0.0, 1.25, (h, w)).astype(np.float32) * clamp_weight * (1.0 - 0.15 * glare_mask)
        lab_out[:, :, 0] = np.clip(lab_out[:, :, 0] + dermal_micro_grain, 0.0, 100.0)

    adjusted = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR) * 255.0
    result = img_bgr.copy()
    blend_weight = np.clip(alpha_mask * soft_skin_gate, 0.0, 1.0)[:, :, None]
    result = np.rint(np.clip(adjusted * blend_weight + img_bgr.astype(np.float32) * (1.0 - blend_weight), 0, 255)).astype(np.uint8)

    output = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    output._cached_clothing_labels = labels
    if return_info:
        encoded_lab = lab_out.copy()
        encoded_lab[:, :, 0] *= 2.55
        encoded_lab[:, :, 1:] += 128.0
        info = {
            "target_lab": target.tolist(), "active_zones": [{
                "name": "body_chroma", "dl": float(delta_l),
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

def clamp_skin_tone_robust(
    input_face_bgr: np.ndarray,
    generated_scene_bgr: np.ndarray,
    input_skin_mask: np.ndarray,
    scene_skin_mask: np.ndarray,
) -> np.ndarray:
    """
    Matches generated body and face skin tone to the input image while filtering out 
    harsh shadows and specular glare to prevent orange over-saturation or pinkish lightening (Bug B6).
    """
    in_lab = cv2.cvtColor(input_face_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    out_lab = cv2.cvtColor(generated_scene_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    in_pixels = in_lab[input_skin_mask > 128]
    out_pixels = out_lab[scene_skin_mask > 128]

    if len(in_pixels) == 0 or len(out_pixels) == 0:
        return generated_scene_bgr

    # Filter out extreme shadows (<15th percentile) and glare (>85th percentile) on L, a, b channels
    in_l_low, in_l_high = np.percentile(in_pixels[:, 0], [15, 85])
    out_l_low, out_l_high = np.percentile(out_pixels[:, 0], [15, 85])

    in_valid = in_pixels[(in_pixels[:, 0] >= in_l_low) & (in_pixels[:, 0] <= in_l_high)]
    out_valid = out_pixels[(out_pixels[:, 0] >= out_l_low) & (out_pixels[:, 0] <= out_l_high)]

    if len(in_valid) == 0 or len(out_valid) == 0:
        in_valid, out_valid = in_pixels, out_pixels

    in_mean, in_std = np.mean(in_valid, axis=0), np.std(in_valid, axis=0) + 1e-6
    out_mean, out_std = np.mean(out_valid, axis=0), np.std(out_valid, axis=0) + 1e-6

    # Build continuous skin gate on scene image (Cr in [132, 178], Cb in [75, 132], S in [12, 210], L >= 28)
    ycrcb = cv2.cvtColor(generated_scene_bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(generated_scene_bgr, cv2.COLOR_BGR2HSV)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    sat = hsv[:, :, 1]
    lum = out_lab[:, :, 0]
    
    skin_gate = (
        (cr >= 132) & (cr <= 178) &
        (cb >= 75) & (cb <= 132) & 
        (sat >= 12) & (sat <= 210) &
        (lum >= 28)
    ).astype(np.float32)
    skin_gate = cv2.GaussianBlur(skin_gate, (15, 15), 0)

    # Multiply mask_f by skin_gate before blur
    raw_mask_f = (scene_skin_mask.astype(np.float32) / 255.0) if scene_skin_mask.max() > 1.0 else scene_skin_mask.astype(np.float32)
    mask_f = raw_mask_f * skin_gate
    smooth_skin_mask = cv2.GaussianBlur(mask_f, (41, 41), 0)
    # Multiply by skin_gate after blur so neutral walls (a=128, b=128) have 0 halo
    mask_f = smooth_skin_mask * skin_gate

    corrected_lab = out_lab.copy()
    for c in range(3):
        scale = np.clip(in_std[c] / out_std[c], 0.65, 1.25)
        if c == 0:
            scale *= 0.8
        delta = (out_lab[:, :, c] - out_mean[c]) * scale + in_mean[c] - out_lab[:, :, c]
        if c > 0:
            glare_mask = np.clip((lum - 180.0) / 45.0, 0.0, 1.0)
            glare_damp = 1.0 - 0.15 * glare_mask * mask_f
            corrected_lab[:, :, c] = (out_lab[:, :, c] + delta * mask_f) * glare_damp
        else:
            corrected_lab[:, :, c] = out_lab[:, :, c] + delta * mask_f

    corrected_lab = np.clip(corrected_lab, 0, 255).astype(np.uint8)
    corrected_bgr = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
    
    m_3d = mask_f[..., None]
    return (corrected_bgr.astype(np.float32) * m_3d + generated_scene_bgr.astype(np.float32) * (1.0 - m_3d)).clip(0, 255).astype(np.uint8)
