"""Universal Prompt & Melanin Luminance Sync Engine with Anatomy & Body Sizing.

Ensures:
1. 100% User Prompt Priority: User prompt is placed strictly first with elevated weight.
2. Universal Melanin Profile: Integrates universal 8-tier Fitzpatrick skin profiling
   from core.skin_engine, covering all world ethnicities (South Indian, North Indian,
   African, European, East Asian, Hispanic, Middle Eastern) with zero whitewashing or darkening.
3. Dynamic Anatomical Sizing: Robust regex parsing for bust size, waist size, curves,
   petite, plus-size, and athletic builds, elevating attention weight to (:1.35).
4. Strict Anti-Pollution Safeguards: Suppresses stereotyping (bindi, sindoor), plastic/waxy
   smoothing, and two-tone head/body mismatches.
"""
import re
import cv2
import numpy as np
from PIL import Image, ImageOps
from core import config
from core.skin_engine import extract_universal_skin_profile


def parse_moss_typography(prompt: str) -> tuple[str, list[str]]:
    """Extract quoted text, slogans, brand words, or typography directives from prompt.
    
    Returns:
        (typography_enhancement_clause, detected_texts)
    """
    if not prompt:
        return "", []

    detected_texts = []
    # Match double quotes "...", single quotes '...', or explicit slogan/text patterns
    quote_matches = re.findall(r'["\u201c\u201d]([^"\u201c\u201d]{1,60})["\u201c\u201d]', prompt)
    if not quote_matches:
        quote_matches = re.findall(r"['\u2018\u2019]([a-zA-Z0-9\s\.\-!?:]{2,40})['\u2018\u2019]", prompt)

    kw_matches = re.findall(r'\b(?:text|slogan|words|letters|quote|title)\s*[:=]\s*["\']?([^,"\';\n]{1,60})["\']?', prompt, re.IGNORECASE)

    all_raw = list(dict.fromkeys(quote_matches + kw_matches))
    for t in all_raw:
        clean = t.strip()
        if len(clean) >= 2 and clean.lower() not in ("a", "an", "the", "in", "on", "at", "to", "for", "with"):
            detected_texts.append(clean)

    if not detected_texts:
        return "", []

    clauses = []
    for t in detected_texts:
        clauses.append(f'prominently featuring the exact text "{t}" with razor-sharp typography, perfect spelling, high-definition character edges, realistic fabric print texture')

    combined_clause = ", ".join(clauses)
    return combined_clause, detected_texts



def analyze_skin_melanin(
    image: Image.Image,
    face_bbox: tuple = None,
    src_faces: list = None,
    skin_profile: dict = None
) -> tuple[str, str, str]:
    """Analyze real skin tone from face pixels using universal 8-tier Fitzpatrick profiling.
    
    Returns:
        (melanin_descriptor, civitai_trigger, anti_skin_negative)
    """
    try:
        if src_faces is None and hasattr(image, "_cached_src_faces"):
            src_faces = getattr(image, "_cached_src_faces")
        if face_bbox is None and hasattr(image, "_cached_face_bbox"):
            face_bbox = getattr(image, "_cached_face_bbox")
        if skin_profile is None and hasattr(image, "_cached_skin_profile"):
            skin_profile = getattr(image, "_cached_skin_profile")

        img = ImageOps.exif_transpose(image).convert('RGB')
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if src_faces is not None:
            faces = src_faces
        elif face_bbox is None:
            from core.face_engine import get_face_modules, detect_faces_robust
            app, _, _, _, _ = get_face_modules()
            faces = detect_faces_robust(app, bgr)
            try:
                image._cached_src_faces = faces
            except Exception:
                pass
        else:
            faces = None
        
        is_natural_face = True
        
        if faces:
            # Select largest face
            f = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
            face_bbox = f.bbox
            
            # Natural makeup vs Glam makeup detection:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            h, w = bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face_crop = bgr[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                crop_hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
                glam_cosmetic_pixels = (crop_hsv[:, :, 1] > 135) & (crop_hsv[:, :, 2] > 90) & ((crop_hsv[:, :, 0] < 20) | (crop_hsv[:, :, 0] > 160))
                glam_ratio = float(np.count_nonzero(glam_cosmetic_pixels)) / float(max(1, face_crop.shape[0] * face_crop.shape[1]))
                is_natural_face = glam_ratio < 0.035
        elif face_bbox is not None and len(face_bbox) >= 4:
            x1, y1, x2, y2 = [int(v) for v in face_bbox]
            h, w = bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face_crop = bgr[y1:y2, x1:x2]
            if face_crop.size > 0:
                crop_hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
                glam_cosmetic_pixels = (crop_hsv[:, :, 1] > 135) & (crop_hsv[:, :, 2] > 90) & ((crop_hsv[:, :, 0] < 20) | (crop_hsv[:, :, 0] > 160))
                glam_ratio = float(np.count_nonzero(glam_cosmetic_pixels)) / float(max(1, face_crop.shape[0] * face_crop.shape[1]))
                is_natural_face = glam_ratio < 0.035

        if skin_profile is not None:
            profile = skin_profile
        else:
            profile = extract_universal_skin_profile(image, face_bbox=face_bbox)
            try:
                image._cached_skin_profile = profile
            except Exception:
                pass
        
        makeup_pos = " (natural bare skin texture, clean natural face:1.15), (minimal natural makeup:1.1)," if is_natural_face else ""
        makeup_neg = ", (heavy makeup:1.4), (thick lipstick:1.4), (bold eyeliner:1.35), (painted eyelashes:1.35), (glamour makeup:1.35), (cakey foundation:1.35), (doll face:1.4), (wax figure:1.4)" if is_natural_face else ""

        # Positive tokens with undertone anchoring
        pos = f"{profile['positive_tokens']},{makeup_pos} ({profile['undertone']} undertone:1.15), (authentic skin micro pores:1.1), (natural skin texture:1.1)"
        
        # Anti-conflict negative tokens
        anti_tokens = [f"({token}:1.5)" for token in profile["anti_conflict_tokens"]]
        neg = ", ".join(anti_tokens) + makeup_neg

        civitai = f"skin tone style, {profile['tier_name'].lower()}, natural skin texture, realistic skin"
        return pos, civitai, neg

    except Exception as e:
        print(f"[PromptEngine] Universal skin profiling fallback: {e}")
        return (
            "(natural skin tone:0.9), (authentic skin micro pores:0.9)",
            "skin tone style, detailed skin pore, realistic skin",
            "(plastic face:1.4), (waxy skin:1.4), (airbrushed skin:1.4)"
        )


def garment_requirements(prompt: str) -> dict:
    """Extract explicit garment attributes for conditioning and review."""
    words = re.findall(r"[a-z]+(?:-[a-z]+)*", prompt.lower())
    colors = {"peach", "black", "white", "ivory", "cream", "beige", "red", "crimson", "burgundy",
              "pink", "magenta", "purple", "violet", "blue", "navy", "teal", "green", "emerald",
              "yellow", "orange", "brown", "grey", "gray", "silver", "gold"}
    result = {}
    for index, word in enumerate(words):
        if word in {"dress", "gown", "turtleneck", "shirt", "blouse", "top", "skirt", "pants", "suit", "jacket", "hoodie"}:
            color = next((value for value in reversed(words[max(0,index-5):index]) if value in colors), None)
            if color:
                result["color"] = color
                break
    lower = prompt.lower()
    if re.search(r"\b(sleeveless|strapless|off[- ]shoulder)\b", lower):
        result["sleeves"] = "bare arms"
    elif re.search(r"\b(long[- ]sleeves?|full[- ]sleeves?)\b", lower):
        result["sleeves"] = "long sleeves"
    if re.search(r"\b(turtleneck|high collar|mock neck|polo neck)\b", lower):
        result["neckline"] = "high collar"
    return result


def merge_negative_prompt(items: list[str], user_prompt: str) -> str:
    """Keep one copy of each negative term, retaining its strongest weight."""
    user_lower = user_prompt.lower()
    allowed = set()
    for pattern, terms in (
        (r"\b(bindi|forehead dot)\b", ("bindi", "forehead dot", "forehead sticker", "red mark on forehead")),
        (r"\bsindoor\b", ("sindoor",)),
        (r"\btilak\b", ("tilak", "religious forehead paint")),
        (r"\b(text|slogan|typography|lettering|words?|logo)\b", ("text", "logo")),
    ):
        if re.search(pattern, user_lower) and not re.search(r"\b(?:no|without)\s+" + pattern, user_lower):
            allowed.update(terms)

    merged = {}
    for item in items:
        for match in re.finditer(r"\(([^()]*)\:([0-9.]+)\)|([^(),]+)", item):
            text = match.group(1) if match.group(1) is not None else match.group(3)
            weight = float(match.group(2)) if match.group(2) else 1.0
            for term in text.split(","):
                term = term.strip()
                key = term.lower()
                if key and key not in allowed:
                    previous = merged.get(key)
                    if previous is None or weight > previous[1]:
                        merged[key] = (term, weight)
    return ", ".join(term if weight == 1.0 else f"({term}:{weight:g})" for term, weight in merged.values())


def build_zero_pollution_prompt(
    user_prompt: str,
    source_image: Image.Image = None,
    mode: str = "in_place_swap",
    model_type: str = "sdxl",
    framing: str = "editorial",
    skin_profile: dict = None,
    cached_src_faces: list = None,
    lora_meta: dict = None,
    **kwargs
) -> tuple[str, str]:
    """Construct an unrestricted prompt with 100% user priority, authentic melanin sync,
    anatomy/body size accuracy, framing control, and anti-stereotype safeguards.
    
    Returns:
        (positive_prompt, negative_prompt)
    """
    clean_user = user_prompt.strip()
    if not clean_user:
        clean_user = "elegant luxury designer outfit, tailored fit, beautiful fabric texture"
    
    # Strip any trailing period
    clean_user = clean_user.rstrip(".")
    user_lower = clean_user.lower()

    has_dramatic_lighting = any(w in user_lower for w in [
        "sunset", "golden hour", "night", "neon", "rain", "dramatic light", "shadows", "backlight", "rim light"
    ])
    effective_framing = framing or "editorial"
    
    # Parse anatomy & body sizing requirements from prompt
    from core.anatomy_engine import parse_body_anatomy, build_anatomy_conditioning
    parsed_anatomy = parse_body_anatomy(clean_user)
    anatomy_pos_list, anatomy_neg_list = build_anatomy_conditioning(parsed_anatomy)
    anatomy_clause = ", ".join(anatomy_pos_list) if anatomy_pos_list else "(natural proportional body shape:1.1), (authentic human physique:1.1)"
    matched_anatomy = parsed_anatomy.get("raw_matches", [])
    
    is_in_place = mode in ("in_place", "in_place_swap")
    
    skin_pos = ""
    civitai_trigger = ""
    anti_skin = ""
    hair_pos = "natural hair texture"
    anti_hair = ""

    if source_image is not None and not is_in_place:
        try:
            skin_pos, civitai_trigger, anti_skin = analyze_skin_melanin(
                source_image,
                src_faces=cached_src_faces,
                skin_profile=skin_profile
            )
        except Exception as e:
            print(f"[PromptEngine] Skin melanin profiling warning: {e}")

    if is_in_place:
        # Outfit Inpainting Mode - Ensure exposed skin seamlessly matches the authentic neck and shoulders,
        # with direct positive affirmations of bare, unadorned skin (bypassing text encoder negation blindness)
        # and strict negative guards against unrequested jewelry, necklaces, or mismatched skin tones.
        skin_clause = ", accurate garment coverage and neckline, realistic fabric texture"
        anatomy_inject = f", {anatomy_clause}" if matched_anatomy else ""
        
        if model_type == "flux":
            positive = f"A photo of the person wearing {clean_user}{anatomy_inject}{skin_clause}, perfect fit, natural drape, photorealistic fabric weave, sharp seams, 8k uhd."
        elif model_type == "moss":
            typo_clause, _ = parse_moss_typography(clean_user)
            typo_chunk = f", {typo_clause}" if typo_clause else ""
            positive = f"A high-fashion photograph of the person wearing {clean_user}{anatomy_inject}{typo_chunk}{skin_clause}, perfect fit, natural drape, photorealistic fabric weave, sharp seams, 8k uhd."
        else:
            positive = f"({clean_user}:1.30){skin_clause}{anatomy_inject}, worn on body, perfect fit, natural drape, photorealistic fabric weave, sharp seams, 8k uhd"
        attributes = garment_requirements(clean_user)
        if attributes.get("color"):
            positive += f", ({attributes['color']} fabric color:1.4)" if model_type == "sdxl" else f", {attributes['color']} fabric color"
        if attributes.get("sleeves"):
            positive += f", {attributes['sleeves']}"
    else:
        if framing == "editorial" or not framing:
            if re.search(r"\b(close[- ]up|portrait|headshot|beauty shot)\b", user_lower):
                effective_framing = "portrait"
            elif re.search(r"\b(full[- ]body|head to toe|full[- ]length)\b", user_lower):
                effective_framing = "full_body"
        frame_description = {
            "full_body": "full-length photograph, complete head and outfit and shoes inside frame",
            "portrait": "head and shoulders portrait photograph, complete head inside frame",
        }.get(effective_framing, "waist-up fashion photograph, complete head and torso inside frame")
        has_custom_lighting = bool(re.search(
            r"\b(light(?:ing)?|sun(?:light|lit|set)?|golden hour|night|neon|candlelight|shadows?|backlight|rim light|moody)\b",
            user_lower))
        parts = [f"({clean_user}:1.3)" if model_type == "sdxl" else clean_user, frame_description]
        if skin_pos:
            # Complexion supports the reference identity without overpowering
            # explicit garment color, mood, pose or lighting in the user request.
            parts.append(re.sub(r":1\.[0-9]+\)", ":1.05)", skin_pos))
        if matched_anatomy:
            parts.append(anatomy_clause)
        if not has_custom_lighting:
            parts.append("soft natural lighting")
        parts.append("photorealistic fabric and skin detail, natural human proportions")
        positive = ", ".join(parts)
        if model_type == "moss":
            typo_clause, _ = parse_moss_typography(clean_user)
            if typo_clause:
                positive += f", {typo_clause}"

    # Strict negative prompt: Anti-crop safeguards, anti-stereotype, anti-lingerie
    user_lower = clean_user.lower()
    neg_items = []

    # Safeguard against headless catalog shots and cut off heads (weighted heavily at start)
    if not is_in_place:
        if effective_framing == "full_body":
            neg_items.append("(portrait:1.5), (headshot:1.6), (bust shot:1.5), (close up, extreme close up, tight headshot, macro shot, face zoomed in, cropped body, cropped dress, cutoff dress, cropped legs:1.5)")
            neg_items.append("(mutated hands, deformed hands, fused fingers, extra fingers, missing fingers, malformed limbs, twisted wrists:1.45)")

        # Negative guards against disproportionate anatomy, thick necks, and perspective distortion
        neg_items.append("(disproportionate head, small head, pinhead, giant body, thick muscular neck, swollen neck, bull neck, pasted head, floating head, head-to-body scale mismatch, wide angle perspective distortion, foreshortening distortion:1.35)")
        neg_items.append("(cropped head, cut off head, missing head, headless, decapitated, face cropped, head out of frame, partial head, cut off forehead, headless mannequin, clothes on hanger, torso without head, out of frame:1.35)")
        neg_items.append("(blurry, soft focus, out of focus, motion blur, low resolution, hazy, bad clarity:1.3)")
        if anti_hair:
            neg_items.append(anti_hair)
    
    # Melanin-specific anti-skin penalties (apply in BOTH in-place and full scene)
    if anti_skin and not is_in_place:
        neg_items.append(anti_skin)

    # Contradictory anatomy elimination (e.g. flat chest if large bust, skinny if plus size)
    if anatomy_neg_list:
        neg_items.extend(anatomy_neg_list)

    # Universal skin realism, anti-oily, anti-plastic, and anti-two-tone safeguards
    if not is_in_place:
        if has_dramatic_lighting:
            neg_items.append(
                "(orange skin, oversaturated skin, fake tan, sunburn, "
                "oily skin, greasy skin, shiny skin, reflective skin, "
                "two-tone skin, mismatched skin tone, mismatched head and body, floating head, plastic face, waxy skin:1.3)"
            )
        else:
            neg_items.append(
                "(orange skin, oversaturated skin, fake tan, sunburn, dark patches, "
                "oily skin, greasy skin, shiny skin, specular glare, reflective skin, "
                "two-tone skin, mismatched skin tone, mismatched head and body, floating head, plastic face, waxy skin:1.4)"
            )


    # In-place swap anti-hallucination guards (jewelry, necklaces, pendants unless requested)
    if is_in_place and not any(w in user_lower for w in ["necklace", "jewelry", "pendant", "chain", "choker"]):
        neg_items.append("(jewelry, necklace, pendant, gold chain, silver chain, choker, medal, medallion, neck ornament:1.45)")
    
    # Anti-stereotype checks
    if "bindi" not in user_lower and "dot" not in user_lower:
        neg_items.append("bindi, forehead dot, forehead sticker, red mark on forehead")
    if "sindoor" not in user_lower and "tilak" not in user_lower:
        neg_items.append("sindoor, tilak, religious forehead paint")
    
    # Anti-sheer / underwear bleed checks
    if not any(w in user_lower for w in ["nude", "lingerie", "bra", "underwear", "sheer"]):
        neg_items.append("nude lining, beige bra, visible bra cups, underwear, lingerie")
        
    # Anti-bridal bleed if user requested specific non-bridal styles
    if not any(w in user_lower for w in ["white", "wedding", "bridal", "bride", "gown"]):
        neg_items.append("wedding dress, bridal gown, veil")
        
    neg_items.append(config.DEFAULT_NEGATIVE_PROMPT)
    negative = merge_negative_prompt(neg_items, clean_user)
    
    return positive, negative
