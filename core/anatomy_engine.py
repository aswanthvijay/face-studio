"""Dynamic Anatomical Sizing & Body Build Parsing Engine.

Ensures that user-specified body dimensions (bust size, waist, curves, petite, plus size)
are accurately realized in generated imagery with high attention weighting (:1.35)
and aggressive elimination of contradictory anatomy in negative conditioning.
"""
import re


BUST_PATTERNS = [
    (r"\b(bust size\s*[a-e]|cup size\s*[a-e]|[a-e]\s*cup|[a-e]-cup)\b", "bust_specific"),
    (r"\b(large bust|full bust|voluptuous bust|heavy bust|ample bust|big breasts|large breasts|ample bosom)\b", "bust_large"),
    (r"\b(small bust|petite bust|flat chest|modest bust|small breasts)\b", "bust_small"),
    (r"\b(cleavage|deep cleavage|dramatic cleavage)\b", "bust_cleavage"),
]

WAIST_PATTERNS = [
    (r"\b(slender waist|tiny waist|cinched waist|hourglass waist|narrow waist|wasp waist|corseted waist)\b", "waist_slender"),
    (r"\b(thick waist|wide waist|straight waist)\b", "waist_wide"),
]

HIP_PATTERNS = [
    (r"\b(curvy hips|wide hips|hourglass figure|full curves|ample hips|thicc|pear shaped)\b", "hips_curvy"),
]

BUILD_PATTERNS = [
    (r"\b(petite frame|petite body|petite woman|short petite)\b", "build_petite"),
    (r"\b(plus size woman|plus size body|curvy plus size|full figured woman|chubby)\b", "build_plus_size"),
    (r"\b(athletic toned body|fitness model body|toned physique|muscular toned)\b", "build_athletic"),
    (r"\b(tall lean physique|tall slender woman|supermodel physique|slender body|slim body)\b", "build_slender"),
]


def parse_body_anatomy(user_prompt: str) -> dict:
    """Scan user prompt for specific body dimensions, sizing, and silhouette requests."""
    prompt_lower = user_prompt.lower()
    matched = {
        "bust": None,
        "waist": None,
        "hips": None,
        "build": None,
        "raw_matches": []
    }

    # Match bust
    for pattern, kind in BUST_PATTERNS:
        for m in re.finditer(pattern, prompt_lower):
            matched_str = m.group(0)
            # Guard against beverage false positives (e.g. "a cup of coffee", "drinking a cup", "holding a cup")
            if re.sub(r"\s+", " ", matched_str).strip() == "a cup":
                start, end = m.start(), m.end()
                after = prompt_lower[end:end+20]
                # If followed by "of" (e.g. "a cup of coffee")
                if re.match(r"^\s+of\b", after):
                    continue
                # If in a drink/holding context without any lingerie/breast/bra/size words
                is_beverage_ctx = any(w in prompt_lower for w in ["coffee", "tea", "water", "juice", "drink", "cafe", "mug", "sip", "warm water", "holding a cup", "drinking a cup", "cup in her hand", "cup in hand"])
                has_bra_ctx = any(w in prompt_lower for w in ["bra", "lingerie", "bust", "breast", "size", "cup size", "silhouette", "body", "frame", "cleavage", "model", "woman", "girl"])
                if is_beverage_ctx and not any(w in prompt_lower for w in ["bra", "lingerie", "bust", "breast", "size"]):
                    continue
                if prompt_lower.strip() in ("holding a cup", "a cup"):
                    continue
            matched["bust"] = (kind, matched_str)
            matched["raw_matches"].append(matched_str)
            break
        if matched["bust"]:
            break

    # Match waist
    for pattern, kind in WAIST_PATTERNS:
        m = re.search(pattern, prompt_lower)
        if m:
            matched["waist"] = (kind, m.group(0))
            matched["raw_matches"].append(m.group(0))
            break

    # Match hips
    for pattern, kind in HIP_PATTERNS:
        m = re.search(pattern, prompt_lower)
        if m:
            matched["hips"] = (kind, m.group(0))
            matched["raw_matches"].append(m.group(0))
            break

    # Match build
    for pattern, kind in BUILD_PATTERNS:
        m = re.search(pattern, prompt_lower)
        if m:
            matched["build"] = (kind, m.group(0))
            matched["raw_matches"].append(m.group(0))
            break

    return matched


def build_anatomy_conditioning(parsed_anatomy: dict) -> tuple[list[str], list[str]]:
    """
    Construct positive anatomy reinforcement clauses with elevated weight (:1.35)
    and contradictory negative tokens to prevent model dilution.
    """
    positive_clauses = []
    negative_tokens = []

    # Process Bust
    if parsed_anatomy.get("bust"):
        kind, text = parsed_anatomy["bust"]
        positive_clauses.append(f"({text}:1.35)")
        if kind == "bust_specific":
            if re.search(r"\b(a-cup|cup size\s*a|bust size\s*a)\b", text):
                negative_tokens.extend(["(large bust:1.4)", "(huge breasts:1.4)"])
            else:
                negative_tokens.extend(["(flat chest:1.4)", "(small bust:1.3)", "(small breasts:1.3)"])
        elif kind in ("bust_large", "bust_cleavage"):
            negative_tokens.extend(["(flat chest:1.4)", "(small bust:1.3)", "(small breasts:1.3)"])
        elif kind == "bust_small":
            negative_tokens.extend(["(large bust:1.4)", "(cleavage:1.3)", "(huge breasts:1.4)"])

    # Process Waist
    if parsed_anatomy.get("waist"):
        kind, text = parsed_anatomy["waist"]
        positive_clauses.append(f"({text}:1.35)")
        if kind == "waist_slender":
            negative_tokens.extend(["(thick waist:1.4)", "(wide waist:1.3)", "(boxy torso:1.3)"])

    # Process Hips
    if parsed_anatomy.get("hips"):
        kind, text = parsed_anatomy["hips"]
        positive_clauses.append(f"({text}:1.35)")
        negative_tokens.extend(["(straight narrow hips:1.3)", "(boxy figure:1.3)"])

    # Process Build
    if parsed_anatomy.get("build"):
        kind, text = parsed_anatomy["build"]
        positive_clauses.append(f"({text}:1.35)")
        if kind == "build_plus_size":
            negative_tokens.extend(["(skinny:1.4)", "(anorexic:1.4)", "(slender:1.3)", "(underweight:1.4)"])
        elif kind in ("build_petite", "build_slender"):
            negative_tokens.extend(["(plus size:1.4)", "(obese:1.4)", "(heavy body:1.3)", "(overweight:1.3)"])
        elif kind == "build_athletic":
            negative_tokens.extend(["(flabby:1.3)", "(skinny fat:1.3)", "(untrained:1.2)"])

    return positive_clauses, negative_tokens
