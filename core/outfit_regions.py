"""Single owner of source-coordinate regions for the automatic outfit path.

No candidate mask is allowed to redefine foreground ownership. All arrays use
the already-normalized source dimensions; local detector coordinates are mapped
back here, once, before synthesis.
"""
from dataclasses import dataclass, field
import re
import cv2
import numpy as np
from PIL import Image

GARMENT = (4, 5, 6, 7, 8, 17)
SKIN = (11, 12, 13, 14, 15)


def dilate(mask, radius):
    return cv2.dilate(mask.astype('uint8'), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)) > 0


def core(mask, radius=2):
    return cv2.erode(mask.astype('uint8'), np.ones((radius*2+1,)*2, 'uint8')) > 0


def coverage_intent(prompt):
    p = prompt.lower()
    bare = bool(re.search(r'\b(sleeveless|strapless|off.shoulder|one.shoulder|tank|halter|short[ -]sleeve\w*)\b', p))
    long = bool(re.search(r'\b(long[ -]sleeve\w*|full[ -]sleeve\w*)\b', p))
    return {'bare_arms': bare, 'long_sleeves': long and not bare,
            'short_hem': bool(re.search(r'\b(mini|short|shorter|above.the.knee)\b', p)),
            'neckline_change': bool(re.search(r'\b(strapless|off.shoulder|one.shoulder|halter|v.neck|low.cut|plunging|high.neck|turtleneck)\b', p))}


@dataclass
class SourceRegions:
    labels: np.ndarray
    regions: dict
    garment: np.ndarray
    coverage: np.ndarray
    approved: np.ndarray
    synthesis: np.ndarray
    alpha: np.ndarray
    intent: dict
    reasons: list = field(default_factory=list)
    decisions: dict = field(default_factory=dict)

    @property
    def protected(self):
        return np.logical_or.reduce(list(self.regions.values()))

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        for name, mask in {**self.regions, 'garment': self.garment,
                           'coverage': self.coverage, 'approved': self.approved,
                           'synthesis': self.synthesis}.items():
            Image.fromarray(mask.astype('uint8')*255).save(directory / (name + '.png'))
        Image.fromarray(self.alpha).save(directory / 'alpha.png')
        Image.fromarray(self.labels).save(directory / 'labels.png')


def clip_proposals(image, queries):
    # Do not use segment_with_clipseg: its silent zero-mask fallback conceals
    # missing assets/inference failures in a protection-critical detector.
    import torch
    from core.clothing_engine import get_clipseg
    processor, model = get_clipseg()
    result = {}
    for name, query in queries.items():
        inputs = processor(text=[query], images=[image], padding=True, return_tensors='pt')
        with torch.inference_mode():
            values = model(**inputs).logits.sigmoid().cpu().numpy().squeeze()
        result[name] = cv2.resize(values, image.size, interpolation=cv2.INTER_LINEAR)
    return result


def refine_hair_boundary(image, labels):
    """Resolve garment/hair contact with contextual 512px local parser views."""
    from core.clothing_engine import parse_clothing_labels
    hair = labels == 2
    garments = np.isin(labels, GARMENT)
    boundary = dilate(hair, 6) & dilate(garments, 6)
    votes = np.zeros(labels.shape, np.uint8)
    seen = np.zeros(labels.shape, np.uint8)
    h, w = labels.shape
    for y in sorted(set(range(0, max(1, h-512), 384)) | {max(0, h-512)}):
        for x in sorted(set(range(0, max(1, w-512), 384)) | {max(0, w-512)}):
            x2, y2 = min(w, x+512), min(h, y+512)
            if not boundary[y:y2, x:x2].any():
                continue
            local = parse_clothing_labels(image.crop((x, y, x2, y2)))
            votes[y:y2, x:x2] += local == 2
            seen[y:y2, x:x2] += 1
    near = dilate(boundary, 12) & (seen > 0)
    uncertain = near & (votes > 0) & (votes < seen)
    refined = hair.copy()
    refined[near & (votes == seen)] = True
    refined[near & (votes == 0) & garments] = False
    # Foreground is authoritative on disagreement. Do not shave hair to erase
    # an old-clothing fringe; validation will reject unresolved remnants.
    return refined | uncertain, uncertain


def build_source_regions(image, prompt, progress=None, target='dress'):
    from core import config
    from core.clothing_engine import parse_clothing_labels, detect_face_landmarks
    from core.hand_regions import hand_protection
    from core.text_regions import detect_text
    if progress:
        progress('Mapping hair, hands, objects and existing text...')
    labels = parse_clothing_labels(image)
    garment_ids = (4,17) if target in ('upper','top','shirt') else (5,6,8) if target in ('lower','bottom','pants','skirt') else GARMENT
    garment = np.isin(labels, garment_ids)
    intent = coverage_intent(prompt)
    reasons = []
    decisions = {}
    empty = np.zeros(labels.shape, bool)
    face = labels == 11
    faces = detect_face_landmarks(image)
    if not faces:
        reasons.append('No reliable face boundary. Use a clearer source photo.')
    for item in faces:
        x, y, w, h = map(round, item['box'])
        face[max(0,y):min(image.height,y+h), max(0,x):min(image.width,x+w)] = True
    hair, uncertain = refine_hair_boundary(image, labels)
    hand_assets = config.WORKSPACE_ROOT / 'models' / 'hands'
    if not all((hand_assets / n).exists() for n in (
            'palm_detection_mediapipe_2023feb.onnx', 'handpose_estimation_mediapipe_2023feb.onnx')):
        reasons.append('Hand protection models are missing. Run scripts/setup_hand_landmarks.py.')
    hands, detections = hand_protection(image, labels)  # unconditional, including sleeveless
    hand_context = empty.copy()
    for detection in detections:
        cv2mask = np.zeros(labels.shape, np.uint8)
        cv2.fillConvexPoly(cv2mask, cv2.convexHull(np.rint(detection['points']).astype('int32')), 1)
        # A crossed hand may be mislabeled as clothing. Landmark interiors
        # remain protected even when semantic labels disagree.
        hands |= cv2mask > 0
        hand_context |= dilate(cv2mask > 0, max(20, round(max(image.size)*0.055)))
    if not detections and np.isin(labels, (14,15)).any():
        # No destructive inference about which arm end is a hand.
        hands |= np.isin(labels, (14,15))
        reasons.append('Hands could not be localized reliably; arm/held-object boundaries need a clearer photo.')
    decisions['detected_hands'] = len(detections)
    bag = labels == 16  # ATR bag category is explicitly protected
    body_context = dilate(labels != 0, max(8, round(max(image.size)*0.02)))
    objects = empty.copy()
    foreground = empty.copy()
    try:
        proposals = clip_proposals(image, {
            'phone': 'a mobile phone held in a hand', 'bag': 'a handbag',
            'strap': 'a shoulder bag strap', 'jewelry': 'jewelry necklace bracelet wristwatch',
            'graphics': 'a watermark logo', 'obstruction': 'a blurry foreground object at the bottom of the photo'})
        objects |= (proposals['phone'] > .22) & hand_context
        bag |= (proposals['bag'] > .3) & body_context
        strap_context = dilate(bag, max(30, round(image.height*.10))) | (body_context & (labels != 0))
        objects |= (proposals['strap'] > .3) & strap_context
        objects |= (proposals['jewelry'] > .35) & (dilate(face | hands, 30) | np.isin(labels,(14,15)))
        foreground |= proposals['graphics'] > .4
        lower = np.indices(labels.shape)[0] > image.height*.75
        foreground |= (proposals['obstruction'] > .4) & lower
    except Exception as exc:
        reasons.append('Object protection unavailable: ' + str(exc))
    text = empty.copy()
    try:
        boxes = detect_text(image)
        text_pixels = np.zeros(labels.shape, np.uint8)
        for box in boxes:
            cv2.fillPoly(text_pixels, [np.rint(box['points']).astype('int32')], 1)
        text = dilate(text_pixels > 0, 2)
        decisions['text_regions'] = len(boxes)
        if (text & garment).sum() > 8:
            reasons.append('Existing text overlaps the clothing. Preserving its source pixels would retain old fabric; use a source without the overlapping graphic.')
    except Exception as exc:
        reasons.append('Text protection unavailable: ' + str(exc))
    if (foreground & garment).sum() > max(20, garment.sum()*.005):
        reasons.append('A graphic or foreground obstruction overlaps clothing and cannot be separated reliably.')
    coverage = empty.copy()
    if intent['long_sleeves'] and target not in ('lower','bottom','pants','skirt'):
        coverage = np.isin(labels, (14,15)) & ~dilate(hands, 4)
    skin = np.isin(labels, SKIN) & ~coverage
    regions = {'face': face, 'hair': dilate(hair, 2), 'hands': dilate(hands, 2),
               'skin': skin, 'bag': dilate(bag, 4), 'objects': dilate(objects, 4),
               'graphics': foreground | text, 'uncertain': uncertain,
               'other_clothes': np.isin(labels, GARMENT) & ~garment,
               'shoes': np.isin(labels, (9,10))}
    protected = np.logical_or.reduce(list(regions.values()))
    approved = (dilate(garment, 3) | coverage) & ~protected
    synthesis = dilate(approved, 10) & ~protected
    # The support includes context; only approved source pixels can be pasted.
    # One-pixel inward edge softening avoids a wide old-clothing fringe.
    distance = cv2.distanceTransform(approved.astype('uint8'), cv2.DIST_L2, 5)
    alpha = np.minimum(distance, 1.0) * 255
    if approved.sum() < image.width*image.height*.005:
        reasons.append('No sufficiently large, safely editable garment region was found.')
    decisions['coordinate_space'] = 'normalized_source'
    return SourceRegions(labels, regions, garment, coverage, approved, synthesis,
                         alpha.astype('uint8'), intent, reasons, decisions)
