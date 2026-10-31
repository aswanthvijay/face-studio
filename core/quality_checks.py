"""Small, inspectable quality measurements; scores are not identity probabilities."""
import cv2
import numpy as np
from PIL import Image


def face_similarity(source_face, target_face) -> float | None:
    source = getattr(source_face, "normed_embedding", None)
    target = getattr(target_face, "normed_embedding", None)
    if source is None or target is None:
        return None
    return float(np.dot(source, target))


def garment_color_check(image: Image.Image, labels: np.ndarray, color: str) -> dict:
    garment = np.isin(labels, (4,5,6,7))
    if np.count_nonzero(garment) < 200:
        return {"requested": color, "matches": None, "reason": "insufficient garment pixels"}
    rgb = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb.astype(np.float32)/255.0, cv2.COLOR_RGB2LAB)
    lightness = float(np.median(lab[:,:,0][garment]))
    saturation = float(np.median(hsv[:,:,1][garment]))
    colorful = garment & (hsv[:,:,1] > 20)
    hue = float(np.median(hsv[:,:,0][colorful])) if np.count_nonzero(colorful) > 100 else None
    matches = None
    if color == "black":
        matches = lightness < 45.0
    elif color in ("white", "ivory", "cream"):
        matches = lightness > 65.0 and saturation < 75.0
    elif color == "peach" and hue is not None:
        warmth = np.median(lab[:,:,1:][garment],axis=0)
        matches = bool(2.0 <= hue <= 25.0 and lightness > 45.0 and warmth[0] > 4.0 and warmth[1] > 8.0)
    return {"requested": color, "matches": matches, "median_lightness": lightness,
            "median_saturation": saturation, "median_hue": hue}
