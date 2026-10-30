"""Face contours and bounded garment-repair blending regions."""
import threading

import cv2
import numpy as np

from core import config

_landmark_model = None
_landmark_lock = threading.Lock()


def face_contour(image, face):
    """Return a local CPU landmark contour; no face swap/restoration models load."""
    global _landmark_model
    path = config.INSIGHTFACE_DIR / 'buffalo_l' / '2d106det.onnx'
    if not path.is_file():
        return None
    from insightface.model_zoo import get_model
    from insightface.app.common import Face
    x, y, w, h = face['box']
    with _landmark_lock:
        if _landmark_model is None:
            _landmark_model = get_model(str(path), download=False, providers=['CPUExecutionProvider'])
        points = _landmark_model.get(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR),
                                    Face(bbox=np.array([x, y, x+w, y+h])))
    if not np.isfinite(points).all():
        return None
    # Extend the landmark hull to cover the forehead while keeping its curved jaw.
    points = np.vstack([points, [x, y], [x+w, y]])
    mask = np.zeros((image.height, image.width), dtype='uint8')
    cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(points).astype('int32')), 255)
    return mask > 0


def composite_alpha(candidate, protected, radius=14):
    """Replace the whole old garment and fade outward into known background.

    Inward feathering would mix original fabric back into a changed garment.
    Foreground protection remains exact; image edges do not acquire a false rim.
    """
    candidate = np.asarray(candidate, dtype=bool) & ~protected
    if not candidate.any():
        return np.zeros(candidate.shape, dtype='uint8')
    distance = cv2.distanceTransform((~candidate).astype('uint8'), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    alpha = np.clip(1-distance/max(1, radius), 0, 1)
    alpha = alpha*alpha*(3-2*alpha)
    alpha[protected] = 0
    return np.rint(alpha*255).astype('uint8')


