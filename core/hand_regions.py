"""Optional local hand landmarks for garment-mask foreground protection."""
import threading
import cv2
import numpy as np
from core import config

_models = None
_lock = threading.Lock()


def detect_hands(image, labels):
    global _models
    folder = config.WORKSPACE_ROOT/'models'/'hands'
    palm_file = folder/'palm_detection_mediapipe_2023feb.onnx'
    hand_file = folder/'handpose_estimation_mediapipe_2023feb.onnx'
    if not palm_file.is_file() or not hand_file.is_file():
        return []
    rgb = np.asarray(image.convert('RGB'))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = labels.shape
    # Tiled views retain small hands on full-body images. Only tiles containing
    # semantic arm/hand pixels are examined, and overlapping detections merge.
    side = min(512, max(h, w))
    xs = sorted(set(list(range(0, max(1, w-side), max(1, side*3//4))) + [max(0, w-side)]))
    ys = sorted(set(list(range(0, max(1, h-side), max(1, side*3//4))) + [max(0, h-side)]))
    regions = [(x,y,min(w,x+side),min(h,y+side)) for y in ys for x in xs
               if np.count_nonzero(np.isin(labels[y:y+side,x:x+side], (14,15))) > 100]
    hands = []
    with _lock:
        if _models is None:
            from core.vendor.opencv_mediapipe.mp_palmdet import MPPalmDet
            from core.vendor.opencv_mediapipe.mp_handpose import MPHandPose
            _models = (MPPalmDet(str(palm_file),scoreThreshold=0.6), MPHandPose(str(hand_file),confThreshold=0.8))
        palm_model, hand_model = _models
        for x,y,x2,y2 in regions:
            crop = bgr[y:y2,x:x2]
            for palm in palm_model.infer(crop):
                hand = hand_model.infer(crop, palm)
                if hand is None:
                    continue
                points = hand[4:67].reshape(21,3)[:,:2] + [x,y]
                if not np.isfinite(points).all():
                    continue
                span = np.linalg.norm(np.ptp(points,axis=0))
                if span < 15:
                    continue
                center = points.mean(axis=0)
                if any(np.linalg.norm(center - old['points'].mean(axis=0)) < span*0.4 for old in hands):
                    continue
                hands.append({'points':points,'confidence':float(hand[-1])})
    return hands


def hand_protection(image, labels):
    """Select semantic hand pixels around detected landmarks, preserving gaps."""
    hands = detect_hands(image, labels)
    mask = np.zeros(labels.shape, dtype='uint8')
    for hand in hands:
        points = hand['points']
        extent = np.linalg.norm(np.ptp(points,axis=0))
        radius = max(3, round(extent*0.12))
        region = np.zeros_like(mask)
        cv2.fillConvexPoly(region, cv2.convexHull(np.rint(points).astype('int32')), 1)
        region = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,)*2))
        mask |= region
    return (mask > 0) & np.isin(labels,(14,15)), hands
