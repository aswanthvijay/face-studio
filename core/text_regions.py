"""Local PP-OCRv3 text boxes. Detection is not recognition or alpha matting."""
import hashlib
import json
import threading
import cv2
import numpy as np
from core import config

_model = None
_lock = threading.Lock()


def detect_text(image):
    global _model
    folder = config.WORKSPACE_ROOT / 'models' / 'text'
    with _lock:
        if _model is None:
            asset = json.loads((folder / 'manifest.json').read_text())[0]
            path = folder / asset['file']
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != asset['sha256']:
                raise RuntimeError('Text protection unavailable: run scripts/setup_outfit_text_detector.py')
            detector = cv2.dnn_TextDetectionModel_DB(cv2.dnn.readNet(str(path)))
            detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            detector.setBinaryThreshold(0.3)
            detector.setPolygonThreshold(0.5)
            detector.setUnclipRatio(2.0)
            detector.setMaxCandidates(200)
            detector.setInputSize((736, 736))
            detector.setInputMean((123.675, 116.28, 103.53))
            detector.setInputScale(1.0 / 255.0 / np.array([0.229, 0.224, 0.225]))
            _model = detector
        # Aspect-preserving letterbox; map polygons back to normalized source pixels.
        rgb = np.asarray(image.convert('RGB'))
        h, w = rgb.shape[:2]
        scale = 736 / max(w, h)
        sw, sh = round(w * scale), round(h * scale)
        canvas = np.zeros((736, 736, 3), np.uint8)
        canvas[:sh, :sw] = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (sw, sh))
        polygons, scores = _model.detect(canvas)
        return [{'points': np.clip(np.asarray(p, dtype=float) / scale, [0, 0], [w-1, h-1]),
                 'confidence': float(s)} for p, s in zip(polygons, scores)]
