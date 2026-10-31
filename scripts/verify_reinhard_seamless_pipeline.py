"""Verification test script for Reinhard Lab Skin Tone Transfer and Poisson Seamless Face Blend.

Evaluates:
1. Melanin Level & Skin Tone Preservation (Source Face vs. Output Face vs. Output Body).
2. Elimination of Skin Lightening Bias on generated bodies/limbs.
3. Boundary Continuity across Face-to-Neck Seams (Poisson Seamless Blend vs. Edge Steps).
4. Facial & Dermal Micro-Texture Realism (Anti-Plasticity Index).
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from core import config
from core.face_engine import (
    get_face_modules,
    detect_faces_robust,
    extract_face_skin_profile,
    match_skin_tone_lab,
    seamless_face_blend,
    apply_face_lock
)
from core.clothing_engine import parse_clothing_labels
from core.skin_engine import harmonize_unified_canvas_skin


def compute_lab_delta_e(lab1: np.ndarray, lab2: np.ndarray, chroma_only: bool = False) -> float:
    """Computes Euclidean Delta-E in CIE-Lab space."""
    if chroma_only:
        return float(np.sqrt(np.sum((lab1[1:].astype(np.float32) - lab2[1:].astype(np.float32)) ** 2)))
    return float(np.sqrt(np.sum((lab1.astype(np.float32) - lab2.astype(np.float32)) ** 2)))


def run_verification(src_path: Path, scene_path: Path):
    print("=" * 80)
    print("VERIFICATION: REINHARD LAB SKIN TRANSFER & POISSON SEAMLESS FACE BLEND")
    print(f"Source Image: {src_path}")
    print(f"Target Scene: {scene_path}")
    print("=" * 80)

    src_img = Image.open(src_path).convert("RGB")
    scene_img = Image.open(scene_path).convert("RGB")

    src_bgr = cv2.cvtColor(np.array(src_img), cv2.COLOR_RGB2BGR)
    scene_bgr = cv2.cvtColor(np.array(scene_img), cv2.COLOR_RGB2BGR)

    app, _, _, _, _ = get_face_modules()

    # Detect faces in source image
    src_faces = detect_faces_robust(app, src_bgr)
    assert src_faces, "No face detected in source image!"
    src_face = sorted(src_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[0]
    src_face_lab, src_face_std = extract_face_skin_profile(src_bgr, src_face.bbox)
    print(f"\n[Source Anchor] Face Skin LAB: {src_face_lab.round(2).tolist()}")

    # 1. Test match_skin_tone_lab on Target Body Skin
    # Parse clothing and skin segmentation
    h, w = scene_bgr.shape[:2]
    tgt_labels = parse_clothing_labels(scene_img)
    if tgt_labels.shape[:2] != (h, w):
        tgt_labels = cv2.resize(tgt_labels, (w, h), interpolation=cv2.INTER_NEAREST)

    # Body skin: labels 11 (below chin), 12, 13 (legs), 14, 15 (arms)
    tf_faces = detect_faces_robust(app, scene_bgr)
    if tf_faces:
        tf_box = tf_faces[0].bbox
        rx1, ry1, rx2, ry2 = [int(v) for v in tf_box]
        bw, bh = rx2 - rx1, ry2 - ry1
        face_ellipse = np.zeros((h, w), dtype=bool)
        cv2.ellipse(face_ellipse.view(np.uint8), ((rx1 + rx2) // 2, (ry1 + ry2) // 2),
                    (int(bw * 0.48), int(bh * 0.50)), 0, 0, 360, 1, -1)
        face_ellipse = face_ellipse > 0
        torso_skin = (tgt_labels == 11) & (np.arange(h)[:, None] > (ry2 + int(bh * 0.05)))
    else:
        face_ellipse = np.zeros((h, w), dtype=bool)
        torso_skin = (tgt_labels == 11)

    arm_skin = np.isin(tgt_labels, (14, 15))
    leg_skin = np.isin(tgt_labels, (12, 13))
    raw_body_skin = (arm_skin | leg_skin | torso_skin) & ~face_ellipse

    # Dermatological validity check
    ycrcb = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
    valid_body_skin = raw_body_skin & (cr >= 120) & (cr <= 190) & (cb >= 65) & (cb <= 145) & (gray >= 30) & (gray <= 250)

    # Bridge sheer fabric folds
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tgt_skin_mask = cv2.morphologyEx((valid_body_skin.astype(np.uint8) * 255), cv2.MORPH_CLOSE, close_kernel)

    # Source skin mask
    src_labels = parse_clothing_labels(src_img)
    if src_labels.shape[:2] != src_bgr.shape[:2]:
        src_labels = cv2.resize(src_labels, (src_bgr.shape[1], src_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    src_skin_mask = np.isin(src_labels, (11, 12, 13, 14, 15)).astype(np.uint8) * 255

    # Run match_skin_tone_lab
    matched_scene_bgr = match_skin_tone_lab(
        source_img_bgr=src_bgr,
        target_img_bgr=scene_bgr,
        source_mask=src_skin_mask,
        target_mask=tgt_skin_mask
    )

    # 2. Run apply_face_lock with CodeFormer fidelity <= 0.5 and seamless_face_blend
    face_locked_img = apply_face_lock(
        source_image=src_img,
        target_image=Image.fromarray(cv2.cvtColor(matched_scene_bgr, cv2.COLOR_BGR2RGB)),
        fidelity_weight=0.45,
        enable_pores=True,
        enable_grain=True,
        src_faces=src_faces
    )

    # 3. Post-harmonization
    final_img = harmonize_unified_canvas_skin(
        canvas_image=face_locked_img,
        face_bbox=tf_box if tf_faces else None,
        blend_strength=0.85,
        align_luminance=True
    )

    final_bgr = cv2.cvtColor(np.array(final_img), cv2.COLOR_RGB2BGR)
    final_lab = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Extract final face and body metrics
    fin_faces = detect_faces_robust(app, final_bgr)
    assert fin_faces, "No face detected in final image!"
    fin_face = sorted(fin_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[0]
    fin_face_lab, fin_face_std = extract_face_skin_profile(final_bgr, fin_face.bbox)

    fin_body_lab = np.mean(final_lab[valid_body_skin], axis=0)

    # 4. Compute Metrics
    delta_e_face_src = compute_lab_delta_e(src_face_lab, fin_face_lab, chroma_only=True)
    delta_e_body_src = compute_lab_delta_e(src_face_lab, fin_body_lab, chroma_only=True)
    delta_e_face_body = compute_lab_delta_e(fin_face_lab, fin_body_lab, chroma_only=True)

    # Seam continuity gradient across jawline
    fx1, fy1, fx2, fy2 = [int(v) for v in fin_face.bbox]
    jaw_y = fy2
    jaw_band = final_lab[max(0, jaw_y - 10):min(h, jaw_y + 10), max(0, fx1):min(w, fx2)]
    ga_y = cv2.Sobel(jaw_band[:, :, 1], cv2.CV_32F, 0, 1, ksize=3)
    gb_y = cv2.Sobel(jaw_band[:, :, 2], cv2.CV_32F, 0, 1, ksize=3)
    jaw_seam_grad = float(np.mean(cv2.magnitude(ga_y, gb_y)))

    # Dermal texture variance (anti-plasticity)
    final_gray = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(final_gray, cv2.CV_64F)
    face_x1, face_y1, face_x2, face_y2 = [int(v) for v in fin_face.bbox]
    face_gray_crop = final_gray[face_y1:face_y2, face_x1:face_x2]
    face_texture_var = float(np.var(cv2.Laplacian(face_gray_crop, cv2.CV_64F)))
    body_texture_var = float(np.var(laplacian[valid_body_skin]))

    print("\n--- Quantitative Evaluation ---")
    print(f"Final Face Skin LAB:            {fin_face_lab.round(2).tolist()}")
    print(f"Final Body Skin LAB:            {fin_body_lab.round(2).tolist()}")
    print(f"Chroma Delta-E (Input Face vs Output Face): {delta_e_face_src:.2f} (Threshold: <= 4.0) -> {'PASS' if delta_e_face_src <= 4.0 else 'FAIL'}")
    print(f"Chroma Delta-E (Input Face vs Output Body): {delta_e_body_src:.2f} (Threshold: <= 4.0) -> {'PASS' if delta_e_body_src <= 4.0 else 'FAIL'}")
    print(f"Chroma Delta-E (Output Face vs Output Body): {delta_e_face_body:.2f} (Threshold: <= 3.5) -> {'PASS' if delta_e_face_body <= 3.5 else 'FAIL'}")
    print(f"Jawline Seam Vertical Gradient: {jaw_seam_grad:.2f} (Threshold: <= 25.0) -> {'PASS' if jaw_seam_grad <= 25.0 else 'FAIL'}")
    print(f"Facial Skin Texture Variance:   {face_texture_var:.2f} (Threshold: >= 50.0) -> {'PASS' if face_texture_var >= 50.0 else 'FAIL'}")
    print(f"Body Skin Texture Variance:     {body_texture_var:.2f} (Threshold: >= 25.0) -> {'PASS' if body_texture_var >= 25.0 else 'FAIL'}")

    all_pass = (delta_e_face_src <= 4.0) and (delta_e_body_src <= 4.0) and (delta_e_face_body <= 3.5) and (jaw_seam_grad <= 25.0) and (face_texture_var >= 50.0) and (body_texture_var >= 25.0)

    print("-" * 80)
    print(f"OVERALL STATUS: {'[PASS] ALL CRITERIA MET' if all_pass else '[FAIL] CRITERIA NOT MET'}")
    print("=" * 80)

    out_save_path = Path("D:/face_studio/outputs/verify_reinhard_seamless_result.png")
    final_img.save(out_save_path)
    print(f"Saved verification output to: {out_save_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    src = Path("D:/face_studio/inputs/src_820663bb.png")
    scene = Path("D:/face_studio/outputs/out_820663bb_cyberrealistic_xl_full_scene.png")
    sys.exit(run_verification(src, scene))

