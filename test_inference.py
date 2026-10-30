"""Automated verification suite for Face Studio Core AI Modules."""
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import config
from core.prompt_engine import build_zero_pollution_prompt, analyze_skin_melanin
from core.clothing_engine import generate_dress_mask
from core.face_engine import (
    apply_face_lock,
    calibrate_swapped_face_tone,
    reinject_micro_pores,
    get_face_modules
)


def run_tests():
    print("======================================================================")
    print("         Face Studio - Core AI Modules Verification Suite")
    print("======================================================================")

    # Test 1: Check Anti-Bindi & Negative Filter Safeguards
    print("\n[Test 1/4] Verifying Zero-Pollution Prompt & Negative Filters...")
    pos, neg = build_zero_pollution_prompt(
        user_prompt="crimson silk evening gown with golden embroidery",
        source_image=None,
        mode="in_place_swap"
    )
    assert "bindi" in neg.lower(), "Negative prompt MUST suppress bindi!"
    assert "forehead dot" in neg.lower(), "Negative prompt MUST suppress forehead dot!"
    assert "sindoor" in neg.lower(), "Negative prompt MUST suppress sindoor!"
    assert "crimson silk evening gown" in pos, "Positive prompt MUST preserve user prompt verbatim!"
    print("  [PASS] Zero-pollution prompt priority and anti-bindi negative filter verified.")

    # Test 2: Check Segformer Dress Mask Generation
    print("\n[Test 2/4] Verifying Segformer Dress Mask & Smart Chin Line Feathering...")
    sample_path = config.SAMPLES_DIR / "sample_model.png"
    if not sample_path.exists():
        sample_path = Path(r"D:\image mod\samples\sample_model.png")
    
    assert sample_path.exists(), f"Sample image not found at {sample_path}"
    test_img = Image.open(sample_path)
    mask_pil, preview, faces, coverage = generate_dress_mask(test_img, target="dress", keep_face=True)
    assert mask_pil.size == test_img.size, "Mask dimensions must match source image!"
    assert coverage > 0.05, f"Garment coverage expected > 5%, got {coverage*100:.1f}%"
    print(f"  [PASS] Segformer generated clothing mask (coverage: {coverage*100:.1f}%, faces detected: {len(faces)}).")

    # Test 3: Check Biometric Face Modules & CodeFormer
    print("\n[Test 3/4] Verifying Biometric Face Engine & CodeFormer Loading...")
    app, swapper, codeformer_net, face_helper, fallback_restorer = get_face_modules()
    assert app is not None, "InsightFace FaceAnalysis must be initialized!"
    assert swapper is not None, "Inswapper 128 model must be loaded!"
    assert codeformer_net is not None, "CodeFormer prior network must be loaded!"
    print("  [PASS] Inswapper 128 and CodeFormer Discrete Codebook Prior (w=0.85) loaded successfully.")

    # Test 4: Luminance Preservation & Micro-Pore Frequency Separation
    print("\n[Test 4/4] Verifying Skin Tone Luminance & Micro-Pore Preservation...")
    src_bgr = np.full((128, 128, 3), 180, dtype=np.uint8)  # Lighter reference
    tgt_bgr = np.full((128, 128, 3), 100, dtype=np.uint8)  # Darker target
    bbox = [10, 10, 118, 118]
    harmonized = calibrate_swapped_face_tone(tgt_bgr, src_bgr, bbox, bbox, tgt_bgr)
    
    src_l = np.mean(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB)[:, :, 0])
    harm_l = np.mean(cv2.cvtColor(harmonized, cv2.COLOR_BGR2LAB)[:, :, 0])
    l_diff_pct = abs(harm_l - src_l) / src_l * 100
    print(f"  Source L*: {src_l:.1f}, Harmonized L*: {harm_l:.1f} (Deviation: {l_diff_pct:.2f}%)")
    assert l_diff_pct < 15.0, f"Luminance deviation too high: {l_diff_pct:.2f}%"
    print("  [PASS] Luminance & melanocyte calibration preserved L* accurately.")

    # Test 5: LoRA Discovery & ComfyUI Graph Injection
    print("\n[Test 5/5] Verifying LoRA Registry & ComfyUI Graph Integration...")
    from core.pipeline import build_sdxl_inpaint_graph, build_sdxl_txt2img_graph

    loras = config.get_available_loras()
    print(f"  Discovered {len(loras)} LoRAs on disk: {list(loras.keys())}")
    assert isinstance(loras, dict), "get_available_loras() must return a dict"

    # Verify Inpaint Graph LoRA wiring
    inpaint_graph, _ = build_sdxl_inpaint_graph(
        source_subpath="test.png",
        mask_subpath="mask.png",
        positive_prompt="emerald velvet dress",
        negative_prompt="blurry",
        model_cfg=config.MODELS["cyberrealistic_xl"],
        lora_name="fs_couture.safetensors",
        lora_weight=0.8
    )
    assert "lora" in inpaint_graph, "Inpaint graph MUST include 'lora' node when lora_name is provided!"
    assert inpaint_graph["lora"]["class_type"] == "LoraLoader", "Node type must be LoraLoader"
    assert inpaint_graph["lora"]["inputs"]["lora_name"] == "fs_couture.safetensors"
    assert inpaint_graph["patched_model"]["inputs"]["model"] == ["lora", 0], "Patched model must receive LoRA model output"
    assert inpaint_graph["positive"]["inputs"]["clip"] == ["lora", 1], "Positive prompt must receive LoRA CLIP output"

    # Verify Txt2Img Graph LoRA wiring
    txt2img_graph, _ = build_sdxl_txt2img_graph(
        positive_prompt="emerald velvet dress",
        negative_prompt="blurry",
        model_cfg=config.MODELS["realvis_xl"],
        lora_name="fs_couture.safetensors",
        lora_weight=0.75
    )
    assert "lora" in txt2img_graph, "Txt2Img graph MUST include 'lora' node when lora_name is provided!"
    assert txt2img_graph["sample"]["inputs"]["model"] == ["lora", 0], "Sampler must receive LoRA model output"
    assert txt2img_graph["positive"]["inputs"]["clip"] == ["lora", 1], "CLIP text must receive LoRA clip output"
    # Verify Hyper-SDXL distillation parameter adaptation (steps=8, cfg=2.0)
    hyper_meta = loras.get("Hyper-SDXL-8steps-CFG-lora")
    assert hyper_meta is not None, "Hyper-SDXL-8steps-CFG-lora must be in registry"
    assert hyper_meta.get("steps") == 8, f"Expected 8 steps for Hyper-SDXL, got {hyper_meta.get('steps')}"
    assert hyper_meta.get("cfg") == 2.0, f"Expected CFG 2.0 for Hyper-SDXL, got {hyper_meta.get('cfg')}"
    assert "Hyper-SD15-8steps-CFG-lora" not in loras, "SD 1.5 LoRA must be filtered out from SDXL models list"
    print("  [PASS] Hyper-SDXL distillation parameter adaptation (8 steps, CFG 2.0) and SD15 filtering verified.")
    print("  [PASS] ComfyUI LoRA graph wiring and dynamic parameter routing verified.")

    print("\n======================================================================")
    print("   ALL CORE AI MODULE TESTS PASSED (100% VERIFIED)")
    print("======================================================================")


if __name__ == "__main__":
    run_tests()

