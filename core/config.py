import os
from pathlib import Path

# Workspace Root
# Workspace Root dynamically resolved (Bug B12)
_env_root = os.environ.get("FACE_STUDIO_ROOT")
if _env_root:
    ROOT_DIR = Path(_env_root).resolve()
elif Path(r"D:\face_studio").exists():
    ROOT_DIR = Path(r"D:\face_studio").resolve()
else:
    ROOT_DIR = Path(__file__).resolve().parent.parent

WORKSPACE_ROOT = ROOT_DIR
IMAGE_AI_ROOT = Path(r"D:\Image AI L model").resolve()
STABILITY_MATRIX_ROOT = Path(r"D:\Image AI\StabilityMatrix-win-x64").resolve()

# Python Runtime
COMFY_PYTHON = STABILITY_MATRIX_ROOT / "Data" / "Packages" / "ComfyUI" / "venv" / "Scripts" / "python.exe"
PYTHON_EXE = COMFY_PYTHON if COMFY_PYTHON.exists() else (IMAGE_AI_ROOT / "runtime" / "python.exe")
if not PYTHON_EXE.exists():
    PYTHON_EXE = STABILITY_MATRIX_ROOT / "Data" / "Assets" / "Python" / "cpython-3.12.14-windows-x86_64-none" / "python.exe"

# Backend configuration
COMFYUI_DIR = STABILITY_MATRIX_ROOT / "Data" / "Packages" / "ComfyUI"
BACKEND_PORT = int(os.environ.get("FACE_STUDIO_BACKEND_PORT", 8189))
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"
# The bundled offline runtime rejects wildcard/LAN socket binds.
# Expose this loopback service through the Windows LAN forwarding helper.
HOST = os.environ.get("FACE_STUDIO_HOST", "127.0.0.1")
PORT = int(os.environ.get("FACE_STUDIO_PORT", 7870))

# Model Paths
MODELS_DIR = STABILITY_MATRIX_ROOT / "Data" / "Models" / "StableDiffusion"
DIFFUSION_MODELS_DIR = STABILITY_MATRIX_ROOT / "Data" / "Models" / "DiffusionModels"
TEXT_ENCODERS_DIR = STABILITY_MATRIX_ROOT / "Data" / "Models" / "TextEncoders"
VAE_DIR = STABILITY_MATRIX_ROOT / "Data" / "Models" / "VAE"
ASSETS_DIR = Path(os.environ.get("FACE_STUDIO_ASSETS_DIR", r"D:\Photovance AI\models"))
CLOTHING_PARSER_DIR = ASSETS_DIR / "clothing-parser" if (ASSETS_DIR / "clothing-parser").exists() else (IMAGE_AI_ROOT / "assets" / "clothing-parser")
CLIPSEG_DIR = Path(os.environ.get("FACE_STUDIO_CLIPSEG_DIR", WORKSPACE_ROOT / "models" / "clipseg"))
FACE_DETECTOR_PATH = ASSETS_DIR / "face_detection_yunet_2023mar.onnx" if (ASSETS_DIR / "face_detection_yunet_2023mar.onnx").exists() else (IMAGE_AI_ROOT / "assets" / "face_detection_yunet_2023mar.onnx")
INSIGHTFACE_DIR = Path(os.environ.get("FACE_STUDIO_INSIGHTFACE_DIR", Path.home() / ".insightface" / "models"))
LORA_DIR = STABILITY_MATRIX_ROOT / "Data" / "Models" / "Lora"
LOCAL_LORA_DIR = WORKSPACE_ROOT / "models" / "loras"
LOCAL_LORA_DIR.mkdir(parents=True, exist_ok=True)

# VAE
SD_VAE = "vae-ft-mse-840000-ema-pruned.safetensors"
FLUX_VAE = "flux2-vae.safetensors"
QWEN_VAE = "qwen_image_2.1_vae_bf16.safetensors"
ZIMAGE_VAE = "ae.safetensors"

# Dynamic LoRA Registry
def get_available_loras() -> dict:
    """Scan and list all available LoRA models from StabilityMatrix and local directory."""
    loras = {}
    known_metadata = {
        "fs_couture": {
            "compat": "sdxl",
            "name": "Haute Couture & Designer Gowns",
            "trigger": "fs_couture, luxury haute couture embroidery, elegant draped fabric",
            "default_weight": 0.8,
            "badge": "Couture Specialist",
            "category": "fashion",
            "desc": "Custom trained luxury fashion LoRA for elegant drapes and embroidery."
        },
        "Hyper-SDXL-8steps-CFG-lora": {
            "compat": "sdxl",
            "name": "Hyper-SDXL 8-Step Turbo",
            "trigger": "",
            "default_weight": 0.85,
            "badge": "Turbo Speed",
            "category": "speed",
            "steps": 8,
            "cfg": 2.0,
            "sampler": "euler",
            "scheduler": "karras",
            "desc": "ByteDance 8-step distillation. Automatically uses 8 steps at CFG 2.0 to eliminate distortion and blur."
        },
        "sd_xl_offset_example-lora_1.0": {
            "compat": "sdxl",
            "name": "SDXL High Dynamic Contrast & Shadow",
            "trigger": "",
            "default_weight": 0.6,
            "badge": "Lighting & Contrast",
            "category": "style",
            "desc": "Enhances true deep shadows and dynamic range in studio lighting."
        },
        "skin_tone_style_zib_v1.1": {
            "name": "Z-Image Skin Melanin Sync (Civitai #562884 - Z-Image Turbo Only)",
            "trigger": "",
            "default_weight": 0.75,
            "badge": "Z-Image Only",
            "category": "skin",
            "compat": "zimage",
            "desc": "Bespoke melanin & epidermal complexion adapter for Z-Image Turbo only."
        },
        "skin_tone_style_v4": {
            "compat": "sdxl",
            "name": "SDXL Skin Melanin Sync (Civitai #562884)",
            "trigger": "skin tone style",
            "default_weight": 0.75,
            "badge": "Melanin Sync",
            "category": "skin",
            "desc": "Calibrated human skin melanin matching from fair to bronze/brown without whitewashing."
        },
        "skin_texture_style_v4": {
            "compat": "sdxl",
            "name": "SDXL Realistic Micro-Pores (Civitai #580857)",
            "trigger": "skin texture style, detailed skin pore, realistic skin",
            "default_weight": 0.70,
            "badge": "Micro-Pores",
            "category": "skin",
            "desc": "Restores epidermal micro-pores, fine dermal grain, and natural sub-surface scattering."
        },
        "skin_tone_slider-qwen-2.1-v1": {
            "compat": "qwen",
            "name": "Qwen 2.1 Skin Tone Slider (Civitai #766017)",
            "trigger": "Dark skin tone, Pale skin tone",
            "default_weight": 0.75,
            "badge": "Melanin Slider",
            "category": "skin",
            "desc": "Continuous scalar melanin slider for Qwen 2.1 Image generation."
        },
        "ImpurePores": {
            "compat": "flux",
            "name": "Flux Natural Micro-Pores (Civitai #902169)",
            "trigger": "xlr_skin",
            "default_weight": 0.65,
            "badge": "Flux Micro-Pores",
            "category": "skin",
            "desc": "Ultra-lightweight pore and dermal roughness injector for Flux models."
        },
        "moss_image_color_depth_control": {
            "name": "MOSS Color & Depth Control (OpenMOSS)",
            "trigger": "cinematic lighting, rich color depth, volumetric contrast",
            "default_weight": 0.70,
            "badge": "MOSS Specialist",
            "category": "style",
            "compat": "moss",
            "desc": "Official OpenMOSS styling adapter for enhanced color depth, shadow range, and cinematic composition."
        }
    }

    search_dirs = [LORA_DIR, LOCAL_LORA_DIR]
    for directory in search_dirs:
        if not directory.exists():
            continue
        for file in directory.glob("*.safetensors"):
            # Ignore corrupt or tiny stub downloads (e.g. Civitai 401 error pages)
            if file.stat().st_size < 100_000:
                continue
            stem = file.stem
            # Exclude SD 1.5 LoRAs from the SDXL model suite
            stem_lower = stem.lower()
            if "sd15" in stem_lower or "sd1.5" in stem_lower or "sd-1-5" in stem_lower:
                continue

            meta = known_metadata.get(stem, {
                "name": stem.replace("_", " ").replace("-", " ").title(),
                "trigger": stem.lower(),
                "default_weight": 0.75,
                "badge": "Custom LoRA",
                "desc": "Custom LoRA styling adapter."
            })
            item = {
                "id": stem,
                "filename": file.name,
                "name": meta["name"],
                "trigger": meta.get("trigger", ""),
                "default_weight": meta.get("default_weight", 0.75),
                "badge": meta.get("badge", "LoRA"),
                "category": meta.get("category", "fashion"),
                "desc": meta.get("desc", ""),
                "path": str(file)
            }
            for opt_key in ("steps", "cfg", "sampler", "scheduler", "compat"):
                if opt_key in meta:
                    item[opt_key] = meta[opt_key]
            loras[stem] = item
    return loras


def resolve_lora(lora_id: str, model_type: str) -> dict | None:
    """Resolve IDs and filenames through the same installed adapter registry."""
    if not lora_id or str(lora_id).lower() in ("none", "null", ""):
        return None
    available = get_available_loras()
    meta = available.get(lora_id)
    if meta is None:
        meta = next((item for item in available.values() if item["filename"] == lora_id), None)
    if meta is None:
        raise ValueError(f"LoRA '{lora_id}' is not installed.")
    compat = meta.get("compat")
    if compat is None:
        name = meta["filename"].lower()
        compat = next((family for family, words in (
            ("zimage", ("zimage", "zib", "auraflow")),
            ("qwen", ("qwen",)), ("flux", ("flux",)), ("moss", ("moss",)),
            ("sdxl", ("sdxl", "sd_xl")),
        ) if any(word in name for word in words)), None)
    if compat != model_type:
        required = compat or "an unspecified architecture"
        raise ValueError(f"LoRA '{lora_id}' requires {required}; selected model uses {model_type}.")
    if model_type == "flux":
        raise ValueError("LoRA loading is not supported by the Flux-Klein workflow.")
    return meta


# Helper function to find best matching checkpoint file
def get_best_checkpoint(primary_name: str, fallback_names: list) -> str:
    for name in [primary_name] + fallback_names:
        if (MODELS_DIR / name).exists():
            return name
    return primary_name


def get_best_moss_unet() -> str:
    """Find available MOSS-Image DiT weights with graceful fallback to Flux-2-Klein."""
    candidates = [
        "moss_image_sft_int8_convrot.safetensors",
        "moss_image_sft_bf16.safetensors",
        "moss_image_base_int8_convrot.safetensors",
        "moss_image_base_bf16.safetensors"
    ]
    for c in candidates:
        if (DIFFUSION_MODELS_DIR / c).exists():
            return c
    # If MOSS weights are currently downloading, fallback gracefully to existing Flux-2-Klein
    if (DIFFUSION_MODELS_DIR / "flux-2-klein-4b-fp8.safetensors").exists():
        return "flux-2-klein-4b-fp8.safetensors"
    return "moss_image_sft_int8_convrot.safetensors"


def get_best_moss_clip() -> str:
    """Find available Qwen3.5/Qwen3 text encoder for MOSS-Image."""
    candidates = [
        "qwen3.5_9b_int8_convrot.safetensors",
        "qwen3.5_9b_w4a8_gptq.safetensors",
        "qwen3.5_9b_bf16.safetensors",
        "qwen3-4b-heretic_fp8_e4m3fn.safetensors",
        "qwen_3_4b.safetensors"
    ]
    for c in candidates:
        if (TEXT_ENCODERS_DIR / c).exists():
            return c
    return "qwen3-4b-heretic_fp8_e4m3fn.safetensors"

# Streamlined Top-3 Photorealistic Model Suite (CyberRealistic XL Anchor)
MODELS = {
    "cyberrealistic_xl": {
        "id": "cyberrealistic_xl",
        "name": "CyberRealistic XL (DSLR Pores & Natural Skin)",
        "badge": "Top Photorealism ★",
        "desc": "True DSLR micro-pores, natural epidermal depth, zero AI waxy look.",
        "type": "sdxl",
        "checkpoint": get_best_checkpoint(
            "CyberRealisticXLPlay_V7.0_FP16.safetensors",
            ["RealVisXL_V5.0_fp16.safetensors", "epicrealismXL_v7FinalDestination.safetensors", "juggernautXL_v8Rundiffusion.safetensors"]
        ),
        "target_size": 1024,
        "cfg": 5.5,
        "steps": 25,
        "sampler": "dpmpp_2m",
        "scheduler": "karras"
    },
    "realvis_xl": {
        "id": "realvis_xl",
        "name": "RealVisXL V5.0 (Studio Portrait Benchmark)",
        "badge": "Studio Portrait Specialist",
        "desc": "Razor-sharp photorealism, lifelike eye catchlights, and cinematic studio lighting.",
        "type": "sdxl",
        "checkpoint": "RealVisXL_V5.0_fp16.safetensors",
        "target_size": 896,
        "cfg": 4.5,
        "steps": 18,
        "sampler": "dpmpp_2m",
        "scheduler": "karras"
    },
    "flux_klein": {
        "id": "flux_klein",
        "name": "Flux-2-Klein 4B Turbo (Wardrobe Master DiT)",
        "badge": "Wardrobe Master DiT",
        "desc": "Next-gen Diffusion Transformer with Qwen3 Heretic text encoder for unrestricted prompt adherence and natural fabric drape.",
        "type": "flux",
        "unet": "flux-2-klein-4b-fp8.safetensors",
        "clip": "qwen3-4b-heretic_fp8_e4m3fn.safetensors",
        "vae": FLUX_VAE,
        "target_size": 896,
        "steps": 4,
        "sampler": "euler"
    }
}

DEFAULT_MODEL = "cyberrealistic_xl"

# Cleanly Archived Inactive Models (Preserved for Reference)
ARCHIVED_MODELS = {
    "epicrealism_xl": {
        "id": "epicrealism_xl",
        "name": "epiCRealism XL (Cinematic Portraits & Studio Light)",
        "badge": "Portrait Specialist",
        "desc": "Studio lighting, cinematic contrast, lifelike hair and authentic fabric drape.",
        "type": "sdxl",
        "checkpoint": get_best_checkpoint(
            "epicrealismXL_v7FinalDestination.safetensors",
            ["juggernautXL_v8Rundiffusion.safetensors", "RealVisXL_V5.0_fp16.safetensors"]
        ),
        "target_size": 896,
        "cfg": 5.0,
        "steps": 20,
        "sampler": "dpmpp_2m",
        "scheduler": "karras"
    },
    "qwen_image_21": {
        "id": "qwen_image_21",
        "name": "Qwen-Image-2.1 Turbo (Single-Stream DiT & Qwen3-VL)",
        "badge": "Qwen 2.1 DiT",
        "desc": "Alibaba next-gen 7B DiT with native Qwen3-VL comprehension for high-fidelity photorealism.",
        "type": "qwen",
        "unet": "qwen_image_2.1_turbo_int8_convrot.safetensors",
        "clip": "qwen3vl_8b_int8_convrot.safetensors",
        "vae": QWEN_VAE,
        "target_size": 1024,
        "cfg": 1.0,
        "steps": 8,
        "sampler": "euler",
        "scheduler": "simple"
    },
    "z_image_turbo": {
        "id": "z_image_turbo",
        "name": "Z-Image Turbo (Tongyi 6B DiT)",
        "badge": "Tongyi 6B DiT",
        "desc": "Tongyi-MAI 6B DiT distilled 8-step generator with Qwen3-4B text encoder.",
        "type": "zimage",
        "unet": "z_image_turbo_int8_convrot.safetensors",
        "clip": "qwen_3_4b.safetensors",
        "vae": ZIMAGE_VAE,
        "target_size": 1024,
        "cfg": 1.0,
        "steps": 8,
        "sampler": "res_multistep",
        "scheduler": "simple"
    },
    "moss_image": {
        "id": "moss_image",
        "name": "MOSS-Image 5B SFT (OpenMOSS DiT & Typography Master)",
        "badge": "OpenMOSS 5B DiT",
        "desc": "Next-gen 5B DiT with Qwen3.5 text encoder, 4K prompt comprehension, and perfect typographic text rendering on garments.",
        "type": "moss",
        "unet": get_best_moss_unet(),
        "clip": get_best_moss_clip(),
        "vae": FLUX_VAE,
        "target_size": 1024,
        "cfg": 3.0,
        "steps": 18,
        "shift": 5.0,
        "sampler": "euler",
        "scheduler": "simple"
    }
}

ALL_KNOWN_MODELS = {**MODELS, **ARCHIVED_MODELS}

# Directories for runtime data
# Staged rollout: enable only in a qualification run until fixture visual review
# and portrait regressions pass. The default remains unchanged during rollout.
AUTOMATIC_OUTFIT_REPAIR = os.environ.get('FACE_STUDIO_AUTOMATIC_OUTFIT_REPAIR', '0') == '1'
INPUTS_DIR = WORKSPACE_ROOT / "inputs"
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"
SAMPLES_DIR = WORKSPACE_ROOT / "samples"

INPUTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Strict Negative Filter (Melanin-neutral; skin color protection is handled dynamically by skin_engine)
DEFAULT_NEGATIVE_PROMPT = (
    "bindi, forehead dot, forehead sticker, sindoor, tilak, "
    "oily skin, greasy skin, shiny skin, specular glare, reflective skin, "
    "two-tone skin, mismatched skin tone, mismatched head and body, floating head, "
    "exaggerated grimace, cartoon grin, wide angle distortion, selfie, flat flash, fisheye, "
    "cartoon, painting, illustration, 3d render, CGI, plastic skin, doll, airbrushed, "
    "smooth plastic, mannequin, wax figure, fake eyes, artificial, "
    "oversaturated, blurry, out of focus, bad anatomy, deformed, mutated hands, "
    "extra fingers, fused fingers, extra limbs, asymmetric eyes, bad proportions, "
    "watermark, signature, text, logo, distorted face"
)
