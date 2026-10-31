import threading
"""End-to-End Face Studio & Photovance AI Pipeline.



Covers:

1. Dual Modes:

   - Mode 1: In-Place Outfit Swap (Inpaints *only* the garment mask; body, pose, background, face untouched).

   - Mode 2: Full Scene Generation (Generates new scene & outfit from prompt, then transfers 1:1 biometric face).

2. Multi-Model Support:

   - CyberRealistic XL (DSLR Pores & Natural Skin)

   - epiCRealism XL (Cinematic Portraits)

   - RealVisXL V5.0 (Ultra-Fast Benchmark)

   - Flux-2-Klein 4B Turbo (Unrestricted DiT)

3. Biometric Identity Lock:

   - CodeFormer w=0.85 + Inswapper 128 + Luminance Matching + High-Frequency Skin Pores.

"""

import io

import math

import random

import subprocess

import time

import uuid

from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

import cv2

try:

    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

except Exception:

    pass

import numpy as np

import requests

import torch

class _MaxResult(tuple):
    @property
    def values(self):
        return self[0]

    @property
    def indices(self):
        return self[1]


class _T(np.ndarray):
    def sigmoid(self):
        return 1.0 / (1.0 + np.exp(-np.asarray(self, dtype=np.float32)))

    def max(self, dim=None, **kwargs):
        vals = np.max(np.asarray(self), axis=dim).view(_T)
        idxs = np.argmax(np.asarray(self), axis=dim).view(_T)
        return _MaxResult((vals, idxs))

    def numpy(self):
        return np.asarray(self)


class _TorchStub:
    _MaxResult = _MaxResult
    _T = _T

    @staticmethod
    def from_numpy(arr):
        return np.asarray(arr).view(_T)

    @staticmethod
    def tensor(arr, **kwargs):
        return np.asarray(arr).view(_T)

    @staticmethod
    def zeros(*args, **kwargs):
        return np.zeros(*args).view(_T)

    @staticmethod
    def full(shape, val, **kwargs):
        return np.full(shape, val).view(_T)

    class no_grad:
        def __enter__(self): return self
        def __exit__(self, *args): pass




from core import config

from core.prompt_engine import build_zero_pollution_prompt, garment_requirements

from core.clothing_engine import generate_dress_mask, normalize_input_image

from core.face_engine import apply_face_lock



SESSION = requests.Session()

SESSION.trust_env = False





_BACKEND_LOCK = threading.Lock()





def ensure_backend() -> bool:

    """Verify ComfyUI backend is running on port 8189, or start it (thread-safe, Bug B9)."""

    with _BACKEND_LOCK:

        try:

            res = SESSION.get(f"{config.BACKEND_URL}/system_stats", timeout=3)

            if res.status_code == 200:

                return True

        except requests.RequestException:

            pass



        bootstrap = config.IMAGE_AI_ROOT / "engine_bootstrap.py"

        if not bootstrap.exists():

            bootstrap = config.COMFYUI_DIR / "main.py"

        if not bootstrap.exists():

            raise RuntimeError(f"Engine bootstrap script not found at {bootstrap}")



        for folder in ("engine-input", "engine-output", "engine-user", "engine-temp", "logs"):

            (config.IMAGE_AI_ROOT / folder).mkdir(exist_ok=True)



        log_file = (config.IMAGE_AI_ROOT / "logs" / "engine.log").open("a", encoding="utf-8")

        subprocess.Popen(

            [

                str(config.PYTHON_EXE),

                "-u",

                str(bootstrap),

                "--listen", "127.0.0.1",

                "--port", "8189",

                "--lowvram",

                "--reserve-vram", "1.5",

                "--enable-dynamic-vram",

                "--fast-disk",

                "--async-offload", "2",

                "--use-pytorch-cross-attention",

                "--preview-method", "none",

                "--offline",

                "--disable-auto-launch",

                "--extra-model-paths-config", str(config.WORKSPACE_ROOT / "extra_models.yaml"),

                "--input-directory", str(config.IMAGE_AI_ROOT / "engine-input"),

                "--output-directory", str(config.IMAGE_AI_ROOT / "engine-output"),

                "--user-directory", str(config.IMAGE_AI_ROOT / "engine-user"),

                "--temp-directory", str(config.IMAGE_AI_ROOT / "engine-temp"),

            ],

            cwd=config.COMFYUI_DIR,

            stdin=subprocess.DEVNULL,

            stdout=log_file,

            stderr=subprocess.STDOUT,

            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,

            close_fds=True

        )

        log_file.close()



        deadline = time.monotonic() + 180

        while time.monotonic() < deadline:

            try:

                res = SESSION.get(f"{config.BACKEND_URL}/system_stats", timeout=4)

                if res.status_code == 200:

                    return True

            except requests.RequestException:

                time.sleep(2)



        raise RuntimeError("Timed out waiting for local image engine on port 8189.")





def comfy_free_memory(unload_models: bool = True, free_memory: bool = True):

    """Signal ComfyUI to unload inactive models and free GPU VRAM."""

    try:

        SESSION.post(

            f"{config.BACKEND_URL}/free",

            json={"unload_models": unload_models, "free_memory": free_memory},

            timeout=5

        )

    except Exception:

        pass





def upload_image(path: Path) -> str:

    """Upload image to local ComfyUI input folder."""

    with path.open("rb") as stream:

        res = SESSION.post(

            f"{config.BACKEND_URL}/upload/image",

            files={"image": (path.name, stream, "image/png")},

            data={"subfolder": "FaceStudio", "type": "input", "overwrite": "true"},

            timeout=60

        )

    res.raise_for_status()

    data = res.json()

    return f"{data['subfolder']}/{data['name']}" if data.get("subfolder") else data["name"]





def calculate_working_dimensions(orig_w: int, orig_h: int, target_size: int = 1024, aspect_ratio: str = "original", is_in_place: bool = False):

    """Scale source dimensions preserving aspect ratio using model-optimal buckets (multiple of 16)."""

    factor = float(target_size) / 1024.0



    if is_in_place:

        target_dim = float(target_size) * factor

        scale = target_dim / math.sqrt(orig_w * orig_h)

        w = max(64, round(orig_w * scale / 16) * 16)

        h = max(64, round(orig_h * scale / 16) * 16)

        return w, h



    if aspect_ratio in ("story", "full_body"):  # 9:16

        base_w, base_h = 768, 1344

    elif aspect_ratio == "square":  # 1:1

        base_w, base_h = 1024, 1024

    elif aspect_ratio in ("portrait", "editorial", "waist_up"):  # 3:4 / 4:5

        base_w, base_h = 896, 1152

    else:

        aspect = orig_w / max(1, orig_h)

        if aspect <= 0.65:      # ~9:16

            base_w, base_h = 768, 1344

        elif aspect <= 0.82:    # ~3:4

            base_w, base_h = 896, 1152

        elif aspect <= 0.95:    # ~4:5

            base_w, base_h = 832, 1024

        elif aspect <= 1.08:    # ~1:1

            base_w, base_h = 1024, 1024

        elif aspect <= 1.25:    # ~5:4

            base_w, base_h = 1024, 832

        elif aspect <= 1.55:    # ~4:3

            base_w, base_h = 1152, 896

        else:                   # ~16:9

            base_w, base_h = 1344, 768



    w = max(64, round(base_w * factor / 16) * 16)

    h = max(64, round(base_h * factor / 16) * 16)

    return w, h





def laplacian_pyramid_blend(

    fg_img: Image.Image,

    bg_img: Image.Image,

    mask_img: Image.Image,

    num_levels: int = 4

) -> Image.Image:

    """Seamless Multi-Band Laplacian Pyramid Compositing.



    Blends low frequencies (color, lighting, ambient exposure) across a wide spatial scale

    to eliminate hard boundary color patches, while preserving high frequencies (textures,

    seams, fine details) sharply aligned with the mask boundary.

    """

    try:

        fg = np.array(fg_img.convert("RGB"), dtype=np.float32) / 255.0

        bg = np.array(bg_img.convert("RGB"), dtype=np.float32) / 255.0

        m_raw = np.array(mask_img.convert("L"), dtype=np.float32) / 255.0

        m = m_raw[:, :, np.newaxis]

        m = np.repeat(m, 3, axis=2)



        h, w = fg.shape[:2]

        div = 1 << num_levels

        pad_h = (div - (h % div)) if (h % div) != 0 else 0

        pad_w = (div - (w % div)) if (w % div) != 0 else 0



        if pad_h > 0 or pad_w > 0:

            fg = cv2.copyMakeBorder(fg, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

            bg = cv2.copyMakeBorder(bg, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

            m = cv2.copyMakeBorder(m, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)



        # Gaussian pyramids

        g_fg = [fg]

        g_bg = [bg]

        g_m = [m]

        for _ in range(num_levels):

            g_fg.append(cv2.pyrDown(g_fg[-1]))

            g_bg.append(cv2.pyrDown(g_bg[-1]))

            g_m.append(cv2.pyrDown(g_m[-1]))



        # Laplacian pyramids

        l_fg = [g_fg[-1]]

        l_bg = [g_bg[-1]]

        for i in range(num_levels, 0, -1):

            size = (g_fg[i - 1].shape[1], g_fg[i - 1].shape[0])

            l_fg.append(g_fg[i - 1] - cv2.pyrUp(g_fg[i], dstsize=size))

            l_bg.append(g_bg[i - 1] - cv2.pyrUp(g_bg[i], dstsize=size))



        # Blend levels

        blended_pyr = [l_fg[0] * g_m[-1] + l_bg[0] * (1.0 - g_m[-1])]

        for i in range(1, len(l_fg)):

            m_curr = g_m[num_levels - i]

            blended_pyr.append(l_fg[i] * m_curr + l_bg[i] * (1.0 - m_curr))



        # Reconstruct

        blended = blended_pyr[0]

        for i in range(1, len(blended_pyr)):

            size = (blended_pyr[i].shape[1], blended_pyr[i].shape[0])

            blended = cv2.pyrUp(blended, dstsize=size) + blended_pyr[i]



        if pad_h > 0 or pad_w > 0:

            blended = blended[:h, :w]



        blended_uint8 = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

        return Image.fromarray(blended_uint8)

    except Exception as e:

        print(f"[Pipeline] Multi-band blending fallback to alpha composite: {e}")

        return Image.composite(fg_img, bg_img, mask_img)





def build_flux_graph(

    source_subpath: str,

    prompt: str,

    model_cfg: dict,

    width: int = 768,

    height: int = 1024,

    seed: int = None,

    reference_subpath: str = None,

    is_in_place: bool = False

):

    """Flux-2-Klein DiT workflow with Qwen3 Heretic text encoder (clean prompt synthesis in Full Scene mode)."""

    if seed is None:

        seed = random.randint(1, 2147483647)



    graph = {

        "model": {

            "class_type": "UNETLoader",

            "inputs": {

                "unet_name": model_cfg.get("unet", "flux-2-klein-4b-fp8.safetensors"),

                "weight_dtype": "default"

            }

        },

        "clip": {

            "class_type": "CLIPLoader",

            "inputs": {

                "clip_name": model_cfg.get("clip", "qwen3-4b-heretic_fp8_e4m3fn.safetensors"),

                "type": "flux2",

                "device": "default"

            }

        },

        "vae": {

            "class_type": "VAELoader",

            "inputs": {"vae_name": model_cfg.get("vae", config.FLUX_VAE)}

        },

        "text": {

            "class_type": "CLIPTextEncode",

            "inputs": {

                "clip": ["clip", 0],

                "text": prompt

            }

        },

        "latent": {

            "class_type": "EmptyFlux2LatentImage",

            "inputs": {

                "width": width,

                "height": height,

                "batch_size": 1

            }

        }

    }



    # In-place outfit swap mode uses reference latent from original photo to anchor pose/garment context

    if is_in_place and reference_subpath:

        graph["source"] = {

            "class_type": "LoadImage",

            "inputs": {"image": reference_subpath}

        }

        graph["resize"] = {

            "class_type": "ImageScale",

            "inputs": {

                "image": ["source", 0],

                "upscale_method": "lanczos",

                "width": width,

                "height": height,

                "crop": "center"

            }

        }

        graph["reference"] = {

            "class_type": "VAEEncode",

            "inputs": {

                "pixels": ["resize", 0],

                "vae": ["vae", 0]

            }

        }

        graph["condition"] = {

            "class_type": "ReferenceLatent",

            "inputs": {

                "conditioning": ["text", 0],

                "latent": ["reference", 0]

            }

        }

        conditioning_target = ["condition", 0]

    else:

        # Full scene mode: pure text conditioning, eliminating any ghosting of prior clothing or stretched face crops

        conditioning_target = ["text", 0]



    graph.update({

        "guider": {

            "class_type": "BasicGuider",

            "inputs": {

                "model": ["model", 0],

                "conditioning": conditioning_target

            }

        },

        "schedule": {

            "class_type": "Flux2Scheduler",

            "inputs": {

                "steps": model_cfg.get("steps", 4),

                "width": width,

                "height": height

            }

        },

        "sampler": {

            "class_type": "KSamplerSelect",

            "inputs": {

                "sampler_name": model_cfg.get("sampler", "euler")

            }

        },

        "noise": {

            "class_type": "RandomNoise",

            "inputs": {"noise_seed": seed}

        },

        "sample": {

            "class_type": "SamplerCustomAdvanced",

            "inputs": {

                "noise": ["noise", 0],

                "guider": ["guider", 0],

                "sampler": ["sampler", 0],

                "sigmas": ["schedule", 0],

                "latent_image": ["latent", 0]

            }

        },

        "decode": {

            "class_type": "VAEDecodeTiled",

            "inputs": {

                "samples": ["sample", 0],

                "vae": ["vae", 0],

                "tile_size": 512,

                "overlap": 64,

                "temporal_size": 64,

                "temporal_overlap": 8

            }

        },

        "save": {

            "class_type": "SaveImage",

            "inputs": {

                "images": ["decode", 0],

                "filename_prefix": "FaceStudio_Flux"

            }

        }

    })

    return graph, seed





def build_flux_inpaint_graph(

    source_subpath: str,

    mask_subpath: str,

    prompt: str,

    model_cfg: dict,

    width: int = 768,

    height: int = 1024,

    seed: int = None

):

    """Flux-2-Klein DiT inpainting workflow with ReferenceLatent, SetLatentNoiseMask, and 1:1 pixel registration."""

    graph, seed = build_flux_graph(

        source_subpath=source_subpath,

        prompt=prompt,

        model_cfg=model_cfg,

        width=width,

        height=height,

        seed=seed,

        reference_subpath=source_subpath,

        is_in_place=True

    )

    if "resize" in graph and "inputs" in graph["resize"]:

        graph["resize"]["inputs"]["crop"] = "disabled"

    if mask_subpath:

        graph["mask_img"] = {

            "class_type": "LoadImage",

            "inputs": {"image": mask_subpath}

        }

        graph["scale_mask"] = {

            "class_type": "ImageScale",

            "inputs": {

                "image": ["mask_img", 0],

                "upscale_method": "bilinear",

                "width": width,

                "height": height,

                "crop": "disabled"

            }

        }

        graph["mask_node"] = {

            "class_type": "ImageToMask",

            "inputs": {"image": ["scale_mask", 0], "channel": "red"}

        }

        graph["set_mask"] = {

            "class_type": "SetLatentNoiseMask",

            "inputs": {

                "samples": ["reference", 0],

                "mask": ["mask_node", 0]

            }

        }

        if "sample" in graph and "inputs" in graph["sample"]:

            graph["sample"]["inputs"]["latent_image"] = ["set_mask", 0]

        graph.pop("latent", None)

    return graph, seed





def build_moss_graph(

    source_subpath: str,

    prompt: str,

    model_cfg: dict,

    width: int = 1024,

    height: int = 1024,

    seed: int = None,

    reference_subpath: str = None,

    lora_name: str = None,

    lora_weight: float = 0.75,

    is_in_place: bool = False

):

    """MOSS-Image 5B DiT workflow with Qwen3.5 text encoder, FLUX.2 VAE, and optional LoRA."""

    if seed is None:

        seed = random.randint(1, 2147483647)



    has_lora = bool(lora_name and str(lora_name).lower() not in ("none", "null", ""))



    graph = {

        "model": {

            "class_type": "UNETLoader",

            "inputs": {

                "unet_name": model_cfg.get("unet", "moss_image_sft_int8_convrot.safetensors"),

                "weight_dtype": "default"

            }

        }

    }



    if has_lora:

        graph["lora"] = {

            "class_type": "LoraLoaderModelOnly",

            "inputs": {

                "model": ["model", 0],

                "lora_name": lora_name,

                "strength_model": float(lora_weight)

            }

        }



    model_source = ["lora", 0] if has_lora else ["model", 0]



    graph.update({

        "clip": {

            "class_type": "CLIPLoader",

            "inputs": {

                "clip_name": model_cfg.get("clip", "qwen3-4b-heretic_fp8_e4m3fn.safetensors"),

                "type": model_cfg.get("clip_type", "flux2"),

                "device": "default"

            }

        },

        "vae": {

            "class_type": "VAELoader",

            "inputs": {"vae_name": model_cfg.get("vae", config.FLUX_VAE)}

        },

        "text": {

            "class_type": "CLIPTextEncode",

            "inputs": {

                "clip": ["clip", 0],

                "text": prompt

            }

        },

        "latent": {

            "class_type": "EmptyFlux2LatentImage",

            "inputs": {

                "width": width,

                "height": height,

                "batch_size": 1

            }

        }

    })



    if is_in_place and reference_subpath:

        graph["source"] = {

            "class_type": "LoadImage",

            "inputs": {"image": reference_subpath}

        }

        graph["resize"] = {

            "class_type": "ImageScale",

            "inputs": {

                "image": ["source", 0],

                "upscale_method": "lanczos",

                "width": width,

                "height": height,

                "crop": "disabled" if is_in_place else "center"

            }

        }

        graph["reference"] = {

            "class_type": "VAEEncode",

            "inputs": {

                "pixels": ["resize", 0],

                "vae": ["vae", 0]

            }

        }

        graph["condition"] = {

            "class_type": "ReferenceLatent",

            "inputs": {

                "conditioning": ["text", 0],

                "latent": ["reference", 0]

            }

        }

        conditioning_target = ["condition", 0]

    else:

        conditioning_target = ["text", 0]



    graph.update({

        "guider": {

            "class_type": "BasicGuider",

            "inputs": {

                "model": model_source,

                "conditioning": conditioning_target

            }

        },

        "schedule": {

            "class_type": "Flux2Scheduler",

            "inputs": {

                "steps": model_cfg.get("steps", 18),

                "width": width,

                "height": height

            }

        },

        "sampler": {

            "class_type": "KSamplerSelect",

            "inputs": {

                "sampler_name": model_cfg.get("sampler", "euler")

            }

        },

        "noise": {

            "class_type": "RandomNoise",

            "inputs": {"noise_seed": seed}

        },

        "sample": {

            "class_type": "SamplerCustomAdvanced",

            "inputs": {

                "noise": ["noise", 0],

                "guider": ["guider", 0],

                "sampler": ["sampler", 0],

                "sigmas": ["schedule", 0],

                "latent_image": ["latent", 0]

            }

        },

        "decode": {

            "class_type": "VAEDecodeTiled",

            "inputs": {

                "samples": ["sample", 0],

                "vae": ["vae", 0],

                "tile_size": 512,

                "overlap": 64,

                "temporal_size": 64,

                "temporal_overlap": 8

            }

        },

        "save": {

            "class_type": "SaveImage",

            "inputs": {

                "images": ["decode", 0],

                "filename_prefix": "FaceStudio_MOSS"

            }

        }

    })

    return graph, seed





def build_moss_inpaint_graph(

    source_subpath: str,

    mask_subpath: str,

    positive_prompt: str,

    negative_prompt: str,

    model_cfg: dict,

    width: int = 768,

    height: int = 1024,

    seed: int = None,

    lora_name: str = None,

    lora_weight: float = 0.75

):

    """MOSS-Image 5B DiT inpainting workflow with ReferenceLatent, SetLatentNoiseMask, and optional LoRA."""

    graph, seed = build_moss_graph(

        source_subpath=source_subpath,

        prompt=positive_prompt,

        model_cfg=model_cfg,

        width=width,

        height=height,

        seed=seed,

        reference_subpath=source_subpath,

        lora_name=lora_name,

        lora_weight=lora_weight,

        is_in_place=True

    )

    if "resize" in graph and "inputs" in graph["resize"]:

        graph["resize"]["inputs"]["crop"] = "disabled"

    if mask_subpath:

        graph["mask_img"] = {

            "class_type": "LoadImage",

            "inputs": {"image": mask_subpath}

        }

        graph["scale_mask"] = {

            "class_type": "ImageScale",

            "inputs": {

                "image": ["mask_img", 0],

                "upscale_method": "bilinear",

                "width": width,

                "height": height,

                "crop": "disabled"

            }

        }

        graph["mask_node"] = {

            "class_type": "ImageToMask",

            "inputs": {"image": ["scale_mask", 0], "channel": "red"}

        }

        graph["set_mask"] = {

            "class_type": "SetLatentNoiseMask",

            "inputs": {

                "samples": ["reference", 0],

                "mask": ["mask_node", 0]

            }

        }

        if "sample" in graph and "inputs" in graph["sample"]:

            graph["sample"]["inputs"]["latent_image"] = ["set_mask", 0]

    return graph, seed





def build_qwen_graph(

    positive_prompt: str,

    negative_prompt: str,

    model_cfg: dict,

    width: int = 1024,

    height: int = 1024,

    seed: int = None

):

    """Qwen-Image-2.1 Single-Stream DiT graph with Qwen3-VL text encoder."""

    if seed is None:

        seed = random.randint(1, 2147483647)



    graph = {

        "model": {

            "class_type": "UNETLoader",

            "inputs": {

                "unet_name": model_cfg.get("unet", "qwen_image_2.1_turbo_int8_convrot.safetensors"),

                "weight_dtype": "default"

            }

        },

        "clip": {

            "class_type": "CLIPLoader",

            "inputs": {

                "clip_name": model_cfg.get("clip", "qwen3vl_8b_int8_convrot.safetensors"),

                "type": "qwen_image",

                "device": "default"

            }

        },

        "vae": {

            "class_type": "VAELoader",

            "inputs": {

                "vae_name": model_cfg.get("vae", config.QWEN_VAE)

            }

        },

        "sampling": {

            "class_type": "ModelSamplingAuraFlow",

            "inputs": {

                "model": ["model", 0],

                "shift": 3.1

            }

        },

        "cache": {

            "class_type": "QwenImage21Cache",

            "inputs": {

                "model": ["sampling", 0],

                "device": "auto",

                "dtype": "default"

            }

        },

        "encode": {

            "class_type": "TextEncodeQwenImage21",

            "inputs": {

                "clip": ["clip", 0],

                "prompt": positive_prompt,

                "negative_prompt": negative_prompt or "",

                "resolution": max(width, height)

            }

        },

        "latent": {

            "class_type": "EmptyLatentImage",

            "inputs": {

                "width": width,

                "height": height,

                "batch_size": 1

            }

        },

        "sampler": {

            "class_type": "KSampler",

            "inputs": {

                "model": ["cache", 0],

                "positive": ["encode", 0],

                "negative": ["encode", 1],

                "latent_image": ["latent", 0],

                "seed": seed,

                "steps": model_cfg.get("steps", 8),

                "cfg": model_cfg.get("cfg", 1.0),

                "sampler_name": model_cfg.get("sampler", "euler"),

                "scheduler": model_cfg.get("scheduler", "simple"),

                "denoise": 1.0

            }

        },

        "decode": {

            "class_type": "VAEDecode",

            "inputs": {

                "samples": ["sampler", 0],

                "vae": ["vae", 0]

            }

        },

        "save": {

            "class_type": "SaveImage",

            "inputs": {

                "images": ["decode", 0],

                "filename_prefix": "FaceStudio_Qwen21"

            }

        }

    }

    return graph, seed





def build_qwen_inpaint_graph(

    source_subpath: str,

    mask_subpath: str,

    positive_prompt: str,

    negative_prompt: str,

    model_cfg: dict,

    width: int = 768,

    height: int = 1024,

    seed: int = None

):

    """Qwen-Image-2.1 DiT inpainting workflow with SetLatentNoiseMask and reference latents."""

    if seed is None:

        seed = random.randint(1, 2147483647)



    graph = {

        "model": {

            "class_type": "UNETLoader",

            "inputs": {

                "unet_name": model_cfg.get("unet", "qwen_image_2.1_turbo_int8_convrot.safetensors"),

                "weight_dtype": "default"

            }

        },

        "clip": {

            "class_type": "CLIPLoader",

            "inputs": {

                "clip_name": model_cfg.get("clip", "qwen3vl_8b_int8_convrot.safetensors"),

                "type": "qwen_image",

                "device": "default"

            }

        },

        "vae": {

            "class_type": "VAELoader",

            "inputs": {

                "vae_name": model_cfg.get("vae", config.QWEN_VAE)

            }

        },

        "sampling": {

            "class_type": "ModelSamplingAuraFlow",

            "inputs": {

                "model": ["model", 0],

                "shift": 3.1

            }

        },

        "cache": {

            "class_type": "QwenImage21Cache",

            "inputs": {

                "model": ["sampling", 0],

                "device": "auto",

                "dtype": "default"

            }

        },

        "encode": {

            "class_type": "TextEncodeQwenImage21",

            "inputs": {

                "clip": ["clip", 0],

                "prompt": positive_prompt,

                "negative_prompt": negative_prompt or "",

                "resolution": max(width, height)

            }

        },

        "source_img": {

            "class_type": "LoadImage",

            "inputs": {"image": source_subpath}

        },

        "mask_img": {

            "class_type": "LoadImage",

            "inputs": {"image": mask_subpath}

        },

        "scale_src": {

            "class_type": "ImageScale",

            "inputs": {

                "image": ["source_img", 0],

                "upscale_method": "lanczos",

                "width": width,

                "height": height,

                "crop": "disabled"

            }

        },

        "scale_mask": {

            "class_type": "ImageScale",

            "inputs": {

                "image": ["mask_img", 0],

                "upscale_method": "bilinear",

                "width": width,

                "height": height,

                "crop": "disabled"

            }

        },

        "mask_node": {

            "class_type": "ImageToMask",

            "inputs": {"image": ["scale_mask", 0], "channel": "red"}

        },

        "vae_encode": {

            "class_type": "VAEEncode",

            "inputs": {

                "pixels": ["scale_src", 0],

                "vae": ["vae", 0]

            }

        },

        "set_mask": {

            "class_type": "SetLatentNoiseMask",

            "inputs": {

                "samples": ["vae_encode", 0],

                "mask": ["mask_node", 0]

            }

        },

        "sampler": {

            "class_type": "KSampler",

            "inputs": {

                "model": ["cache", 0],

                "positive": ["encode", 0],

                "negative": ["encode", 1],

                "latent_image": ["set_mask", 0],

                "seed": seed,

                "steps": model_cfg.get("steps", 8),

                "cfg": model_cfg.get("cfg", 1.0),

                "sampler_name": model_cfg.get("sampler", "euler"),

                "scheduler": model_cfg.get("scheduler", "simple"),

                "denoise": 1.0

            }

        },

        "decode": {

            "class_type": "VAEDecode",

            "inputs": {

                "samples": ["sampler", 0],

                "vae": ["vae", 0]

            }

        },

        "save": {

            "class_type": "SaveImage",

            "inputs": {

                "images": ["decode", 0],

                "filename_prefix": "FaceStudio_Qwen21"

            }

        }

    }

    return graph, seed





def build_zimage_graph(

    positive_prompt: str,

    negative_prompt: str,

    model_cfg: dict,

    width: int = 1024,

    height: int = 1024,

    seed: int = None,

    lora_name: str = None,

    lora_weight: float = 0.8

):

    """Z-Image Turbo 6B DiT workflow with Qwen3-4B text encoder, optional LoRA, and ModelSamplingAuraFlow."""

    if seed is None:

        seed = random.randint(1, 2147483647)



    has_lora = bool(lora_name and str(lora_name).lower() not in ("none", "null", ""))



    graph = {

        "model": {

            "class_type": "UNETLoader",

            "inputs": {

                "unet_name": model_cfg.get("unet", "z_image_turbo_int8_convrot.safetensors"),

                "weight_dtype": "default"

            }

        }

    }



    if has_lora:

        graph["lora"] = {

            "class_type": "LoraLoaderModelOnly",

            "inputs": {

                "model": ["model", 0],

                "lora_name": lora_name,

                "strength_model": float(lora_weight)

            }

        }



    model_source = ["lora", 0] if has_lora else ["model", 0]



    graph.update({

        "sampling": {

            "class_type": "ModelSamplingAuraFlow",

            "inputs": {

                "model": model_source,

                "shift": 3.0

            }

        },

        "clip": {

            "class_type": "CLIPLoader",

            "inputs": {

                "clip_name": model_cfg.get("clip", "qwen_3_4b.safetensors"),

                "type": "lumina2",

                "device": "default"

            }

        },

        "vae": {

            "class_type": "VAELoader",

            "inputs": {

                "vae_name": model_cfg.get("vae", config.ZIMAGE_VAE)

            }

        },

        "positive": {

            "class_type": "CLIPTextEncode",

            "inputs": {

                "clip": ["clip", 0],

                "text": positive_prompt

            }

        },

        "negative": {

            "class_type": "CLIPTextEncode" if (negative_prompt and negative_prompt.strip()) else "ConditioningZeroOut",

            "inputs": {

                "clip": ["clip", 0],

                "text": negative_prompt

            } if (negative_prompt and negative_prompt.strip()) else {

                "conditioning": ["positive", 0]

            }

        },

        "latent": {

            "class_type": "EmptySD3LatentImage",

            "inputs": {

                "width": width,

                "height": height,

                "batch_size": 1

            }

        },

        "sampler": {

            "class_type": "KSampler",

            "inputs": {

                "model": ["sampling", 0],

                "positive": ["positive", 0],

                "negative": ["negative", 0],

                "latent_image": ["latent", 0],

                "seed": seed,

                "steps": model_cfg.get("steps", 8),

                "cfg": max(float(model_cfg.get("cfg", 1.0)), 1.5) if (negative_prompt and negative_prompt.strip()) else float(model_cfg.get("cfg", 1.0)),

                "sampler_name": model_cfg.get("sampler", "res_multistep"),

                "scheduler": model_cfg.get("scheduler", "simple"),

                "denoise": 1.0

            }

        },

        "decode": {

            "class_type": "VAEDecode",

            "inputs": {

                "samples": ["sampler", 0],

                "vae": ["vae", 0]

            }

        },

        "save": {

            "class_type": "SaveImage",

            "inputs": {

                "images": ["decode", 0],

                "filename_prefix": "FaceStudio_ZImageTurbo"

            }

        }

    })

    return graph, seed





def build_zimage_inpaint_graph(

    source_subpath: str,

    mask_subpath: str,

    positive_prompt: str,

    negative_prompt: str,

    model_cfg: dict,

    width: int = 768,

    height: int = 1024,

    seed: int = None,

    lora_name: str = None,

    lora_weight: float = 0.8

):

    """Z-Image Turbo 6B DiT inpainting workflow with optional LoRA and SetLatentNoiseMask."""

    if seed is None:

        seed = random.randint(1, 2147483647)



    has_lora = bool(lora_name and str(lora_name).lower() not in ("none", "null", ""))



    graph = {

        "model": {

            "class_type": "UNETLoader",

            "inputs": {

                "unet_name": model_cfg.get("unet", "z_image_turbo_int8_convrot.safetensors"),

                "weight_dtype": "default"

            }

        }

    }



    if has_lora:

        graph["lora"] = {

            "class_type": "LoraLoaderModelOnly",

            "inputs": {

                "model": ["model", 0],

                "lora_name": lora_name,

                "strength_model": float(lora_weight)

            }

        }



    model_source = ["lora", 0] if has_lora else ["model", 0]



    graph.update({

        "sampling": {

            "class_type": "ModelSamplingAuraFlow",

            "inputs": {

                "model": model_source,

                "shift": 3.0

            }

        },

        "clip": {

            "class_type": "CLIPLoader",

            "inputs": {

                "clip_name": model_cfg.get("clip", "qwen_3_4b.safetensors"),

                "type": "lumina2",

                "device": "default"

            }

        },

        "vae": {

            "class_type": "VAELoader",

            "inputs": {

                "vae_name": model_cfg.get("vae", config.ZIMAGE_VAE)

            }

        },

        "positive": {

            "class_type": "CLIPTextEncode",

            "inputs": {

                "clip": ["clip", 0],

                "text": positive_prompt

            }

        },

        "negative": {

            "class_type": "CLIPTextEncode" if (negative_prompt and negative_prompt.strip()) else "ConditioningZeroOut",

            "inputs": {

                "clip": ["clip", 0],

                "text": negative_prompt

            } if (negative_prompt and negative_prompt.strip()) else {

                "conditioning": ["positive", 0]

            }

        },

        "source_img": {

            "class_type": "LoadImage",

            "inputs": {"image": source_subpath}

        },

        "mask_img": {

            "class_type": "LoadImage",

            "inputs": {"image": mask_subpath}

        },

        "scale_src": {

            "class_type": "ImageScale",

            "inputs": {

                "image": ["source_img", 0],

                "upscale_method": "lanczos",

                "width": width,

                "height": height,

                "crop": "disabled"

            }

        },

        "scale_mask": {

            "class_type": "ImageScale",

            "inputs": {

                "image": ["mask_img", 0],

                "upscale_method": "bilinear",

                "width": width,

                "height": height,

                "crop": "disabled"

            }

        },

        "mask_node": {

            "class_type": "ImageToMask",

            "inputs": {"image": ["scale_mask", 0], "channel": "red"}

        },

        "vae_encode": {

            "class_type": "VAEEncode",

            "inputs": {

                "pixels": ["scale_src", 0],

                "vae": ["vae", 0]

            }

        },

        "set_mask": {

            "class_type": "SetLatentNoiseMask",

            "inputs": {

                "samples": ["vae_encode", 0],

                "mask": ["mask_node", 0]

            }

        },

        "sampler": {

            "class_type": "KSampler",

            "inputs": {

                "model": ["sampling", 0],

                "positive": ["positive", 0],

                "negative": ["negative", 0],

                "latent_image": ["set_mask", 0],

                "seed": seed,

                "steps": model_cfg.get("steps", 8),

                "cfg": max(float(model_cfg.get("cfg", 1.0)), 1.5) if (negative_prompt and negative_prompt.strip()) else float(model_cfg.get("cfg", 1.0)),

                "sampler_name": model_cfg.get("sampler", "res_multistep"),

                "scheduler": model_cfg.get("scheduler", "simple"),

                "denoise": 1.0

            }

        },

        "decode": {

            "class_type": "VAEDecode",

            "inputs": {

                "samples": ["sampler", 0],

                "vae": ["vae", 0]

            }

        },

        "save": {

            "class_type": "SaveImage",

            "inputs": {

                "images": ["decode", 0],

                "filename_prefix": "FaceStudio_ZImageTurbo_Inpaint"

            }

        }

    })

    return graph, seed





_COMFY_OBJECT_INFO_CACHE = None





def comfy_supports_node(node_name: str) -> bool:

    """Check if node_name exists in ComfyUI's object_info registry.

    Returns False on failure or absence to guarantee safe fallback to native nodes.

    """

    global _COMFY_OBJECT_INFO_CACHE

    if _COMFY_OBJECT_INFO_CACHE is not None:

        return node_name in _COMFY_OBJECT_INFO_CACHE

    try:

        res = SESSION.get(f"{config.BACKEND_URL}/object_info/{node_name}", timeout=3)

        if res.status_code == 200:

            data = res.json()

            if node_name in data and bool(data[node_name]):

                return True

            if data == {}:

                return False

    except Exception:

        pass

    try:

        res = SESSION.get(f"{config.BACKEND_URL}/object_info", timeout=4)

        if res.status_code == 200:

            _COMFY_OBJECT_INFO_CACHE = res.json()

            return node_name in _COMFY_OBJECT_INFO_CACHE

    except Exception:

        pass

    return False



_DEFAULT_COMFY_SUPPORTS_NODE = comfy_supports_node





def build_sdxl_inpaint_graph(

    source_subpath: str,

    mask_subpath: str,

    positive_prompt: str,

    negative_prompt: str,

    model_cfg: dict,

    width: int = 768,

    height: int = 1024,

    seed: int = None,

    lora_name: str = None,

    lora_weight: float = 0.8,

    prefer_native: bool = False

):

    """SDXL Inpainting graph with Fooocus Inpainting Patch and automated native InpaintModelConditioning fallback."""

    if seed is None:

        seed = random.randint(1, 2147483647)



    has_lora = bool(lora_name and str(lora_name).lower() not in ("none", "null", ""))



    graph = {

        "checkpoint": {

            "class_type": "CheckpointLoaderSimple",

            "inputs": {"ckpt_name": model_cfg["checkpoint"]}

        }

    }



    if has_lora:

        graph["lora"] = {

            "class_type": "LoraLoader",

            "inputs": {

                "model": ["checkpoint", 0],

                "clip": ["checkpoint", 1],

                "lora_name": lora_name,

                "strength_model": float(lora_weight),

                "strength_clip": float(lora_weight)

            }

        }



    clip_source = ["lora", 1] if has_lora else ["checkpoint", 1]

    model_source = ["lora", 0] if has_lora else ["checkpoint", 0]



    if prefer_native:

        use_fooocus = False

    elif comfy_supports_node is not _DEFAULT_COMFY_SUPPORTS_NODE:

        use_fooocus = bool(comfy_supports_node("INPAINT_VAEEncodeInpaintConditioning"))

    else:

        use_fooocus = True

        try:

            r = SESSION.get(f"{config.BACKEND_URL}/system_stats", timeout=0.5)

            if r.status_code == 200:

                use_fooocus = comfy_supports_node("INPAINT_VAEEncodeInpaintConditioning")

        except Exception:

            use_fooocus = True



    graph.update({

        "positive": {

            "class_type": "CLIPTextEncode",

            "inputs": {"clip": clip_source, "text": positive_prompt}

        },

        "negative": {

            "class_type": "CLIPTextEncode",

            "inputs": {"clip": clip_source, "text": negative_prompt}

        },

        "source": {

            "class_type": "LoadImage",

            "inputs": {"image": source_subpath}

        },

        "resize_source": {

            "class_type": "ImageScale",

            "inputs": {

                "image": ["source", 0],

                "upscale_method": "lanczos",

                "width": width,

                "height": height,

                "crop": "disabled"

            }

        },

        "mask_source": {

            "class_type": "LoadImage",

            "inputs": {"image": mask_subpath}

        },

        "resize_mask": {

            "class_type": "ImageScale",

            "inputs": {

                "image": ["mask_source", 0],

                "upscale_method": "bilinear",

                "width": width,

                "height": height,

                "crop": "disabled"

            }

        },

        "mask": {

            "class_type": "ImageToMask",

            "inputs": {"image": ["resize_mask", 0], "channel": "red"}

        }

    })



    if use_fooocus:

        graph.update({

            "neutral_source": {

                "class_type": "INPAINT_MaskedFill",

                "inputs": {"image": ["resize_source", 0], "mask": ["mask", 0], "fill": "neutral", "falloff": 16}

            },

            "inpaint_conditioning": {

                "class_type": "INPAINT_VAEEncodeInpaintConditioning",

                "inputs": {

                    "positive": ["positive", 0],

                    "negative": ["negative", 0],

                    "vae": ["checkpoint", 2],

                    "pixels": ["neutral_source", 0],

                    "mask": ["mask", 0]

                }

            },

            "inpaint_patch": {

                "class_type": "INPAINT_LoadFooocusInpaint",

                "inputs": {

                    "head": "fooocus_inpaint_head.pth",

                    "patch": "inpaint_v26.fooocus.patch"

                }

            },

            "patched_model": {

                "class_type": "INPAINT_ApplyFooocusInpaint",

                "inputs": {

                    "model": model_source,

                    "patch": ["inpaint_patch", 0],

                    "latent": ["inpaint_conditioning", 2]

                }

            },

            "sample": {

                "class_type": "KSampler",

                "inputs": {

                    "model": ["patched_model", 0],

                    "positive": ["inpaint_conditioning", 0],

                    "negative": ["inpaint_conditioning", 1],

                    "latent_image": ["inpaint_conditioning", 3],

                    "seed": seed,

                    "steps": model_cfg.get("steps", 20),

                    "cfg": model_cfg.get("cfg", 5.0),

                    "sampler_name": model_cfg.get("sampler", "dpmpp_2m"),

                    "scheduler": model_cfg.get("scheduler", "karras"),

                    "denoise": 1.0

                }

            }

        })

    else:

        # Native ComfyUI fallback (InpaintModelConditioning + SetLatentNoiseMask)

        graph.update({

            "inpaint_conditioning": {

                "class_type": "InpaintModelConditioning",

                "inputs": {

                    "positive": ["positive", 0],

                    "negative": ["negative", 0],

                    "vae": ["checkpoint", 2],

                    "pixels": ["resize_source", 0],

                    "mask": ["mask", 0]

                }

            },

            "latent_noise_mask": {

                "class_type": "SetLatentNoiseMask",

                "inputs": {

                    "samples": ["inpaint_conditioning", 2],

                    "mask": ["mask", 0]

                }

            },

            "sample": {

                "class_type": "KSampler",

                "inputs": {

                    "model": model_source,

                    "positive": ["inpaint_conditioning", 0],

                    "negative": ["inpaint_conditioning", 1],

                    "latent_image": ["latent_noise_mask", 0],

                    "seed": seed,

                    "steps": model_cfg.get("steps", 20),

                    "cfg": model_cfg.get("cfg", 5.0),

                    "sampler_name": model_cfg.get("sampler", "dpmpp_2m"),

                    "scheduler": model_cfg.get("scheduler", "karras"),

                    "denoise": 1.0

                }

            }

        })

    graph.update({

        "decode": {

            "class_type": "VAEDecodeTiled",

            "inputs": {

                "samples": ["sample", 0],

                "vae": ["checkpoint", 2],

                "tile_size": 512,

                "overlap": 64,

                "temporal_size": 64,

                "temporal_overlap": 8

            }

        },

        "save": {

            "class_type": "SaveImage",

            "inputs": {

                "images": ["decode", 0],

                "filename_prefix": f"FaceStudio_{model_cfg.get('id', 'sdxl')}"

            }

        }

    })

    return graph, seed





def build_sdxl_txt2img_graph(

    positive_prompt: str,

    negative_prompt: str,

    model_cfg: dict,

    width: int = 768,

    height: int = 1024,

    seed: int = None,

    lora_name: str = None,

    lora_weight: float = 0.8,

    face_image_subpath: str = None,

    enable_face_id: bool = True,

    face_id_weight: float = 0.85,

    framing: str = "editorial"

):

    """SDXL Text-to-Image scene generation graph with optional LoRA styling and IP-Adapter FaceID guidance."""

    if seed is None:

        seed = random.randint(1, 2147483647)



    has_lora = bool(lora_name and str(lora_name).lower() not in ("none", "null", ""))



    graph = {

        "checkpoint": {

            "class_type": "CheckpointLoaderSimple",

            "inputs": {"ckpt_name": model_cfg["checkpoint"]}

        }

    }



    if has_lora:

        graph["lora"] = {

            "class_type": "LoraLoader",

            "inputs": {

                "model": ["checkpoint", 0],

                "clip": ["checkpoint", 1],

                "lora_name": lora_name,

                "strength_model": float(lora_weight),

                "strength_clip": float(lora_weight)

            }

        }



    clip_source = ["lora", 1] if has_lora else ["checkpoint", 1]

    model_source = ["lora", 0] if has_lora else ["checkpoint", 0]



    # Inject IP-Adapter FaceID for 1:1 facial identity guidance

    if enable_face_id and face_image_subpath:

        # If full body or wide editorial framing is requested, delay IP-Adapter start slightly

        # so the model can first anchor full-body proportions and footwear before facial identity conditioning begins

        is_wide_or_full = framing in ("full_body", "editorial", "wide")

        ip_start = 0.12 if is_wide_or_full else 0.0

        ip_end = 0.58 if is_wide_or_full else 0.65



        model_id = model_cfg.get("id", "")

        # For epiCRealism XL, its aesthetic prior tends to narrow chin and pull facial landmarks towards

        # generic influencer aesthetics; slightly elevate FaceID weight to pin user's authentic facial geometry

        if model_id == "epicrealism_xl":

            effective_ip_weight = min(0.74, max(0.50, float(face_id_weight) * 0.95))

            effective_v2_weight = round(min(0.35, float(face_id_weight) * 0.48), 2)

        else:

            effective_ip_weight = min(0.70, max(0.40, float(face_id_weight)))

            effective_v2_weight = round(min(0.30, float(face_id_weight) * 0.45), 2)



        graph.update({

            "ip_model": {

                "class_type": "IPAdapterModelLoader",

                "inputs": {"ipadapter_file": "ip-adapter-faceid-portrait_sdxl.bin"}

            },

            "ip_insight": {

                "class_type": "IPAdapterInsightFaceLoader",

                "inputs": {"provider": "CPU", "model_name": "buffalo_l"}

            },

            "clip_vision": {

                "class_type": "CLIPVisionLoader",

                "inputs": {"clip_name": "clip-vision_vit-h.safetensors"}

            },

            "face_img": {

                "class_type": "LoadImage",

                "inputs": {"image": face_image_subpath}

            },

            "ip_apply": {

                "class_type": "IPAdapterFaceID",

                "inputs": {

                    "model": model_source,

                    "ipadapter": ["ip_model", 0],

                    "image": ["face_img", 0],

                    "clip_vision": ["clip_vision", 0],

                    "insightface": ["ip_insight", 0],

                    "weight": effective_ip_weight,

                    "weight_faceidv2": effective_v2_weight,

                    "weight_type": "linear",

                    "combine_embeds": "concat",

                    "start_at": ip_start,

                    "end_at": ip_end,

                    "embeds_scaling": "V only"

                }

            }

        })

        model_source = ["ip_apply", 0]



    graph.update({

        "positive": {

            "class_type": "CLIPTextEncode",

            "inputs": {"clip": clip_source, "text": positive_prompt}

        },

        "negative": {

            "class_type": "CLIPTextEncode",

            "inputs": {"clip": clip_source, "text": negative_prompt}

        },

        "latent": {

            "class_type": "EmptyLatentImage",

            "inputs": {

                "width": width,

                "height": height,

                "batch_size": 1

            }

        },

        "sample": {

            "class_type": "KSampler",

            "inputs": {

                "model": model_source,

                "positive": ["positive", 0],

                "negative": ["negative", 0],

                "latent_image": ["latent", 0],

                "seed": seed,

                "steps": model_cfg.get("steps", 18),

                "cfg": model_cfg.get("cfg", 5.0),

                "sampler_name": model_cfg.get("sampler", "dpmpp_2m"),

                "scheduler": model_cfg.get("scheduler", "karras"),

                "denoise": 1.0

            }

        },

        "decode": {

            "class_type": "VAEDecodeTiled",

            "inputs": {

                "samples": ["sample", 0],

                "vae": ["checkpoint", 2],

                "tile_size": 512,

                "overlap": 64,

                "temporal_size": 64,

                "temporal_overlap": 8

            }

        },

        "save": {

            "class_type": "SaveImage",

            "inputs": {

                "images": ["decode", 0],

                "filename_prefix": f"FaceStudio_{model_cfg.get('id', 'sdxl')}"

            }

        }

    })

    return graph, seed





_CURRENT_MODEL_TYPE = None





def execute_workflow(graph: dict, progress_cb=None) -> Image.Image:

    """Submit workflow graph to ComfyUI, wait for completion, and return resulting PIL image."""

    client_id = str(uuid.uuid4())

    res = SESSION.post(

        f"{config.BACKEND_URL}/prompt",

        json={"prompt": graph, "client_id": client_id},

        timeout=30

    )

    if res.status_code != 200:

        raise RuntimeError(f"Engine rejected workflow: {res.text[:1000]}")

    data = res.json()

    if "node_errors" in data and data["node_errors"]:

        raise RuntimeError(f"Engine workflow validation error: {data['node_errors']}")

    prompt_id = data.get("prompt_id")

    if not prompt_id:

        raise RuntimeError(f"Engine did not return a valid prompt_id: {data}")



    start_time = time.monotonic()

    while True:

        elapsed = round(time.monotonic() - start_time)

        if elapsed > 300:

            raise TimeoutError("Engine workflow execution timed out after 300 seconds.")

        if progress_cb:

            progress_cb(f"Rendering photorealistic outfit... ({elapsed}s)")



        res = SESSION.get(f"{config.BACKEND_URL}/history/{prompt_id}", timeout=15)

        res.raise_for_status()

        data = res.json().get(prompt_id)



        if data:

            # Check for node validation/execution errors immediately

            node_errors = data.get("node_errors")

            if node_errors:

                raise RuntimeError(f"Engine node error: {node_errors}")



            status = data.get("status", {})

            messages = status.get("messages", [])

            for m in messages:

                if isinstance(m, (list, tuple)) and len(m) > 1 and m[0] == "execution_error":

                    ex = m[1].get("exception_message") or m[1].get("exception_type") or "Execution error"

                    raise RuntimeError(f"Engine node execution error: {ex}")



            if status.get("status_str") == "error":

                err = next((m[1].get("exception_message") for m in reversed(messages) if isinstance(m, (list, tuple)) and len(m) > 1 and m[0] == "execution_error"), "Engine execution error")

                raise RuntimeError(err)



            if not status.get("completed", False):

                time.sleep(0.5)

                continue



            save_output = data.get("outputs", {}).get("save", {})

            images = save_output.get("images", [])

            if not images:

                raise RuntimeError("No image returned from engine.")



            # Resilient loader: retry in case Windows file stream/cache is still flushing

            for attempt in range(5):

                try:

                    img_res = SESSION.get(f"{config.BACKEND_URL}/view", params=images[0], timeout=60)

                    img_res.raise_for_status()

                    buf = io.BytesIO(img_res.content)

                    with Image.open(buf) as generated:

                        generated.load()

                        return generated.copy()

                except (OSError, IOError, SyntaxError, Exception) as load_err:

                    if attempt == 4:

                        raise RuntimeError(f"Failed to load generated image after 5 attempts: {load_err}")

                    time.sleep(1.0)



        time.sleep(1)





def run_face_studio_pipeline(

    source_image: Image.Image,

    prompt: str,

    target_garment: str = "dress",

    model_id: str = "cyberrealistic_xl",

    mode: str = "in_place_swap",

    aspect_ratio: str = "original",

    framing: str = "editorial",

    variations: int = 1,

    seed: int = None,

    lora_id: str = None,

    lora_weight: float = 0.8,

    enable_face_id: bool = True,

    face_id_weight: float = 0.85,

    progress_cb = None,

    negative_prompt: str = "",
    custom_negative_prompt: str = None
) -> dict:

    """

    Executes the complete Face Studio pipeline:

    1. Segments clothing & calculates working dimensions.

    2. Builds zero-pollution prompt with skin luminance descriptors.

    3. Resolves and applies optional Custom Fashion LoRA weights & trigger tokens.

    4. Runs ComfyUI diffusion inpainting or scene generation.

    5. Applies 1:1 biometric identity lock + CodeFormer w=0.85 + micro-skin pores (for Full Scene mode).

    6. In In-Place mode: Composites the newly generated outfit back onto original photo without redundant face-lock latency.

    """

    is_in_place = mode in ("in_place", "in_place_swap")

    normalized_mode = "in_place_swap" if is_in_place else "full_scene"



    if model_id not in config.MODELS:

        raise ValueError(f"Unknown model '{model_id}'.")

    model_cfg = config.MODELS[model_id]

    model_type = model_cfg.get("type", "sdxl")

    lora_meta = config.resolve_lora(lora_id, model_type)

    resolved_lora_filename = lora_meta["filename"] if lora_meta else None

    if lora_meta:

        trigger = lora_meta.get("trigger", "").strip()

        if trigger and trigger.lower() not in prompt.lower():

            prompt = f"{trigger}, {prompt}"

    requested_variations = max(1, min(int(variations), 4))

    warnings = []

    if is_in_place and config.AUTOMATIC_OUTFIT_REPAIR:

        from core.automatic_outfit import run_automatic_outfit

        source_image = normalize_input_image(source_image)

        effective_cfg = dict(model_cfg)

        if lora_meta:

            for key in ('steps', 'cfg', 'sampler', 'scheduler'):

                if key in lora_meta:

                    effective_cfg[key] = lora_meta[key]

        # Detection/preflight does not need to launch or unload the GPU backend.

        return run_automatic_outfit(source_image, prompt, effective_cfg, seed,

            requested_variations, progress_cb, resolved_lora_filename, lora_weight,

            target=target_garment, custom_negative_prompt=custom_negative_prompt or negative_prompt)

    ensure_backend()



    # Step 1: Normalize source image & save temporary file

    source_image = normalize_input_image(source_image)

    request_id = str(uuid.uuid4())[:8]

    temp_src_path = config.INPUTS_DIR / f"src_{request_id}.png"

    source_image.save(temp_src_path, "PNG")



    # Step 1.2: Model configuration & working dimensions

    global _CURRENT_MODEL_TYPE

    if _CURRENT_MODEL_TYPE is not None and _CURRENT_MODEL_TYPE != model_type:

        print(f"[Pipeline] Switching model architecture from {_CURRENT_MODEL_TYPE} to {model_type}; freeing VRAM...")

        comfy_free_memory(unload_models=True, free_memory=True)

    _CURRENT_MODEL_TYPE = model_type



    # Step 1.5: Validate face presence and extract source face bbox unconditionally

    src_face_bbox = None

    cached_src_faces = None

    from core.face_engine import get_face_modules, detect_faces_robust

    app, _, _, _, _ = get_face_modules()

    src_rgb = np.array(ImageOps.exif_transpose(source_image).convert("RGB"))

    src_bgr = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR)

    cached_src_faces = detect_faces_robust(app, src_bgr)

    if cached_src_faces:

        src_face_bbox = cached_src_faces[0].bbox

    elif not is_in_place:

        raise ValueError("No face detected in reference photo. Please upload a photo with a clearly visible face for Full Scene generation, or switch to In-Place Outfit Swap.")



    try:

        source_image._cached_src_faces = cached_src_faces

        source_image._cached_face_bbox = src_face_bbox

    except Exception:

        pass



    cached_skin_profile = None

    try:

        from core.skin_engine import extract_universal_skin_profile

        cached_skin_profile = extract_universal_skin_profile(source_image, face_bbox=src_face_bbox)

        source_image._cached_skin_profile = cached_skin_profile

    except Exception as e:

        print(f"[Pipeline] Skin profile extraction note: {e}")



    # Adapt sampling parameters if distillation LoRA is active (e.g. Hyper-SDXL 8-steps, CFG 2.0)

    effective_model_cfg = dict(model_cfg)

    if lora_meta:

        for opt_key in ("steps", "cfg", "sampler", "scheduler"):

            if opt_key in lora_meta:

                effective_model_cfg[opt_key] = lora_meta[opt_key]

    elif resolved_lora_filename and ("hyper" in resolved_lora_filename.lower() or "8step" in resolved_lora_filename.lower()):

        effective_model_cfg["steps"] = 8

        effective_model_cfg["cfg"] = 2.0

        effective_model_cfg["sampler"] = "euler"

        effective_model_cfg["scheduler"] = "karras"



    w, h = calculate_working_dimensions(

        source_image.width, source_image.height,

        target_size=effective_model_cfg.get("target_size", 1024),

        aspect_ratio=aspect_ratio if not is_in_place else "original",

        is_in_place=is_in_place

    )



    # Step 3: Dress Mask Segmentation (Only required for In-Place Outfit Swap)

    mask_pil = None

    coverage = 0.0

    mask_subpath = None

    if is_in_place:

        if effective_model_cfg.get("type") == "flux":

            from core.clothing_engine import detect_dress_mask_flux

            res = detect_dress_mask_flux(

                source_image,

                prompt=prompt,

                target=target_garment,

                keep_face=True,

                progress_cb=progress_cb,

                return_flux_image=False,

                seed=seed if seed is not None else 42

            )

            mask_pil, preview_overlay, faces, coverage = res[:4]

        else:

            # High-precision SegFormer B2 clothing parser on authentic source photograph

            mask_pil, preview_overlay, faces, coverage = generate_dress_mask(

                source_image,

                target=target_garment,

                prompt=prompt,

                keep_face=True,

                expand_pixels=14,

                feather_sigma=3.5

            )

        if coverage < 0.005:

            raise ValueError(

                "No garment detected for in-place swap. "

                "Please use Full Scene mode or upload a clearer portrait showing clothing."

            )

        temp_mask_path = config.INPUTS_DIR / f"mask_{request_id}.png"

        mask_pil.save(temp_mask_path, "PNG")

        mask_subpath = upload_image(temp_mask_path)



    # Step 4: Prompt Construction

    pos_prompt, neg_prompt = build_zero_pollution_prompt(

        user_prompt=prompt,

        source_image=source_image,

        mode=normalized_mode,

        model_type=effective_model_cfg.get("type", "sdxl"),

        framing=framing,

        skin_profile=cached_skin_profile,

        cached_src_faces=cached_src_faces,

        lora_meta=lora_meta

    )

    custom_neg = custom_negative_prompt if custom_negative_prompt is not None else negative_prompt
    if custom_neg and str(custom_neg).strip():
        from core.prompt_engine import merge_negative_prompt
        neg_prompt = merge_negative_prompt([neg_prompt, str(custom_neg).strip()], prompt)



    # Step 5: Upload authentic source image to ComfyUI

    src_subpath = upload_image(temp_src_path)



    # Clean 1.8x portrait crop for IP-Adapter FaceID (prevents room shadow & clothing color bleed / green hair)

    face_id_subpath = src_subpath

    if enable_face_id and not is_in_place:

        try:

            if cached_src_faces:

                f_obj = sorted(cached_src_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]

                fx1, fy1, fx2, fy2 = [int(v) for v in f_obj.bbox]

                fw, fh = fx2 - fx1, fy2 - fy1

                cx, cy = (fx1 + fx2) // 2, (fy1 + fy2) // 2

                crop_dim = int(max(fw, fh) * 1.8)

                h_dim = crop_dim // 2

                c_x1 = max(0, cx - h_dim)

                c_y1 = max(0, cy - h_dim)

                c_x2 = min(src_bgr.shape[1], cx + h_dim)

                c_y2 = min(src_bgr.shape[0], cy + h_dim)

                face_crop_bgr = src_bgr[c_y1:c_y2, c_x1:c_x2]

                if face_crop_bgr.shape[0] > 64 and face_crop_bgr.shape[1] > 64:

                    face_crop_path = config.INPUTS_DIR / f"face_crop_{request_id}.png"

                    cv2.imwrite(str(face_crop_path), face_crop_bgr)

                    face_id_subpath = upload_image(face_crop_path)

        except Exception as e:

            print(f"[Pipeline] Portrait face crop for FaceID warning: {e}")

            face_id_subpath = src_subpath



    # Step 6: Construct workflow graph

    def make_graph(current_seed, prefer_native: bool = False):

        if effective_model_cfg.get("type") == "flux":

            if is_in_place:

                return build_flux_inpaint_graph(

                    source_subpath=src_subpath,

                    mask_subpath=mask_subpath,

                    prompt=pos_prompt,

                    model_cfg=effective_model_cfg,

                    width=w, height=h,

                    seed=current_seed

                )

            else:

                return build_flux_graph(

                    source_subpath=src_subpath,

                    prompt=pos_prompt,

                    model_cfg=effective_model_cfg,

                    width=w, height=h,

                    seed=current_seed,

                    reference_subpath=None,

                    is_in_place=False

                )

        elif effective_model_cfg.get("type") == "qwen":

            if is_in_place:

                return build_qwen_inpaint_graph(

                    source_subpath=src_subpath,

                    mask_subpath=mask_subpath,

                    positive_prompt=pos_prompt,

                    negative_prompt=neg_prompt,

                    model_cfg=effective_model_cfg,

                    width=w, height=h,

                    seed=current_seed

                )

            else:

                return build_qwen_graph(

                    positive_prompt=pos_prompt,

                    negative_prompt=neg_prompt,

                    model_cfg=effective_model_cfg,

                    width=w, height=h,

                    seed=current_seed

                )

        elif effective_model_cfg.get("type") == "zimage":

            if is_in_place:

                return build_zimage_inpaint_graph(

                    source_subpath=src_subpath,

                    mask_subpath=mask_subpath,

                    positive_prompt=pos_prompt,

                    negative_prompt=neg_prompt,

                    model_cfg=effective_model_cfg,

                    width=w, height=h,

                    seed=current_seed,

                    lora_name=resolved_lora_filename,

                    lora_weight=lora_weight

                )

            else:

                return build_zimage_graph(

                    positive_prompt=pos_prompt,

                    negative_prompt=neg_prompt,

                    model_cfg=effective_model_cfg,

                    width=w, height=h,

                    seed=current_seed,

                    lora_name=resolved_lora_filename,

                    lora_weight=lora_weight

                )

        elif effective_model_cfg.get("type") == "moss":

            if is_in_place:

                return build_moss_inpaint_graph(

                    source_subpath=src_subpath,

                    mask_subpath=mask_subpath,

                    positive_prompt=pos_prompt,

                    negative_prompt=neg_prompt,

                    model_cfg=effective_model_cfg,

                    width=w, height=h,

                    seed=current_seed,

                    lora_name=resolved_lora_filename,

                    lora_weight=lora_weight

                )

            else:

                return build_moss_graph(

                    source_subpath=src_subpath,

                    prompt=pos_prompt,

                    model_cfg=effective_model_cfg,

                    width=w, height=h,

                    seed=current_seed,

                    reference_subpath=src_subpath if is_in_place else None,

                    lora_name=resolved_lora_filename,

                    lora_weight=lora_weight,

                    is_in_place=is_in_place

                )

        elif is_in_place:

            return build_sdxl_inpaint_graph(

                source_subpath=src_subpath,

                mask_subpath=mask_subpath,

                positive_prompt=pos_prompt,

                negative_prompt=neg_prompt,

                model_cfg=effective_model_cfg,

                width=w, height=h,

                seed=current_seed,

                lora_name=resolved_lora_filename,

                lora_weight=lora_weight,

                prefer_native=prefer_native

            )

        else:

            return build_sdxl_txt2img_graph(

                positive_prompt=pos_prompt,

                negative_prompt=neg_prompt,

                model_cfg=effective_model_cfg,

                width=w, height=h,

                seed=current_seed,

                lora_name=resolved_lora_filename,

                lora_weight=lora_weight,

                face_image_subpath=face_id_subpath,

                enable_face_id=enable_face_id,

                face_id_weight=face_id_weight,

                framing=framing

            )



    used_seed = seed if seed is not None else random.randint(1, 2147483647)



    # Step 7: Execute diffusion workflow

    active_prefer_native = False

    try:

        graph, used_seed = make_graph(used_seed, prefer_native=active_prefer_native)

        raw_rendered = execute_workflow(graph, progress_cb=progress_cb)

    except Exception as e:

        if is_in_place and any(k in str(e).lower() for k in ["inpaint_", "fooocus", "missing_node", "not found", "class not found", "validation"]):

            print(f"[Pipeline] Custom inpainting node error ({e}). Retrying with native InpaintModelConditioning fallback...")

            active_prefer_native = True

            graph, used_seed = make_graph(used_seed, prefer_native=True)

            raw_rendered = execute_workflow(graph, progress_cb=progress_cb)

        else:

            raise



    # Phase-Shift VRAM Relay: Free ComfyUI diffusion models from VRAM before starting FaceEngine / Post-processing

    # Keep models resident in VRAM between iterations when variations > 1; unload only if single variation

    comfy_free_memory(unload_models=True if (not is_in_place and enable_face_id) or requested_variations <= 1 else False, free_memory=True)

    if torch.cuda.is_available():

        torch.cuda.empty_cache()



    def process_output(rendered_img):

        if is_in_place:

            base_img = source_image

            rendered_scaled = rendered_img.resize(source_image.size, Image.Resampling.LANCZOS)

            if mask_pil is not None:

                mask_scaled = mask_pil.resize(source_image.size, Image.Resampling.LANCZOS)

                mask_np = np.array(mask_scaled)



                # Pre-composite Lab skin tone matching to align inpaint skin elements to source photo

                try:

                    from core.face_engine import match_skin_tone_lab

                    from core.clothing_engine import parse_clothing_labels

                    src_bgr = cv2.cvtColor(np.array(source_image.convert("RGB")), cv2.COLOR_RGB2BGR)

                    rend_bgr = cv2.cvtColor(np.array(rendered_scaled.convert("RGB")), cv2.COLOR_RGB2BGR)

                    src_labels = parse_clothing_labels(source_image)

                    if src_labels.shape[:2] != src_bgr.shape[:2]:

                        src_labels = cv2.resize(src_labels, (src_bgr.shape[1], src_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

                    src_smask = np.isin(src_labels, (11, 12, 13, 14, 15)).astype(np.uint8) * 255



                    inpaint_mask_np = np.array(mask_scaled)

                    tgt_labels = parse_clothing_labels(rendered_scaled)

                    if tgt_labels.shape[:2] != rend_bgr.shape[:2]:

                        tgt_labels = cv2.resize(tgt_labels, (rend_bgr.shape[1], rend_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

                    tgt_smask = (np.isin(tgt_labels, (11, 12, 13, 14, 15)) & (inpaint_mask_np > 10)).astype(np.uint8) * 255



                    if np.count_nonzero(src_smask) > 100 and np.count_nonzero(tgt_smask) > 100:

                        matched_inpaint = match_skin_tone_lab(src_bgr, rend_bgr, source_mask=src_smask, target_mask=tgt_smask)

                        rendered_scaled = Image.fromarray(cv2.cvtColor(matched_inpaint, cv2.COLOR_BGR2RGB))

                except Exception as e:

                    print(f"[Pipeline] In-place skin matching note: {e}")



                # Bug B10: Pre-composite harmonization removed; single post-composite pass runs after foreground compositing



                # Smooth Alpha Compositing using mask_scaled

                composited = Image.composite(rendered_scaled, base_img.convert("RGB"), mask_scaled)



                # Phase 2D: Authentic foreground draped hair re-compositing

                # Guarantees that natural hair strands, curls, and flyaways cascading over shoulders are 100% preserved

                src_labels = None

                try:

                    from core.clothing_engine import parse_clothing_labels

                    src_labels = parse_clothing_labels(source_image)

                    hair_mask = (src_labels == 2) & ~np.isin(src_labels, (4, 5, 6, 7, 8, 17))

                    hair_clean = cv2.erode(hair_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0

                    hair_overlap = hair_clean & (mask_np > 30)

                    if np.count_nonzero(hair_overlap) > 100:

                        hair_alpha = cv2.GaussianBlur(hair_overlap.astype(np.float32), (0, 0), sigmaX=1.5, sigmaY=1.5)

                        hair_alpha = np.clip(hair_alpha, 0.0, 1.0)[:, :, None]

                        comp_np = np.array(composited)

                        base_np = np.array(base_img.convert("RGB"))

                        hair_blended = (comp_np * (1.0 - hair_alpha) + base_np * hair_alpha).clip(0, 255).astype(np.uint8)

                        composited = Image.fromarray(hair_blended)

                except Exception as e:

                    print(f"[Pipeline] Hair re-compositing note: {e}")



                # Unified Dermal Harmonization on final composite (guarantees seamless transition across midriff & limbs)

                try:

                    from core.skin_engine import extract_universal_skin_profile, harmonize_unified_canvas_skin

                    skin_prof = cached_skin_profile if cached_skin_profile is not None else extract_universal_skin_profile(source_image, face_bbox=src_face_bbox)

                    composited = harmonize_unified_canvas_skin(

                        canvas_image=composited,

                        face_bbox=src_face_bbox,

                        src_skin_profile=skin_prof,

                        blend_strength=0.90,

                        clothing_labels=None,

                        edit_mask=mask_scaled,

                        align_luminance=True

                    )

                except Exception as e:

                    print(f"[Pipeline] Post-composite skin harmonization note: {e}")



                # Guarantee exact bitwise preservation strictly outside the inpainting mask (mask_np == 0)

                # while preserving 100% of the smooth feathered transition across all mask_np > 0 pixels

                comp_np = np.array(composited)

                outside = (mask_np == 0)

                comp_np[outside] = np.array(base_img.convert("RGB"))[outside]

                composited = Image.fromarray(comp_np)

            else:

                composited = rendered_scaled

            return composited, False

        else:

            # Full scene mode:

            # Step 0: Direct Lab skin tone transfer from input portrait to prevent skin lightening bias

            try:

                from core.face_engine import match_skin_tone_lab

                from core.clothing_engine import parse_clothing_labels



                src_bgr = cv2.cvtColor(np.array(source_image.convert("RGB")), cv2.COLOR_RGB2BGR)

                rend_bgr = cv2.cvtColor(np.array(rendered_img.convert("RGB")), cv2.COLOR_RGB2BGR)

                rh, rw = rend_bgr.shape[:2]



                src_labels = parse_clothing_labels(source_image)

                if src_labels.shape[:2] != src_bgr.shape[:2]:

                    src_labels = cv2.resize(src_labels, (src_bgr.shape[1], src_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

                src_skin_mask = np.isin(src_labels, (11, 12, 13, 14, 15)).astype(np.uint8) * 255



                tgt_labels = parse_clothing_labels(rendered_img)

                if tgt_labels.shape[:2] != (rh, rw):

                    tgt_labels = cv2.resize(tgt_labels, (rw, rh), interpolation=cv2.INTER_NEAREST)

                tgt_skin_mask = np.isin(tgt_labels, (11, 12, 13, 14, 15)).astype(np.uint8) * 255



                # Morphological closing to bridge sheer fabric folds and prevent disjointed oval patches

                close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

                tgt_skin_closed = cv2.morphologyEx(tgt_skin_mask, cv2.MORPH_CLOSE, close_kernel)



                # Dermatological YCrCb skin chrominance validity filter

                ycrcb_t = cv2.cvtColor(rend_bgr, cv2.COLOR_BGR2YCrCb)

                cr_t = ycrcb_t[:, :, 1]

                cb_t = ycrcb_t[:, :, 2]

                gray_t = cv2.cvtColor(rend_bgr, cv2.COLOR_BGR2GRAY)

                valid_tgt_skin = (tgt_skin_closed > 0) & (cr_t >= 120) & (cr_t <= 190) & (cb_t >= 65) & (cb_t <= 145) & (gray_t >= 30) & (gray_t <= 250)

                tgt_skin_final = (valid_tgt_skin.astype(np.uint8) * 255)



                if np.count_nonzero(src_skin_mask) > 100 and np.count_nonzero(tgt_skin_final) > 100:

                    matched_bgr = match_skin_tone_lab(src_bgr, rend_bgr, source_mask=src_skin_mask, target_mask=tgt_skin_final)

                    rendered_img = Image.fromarray(cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB))

            except Exception as e:

                print(f"[Pipeline] Skin tone matching note: {e}")



            # Step 1: Biometric Identity & Discrete Codebook Prior Transfer (apply_face_lock)

            face_locked_img = rendered_img

            face_swapped = False

            if enable_face_id:

                lock_fidelity = min(0.50, max(0.25, float(face_id_weight)))

                face_locked_img, face_swapped = apply_face_lock(

                    source_image=source_image,

                    target_image=rendered_img,

                    fidelity_weight=lock_fidelity,

                    enable_pores=True,

                    enable_grain=True,

                    return_status=True,

                    src_faces=cached_src_faces

                )



            # Step 2: Unified Post-Swap Dermal Harmonization

            # Executes strictly AFTER face locking so that the face on the canvas is the definitive

            # calibrated ground truth. Harmonizes all exposed body skin (submental throat, chest,

            # cleavage, arms, legs) with 100% mathematical unity to the face.

            try:

                from core.skin_engine import extract_universal_skin_profile, harmonize_unified_canvas_skin

                skin_prof = cached_skin_profile if cached_skin_profile is not None else extract_universal_skin_profile(source_image, face_bbox=src_face_bbox)

                target_face_bbox = None

                try:

                    from core.face_engine import get_face_modules, detect_faces_robust

                    app, _, _, _, _ = get_face_modules()

                    fl_bgr = cv2.cvtColor(np.array(face_locked_img.convert("RGB")), cv2.COLOR_RGB2BGR)

                    fl_faces = detect_faces_robust(app, fl_bgr)

                    if fl_faces:

                        target_face_bbox = sorted(fl_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0].bbox

                except Exception:

                    pass



                final_img = harmonize_unified_canvas_skin(

                    canvas_image=face_locked_img,

                    face_bbox=target_face_bbox,

                    src_skin_profile=skin_prof,

                    blend_strength=0.85,

                    align_luminance=True

                )

                return final_img, face_swapped

            except Exception as e:

                print(f"[Pipeline] Post-swap dermal harmonization note: {e}")

                return face_locked_img, face_swapped



    primary_image, face_swapped = process_output(raw_rendered)



    # Automatic safety fallback: If Full Scene mode generated an image with no face, retry with alternate seed

    if not is_in_place and enable_face_id and not face_swapped:

        print("[Pipeline] Face transfer failed. Retrying with an alternate seed...")

        try:

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

            retry_seed = random.randint(1, 2147483647)

            retry_graph, _ = make_graph(retry_seed)

            retry_rendered = execute_workflow(retry_graph, progress_cb=progress_cb)

            comfy_free_memory(unload_models=True, free_memory=True)

            retry_image, retry_swapped = process_output(retry_rendered)

            if retry_swapped:

                primary_image = retry_image

                face_swapped = True

                used_seed = retry_seed

                print(f"[Pipeline] Retry succeeded with seed {retry_seed}: Biometric identity locked.")

        except Exception as retry_err:

            print(f"[Pipeline] Retry error: {retry_err}")

        if not face_swapped:

            raise RuntimeError("Could not transfer the reference face after two attempts. Try a clearer reference or a closer framing.")



    required_color = garment_requirements(prompt).get("color")

    def assess_quality(result_image, transfer_applied, index):

        quality = {}

        if is_in_place:

            quality["review_required"] = True

            repair_info = getattr(result_image, "_garment_repair_info", None)

            if repair_info:

                quality["garment_repair"] = repair_info

                if repair_info.get("unexpected_arm_skin_after", 0) > 0:

                    warnings.append(f"Result {index}: generated skin near the garment needs review for extra fingers or uncovered arms.")

        from core.quality_checks import face_similarity, garment_color_check

        if transfer_applied and cached_src_faces:

            target_faces = detect_faces_robust(app, cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR))

            if target_faces:

                target_face = max(target_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

                quality["identity_similarity"] = face_similarity(cached_src_faces[0], target_face)

        if required_color:

            try:

                from core.clothing_engine import parse_clothing_labels

                color_result = garment_color_check(result_image, parse_clothing_labels(result_image), required_color)

                quality["garment_color"] = color_result

                if color_result["matches"] is False:

                    warnings.append(f"Result {index}: the clothing may not match the requested {required_color} color. Please review it.")

            except Exception as error:

                quality["garment_color"] = {"requested": required_color, "matches": None}

                print(f"[Pipeline] Garment color could not be checked: {error}")



        return quality



    qualities = [assess_quality(primary_image, face_swapped, 1)]



    out_filename = f"out_{request_id}_{model_id}_{normalized_mode}.png"

    out_path = config.OUTPUTS_DIR / out_filename

    primary_image.save(out_path, "PNG")



    all_results = [f"/outputs/{out_filename}"]



    # Multi-variation batch support

    if requested_variations > 1:

        for v_idx in range(1, requested_variations):

            try:

                comfy_free_memory(unload_models=False, free_memory=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                var_graph, var_seed = make_graph(random.randint(1, 2147483647))

                var_raw = execute_workflow(var_graph, progress_cb=progress_cb)

                comfy_free_memory(unload_models=(v_idx == requested_variations - 1) or (not is_in_place and enable_face_id), free_memory=True)
                var_final, var_swapped = process_output(var_raw)

                if not is_in_place and enable_face_id and not var_swapped:

                    raise RuntimeError("reference face transfer failed")

                var_name = f"out_{request_id}_var{v_idx}_{model_id}_{normalized_mode}.png"

                var_final.save(config.OUTPUTS_DIR / var_name, "PNG")

                all_results.append(f"/outputs/{var_name}")

                qualities.append(assess_quality(var_final, var_swapped, v_idx + 1))

            except Exception as e:

                print(f"[Pipeline] Variation {v_idx} generation warning: {e}")

                warnings.append(f"Variation {v_idx + 1} failed: {e}")



    # Temporary files are safely pruned by the background cache cleaner (24-hr TTL)

    # Avoiding immediate unlinking prevents race conditions with async ComfyUI workers



    return {

        "output_path": str(out_path),

        "filename": out_filename,

        "width": primary_image.width,

        "height": primary_image.height,

        "mode": normalized_mode,

        "model_id": model_id,

        "model_name": model_cfg.get("name", model_id),

        "lora_id": lora_id,

        "lora_weight": lora_weight,

        "seed": used_seed,

        "coverage": coverage,

        "face_swapped": face_swapped,

        "enable_face_id": enable_face_id,

        "face_id_weight": face_id_weight,

        "results": all_results,

        "requested_variations": requested_variations,

        "generated_variations": len(all_results),

        "warnings": warnings,

        "quality": qualities[0] if 'qualities' in locals() and qualities else {},

        "quality_results": qualities if 'qualities' in locals() else []

    }

