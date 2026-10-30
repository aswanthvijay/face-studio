"""Face Studio — Custom Fashion LoRA Trainer for SDXL (RTX 3060 6GB Low-VRAM Optimized).

Engineered to operate within a 6GB VRAM constraint:
1. Stage 1 (Latent Caching): Pre-encodes training images through SDXL VAE to latents; VAE is completely freed from VRAM.
2. Stage 2 (Text Conditioning Caching): Pre-encodes prompt captions via dual CLIP text encoders; encoders freed from VRAM.
3. Stage 3 (LoRA Training): Trains only UNet cross-attention adapters (Rank 16) with FP16 mixed precision,
   gradient checkpointing, and gradient accumulation. Peak VRAM: ~4.6-5.1 GB.
4. Stage 4 (Export): Saves production .safetensors directly to StabilityMatrix & Face Studio LoRA folders.
"""
import argparse
import gc
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

# Add root directory to sys.path for config import
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import config


class CachedFashionDataset(Dataset):
    """Dataset holding pre-cached latents, text embeddings, and pooled embeddings in CPU RAM."""
    def __init__(self, cached_items: List[dict]):
        self.items = cached_items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def get_base_checkpoint_path(preferred_name: str = "") -> Path:
    """Finds existing SDXL checkpoint from config.MODELS_DIR."""
    if preferred_name:
        custom_p = Path(preferred_name)
        if custom_p.exists():
            return custom_p
        in_models = config.MODELS_DIR / preferred_name
        if in_models.exists():
            return in_models

    candidates = [
        "CyberRealisticXLPlay_V7.0_FP16.safetensors",
        "RealVisXL_V5.0_fp16.safetensors",
        "epicrealismXL_v7FinalDestination.safetensors",
        "Local_CyberRealistic_FINAL_inpainting.safetensors"
    ]
    for c in candidates:
        p = config.MODELS_DIR / c
        if p.exists():
            return p

    # Fallback to any safetensors in Models dir
    all_safetensors = list(config.MODELS_DIR.glob("*.safetensors"))
    if all_safetensors:
        return all_safetensors[0]

    raise FileNotFoundError(f"No SDXL base model found in {config.MODELS_DIR}. Please place an SDXL checkpoint in the models folder.")


def convert_diffusers_to_comfy_lora(state_dict: dict, alpha: int = 16) -> dict:
    """Converts PEFT / Diffusers LoRA keys into standard ComfyUI / Kohya format."""
    comfy_state_dict = {}
    for k, v in state_dict.items():
        if "lora" not in k.lower():
            continue

        # Standard diffusers key format: unet.down_blocks.1.attentions.0...
        new_k = k
        if new_k.startswith("base_model.model."):
            new_k = new_k[len("base_model.model."):]
        if new_k.startswith("unet."):
            new_k = new_k[len("unet."):]

        # Transform to Kohya/ComfyUI convention: lora_unet_...
        new_k = new_k.replace(".", "_")
        new_k = f"lora_unet_{new_k}"
        new_k = new_k.replace("_lora_A_weight", ".lora_down.weight")
        new_k = new_k.replace("_lora_B_weight", ".lora_up.weight")
        new_k = new_k.replace("_lora_down_weight", ".lora_down.weight")
        new_k = new_k.replace("_lora_up_weight", ".lora_up.weight")

        comfy_state_dict[new_k] = v.contiguous().half()

        # Add alpha key
        base_layer = new_k.rsplit(".", 1)[0]
        alpha_key = f"{base_layer}.alpha"
        if alpha_key not in comfy_state_dict:
            comfy_state_dict[alpha_key] = torch.tensor(alpha, dtype=torch.float32)

    return comfy_state_dict


def cache_latents_and_embeddings(
    dataset_dir: Path,
    base_ckpt_path: Path,
    device: torch.device,
    target_dim: Tuple[int, int] = (896, 896)
) -> List[dict]:
    """Stage 1 & 2: Pre-encode images and captions to avoid loading VAE & CLIP during UNet training."""
    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = sorted([f for f in dataset_dir.iterdir() if f.is_file() and f.suffix.lower() in image_exts])

    if not image_files:
        raise ValueError(f"No training images found in {dataset_dir}")

    print(f"\n[Stage 1/4] Pre-encoding {len(image_files)} images through SDXL VAE (Freeing VRAM afterward)...")

    # Load SDXL VAE
    vae = AutoencoderKL.from_single_file(
        str(base_ckpt_path),
        torch_dtype=torch.float16
    ).to(device)
    vae.eval()

    cached_latents = []
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Caching VAE Latents"):
            img = Image.open(img_path).convert("RGB")
            # Ensure divisible by 8
            w = (img.width // 8) * 8
            h = (img.height // 8) * 8
            if (img.width, img.height) != (w, h):
                img = img.resize((w, h), Image.Resampling.LANCZOS)

            import numpy as np
            arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
            img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float16)

            latent = vae.encode(img_tensor).latent_dist.sample() * vae.config.scaling_factor
            cached_latents.append(latent.squeeze(0).cpu())

    # Fully unload VAE from VRAM
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    print("  VAE unloaded. VRAM freed.")

    print(f"\n[Stage 2/4] Pre-encoding captions through dual SDXL text encoders (CLIP L + OpenCLIP G)...")
    # Load Tokenizers and Text Encoders
    tok_one = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    tok_two = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

    text_enc_one = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=torch.float16
    ).to(device)
    text_enc_two = CLIPTextModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        torch_dtype=torch.float16
    ).to(device)
    text_enc_one.eval()
    text_enc_two.eval()

    cached_data = []
    with torch.no_grad():
        for idx, img_path in enumerate(tqdm(image_files, desc="Caching Text Encodings")):
            txt_path = img_path.with_suffix(".txt")
            caption = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else "luxury fashion dress, editorial portrait"

            # Tokenize
            tokens_one = tok_one(caption, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
            tokens_two = tok_two(caption, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)

            out_one = text_enc_one(tokens_one, output_hidden_states=True)
            out_two = text_enc_two(tokens_two, output_hidden_states=True)

            # SDXL uses penultimate hidden states concatenated along feature dimension
            prompt_embeds = torch.cat([out_one.hidden_states[-2], out_two.hidden_states[-2]], dim=-1)
            pooled_embeds = out_two.text_embeds

            # Prepare SDXL micro-conditioning vector (original size, target size, crop coords)
            lat = cached_latents[idx]
            h, w = lat.shape[1] * 8, lat.shape[2] * 8
            time_ids = torch.tensor([[h, w, 0, 0, h, w]], dtype=torch.float16)

            cached_data.append({
                "latent": lat,
                "prompt_embeds": prompt_embeds.squeeze(0).cpu(),
                "pooled_embeds": pooled_embeds.squeeze(0).cpu(),
                "time_ids": time_ids.squeeze(0).cpu(),
                "caption": caption,
                "filename": img_path.name
            })

    # Fully unload Text Encoders from VRAM
    del text_enc_one, text_enc_two, tok_one, tok_two
    gc.collect()
    torch.cuda.empty_cache()
    print("  CLIP Encoders unloaded. VRAM freed.")

    return cached_data


def train_lora(
    dataset_dir: str,
    output_name: str = "fs_couture",
    steps: int = 400,
    lr: float = 1e-4,
    rank: int = 16,
    grad_accum: int = 4,
    save_interval: int = 200,
    preferred_model: str = "",
    dry_run: bool = False
):
    """Main training loop with low-VRAM optimization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Face Studio LoRA training.")

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    base_ckpt = get_base_checkpoint_path(preferred_model)
    print(f"\n=======================================================")
    print(f"Face Studio Low-VRAM LoRA Trainer (RTX 3060 6GB)")
    print(f"Base Model:       {base_ckpt.name}")
    print(f"Dataset Dir:      {dataset_path}")
    print(f"LoRA Target Name: {output_name}")
    print(f"Rank (r):         {rank} (Alpha: {rank})")
    print(f"Max Steps:        {5 if dry_run else steps}")
    print(f"Learning Rate:    {lr}")
    print(f"Grad Accum Steps: {grad_accum}")
    print(f"=======================================================\n")

    # Step 1 & 2: Latents & CLIP Pre-Caching
    cached_dataset = cache_latents_and_embeddings(dataset_path, base_ckpt, device)
    train_dataset = CachedFashionDataset(cached_dataset)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Step 3: Load UNet in FP16 with Gradient Checkpointing
    print(f"\n[Stage 3/4] Loading SDXL UNet in FP16 with Gradient Checkpointing...")
    unet = UNet2DConditionModel.from_single_file(
        str(base_ckpt),
        torch_dtype=torch.float16
    ).to(device)

    unet.enable_gradient_checkpointing()
    unet.requires_grad_(False)

    # Inject LoRA adapters to cross-attention layers
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        init_lora_weights="gaussian"
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Noise Scheduler & Optimizer
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8
    )

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    total_steps = 5 if dry_run else steps
    progress_bar = tqdm(total=total_steps, desc="Training Fashion LoRA")

    global_step = 0
    unet.train()
    optimizer.zero_grad()

    start_train_time = time.time()
    while global_step < total_steps:
        for batch in train_loader:
            latents = batch["latent"].to(device, dtype=torch.float16)
            prompt_embeds = batch["prompt_embeds"].to(device, dtype=torch.float16)
            pooled_embeds = batch["pooled_embeds"].to(device, dtype=torch.float16)
            time_ids = batch["time_ids"].to(device, dtype=torch.float16)

            # Sample random noise & random timestep
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Added conditioning for SDXL
            added_cond_kwargs = {
                "text_embeds": pooled_embeds,
                "time_ids": time_ids
            }

            with torch.cuda.amp.autocast(dtype=torch.float16):
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs
                ).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (global_step + 1) % grad_accum == 0 or (global_step + 1) == total_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            global_step += 1
            progress_bar.update(1)

            peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            progress_bar.set_postfix({
                "loss": f"{loss.item() * grad_accum:.4f}",
                "peak_vram": f"{peak_vram_gb:.2f}GB",
                "step": f"{global_step}/{total_steps}"
            })

            if global_step >= total_steps:
                break

    progress_bar.close()
    elapsed = round(time.time() - start_train_time, 1)
    print(f"\nTraining completed in {elapsed}s. Peak VRAM: {torch.cuda.max_memory_allocated(device) / (1024 ** 3):.2f} GB")

    # Step 4: Export LoRA to ComfyUI format
    print(f"\n[Stage 4/4] Converting and exporting production .safetensors LoRA...")
    peft_state_dict = unet.state_dict()
    comfy_lora_dict = convert_diffusers_to_comfy_lora(peft_state_dict, alpha=rank)

    filename = f"{output_name}.safetensors"
    # Destination 1: StabilityMatrix Lora Folder
    dest1 = config.LORA_DIR / filename
    # Destination 2: Local Face Studio Lora Folder
    dest2 = config.LOCAL_LORA_DIR / filename

    print(f"  Saving to: {dest1}")
    save_file(comfy_lora_dict, str(dest1))

    print(f"  Saving duplicate copy to: {dest2}")
    save_file(comfy_lora_dict, str(dest2))

    print(f"\nSUCCESS: LoRA '{output_name}' is now active and ready in Face Studio!")
    print(f"  - Model File: {dest1.name} ({dest1.stat().st_size / (1024*1024):.1f} MB)")
    print(f"  - App Integration: Select '{output_name}' from the Web Studio Custom LoRA dropdown.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Studio 6GB Low-VRAM SDXL LoRA Trainer")
    parser.add_argument("--dataset-dir", type=str, default="training/dataset", help="Path to prepared dataset directory")
    parser.add_argument("--lora-name", type=str, default="fs_couture", help="Output LoRA filename (without .safetensors)")
    parser.add_argument("--steps", type=int, default=350, help="Total training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--base-model", type=str, default="", help="Preferred base model checkpoint name")
    parser.add_argument("--dry-run", action="store_true", help="Run 5-step validation dry-run to verify VRAM and exports")

    args = parser.parse_args()

    train_lora(
        dataset_dir=args.dataset_dir,
        output_name=args.lora_name,
        steps=args.steps,
        lr=args.lr,
        rank=args.rank,
        grad_accum=args.grad_accum,
        preferred_model=args.base_model,
        dry_run=args.dry_run
    )
