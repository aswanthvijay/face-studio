# Face Studio — Custom Dress & Fashion LoRA Training Guide

This guide covers training custom **Dress, Wardrobe & Fashion LoRAs** for Face Studio locally on an **NVIDIA RTX 3060 (6GB VRAM)**.

---

## 🌟 Overview

The Face Studio LoRA training engine specializes SDXL checkpoints (such as *CyberRealistic XL*, *RealVisXL*, or *epiCRealism XL*) on bespoke fashion aesthetics, designer garment collections, fabric drapes, and embroidery styles.

Once trained, your LoRA seamlessly integrates with:
- **In-Place Outfit Swap**: Inpaints *only* the segmented garment while keeping 100% of the original subject's face, pose, and background untouched.
- **Full Scene Generation**: Synthesizes the full editorial environment and styling while locking 1:1 facial identity.

---

## 🚀 Quick Start (1-Click Workflow)

Double-click `training\run_train.bat` to access the interactive studio launcher:
- **[1] Quick LoRA Validation**: Runs a 5-step validation training on the detected dress dataset, verifying VRAM headroom and Kohya export in ~60 seconds.
- **[2] Download Curated 30-Dress Dataset**: Automatically fetches 30 high-resolution couture and evening dress photos into `training\raw_images\`.
- **[3] Run SegFormer Dress Detection & Prepper**: Runs Face Studio's SegFormer parser on raw images to detect dresses, calculate coverage, crop/bucket to 896x896 / 768x1024, and generate paired `.txt` caption files with `fs_couture`.
- **[4] Full Fashion LoRA Training**: Executes customized training (e.g., 200–350 steps) for production use.
- **[5] Launch Web Studio**: Starts Face Studio on `http://127.0.0.1:7870`.

---

## 📸 Dataset Guidelines & Automated Dress Detection

For optimal photorealism and fabric drape learning:
- **Automated Detection**: `training/detect_and_crop_dresses.py` runs SegFormer clothing parsing to verify that garments are clearly distinguished from the background.
- **Quantity**: 15–30 high-resolution images (the built-in dataset includes 30 curated high-fashion dresses).
- **Diversity**: Full-body, three-quarter, and detail crops showing fabric texture, embroidery, and cuts.
- **Trigger Token**: The script automatically pairs each image with a `.txt` file containing your trigger token (`fs_couture`):
  ```text
  fs_couture, crimson silk evening gown, elegant fabric texture, haute couture tailoring, luxury dress design, editorial studio lighting
  ```

---

## ⚡ 6GB VRAM Optimization Architecture

To train a full SDXL LoRA within 6GB VRAM on an RTX 3060:
1. **Pre-Cached Latents**: The SDXL VAE encodes images to latents once and is unloaded completely from VRAM before training begins.
2. **Pre-Cached CLIP Embeddings**: Dual CLIP encoders (CLIP ViT-L and OpenCLIP ViT-bigG) encode all captions and are unloaded from VRAM.
3. **Gradient Checkpointing**: Activations are freed during forward pass, reducing UNet VRAM consumption by ~40%.
4. **Targeted LoRA Adapters**: Injects low-rank adapters ($r=16, \alpha=16$) into cross-attention projections (`to_k`, `to_q`, `to_v`, `to_out.0`).
5. **Peak VRAM**: Operates reliably at **~5.1 GB to 5.6 GB**, leaving safe overhead on your 6GB RTX 3060.

---

## 📁 Output & App Integration

When training completes, the `.safetensors` model is automatically saved to:
- `D:\Image AI\StabilityMatrix-win-x64\Data\Models\Lora\<lora_name>.safetensors`
- `D:\face_studio\models\loras\<lora_name>.safetensors`

### Using Your LoRA in Face Studio:
1. Launch the Web Studio: `run.bat` (Port 7870).
2. Look under **Section 3: Custom Style & Fashion LoRA**.
3. Select **Haute Couture & Designer Gowns (`fs_couture`)** from the dropdown.
4. Set the **LoRA Strength Slider** (recommended: `0.75` - `0.85`).
5. Click the **Trigger Tag** badge to insert `fs_couture` into your prompt.
6. Click **Generate Photorealistic Outfit**!
