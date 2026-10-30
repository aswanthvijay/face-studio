# Face Studio & Photovance AI

## Repository setup

This repository includes the Python web studio, native Android source, store assets,
and prebuilt APK/AAB files in `release_binaries/`. Model weights, build caches,
uploaded/generated photos, local test fixtures, training images, and signing keys
are kept outside Git. Tests that use those image fixtures require local copies.

Configure model paths in `core/config.py` for your machine and obtain the required
weights separately. The helpers in `scripts/` download supported model assets;
model configuration, manifests, and licenses are included under `models/`.

For signed Android builds, copy `android/signing.properties.example` to
`android/signing.properties`, enter your credentials, and supply your private
keystore. You can also set `ANDROID_KEYSTORE_FILE`, `ANDROID_KEYSTORE_PASSWORD`,
`ANDROID_KEY_ALIAS`, and `ANDROID_KEY_PASSWORD`. Set your Android SDK path in
the local `android/local.properties` file before building.

### Enterprise AI Portrait & Outfit Studio + Google Play Store Ready Android App

---

## 🌟 Executive Overview
**Face Studio AI** is an ultra-realistic, biometrically locked AI fashion and portrait studio built in `D:\face_studio`. It provides both a modern glassmorphism Web Studio operating on **Port 7870** and a production-grade, Google Play Store ready Android application (`com.photovance.app`).

### Key Capabilities:
1. **Competitive Multi-Model Suite**:
   - **CyberRealistic XL**: The #1 King for true DSLR skin micro-pores and authentic epidermal depth.
   - **epiCRealism XL**: Specialized for cinematic lighting, portraits, and lifelike fabric drape.
   - **RealVisXL V5.0**: High-speed benchmark for sharp all-rounder portrait generation.
   - **Flux-2-Klein 4B Turbo**: Unrestricted Diffusion Transformer with Qwen3 Heretic text encoder.
2. **Dual Operation Modes**:
   - **Mode 1: In-Place Outfit Swap**: Inpaints *only* the masked garment using Segformer clothing parsing and smart chin-line feathering. The original body, pose, background, and head remain 100% pixel-perfect.
   - **Mode 2: Full Scene Generation**: Synthesizes a brand new editorial scene and outfit from prompt, then transplants 1:1 biometric identity with CodeFormer ($w=0.85$), luminance matching, and micro-pore reinjection.
3. **100% Biometric Identity & Melanin Preservation**:
   - Discrete Codebook Prior ($w=0.85$) + Inswapper 128.
   - Perceptual lightness ($L^*$) matching eliminates dark muddy or bronze skin tone discoloration.
   - High-frequency Gaussian spatial separation re-injects authentic camera skin pores ($0.35$ factor).
4. **Zero-Pollution Prompting & Strict Negative Safeguards**:
   - User input prompt is strictly conditioned first with zero boilerplate pollution.
   - Melanin-anchored physical descriptors prevent stereotype hallucination (strict suppression of unwanted bindi, forehead dots, sindoor, and artificial plastic smoothing).

---

## 🚀 Quick Start Guide

### 1. Launch the Web Studio (Port 7870)
Double-click `run.bat` or run:
```bat
D:\face_studio\run.bat
```
The Web Studio opens automatically at `http://127.0.0.1:7870`.

For Android on the same Wi-Fi, run `C:\Users\Aswanth Vijay\Documents\antigravity\wise-planck\start_ai_for_android.bat`. It starts both studios and forwards the PC's LAN address to their loopback listeners, which the bundled offline Python runtime requires. Use the printed `http://<PC-LAN-IP>:7870/` in the Android app's Settings. The old `10.0.2.2` default is for an emulator, not a physical phone. Existing installations can update Settings without reinstalling.

### 2. Verify Core AI Modules
Run the automated test suite:
```bat
python test_inference.py
```

### 3. Compile Google Play Store Release App
Double-click `scripts\build_android_app.bat` or run:
```bat
cd D:\face_studio\android
scripts\build_app.bat
```
Produces:
- **Play Store Signed Bundle:** `release_binaries\app-release.aab`
- **Direct Release APK:** `release_binaries\app-release.apk`

### 4. Enable Worldwide Mobile & Reviewer Access
Double-click `scripts\start_public_tunnel.bat` to launch a Cloudflare HTTPS tunnel:
```bat
scripts\start_public_tunnel.bat
```
Copy the printed `https://xyz.trycloudflare.com` URL into the Android App Settings sheet for instant worldwide access without port forwarding.

---

## 📁 Project Directory Structure
```
D:\face_studio\
├── core\                          # Python AI Inference Engine
│   ├── config.py                  # Model paths, ports, and global settings
│   ├── prompt_engine.py           # Zero-pollution prompts & melanin analysis
│   ├── clothing_engine.py         # Segformer clothing parser & chin feathering
│   ├── face_engine.py             # CodeFormer w=0.85, Inswapper 128 & skin pores
│   └── pipeline.py                # ComfyUI GPU orchestration & compositing
├── static\                        # Modern Dark Glassmorphism Web UI
│   ├── index.html                 # Canvas, interactive split slider & prompt chips
│   ├── style.css                  # Atmospheric glow, glassmorphism & responsive CSS
│   └── app.js                     # Touch/drag slider logic & API client
├── release_binaries\              # Pre-compiled Google Play Store Binaries
│   ├── app-release.aab            # Signed Android App Bundle ready for Play Console
│   └── app-release.apk            # Standalone release APK
├── android\                       # Native Jetpack Compose Android Project
│   ├── app\src\main\java\com\photovance\app\
│   ├── play_store_assets\         # 512x512 icon & 1024x500 feature graphic
│   └── scripts\build_app.bat      # Automated compilation script
├── scripts\                       # Utility & Automation Scripts
│   ├── build_android_app.bat      # Quick-compile Android AAB/APK
│   ├── start_public_tunnel.bat    # Cloudflare HTTPS tunnel
│   └── download_models.py         # Additional XL model downloader
├── server.py                      # REST API server (Port 7870)
├── run.bat                        # One-click Web Studio launcher
├── test_inference.py              # Automated test verification suite
├── privacy.html                   # Play Store compliant privacy policy
└── PLAY_STORE_GUIDE.md            # Play Console submission & Data Safety guide
```

---

## 🔒 Security & Privacy Architecture
- **Offline GPU Execution**: All diffusion inference, face swapping, and clothing parsing run strictly on local hardware (NVIDIA RTX GPU).
- **Ephemeral Storage**: Uploaded reference photos and masks are transiently processed and automatically cleaned up.
- **Encrypted in Transit**: Mobile traffic flows over TLS 1.3 / HTTPS via Cloudflare tunnels.
- **Biometric Integrity**: Biometric facial embeddings are never stored permanently, monetized, or exported.
