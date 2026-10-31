# Face Studio & Photovance AI

## Repository setup

This repository includes the Python web studio, native Android source, store assets,
and prebuilt APK/AAB files in `release_binaries/`. Model weights, build caches,
uploaded/generated photos, local test fixtures, training images, and signing keys
are kept outside Git. Tests that use those image fixtures require local copies.

Configure model paths in `core/config.py` for your machine and obtain the required
weights separately. The helpers in `scripts/` download supported model assets;
model configuration, manifests, and licenses are included under `models/`.

### Windows desktop app

The desktop shell opens Face Studio in its own Windows window and adds multiple
saved generation chats. Chats keep their own reference image, model, mode,
garment target, variation count, prompts, and generated results. Chat history is
stored locally in `%LOCALAPPDATA%\FaceStudio\chats.json`; chat images are kept in
its `assets` folder and removed when no saved chat references them.

1. Run `install_windows_app.bat` once to install the small WebView desktop dependency
   into the Face Studio Python runtime.
2. Run `run_windows_app.bat` to open the desktop app. The local image server is
   started automatically when it is not already running. The app checks the
   configured ComfyUI port, then common local ports 8189 and 8200, so it can
   connect to an already-running ComfyUI instance on either port. Set
   `FACE_STUDIO_BACKEND_PORT` to use a different local port.
3. Optionally run `package_windows_app.bat` to build a distributable folder at
   `dist\windows\FaceStudio`. Distribute the entire folder, not only the `.exe`;
   the target PC still needs the Face Studio ComfyUI Python runtime, models, and
   inference assets. Set `FACE_STUDIO_PYTHON` if that runtime is installed elsewhere.
4. The Windows release workflow also builds `FaceStudio-Setup.exe`, a per-user
   installer with Start menu and optional desktop shortcuts. It installs under
   `%LOCALAPPDATA%\Programs\FaceStudio` and does not require administrator rights.
   If Inno Setup 6 is installed locally, `package_windows_app.bat` builds this
   installer alongside the portable app folder.

Every update pushed to `main` triggers a versioned Windows prerelease. The desktop
app checks GitHub Releases when it starts and also provides **Check for updates**.
When a newer build is available, choose **Update now** to download and verify the
installer checksum before opening the standard Windows setup wizard. The app only
contacts GitHub for release metadata during an update check; photos and chat history
remain local.

The desktop app still requires the configured ComfyUI runtime, model files, and
inference assets. It does not upload chats or photos to a third-party service.
JPEG, PNG, and WebP uploads are supported. Files named `.CR2` are accepted when
their contents are JPEG-encoded; actual camera RAW files must be exported to
JPEG, PNG, or WebP before upload.

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
   - **Mode 1: In-Place Outfit Swap**: Inpaints the detected garment with SegFormer parsing and a feathered boundary. Pixels outside the active edit mask are preserved; the feathered transition can also change.
   - **Mode 2: Full Scene Generation**: Synthesizes a new editorial scene from the prompt, then transfers facial identity and calibrates face tone toward the reference while retaining most generated scene lighting.
3. **Face Identity & Complexion Matching**:
   - Uses face-restoration and identity-transfer models; similarity depends on the source photo and generated scene.
   - Perceptual lightness ($L^*$) calibration anchors the transferred face to the reference complexion without discarding scene exposure.
   - Optional high-frequency detail processing preserves natural-looking skin texture.
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
- **Local Image Retention**: The desktop app stores chat reference and result images locally; temporary server uploads and outputs are automatically cleaned up.
- **Encrypted in Transit**: Mobile traffic flows over TLS 1.3 / HTTPS via Cloudflare tunnels.
- **Biometric Integrity**: Biometric facial embeddings are never stored permanently, monetized, or exported.
