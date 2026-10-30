# Google Play Store Publishing Guide: Photovance AI
**Package Name:** `com.photovance.app`  
**Application Title:** `Photovance AI — Photo & Outfit Studio`  
**Target SDK:** 34 (Android 14) / Min SDK 26 (Android 8.0)  

---

## 1. Release Keystore Credentials
The production Android App Bundle (`.aab`) and APK are digitally signed with `release.keystore`:
- **File Location:** `android/app/release.keystore`
- **Keystore Password:** Set privately in `android/signing.properties` or `ANDROID_KEYSTORE_PASSWORD`.
- **Key Alias:** `photovance`
- **Key Password:** Set privately in `android/signing.properties` or `ANDROID_KEY_PASSWORD`.

---

## 2. Compiling the Production Release Bundle
Run the automated build script:
```bat
D:\face_studio\android\scripts\build_app.bat
```
Upon completion, the build script produces:
1. **Play Store App Bundle:** `android/app/build/outputs/bundle/release/app-release.aab`  
   *(Upload this file directly to Google Play Console under Production / Closed Testing).*
2. **Direct Install APK:** `android/app/build/outputs/apk/release/app-release.apk`  
   *(For sideloading or direct tester installations).*

---

## 3. Play Store Assets Checklist (`android/play_store_assets/`)
| Asset | Dimensions | Status |
| :--- | :--- | :--- |
| **App Icon** | 512 × 512 px (32-bit PNG) | `android/play_store_assets/icon_512.png` |
| **Feature Graphic** | 1024 × 500 px (PNG) | `android/play_store_assets/feature_graphic_1024x500.png` |
| **Phone Screenshots** | 1080 × 2400 px | Templates ready in `android/play_store_assets/` |
| **Privacy Policy URL** | Hosted HTTPS URL | Host `privacy.html` on GitHub Pages or custom domain |

---

## 4. Google Play Console Data Safety Questionnaire Answers
When filling out the **Data Safety** section on Play Console:

1. **Does your app collect or share any user data?**  
   - Select **Yes**.
2. **Is all of the user data collected encrypted in transit?**  
   - Select **Yes** (all API communications use HTTPS / TLS 1.3).
3. **Do you provide a way for users to request data deletion?**  
   - Select **Yes** (via contact email or app cache clear).
4. **Data Types Collected:**
   - **Photos and Videos (Photos):**
     - Purpose: **App functionality** (outfit replacement and portrait rendering).
     - Ephemeral processing: **Yes** (not stored permanently; transient inference only).
   - **Device or other IDs (Advertising ID):**
     - Purpose: **Advertising or marketing, Analytics** (collected by Google Mobile Ads / AdMob SDK).
   - **Crash logs & Diagnostics:**
     - Purpose: **Analytics & App performance** (collected by Firebase Crashlytics).

---

## 5. Google Play Generative AI Policy Declaration
1. **Does your app generate AI content?**  
   - Select **Yes**.
2. **Does your app implement safeguards against harmful content?**  
   - Select **Yes** (server-side negative prompting and keyword filters prevent NSFW, hate, and deepfake abuses).
3. **Can users report offensive content?**  
   - Select **Yes** (report mechanism in app settings).

---

## 6. One-Click Worldwide Reviewer Access (Cloudflare Tunnel)
To allow Google Play app reviewers or remote test devices worldwide to use the app without port forwarding:
```bat
D:\face_studio\android\scripts\start_public_tunnel.bat
```
This prints a live public HTTPS endpoint:
```
https://xxxx-xxxx.trycloudflare.com
```
Simply paste this URL into the Photovance Android App's **Settings Sheet** to connect instantly!
