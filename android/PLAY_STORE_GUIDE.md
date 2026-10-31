# Google Play Store Publishing Guide: Photovance AI

This document provides step-by-step instructions for publishing **Photovance AI** to the Google Play Store.

---

## 0. Production services

The Android client uses Firebase Authentication (Google, email/password, phone SMS, and anonymous guest sessions) and Play Billing subscriptions. The Firebase project, Play Console listing, and subscription products still need to be created under the owner's accounts; no cloud project or paid account is created by this repository. Do not send Firebase configuration files, keystore files, or service-account credentials in chat or commit them.

### Firebase project setup

1. Create a Firebase project in the owner's Google account and register an Android app with package name `com.photovance.app`.
2. Enable **Google**, **Email/Password**, **Phone**, and **Anonymous** sign-in providers in Firebase Authentication.
3. Add the SHA-1 and SHA-256 fingerprints for debug and release signing certificates to the Firebase Android app. Configure the Google sign-in OAuth client for the release package and fingerprints.
4. Download the app's `google-services.json` into `android\app\google-services.json`. It is ignored by Git. The Android Gradle build applies the Google Services plugin only when this file exists; without it, sign-in reports a configuration error rather than simulating authentication.
5. Link the repository to the Firebase project with the Firebase CLI, install Functions dependencies (`npm install` in `functions`), and deploy the callable purchase verifier and Firestore rules:

   ```bat
   firebase use --add
   firebase deploy --only functions,firestore:rules
   ```

6. Enable the Google Play Developer API for the Firebase project's Google Cloud project. Grant the Cloud Functions runtime service account the minimum Android Publisher permissions needed to read subscriptions. Add that service account to Play Console with access to subscription/order data. Do not download or commit a service-account key.

Deploying Cloud Functions requires the Firebase project's Blaze plan and a Google Cloud billing account; phone verification and Functions/Firestore usage may incur charges. Review Google's current pricing and configure budget alerts before enabling paid services.

The callable `verifyPlayPurchase` verifies the purchase token against Google Play before the app marks VIP active. Each purchase is bound to the signed-in Firebase UID using Play Billing's obfuscated account identifier; restoration must use that same Photovance account. It stores only the SHA-256 hash of the token in Firestore; client access to those records is denied by `firestore.rules`. The `deleteAccountData` callable requires a recent sign-in, removes the account's Firestore data, then deletes the Firebase Auth account.

### Downloadable Android preview APK

The `Android APK preview` GitHub Actions workflow runs Android unit tests and builds a debug-signed APK for pull requests targeting `main`. Download the `photovance-android-<version>` artifact from the successful Actions run. Merging the workflow to `main` builds and attaches the APK and its SHA-256 checksum to a prerelease named `android-v<version>`. This is a sideload/testing build, not a Play production release. The version can be set when manually dispatching the workflow from `main`.

### Play Console setup

1. Create the Play Console app using package name `com.photovance.app`, enable Play App Signing, and keep the upload key private.
   The Android build targets API 35 for current Play submission requirements.
2. Create subscription products `vip_weekly` and `vip_yearly`. Configure the weekly base plan at the intended $4.99/week price and a 3-day free-trial offer; configure the annual base plan at the intended $29.99/year price. The app reads localized prices from Google Play, so the console is authoritative.
3. Add license testers and use an internal testing track to verify purchases, restore, cancellation, renewal, and expired entitlements before release.
4. Publish Play installs through Play tracks. Play-installed copies use Google Play in-app updates. Do not distribute a direct APK update to Play-installed copies.
5. For separately sideloaded Android builds, attach a signed APK named `photovance-android-<version>.apk` to the official GitHub release. The app opens that official asset for download; Android's package installer performs the user-confirmed installation and verifies the app signing identity.

**Local Windows GPU pairing is not shipped yet.** The app currently accepts a manually entered HTTPS server URL and reconnects to a saved HTTPS URL. Do not expose the current unauthenticated loopback API directly to a LAN or use a cleartext LAN URL. Secure Windows discovery, pairing, TLS, and server-side authorization remain required before the requested automatic same-Wi-Fi connection is production-ready.

---

## 1. Prerequisites

1. **Google Play Developer Account**:
   - Register at [play.google.com/console](https://play.google.com/console) ($25 one-time registration fee).
2. **Release App Bundle**:
   - The compiled `.aab` file is generated by running `android\scripts\build_app.bat`.
   - File output path: `android\app\build\outputs\bundle\release\app-release.aab`.
3. **Release Keystore**:
   - Keep `android\app\release.keystore` private (alias: `photovance`). Configure passwords in the ignored `android/signing.properties` file; see `signing.properties.example`.
   - Keystore valid for 10,000 days. Keep this file safe for future app updates.

---

## 2. Store Listing Assets (Ready in `android\play_store_assets\`)

* **App Icon**: `android\play_store_assets\icon_512.png` (512 × 512 px PNG, 32-bit).
* **Feature Graphic**: `android\play_store_assets\feature_graphic_1024x500.png` (1024 × 500 px PNG).
* **Screenshots**: Take 2–4 screenshots from your phone running the app showing the Before/After comparison slider and preset chips.

---

## 3. Play Store Listing Copy (Optimized for ASO & Downloads)

* **App Name**: `Photovance AI: Pro Photo Studio`
* **Short Description** (up to 80 chars):
  `AI Photo Editor: Swap outfits, try designer dresses & explore glamorous scenes.`
* **Full Description** (up to 4000 chars):
```text
Transform your look and step into any scene with Photovance AI — the ultimate AI Photo Studio and Outfit Editor.

Whether you want to try on a luxury evening gown, walk the red carpet at Cannes, relax on a superyacht in Monaco, or change your outfit in an existing portrait, Photovance AI delivers studio-quality photorealistic results while keeping your 100% exact facial identity and natural skin tone.

✦ KEY FEATURES:
• Face Studio (Unrestricted): Place your face into any prompt, scene, pose, or world location.
• In-Place Outfit Swap: Keep your original pose, hair, and background while instantly changing clothes.
• Biometric Face Identity Lock: 1:1 facial preservation with HD face restoration — your smile, eyes, and expressions stay authentic.
• Touch Before & After Slider: Compare your original photo with the AI-transformed result in real time.
• Save HD & Share: Export high-resolution images directly to your device Gallery or share to Instagram, WhatsApp, and TikTok.
• Multiple AI Engines: Choose between ultra-fast generation or maximum fabric micro-texture precision.

Elevate your personal style and create unforgettable portraits with Photovance AI today!
```

---

## 4. Privacy Policy & Data Safety Form

* **Privacy Policy URL**:
  - Publish a reviewed policy on a stable HTTPS domain before submitting the listing. Do not use a temporary tunnel URL as the long-term policy URL.
  - The policy must explain Firebase account identifiers/email/phone, Google Play subscription data, and how selected photos are sent to the configured studio server and retained/deleted.
* **Data Safety**:
  - Complete the form from the actual release behavior and Firebase/Play Console configuration; do not copy guessed answers.
  - Account identifiers and any enabled email/phone fields, photos/videos, and purchase/subscription status must be assessed.
  - The app rejects cleartext connections to non-loopback hosts. The selected studio endpoint and Firebase/Google Play data flows still need to be described accurately in the published policy.
  - The in-app account deletion flow is implemented. A stable public web deletion-request page and its Play Console URL still need to be deployed and verified before enabling account creation for a public Play release.

---

## 5. Live Backend Connectivity

The Android client currently reconnects automatically only to a previously saved HTTPS URL. For manual setup, open **Settings → Your studio connection**, enter an HTTPS endpoint, and choose **Save and connect**. Do not use the old LAN HTTP forwarding instructions: the in-repository Windows server has not yet been given the secure pairing bridge required for same-Wi-Fi discovery. A public HTTPS GPU endpoint can be configured manually, but it must be secured, monitored, and approved for handling user photos before Play review.
