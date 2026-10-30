# Photovance · by Dovrali Studio

The Android redesign implements the selected Obsidian Violet dark appearance and Ivory Editorial light appearance with the same Photo → Style → Result flow. Settings includes System, Light and Dark choices; the selection is saved locally. The folded-light mark is a native vector and adaptive launcher icon with a monochrome layer.

## Review the design

- Install the debug APK from `app/build/outputs/apk/debug/app-debug.apk`.
- The opening screen includes Google, Apple, email/password, signup, password-reset and email-link design previews. These do **not** authenticate, create an account, send email, store passwords or contact a provider. Use **Explore** in the header or **Explore as a guest** to enter the studio.
- Select a photo with the system photo picker, or choose the bundled sample. Browse styles and edit the description, mode, output format and outfit area.
- In Settings, configure your existing studio server. A photo is uploaded when you press **Generate photo**. Actual AI generation still requires that server and its models.
- Compare the result by dragging the divider or using **View original**. Save to the gallery or share an actual image file. Recent looks are available for the current app session and retain their original photo/source for comparison and restyling.

## Compose files

- `ui/theme/Theme.kt`: both palettes, typography and appearance selection.
- `ui/components/StudioComponents.kt`: shared headers, step navigation, buttons and notices.
- `ui/screens/AuthScreen.kt`: `LoginScreen` and `SignupScreen` callbacks; no auth dependency.
- `ui/screens/HomeScreen.kt`: `PhotoScreen`, `StyleScreen`, `ResultScreen` and the editor shell.
- `ui/screens/SettingsSheet.kt`: appearance, connection and advanced creation controls.
- `ui/StudioViewModel.kt`: lifecycle-aware editor state, generation, history and export actions.
- `media/StudioMedia.kt`: bounded image preparation with EXIF orientation, gallery save and image sharing.

## Validation and previews

Completed on 15 October 2026: debug APK build passed; Android lint passed with zero errors (existing/dependency/resource warnings remain); all five local UI tests passed. `git diff --check` passed.

`StudioFlowTest` uses Robolectric native graphics, Compose UI tests, Roborazzi and a local MockWebServer. It exercises guest navigation, signup validation/provider-preview behaviour, generation payloads, original/result pairing, failure/retry, saved appearance and small-screen large-text controls. Preview PNGs are generated in `app/build/design-preview/`. Result preview images are explicitly named `*-fixture.png`: they use test images and do not demonstrate a live AI generation.

Run `gradlew.bat :app:assembleDebug :app:lintDebug :app:testDebugUnitTest`. This workstation also has Gradle at `D:\gradle-8.7\bin\gradle.bat`. If the desktop host's Java temporary-directory environment causes a loopback error, use an existing writable Windows temp path for the process's `TEMP`, `TMP` and `-Djava.io.tmpdir` settings. No global environment settings were changed for this task.

No emulator or physical device was attached during this work. The debug APK, automated local tests and native rendered previews support design review; live server generation and device gallery/share targets still need a device pass. This is a design build, not a Play Store release or a completed authentication integration.

## Photography privacy update

All six bundled private photographs have been replaced with stock photos whose individual source pages state CC0. Login, the sample portrait, style cards and local test fixtures now use those assets. See [IMAGE_SOURCES.md](IMAGE_SOURCES.md) for sources and licence details. The originals are preserved outside the app in the ignored private_assets/android-originals directory. Only the newly rebuilt APK and regenerated previews should be distributed.

The rebuilt APK was inspected as a ZIP: exactly six JPEG photo entries match the packaged licence manifest SHA-256 values, none of the six original private photo filenames or hashes occurs among its photo entries, and the manifest is included. All nine design previews were regenerated. The original files were verified against their Git versions after preservation. Debug build, lint and five local UI tests passed after replacement.
