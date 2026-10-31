import hashlib
import json
import re
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import PropertyMock, patch

import requests

import desktop_app
from desktop_app import BACKEND_URL, ChatStore, DesktopBridge, detect_comfyui_backend_port


class ChatStoreTests(unittest.TestCase):
    def test_saves_and_reloads_multiple_conversations(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatStore(Path(directory) / "chats.json")
            conversations = [
                {"id": "chat-1", "title": "First look", "messages": [{"text": "Blue coat"}]},
                {"id": "chat-2", "title": "Portrait ✨", "messages": []},
            ]

            store.save(json.dumps(conversations, ensure_ascii=False))

            self.assertEqual(store.load(), conversations)

    def test_atomically_creates_one_initial_chat(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatStore(Path(directory) / "chats.json")

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(lambda _: store.load_or_create(), range(16)))

            self.assertTrue(all(result == results[0] for result in results))
            self.assertEqual(len(results[0]), 1)
            self.assertEqual(results[0][0]["title"], "New chat")
            self.assertEqual(store.load(), results[0])

    def test_rejects_non_list_chat_data(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatStore(Path(directory) / "chats.json")

            with self.assertRaisesRegex(ValueError, "must be a list"):
                store.save(json.dumps({"id": "not-a-list"}))

    def test_reports_corrupt_history_instead_of_silently_discarding_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chats.json"
            path.write_text("{", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Could not read saved chats"):
                ChatStore(path).load()

    def test_removes_assets_only_after_their_chat_references_are_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatStore(Path(directory) / "chats.json")
            store.asset_directory.mkdir()
            referenced = store.asset_directory / ("a" * 32 + ".png")
            orphaned = store.asset_directory / ("b" * 32 + ".jpg")
            referenced.write_bytes(b"kept")
            orphaned.write_bytes(b"removed")
            chats = [{"sourceUrl": f"asset:{referenced.name}", "messages": []}]

            store.save(json.dumps(chats))

            self.assertTrue(referenced.exists())
            self.assertFalse(orphaned.exists())
            store.save("[]")
            self.assertFalse(referenced.exists())


class ComfyUIBackendDiscoveryTests(unittest.TestCase):
    def test_prefers_configured_healthy_port(self):
        response = requests_response(b"{}", "application/json")
        with patch.dict("os.environ", {"FACE_STUDIO_BACKEND_PORT": "8189"}):
            with patch("desktop_app.requests.get", return_value=response) as get:
                self.assertEqual(detect_comfyui_backend_port(), 8189)
        get.assert_called_once_with("http://127.0.0.1:8189/system_stats", timeout=0.75)

    def test_falls_back_to_comfyui_port_8200(self):
        healthy = requests_response(b"{}", "application/json")
        with patch.dict("os.environ", {"FACE_STUDIO_BACKEND_PORT": "8189"}):
            with patch(
                "desktop_app.requests.get",
                side_effect=[requests.ConnectionError(), healthy]
            ) as get:
                self.assertEqual(detect_comfyui_backend_port(), 8200)
        self.assertEqual(get.call_count, 2)

    def test_returns_none_when_no_backend_is_healthy(self):
        with patch.dict("os.environ", {"FACE_STUDIO_BACKEND_PORT": "8189"}):
            with patch("desktop_app.requests.get", side_effect=requests.ConnectionError()):
                self.assertIsNone(detect_comfyui_backend_port())

    def test_rejects_invalid_configured_port(self):
        with patch.dict("os.environ", {"FACE_STUDIO_BACKEND_PORT": "not-a-port"}):
            with self.assertRaisesRegex(RuntimeError, "valid TCP port"):
                detect_comfyui_backend_port()


class DesktopAssetTests(unittest.TestCase):
    def create_bridge(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        bridge = DesktopBridge(ChatStore(Path(directory.name) / "chats.json"))
        self.addCleanup(bridge.stop_backend)
        return bridge

    def test_persists_and_serves_a_studio_image(self):
        bridge = self.create_bridge()
        response = requests_response(b"image-bytes", "image/png")
        with patch("desktop_app.requests.get", return_value=response) as get:
            reference = bridge.save_asset("/outputs/generated.png")
        get.assert_called_once_with(f"{BACKEND_URL}/outputs/generated.png", timeout=60)

        name = reference.removeprefix("asset:")
        served = requests.get(f"{bridge.assets_url}/assets/{name}", timeout=3)
        self.assertEqual(served.status_code, 200)
        self.assertEqual(served.content, b"image-bytes")
        self.assertEqual(served.headers["Content-Type"], "image/png")

    def test_rejects_non_studio_image_urls(self):
        bridge = self.create_bridge()
        with patch("desktop_app.requests.get") as get:
            with self.assertRaisesRegex(ValueError, "Only Face Studio"):
                bridge.save_asset("http://example.com/image.png")
            with self.assertRaisesRegex(ValueError, "Only Face Studio"):
                bridge.save_asset("/outputs/%2e%2e/private.png")
        get.assert_not_called()

    def test_restores_a_saved_reference_to_a_new_backend_image_id(self):
        bridge = self.create_bridge()
        bridge.store.asset_directory.mkdir(exist_ok=True)
        name = "c" * 32 + ".png"
        (bridge.store.asset_directory / name).write_bytes(b"saved-reference")
        response = requests_response(b'{"image_id":"new-image-id"}', "application/json")
        with patch("desktop_app.requests.post", return_value=response) as post:
            image_id = bridge.restore_source(f"asset:{name}")

        self.assertEqual(image_id, "new-image-id")
        self.assertEqual(post.call_args.args[0], f"{BACKEND_URL}/api/upload")
        self.assertEqual(post.call_args.kwargs["files"]["image"][1], b"saved-reference")
        self.assertEqual(post.call_args.kwargs["timeout"], 180)

    def test_rejects_invalid_or_missing_saved_reference(self):
        bridge = self.create_bridge()
        with patch("desktop_app.requests.post") as post:
            with self.assertRaisesRegex(ValueError, "saved reference image is invalid"):
                bridge.restore_source("asset:../../private.png")
            with self.assertRaisesRegex(RuntimeError, "Could not read the saved reference"):
                bridge.restore_source("asset:" + "d" * 32 + ".png")
        post.assert_not_called()


class WindowsPackagingTests(unittest.TestCase):
    def test_frozen_workspace_copies_only_into_user_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            workspace = root / "app-data" / "workspace"
            local_loras = root / "app-data" / "loras"
            (bundle / "static").mkdir(parents=True)
            (bundle / "models").mkdir()
            (bundle / "static" / "index.html").write_text("bundled", encoding="utf-8")
            (bundle / "models" / "manifest.json").write_text("{}", encoding="utf-8")
            (bundle / "extra_models.yaml").write_text("models: []", encoding="utf-8")

            with (patch.object(desktop_app, "IS_FROZEN", True),
                  patch.object(desktop_app, "ROOT", bundle),
                  patch.object(desktop_app, "WORKSPACE_ROOT", workspace),
                  patch.object(desktop_app, "LOCAL_LORA_DIRECTORY", local_loras)):
                desktop_app.prepare_workspace()

            self.assertEqual((workspace / "static" / "index.html").read_text(encoding="utf-8"), "bundled")
            self.assertEqual((workspace / "models" / "manifest.json").read_text(encoding="utf-8"), "{}")
            self.assertTrue((workspace / "inputs").is_dir())
            self.assertTrue(local_loras.is_dir())

    def test_frozen_backend_uses_explicit_runtime_path(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "python.exe"
            runtime.write_bytes(b"test runtime placeholder")
            with (patch.object(desktop_app, "IS_FROZEN", True),
                  patch.dict("os.environ", {"FACE_STUDIO_PYTHON": str(runtime)})):
                self.assertEqual(desktop_app.backend_python(), str(runtime))

    def test_windows_home_has_eight_prebuilt_prompt_cards(self):
        source = (Path(__file__).parent / "desktop" / "index.html").read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r'class="prompt-card"', source)), 8)

    def test_reference_images_in_messages_use_compact_thumbnail_sizing(self):
        root = Path(__file__).parent / "desktop"
        script = (root / "app.js").read_text(encoding="utf-8")
        styles = (root / "style.css").read_text(encoding="utf-8")
        self.assertIn('"Reference photo", "source-thumb"', script)
        self.assertIn('if (imageClass) image.classList.add(imageClass);', script)
        self.assertRegex(styles, r"\.source-thumb\s*\{\s*display:block;\s*width:56px;\s*height:72px;")

    def test_updater_does_not_report_up_to_date_without_a_published_release(self):
        source = (Path(__file__).parent / "desktop" / "app.js").read_text(encoding="utf-8")
        self.assertIn('} else if (!result.release_published) {', source)
        self.assertIn('elements.updateStatus.textContent = "No Windows release published";', source)

    def test_windows_ui_defaults_to_natural_face_detail(self):
        root = Path(__file__).parent / "desktop"
        script = (root / "app.js").read_text(encoding="utf-8")
        page = (root / "index.html").read_text(encoding="utf-8")
        self.assertIn("faceIdWeight: 0.80", script)
        self.assertRegex(page, r'id="faceIdWeight"[^>]*max="0\.90"[^>]*value="0\.80"')

    def test_release_workflow_versions_exceed_manually_built_1_0_releases(self):
        source = (Path(__file__).parent / ".github" / "workflows" / "windows-desktop-release.yml")
        self.assertIn("APP_VERSION: 1.1.${{ github.run_number }}", source.read_text(encoding="utf-8"))

    def test_installer_is_per_user_and_creates_shortcuts(self):
        source = (Path(__file__).parent / "windows_installer.iss").read_text(encoding="utf-8")
        self.assertIn("DefaultDirName={localappdata}\\Programs\\FaceStudio", source)
        self.assertIn("PrivilegesRequired=lowest", source)
        self.assertIn("{autoprograms}\\Face Studio", source)


class UpdateTests(unittest.TestCase):
    def create_bridge(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        bridge = DesktopBridge(ChatStore(Path(directory.name) / "chats.json"))
        self.addCleanup(bridge.stop_backend)
        return bridge

    def test_finds_newest_desktop_release_and_compares_versions(self):
        bridge = self.create_bridge()
        response = requests_response(json.dumps([
            {"tag_name": "desktop-v1.0.2", "draft": False, "body": "Second build"},
            {"tag_name": "desktop-v1.0.1", "draft": False, "body": "First build"},
            {"tag_name": "v2.0.0", "draft": False, "body": "Unrelated release"},
        ]).encode(), "application/json")
        with patch.object(desktop_app, "APP_VERSION", "1.0.1"), \
             patch("desktop_app.requests.get", return_value=response):
            result = bridge.check_for_updates()

        self.assertTrue(result["success"])
        self.assertTrue(result["update_available"])
        self.assertEqual(result["version"], "1.0.2")
        self.assertEqual(result["release_notes"], "Second build")

    def test_update_check_does_not_depend_on_chardet(self):
        bridge = self.create_bridge()
        response = requests_response(json.dumps([
            {"tag_name": "desktop-v1.0.2", "draft": False, "body": "Update"}
        ]).encode(), "application/json")
        with patch.object(desktop_app, "APP_VERSION", "1.0.1"), \
             patch("desktop_app.requests.get", return_value=response), \
             patch.object(
                 requests.Response,
                 "text",
                 new_callable=PropertyMock,
                 side_effect=AttributeError("module 'chardet' has no attribute 'detect'")
             ):
            result = bridge.check_for_updates()

        self.assertTrue(result["success"])
        self.assertTrue(result["update_available"])
        self.assertEqual(result["version"], "1.0.2")

    def test_reports_current_version_when_no_windows_release_exists(self):
        bridge = self.create_bridge()
        response = requests_response(b"[]", "application/json")
        with patch("desktop_app.requests.get", return_value=response):
            result = bridge.check_for_updates()
        self.assertEqual(result, {
            "success": True,
            "update_available": False,
            "release_published": False,
            "version": desktop_app.APP_VERSION
        })

    def test_downloads_only_a_checksum_verified_windows_installer(self):
        bridge = self.create_bridge()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        installer = b"MZ" + b"windows-installer-payload"
        checksum = hashlib.sha256(installer).hexdigest()
        api_response = requests_response(json.dumps([{
            "tag_name": "desktop-v1.0.2",
            "draft": False,
            "body": "",
            "assets": [
                {"name": "FaceStudio-Setup.exe", "browser_download_url":
                    "https://github.com/aswanthvijay/face-studio/releases/download/desktop-v1.0.2/FaceStudio-Setup.exe"},
                {"name": "FaceStudio-Setup.exe.sha256", "browser_download_url":
                    "https://github.com/aswanthvijay/face-studio/releases/download/desktop-v1.0.2/FaceStudio-Setup.exe.sha256"},
            ],
        }]).encode(), "application/json")
        checksum_response = requests_response(f"{checksum}  FaceStudio-Setup.exe\n".encode(), "text/plain")
        installer_response = requests_response(installer, "application/octet-stream")
        installer_response._content_consumed = True
        installer_response.headers["Content-Length"] = str(len(installer))
        with (patch.object(desktop_app, "APP_VERSION", "1.0.1"),
              patch.object(desktop_app, "APP_DATA", Path(directory.name)),
              patch("desktop_app.requests.get", side_effect=[api_response, checksum_response, installer_response]),
              patch.object(
                  requests.Response,
                  "text",
                  new_callable=PropertyMock,
                  side_effect=AttributeError("module 'chardet' has no attribute 'detect'")
              ),
              patch("desktop_app.subprocess.Popen") as popen):
            result = bridge.install_update()

        self.assertTrue(result["success"])
        self.assertTrue(result["installer_started"])
        launched_path = Path(popen.call_args.args[0][0])
        self.assertEqual(launched_path.read_bytes(), installer)

    def test_rejects_installer_checksum_mismatch(self):
        bridge = self.create_bridge()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        payload = b"MZtampered-installer"
        api_response = requests_response(json.dumps([{
            "tag_name": "desktop-v1.0.2", "draft": False, "body": "",
            "assets": [
                {"name": "FaceStudio-Setup.exe", "browser_download_url":
                    "https://github.com/aswanthvijay/face-studio/releases/download/desktop-v1.0.2/FaceStudio-Setup.exe"},
                {"name": "FaceStudio-Setup.exe.sha256", "browser_download_url":
                    "https://github.com/aswanthvijay/face-studio/releases/download/desktop-v1.0.2/FaceStudio-Setup.exe.sha256"},
            ],
        }]).encode(), "application/json")
        checksum_response = requests_response((hashlib.sha256(b"expected").hexdigest() + "\n").encode(), "text/plain")
        installer_response = requests_response(payload, "application/octet-stream")
        installer_response._content_consumed = True
        with (patch.object(desktop_app, "APP_VERSION", "1.0.1"),
              patch.object(desktop_app, "APP_DATA", Path(directory.name)),
              patch("desktop_app.requests.get", side_effect=[api_response, checksum_response, installer_response]),
              patch("desktop_app.subprocess.Popen") as popen):
            result = bridge.install_update()

        self.assertFalse(result["success"])
        self.assertIn("SHA-256", result["error"])
        popen.assert_not_called()


def requests_response(content, content_type):
    response = requests.Response()
    response.status_code = 200
    response._content = content
    response.headers["Content-Type"] = content_type
    return response


if __name__ == "__main__":
    unittest.main()
