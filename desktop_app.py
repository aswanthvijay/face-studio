"""Windows desktop shell for Face Studio with locally saved generation chats."""
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import requests


IS_FROZEN = bool(getattr(sys, "frozen", False))
ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)).resolve()
PORT = int(os.environ.get("FACE_STUDIO_PORT", "7870"))
BACKEND_URL = f"http://127.0.0.1:{PORT}"
APP_DATA = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "FaceStudio"
CHAT_FILE = APP_DATA / "chats.json"
GITHUB_REPOSITORY = "aswanthvijay/face-studio"
GITHUB_RELEASES_API = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/releases?per_page=30"
APP_VERSION = "1.0.0-dev"
BUILD_INFO_PATH = ROOT / "desktop" / "build_info.json"
if BUILD_INFO_PATH.is_file():
    try:
        BUILD_INFO = json.loads(BUILD_INFO_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Desktop build version metadata is invalid: {error}") from error
    version = BUILD_INFO.get("version") if isinstance(BUILD_INFO, dict) else None
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?", version):
        raise RuntimeError("Desktop build version metadata must contain a valid version string.")
    APP_VERSION = version
WORKSPACE_ROOT = Path(os.environ.get(
    "FACE_STUDIO_WORKSPACE_ROOT",
    APP_DATA / "workspace" / APP_VERSION if IS_FROZEN else ROOT
)).resolve()
LOCAL_LORA_DIRECTORY = Path(os.environ.get(
    "FACE_STUDIO_LOCAL_LORA_DIR",
    APP_DATA / "loras" if IS_FROZEN else WORKSPACE_ROOT / "models" / "loras"
)).resolve()
ASSET_NAME_PATTERN = re.compile(r"[0-9a-f]{32}\.(?:png|jpg|webp)\Z")


def prepare_workspace():
    if not IS_FROZEN:
        return
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("static", "models"):
        source = ROOT / name
        destination = WORKSPACE_ROOT / name
        if source.is_dir() and not destination.exists():
            shutil.copytree(source, destination)
    for name in ("extra_models.yaml", "privacy.html"):
        source = ROOT / name
        destination = WORKSPACE_ROOT / name
        if source.is_file() and not destination.exists():
            shutil.copy2(source, destination)
    for name in ("inputs", "outputs", "samples"):
        (WORKSPACE_ROOT / name).mkdir(parents=True, exist_ok=True)
    LOCAL_LORA_DIRECTORY.mkdir(parents=True, exist_ok=True)


def backend_python():
    if not IS_FROZEN:
        return sys.executable
    configured = os.environ.get("FACE_STUDIO_PYTHON")
    candidates = [
        Path(configured) if configured else None,
        Path(r"D:\Image AI\StabilityMatrix-win-x64\Data\Packages\ComfyUI\venv\Scripts\python.exe"),
        Path(r"D:\Image AI L model\runtime\python.exe")
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return str(candidate)
    raise RuntimeError(
        "Face Studio's Python inference runtime was not found. Set FACE_STUDIO_PYTHON "
        "to the Python executable in your ComfyUI environment."
    )


def detect_comfyui_backend_port():
    """Return a healthy local ComfyUI port, preferring the configured port."""
    configured_port = os.environ.get("FACE_STUDIO_BACKEND_PORT", "8189")
    try:
        preferred_port = int(configured_port)
    except ValueError as error:
        raise RuntimeError("FACE_STUDIO_BACKEND_PORT must be a valid TCP port.") from error
    if not 1 <= preferred_port <= 65535:
        raise RuntimeError("FACE_STUDIO_BACKEND_PORT must be between 1 and 65535.")

    ports = dict.fromkeys((preferred_port, 8189, 8200))
    for port in ports:
        try:
            response = requests.get(f"http://127.0.0.1:{port}/system_stats", timeout=0.75)
            if response.status_code == 200:
                return port
        except requests.RequestException:
            continue
    return None


class LocalAssetHandler(BaseHTTPRequestHandler):
    asset_directory = None

    def do_GET(self):
        name = urlsplit(self.path).path.removeprefix("/assets/")
        if not ASSET_NAME_PATTERN.fullmatch(name):
            self.send_error(404)
            return
        path = (self.asset_directory / name).resolve()
        if path.parent != self.asset_directory.resolve() or not path.is_file():
            self.send_error(404)
            return
        try:
            content = path.read_bytes()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "private, max-age=3600")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        return


class ChatStore:
    def __init__(self, path=CHAT_FILE):
        self.path = Path(path)
        self.asset_directory = self.path.parent / "assets"
        self.lock = threading.Lock()

    def load(self):
        with self.lock:
            return self._load_unlocked()

    def load_or_create(self):
        with self.lock:
            chats = self._load_unlocked()
            if chats:
                return chats
            chats = [{
                "id": str(uuid.uuid4()),
                "title": "New chat",
                "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "messages": [],
                "imageId": None,
                "sourceUrl": None,
                "sourceName": None,
                "modelId": "",
                "mode": "in_place_swap",
                "target": "dress",
                "variations": 1,
                "enableFaceId": True,
                "faceIdWeight": 0.65
            }]
            self._save_unlocked(chats)
            return chats

    def _load_unlocked(self):
        if not self.path.exists():
            return []
        try:
            chats = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Could not read saved chats: {error}") from error
        if not isinstance(chats, list) or not all(isinstance(chat, dict) for chat in chats):
            raise RuntimeError("Saved chat history has an invalid format.")
        return chats

    def save(self, payload):
        try:
            chats = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Chat history is not valid JSON: {error}") from error
        if not isinstance(chats, list) or not all(isinstance(chat, dict) for chat in chats):
            raise ValueError("Chat history must be a list of conversations.")

        with self.lock:
            self._save_unlocked(chats)
        return True

    def _save_unlocked(self, chats):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(chats, ensure_ascii=False), encoding="utf-8")
            temporary.replace(self.path)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Could not save chat history: {error}") from error
        self._remove_unreferenced_assets(chats)

    def _remove_unreferenced_assets(self, chats):
        referenced = set()
        for chat in chats:
            urls = [chat.get("sourceUrl")]
            for message in chat.get("messages", []):
                urls.append(message.get("sourceUrl"))
                urls.extend(message.get("results", []))
            for url in urls:
                if isinstance(url, str) and url.startswith("asset:"):
                    name = url.removeprefix("asset:")
                    if ASSET_NAME_PATTERN.fullmatch(name):
                        referenced.add(name)
        if self.asset_directory.is_dir():
            for asset in self.asset_directory.iterdir():
                if ASSET_NAME_PATTERN.fullmatch(asset.name) and asset.name not in referenced and asset.is_file():
                    asset.unlink()


class DesktopBridge:
    def __init__(self, store=None):
        self.store = store or ChatStore()
        self.server = None
        self.server_log = None
        self.window = None
        self.available_release = None
        self.start_lock = threading.Lock()
        self.store.asset_directory.mkdir(parents=True, exist_ok=True)
        LocalAssetHandler.asset_directory = self.store.asset_directory
        self.asset_server = ThreadingHTTPServer(("127.0.0.1", 0), LocalAssetHandler)
        self.asset_server_thread = threading.Thread(target=self.asset_server.serve_forever, daemon=True)
        self.asset_server_thread.start()
        self.assets_url = f"http://127.0.0.1:{self.asset_server.server_address[1]}"

    def backend_url(self):
        return BACKEND_URL

    def load_chats(self):
        return self.store.load_or_create()

    def save_chats(self, payload):
        return self.store.save(payload)

    def check_for_updates(self):
        try:
            response = requests.get(
                GITHUB_RELEASES_API,
                headers={"Accept": "application/vnd.github+json", "User-Agent": "FaceStudio-Desktop"},
                timeout=15
            )
            response.raise_for_status()
            releases = response.json()
        except (requests.RequestException, requests.exceptions.JSONDecodeError) as error:
            return {"success": False, "error": f"Could not check for updates: {error}"}
        if not isinstance(releases, list):
            return {"success": False, "error": "The update service returned an invalid release list."}

        candidates = []
        for release in releases:
            if not isinstance(release, dict) or release.get("draft"):
                continue
            tag = release.get("tag_name", "")
            version_match = re.fullmatch(r"desktop-v(\d+\.\d+\.\d+)", tag) if isinstance(tag, str) else None
            if version_match:
                candidates.append((tuple(int(part) for part in version_match.group(1).split(".")), release))
        if not candidates:
            self.available_release = None
            return {
                "success": True,
                "update_available": False,
                "release_published": False,
                "version": APP_VERSION
            }

        latest_version, release = max(candidates, key=lambda candidate: candidate[0])
        current_match = re.match(r"^(\d+\.\d+\.\d+)", APP_VERSION)
        current_version = tuple(int(part) for part in current_match.group(1).split(".")) if current_match else (0, 0, 0)
        update_available = latest_version > current_version
        self.available_release = release if update_available else None
        return {
            "success": True,
            "update_available": update_available,
            "release_published": True,
            "version": ".".join(str(part) for part in latest_version),
            "current_version": APP_VERSION,
            "release_notes": (release.get("body") or "")[:4000] if update_available else ""
        }

    def install_update(self):
        checked = self.check_for_updates()
        if not checked.get("success"):
            return checked
        if not checked.get("update_available"):
            return {"success": False, "error": "Face Studio is already up to date."}

        release = self.available_release
        assets = release.get("assets", []) if isinstance(release, dict) else []
        if not isinstance(assets, list):
            return {"success": False, "error": "The latest release has an invalid asset list."}
        asset_by_name = {
            asset.get("name"): asset for asset in assets
            if isinstance(asset, dict) and isinstance(asset.get("name"), str)
        }
        installer_asset = asset_by_name.get("FaceStudio-Setup.exe")
        checksum_asset = asset_by_name.get("FaceStudio-Setup.exe.sha256")
        if installer_asset is None or checksum_asset is None:
            return {"success": False, "error": "The latest release is missing its installer or checksum."}

        update_directory = APP_DATA / "updates"
        update_directory.mkdir(parents=True, exist_ok=True)
        installer_path = update_directory / f"FaceStudio-Setup-{checked['version']}.exe"
        try:
            checksum_url = checksum_asset.get("browser_download_url", "")
            installer_url = installer_asset.get("browser_download_url", "")
            tag = release["tag_name"]
            for url, asset_name in (
                (checksum_url, "FaceStudio-Setup.exe.sha256"),
                (installer_url, "FaceStudio-Setup.exe")
            ):
                parsed_url = urlsplit(url)
                expected_path = f"/{GITHUB_REPOSITORY}/releases/download/{tag}/{asset_name}"
                if (parsed_url.scheme != "https" or parsed_url.netloc != "github.com"
                        or parsed_url.path != expected_path or parsed_url.query or parsed_url.fragment):
                    raise ValueError("The release contains an unexpected installer download URL.")
            checksum_response = requests.get(checksum_url, timeout=30)
            checksum_response.raise_for_status()
            expected_hash = checksum_response.text.strip().split()[0].lower()
            if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
                raise ValueError("The installer checksum file is invalid.")

            download = requests.get(installer_url, stream=True, timeout=(15, 300))
            download.raise_for_status()
            content_length = int(download.headers.get("Content-Length", "0") or 0)
            if content_length > 1024 * 1024 * 1024:
                raise ValueError("The Windows installer exceeds the 1 GB download limit.")
            temporary_path = installer_path.with_suffix(".download")
            digest = hashlib.sha256()
            total = 0
            try:
                with temporary_path.open("wb") as target:
                    for chunk in download.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > 1024 * 1024 * 1024:
                            raise ValueError("The Windows installer exceeds the 1 GB download limit.")
                        digest.update(chunk)
                        target.write(chunk)
                if digest.hexdigest() != expected_hash:
                    raise ValueError("The downloaded installer failed its SHA-256 integrity check.")
                with temporary_path.open("rb") as downloaded:
                    signature = downloaded.read(2)
                if signature != b"MZ":
                    raise ValueError("The downloaded file is not a valid Windows installer.")
                temporary_path.replace(installer_path)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise
        except (OSError, ValueError, requests.RequestException, IndexError) as error:
            return {"success": False, "error": f"Could not prepare the update: {error}"}

        try:
            subprocess.Popen(
                [str(installer_path)],
                cwd=str(update_directory),
                close_fds=True
            )
        except OSError as error:
            return {"success": False, "error": f"Could not launch the Windows installer: {error}"}
        return {"success": True, "version": checked["version"], "installer_started": True}

    def save_asset(self, relative_url):
        parsed = urlsplit(relative_url) if isinstance(relative_url, str) else None
        path_parts = unquote(parsed.path).strip("/").split("/") if parsed is not None else []
        if (parsed is None or parsed.scheme or parsed.netloc or parsed.query or parsed.fragment
                or not parsed.path.startswith(("/inputs/", "/outputs/"))
                or len(path_parts) != 2 or path_parts[0] not in ("inputs", "outputs")
                or not path_parts[1] or path_parts[1] in (".", "..")
                or "/" in path_parts[1] or "\\" in path_parts[1]):
            raise ValueError("Only Face Studio input and output images can be saved.")
        response = requests.get(f"{BACKEND_URL}{parsed.path}", timeout=60)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
        extension = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp"
        }.get(content_type)
        if extension is None:
            raise ValueError("The studio returned an unsupported image type.")
        content = response.content
        if len(content) > 50 * 1024 * 1024:
            raise ValueError("The saved image exceeds the 50 MB limit.")
        self.store.asset_directory.mkdir(parents=True, exist_ok=True)
        name = uuid.uuid4().hex + extension
        target = self.store.asset_directory / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            temporary.write_bytes(content)
            temporary.replace(target)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Could not preserve the image in this chat: {error}") from error
        return f"asset:{name}"

    def save_assets(self, payload):
        try:
            urls = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Image list is not valid JSON: {error}") from error
        if not isinstance(urls, list) or not all(isinstance(url, str) for url in urls):
            raise ValueError("Image list must be an array of studio image paths.")
        return [self.save_asset(url) for url in urls]

    def restore_source(self, asset_reference):
        if not isinstance(asset_reference, str) or not asset_reference.startswith("asset:"):
            raise ValueError("The saved reference image is invalid.")
        name = asset_reference.removeprefix("asset:")
        if not ASSET_NAME_PATTERN.fullmatch(name):
            raise ValueError("The saved reference image is invalid.")
        path = self.store.asset_directory / name
        try:
            content = path.read_bytes()
        except OSError as error:
            raise RuntimeError(f"Could not read the saved reference image: {error}") from error
        if len(content) > 50 * 1024 * 1024:
            raise ValueError("The saved reference image exceeds the 50 MB limit.")
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        response = requests.post(
            f"{BACKEND_URL}/api/upload",
            files={"image": (path.name, content, content_type)},
            timeout=180
        )
        response.raise_for_status()
        try:
            result = response.json()
        except requests.exceptions.JSONDecodeError as error:
            raise RuntimeError("The studio returned an invalid response while restoring the reference image.") from error
        image_id = result.get("image_id") if isinstance(result, dict) else None
        if not image_id:
            raise RuntimeError(result.get("error", "The studio could not restore the reference image.")
                               if isinstance(result, dict) else
                               "The studio could not restore the reference image.")
        return image_id

    def start_backend(self):
        prepare_workspace()
        with self.start_lock:
            try:
                response = requests.get(f"{BACKEND_URL}/health", timeout=1.5)
                if response.ok:
                    return {"ready": True, "url": BACKEND_URL, "assets_url": self.assets_url, "started_here": False}
            except requests.RequestException:
                pass

            if self.server is None or self.server.poll() is not None:
                APP_DATA.mkdir(parents=True, exist_ok=True)
                environment = os.environ.copy()
                environment["FACE_STUDIO_WORKSPACE_ROOT"] = str(WORKSPACE_ROOT)
                environment["FACE_STUDIO_LOCAL_LORA_DIR"] = str(LOCAL_LORA_DIRECTORY)
                environment["FACE_STUDIO_HOST"] = "127.0.0.1"
                backend_port = detect_comfyui_backend_port()
                if backend_port is not None:
                    environment["FACE_STUDIO_BACKEND_PORT"] = str(backend_port)
                self.server_log = (APP_DATA / "server.log").open("a", encoding="utf-8")
                creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                self.server = subprocess.Popen(
                    [backend_python(), "-u", str(ROOT / "server.py")],
                    cwd=ROOT,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=self.server_log,
                    stderr=subprocess.STDOUT,
                    creationflags=creation_flags
                )

        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if self.server is not None and self.server.poll() is not None:
                raise RuntimeError(
                    f"Face Studio server stopped during startup. Check {APP_DATA / 'server.log'}."
                )
            try:
                response = requests.get(f"{BACKEND_URL}/health", timeout=1.5)
                if response.ok:
                    return {"ready": True, "url": BACKEND_URL, "assets_url": self.assets_url, "started_here": True}
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError(
            f"Face Studio did not start within 5 minutes. Check {APP_DATA / 'server.log'}."
        )

    def stop_backend(self):
        if self.server is not None and self.server.poll() is None:
            self.server.terminate()
            try:
                self.server.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait(timeout=3)
        if self.server_log is not None:
            self.server_log.close()
        self.asset_server.shutdown()
        self.asset_server.server_close()
        self.asset_server_thread.join(timeout=3)


def main():
    if sys.platform == "win32" and IS_FROZEN:
        import ctypes
        register_restart = ctypes.WinDLL("kernel32", use_last_error=True).RegisterApplicationRestart
        register_restart.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong]
        register_restart.restype = ctypes.c_ulong
        result = register_restart(None, 0)
        if result != 0:
            print(f"Windows app restart registration failed with code {result}.")
    try:
        import webview
    except ImportError as error:
        raise SystemExit(
            "The desktop window package is missing. Run install_windows_app.bat first."
        ) from error

    prepare_workspace()
    bridge = DesktopBridge()
    window = webview.create_window(
        "Face Studio — AI Image Studio",
        (ROOT / "desktop" / "index.html").as_uri(),
        js_api=bridge,
        width=1440,
        height=920,
        min_size=(960, 640),
        background_color="#090c10"
    )
    bridge.window = window
    try:
        webview.start(gui="edgechromium", debug=False)
    finally:
        bridge.stop_backend()


if __name__ == "__main__":
    main()
