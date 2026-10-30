import base64
import json
import os
import sys
import time
import urllib.request
from PIL import Image

def test_live_change_dress():
    print("=== Testing Live /api/change-dress In-Place Swap ===")
    sample_path = os.path.join("samples", "sample_model.png")
    if not os.path.exists(sample_path):
        print(f"Sample path {sample_path} not found!")
        sys.exit(1)

    with open(sample_path, "rb") as f:
        img_b64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"

    payload = {
        "image": img_b64,
        "prompt": "royal emerald green silk evening gown, delicate gold floral embroidery, cinematic studio lighting",
        "model": "cyberrealistic_xl",
        "mode": "in_place_swap",
        "target_garment": "dress",
        "variations": 1
    }

    req = urllib.request.Request(
        "http://127.0.0.1:7870/api/change-dress",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            elapsed = time.time() - start
            data = json.loads(resp.read().decode())
            print(f"[OK] Response 200 OK received in {elapsed:.2f}s!")
            print(f"Result URL: {data.get('result_url')}")
            print(f"Original URL: {data.get('original_url')}")
            print(f"Seed: {data.get('seed')}")
            print(f"Timing breakdown: {data.get('timing')}")

            # Verify the result URL is downloadable
            result_url = data.get("result_url")
            full_url = f"http://127.0.0.1:7870{result_url}"
            with urllib.request.urlopen(full_url, timeout=10) as img_resp:
                img_data = img_resp.read()
                print(f"[OK] Result image downloaded, size: {len(img_data)} bytes")

            print("=== Live Inference Test PASSED! ===")
            return True
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        print(f"[FAIL] HTTP {e.code}: {err_body}")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_live_change_dress()
    sys.exit(0 if success else 1)
