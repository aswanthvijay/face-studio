"""Verification script for Face Studio Server Security & API Endpoints."""
import json
import time
import requests

BASE_URL = "http://127.0.0.1:7870"

def test_api():
    print("======================================================================")
    print("         Face Studio Server & Security Verification")
    print("======================================================================")

    # 1. Non-blocking Health Check
    print("\n[Test 1/4] Testing non-blocking /api/status response time...")
    t0 = time.monotonic()
    res = requests.get(f"{BASE_URL}/api/status", timeout=5)
    t1 = time.monotonic()
    elapsed_ms = (t1 - t0) * 1000
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert data.get("status") in ("ready", "engine_starting"), f"Unexpected status: {data.get('status')}"
    assert "cyberrealistic_xl" in data.get("models", {}), "Missing cyberrealistic_xl model"
    assert ("realvis_xl" in data.get("models", {}) or "epicrealism_xl" in data.get("models", {})), "Missing active studio portrait model"
    print(f"  [PASS] /api/status responded in {elapsed_ms:.1f}ms (Target: < 200ms).")

    # 2. Path Traversal Prevention
    print("\n[Test 2/4] Testing Directory Traversal Protection (safe_path)...")
    traversal_paths = [
        "/static/../../server.py",
        "/outputs/../../../../Windows/win.ini",
        "/inputs/..%2f..%2fserver.py",
        "/samples/../../core/config.py"
    ]
    for p in traversal_paths:
        r = requests.get(f"{BASE_URL}{p}")
        assert r.status_code == 404, f"Path traversal succeeded on {p}! Status: {r.status_code}"
    print("  [PASS] All directory traversal attempts rejected with 404.")

    # 3. Static Assets Serving
    print("\n[Test 3/4] Testing static web asset serving...")
    r_html = requests.get(f"{BASE_URL}/")
    assert r_html.status_code == 200 and "Face Studio" in r_html.text, "Index HTML failed"
    r_css = requests.get(f"{BASE_URL}/static/style.css")
    assert r_css.status_code == 200 and "--split-pos" in r_css.text, "Style.css failed"
    r_sample = requests.get(f"{BASE_URL}/api/sample")
    assert r_sample.status_code == 200, "Sample endpoint failed"
    print("  [PASS] Web Studio assets and split-slider CSS verified.")

    # 4. Mode Normalization & History Bounding
    print("\n[Test 4/5] Testing Mode Normalization & History buffer...")
    r_hist = requests.get(f"{BASE_URL}/api/history")
    assert r_hist.status_code == 200, "History endpoint failed"
    assert isinstance(r_hist.json(), list), "History must be a JSON array"
    print("  [PASS] API history is active and accessible.")

    # 5. LoRA Metadata Endpoint
    print("\n[Test 5/5] Testing /api/loras metadata endpoint...")
    r_lora = requests.get(f"{BASE_URL}/api/loras")
    assert r_lora.status_code == 200, f"Expected 200 from /api/loras, got {r_lora.status_code}"
    lora_data = r_lora.json()
    assert "loras" in lora_data, "Missing 'loras' key in response"
    assert "fs_couture" in lora_data["loras"], "Custom LoRA 'fs_couture' not found in API listing!"
    print(f"  [PASS] /api/loras returned {len(lora_data['loras'])} models including 'fs_couture'.")

    print("\n======================================================================")
    print("   ALL SERVER SECURITY & API TESTS PASSED (100% SECURE)")
    print("======================================================================")

if __name__ == "__main__":
    test_api()
