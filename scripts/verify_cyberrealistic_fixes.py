"""Exercise the real API on this copy's ports and save reviewable render evidence."""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import config
from core.clothing_engine import generate_dress_mask, normalize_input_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=f"http://127.0.0.1:{config.PORT}")
    parser.add_argument("--source", type=Path, default=config.SAMPLES_DIR / "sample_model.png")
    parser.add_argument("--case", choices=("in_place_peach_turtleneck", "full_scene_black_turtleneck"))
    parser.add_argument("--variations", type=int, default=2)
    args = parser.parse_args()
    session = requests.Session()
    session.trust_env = False
    report_dir = config.OUTPUTS_DIR / "cyberrealistic_verification"
    report_dir.mkdir(parents=True, exist_ok=True)
    status = session.get(args.base_url + "/api/status", timeout=10)
    status.raise_for_status()
    assert status.json()["ready"], "Image backend is not ready"
    report_path = report_dir / "report.json"
    report = {"studio_url": args.base_url, "backend_url": config.BACKEND_URL, "cases": []}
    if args.case and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    for request in (
        {"id": "in_place_peach_turtleneck", "mode": "in_place_swap", "variations": 2,
         "prompt": "peach colored high collar turtleneck dress, opaque peach fabric, fitted long sleeves, realistic knit texture"},
        {"id": "full_scene_black_turtleneck", "mode": "full_scene", "variations": 2,
         "prompt": "woman wearing a black high collar turtleneck dress, waist up portrait in a photography studio, natural soft lighting"},
    ):
        if args.case and args.case != request["id"]:
            continue
        request["variations"] = args.variations
        report["cases"] = [case for case in report["cases"] if case["id"] != request["id"]]
        print(f"Starting {request['id']} ({request['variations']} variations)", flush=True)
        start = time.monotonic()
        payload = {key: value for key, value in request.items() if key != "id"}
        payload.update({"image_id": "sample_model", "model_id": "cyberrealistic_xl", "seed": 7041,
                        "framing": "portrait", "aspect_ratio": "portrait", "enable_face_id": True})
        # The sample ID resolves in this workspace; a custom source uses upload.
        if args.source != config.SAMPLES_DIR / "sample_model.png":
            with args.source.open("rb") as source_file:
                uploaded = session.post(args.base_url + "/api/upload", files={"image": source_file}, timeout=180)
            uploaded.raise_for_status()
            payload["image_id"] = uploaded.json()["image_id"]
        response = session.post(args.base_url + "/api/change-dress", json=payload, timeout=1200)
        case = {"id": request["id"], "http_status": response.status_code,
                "elapsed_seconds": round(time.monotonic() - start, 2), "response": response.json()}
        report["cases"].append(case)
        (report_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        response.raise_for_status()
        data = case["response"]
        assert len(data["results"]) == request["variations"], data.get("warnings")
        if request["mode"] == "full_scene":
            assert data["face_swapped"], "Primary result did not transfer identity"
        source = normalize_input_image(Image.open(args.source))
        if request["mode"] == "in_place_swap":
            mask, _, _, _ = generate_dress_mask(source, target="dress", prompt=request["prompt"],
                                                keep_face=True, expand_pixels=14, feather_sigma=3.5)
            mask.save(report_dir / "in_place_edit_mask.png")
            outside = np.array(mask) == 0
            case["outside_mask_changed_pixels"] = []
        for index, url in enumerate(data["results"]):
            image_response = session.get(args.base_url + url, timeout=30)
            image_response.raise_for_status()
            with Image.open(io.BytesIO(image_response.content)) as generated:
                generated = generated.convert("RGB")
                generated.save(report_dir / f"{request['id']}_{index + 1}.png")
                if request["mode"] == "in_place_swap":
                    changed = np.any(np.array(generated) != np.array(source), axis=2)
                    count = int(np.count_nonzero(changed & outside))
                    case["outside_mask_changed_pixels"].append(count)
                    assert count == 0, f"{count} protected pixels changed"
        (report_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Passed {request['id']} in {case['elapsed_seconds']}s", flush=True)
    print(f"Verification report: {report_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
