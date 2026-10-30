"""Local qualification for the pinned photo. A rejection is NOT visual acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
from core import config
from core.automatic_outfit import run_automatic_outfit, OutfitRejected
from core.outfit_validation import preservation_checks


def reference_masks(source):
    folder = ROOT/'tests/fixtures/outfit_596663721'
    spec = json.loads((folder/'reference_annotations.json').read_text())
    assert list(source.size) == spec['size']
    masks = {}
    for name, polygons in spec['cores'].items():
        mask = np.zeros((source.height,source.width),np.uint8)
        cv2.fillPoly(mask,[np.array(p,np.int32) for p in polygons],1)
        selected = mask > 0
        if name in spec['bright_pixel_cores']:
            selected &= np.min(np.asarray(source),axis=2) > 185
        masks[name] = selected
    mask = np.zeros((source.height,source.width),np.uint8)
    cv2.fillPoly(mask,[np.array(spec['approved_outline'],np.int32)],1)
    masks['outside_reference_edit_union'] = mask == 0
    return masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence-only',action='store_true')
    parser.add_argument('--diagnostic-stages',action='store_true',
        help='Exercise GPU stages despite preflight rejection; never publish any candidate.')
    args = parser.parse_args()
    folder = ROOT/'tests/fixtures/outfit_596663721'
    manifest = json.loads((folder/'manifest.json').read_text())
    for filename, checksum in manifest['sha256'].items():
        assert hashlib.sha256((folder/filename).read_bytes()).hexdigest() == checksum, filename
    source = Image.open(folder/'source.png').convert('RGB')
    failed = Image.open(folder/'failed.png').convert('RGB')
    masks = reference_masks(source)
    changed = np.any(np.asarray(source) != np.asarray(failed),axis=2)
    evidence = {name:{'pixels':int(mask.sum()),'changed':int((mask&changed).sum())} for name,mask in masks.items()}
    (folder/'failed_reference_checks.json').write_text(json.dumps(evidence,indent=2))
    # Annotation overlay is for review, never fed into region discovery.
    overlay = np.asarray(source).copy()
    for name,mask in masks.items():
        if name != 'outside_reference_edit_union':
            contours,_ = cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay,contours,-1,(255,40,40),1)
    Image.fromarray(overlay).save(folder/'reference_overlay.png')
    print(json.dumps(evidence,indent=2),flush=True)
    if args.evidence_only:
        return
    try:
        result = run_automatic_outfit(source,manifest['prompt'],config.MODELS['cyberrealistic_xl'],
            manifest['seed'],progress_cb=lambda message: print(message,flush=True), reference_cores=masks,
            diagnostic_only=args.diagnostic_stages)
        report = {'automatic_result':result,'visual_acceptance':'pending'}
    except OutfitRejected as exc:
        report = {'automatic_result':exc.quality,'visual_acceptance':'failed'}
    report_name = 'diagnostic_stages.json' if args.diagnostic_stages else 'qualification.json'
    (folder/report_name).write_text(json.dumps(report,indent=2))
    print(json.dumps({'report':str(folder/report_name),'visual_acceptance':report['visual_acceptance'],
                     'failure_reasons':report['automatic_result'].get('failure_reasons',[])},indent=2),flush=True)


if __name__ == '__main__':
    main()
