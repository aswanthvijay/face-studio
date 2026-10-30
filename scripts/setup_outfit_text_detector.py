"""Install the pinned float PP-OCRv3 detector; runtime inference is offline."""
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]


def install():
    folder = ROOT / 'models' / 'text'
    for asset in json.loads((folder / 'manifest.json').read_text()):
        license_path = folder / asset['license_file']
        if not license_path.is_file() or hashlib.sha256(license_path.read_bytes()).hexdigest() != asset['license_sha256']:
            raise RuntimeError('Restore the packaged Apache-2.0 LICENSE before installing the detector.')
        dest = folder / asset['file']
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == asset['sha256']:
            print('Verified', dest.name)
            continue
        response = requests.get(asset['url'], timeout=90)
        response.raise_for_status()
        data = response.content
        if len(data) != asset['bytes'] or hashlib.sha256(data).hexdigest() != asset['sha256']:
            raise RuntimeError('Checksum mismatch: ' + dest.name)
        temporary = dest.with_suffix('.part')
        temporary.write_bytes(data)
        temporary.replace(dest)
        print('Installed', dest.name)


if __name__ == '__main__':
    install()
