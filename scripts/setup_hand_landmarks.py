"""Explicitly restore the two optional hand models from the checked manifest."""
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT/'models'/'hands'
for asset in json.loads((folder/'manifest.json').read_text()):
    destination = folder/asset['file']
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == asset['sha256']:
        print(f"Verified {destination.name}")
        continue
    response = requests.get(asset['url'], timeout=120)
    response.raise_for_status()
    data = response.content
    if len(data) != asset['bytes'] or hashlib.sha256(data).hexdigest() != asset['sha256']:
        raise RuntimeError(f"Checksum mismatch for {destination.name}")
    temporary = destination.with_suffix('.part')
    temporary.write_bytes(data)
    temporary.replace(destination)
    print(f"Installed {destination.name}")
