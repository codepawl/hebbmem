"""Download LoCoMo dataset if not present."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)
CACHE_DIR = Path(__file__).parent / "data"
CACHE_FILE = CACHE_DIR / "locomo10.json"


def download_locomo() -> Path:
    """Download LoCoMo dataset, return path to cached file."""
    if CACHE_FILE.exists():
        return CACHE_FILE
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading LoCoMo dataset to {CACHE_FILE}...")
    urllib.request.urlretrieve(LOCOMO_URL, CACHE_FILE)
    print("Done.")
    return CACHE_FILE


def load_locomo() -> list[dict]:
    """Load LoCoMo dataset as raw JSON."""
    path = download_locomo()
    with open(path) as f:
        return json.load(f)
