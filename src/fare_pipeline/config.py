"""Load project settings from params.yaml."""
import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_params(path="params.yaml"):
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def sha256_of(path):
    """SHA-256 of a file's raw bytes (used as the `data_sha256` MLflow param later)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
