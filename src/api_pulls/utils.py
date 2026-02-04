import json
import requests
import polars as pl
from pathlib import Path

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_seen_ids(path: Path) -> set:
    if not path.exists():
        return set()
    df = pl.read_parquet(path)
    return set(df["source_image_id"].to_list())


def save_seen_ids(path: Path, ids: set):
    pl.DataFrame({"source_image_id": list(ids)}).write_parquet(path)


def download_image(url: str, out_path: Path) -> bool:
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            out_path.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False