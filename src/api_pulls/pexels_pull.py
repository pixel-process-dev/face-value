import os
import time
import requests
from pathlib import Path
from datetime import datetime

import polars as pl

from utils import (
    load_config,
    ensure_dir,
    load_seen_ids,
    save_seen_ids,
    download_image
)

def main(config_path: str):
    cfg = load_config(config_path)

    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        raise RuntimeError(f"Missing API key")

    headers = {"Authorization": api_key}

    search_cfg = cfg["search"]
    output_cfg = cfg["output"]
    rate_cfg = cfg["rate_limit"]
    stop_cfg = cfg["early_stop"]

    base_image_dir = Path(output_cfg["raw_image_dir"])
    ensure_dir(base_image_dir)

    seen_ids_path = base_image_dir /  output_cfg["seen_ids_filename"]
    seen_ids = load_seen_ids(seen_ids_path)

    manifest_records = []
    run_records = []

    request_count = 0

    for emotion, search_terms in search_cfg["emotions"].items():
        for search_term in search_terms:

            keyword = search_cfg["keyword_template"].format(keyword=search_term)
            emotion_dir = base_image_dir / emotion / search_term
            ensure_dir(emotion_dir)

            print(f"\n=== Emotion: {emotion} | Query: '{keyword}' ===")

            for page in range(1, search_cfg["max_pages"] + 1):
                params = {
                    "query": keyword,
                    "per_page": search_cfg["per_page"],
                    "page": page,
                }

                response = requests.get(
                    search_cfg["api_url"],
                    headers=headers,
                    params=params,
                    timeout=30,
                )

                request_count += 1

                if response.status_code == 429:
                    print("Rate limit hit (429). Exiting cleanly.")
                    save_seen_ids(seen_ids_path, seen_ids)
                    return

                response.raise_for_status()
                data = response.json()

                photos = data.get("photos", [])
                if not photos:
                    print("No results returned; stopping pagination.")
                    break

                new_count = 0

                for photo in photos:
                    pid = str(photo["id"])
                    if pid in seen_ids:
                        continue

                    seen_ids.add(pid)
                    new_count += 1

                    img_url = photo["src"]["original"]
                    img_path = emotion_dir / f"pexels_{pid}.jpg"

                    downloaded = download_image(img_url, img_path)

                    manifest_records.append({
                        "source": "pexels",
                        "emotion": emotion,
                        "query": keyword,
                        "source_image_id": pid,
                        "image_path": str(img_path),
                        "url": img_url,
                        "width": photo["width"],
                        "height": photo["height"],
                        "photographer": photo.get("photographer"),
                        "downloaded": downloaded,
                        "page": page,
                        "retrieved_at": datetime.now(),
                    })

                novelty_fraction = new_count / len(photos)

                run_records.append({
                    "emotion": emotion,
                    "query": keyword,
                    "page": page,
                    "n_returned": len(photos),
                    "n_new": new_count,
                    "novelty_fraction": novelty_fraction,
                    "request_index": request_count,
                    "rate_limit_remaining": response.headers.get("X-Ratelimit-Remaining"),
                    "timestamp": datetime.now(),
                })

                print(
                    f"Page {page}: {new_count}/{len(photos)} new "
                    f"(novelty={novelty_fraction:.2f})"
                )

                if novelty_fraction < stop_cfg["min_new_fraction"]:
                    print("Novelty below threshold; stopping pagination for this emotion.")
                    break

                remaining = response.headers.get("X-Ratelimit-Remaining")
                if remaining is not None and int(remaining) < rate_cfg["min_remaining_requests"]:
                    print("Low remaining quota; sleeping briefly.")
                    time.sleep(rate_cfg["sleep_seconds_on_low_remaining"])

    # Save outputs
    if manifest_records:
        manifest_path = base_image_dir / output_cfg["manifest_filename"]
        pl.DataFrame(manifest_records).write_parquet(
            manifest_path
        )

    if run_records:
        run_log_path = base_image_dir / output_cfg["run_log_filename"]
        pl.DataFrame(run_records).write_parquet(
            run_log_path
        )

    save_seen_ids(seen_ids_path, seen_ids)

    print("\nIngestion + download complete.")
    print(f"Total API requests used: {request_count}")
    print(f"Total unique images seen: {len(seen_ids)}")
    

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: python pexels_pull.py <config.json>")
    main(sys.argv[1])
