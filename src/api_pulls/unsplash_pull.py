import os
import time
import json
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

    access_key = os.getenv("UNSPLASH_ACCESS_KEY")
    if not access_key:
        raise RuntimeError("Missing UNSPLASH_ACCESS_KEY environment variable")

    search_cfg = cfg["search"]
    output_cfg = cfg["output"]
    stop_cfg = cfg["early_stop"]

    base_image_dir = Path(output_cfg["raw_image_dir"])
    ensure_dir(base_image_dir)

    seen_ids_path = base_image_dir /  output_cfg["seen_ids_filename"]
    seen_ids = load_seen_ids(seen_ids_path)

    manifest_records = []
    run_records = []

    request_count = 0

    headers = {
        "Authorization": f"Client-ID {access_key}"
    }

    for emotion, keywords in search_cfg["emotions"].items():
        for keyword_base in keywords:

            query = search_cfg["keyword_template"].format(
                emotion=keyword_base
            )

            emotion_dir = base_image_dir / emotion / keyword_base
            ensure_dir(emotion_dir)

            print(f"\n=== Emotion: {emotion} | Query: '{query}' ===")

            max_pages_available = None

            for page in range(1, search_cfg["max_pages"] + 1):
                params = {
                    "query": query,
                    "page": page,
                    "per_page": search_cfg["per_page"],
                    "order_by": search_cfg.get("order_by", "relevant"),
                }

                response = requests.get(
                    search_cfg["api_url"],
                    headers=headers,
                    params=params,
                    timeout=30,
                )

                request_count += 1

                try:
                    response.raise_for_status()
                except requests.HTTPError as e:
                    run_records.append({
                        "source": "unsplash",
                        "emotion": emotion,
                        "keyword": keyword_base,
                        "query": query,
                        "page": page,
                        "error_type": "http_error",
                        "status_code": response.status_code,
                        "error_message": str(e),
                        "request_index": request_count,
                        "timestamp": datetime.utcnow().isoformat(),
                    })

                    if response.status_code == 429:
                        print("429 Rate limit hit. Exiting cleanly.")
                        save_seen_ids(seen_ids_path, seen_ids)
                        return

                    print(f"HTTP error {response.status_code}. Skipping keyword.")
                    break

                data = response.json()
                hits = data.get("results", [])
                total_hits = data.get("total", 0)

                if max_pages_available is None:
                    max_pages_available = max(
                        1,
                        (total_hits + search_cfg["per_page"] - 1)
                        // search_cfg["per_page"],
                    )

                if page > max_pages_available:
                    print("No more valid pages available; stopping pagination.")
                    break

                if not hits:
                    print("No results returned; stopping pagination.")
                    break

                new_count = 0

                for hit in hits:
                    pid = f"unsplash:{hit['id']}"
                    if pid in seen_ids:
                        continue

                    seen_ids.add(pid)
                    new_count += 1

                    img_url = hit["urls"].get("full") or hit["urls"].get("regular")
                    img_path = emotion_dir / f"{pid.replace(':', '_')}.jpg"

                    downloaded = False
                    if img_url:
                        downloaded = download_image(img_url, img_path)

                    manifest_records.append({
                        "source": "unsplash",
                        "emotion": emotion,
                        "keyword": keyword_base,
                        "query": query,
                        "source_image_id": pid,
                        "image_path": str(img_path),
                        "url": img_url,
                        "width": hit.get("width"),
                        "height": hit.get("height"),
                        "user": hit["user"].get("username"),
                        "page": page,
                        "downloaded": downloaded,
                        "retrieved_at": datetime.now(),
                    })

                novelty_fraction = new_count / len(hits)

                run_records.append({
                    "source": "unsplash",
                    "emotion": emotion,
                    "keyword": keyword_base,
                    "query": query,
                    "page": page,
                    "n_returned": len(hits),
                    "n_new": new_count,
                    "novelty_fraction": novelty_fraction,
                    "request_index": request_count,
                    "timestamp": datetime.now(),
                })

                print(
                    f"Page {page}: {new_count}/{len(hits)} new "
                    f"(novelty={novelty_fraction:.2f})"
                )

                if novelty_fraction < stop_cfg["min_new_fraction"]:
                    print("Novelty below threshold; stopping pagination.")
                    break

                time.sleep(0.25)

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
        raise RuntimeError("Usage: python unsplash_pull.py <config.json>")
    main(sys.argv[1])
