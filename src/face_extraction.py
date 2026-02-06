from pathlib import Path
import mediapipe as mp

import numpy as np
import polars as pl
from PIL import Image
import plotly.express as px


from utils.fv_utils import (
    ensure_dir, 
    list_images, 
    rel_id,
    get_detector
)

detector = get_detector()

# -----------------------------
# Core processing 
# -----------------------------

def process_image(image_path: Path, raw_root: Path, face_out_dir: Path):
    emotion = image_path.parts[-3]
    mp_image = mp.Image.create_from_file(str(image_path))
    img = mp_image.numpy_view()

    H, W = mp_image.height, mp_image.width
    rel_image_id = rel_id(image_path, raw_root)

    if img.shape[2] !=3:
        print(f'Image path: {image_path}')
        print(f'Bad image dimensions: {img.shape}')
        detections = []
    else:
        result = detector.detect(mp_image)
        detections = result.detections or []

    image_record = {
        "image_id": rel_image_id,
        "image_path": str(image_path),
        "raw_root": raw_root.name,
        "emotion": emotion,
        "width": W,
        "height": H,
        "channels": mp_image.channels,
        "image_format": mp_image.image_format,
        "dtype": str(img.dtype),
        "n_faces": len(detections),
    }

    face_records = []

    for i, det in enumerate(detections):
        bbox = det.bounding_box

        x1 = max(0, bbox.origin_x)
        y1 = max(0, bbox.origin_y)
        x2 = min(W, x1 + bbox.width)
        y2 = min(H, y1 + bbox.height)

        if x2 <= x1 or y2 <= y1:
            continue

        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        face_h, face_w = face_crop.shape[:2]
        face_area = face_h * face_w

        face_name = f"{rel_image_id}_face_{i}.jpg"
        face_path = face_out_dir / face_name
        ensure_dir(face_path.parent)

        Image.fromarray(face_crop).save(face_path)

        face_records.append({
            "image_id": rel_image_id,
            "face_index": i,
            "face_path": str(face_path),
            "emotion": emotion,
            "bbox_x": x1,
            "bbox_y": y1,
            "bbox_w": bbox.width,
            "bbox_h": bbox.height,
            "score": det.categories[0].score,
            "face_width": face_w,
            "face_height": face_h,
            "face_area": face_area,
        })

    return image_record, face_records

# -----------------------------
# EDA post processing 
# -----------------------------

def image_eda(images_df: pl.DataFrame, faces_df: pl.DataFrame, eda_dir: Path):
    by_emotion = (
        images_df
        .group_by("emotion")
        .agg(
            n_images=pl.len(),
            n_with_face=(pl.col("n_faces") > 0).sum(),
        )
        .with_columns(
            pct_with_face=pl.col("n_with_face") / pl.col("n_images")
        )
        .sort("emotion")
    )

    fig = px.bar(
        by_emotion.to_pandas(),
        x="emotion",
        y="n_images",
        title="Images per Emotion",
        text="n_images",
    )
    fig.write_html(eda_dir / "image_counts.html")

    fig = px.bar(
        by_emotion.to_pandas(),
        x="emotion",
        y="pct_with_face",
        title="Proportion of Images with ≥1 Face",
    )
    fig.write_html(eda_dir / "face_yield.html")

    primary_faces = (
            faces_df
            .sort("face_area", descending=True)
            .group_by("image_id")
            .first()
        )

    images_with_faces = images_df.join(
        primary_faces,
        on="image_id",
        how="left"
    )

    images_with_faces = images_with_faces.with_columns(
        relative_face_area=(
            pl.col("face_area") /
            (pl.col("width") * pl.col("height"))
        )
    )

    fig = px.box(
        images_with_faces.drop_nulls("face_area").to_pandas(),
        x="emotion",
        y="face_area",
        title="Largest Face Area per Image",
    )
    fig.write_html(eda_dir / "face_area_box.html")

    fig = px.box(
        images_with_faces.drop_nulls("relative_face_area").to_pandas(),
        x="emotion",
        y="relative_face_area",
        title="Relative Face Area per Image",
    )
    fig.write_html(eda_dir / "face_area_relative_box.html")

    return images_with_faces

# -----------------------------
# Run-level pipeline
# -----------------------------

def run_extraction(raw_dir: Path, out_dir: Path, force: bool = False):
    summaries_dir = out_dir / "summaries"
    faces_dir = out_dir / "faces"
    eda_dir = summaries_dir / "eda"

    image_summary_path = summaries_dir / "image_level.parquet"
    face_summary_path = summaries_dir / "face_level.parquet"
    aggregate_path = summaries_dir / "aggregate_stats.parquet"
    training_path = summaries_dir / "training_data.parquet"

    # ---- DO NOT RUN AGAIN GUARD ----
    if aggregate_path.exists() and not force:
        print(
            f"[SKIP] Existing summary found:\n"
            f"  {aggregate_path}\n"
            f"Use --force to rerun intentionally."
        )
        return

    ensure_dir(summaries_dir)
    ensure_dir(faces_dir)
    ensure_dir(eda_dir)

    image_records = []
    face_records = []

    image_paths = list_images(raw_dir)
    print(f"Processing {len(image_paths)} images from {raw_dir.name}")

    for img_path in image_paths:
        try:
            img_rec, face_recs = process_image(
                img_path,
                raw_root=raw_dir,
                face_out_dir=faces_dir,
            )
            image_records.append(img_rec)
            face_records.extend(face_recs)

        except Exception as e:
            image_records.append({
                "image_id": rel_id(img_path, raw_dir),
                "image_path": str(img_path),
                "raw_root": raw_dir.name,
                "error": str(e),
                "n_faces": None,
            })

    img_df = pl.DataFrame(image_records)
    face_df = pl.DataFrame(face_records)

    img_df.write_parquet(image_summary_path)
    face_df.write_parquet(face_summary_path)

    # -----------------------------
    # Aggregate stats (EDA-ready)
    # -----------------------------

    agg = (
        img_df
        .group_by("raw_root")
        .agg([
            pl.count().alias("n_images"),
            pl.col("n_faces").sum().alias("n_faces_total"),
            (pl.col("n_faces") > 0).sum().alias("n_images_with_face"),
        ])
        .with_columns([
            (pl.col("n_images_with_face") / pl.col("n_images"))
            .alias("pct_images_with_face")
        ])
    )

    agg.write_parquet(aggregate_path)

    print("Aggredation complete")
    print(agg)

    training = image_eda(images_df=img_df, faces_df=face_df, eda_dir=eda_dir)
    training.write_parquet(training_path)

    print("EDA complete")
    print("Extraction done")

# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    run_extraction(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        force=args.force,
    )
