#!/usr/bin/env python
# coding: utf-8

"""
Standalone movie evaluation script.
Evaluates a trained model on movies and logs results to MLflow.
"""

from pathlib import Path
import json
from typing import List, Dict

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import polars as pl
import numpy as np
import mlflow

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm.auto import tqdm


# -------------------------
# Model Loading
# -------------------------

def load_model_from_checkpoint(checkpoint_path: Path, device: str = "cuda"):
    """
    Load model from checkpoint file.
    
    Returns:
        model: Loaded model
        emotions: List of emotion labels
        transform: Transform to use for evaluation
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    emotions = checkpoint["emotions"]
    num_classes = len(emotions)
    
    # Get config if available (for new checkpoints)
    config = checkpoint.get("config", {})
    
    # Rebuild model
    model_arch = config.get("model", {}).get("architecture", "resnet18")
    
    if model_arch == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
    elif model_arch == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
    else:
        raise ValueError(f"Unknown architecture: {model_arch}")
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Create evaluation transform
    image_size = config.get("data", {}).get("image_size", 224)
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
    return model, emotions, eval_transform


# -------------------------
# Movie Processing
# -------------------------

def get_largest_detection(detections):
    """Get largest face from detections."""
    if not detections.detections:
        return None
    
    largest = None
    max_area = 0
    
    for detection in detections.detections:
        bbox = detection.bounding_box
        area = bbox.width * bbox.height
        if area > max_area:
            max_area = area
            largest = detection
    
    return largest


def crop_face(frame, bbox):
    """Crop face from frame using bounding box."""
    h, w = frame.shape[:2]
    x_min = int(bbox.origin_x)
    y_min = int(bbox.origin_y)
    x_max = int(bbox.origin_x + bbox.width)
    y_max = int(bbox.origin_y + bbox.height)
    
    # Clamp to frame boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)
    
    return frame[y_min:y_max, x_min:x_max]


def process_movie(
    movie_path: Path,
    model,
    emotions: List[str],
    eval_transform,
    face_detector,
    device: str,
    frame_stride: int = 100,
    min_face_confidence: float = 0.5,
) -> pl.DataFrame:
    """
    Process a movie and return emotion predictions for detected faces.
    """
    cap = cv2.VideoCapture(str(movie_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {movie_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    results = []
    frame_idx = 0
    
    pbar = tqdm(total=total_frames, desc=movie_path.stem, unit="frame")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        pbar.update(1)
        
        # Only process every Nth frame
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect faces
        detections = face_detector.detect(mp_image)
        
        # Get largest face
        detection = get_largest_detection(detections)
        
        if detection is not None and detection.categories[0].score >= min_face_confidence:
            bbox = detection.bounding_box
            
            # Crop face
            face_crop = crop_face(rgb_frame, bbox)
            
            if face_crop.size > 0:
                # Convert to PIL and apply transform
                face_pil = Image.fromarray(face_crop)
                face_tensor = eval_transform(face_pil).unsqueeze(0).to(device)
                
                # Predict emotion
                with torch.no_grad():
                    logits = model(face_tensor)
                    probs = torch.softmax(logits, dim=1)
                    pred = probs.argmax(dim=1).item()
                    confidence = probs[0, pred].item()
                
                # Calculate face statistics
                face_area = bbox.width * bbox.height
                frame_area = frame.shape[0] * frame.shape[1]
                relative_face_area = face_area / frame_area
                
                results.append({
                    "movie": movie_path.stem,
                    "frame_idx": frame_idx,
                    "timestamp_sec": frame_idx / fps,
                    "face_area": int(face_area),
                    "relative_face_area": relative_face_area,
                    "emotion": emotions[pred],
                    "confidence": confidence,
                })
        
        frame_idx += 1
    
    cap.release()
    pbar.close()
    
    return pl.DataFrame(results)


def evaluate_movies(
    movie_paths: List[Path],
    model,
    emotions: List[str],
    eval_transform,
    device: str,
    output_dir: Path,
    face_detector_path: Path,
    frame_stride: int = 100,
    min_face_confidence: float = 0.5,
) -> Dict:
    """
    Evaluate model on multiple movies.
    
    Returns summary statistics for MLflow logging.
    """
    # Initialize face detector
    print("Initializing face detector...")
    base_options = python.BaseOptions(model_asset_path=str(face_detector_path))
    face_options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=min_face_confidence
    )
    face_detector = vision.FaceDetector.create_from_options(face_options)
    
    # Process each movie
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    
    for movie_path in movie_paths:
        print(f"\nProcessing: {movie_path.name}")
        
        # Check if already processed
        out_path = output_dir / f"{movie_path.stem}.parquet"
        if out_path.exists():
            print(f"  Already processed, loading from {out_path}")
            df = pl.read_parquet(out_path)
        else:
            df = process_movie(
                movie_path,
                model,
                emotions,
                eval_transform,
                face_detector,
                device,
                frame_stride=frame_stride,
                min_face_confidence=min_face_confidence,
            )
            df.write_parquet(out_path)
            print(f"  Saved to {out_path}")
            
            # Print quick summary
            print(f"  Faces detected: {len(df)}")
            print(f"  Avg confidence: {df['confidence'].mean():.3f}")
            emotion_counts = df["emotion"].value_counts().sort("count", descending=True)
            print(f"  Top emotion: {emotion_counts[0, 'emotion']} ({emotion_counts[0, 'count']} faces)")
        
        all_results.append(df)
    
    # Combine all results
    print("\nCombining results...")
    combined_df = pl.concat(all_results)
    
    # Save combined results
    combined_path = output_dir / "combined_results.parquet"
    combined_df.write_parquet(combined_path)
    print(f"Saved combined results to {combined_path}")
    
    # Calculate summary statistics
    emotion_dist = combined_df["emotion"].value_counts().sort("count", descending=True)
    
    # Calculate dominant emotion per movie
    movie_dominant = (
        combined_df
        .group_by(["movie", "emotion"])
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .group_by("movie")
        .head(1)
    )
    
    dominant_emotion_counts = movie_dominant["emotion"].value_counts().sort("count", descending=True)
    
    # Print summary
    print(f"\n{'='*60}")
    print("MOVIE EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total faces detected: {len(combined_df)}")
    print(f"Movies processed: {len(movie_paths)}")
    print(f"Avg faces per movie: {len(combined_df) / len(movie_paths):.1f}")
    print(f"Avg confidence: {combined_df['confidence'].mean():.3f}")
    print()
    
    print("Emotion Distribution (all faces):")
    for row in emotion_dist.iter_rows(named=True):
        pct = row["count"] / len(combined_df) * 100
        print(f"  {row['emotion']:>10s}: {row['count']:5d} ({pct:5.1f}%)")
    print()
    
    print("Dominant Emotion Distribution (per movie):")
    for row in dominant_emotion_counts.iter_rows(named=True):
        pct = row["count"] / len(movie_paths) * 100
        print(f"  {row['emotion']:>10s}: {row['count']:3d} movies ({pct:5.1f}%)")
    print(f"{'='*60}\n")
    
    # Summary statistics
    summary = {
        "total_faces_detected": len(combined_df),
        "total_movies": len(movie_paths),
        "avg_faces_per_movie": len(combined_df) / len(movie_paths),
        "avg_confidence": combined_df["confidence"].mean(),
        "emotion_distribution": emotion_dist.to_dicts(),
        "dominant_emotion_distribution": dominant_emotion_counts.to_dicts(),
    }
    
    return summary, combined_df


# -------------------------
# Main Function
# -------------------------

def evaluate_model_on_movies(
    run_id: str,
    checkpoint_path: Path,
    movies_dir: Path,
    movie_list: List[str],
    output_dir: Path,
    face_detector_path: Path,
    frame_stride: int = 100,
    min_face_confidence: float = 0.5,
    device: str = "cuda",
):
    """
    Evaluate model on movies and log results to MLflow.
    """
    print(f"\n{'='*60}")
    print("MOVIE EVALUATION")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Movies directory: {movies_dir}")
    print(f"Number of movies: {len(movie_list)}")
    print(f"Output directory: {output_dir}")
    print(f"Frame stride: {frame_stride}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Load model
    print("Loading model...")
    model, emotions, eval_transform = load_model_from_checkpoint(checkpoint_path, device)
    print(f"Model loaded. Emotions: {emotions}\n")
    
    # Build movie paths
    movie_paths = [movies_dir / movie_name for movie_name in movie_list]
    
    # Check that movies exist
    missing_movies = [p for p in movie_paths if not p.exists()]
    if missing_movies:
        print("Warning: The following movies were not found:")
        for p in missing_movies:
            print(f"  - {p}")
        movie_paths = [p for p in movie_paths if p.exists()]
        print(f"Proceeding with {len(movie_paths)} movies\n")
    
    if not movie_paths:
        print("Error: No valid movie files found")
        return
    
    # Evaluate on movies
    summary, movie_df = evaluate_movies(
        movie_paths=movie_paths,
        model=model,
        emotions=emotions,
        eval_transform=eval_transform,
        device=device,
        output_dir=output_dir,
        face_detector_path=face_detector_path,
        frame_stride=frame_stride,
        min_face_confidence=min_face_confidence,
    )
    
    # # Log to MLflow
    # print("Logging results to MLflow...")
    # with mlflow.start_run(run_id=run_id):
    #     # Log movie evaluation metrics
    #     mlflow.log_metric("eval_movies_total_faces", summary["total_faces_detected"])
    #     mlflow.log_metric("eval_movies_avg_confidence", summary["avg_confidence"])
    #     mlflow.log_metric("eval_movies_count", summary["total_movies"])
    #     mlflow.log_metric("eval_movies_avg_per_movie", summary["avg_faces_per_movie"])
        
    #     # Log emotion distribution in movies
    #     for item in summary["emotion_distribution"]:
    #         emotion = item["emotion"]
    #         count = item["count"]
    #         pct = count / summary["total_faces_detected"] * 100
    #         mlflow.log_metric(f"eval_movies_{emotion}_count", count)
    #         mlflow.log_metric(f"eval_movies_{emotion}_pct", pct)
        
    #     # Log dominant emotion distribution
    #     for item in summary["dominant_emotion_distribution"]:
    #         emotion = item["emotion"]
    #         count = item["count"]
    #         pct = count / summary["total_movies"] * 100
    #         mlflow.log_metric(f"eval_movies_dominant_{emotion}_count", count)
    #         mlflow.log_metric(f"eval_movies_dominant_{emotion}_pct", pct)
        
    #     # Save combined movie results as artifact
    #     mlflow.log_artifact(str(output_dir / "combined_results.parquet"))
        
    #     # Save summary as JSON
    #     summary_path = output_dir / "movie_evaluation_summary.json"
    #     with open(summary_path, "w") as f:
    #         json.dump(summary, f, indent=2, default=str)
    #     mlflow.log_artifact(str(summary_path))
        
    #     # Tag as evaluated
    #     mlflow.set_tag("movie_evaluation_status", "complete")
    
    # print("\nMLflow logging complete!")
    # print(f"\n{'='*60}")
    # print("EVALUATION COMPLETE")
    # print(f"{'='*60}\n")


# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on movies"
    )
    parser.add_argument("--run-id", required=True, help="MLflow run ID")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--movies-dir", required=True, help="Directory containing movies")
    parser.add_argument("--movie-list", required=True, help="Path to file with movie filenames (one per line)")
    parser.add_argument("--output-dir", required=True, help="Directory to save results")
    parser.add_argument("--face-detector", required=True, help="Path to MediaPipe face detector")
    parser.add_argument("--frame-stride", type=int, default=100, help="Process every Nth frame (default: 100)")
    parser.add_argument("--min-face-confidence", type=float, default=0.5, help="Minimum face detection confidence (default: 0.5)")
    parser.add_argument("--device", default="cuda", help="Device to use (default: cuda)")
    
    args = parser.parse_args()
    
    # Load movie list
    with open(args.movie_list) as f:
        movie_list = [line.strip() for line in f if line.strip()]
    
    # Run evaluation
    evaluate_model_on_movies(
        run_id=args.run_id,
        checkpoint_path=Path(args.checkpoint),
        movies_dir=Path(args.movies_dir),
        movie_list=movie_list,
        output_dir=Path(args.output_dir),
        face_detector_path=Path(args.face_detector),
        frame_stride=args.frame_stride,
        min_face_confidence=args.min_face_confidence,
        device=args.device,
    )

