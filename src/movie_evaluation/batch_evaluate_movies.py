#!/usr/bin/env python
# coding: utf-8

"""
Batch evaluate all trained models on movies.
Useful for running overnight.
"""

import mlflow
from pathlib import Path
import subprocess
import time


def get_runs_to_evaluate(experiment_name: str, only_unevaluated: bool = True):
    """
    Get list of runs from experiment that need movie evaluation.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    # Search for runs
    filter_string = None
    if only_unevaluated:
        filter_string = "tags.movie_evaluation_status != 'complete'"
    
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=["start_time DESC"]
    )
    
    if len(runs_df) == 0:
        print("No runs found to evaluate")
        return []
    
    runs_info = []
    client = mlflow.tracking.MlflowClient()
    
    for _, row in runs_df.iterrows():
        run_id = row["run_id"]
        run_name = row.get("tags.mlflow.runName", "unknown")
        
        # Get checkpoint path from artifacts
        artifacts = client.list_artifacts(run_id)
        
        # Find model.pt artifact
        checkpoint_path = None
        for artifact in artifacts:
            if artifact.path == "model.pt":
                # Download artifact to local path
                local_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path="model.pt"
                )
                checkpoint_path = local_path
                break
        
        if checkpoint_path:
            runs_info.append({
                "run_id": run_id,
                "run_name": run_name,
                "checkpoint_path": checkpoint_path,
            })
        else:
            print(f"⚠️  Warning: No checkpoint found for run {run_name} ({run_id})")
    
    return runs_info


def evaluate_run(
    run_info: dict,
    movies_dir: Path,
    movie_list_path: Path,
    output_base_dir: Path,
    face_detector_path: Path,
    frame_stride: int = 100,
    device: str = "cuda",
    dry_run: bool = False,
):
    """
    Evaluate a single run on movies.
    """
    output_dir = output_base_dir / run_info["run_name"]
    
    cmd = [
        "python", "evaluate_movies.py",
        "--run-id", run_info["run_id"],
        "--checkpoint", run_info["checkpoint_path"],
        "--movies-dir", str(movies_dir),
        "--movie-list", str(movie_list_path),
        "--output-dir", str(output_dir),
        "--face-detector", str(face_detector_path),
        "--frame-stride", str(frame_stride),
        "--device", device,
    ]
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {run_info['run_name']}")
    print(f"Run ID: {run_info['run_id']}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    if dry_run:
        print("[DRY RUN] Would execute:")
        print(" ".join(cmd))
        return True
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch evaluate all models on movies"
    )
    parser.add_argument(
        "--experiment",
        default="emotion-movie-transfer",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--movies-dir",
        required=True,
        help="Directory containing movies"
    )
    parser.add_argument(
        "--movie-list",
        required=True,
        help="Path to file with movie filenames (one per line)"
    )
    parser.add_argument(
        "--output-base",
        required=True,
        help="Base directory for movie evaluation results"
    )
    parser.add_argument(
        "--face-detector",
        required=True,
        help="Path to MediaPipe face detector model"
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=100,
        help="Process every Nth frame (default: 100)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--only-unevaluated",
        action="store_true",
        help="Only evaluate runs without movie_evaluation_status=complete tag"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop if any evaluation fails"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    movies_dir = Path(args.movies_dir)
    if not movies_dir.exists():
        print(f"Error: Movies directory not found: {movies_dir}")
        return
    
    movie_list_path = Path(args.movie_list)
    if not movie_list_path.exists():
        print(f"Error: Movie list file not found: {movie_list_path}")
        return
    
    face_detector_path = Path(args.face_detector)
    if not face_detector_path.exists():
        print(f"Error: Face detector not found: {face_detector_path}")
        return
    
    # Load movie list
    with open(movie_list_path) as f:
        movie_list = [line.strip() for line in f if line.strip()]
    
    print(f"Movies to process: {len(movie_list)}")
    print(f"Frame stride: {args.frame_stride} (every {args.frame_stride}th frame)")
    print()
    
    # Get runs to evaluate
    print(f"Searching for runs in experiment: {args.experiment}")
    runs = get_runs_to_evaluate(args.experiment, only_unevaluated=args.only_unevaluated)
    
    if not runs:
        print("No runs to evaluate")
        return
    
    print(f"\nFound {len(runs)} runs to evaluate:")
    for i, run in enumerate(runs, 1):
        print(f"{i}. {run['run_name']} ({run['run_id'][:8]}...)")
    print()
    
    # Estimate time
    # Rough estimate: ~5 minutes per movie per model (with frame_stride=100)
    estimated_minutes = len(runs) * len(movie_list) * 5
    print(f"Estimated time: {estimated_minutes/60:.1f} hours")
    print(f"  ({len(runs)} models × {len(movie_list)} movies × ~5 min/movie)")
    print()
    
    if not args.dry_run:
        response = input(f"Evaluate {len(runs)} runs on {len(movie_list)} movies? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled")
            return
    
    # Evaluate each run
    results = []
    overall_start = time.time()
    
    for i, run in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}]")
        success = evaluate_run(
            run,
            movies_dir=movies_dir,
            movie_list_path=movie_list_path,
            output_base_dir=Path(args.output_base),
            face_detector_path=face_detector_path,
            frame_stride=args.frame_stride,
            device=args.device,
            dry_run=args.dry_run,
        )
        
        results.append({
            "run_name": run["run_name"],
            "success": success,
        })
        
        if not success and args.stop_on_error:
            print("\n⚠️  Stopping due to error")
            break
    
    overall_elapsed = time.time() - overall_start
    
    # Print summary
    print(f"\n{'='*60}")
    print("MOVIE EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {overall_elapsed/3600:.1f} hours")
    print(f"Completed: {sum(r['success'] for r in results)}/{len(results)}")
    print()
    
    for result in results:
        status = "✓" if result["success"] else "✗"
        print(f"{status} {result['run_name']}")
    
    print(f"{'='*60}\n")
    
    print("To view results:")
    print("  mlflow ui")
    print("  # Then navigate to experiment and sort by 'eval_movies_dominant_fear_pct'")


if __name__ == "__main__":
    main()
