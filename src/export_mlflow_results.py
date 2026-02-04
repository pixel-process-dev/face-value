"""
Code snippets to export MLflow results for sharing/analysis.

Run these in a Jupyter notebook or Python script to extract your 9 runs.
"""

import mlflow
import polars as pl
import json


# ====================
# OPTION 1: Quick Summary Table (RECOMMENDED FOR INITIAL SHARING)
# ====================

def export_runs_summary():
    """
    Export key metrics from all runs to a simple table.
    Best for initial discussion and comparison.
    """
    runs = mlflow.search_runs(
        experiment_names=["emotion-movie-transfer"],
        order_by=["start_time DESC"]
    )
    
    # Select relevant columns
    columns = [
        # Identifiers
        "run_id",
        "tags.run_name",
        "tags.data_source",
        "tags.mlflow.runName",
        
        # Training config
        "params.train_epochs",
        "params.train_learning_rate",
        "params.data_augmentation",
        "params.train_samples",
        "params.val_samples",
        
        # Training performance
        "metrics.val_acc",
        "metrics.best_val_acc",
        "metrics.recall_angry",
        "metrics.recall_fear",
        "metrics.recall_happy",
        "metrics.recall_sad",
        "metrics.recall_surprise",
        
        # Movie evaluation (if completed)
        "metrics.eval_movies_total_faces",
        "metrics.eval_movies_dominant_sad_pct",
        "metrics.eval_movies_dominant_fear_pct",
        "metrics.eval_movies_dominant_happy_pct",
        "metrics.eval_movies_sad_pct",
        "metrics.eval_movies_fear_pct",
        "metrics.eval_movies_happy_pct",
    ]
    
    # Filter to only available columns
    available_cols = [col for col in columns if col in runs.columns]
    summary = runs[available_cols].copy()
    
    # Clean up column names
    summary.columns = [
        col.replace("metrics.", "").replace("params.", "").replace("tags.", "")
        for col in summary.columns
    ]
    
    # Sort by validation accuracy
    summary = summary.sort_values("val_acc", ascending=False)
    
    # Save to CSV
    summary.to_csv("mlflow_runs_summary.csv", index=False)
    print("Saved to: mlflow_runs_summary.csv")
    
    # Also print to console
    print("\n" + "="*100)
    print("RUNS SUMMARY")
    print("="*100)
    print(summary.to_string())
    print("="*100 + "\n")
    
    return summary


# ====================
# OPTION 2: Detailed Metrics Export
# ====================

def export_detailed_metrics():
    """
    Export all metrics for deeper analysis.
    Includes per-class metrics and training history.
    """
    runs = mlflow.search_runs(
        experiment_names=["emotion-movie-transfer"],
        order_by=["start_time DESC"]
    )
    
    detailed_data = []
    
    for _, row in runs.iterrows():
        run_data = {
            "run_id": row["run_id"],
            "run_name": row.get("tags.run_name", "unknown"),
            "data_source": row.get("tags.data_source", "unknown"),
        }
        
        # Extract all metrics and params
        for col in row.index:
            if col.startswith("metrics.") or col.startswith("params."):
                key = col.replace("metrics.", "").replace("params.", "")
                run_data[key] = row[col]
        
        detailed_data.append(run_data)
    
    # Convert to DataFrame
    detailed_df = pl.DataFrame(detailed_data)
    
    # Save to parquet (preserves types better than CSV)
    detailed_df.write_parquet("mlflow_runs_detailed.parquet")
    print("Saved to: mlflow_runs_detailed.parquet")
    
    # Also save as JSON for easy viewing
    detailed_df.write_json("mlflow_runs_detailed.json")
    print("Saved to: mlflow_runs_detailed.json")
    
    return detailed_df


# ====================
# OPTION 3: Confusion Matrices
# ====================

def export_confusion_matrices():
    """
    Export confusion matrices for all runs.
    Useful for understanding misclassification patterns.
    """
    import mlflow.artifacts
    
    runs = mlflow.search_runs(
        experiment_names=["emotion-movie-transfer"],
        order_by=["start_time DESC"]
    )
    
    cm_data = {}
    
    for _, row in runs.iterrows():
        run_id = row["run_id"]
        run_name = row.get("tags.run_name", "unknown")
        
        try:
            # Download confusion matrix artifact
            cm_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="confusion_matrix_normalized.parquet"
            )
            
            # Load confusion matrix
            cm_df = pl.read_parquet(cm_path)
            cm_data[run_name] = cm_df
            
        except Exception as e:
            print(f"Could not load confusion matrix for {run_name}: {e}")
    
    # Save all confusion matrices
    for run_name, cm_df in cm_data.items():
        safe_name = run_name.replace("/", "_").replace(" ", "_")
        output_path = f"confusion_matrices/{safe_name}.parquet"
        Path("confusion_matrices").mkdir(exist_ok=True)
        cm_df.write_parquet(output_path)
        print(f"Saved: {output_path}")
    
    return cm_data


# ====================
# OPTION 4: Custom Comparison Table
# ====================

def create_custom_comparison():
    """
    Create a custom comparison focused on your specific questions.
    Tailored for pexels/pixabay/combined comparison.
    """
    runs = mlflow.search_runs(
        experiment_names=["emotion-movie-transfer"],
        order_by=["start_time DESC"]
    )
    
    comparison_data = []
    
    for _, row in runs.iterrows():
        run_name = row.get("tags.run_name", "unknown")
        data_source = row.get("tags.data_source", "unknown")
        
        # Parse source and version from run name
        # Assuming format like "pixabay_v1" or "pexels_v2"
        source = data_source
        version = "v1" if "v1" in run_name.lower() else "v2" if "v2" in run_name.lower() else "unknown"
        
        comparison_data.append({
            "run_name": run_name,
            "source": source,
            "version": version,
            "search_method": "[emotion] + face" if version == "v1" else "3 targeted keywords",
            
            # Sample sizes
            "train_samples": row.get("params.train_samples", None),
            "val_samples": row.get("params.val_samples", None),
            
            # Training performance
            "val_acc": row.get("metrics.val_acc", None),
            "best_val_acc": row.get("metrics.best_val_acc", None),
            
            # Per-emotion recall (validation)
            "fear_recall": row.get("metrics.recall_fear", None),
            "happy_recall": row.get("metrics.recall_happy", None),
            "sad_recall": row.get("metrics.recall_sad", None),
            "angry_recall": row.get("metrics.recall_angry", None),
            "surprise_recall": row.get("metrics.recall_surprise", None),
            
            # Movie evaluation (if available)
            "movie_total_faces": row.get("metrics.eval_movies_total_faces", None),
            "movie_sad_dominant_pct": row.get("metrics.eval_movies_dominant_sad_pct", None),
            "movie_fear_dominant_pct": row.get("metrics.eval_movies_dominant_fear_pct", None),
            "movie_happy_dominant_pct": row.get("metrics.eval_movies_dominant_happy_pct", None),
        })
    
    comparison_df = pl.DataFrame(comparison_data)
    
    # Save
    comparison_df.write_csv("model_comparison.csv")
    print("Saved to: model_comparison.csv")
    
    # Print formatted table
    print("\n" + "="*120)
    print("MODEL COMPARISON")
    print("="*120)
    print(comparison_df.to_pandas().to_string())
    print("="*120 + "\n")
    
    # Group by source and version
    print("\nAVERAGE PERFORMANCE BY SOURCE:")
    grouped = comparison_df.group_by(["source", "version"]).agg([
        pl.col("val_acc").mean().alias("avg_val_acc"),
        pl.col("fear_recall").mean().alias("avg_fear_recall"),
        pl.col("happy_recall").mean().alias("avg_happy_recall"),
        pl.col("train_samples").mean().alias("avg_samples"),
    ])
    print(grouped)
    
    return comparison_df


# ====================
# OPTION 5: Training History Export
# ====================

def export_training_histories():
    """
    Export epoch-by-epoch training history for all runs.
    Useful for understanding training dynamics.
    """
    client = mlflow.tracking.MlflowClient()
    
    runs = mlflow.search_runs(
        experiment_names=["emotion-movie-transfer"],
        order_by=["start_time DESC"]
    )
    
    histories = {}
    
    for _, row in runs.iterrows():
        run_id = row["run_id"]
        run_name = row.get("tags.run_name", "unknown")
        
        # Get training history metrics
        train_loss_history = client.get_metric_history(run_id, "train_loss")
        val_acc_history = client.get_metric_history(run_id, "val_acc")
        
        if train_loss_history and val_acc_history:
            history_data = {
                "epoch": [m.step for m in train_loss_history],
                "train_loss": [m.value for m in train_loss_history],
                "val_acc": [m.value for m in val_acc_history],
            }
            
            histories[run_name] = pl.DataFrame(history_data)
    
    # Save all histories
    Path("training_histories").mkdir(exist_ok=True)
    for run_name, history_df in histories.items():
        safe_name = run_name.replace("/", "_").replace(" ", "_")
        output_path = f"training_histories/{safe_name}.parquet"
        history_df.write_parquet(output_path)
        print(f"Saved: {output_path}")
    
    return histories


# ====================
# RECOMMENDED: All-in-One Export
# ====================

def export_all_for_sharing():
    """
    Export everything in one go.
    Creates a complete snapshot of your experiments.
    """
    from pathlib import Path
    from datetime import datetime
    
    # Create export directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path(f"mlflow_export_{timestamp}")
    export_dir.mkdir(exist_ok=True)
    
    print(f"Exporting to: {export_dir}")
    print()
    
    # 1. Summary table
    print("1. Exporting summary table...")
    summary = export_runs_summary()
    summary.to_csv(export_dir / "runs_summary.csv", index=False)
    
    # 2. Detailed metrics
    print("\n2. Exporting detailed metrics...")
    detailed = export_detailed_metrics()
    detailed.write_parquet(export_dir / "runs_detailed.parquet")
    detailed.write_json(export_dir / "runs_detailed.json")
    
    # 3. Custom comparison
    print("\n3. Creating custom comparison...")
    comparison = create_custom_comparison()
    comparison.write_csv(export_dir / "model_comparison.csv")
    
    # 4. Confusion matrices
    print("\n4. Exporting confusion matrices...")
    (export_dir / "confusion_matrices").mkdir(exist_ok=True)
    cm_data = export_confusion_matrices()
    for run_name, cm_df in cm_data.items():
        safe_name = run_name.replace("/", "_").replace(" ", "_")
        cm_df.write_parquet(export_dir / "confusion_matrices" / f"{safe_name}.parquet")
    
    # 5. Training histories
    print("\n5. Exporting training histories...")
    (export_dir / "training_histories").mkdir(exist_ok=True)
    histories = export_training_histories()
    for run_name, history_df in histories.items():
        safe_name = run_name.replace("/", "_").replace(" ", "_")
        history_df.write_parquet(export_dir / "training_histories" / f"{safe_name}.parquet")
    
    # Create README
    readme = f"""# MLflow Export - {timestamp}

This directory contains a complete export of your emotion classification experiments.

## Files:

1. **runs_summary.csv** - High-level comparison of all runs
2. **runs_detailed.parquet/json** - All metrics and parameters
3. **model_comparison.csv** - Custom comparison by source/version
4. **confusion_matrices/** - Normalized confusion matrices per run
5. **training_histories/** - Epoch-by-epoch training metrics

## Quick Analysis:

Load the summary:
```python
import polars as pl
summary = pl.read_csv("runs_summary.csv")
print(summary)
```

Compare sources:
```python
comparison = pl.read_csv("model_comparison.csv")
print(comparison.group_by("source").agg(pl.col("val_acc").mean()))
```
"""
    
    with open(export_dir / "README.md", "w") as f:
        f.write(readme)
    
    print(f"\n{'='*60}")
    print(f"Export complete! All files saved to: {export_dir}")
    print(f"{'='*60}\n")
    print(f"To share with Claude, you can either:")
    print(f"1. Upload {export_dir}/runs_summary.csv (simplest)")
    print(f"2. Upload {export_dir}/model_comparison.csv (custom analysis)")
    print(f"3. Copy and paste the contents of runs_summary.csv")
    
    return export_dir


# ====================
# USAGE
# ====================

if __name__ == "__main__":
    # Choose one:
    
    # Option 1: Quick summary (good for initial discussion)
    summary = export_runs_summary()
    
    # Option 2: Custom comparison (best for your specific questions)
    # comparison = create_custom_comparison()
    
    # Option 3: Everything (most comprehensive)
    # export_dir = export_all_for_sharing()