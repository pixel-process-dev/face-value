#!/usr/bin/env python
# coding: utf-8

"""
Train emotion classification model using JSON config files.
"""

from pathlib import Path
import json
import shutil
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torchmetrics.classification import ConfusionMatrix

import polars as pl
import mlflow
import mlflow.pytorch

from utils.fv_utils import (
    ensure_dir
)

from utils.model_utils import (
    FaceDataset,
    evaluate_with_outputs
)

from model_config import (
    get_transform_preset,
    get_optimizer,
    get_scheduler,
    get_config_summary,
    validate_config,
)
# -------------------------
# Training helpers
# -------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# -------------------------
# Main training function
# -------------------------

def train_from_config(config_path: Path):
    """
    Train model using a JSON config file.
    """
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Validate config
    validate_config(config)
    
    # Print config summary
    print(get_config_summary(config))
    print()
    
    # Extract config sections
    training_cfg = config["training"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    mlflow_cfg = config["mlflow"]
    
    # Set seed
    torch.manual_seed(training_cfg["seed"])
    
    # Setup paths
    metadata_path = Path(config["metadata_path"])
    output_dir = Path(config["output_dir"])
    ensure_dir(output_dir)
    
    # Copy config and metadata to output dir for persistence
    config_copy_path = output_dir / "config.json"
    shutil.copy(config_path, config_copy_path)
    print(f"Copied config to: {config_copy_path}")
    
    # Copy metadata file to output dir
    metadata_copy_path = output_dir / "metadata.parquet"
    shutil.copy(metadata_path, metadata_copy_path)
    print(f"Copied metadata to: {metadata_copy_path}")
    print()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # ---- MLflow setup ----
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    
    # Start MLflow run
    run_name = config.get("run_name")
    
    with mlflow.start_run(run_name=run_name) as run:
        # Log entire config as params
        mlflow.log_params({
            "config_file": str(config_path.name),
            "data_source": config.get("data_source", "unknown"),
            "run_name": run_name,
        })
        
        # Log training params
        for key, value in training_cfg.items():
            if key not in ["optimizer_kwargs", "scheduler_kwargs"]:
                mlflow.log_param(f"train_{key}", value)
        
        # Log optimizer kwargs
        for key, value in training_cfg.get("optimizer_kwargs", {}).items():
            mlflow.log_param(f"opt_{key}", value)
        
        # Log scheduler kwargs
        for key, value in training_cfg.get("scheduler_kwargs", {}).items():
            mlflow.log_param(f"sched_{key}", value)
        
        # Log data params
        for key, value in data_cfg.items():
            mlflow.log_param(f"data_{key}", value)
        
        # Log model params
        for key, value in model_cfg.items():
            if key != "unfrozen_layers":
                mlflow.log_param(f"model_{key}", value)
            else:
                mlflow.log_param("model_unfrozen_layers", ",".join(value))
        
        mlflow.log_param("device", device)
        
        # Log config file as artifact
        mlflow.log_artifact(str(config_copy_path))
        
        # ---- Load + filter data ----
        df = pl.read_parquet(metadata_path)
        
        emotions = sorted(df["emotion"].unique().to_list())
        label_map = {e: i for i, e in enumerate(emotions)}
        
        df = df.with_columns(
            pl.col("emotion").replace(label_map).alias("label").cast(pl.Int64)
        )
        
        # Log dataset info
        mlflow.log_params({
            "num_classes": len(emotions),
            "emotions": ",".join(emotions),
            "total_samples": len(df),
        })
        
        # Log per-class distribution
        class_dist = df.group_by("emotion").agg(pl.count().alias("count"))
        for row in class_dist.iter_rows(named=True):
            mlflow.log_param(f"class_{row['emotion']}_count", row['count'])
        
        # Deterministic split
        df = df.with_row_count("row_id")
        train_df = df.filter(pl.col("row_id") % 5 != 0)
        val_df = df.filter(pl.col("row_id") % 5 == 0)
        
        mlflow.log_params({
            "train_samples": len(train_df),
            "val_samples": len(val_df),
        })
        
        print(f"Loaded {len(df)} samples")
        print(f"  - Train: {len(train_df)}")
        print(f"  - Val: {len(val_df)}")
        print(f"  - Classes: {emotions}")
        print()
        
        # ---- Transforms ----
        train_transform = get_transform_preset(
            data_cfg["transform_preset"],
            image_size=data_cfg["image_size"],
            augmentation=data_cfg.get("augmentation", "none"),
        )
        
        # Validation transform (no augmentation)
        val_transform = get_transform_preset(
            data_cfg["transform_preset"],
            image_size=data_cfg["image_size"],
            augmentation="none",
        )
        
        train_ds = FaceDataset(train_df, train_transform)
        val_ds = FaceDataset(val_df, val_transform)
        
        train_loader = DataLoader(
            train_ds,
            batch_size=training_cfg["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=training_cfg["batch_size"],
            shuffle=False,
        )
        
        # ---- Model ----
        if model_cfg["architecture"] == "resnet18":
            model = models.resnet18(weights=model_cfg["pretrained"])
        elif model_cfg["architecture"] == "resnet50":
            model = models.resnet50(weights=model_cfg["pretrained"])
        else:
            raise ValueError(f"Unknown architecture: {model_cfg['architecture']}")
        
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace final layer
        model.fc = nn.Linear(model.fc.in_features, len(emotions))
        
        # Unfreeze specified layers
        unfrozen_layers = model_cfg.get("unfrozen_layers", ["layer3", "layer4", "fc"])
        for name, param in model.named_parameters():
            for layer_name in unfrozen_layers:
                if name.startswith(layer_name):
                    param.requires_grad = True
                    break
        
        model = model.to(device)
        
        # Log model info
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        mlflow.log_params({
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_pct": round(trainable_params / total_params * 100, 2),
        })
        
        print(f"Model: {model_cfg['architecture']}")
        print(f"  - Total params: {total_params:,}")
        print(f"  - Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print()
        
        # ---- Optimizer & Scheduler ----
        criterion = nn.CrossEntropyLoss()
        
        optimizer = get_optimizer(
            training_cfg["optimizer"],
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=training_cfg["learning_rate"],
            **training_cfg.get("optimizer_kwargs", {})
        )
        
        scheduler = get_scheduler(
            training_cfg.get("scheduler", "none"),
            optimizer,
            **training_cfg.get("scheduler_kwargs", {})
        )
        
        print(f"Optimizer: {training_cfg['optimizer']}")
        if scheduler is not None:
            print(f"Scheduler: {training_cfg.get('scheduler')}")
        print()
        
        # ---- Training loop ----
        records = []
        best_val_acc = 0.0
        
        print("Starting training...")
        print("="*60)
        
        for epoch in range(1, training_cfg["epochs"] + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            
            outputs = evaluate_with_outputs(model, val_loader, device)
            val_acc = (outputs["preds"] == outputs["labels"]).float().mean().item()
            
            # Log metrics to MLflow
            metrics = {
                "train_loss": train_loss,
                "val_acc": val_acc,
            }
            
            # Log learning rate
            if scheduler is not None:
                current_lr = optimizer.param_groups[0]['lr']
                metrics["learning_rate"] = current_lr
            
            mlflow.log_metrics(metrics, step=epoch)
            
            records.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_acc": val_acc,
            })
            
            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_acc={val_acc:.3f}"
            )
            
            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            # Step scheduler
            if scheduler is not None:
                if training_cfg.get("scheduler") == "plateau":
                    scheduler.step(val_acc)
                else:
                    scheduler.step()
        
        print("="*60)
        print()
        
        # Log best validation accuracy
        mlflow.log_metric("best_val_acc", best_val_acc)
        
        # ---- Final diagnostics ----
        labels = outputs["labels"]
        preds = outputs["preds"]
        probs = outputs["probs"]
        
        cm_norm = ConfusionMatrix(
            task="multiclass",
            num_classes=len(emotions),
            normalize="true",
        )(preds, labels)
        
        # Per-class recall
        recalls = cm_norm.diagonal()
        recall_dict = {}
        
        print("Per-class Recall:")
        for emotion, r in zip(emotions, recalls):
            recall_dict[f"recall_{emotion}"] = r.item()
            print(f"  {emotion:>10s}: {r:.3f}")
        print()
        
        # Log per-class recalls
        mlflow.log_metrics(recall_dict)
        
        # ---- Persist per-sample validation predictions ----
        val_pred_df = (
            val_df
            .with_columns([
                pl.Series("true_label", labels.numpy()),
                pl.Series("pred_label", preds.numpy()),
            ])
        )
        
        for i, emotion in enumerate(emotions):
            val_pred_df = val_pred_df.with_columns(
                pl.Series(f"prob_{emotion}", probs[:, i].numpy())
            )
        
        val_pred_path = output_dir / "val_predictions.parquet"
        val_pred_df.write_parquet(val_pred_path)
        mlflow.log_artifact(str(val_pred_path))
        
        # ---- Save artifacts ----
        model_path = output_dir / "model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "label_map": label_map,
                "emotions": emotions,
                "config": config,
            },
            model_path,
        )
        
        # Log PyTorch model to MLflow
        mlflow.log_artifact(str(model_path))
        
        # Log full model with MLflow's model registry format
        mlflow.pytorch.log_model(
            model,
            "model",
            signature=mlflow.models.infer_signature(
                torch.randn(1, 3, data_cfg["image_size"], data_cfg["image_size"]).to(device),
                model(torch.randn(1, 3, data_cfg["image_size"], data_cfg["image_size"]).to(device)).detach().cpu().numpy()
            )
        )
        
        # Save training log
        train_log_path = output_dir / "train_log.parquet"
        pl.DataFrame(records).write_parquet(train_log_path)
        mlflow.log_artifact(str(train_log_path))
        
        # Save run metadata
        run_meta_path = output_dir / "run_meta.json"
        with open(run_meta_path, "w") as f:
            json.dump(
                {
                    "run_name": run_name,
                    "mlflow_run_id": run.info.run_id,
                    "mlflow_experiment": mlflow_cfg["experiment_name"],
                    "timestamp": datetime.now().isoformat(),
                    "best_val_acc": best_val_acc,
                    "device": device,
                },
                f,
                indent=2,
            )
        mlflow.log_artifact(str(run_meta_path))
        
        print("="*60)
        print(f"Training Complete!")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Experiment: {mlflow_cfg['experiment_name']}")
        print(f"Best Val Accuracy: {best_val_acc:.3f}")
        print(f"Output Directory: {output_dir}")
        print("="*60)


# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train emotion classification model from JSON config"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file"
    )
    
    args = parser.parse_args()
    
    train_from_config(Path(args.config))
