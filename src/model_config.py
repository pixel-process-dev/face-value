#!/usr/bin/env python
# coding: utf-8

"""
Config module for emotion classification training.

Provides reusable components for transforms, optimizers, and data augmentation
that can be referenced by name in JSON config files.
"""

import torch
from torch import optim
from torchvision import transforms
from typing import Dict, Any, Optional


# -------------------------
# Transform Presets
# -------------------------

def get_transform_preset(
    preset_name: str,
    image_size: int = 224,
    augmentation: Optional[str] = None,
) -> transforms.Compose:
    """
    Get a transform pipeline by preset name.
    
    Args:
        preset_name: One of "imagenet", "basic", "grayscale"
        image_size: Target image size (default: 224)
        augmentation: Optional augmentation preset name
    
    Returns:
        transforms.Compose pipeline
    """
    
    # Base transforms (always applied)
    base_transforms = []
    
    # Augmentation (applied before normalization)
    aug_transforms = []
    if augmentation == "light":
        aug_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    elif augmentation == "medium":
        aug_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]
    elif augmentation == "heavy":
        aug_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        ]
    elif augmentation is not None and augmentation != "none":
        raise ValueError(f"Unknown augmentation preset: {augmentation}")
    
    # Normalization presets
    if preset_name == "imagenet":
        # Standard ImageNet normalization (for pretrained models)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    elif preset_name == "basic":
        # Simple normalization to [-1, 1]
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )
    elif preset_name == "grayscale":
        # For grayscale images
        base_transforms.append(transforms.Grayscale(num_output_channels=3))
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    else:
        raise ValueError(f"Unknown transform preset: {preset_name}")
    
    # Combine all transforms
    all_transforms = (
        base_transforms +
        [transforms.Resize((image_size, image_size))] +
        aug_transforms +
        [transforms.ToTensor(), normalize]
    )
    
    return transforms.Compose(all_transforms)


# -------------------------
# Optimizer Presets
# -------------------------

def get_optimizer(
    optimizer_name: str,
    parameters,
    lr: float,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Get an optimizer by name with sensible defaults.
    
    Args:
        optimizer_name: One of "adam", "adamw", "sgd", "rmsprop"
        parameters: Model parameters to optimize
        lr: Learning rate
        **kwargs: Additional optimizer-specific parameters
    
    Returns:
        Optimizer instance
    """
    
    if optimizer_name == "adam":
        return optim.Adam(
            parameters,
            lr=lr,
            betas=kwargs.get("betas", (0.9, 0.999)),
            eps=kwargs.get("eps", 1e-8),
            weight_decay=kwargs.get("weight_decay", 0.0),
        )
    
    elif optimizer_name == "adamw":
        return optim.AdamW(
            parameters,
            lr=lr,
            betas=kwargs.get("betas", (0.9, 0.999)),
            eps=kwargs.get("eps", 1e-8),
            weight_decay=kwargs.get("weight_decay", 0.01),  # Higher default for AdamW
        )
    
    elif optimizer_name == "sgd":
        return optim.SGD(
            parameters,
            lr=lr,
            momentum=kwargs.get("momentum", 0.9),
            weight_decay=kwargs.get("weight_decay", 0.0),
            nesterov=kwargs.get("nesterov", False),
        )
    
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(
            parameters,
            lr=lr,
            alpha=kwargs.get("alpha", 0.99),
            eps=kwargs.get("eps", 1e-8),
            weight_decay=kwargs.get("weight_decay", 0.0),
            momentum=kwargs.get("momentum", 0.0),
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


# -------------------------
# Scheduler Presets (Optional)
# -------------------------

def get_scheduler(
    scheduler_name: Optional[str],
    optimizer: torch.optim.Optimizer,
    **kwargs
):
    """
    Get a learning rate scheduler by name.
    
    Args:
        scheduler_name: One of "step", "cosine", "plateau", or None
        optimizer: Optimizer instance
        **kwargs: Scheduler-specific parameters
    
    Returns:
        Scheduler instance or None
    """
    
    if scheduler_name is None or scheduler_name == "none":
        return None
    
    elif scheduler_name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 10),
            gamma=kwargs.get("gamma", 0.1),
        )
    
    elif scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("T_max", 30),
            eta_min=kwargs.get("eta_min", 0.0),
        )
    
    elif scheduler_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get("mode", "max"),  # "max" for accuracy
            factor=kwargs.get("factor", 0.1),
            patience=kwargs.get("patience", 5),
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


# -------------------------
# Config Utilities
# -------------------------

def get_config_summary(config: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of the config.
    """
    lines = [
        "Configuration Summary:",
        "=" * 60,
        f"Run Name: {config.get('run_name', 'N/A')}",
        f"Data Source: {config.get('data_source', 'N/A')}",
        f"Metadata: {config.get('metadata_path', 'N/A')}",
        "",
        "Training:",
        f"  - Epochs: {config['training']['epochs']}",
        f"  - Batch Size: {config['training']['batch_size']}",
        f"  - Learning Rate: {config['training']['learning_rate']}",
        f"  - Optimizer: {config['training']['optimizer']}",
        f"  - Scheduler: {config['training'].get('scheduler', 'none')}",
        "",
        "Data:",
        f"  - Image Size: {config['data']['image_size']}",
        f"  - Transform Preset: {config['data']['transform_preset']}",
        f"  - Augmentation: {config['data'].get('augmentation', 'none')}",
        "",
        "Model:",
        f"  - Architecture: {config['model']['architecture']}",
        f"  - Pretrained: {config['model']['pretrained']}",
        f"  - Unfrozen Layers: {config['model'].get('unfrozen_layers', 'layer3,layer4,fc')}",
        "=" * 60,
    ]
    return "\n".join(lines)


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that config has all required fields.
    """
    required_fields = {
        "run_name": str,
        "metadata_path": str,
        "output_dir": str,
        "training": dict,
        "data": dict,
        "model": dict,
    }
    
    for field, expected_type in required_fields.items():
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(config[field], expected_type):
            raise ValueError(f"Field '{field}' must be of type {expected_type.__name__}")
    
    # Validate training config
    training_required = ["epochs", "batch_size", "learning_rate", "optimizer"]
    for field in training_required:
        if field not in config["training"]:
            raise ValueError(f"Missing required training field: {field}")
    
    # Validate data config
    data_required = ["image_size", "transform_preset"]
    for field in data_required:
        if field not in config["data"]:
            raise ValueError(f"Missing required data field: {field}")
    
    # Validate model config
    model_required = ["architecture", "pretrained"]
    for field in model_required:
        if field not in config["model"]:
            raise ValueError(f"Missing required model field: {field}")


# -------------------------
# Example Configs
# -------------------------

EXAMPLE_CONFIGS = {
    "baseline": {
        "run_name": "pixabay-baseline",
        "data_source": "pixabay",
        "metadata_path": "data/pixabay_metadata.parquet",
        "output_dir": "outputs/pixabay_baseline",
        "training": {
            "epochs": 30,
            "batch_size": 32,
            "learning_rate": 3e-4,
            "optimizer": "adam",
            "scheduler": "none",
            "seed": 42,
        },
        "data": {
            "image_size": 224,
            "transform_preset": "imagenet",
            "augmentation": "none",
        },
        "model": {
            "architecture": "resnet18",
            "pretrained": "IMAGENET1K_V1",
            "unfrozen_layers": ["layer3", "layer4", "fc"],
        },
        "mlflow": {
            "experiment_name": "emotion-movie-transfer",
        }
    },
    
    "with_augmentation": {
        "run_name": "pixabay-light-aug",
        "data_source": "pixabay",
        "metadata_path": "data/pixabay_metadata.parquet",
        "output_dir": "outputs/pixabay_aug",
        "training": {
            "epochs": 30,
            "batch_size": 32,
            "learning_rate": 3e-4,
            "optimizer": "adam",
            "scheduler": "none",
            "seed": 42,
        },
        "data": {
            "image_size": 224,
            "transform_preset": "imagenet",
            "augmentation": "light",  # Added augmentation
        },
        "model": {
            "architecture": "resnet18",
            "pretrained": "IMAGENET1K_V1",
            "unfrozen_layers": ["layer3", "layer4", "fc"],
        },
        "mlflow": {
            "experiment_name": "emotion-movie-transfer",
        }
    },
    
    "adamw_cosine": {
        "run_name": "pixabay-adamw-cosine",
        "data_source": "pixabay",
        "metadata_path": "data/pixabay_metadata.parquet",
        "output_dir": "outputs/pixabay_adamw",
        "training": {
            "epochs": 30,
            "batch_size": 32,
            "learning_rate": 3e-4,
            "optimizer": "adamw",  # Changed optimizer
            "optimizer_kwargs": {
                "weight_decay": 0.01,
            },
            "scheduler": "cosine",  # Added scheduler
            "scheduler_kwargs": {
                "T_max": 30,
            },
            "seed": 42,
        },
        "data": {
            "image_size": 224,
            "transform_preset": "imagenet",
            "augmentation": "light",
        },
        "model": {
            "architecture": "resnet18",
            "pretrained": "IMAGENET1K_V1",
            "unfrozen_layers": ["layer3", "layer4", "fc"],
        },
        "mlflow": {
            "experiment_name": "emotion-movie-transfer",
        }
    },
}


if __name__ == "__main__":
    # Demo the config system
    print("Available transform presets: imagenet, basic, grayscale")
    print("Available augmentation presets: none, light, medium, heavy")
    print("Available optimizers: adam, adamw, sgd, rmsprop")
    print("Available schedulers: none, step, cosine, plateau")
    print()
    
    # Show example transform
    print("Example: ImageNet transform with light augmentation")
    transform = get_transform_preset("imagenet", image_size=224, augmentation="light")
    print(transform)
    print()
    
    # Show example config
    print("Example baseline config:")
    print(get_config_summary(EXAMPLE_CONFIGS["baseline"]))
