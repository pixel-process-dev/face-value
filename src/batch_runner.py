#!/usr/bin/env python
# coding: utf-8

"""
Batch runner for training multiple configs in sequence.
Useful for running experiments overnight or on remote server.
"""

import subprocess
import json
from pathlib import Path
import time
from datetime import datetime


def run_config(config_path: Path, dry_run: bool = False):
    """
    Run training with a single config file.
    """
    cmd = ["python", "train_from_config.py", "--config", str(config_path)]
    
    print(f"\n{'='*60}")
    print(f"Running: {config_path.name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    if dry_run:
        print("[DRY RUN] Would execute the above command")
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


def run_batch(config_paths: list[Path], dry_run: bool = False, stop_on_error: bool = False):
    """
    Run multiple configs in sequence.
    
    Args:
        config_paths: List of config file paths
        dry_run: If True, just print what would be run
        stop_on_error: If True, stop on first failure
    """
    print(f"\n{'='*60}")
    print(f"BATCH TRAINING")
    print(f"{'='*60}")
    print(f"Total configs: {len(config_paths)}")
    print(f"Dry run: {dry_run}")
    print(f"Stop on error: {stop_on_error}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    results = []
    overall_start = time.time()
    
    for i, config_path in enumerate(config_paths, 1):
        print(f"\n[{i}/{len(config_paths)}] Processing: {config_path.name}")
        
        success = run_config(config_path, dry_run=dry_run)
        
        results.append({
            "config": config_path.name,
            "success": success,
        })
        
        if not success and stop_on_error:
            print("\n⚠️  Stopping due to error (stop_on_error=True)")
            break
    
    overall_elapsed = time.time() - overall_start
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"BATCH TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {overall_elapsed/3600:.1f} hours")
    print(f"Completed: {sum(r['success'] for r in results)}/{len(results)}")
    print()
    
    for result in results:
        status = "✓" if result["success"] else "✗"
        print(f"{status} {result['config']}")
    
    print(f"{'='*60}\n")
    
    # Return True if all succeeded
    return all(r["success"] for r in results)


def load_config_summary(config_path: Path) -> dict:
    """
    Load config and extract key info for display.
    """
    with open(config_path) as f:
        config = json.load(f)
    
    return {
        "run_name": config.get("run_name", "unknown"),
        "data_source": config.get("data_source", "unknown"),
        "lr": config["training"]["learning_rate"],
        "optimizer": config["training"]["optimizer"],
        "augmentation": config["data"].get("augmentation", "none"),
    }


def print_batch_preview(config_paths: list[Path]):
    """
    Print a preview of all configs that will be run.
    """
    print(f"\n{'='*60}")
    print("BATCH PREVIEW")
    print(f"{'='*60}\n")
    
    for i, path in enumerate(config_paths, 1):
        info = load_config_summary(path)
        print(f"{i}. {path.name}")
        print(f"   Run: {info['run_name']}")
        print(f"   Data: {info['data_source']}, LR: {info['lr']}, "
              f"Opt: {info['optimizer']}, Aug: {info['augmentation']}")
        print()
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch runner for multiple training configs"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        help="Config files to run (can use wildcards)"
    )
    parser.add_argument(
        "--config-dir",
        help="Directory containing configs to run (runs all .json files)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop batch if any config fails"
    )
    
    args = parser.parse_args()
    
    # Collect config paths
    config_paths = []
    
    if args.config_dir:
        config_dir = Path(args.config_dir)
        config_paths.extend(sorted(config_dir.glob("*.json")))
    
    if args.configs:
        for pattern in args.configs:
            config_paths.extend(Path().glob(pattern))
    
    if not config_paths:
        print("Error: No config files specified")
        print()
        print("Usage examples:")
        print("  python batch_runner.py --configs configs/baseline.json configs/aug.json")
        print("  python batch_runner.py --config-dir configs/baselines")
        print("  python batch_runner.py --configs 'configs/*.json'")
        exit(1)
    
    # Remove duplicates and sort
    config_paths = sorted(set(config_paths))
    
    # Print preview
    print_batch_preview(config_paths)
    
    # Confirm before running (unless dry run)
    if not args.dry_run:
        response = input(f"Run {len(config_paths)} configs? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled")
            exit(0)
    
    # Run batch
    success = run_batch(
        config_paths,
        dry_run=args.dry_run,
        stop_on_error=args.stop_on_error
    )
    
    exit(0 if success else 1)
