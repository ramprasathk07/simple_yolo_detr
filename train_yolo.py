"""
train_yolo.py – YOLO training with configurable model versions.

Supports: YOLOv3, v5, v6, v8, v9, v10, v11, YOLO-NAS, YOLO-World
All via the Ultralytics library.

Usage:
    python train_yolo.py                                 # config.yaml defaults
    python train_yolo.py --model yolov8n.pt              # specific model
    python train_yolo.py --model yolo11n.pt --epochs 100
    python train_yolo.py --model yolov5su.pt --device cuda --batch_size 32
"""

import argparse
import sys
from pathlib import Path

from utils import load_config, get_device, make_run_dir, init_wandb, finish_wandb
from dataset import build_yolo_data_yaml


# ── Available YOLO versions & their default weights ──────────────────
YOLO_VERSIONS = {
    # YOLOv3
    "yolov3":       "yolov3u.pt",
    "yolov3-tiny":  "yolov3-tinyu.pt",

    # YOLOv5  (u = ultralytics retrained)
    "yolov5n":   "yolov5nu.pt",
    "yolov5s":   "yolov5su.pt",
    "yolov5m":   "yolov5mu.pt",
    "yolov5l":   "yolov5lu.pt",
    "yolov5x":   "yolov5xu.pt",

    # YOLOv6
    "yolov6n":   "yolov6n.pt",
    "yolov6s":   "yolov6s.pt",
    "yolov6m":   "yolov6m.pt",
    "yolov6l":   "yolov6l.pt",

    # YOLOv8
    "yolov8n":   "yolov8n.pt",
    "yolov8s":   "yolov8s.pt",
    "yolov8m":   "yolov8m.pt",
    "yolov8l":   "yolov8l.pt",
    "yolov8x":   "yolov8x.pt",

    # YOLOv9
    "yolov9t":   "yolov9t.pt",
    "yolov9s":   "yolov9s.pt",
    "yolov9m":   "yolov9m.pt",
    "yolov9c":   "yolov9c.pt",
    "yolov9e":   "yolov9e.pt",

    # YOLOv10
    "yolov10n":  "yolov10n.pt",
    "yolov10s":  "yolov10s.pt",
    "yolov10m":  "yolov10m.pt",
    "yolov10l":  "yolov10l.pt",
    "yolov10x":  "yolov10x.pt",

    # YOLO11 (latest)
    "yolo11n":   "yolo11n.pt",
    "yolo11s":   "yolo11s.pt",
    "yolo11m":   "yolo11m.pt",
    "yolo11l":   "yolo11l.pt",
    "yolo11x":   "yolo11x.pt",
}


def resolve_yolo_model(model_str: str) -> str:
    """
    Resolve model string to weights path.
    Accepts:
      - A key from YOLO_VERSIONS (e.g. "yolov8n")
      - A direct .pt path (e.g. "yolov8n.pt" or "/path/to/custom.pt")
    """
    if model_str in YOLO_VERSIONS:
        weights = YOLO_VERSIONS[model_str]
        print(f"  🔖 Resolved '{model_str}' → {weights}")
        return weights
    return model_str


def print_available_models():
    """Print all available YOLO model versions."""
    print("\n📋 Available YOLO models:\n")
    current_family = ""
    for key, weights in YOLO_VERSIONS.items():
        family = "".join(c for c in key if not c.isdigit() and c not in "nsmltcex-")
        version = key.replace(family, "").split("-")[0] if "-" not in key else key
        # Group by version family
        prefix = key.rstrip("nsmltcex")
        if prefix != current_family:
            current_family = prefix
            print(f"  {'─'*40}")
        print(f"    {key:18s} → {weights}")
    print()


def train(cfg: dict, run_dir: Path):
    """Run YOLO training."""
    from ultralytics import YOLO

    device = get_device(cfg["device"])

    # Resolve model version
    model_str = cfg.get("yolo_model", "yolov8n.pt")
    weights = resolve_yolo_model(model_str)
    model = YOLO(weights)

    # Resolve dataset
    data_yaml = build_yolo_data_yaml(cfg)
    print(f"  📄 Data:    {data_yaml}")
    print(f"  🏗  Model:  {weights}")
    print(f"  ⚙  Device: {device}\n")

    # Task detection: most YOLO models do 'detect' by default
    task = cfg.get("yolo_task", "detect")

    train_args = dict(
        data=data_yaml,
        task=task,
        epochs=cfg["epochs"],
        batch=cfg["batch_size"],
        imgsz=cfg["img_size"],
        device=str(device),
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        lr0=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        workers=cfg["num_workers"],
        save_period=cfg.get("save_interval", 0),
        patience=cfg["early_stopping"]["patience"] if cfg["early_stopping"]["enabled"] else 0,
        verbose=True,
    )

    # Optional: augmentation settings from config
    aug = cfg.get("augmentation", {})
    if aug:
        train_args.update({
            "hsv_h": aug.get("hsv_h", 0.015),
            "hsv_s": aug.get("hsv_s", 0.7),
            "hsv_v": aug.get("hsv_v", 0.4),
            "degrees": aug.get("degrees", 0.0),
            "translate": aug.get("translate", 0.1),
            "scale": aug.get("scale", 0.5),
            "flipud": aug.get("flipud", 0.0),
            "fliplr": aug.get("fliplr", 0.5),
            "mosaic": aug.get("mosaic", 1.0),
            "mixup": aug.get("mixup", 0.0),
        })

    print("🚀 Starting YOLO training …\n")
    results = model.train(**train_args)
    print("\n✅ YOLO training complete.")
    return results


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO models (v3/v5/v6/v8/v9/v10/v11).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model", type=str, help="YOLO model version or weights path\n(e.g. yolov8n, yolov5s, yolo11m, /path/to/custom.pt)")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Initial learning rate")
    parser.add_argument("--img_size", type=int, help="Input image size")
    parser.add_argument("--device", type=str, help="Device: cuda, cpu, mps, auto")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--list_models", action="store_true", help="List all available YOLO versions")
    args = parser.parse_args()

    if args.list_models:
        print_available_models()
        sys.exit(0)

    cfg = load_config(args.config)
    cfg["model_type"] = "yolo"

    # CLI overrides
    if args.model:
        cfg["yolo_model"] = args.model
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.lr:
        cfg["learning_rate"] = args.lr
    if args.img_size:
        cfg["img_size"] = args.img_size
    if args.device:
        cfg["device"] = args.device
    if args.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False

    run_dir = make_run_dir(cfg.get("output_dir", "./runs"))
    init_wandb(cfg, "yolo", run_dir)

    model_name = cfg.get("yolo_model", "yolov8n.pt")
    print(f"\n{'='*60}")
    print(f"  YOLO Training")
    print(f"  Model : {model_name}")
    print(f"  Epochs: {cfg['epochs']}  │  Batch: {cfg['batch_size']}  │  LR: {cfg['learning_rate']}")
    print(f"{'='*60}")

    train(cfg, run_dir)
    finish_wandb()


if __name__ == "__main__":
    main()
