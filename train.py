"""
train.py – Unified entry-point that dispatches to train_yolo.py or train_detr.py.

Usage:
    python train.py                          # uses config.yaml defaults
    python train.py --model_type yolo        # train YOLO
    python train.py --model_type detr        # train DETR
    python train.py --model yolov10n         # auto-detects YOLO
    python train.py --model rt-detr-l        # auto-detects DETR

Or call the specific scripts directly:
    python train_yolo.py --model yolov8n --epochs 100
    python train_detr.py --model detr-resnet-50 --epochs 50
"""

import argparse
import sys

from utils import load_config, make_run_dir, init_wandb, finish_wandb


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO or DETR for object detection.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model_type", type=str, choices=["yolo", "detr"], help="Model family: yolo or detr")
    parser.add_argument("--model", type=str, help="Model version (e.g. yolov8n, detr-resnet-50, rt-detr-l)")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--img_size", type=int, help="Input image size")
    parser.add_argument("--device", type=str, help="Device: cuda, cpu, mps, auto")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply CLI overrides
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

    # Determine model type from --model_type or auto-detect from --model
    model_type = args.model_type or cfg.get("model_type", "yolo")

    if args.model:
        model_lower = args.model.lower()
        if any(model_lower.startswith(p) for p in ["yolov", "yolo1"]):
            model_type = "yolo"
            cfg["yolo_model"] = args.model
        else:
            model_type = "detr"
            cfg["detr_model"] = args.model

    if args.model_type:
        model_type = args.model_type

    # Dispatch
    run_dir = make_run_dir(cfg.get("output_dir", "./runs"))
    init_wandb(cfg, model_type, run_dir)

    print(f"\n{'='*60}")
    print(f"  Model type: {model_type.upper()}")
    print(f"  Epochs: {cfg['epochs']}  │  Batch: {cfg['batch_size']}  │  LR: {cfg['learning_rate']}")
    print(f"{'='*60}\n")

    if model_type == "yolo":
        from train_yolo import train
        cfg["model_type"] = "yolo"
        train(cfg, run_dir)

    elif model_type == "detr":
        from train_detr import load_detr_model, train_hf_detr, train_rtdetr
        cfg["model_type"] = "detr"

        model_key = cfg.get("detr_model", "detr-resnet-50")
        num_labels = cfg.get("num_labels", 91)
        model, processor, backend = load_detr_model(model_key, num_labels)

        if backend == "ultralytics":
            train_rtdetr(cfg, run_dir, model)
        else:
            train_hf_detr(cfg, run_dir, model, processor)
    else:
        print(f"❌ Unknown model_type: {model_type}")
        sys.exit(1)

    finish_wandb()


if __name__ == "__main__":
    main()
