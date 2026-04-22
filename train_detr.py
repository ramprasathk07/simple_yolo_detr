"""
train_detr.py – DETR-family training with configurable model versions.

Supports: DETR, Conditional DETR, Deformable DETR, DETA, RT-DETR, YOLOS
All via HuggingFace Transformers.

Usage:
    python train_detr.py                                          # config defaults
    python train_detr.py --model detr-resnet-50                   # specific variant
    python train_detr.py --model rt-detr-l --epochs 100 --device cuda
    python train_detr.py --model conditional-detr-resnet-50 --batch_size 8
    python train_detr.py --list_models
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import load_config, get_device, make_run_dir, init_wandb, log_wandb, finish_wandb
from dataset import build_detr_datasets, detr_collate_fn


# ── Available DETR variants ─────────────────────────────────────────
# Maps a short key → (HuggingFace model ID, model class name, processor class name)
DETR_VERSIONS = {
    # ── DETR (original) ─────────────────────────────────────────────
    "detr-resnet-50": {
        "hf_model": "facebook/detr-resnet-50",
        "model_cls": "DetrForObjectDetection",
        "processor_cls": "DetrImageProcessor",
    },
    "detr-resnet-101": {
        "hf_model": "facebook/detr-resnet-101",
        "model_cls": "DetrForObjectDetection",
        "processor_cls": "DetrImageProcessor",
    },

    # ── Conditional DETR ────────────────────────────────────────────
    "conditional-detr-resnet-50": {
        "hf_model": "microsoft/conditional-detr-resnet-50",
        "model_cls": "ConditionalDetrForObjectDetection",
        "processor_cls": "ConditionalDetrImageProcessor",
    },

    # ── Deformable DETR ─────────────────────────────────────────────
    "deformable-detr": {
        "hf_model": "SenseTime/deformable-detr",
        "model_cls": "DeformableDetrForObjectDetection",
        "processor_cls": "DeformableDetrImageProcessor",
    },

    # ── DETA (DE-formable-DETR + Assignment) ────────────────────────
    "deta-swin-large": {
        "hf_model": "jozhang97/deta-swin-large",
        "model_cls": "DetaForObjectDetection",
        "processor_cls": "DetaImageProcessor",
    },

    # ── RT-DETR (Real-Time DETR by Baidu / Ultralytics) ─────────────
    #    Note: Ultralytics RT-DETR uses the YOLO pipeline
    "rt-detr-l": {
        "hf_model": "rtdetr-l.pt",
        "backend": "ultralytics",
    },
    "rt-detr-x": {
        "hf_model": "rtdetr-x.pt",
        "backend": "ultralytics",
    },

    # ── YOLOS (DETR-style with ViT backbone) ────────────────────────
    "yolos-tiny": {
        "hf_model": "hustvl/yolos-tiny",
        "model_cls": "YolosForObjectDetection",
        "processor_cls": "YolosImageProcessor",
    },
    "yolos-small": {
        "hf_model": "hustvl/yolos-small",
        "model_cls": "YolosForObjectDetection",
        "processor_cls": "YolosImageProcessor",
    },
    "yolos-base": {
        "hf_model": "hustvl/yolos-base",
        "model_cls": "YolosForObjectDetection",
        "processor_cls": "YolosImageProcessor",
    },
}


def print_available_models():
    """Print all available DETR-family models."""
    print("\n📋 Available DETR-family models:\n")
    current_family = ""
    for key, info in DETR_VERSIONS.items():
        family = key.rsplit("-", 1)[0] if "-" in key else key
        if family != current_family:
            current_family = family
            print(f"  {'─'*50}")
        backend = info.get("backend", "transformers")
        hf = info["hf_model"]
        print(f"    {key:38s}  ({backend}) → {hf}")
    print()


def load_detr_model(model_key: str, num_labels: int = 91, device: torch.device = None):
    """
    Load a DETR-family model + processor by key or HF model ID.
    Returns (model, processor, backend).
    """
    import transformers

    # Resolve from registry or treat as raw HF model ID
    if model_key in DETR_VERSIONS:
        info = DETR_VERSIONS[model_key]
    else:
        # Assume it's a raw HF model ID for DETR
        info = {
            "hf_model": model_key,
            "model_cls": "DetrForObjectDetection",
            "processor_cls": "DetrImageProcessor",
        }

    backend = info.get("backend", "transformers")
    hf_model = info["hf_model"]

    if backend == "ultralytics":
        # RT-DETR via Ultralytics
        return hf_model, None, "ultralytics"

    model_cls = getattr(transformers, info["model_cls"])
    proc_cls = getattr(transformers, info["processor_cls"])

    processor = proc_cls.from_pretrained(hf_model)
    model = model_cls.from_pretrained(
        hf_model,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    if device:
        model = model.to(device)

    print(f"  🏗  Model:     {info['model_cls']}")
    print(f"  🔖 Pretrained: {hf_model}")
    print(f"  📏 Labels:     {num_labels}")

    return model, processor, backend


# ══════════════════════════════════════════════════════════════════════
#  RT-DETR TRAINING (via Ultralytics)
# ══════════════════════════════════════════════════════════════════════
def train_rtdetr(cfg: dict, run_dir: Path, weights: str):
    """Train RT-DETR using the Ultralytics pipeline (same as YOLO)."""
    from ultralytics import RTDETR
    from dataset import build_yolo_data_yaml

    device = get_device(cfg["device"])
    model = RTDETR(weights)

    data_yaml = build_yolo_data_yaml(cfg)
    print(f"  📄 Data: {data_yaml}\n")

    train_args = dict(
        data=data_yaml,
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

    print("🚀 Starting RT-DETR training (Ultralytics) …\n")
    results = model.train(**train_args)
    print("\n✅ RT-DETR training complete.")
    return results


# ══════════════════════════════════════════════════════════════════════
#  HF DETR-FAMILY TRAINING
# ══════════════════════════════════════════════════════════════════════
def train_hf_detr(cfg: dict, run_dir: Path, model, processor):
    """Train any HuggingFace DETR-family model."""
    device = get_device(cfg["device"])
    model = model.to(device)

    # ── Datasets ─────────────────────────────────────────────────────
    train_ds, val_ds = build_detr_datasets(cfg, processor)
    print(f"  📊 Train: {len(train_ds)} samples │ Val: {len(val_ds)} samples\n")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=detr_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=detr_collate_fn,
    )

    # ── Optimizer ────────────────────────────────────────────────────
    # Separate backbone LR (10x lower) for better transfer learning
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "backbone" in name or "input_projection" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    base_lr = cfg["learning_rate"]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": base_lr * 0.1},
        {"params": head_params, "lr": base_lr},
    ], weight_decay=cfg["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=base_lr * 0.01
    )

    # ── Training Loop ────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    es = cfg["early_stopping"]

    print("🚀 Starting DETR-family training …\n")

    for epoch in range(1, cfg["epochs"] + 1):
        # ── Train ────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']} [train]", leave=False)

        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # ── Validate ─────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg['epochs']} [val]", leave=False):
                pixel_values = batch["pixel_values"].to(device)
                labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
                outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        scheduler.step()

        # ── Logging ──────────────────────────────────────────────────
        lr = optimizer.param_groups[-1]["lr"]
        print(f"  📈 Epoch {epoch:3d} │ train_loss={avg_train_loss:.4f} │ val_loss={avg_val_loss:.4f} │ lr={lr:.2e}")
        log_wandb({"train/loss": avg_train_loss, "val/loss": avg_val_loss, "lr": lr}, step=epoch)

        # ── Checkpoint ───────────────────────────────────────────────
        save_interval = cfg.get("save_interval", 0)
        if save_interval and epoch % save_interval == 0:
            ckpt = run_dir / f"epoch_{epoch}.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"  💾 Saved checkpoint → {ckpt}")

        if avg_val_loss < best_val_loss - es.get("min_delta", 0):
            best_val_loss = avg_val_loss
            patience_counter = 0
            if cfg.get("save_best", True):
                best_path = run_dir / "best_detr.pth"
                torch.save(model.state_dict(), best_path)
                print(f"  🏆 New best model → {best_path}")
        else:
            patience_counter += 1

        # ── Early Stopping ───────────────────────────────────────────
        if es.get("enabled", False) and patience_counter >= es["patience"]:
            print(f"\n⏹  Early stopping triggered at epoch {epoch}.")
            break

    # Save final
    final_path = run_dir / "last_detr.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ DETR training complete.  Final weights → {final_path}")
    return model


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Train DETR-family models (DETR, Conditional DETR, Deformable DETR, DETA, RT-DETR, YOLOS).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model", type=str, help="DETR variant key or HF model ID\n(e.g. detr-resnet-50, conditional-detr-resnet-50, rt-detr-l, yolos-tiny)")
    parser.add_argument("--num_labels", type=int, help="Number of object classes")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--img_size", type=int, help="Input image size")
    parser.add_argument("--device", type=str, help="Device: cuda, cpu, mps, auto")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--list_models", action="store_true", help="List all available DETR variants")
    args = parser.parse_args()

    if args.list_models:
        print_available_models()
        sys.exit(0)

    cfg = load_config(args.config)
    cfg["model_type"] = "detr"

    # CLI overrides
    if args.model:
        cfg["detr_model"] = args.model
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
    init_wandb(cfg, "detr", run_dir)

    model_key = cfg.get("detr_model", "detr-resnet-50")
    num_labels = args.num_labels or cfg.get("num_labels", 91)

    print(f"\n{'='*60}")
    print(f"  DETR-Family Training")
    print(f"  Model : {model_key}")
    print(f"  Epochs: {cfg['epochs']}  │  Batch: {cfg['batch_size']}  │  LR: {cfg['learning_rate']}")
    print(f"{'='*60}\n")

    model, processor, backend = load_detr_model(model_key, num_labels)

    if backend == "ultralytics":
        train_rtdetr(cfg, run_dir, model)
    else:
        train_hf_detr(cfg, run_dir, model, processor)

    finish_wandb()


if __name__ == "__main__":
    main()
