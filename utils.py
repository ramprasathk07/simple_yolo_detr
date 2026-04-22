"""
utils.py – Shared helpers for config loading, device selection, and W&B init.
"""

import os
import yaml
import torch
import wandb
from pathlib import Path
from datetime import datetime


# ─── Config ──────────────────────────────────────────────────────────
def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config and return as dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ─── Device ──────────────────────────────────────────────────────────
def get_device(device_str: str = "auto") -> torch.device:
    """Resolve device string → torch.device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device_str)
    print(f"⚙  Device → {dev}")
    return dev


# ─── Output dir ──────────────────────────────────────────────────────
def make_run_dir(base: str = "./runs") -> Path:
    """Create a timestamped run directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 Run dir → {run_dir}")
    return run_dir


# ─── Weights & Biases ───────────────────────────────────────────────
def init_wandb(cfg: dict, model_type: str, run_dir: Path):
    """Initialise W&B run if enabled in config."""
    wb = cfg.get("wandb", {})
    if not wb.get("enabled", False):
        print("📊 W&B disabled – skipping init.")
        return None

    run = wandb.init(
        project=wb.get("project", "object-detection"),
        entity=wb.get("entity"),
        name=wb.get("run_name") or f"{model_type}_{run_dir.name}",
        tags=wb.get("tags", []) + [model_type],
        config={
            "model_type": model_type,
            "epochs": cfg["epochs"],
            "batch_size": cfg["batch_size"],
            "lr": cfg["learning_rate"],
            "img_size": cfg["img_size"],
        },
        dir=str(run_dir),
    )
    print(f"📊 W&B run → {run.url}")
    return run


def log_wandb(metrics: dict, step: int | None = None):
    """Log metrics to W&B (no-op if W&B not active)."""
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def finish_wandb():
    """Finish W&B run cleanly."""
    if wandb.run is not None:
        wandb.finish()
