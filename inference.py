"""
inference.py – Unified inference for YOLO & DETR with visualization.

Usage:
    python inference.py                                          # defaults from config.yaml
    python inference.py --model_type yolo --weights runs/best.pt --source test.jpg
    python inference.py --model_type detr --weights runs/best_detr.pth --source ./images/
    python inference.py --device cpu --conf 0.5
"""

import argparse
import os
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from utils import load_config, get_device


# ── Colour palette for drawing ───────────────────────────────────────
COLORS = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
    "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
    "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC",
    "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#FF37C7",
]


def draw_boxes(image: Image.Image, boxes, labels, scores, class_names=None):
    """Draw bounding boxes on a PIL image and return it."""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        color = COLORS[int(label) % len(COLORS)]
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        name = class_names[int(label)] if class_names else str(int(label))
        text = f"{name} {score:.2f}"
        draw.text((x1 + 2, y1 - 16), text, fill=color, font=font)
    return image


# ══════════════════════════════════════════════════════════════════════
#  YOLO INFERENCE
# ══════════════════════════════════════════════════════════════════════
def infer_yolo(cfg: dict, inf: dict):
    from ultralytics import YOLO

    device = get_device(cfg["device"])
    model = YOLO(inf["weights"])

    source = inf["source"]
    print(f"\n🔍 YOLO inference on: {source}")

    results = model.predict(
        source=source,
        device=str(device),
        conf=inf["conf_threshold"],
        iou=inf["iou_threshold"],
        save=inf["save_results"],
        show=inf["show_results"],
        project=cfg.get("output_dir", "./runs"),
        name="inference",
        exist_ok=True,
    )

    for r in results:
        print(f"  📸 {Path(r.path).name}: {len(r.boxes)} detections")

    print("✅ YOLO inference complete.")
    return results


# ══════════════════════════════════════════════════════════════════════
#  DETR INFERENCE
# ══════════════════════════════════════════════════════════════════════
def infer_detr(cfg: dict, inf: dict):
    import torch
    from transformers import DetrForObjectDetection, DetrImageProcessor

    device = get_device(cfg["device"])

    processor = DetrImageProcessor.from_pretrained(cfg["detr_model"])
    model = DetrForObjectDetection.from_pretrained(
        cfg["detr_model"], ignore_mismatched_sizes=True,
    )

    # Load fine-tuned weights if available
    weights_path = inf["weights"]
    if os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        print(f"  📦 Loaded weights: {weights_path}")

    model.to(device).eval()

    # Gather images
    source = Path(inf["source"])
    if source.is_file():
        image_paths = [source]
    elif source.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted([p for p in source.iterdir() if p.suffix.lower() in exts])
    else:
        raise FileNotFoundError(f"Source not found: {source}")

    out_dir = Path(cfg.get("output_dir", "./runs")) / "inference_detr"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔍 DETR inference on {len(image_paths)} images …\n")

    for img_path in tqdm(image_paths, desc="Inference"):
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=device)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=inf["conf_threshold"]
        )[0]

        boxes = results["boxes"].cpu().tolist()
        labels = results["labels"].cpu().tolist()
        scores = results["scores"].cpu().tolist()

        print(f"  📸 {img_path.name}: {len(boxes)} detections")

        if inf["save_results"]:
            annotated = draw_boxes(image.copy(), boxes, labels, scores)
            annotated.save(out_dir / img_path.name)

    print(f"\n✅ DETR inference complete.  Results → {out_dir}")


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Run inference with YOLO or DETR.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model_type", type=str, choices=["yolo", "detr"])
    parser.add_argument("--weights", type=str, help="Path to model weights")
    parser.add_argument("--source", type=str, help="Image file/dir to run on")
    parser.add_argument("--device", type=str, help="Force device (cuda, cpu, mps)")
    parser.add_argument("--conf", type=float, help="Confidence threshold")
    parser.add_argument("--iou", type=float, help="IoU threshold (YOLO only)")
    parser.add_argument("--save", action="store_true", default=None, help="Save annotated results")
    parser.add_argument("--show", action="store_true", default=None, help="Display results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    inf = dict(cfg.get("inference", {}))

    # CLI overrides
    if args.model_type:
        cfg["model_type"] = args.model_type
    if args.device:
        cfg["device"] = args.device
    if args.weights:
        inf["weights"] = args.weights
    if args.source:
        inf["source"] = args.source
    if args.conf is not None:
        inf["conf_threshold"] = args.conf
    if args.iou is not None:
        inf["iou_threshold"] = args.iou
    if args.save is not None:
        inf["save_results"] = args.save
    if args.show is not None:
        inf["show_results"] = args.show

    model_type = cfg["model_type"]

    if model_type == "yolo":
        infer_yolo(cfg, inf)
    elif model_type == "detr":
        infer_detr(cfg, inf)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    main()
