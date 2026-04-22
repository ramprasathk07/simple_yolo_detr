"""
export_onnx.py – Export trained YOLO or DETR models to ONNX format.

Usage:
    python export_onnx.py                                        # defaults from config.yaml
    python export_onnx.py --model_type yolo --weights runs/best.pt
    python export_onnx.py --model_type detr --weights runs/best_detr.pth
    python export_onnx.py --opset 17 --dynamic --simplify
"""

import argparse
import sys
from pathlib import Path

from utils import load_config, get_device


# ══════════════════════════════════════════════════════════════════════
#  YOLO EXPORT
# ══════════════════════════════════════════════════════════════════════
def export_yolo(cfg: dict, exp: dict):
    from ultralytics import YOLO

    weights = exp.get("weights") or cfg.get("inference", {}).get("weights", "yolov8n.pt")
    model = YOLO(weights)

    print(f"\n📦 Exporting YOLO → ONNX  (weights: {weights})")

    export_args = dict(
        format="onnx",
        opset=exp.get("opset", 17),
        dynamic=exp.get("dynamic_axes", True),
        simplify=exp.get("simplify", True),
        imgsz=cfg.get("img_size", 640),
    )

    path = model.export(**export_args)
    print(f"✅ YOLO ONNX exported → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
#  DETR EXPORT
# ══════════════════════════════════════════════════════════════════════
def export_detr(cfg: dict, exp: dict):
    import torch
    import onnx
    from transformers import DetrForObjectDetection

    device = get_device(cfg["device"])

    # Load model
    model = DetrForObjectDetection.from_pretrained(
        cfg["detr_model"], ignore_mismatched_sizes=True,
    )

    weights_path = exp.get("weights") or cfg.get("inference", {}).get("weights")
    if weights_path and Path(weights_path).is_file():
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"  📦 Loaded fine-tuned weights: {weights_path}")

    model.eval()

    # Dummy input
    input_shape = exp.get("input_shape", [1, 3, 640, 640])
    dummy = torch.randn(*input_shape)

    # Export path
    out_dir = Path(cfg.get("output_dir", "./runs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "detr_model.onnx"

    # Dynamic axes
    dynamic_axes = None
    if exp.get("dynamic_axes", True):
        dynamic_axes = {
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
        }

    print(f"\n📦 Exporting DETR → ONNX  (opset={exp.get('opset', 17)})")

    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        opset_version=exp.get("opset", 17),
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        dynamic_axes=dynamic_axes,
    )

    # Validate
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # Optional simplify
    if exp.get("simplify", True):
        try:
            import onnxsim
            onnx_model, ok = onnxsim.simplify(onnx_model)
            if ok:
                onnx.save(onnx_model, str(onnx_path))
                print("  🔧 ONNX model simplified.")
        except ImportError:
            print("  ⚠  onnxsim not installed – skipping simplification. pip install onnxsim")

    print(f"✅ DETR ONNX exported → {onnx_path}")
    return onnx_path


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Export YOLO/DETR to ONNX.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model_type", type=str, choices=["yolo", "detr"])
    parser.add_argument("--weights", type=str, help="Path to model weights")
    parser.add_argument("--opset", type=int, help="ONNX opset version")
    parser.add_argument("--dynamic", action="store_true", default=None, help="Dynamic batch axis")
    parser.add_argument("--simplify", action="store_true", default=None, help="Simplify ONNX graph")
    parser.add_argument("--device", type=str, help="Device (for DETR weight loading)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp = dict(cfg.get("export", {}))

    # CLI overrides
    if args.model_type:
        cfg["model_type"] = args.model_type
    if args.device:
        cfg["device"] = args.device
    if args.weights:
        exp["weights"] = args.weights
    if args.opset:
        exp["opset"] = args.opset
    if args.dynamic is not None:
        exp["dynamic_axes"] = args.dynamic
    if args.simplify is not None:
        exp["simplify"] = args.simplify

    model_type = cfg["model_type"]

    if model_type == "yolo":
        export_yolo(cfg, exp)
    elif model_type == "detr":
        export_detr(cfg, exp)
    else:
        print(f"❌ Unknown model_type: {model_type}")
        sys.exit(1)


if __name__ == "__main__":
    main()
