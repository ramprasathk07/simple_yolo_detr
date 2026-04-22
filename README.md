# Object Detection – YOLO & DETR

Minimal, unified training & inference for **YOLO** (v3–v11) and **DETR-family** (DETR, Conditional DETR, Deformable DETR, DETA, RT-DETR, YOLOS).

## Project Structure

```
.
├── config.yaml        # Single config – models, dataset, training, export
├── requirements.txt   # Python dependencies
├── utils.py           # Shared helpers (device, W&B, config)
├── dataset.py         # Unified dataset: HuggingFace / COCO / YOLO-folder
├── train.py           # Dispatcher → auto-routes to yolo or detr
├── train_yolo.py      # YOLO training (v3/v5/v6/v8/v9/v10/v11)
├── train_detr.py      # DETR-family training (DETR/Cond/Deform/DETA/RT/YOLOS)
├── inference.py       # Inference with visualization
├── export_onnx.py     # Export to ONNX
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt

# Train (uses config.yaml)
python train.py

# Or call specific scripts directly
python train_yolo.py --model yolov8n --epochs 50
python train_detr.py --model detr-resnet-50 --epochs 50
```

---

## Supported Models
             
### YOLO Versions

```bash
python train_yolo.py --list_models    # see all options
```

| Family | Variants |
|--------|----------|
| YOLOv3 | `yolov3`, `yolov3-tiny` |
| YOLOv5 | `yolov5n`, `yolov5s`, `yolov5m`, `yolov5l`, `yolov5x` |
| YOLOv6 | `yolov6n`, `yolov6s`, `yolov6m`, `yolov6l` |
| YOLOv8 | `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x` |
| YOLOv9 | `yolov9t`, `yolov9s`, `yolov9m`, `yolov9c`, `yolov9e` |
| YOLOv10 | `yolov10n`, `yolov10s`, `yolov10m`, `yolov10l`, `yolov10x` |
| YOLO11 | `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x` |

```bash
python train_yolo.py --model yolov10n --epochs 100 --device cuda
python train_yolo.py --model yolov5s --batch_size 32
python train_yolo.py --model /path/to/custom.pt
```

### DETR-Family Variants

```bash
python train_detr.py --list_models    # see all options
```

| Model | Key | Backend |
|-------|-----|---------|
| DETR | `detr-resnet-50`, `detr-resnet-101` | HuggingFace |
| Conditional DETR | `conditional-detr-resnet-50` | HuggingFace |
| Deformable DETR | `deformable-detr` | HuggingFace |
| DETA | `deta-swin-large` | HuggingFace |
| RT-DETR | `rt-detr-l`, `rt-detr-x` | Ultralytics |
| YOLOS | `yolos-tiny`, `yolos-small`, `yolos-base` | HuggingFace |

```bash
python train_detr.py --model conditional-detr-resnet-50 --epochs 50
python train_detr.py --model rt-detr-l --device cuda
python train_detr.py --model yolos-tiny --num_labels 5
```

---

## Dataset Options

Edit `config.yaml → dataset:` section. Three sources supported:

### Option 1 — HuggingFace (auto-download)

```yaml
dataset:
  source: "huggingface"
  name: "cppe-5"              # any HF detection dataset
  bbox_format: "coco"         # coco | pascal_voc | yolo
```

### Option 2 — COCO-JSON (local)

```yaml
dataset:
  source: "coco"
  root: "./data"
  train_ann: "./data/annotations/instances_train.json"
  val_ann: "./data/annotations/instances_val.json"
```

### Option 3 — YOLO Folder (local)

```yaml
dataset:
  source: "yolo"
  root: "./data"              # images/{train,val} + labels/{train,v al}
```

---

##  Training

### Unified dispatcher

```bash
python train.py --model yolov8n             # auto-detects → YOLO
python train.py --model detr-resnet-50      # auto-detects → DETR
python train.py --model_type yolo --epochs 100
```

### Direct scripts

```bash
# YOLO
python train_yolo.py --model yolo11n --epochs 50 --batch_size 16 --device cuda

# DETR
python train_detr.py --model deformable-detr --epochs 100 --lr 0.0001 --no_wandb
```

### Common flags

| Flag | Description |
|------|-------------|
| `--model` | Model version key or weights path |
| `--epochs` | Number of epochs |
| `--batch_size` | Batch size |
| `--lr` | Learning rate |
| `--img_size` | Input image size |
| `--device` | `cuda`, `cpu`, `mps`, `auto` |
| `--no_wandb` | Disable W&B |
| `--list_models` | Show available model versions |

##   Inference

```bash
python inference.py --model_type yolo --weights runs/best.pt --source ./images/
python inference.py --model_type detr --weights runs/best_detr.pth --device cpu
```

## ONNX Export

```bash
python export_onnx.py --model_type yolo --weights runs/best.pt
python export_onnx.py --model_type detr --weights runs/best_detr.pth --opset 17
```

## W&B

```yaml
wandb:
  enabled: true
  project: "my-project"
```

Disable: `python train.py --no_wandb`
