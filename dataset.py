"""
dataset.py – Unified dataset loader for YOLO & DETR.

Supports three dataset sources (set via config.yaml → dataset.source):
  1. "huggingface"  – any HF detection dataset (e.g. "cppe-5", "detection-datasets/coco")
  2. "coco"         – local COCO-JSON format  (images/ + annotations/*.json)
  3. "yolo"         – local YOLO-folder format (images/ + labels/ with .txt per image)

For YOLO model training, datasets are auto-converted to Ultralytics data.yaml.
For DETR model training, datasets produce (pixel_values, target) tuples.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════
#  1. HUGGINGFACE DATASET HELPERS
# ══════════════════════════════════════════════════════════════════════

def load_hf_dataset(
    name: str,
    split: str = "train",
    subset: Optional[str] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = True,
):
    """Load a HuggingFace dataset by name. Returns an HF Dataset object."""
    from datasets import load_dataset

    kwargs = dict(trust_remote_code=trust_remote_code)
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if subset:
        ds = load_dataset(name, subset, split=split, **kwargs)
    else:
        ds = load_dataset(name, split=split, **kwargs)

    print(f"  🤗 Loaded HF dataset '{name}' split='{split}' → {len(ds)} samples")
    return ds


def hf_to_yolo_folder(
    hf_ds,
    output_dir: str,
    split_name: str = "train",
    image_col: str = "image",
    objects_col: str = "objects",
    bbox_col: str = "bbox",
    category_col: str = "category",
    categories: Optional[list] = None,
    bbox_format: str = "coco",
):
    """
    Convert a HuggingFace detection dataset to YOLO folder format.

    Supported bbox_format values:
        "coco"       → [x_min, y_min, width, height]  (absolute pixels)
        "pascal_voc" → [x_min, y_min, x_max, y_max]   (absolute pixels)
        "yolo"       → [cx, cy, w, h]                  (normalised 0-1)

    Output structure:
        output_dir/
        ├── images/<split_name>/
        └── labels/<split_name>/
    """
    img_dir = Path(output_dir) / "images" / split_name
    lbl_dir = Path(output_dir) / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    cat_set = set()

    for idx, sample in enumerate(tqdm(hf_ds, desc=f"Converting {split_name}")):
        # ── Save image ───────────────────────────────────────────────
        img = sample[image_col]
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")

        img_name = f"{idx:06d}.jpg"
        img.save(img_dir / img_name, quality=95)

        w, h = img.size

        # ── Parse annotations ────────────────────────────────────────
        objects = sample.get(objects_col, sample)  # some datasets are flat
        bboxes = objects[bbox_col] if isinstance(objects, dict) else sample[bbox_col]
        cats = objects[category_col] if isinstance(objects, dict) else sample[category_col]

        lines = []
        for box, cat in zip(bboxes, cats):
            cat_set.add(int(cat))

            if bbox_format == "coco":
                # [x_min, y_min, width, height] → YOLO [cx, cy, w, h] normalised
                x_min, y_min, bw, bh = box
                cx = (x_min + bw / 2) / w
                cy = (y_min + bh / 2) / h
                nw = bw / w
                nh = bh / h
            elif bbox_format == "pascal_voc":
                x_min, y_min, x_max, y_max = box
                cx = ((x_min + x_max) / 2) / w
                cy = ((y_min + y_max) / 2) / h
                nw = (x_max - x_min) / w
                nh = (y_max - y_min) / h
            elif bbox_format == "yolo":
                cx, cy, nw, nh = box
            else:
                raise ValueError(f"Unknown bbox_format: {bbox_format}")

            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            lines.append(f"{int(cat)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        lbl_path = lbl_dir / img_name.replace(".jpg", ".txt")
        lbl_path.write_text("\n".join(lines))

    # ── Build class names ────────────────────────────────────────────
    if categories:
        class_names = {i: n for i, n in enumerate(categories)}
    else:
        class_names = {c: str(c) for c in sorted(cat_set)}

    print(f"  ✅ Converted {len(hf_ds)} images → {img_dir.parent.parent}")
    return class_names


def hf_to_data_yaml(
    hf_name: str,
    output_dir: str = "./data_hf",
    subset: Optional[str] = None,
    train_split: str = "train",
    val_split: str = "validation",
    image_col: str = "image",
    objects_col: str = "objects",
    bbox_col: str = "bbox",
    category_col: str = "category",
    bbox_format: str = "coco",
    cache_dir: Optional[str] = None,
) -> str:
    """
    Full pipeline: HuggingFace dataset → YOLO folder + data.yaml.
    Returns the path to the generated data.yaml.
    """
    print(f"\n📥 Downloading & converting HF dataset: {hf_name}")

    # Load splits
    train_ds = load_hf_dataset(hf_name, split=train_split, subset=subset, cache_dir=cache_dir)

    try:
        val_ds = load_hf_dataset(hf_name, split=val_split, subset=subset, cache_dir=cache_dir)
    except Exception:
        # Some datasets use "test" instead of "validation"
        try:
            val_ds = load_hf_dataset(hf_name, split="test", subset=subset, cache_dir=cache_dir)
            val_split = "test"
        except Exception:
            print("  ⚠  No validation split found – using last 20% of train as val.")
            split = train_ds.train_test_split(test_size=0.2, seed=42)
            train_ds = split["train"]
            val_ds = split["test"]
            val_split = "val"

    # Try to get category names from dataset features
    categories = None
    try:
        feat = train_ds.features
        if objects_col in feat:
            inner = feat[objects_col]
            if hasattr(inner, "feature") and category_col in inner.feature:
                cat_feat = inner.feature[category_col]
                if hasattr(cat_feat, "names"):
                    categories = cat_feat.names
                    print(f"  📋 Found {len(categories)} class names from HF features")
    except Exception:
        pass

    # Convert both splits
    conv_kwargs = dict(
        image_col=image_col,
        objects_col=objects_col,
        bbox_col=bbox_col,
        category_col=category_col,
        categories=categories,
        bbox_format=bbox_format,
    )

    class_names = hf_to_yolo_folder(train_ds, output_dir, "train", **conv_kwargs)
    hf_to_yolo_folder(val_ds, output_dir, "val", **conv_kwargs)

    # Use discovered categories if available
    if categories:
        class_names = {i: n for i, n in enumerate(categories)}

    # Write data.yaml
    yaml_path = Path(output_dir) / "data.yaml"
    yaml_content = (
        f"path: {Path(output_dir).resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"nc: {len(class_names)}\n"
        f"names:\n"
    )
    for idx in sorted(class_names.keys()):
        yaml_content += f"  {idx}: {class_names[idx]}\n"

    yaml_path.write_text(yaml_content)
    print(f"  📄 data.yaml → {yaml_path}")
    return str(yaml_path)


# ══════════════════════════════════════════════════════════════════════
#  2. COCO-JSON DATASET (for DETR)
# ══════════════════════════════════════════════════════════════════════

class CocoDetectionDataset(Dataset):
    """
    Loads images + COCO-JSON annotations for HuggingFace DETR.
    """

    def __init__(self, img_dir: str, ann_file: str, processor):
        from pycocotools.coco import COCO

        self.img_dir = img_dir
        self.processor = processor
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        cat_ids = sorted(self.coco.getCatIds())
        self.cat2label = {c: i for i, c in enumerate(cat_ids)}
        self.num_classes = len(cat_ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        w, h = image.size
        boxes, labels = [], []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            boxes.append([cx, cy, bw / w, bh / h])
            labels.append(self.cat2label[ann["category_id"]])

        target = {
            "class_labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
        }

        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)

        return pixel_values, target


# ══════════════════════════════════════════════════════════════════════
#  3. YOLO-FOLDER DATASET (for DETR) – images/ + labels/
# ══════════════════════════════════════════════════════════════════════

class YoloFolderDataset(Dataset):
    """
    Reads YOLO-format folder:
        images/<split>/  → image files
        labels/<split>/  → .txt files with: <class> <cx> <cy> <w> <h>  (normalised)

    Returns (pixel_values, target) compatible with DETR training.
    """

    def __init__(self, img_dir: str, label_dir: str, processor):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.processor = processor

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        self.image_files = sorted(
            [f for f in self.img_dir.iterdir() if f.suffix.lower() in exts]
        )
        print(f"  📂 YoloFolderDataset: {len(self.image_files)} images from {img_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        # Find matching label file
        lbl_path = self.label_dir / (img_path.stem + ".txt")

        boxes, labels = [], []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                boxes.append([cx, cy, bw, bh])
                labels.append(cls_id)

        target = {
            "class_labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
        }

        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)

        return pixel_values, target


# ══════════════════════════════════════════════════════════════════════
#  4. HUGGINGFACE DATASET → DETR  (direct, no folder conversion)
# ══════════════════════════════════════════════════════════════════════

class HFDetectionDataset(Dataset):
    """
    Wraps a HuggingFace detection dataset for direct DETR training
    without needing to save to disk first.
    """

    def __init__(
        self,
        hf_dataset,
        processor,
        image_col: str = "image",
        objects_col: str = "objects",
        bbox_col: str = "bbox",
        category_col: str = "category",
        bbox_format: str = "coco",
    ):
        self.ds = hf_dataset
        self.processor = processor
        self.image_col = image_col
        self.objects_col = objects_col
        self.bbox_col = bbox_col
        self.category_col = category_col
        self.bbox_format = bbox_format

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]

        img = sample[self.image_col]
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        img = img.convert("RGB")
        w, h = img.size

        objects = sample.get(self.objects_col, sample)
        bboxes = objects[self.bbox_col] if isinstance(objects, dict) else sample[self.bbox_col]
        cats = objects[self.category_col] if isinstance(objects, dict) else sample[self.category_col]

        boxes, labels = [], []
        for box, cat in zip(bboxes, cats):
            if self.bbox_format == "coco":
                x_min, y_min, bw, bh = box
                cx = (x_min + bw / 2) / w
                cy = (y_min + bh / 2) / h
                nw, nh = bw / w, bh / h
            elif self.bbox_format == "pascal_voc":
                x_min, y_min, x_max, y_max = box
                cx = ((x_min + x_max) / 2) / w
                cy = ((y_min + y_max) / 2) / h
                nw = (x_max - x_min) / w
                nh = (y_max - y_min) / h
            elif self.bbox_format == "yolo":
                cx, cy, nw, nh = box
            else:
                raise ValueError(f"Unknown bbox_format: {self.bbox_format}")

            boxes.append([
                max(0, min(1, cx)), max(0, min(1, cy)),
                max(0, min(1, nw)), max(0, min(1, nh)),
            ])
            labels.append(int(cat))

        target = {
            "class_labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
        }

        encoding = self.processor(images=img, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)

        return pixel_values, target


# ══════════════════════════════════════════════════════════════════════
#  5. COLLATE FN
# ══════════════════════════════════════════════════════════════════════

def detr_collate_fn(batch):
    """Custom collate: stack pixel_values, keep labels as list of dicts."""
    pixel_values = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}


# ══════════════════════════════════════════════════════════════════════
#  6. BUILDER – pick the right dataset from config
# ══════════════════════════════════════════════════════════════════════

def build_detr_datasets(cfg: dict, processor):
    """
    Reads cfg["dataset"] and returns (train_dataset, val_dataset).
    Both return (pixel_values, target) tuples for DETR.
    """
    ds_cfg = cfg.get("dataset", {})
    source = ds_cfg.get("source", "coco")

    if source == "huggingface":
        hf_name = ds_cfg["name"]
        subset = ds_cfg.get("subset")
        train_split = ds_cfg.get("train_split", "train")
        val_split = ds_cfg.get("val_split", "validation")
        cache_dir = ds_cfg.get("cache_dir")

        train_hf = load_hf_dataset(hf_name, train_split, subset=subset, cache_dir=cache_dir)

        try:
            val_hf = load_hf_dataset(hf_name, val_split, subset=subset, cache_dir=cache_dir)
        except Exception:
            try:
                val_hf = load_hf_dataset(hf_name, "test", subset=subset, cache_dir=cache_dir)
            except Exception:
                split = train_hf.train_test_split(test_size=0.2, seed=42)
                train_hf, val_hf = split["train"], split["test"]

        common = dict(
            processor=processor,
            image_col=ds_cfg.get("image_col", "image"),
            objects_col=ds_cfg.get("objects_col", "objects"),
            bbox_col=ds_cfg.get("bbox_col", "bbox"),
            category_col=ds_cfg.get("category_col", "category"),
            bbox_format=ds_cfg.get("bbox_format", "coco"),
        )
        return HFDetectionDataset(train_hf, **common), HFDetectionDataset(val_hf, **common)

    elif source == "coco":
        img_root = ds_cfg.get("root", cfg.get("detr_data_root", "./data"))
        train_ann = ds_cfg.get("train_ann", cfg.get("detr_train_ann"))
        val_ann = ds_cfg.get("val_ann", cfg.get("detr_val_ann"))

        train_ds = CocoDetectionDataset(f"{img_root}/train", train_ann, processor)
        val_ds = CocoDetectionDataset(f"{img_root}/val", val_ann, processor)
        return train_ds, val_ds

    elif source == "yolo":
        root = ds_cfg.get("root", "./data")
        train_ds = YoloFolderDataset(
            f"{root}/images/train", f"{root}/labels/train", processor,
        )
        val_ds = YoloFolderDataset(
            f"{root}/images/val", f"{root}/labels/val", processor,
        )
        return train_ds, val_ds

    else:
        raise ValueError(
            f"Unknown dataset source: '{source}'. Use 'huggingface', 'coco', or 'yolo'."
        )


def build_yolo_data_yaml(cfg: dict) -> str:
    """
    For YOLO training: resolve the data.yaml path.
    If source is HuggingFace, auto-download & convert first.
    If source is yolo/coco folder, return the data.yaml path.
    """
    ds_cfg = cfg.get("dataset", {})
    source = ds_cfg.get("source", "manual")

    if source == "huggingface":
        output_dir = ds_cfg.get("output_dir", "./data_hf")
        yaml_path = Path(output_dir) / "data.yaml"

        if yaml_path.exists() and not ds_cfg.get("force_redownload", False):
            print(f"  ♻  Using cached HF dataset: {yaml_path}")
            return str(yaml_path)

        return hf_to_data_yaml(
            hf_name=ds_cfg["name"],
            output_dir=output_dir,
            subset=ds_cfg.get("subset"),
            train_split=ds_cfg.get("train_split", "train"),
            val_split=ds_cfg.get("val_split", "validation"),
            image_col=ds_cfg.get("image_col", "image"),
            objects_col=ds_cfg.get("objects_col", "objects"),
            bbox_col=ds_cfg.get("bbox_col", "bbox"),
            category_col=ds_cfg.get("category_col", "category"),
            bbox_format=ds_cfg.get("bbox_format", "coco"),
            cache_dir=ds_cfg.get("cache_dir"),
        )

    else:
        # Manual dataset – just return the configured data.yaml path
        return cfg.get("yolo_data", "data.yaml")
