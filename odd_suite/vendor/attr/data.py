"""Existing split CSVs, with task-specific missing-label masks."""
import ast
import csv
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = {
    "weather": ["Sunny", "Cloudy", "Rain", "Snow", "Fog", "Haze", "Dust"],
    "visibility": ["LT50", "50_200", "200_500", "GE500"],
    "lighting": ["Day", "NightLit", "DawnDusk", "NightUnlit"],
    "road_condition": ["Dry", "Wet", "Snow-covered", "Other"],
    "road_material": ["Asphalt", "Concrete", "Cobblestone", "Other"],
}


def encode(row):
    targets, masks = {}, {}
    for task, classes in CLASSES.items():
        raw = row.get(task + "_label", "").strip()
        flag = row.get("has_" + task + "_label", "").strip().lower()
        allowed = flag not in ("false", "0", "no")
        if task == "weather":
            labels = ast.literal_eval(raw) if raw else []
            if not isinstance(labels, list):
                raise ValueError(f"Invalid weather list: {raw}")
            extra = set(labels) - set(classes) - {"Unknown"}
            if extra:
                raise ValueError(f"Unsupported weather: {extra}")
            valid = [x for x in labels if x in classes]
            targets[task] = torch.tensor([float(x in valid) for x in classes])
            masks[task] = allowed and bool(valid)
        else:
            if raw and raw != "Unknown" and raw not in classes:
                raise ValueError(f"Unsupported {task}: {raw}")
            masks[task] = allowed and raw in classes
            targets[task] = torch.tensor(classes.index(raw) if raw in classes else 0)
    return targets, masks


class OddClassificationDataset(Dataset):
    def __init__(self, path, size=224, augment=False):
        self.path = str(path)
        with open(path) as handle:
            self.rows = list(csv.DictReader(handle))
        self.encoded = [encode(row) for row in self.rows]
        ops = [transforms.Resize((size, size))]
        if augment:
            ops.append(transforms.RandomHorizontalFlip())
        # COCO YOLO weights use RGB / 255; no ImageNet mean/std normalization.
        self.transform = transforms.Compose(ops + [transforms.ToTensor()])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        try:
            with Image.open(row["image_path"]) as img:
                image = self.transform(img.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Image read failed: {row['sample_id']} {row['image_path']}") from exc
        targets, masks = self.encoded[index]
        return {"image": image, "targets": targets, "mask": masks,
                "sample_id": row["sample_id"], "dataset": row["dataset"]}

    def audit(self, check_paths=False):
        stats = {"rows": len(self), "datasets": dict(Counter(r["dataset"] for r in self.rows)),
                 "supervised": {}, "label_counts": {}}
        for task, classes in CLASSES.items():
            valid = [(target[task], mask[task]) for target, mask in self.encoded if mask[task]]
            stats["supervised"][task] = len(valid)
            if task == "weather":
                counts = torch.stack([v for v, _ in valid]).sum(0).tolist() if valid else [0] * len(classes)
            else:
                counts = torch.bincount(torch.tensor([v.item() for v, _ in valid]),
                                        minlength=len(classes)).tolist()
            stats["label_counts"][task] = dict(zip(classes, counts))
        if check_paths:
            missing = [r["image_path"] for r in self.rows if not Path(r["image_path"]).is_file()]
            stats["missing_images"] = missing[:10]
            if missing:
                raise FileNotFoundError(f"{len(missing)} missing images, first: {missing[0]}")
        return stats
