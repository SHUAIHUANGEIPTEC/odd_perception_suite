"""Checkpoint loading and inference recipe for the ODD attribute classifier.

Contract preserved from odd_multitask_yolo/predict.py:
- custom checkpoint dict with ``model`` / ``classes`` / ``config`` / ``epoch``
- strict ``load_state_dict``; class mapping must match exactly
- RGB -> resize(image_size, image_size) -> ToTensor [0,1], no mean/std
- weather: independent sigmoids (threshold 0.5); other tasks: softmax argmax
- BF16 autocast on CUDA when supported
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

from odd_suite.errors import MissingDependencyError, SuiteError
from odd_suite.schema import AttrResult
from odd_suite.vendor.attr.classes import CLASSES


def _require_torch():
    try:
        import torch
        from torchvision import transforms
    except ImportError as e:  # pragma: no cover - exercised only without torch
        raise MissingDependencyError("torch", "attr") from e
    return torch, transforms


def load_checkpoint(path: Path, device: str = "cpu"):
    torch, _ = _require_torch()
    from PIL import Image  # noqa: F401  (import check)
    from odd_suite.vendor.attr.model import MultiTaskYOLO

    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if state["classes"] != CLASSES:
        raise SuiteError("Checkpoint class mapping mismatch")
    model = MultiTaskYOLO(CLASSES)
    model.load_state_dict(state["model"], strict=True)
    model = model.to(torch.device(device)).eval()
    return model, state


def preprocess(image_path: Path, size: int):
    torch, transforms = _require_torch()
    from PIL import Image

    transform = transforms.Compose(
        [transforms.Resize((size, size)), transforms.ToTensor()]
    )
    with Image.open(image_path) as img:
        return transform(img.convert("RGB")).unsqueeze(0)


def predict_probs(model, x, device: str, weather_threshold: float = 0.5) -> dict[str, Any]:
    torch, _ = _require_torch()
    dev = torch.device(device)
    x = x.to(dev)
    precision = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if dev.type == "cuda" and torch.cuda.is_bf16_supported()
        else nullcontext()
    )
    with torch.inference_mode(), precision:
        outputs = model(x)
    prediction: dict[str, Any] = {}
    probabilities: dict[str, dict[str, float]] = {}
    for task, classes in CLASSES.items():
        logits = outputs[task][0].float()
        probs = (logits.sigmoid() if task == "weather" else logits.softmax(0)).cpu()
        probabilities[task] = dict(zip(classes, (float(p) for p in probs.tolist())))
        if task == "weather":
            prediction[task] = [
                c for i, c in enumerate(classes) if float(probs[i]) > weather_threshold
            ]
        else:
            prediction[task] = classes[int(probs.argmax())]
    return {"prediction": prediction, "probabilities": probabilities}


class AttrClassifier:
    """Load once, predict per image. Torch imported lazily at construction."""

    def __init__(self, checkpoint: Path, device: str = "cpu", weather_threshold: float = 0.5):
        self.model, self.state = load_checkpoint(checkpoint, device)
        self.device = device
        self.weather_threshold = weather_threshold
        self.image_size = int(self.state["config"]["image_size"])
        self.epoch = self.state.get("epoch")

    def predict(self, image: Path) -> AttrResult:
        x = preprocess(Path(image), self.image_size)
        out = predict_probs(self.model, x, self.device, self.weather_threshold)
        return AttrResult(
            image=str(Path(image).resolve()),
            prediction=out["prediction"],
            probabilities=out["probabilities"],
            checkpoint_epoch=self.epoch,
        )
