"""Attribute classifier smoke test — only runs when torch is installed."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from odd_suite.paths import weights_dir  # noqa: E402
from odd_suite.vendor.attr.checkpoint import AttrClassifier  # noqa: E402
from odd_suite.vendor.attr.classes import CLASSES  # noqa: E402


CHECKPOINT = weights_dir() / "attr_best.pth"


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="attr_best.pth not present")
def test_checkpoint_loads_and_predicts(tmp_path):
    PIL = pytest.importorskip("PIL")
    image = tmp_path / "probe.png"
    PIL.Image.new("RGB", (224, 224), (120, 130, 150)).save(image)

    classifier = AttrClassifier(CHECKPOINT, device="cpu")
    result = classifier.predict(image)
    assert set(result.prediction) == set(CLASSES)
    assert isinstance(result.prediction["weather"], list)
    for task, classes in CLASSES.items():
        assert set(result.probabilities[task]) == set(classes)
        assert all(0.0 <= p <= 1.0 for p in result.probabilities[task].values())


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="attr_best.pth not present")
def test_classes_module_matches_verbatim_data_module():
    try:
        from odd_suite.vendor.attr.data import CLASSES as DATA_CLASSES
    except ImportError:
        pytest.skip("vendored data.py needs torchvision")
    assert CLASSES == DATA_CLASSES
