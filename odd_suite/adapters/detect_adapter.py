"""3D detection sources -> ``dynamic`` / ``sensors`` Observation groups.

Two entry points:
- ``parse_nuscenes_predictions`` — offline nuScenes submission JSON
- ``Sparse4DDetector`` — real vendor inference (heavy env, invoked via subprocess)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from odd_judge.core.types import Observation
from odd_judge.perception.base import ObservationAdapter
from odd_suite.errors import MissingDependencyError, SuiteError
from odd_suite.mapping.dynamic_rollup import DynamicRollup
from odd_suite.mapping.env_mapper import MappingTables
from odd_suite.mapping.sensor_defaults import SensorDefaults
from odd_suite.paths import vendor_sparse4d_dir
from odd_suite.schema import Detection3D


def parse_nuscenes_predictions(path: Path) -> dict[str, list[Detection3D]]:
    """Parse a nuScenes submission JSON (``{"results": {token: [records]}}``).

    A plain list of records is also accepted and grouped by ``sample_token``.
    """
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("results", data) if isinstance(data, dict) else data
    if isinstance(records, list):
        grouped: dict[str, list[Detection3D]] = {}
        for rec in records:
            det = Detection3D.from_nuscenes_record(rec)
            grouped.setdefault(det.sample_token, []).append(det)
        return grouped
    if not isinstance(records, dict):
        raise SuiteError(f"Unsupported prediction JSON layout in {path}")
    return {
        token: [Detection3D.from_nuscenes_record(rec) for rec in recs]
        for token, recs in records.items()
    }


class NuScenesDetAdapter(ObservationAdapter):
    """Roll detection boxes up into ``dynamic`` and ``sensors`` groups."""

    def __init__(
        self,
        detections: list[Detection3D] | None = None,
        *,
        tables: MappingTables | None = None,
        score_threshold: float = 0.0,
    ):
        self._tables = tables or MappingTables.load()
        self._rollup = DynamicRollup(self._tables, score_threshold)
        self._sensors = SensorDefaults(self._tables)
        self.detections: list[Detection3D] = list(detections or [])

    def set_detections(self, detections: list[Detection3D]) -> "NuScenesDetAdapter":
        self.detections = list(detections)
        return self

    def get_observation(self) -> Observation:
        fields = {
            "dynamic": self._rollup.map(self.detections),
            "sensors": self._sensors.map(self.detections or None),
        }
        return Observation(fields=fields, source="sparse4d")

    def debug(self) -> dict:
        return self._rollup.debug(self.detections)


class Sparse4DDetector:
    """Facade over the vendored Sparse4D runtime.

    The vendor tree runs whole-split nuScenes inference; scene/sample filtering
    is done on the produced prediction JSON. Inference is executed as a
    subprocess (vendor ``run.py``) inside the vendored directory so Hydra and
    the compiled CUDA ops resolve their paths.
    """

    def __init__(
        self,
        ckpt: Path,
        data_root: str | Path | None = None,
        *,
        results_dir: Path | None = None,
        vendor_dir: Path | None = None,
        python: str | None = None,
    ):
        self.ckpt = Path(ckpt)
        self.data_root = str(data_root) if data_root else os.environ.get("NUSCENES_DATA_DIR", "")
        self.vendor_dir = Path(vendor_dir) if vendor_dir else vendor_sparse4d_dir()
        self.results_dir = Path(results_dir) if results_dir else self.vendor_dir / "results"
        self.python = python or sys.executable
        self._cache: dict[str, list[Detection3D]] | None = None

    def _check_env(self) -> None:
        if not self.vendor_dir.is_dir():
            raise MissingDependencyError("vendor/sparse4d", "detect")
        if not self.ckpt.exists():
            raise SuiteError(f"Detection checkpoint not found: {self.ckpt}")
        if not self.data_root:
            raise SuiteError(
                "NUSCENES_DATA_DIR is not set; real Sparse4D inference needs the "
                "nuScenes data root (see vendor/sparse4d/VENDOR.md)"
            )

    def run_inference(self) -> Path:
        """Run vendor val-split inference and return the prediction JSON path."""
        self._check_env()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(
            {
                "PROJECT_ROOT": str(self.vendor_dir),
                "OUTPUT_DIR": str(self.vendor_dir / "outputs"),
                "CKPT_DIR": str(self.vendor_dir / "weights"),
                "SPARSE4D_ANCHOR_PATH": str(
                    self.vendor_dir / "nuscenes_kmeans900_bevvae.npy"
                ),
                "PYTHONPATH": str(self.vendor_dir)
                + os.pathsep
                + env.get("PYTHONPATH", ""),
            }
        )
        cmd = [
            self.python,
            str(self.vendor_dir / "run.py"),
            "infer",
            "--ckpt-path",
            str(self.ckpt),
            "--data-root",
            self.data_root,
            "--results-dir",
            str(self.results_dir),
        ]
        subprocess.run(cmd, cwd=self.vendor_dir, env=env, check=True)
        files = sorted(self.results_dir.glob("*val_results.json"))
        if not files:
            raise SuiteError(f"No *val_results.json produced under {self.results_dir}")
        return files[-1]

    def detect(
        self, *, sample_token: str | None = None, prediction_json: Path | None = None
    ) -> list[Detection3D]:
        if prediction_json is not None:
            grouped = parse_nuscenes_predictions(prediction_json)
        else:
            if self._cache is None:
                self._cache = parse_nuscenes_predictions(self.run_inference())
            grouped = self._cache
        if sample_token is not None:
            return list(grouped.get(sample_token, []))
        merged: list[Detection3D] = []
        for boxes in grouped.values():
            merged.extend(boxes)
        return merged

    def sample_tokens_with_boxes(self, prediction_json: Path | None = None) -> list[str]:
        if prediction_json is not None:
            grouped = parse_nuscenes_predictions(prediction_json)
        else:
            if self._cache is None:
                self._cache = parse_nuscenes_predictions(self.run_inference())
            grouped = self._cache
        return [token for token, boxes in grouped.items() if boxes]
