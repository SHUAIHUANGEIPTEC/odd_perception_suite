"""Combined pipeline report rendering (attrs + detections + judgment)."""

from __future__ import annotations

import json
from typing import Any

from odd_judge.report.render import render as render_judgment
from odd_suite.schema import PipelineResult


def pipeline_to_dict(result: PipelineResult) -> dict[str, Any]:
    detections = result.detections or []
    return {
        "attr": (
            {
                "image": result.attr.image,
                "prediction": result.attr.prediction,
                "probabilities": result.attr.probabilities,
                "checkpoint_epoch": result.attr.checkpoint_epoch,
            }
            if result.attr
            else None
        ),
        "detections": {
            "count": len(detections),
            "by_class": _count_by(detections),
        }
        if result.detections is not None
        else None,
        "observation": {
            "timestamp": result.observation.timestamp,
            "source": result.observation.source,
            "fields": result.observation.fields,
        },
        "judgment": result.judgment.to_dict(),
        "debug": result.debug,
    }


def _count_by(detections) -> dict[str, int]:
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.name] = counts.get(det.name, 0) + 1
    return counts


def pipeline_to_json(result: PipelineResult, indent: int = 2) -> str:
    return json.dumps(pipeline_to_dict(result), ensure_ascii=False, indent=indent, default=str)


def pipeline_to_text(result: PipelineResult) -> str:
    lines: list[str] = []
    lines.append("#" * 72)
    lines.append("# ODD Suite Pipeline Report")
    lines.append("#" * 72)

    if result.attr is not None:
        pred = result.attr.prediction
        lines.append("\n[ODD Attribute Classification]")
        lines.append(f"  image: {result.attr.image}")
        for task in ("weather", "visibility", "lighting", "road_condition", "road_material"):
            if task in pred:
                lines.append(f"  {task:<16} {pred[task]}")
    elif result.debug.get("environment_map") is not None:
        lines.append("\n[ODD Attribute Classification]\n  (synthetic override)")

    if result.detections is not None:
        lines.append("\n[3D Detections]")
        counts = _count_by(result.detections)
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"
        lines.append(f"  boxes: {len(result.detections)} ({summary})")
        dyn = result.observation.fields.get("dynamic")
        if dyn:
            lines.append(f"  dynamic: {dyn}")

    if result.debug.get("environment_map") is not None:
        lines.append("\n[Mapped Environment Fields]")
        for key, value in sorted(result.debug["environment_map"].items()):
            lines.append(f"  {key:<18} {value}")

    lines.append("\n" + render_judgment(result.judgment, fmt="text"))
    return "\n".join(lines)


def render_pipeline(result: PipelineResult, fmt: str = "text") -> str:
    if fmt == "json":
        return pipeline_to_json(result)
    return pipeline_to_text(result)
