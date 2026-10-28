"""Boundary margin helpers and numeric state classification."""

from __future__ import annotations

from typing import Optional

from .types import BoundaryMargin, ScenarioState, ThresholdSpec


def classify_numeric(
    value: float,
    spec: ThresholdSpec,
) -> tuple[ScenarioState, Optional[float], Optional[float]]:
    """Classify a numeric observation against a threshold spec.

    Returns (state, remaining_margin, margin_band).
    remaining_margin is signed distance to the nearest threshold
    (positive = inside, negative = outside).
    """
    op = spec.op
    margin = spec.margin or BoundaryMargin(mode="absolute", value=0.0)

    if op in (">=", ">") and spec.threshold is not None:
        thr = float(spec.threshold)
        band = margin.band_for_threshold(thr)
        remaining = value - thr
        if op == ">=":
            if value >= thr + band:
                return ScenarioState.INSIDE, remaining, band
            if value >= thr:
                return ScenarioState.BOUNDARY, remaining, band
            return ScenarioState.OUTSIDE, remaining, band
        # op == ">"
        if value > thr + band:
            return ScenarioState.INSIDE, remaining, band
        if value > thr:
            return ScenarioState.BOUNDARY, remaining, band
        return ScenarioState.OUTSIDE, remaining, band

    if op in ("<=", "<") and spec.threshold is not None:
        thr = float(spec.threshold)
        band = margin.band_for_threshold(thr)
        remaining = thr - value
        if op == "<=":
            if value <= thr - band:
                return ScenarioState.INSIDE, remaining, band
            if value <= thr:
                return ScenarioState.BOUNDARY, remaining, band
            return ScenarioState.OUTSIDE, remaining, band
        # op == "<"
        if value < thr - band:
            return ScenarioState.INSIDE, remaining, band
        if value < thr:
            return ScenarioState.BOUNDARY, remaining, band
        return ScenarioState.OUTSIDE, remaining, band

    if op == "between" and spec.range is not None:
        lo, hi = float(spec.range[0]), float(spec.range[1])
        band = margin.band_for_range(lo, hi)
        if value < lo:
            remaining = value - lo
            return ScenarioState.OUTSIDE, remaining, band
        if value > hi:
            remaining = hi - value
            return ScenarioState.OUTSIDE, remaining, band
        dist_lo = value - lo
        dist_hi = hi - value
        remaining = min(dist_lo, dist_hi)
        if remaining <= band:
            return ScenarioState.BOUNDARY, remaining, band
        return ScenarioState.INSIDE, remaining, band

    if op == "==" and spec.threshold is not None:
        thr = float(spec.threshold)
        band = margin.band_for_threshold(thr) if margin.value else 0.0
        remaining = thr - value
        if value == thr:
            return ScenarioState.INSIDE, 0.0, band
        if band > 0 and abs(value - thr) <= band:
            return ScenarioState.BOUNDARY, remaining, band
        return ScenarioState.OUTSIDE, remaining, band

    if op == "!=" and spec.threshold is not None:
        thr = float(spec.threshold)
        band = margin.band_for_threshold(thr) if margin.value else 0.0
        if value != thr:
            return ScenarioState.INSIDE, abs(value - thr), band
        return ScenarioState.OUTSIDE, 0.0, band

    raise ValueError(f"Unsupported numeric op for classify_numeric: {op!r}")


def relative_band(threshold: float, fraction: float) -> float:
    return abs(threshold) * fraction
