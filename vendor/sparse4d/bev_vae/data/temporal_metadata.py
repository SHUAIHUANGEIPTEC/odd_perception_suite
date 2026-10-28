"""Normalize legacy nuScenes cache timestamps at the model-input boundary."""
import operator

import numpy as np


def nuscenes_timestamp_to_ns(value, unit="auto"):
    """Return int64 nanoseconds without floating-point epoch arithmetic.

    Some caches retain nuScenes microseconds under the name ``timestamp_ns``.
    Keep those raw values for file/JSON lookups; convert only temporal metadata.
    Auto detection is restricted to Unix epochs between 2000 and 2100. Relative
    clocks require an explicit unit so we never guess from a frame interval.
    """
    timestamp = operator.index(value)
    if unit == "auto":
        for candidate, per_second in (("us", 10**6), ("ns", 10**9)):
            if 946684800 * per_second <= timestamp < 4102444800 * per_second:
                unit = candidate
                break
        else:
            raise ValueError(
                f"Cannot infer nuScenes timestamp unit from {timestamp}; "
                "set temporal_timestamp_unit to 'us' or 'ns' explicitly"
            )
    if unit not in ("us", "ns"):
        raise ValueError("temporal_timestamp_unit must be 'auto', 'us' or 'ns'")
    nanoseconds = timestamp * (1000 if unit == "us" else 1)
    if not np.iinfo(np.int64).min <= nanoseconds <= np.iinfo(np.int64).max:
        raise ValueError("Temporal timestamp overflows int64 nanoseconds; check timestamp unit")
    return np.int64(nanoseconds)
