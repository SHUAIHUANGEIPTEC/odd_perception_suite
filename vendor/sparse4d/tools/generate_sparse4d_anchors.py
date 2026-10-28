"""Generate Sparse4D anchors from BEV-VAE nuScenes annotations.

BEV-VAE's ``scene2stamp2annotation.json`` stores each cuboid as
``[qw, qx, qy, qz, x, y, z, l, w, h, class_id]`` in the ego coordinate frame.
The generated anchor code is
``[x, y, z, log(l), log(w), log(h), sin(yaw), cos(yaw), vx, vy, vz]``.
Only centers are clustered, matching the original Sparse4D generator.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


DEFAULT_PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)


def iter_annotation_frames(
    annotation_data: dict,
    scene_ids: Optional[Set[str]] = None,
    scene2frame: Optional[dict] = None,
) -> Iterable[np.ndarray]:
    for scene_id, scene_frames in annotation_data.items():
        if scene_ids is not None and scene_id not in scene_ids:
            continue
        valid_timestamps = None
        if scene2frame is not None:
            valid_timestamps = {str(timestamp) for timestamp in scene2frame[scene_id]}
        for timestamp, cuboids in scene_frames.items():
            if valid_timestamps is not None and str(timestamp) not in valid_timestamps:
                continue
            if not cuboids:
                continue
            frame = np.asarray(cuboids, dtype=np.float32)
            if frame.ndim != 2 or frame.shape[1] < 11:
                raise ValueError(
                    "Expected BEV-VAE cuboids with at least 11 values, got "
                    f"shape {frame.shape}"
                )
            yield frame


def collect_training_boxes(
    annotation_file: Path,
    pc_range: Sequence[float],
    scene_ids: Optional[Set[str]] = None,
    scene2frame: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    with annotation_file.open("r", encoding="utf-8") as handle:
        annotation_data = json.load(handle)

    centers = []
    dimensions = []
    for frame in iter_annotation_frames(
        annotation_data,
        scene_ids=scene_ids,
        scene2frame=scene2frame,
    ):
        frame_centers = frame[:, 4:7]
        frame_dimensions = frame[:, 7:10]
        valid = (
            (frame_centers[:, 0] > pc_range[0])
            & (frame_centers[:, 0] < pc_range[3])
            & (frame_centers[:, 1] > pc_range[1])
            & (frame_centers[:, 1] < pc_range[4])
            & np.isfinite(frame_centers).all(axis=1)
            & np.isfinite(frame_dimensions).all(axis=1)
            & (frame_dimensions > 0).all(axis=1)
        )
        centers.append(frame_centers[valid])
        dimensions.append(frame_dimensions[valid])

    if not centers:
        raise ValueError(f"No cuboids were found in {annotation_file}")
    centers_array = np.concatenate(centers, axis=0)
    dimensions_array = np.concatenate(dimensions, axis=0)
    if len(centers_array) == 0:
        raise ValueError("No valid cuboids remain after the configured range filter")
    return centers_array, dimensions_array


def load_split_scene_ids(sensor_cache: Path, split: str) -> Set[str]:
    cache = pd.read_feather(sensor_cache, columns=["split", "log_id"])
    scene_ids = set(cache.loc[cache["split"] == split, "log_id"].unique())
    if not scene_ids:
        raise ValueError(f"No scenes with split={split!r} were found in {sensor_cache}")
    return scene_ids


def generate_anchors(
    centers: np.ndarray,
    dimensions: np.ndarray,
    num_anchor: int,
    seed: int,
    max_iter: int,
    verbose: int,
) -> np.ndarray:
    if len(centers) < num_anchor:
        raise ValueError(
            f"Need at least {num_anchor} boxes for K-means, found {len(centers)}"
        )
    estimator = KMeans(
        n_clusters=num_anchor,
        random_state=seed,
        n_init=10,
        max_iter=max_iter,
        verbose=verbose,
    )
    estimator.fit(centers)
    anchors = np.zeros((num_anchor, 11), dtype=np.float32)
    anchors[:, :3] = estimator.cluster_centers_.astype(np.float32)
    anchors[:, 3:6] = np.log(dimensions.mean(axis=0)).astype(np.float32)
    anchors[:, 7] = 1.0
    return anchors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-file",
        type=Path,
        required=True,
        help="Path to nusc/scene2stamp2annotation.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nuscenes_kmeans900_bevvae.npy"),
    )
    parser.add_argument("--num-anchor", type=int, default=900)
    parser.add_argument(
        "--sensor-cache",
        type=Path,
        default=None,
        help=(
            "Path to nusc/sensor_cache.feather. Defaults to the file beside "
            "--annotation-file and is used to prevent validation-scene leakage."
        ),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--scene2frame-file",
        type=Path,
        default=None,
        help=(
            "Path to nusc/scene2frame.json. Defaults to the file beside "
            "--annotation-file and restricts clustering to dataset keyframes."
        ),
    )
    parser.add_argument(
        "--pc-range",
        type=float,
        nargs=6,
        default=DEFAULT_PC_RANGE,
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--verbose", action="count", default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sensor_cache = args.sensor_cache
    if sensor_cache is None:
        sensor_cache = args.annotation_file.parent / "sensor_cache.feather"
    if not sensor_cache.is_file():
        raise FileNotFoundError(
            f"nuScenes sensor cache does not exist: {sensor_cache}. Pass "
            "--sensor-cache explicitly; anchors must be generated from the "
            "training scenes only."
        )
    scene_ids = load_split_scene_ids(sensor_cache, args.split)
    scene2frame_file = args.scene2frame_file
    if scene2frame_file is None:
        scene2frame_file = args.annotation_file.parent / "scene2frame.json"
    if not scene2frame_file.is_file():
        raise FileNotFoundError(
            f"nuScenes keyframe metadata does not exist: {scene2frame_file}. "
            "Pass --scene2frame-file explicitly."
        )
    with scene2frame_file.open("r", encoding="utf-8") as handle:
        scene2frame = json.load(handle)
    missing_scenes = scene_ids.difference(scene2frame)
    if missing_scenes:
        raise KeyError(
            f"scene2frame metadata is missing {len(missing_scenes)} training scenes"
        )
    centers, dimensions = collect_training_boxes(
        args.annotation_file,
        args.pc_range,
        scene_ids=scene_ids,
        scene2frame=scene2frame,
    )
    print(f"Clustering {len(centers)} BEV-VAE ego-frame boxes into {args.num_anchor} anchors")
    anchors = generate_anchors(
        centers,
        dimensions,
        num_anchor=args.num_anchor,
        seed=args.seed,
        max_iter=args.max_iter,
        verbose=args.verbose,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, anchors)
    print(f"Saved {anchors.shape} anchors to {args.output.resolve()}")


if __name__ == "__main__":
    main()
