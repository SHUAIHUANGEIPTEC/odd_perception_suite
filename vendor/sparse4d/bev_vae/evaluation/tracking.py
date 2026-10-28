"""nuScenes tracking protocol: seven classes, global coordinates, scene-safe IDs."""
import math

TRACKING_CLASSES = {"bicycle", "bus", "car", "motorcycle", "pedestrian", "trailer", "truck"}


def validate_tracking_submission(results, nusc=None):
    for token, records in results.items():
        ids = set()
        for row in records:
            if row["sample_token"] != token or row["tracking_name"] not in TRACKING_CLASSES:
                raise ValueError(f"Invalid tracking class/token at {token}")
            identity = row["tracking_id"]
            if not isinstance(identity, str) or not identity or identity in ids:
                raise ValueError(f"Duplicate/invalid tracking ID at {token}")
            ids.add(identity)
            values = row["translation"] + row["size"] + row["rotation"] + row["velocity"]
            if not all(math.isfinite(x) for x in values + [row["tracking_score"]]):
                raise ValueError(f"Non-finite tracking prediction at {token}")
            if min(row["size"]) <= 0 or not 0 <= row["tracking_score"] <= 1:
                raise ValueError(f"Invalid size/score at {token}")
    if nusc is not None:
        from nuscenes.utils.splits import create_splits_scenes
        scenes = set(create_splits_scenes()["val"])
        expected_scenes = {s["token"] for s in nusc.scene if s["name"] in scenes}
        expected = {s["token"] for s in nusc.sample if s["scene_token"] in expected_scenes}
        if set(results) != expected:
            raise ValueError(f"Tracking requires the full val split: missing={len(expected-set(results))}, "
                             f"extra={len(set(results)-expected)}")


def run_tracking_eval(data_root, result_path, output_dir):
    from nuscenes.eval.common.config import config_factory
    TrackingEval = get_tracking_evaluator()
    evaluator = TrackingEval(config=config_factory("tracking_nips_2019"),
                             result_path=str(result_path), eval_set="val",
                             output_dir=str(output_dir), nusc_version="v1.0-trainval",
                             nusc_dataroot=str(data_root), verbose=True)
    return evaluator.main(render_curves=False)


def get_tracking_evaluator():
    try:
        from nuscenes.eval.tracking.evaluate import TrackingEval
    except ImportError as error:
        raise RuntimeError(
            "Official tracking metrics require nuscenes-devkit>=1.2.0 and motmetrics==1.4.0; "
            "install the dependencies in requirements.txt before evaluation."
        ) from error
    return TrackingEval
