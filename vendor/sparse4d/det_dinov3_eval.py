import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

import hydra
import numpy as np
import torch
from einops import rearrange
from hydra.core.global_hydra import GlobalHydra
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from pyquaternion import Quaternion
from tqdm import tqdm



CLASSES = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
]


def infer_attribute(detection_name: str, velocity) -> str:
    speed = float(np.linalg.norm(velocity))
    if detection_name in {"car", "bus", "truck", "trailer", "construction_vehicle"}:
        return "vehicle.moving" if speed > 0.2 else "vehicle.parked"
    if detection_name in {"bicycle", "motorcycle"}:
        return "cycle.with_rider" if speed > 0.2 else "cycle.without_rider"
    if detection_name == "pedestrian":
        return "pedestrian.moving" if speed > 0.2 else "pedestrian.standing"
    return ""


def parse_args(default_task="detection") -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DINOv3-backbone BEV-VAE detection checkpoints on nuScenes."
    )
    parser.add_argument(
        "--experiment",
        default=("dinov3-official-vitb-sta-sparse4d-tracking-1024x576-fpn4_nusc-bboxes_1x4x8x1_4e-4_80000"
                 if default_task == "tracking" else
                 "dinov3-official-vitb-sta-sparse4d-temporal-fixed-1024x576-fpn4_nusc-bboxes_1x4x8x1_4e-4_80000"),
        help="Hydra experiment config name without .yaml.",
    )
    parser.add_argument("--ckpt-path", required=True, help="Checkpoint path to evaluate.")
    parser.add_argument("--config-path", default="./configs")
    parser.add_argument("--config-name", default="train.yaml")
    parser.add_argument("--run-config", help="Saved training .hydra/config.yaml; preserves all architecture overrides.")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("NUSCENES_DATA_DIR", "/data/nusc/"),
        help="nuScenes root used by both the dataset and the official evaluator.",
    )
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--eval-output-dir", default="./nuscenes_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("32", "16-mixed", "bf16-mixed"),
        default="bf16-mixed",
        help="Inference precision. Use bf16-mixed for the fixed temporal A800 experiment.",
    )
    parser.add_argument("--max-num", type=int, default=300)
    parser.add_argument("--task", choices=("detection", "tracking"), default=default_task)
    parser.add_argument("--tracking-threshold", type=float, default=0.2,
                        help="Minimum raw classification score for creating a new track ID.")
    parser.add_argument(
        "--head",
        choices=("auto", "detector", "sparse4d"),
        default="auto",
        help="Detection branch to evaluate. 'auto' selects the only configured branch.",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.0,
        help="Score threshold passed into detector.get_bboxes.",
    )
    parser.add_argument(
        "--dinov3-repo-dir",
        default=None,
        help="Local official DINOv3 repo path, e.g. /data/code/dinov3 or third_party/dinov3.",
    )
    parser.add_argument(
        "--backbone-weight-path",
        default=None,
        help="Override model.backbone.weight_path from the Hydra config.",
    )
    parser.add_argument(
        "--no-pretrained-backbone",
        action="store_true",
        help="Instantiate the DINOv3 architecture without loading the original pretrain weights before ckpt restore.",
    )
    parser.add_argument(
        "--no-run-nuscenes-eval",
        action="store_true",
        help="Only write prediction JSON; skip official nuScenes metric computation.",
    )
    args = parser.parse_args()
    if args.task == "tracking" and args.head == "detector":
        parser.error("Tracking only supports --head=auto or --head=sparse4d")
    if not 0 <= args.score_thr <= 1 or not 1 <= args.max_num <= 500:
        parser.error("--score-thr must be in [0,1] and --max-num in [1,500]")
    return args


def set_default_env(data_root: str) -> None:
    os.environ.setdefault("PROJECT_ROOT", str(REPO_ROOT))
    os.environ.setdefault("WAYVE_DATA_DIR", "/data/wayve/")
    os.environ["NUSCENES_DATA_DIR"] = str(Path(data_root).expanduser())
    os.environ.setdefault("ARGOVERSE_DATA_DIR", "/data/av2/")
    os.environ.setdefault("NUPLAN_DATA_DIR", "/data/nupl/")
    os.environ.setdefault("NUIMAGES_DATA_DIR", "/data/nuim/")
    os.environ.setdefault("MANT_DATA_DIR", "/data/mant/")
    os.environ.setdefault("CKPT_DIR", "/data/ckpt/")
    os.environ.setdefault("OUTPUT_DIR", str(Path.home() / "bev-vae-logs"))


def as_batched_tensor(value, device: torch.device) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.tensor(value)
    return tensor.to(device)[None]


def get_scalar_string(value) -> str:
    if torch.is_tensor(value):
        return str(int(value.item()))
    if isinstance(value, (list, tuple)):
        return str(int(value[0]))
    return str(int(value))


def make_img_metas(batch: Dict, x_img: torch.Tensor, device: torch.device) -> Dict:
    metas = {
        "cam_num": len(batch["cam_names"]),
        "img_shape": x_img.shape[-2:],
        "geometric": as_batched_tensor(batch["geometric"], device),
        "intrinsic": as_batched_tensor(batch["intrinsic"], device),
        "extrinsic": as_batched_tensor(batch["extrinsic"], device),
    }
    if "sequence_id" in batch:
        for name in ("timestamp_ns", "ego_to_global", "sequence_start", "frame_index"):
            metas[name] = as_batched_tensor(batch[name], device)
        metas["sequence_id"] = [batch["sequence_id"]]
        metas["temporal_scope"] = "inference"
    return metas


def resolve_sensor_token(batch: Dict, sensor_dataset) -> str:
    log_id = batch["log_id"][0] if isinstance(batch["log_id"], list) else batch["log_id"]
    if "sequence_id" in batch:
        # Temporal metadata is normalized to ns; cache keys may still be us.
        # nuScenes.__getitem__ preserves the original key in sample_token.
        prefix = f"{log_id}_"
        sample_key = batch.get("sample_token", "")
        if not isinstance(sample_key, str) or not sample_key.startswith(prefix):
            raise ValueError("Temporal evaluation requires the original cache sample_token")
        timestamp_key = sample_key[len(prefix):]
    else:
        timestamp_key = get_scalar_string(batch["timestamp_ns"])
    return sensor_dataset.scene2sensor2stamp2token[log_id][sensor_dataset.refer_sensor][timestamp_key]


def inference_autocast(device: torch.device, precision: str):
    if precision in ("16-mixed", "bf16-mixed"):
        if device.type != "cuda":
            raise ValueError(f"{precision} inference requires a CUDA device")
        dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


@torch.no_grad()
def predict_bboxes(
    model,
    x_img: torch.Tensor,
    img_metas: Dict,
    score_thr: float,
    max_num: int,
    precision: str,
    head: str = "auto",
) -> torch.Tensor:
    with inference_autocast(x_img.device, precision):
        extract_image_features = getattr(model, "extract_image_features", None)
        img_feats = (
            extract_image_features(x_img)
            if callable(extract_image_features)
            else model.backbone(x_img)
        )
        has_detector = getattr(model, "detector", None) is not None
        has_sparse4d = getattr(model, "sparse4d_head", None) is not None
        if head == "auto":
            if has_detector == has_sparse4d:
                raise ValueError(
                    "--head=auto requires exactly one configured detection branch; "
                    "select --head=detector or --head=sparse4d explicitly"
                )
            head = "detector" if has_detector else "sparse4d"

        if head == "sparse4d":
            if not has_sparse4d:
                raise ValueError("The composed model does not contain a Sparse4D head")
            perspective_feats = model.scene_encoder.encode_image_features(
                img_feats, img_metas
            )
            return model.sparse4d_head.get_bboxes(
                perspective_feats,
                img_metas,
                score_thr=score_thr,
                max_num=max_num,
            )

        if not has_detector:
            raise ValueError("The composed model does not contain the DETR detector head")
        proposals = None
        if model.perspective_aux_head is not None and getattr(
            model.detector, "proposal_query_count", 0
        ) > 0:
            scene_feats, perspective_feats = model.scene_encoder(
                img_feats, img_metas, return_img_feats=True
            )
            proposals = model.perspective_aux_head.get_proposals(
                perspective_feats, img_metas
            )
        else:
            scene_feats = model.scene_encoder(img_feats, img_metas)
        scene_feats = model.apply_bev_coordinate_augmentation(scene_feats, img_metas) if hasattr(
            model, "apply_bev_coordinate_augmentation") else scene_feats
        z = model.connector(rearrange(scene_feats, "b c d h w -> b (h w) (d c)"))
        if hasattr(model, "get_task_features"):
            z = model.get_task_features(z, "det")
        return model.detector.get_bboxes(
            z,
            score_thr=score_thr,
            max_num=max_num,
            proposals=proposals,
        )


def build_submission(
    model,
    val_dataset,
    data_root: str,
    device: torch.device,
    score_thr: float,
    max_num: int,
    precision: str,
    head: str = "auto",
    task: str = "detection",
) -> Dict:
    nusc = NuScenes(version="v1.0-trainval", dataroot=data_root, verbose=False)
    sensor_dataset = val_dataset.dataset
    results_dict = {}

    model.eval()
    sparse4d_head = getattr(model, "sparse4d_head", None)
    if task == "tracking" and not getattr(sparse4d_head, "tracking", False):
        raise ValueError("Tracking evaluation requires a temporal Sparse4D tracking head")
    if getattr(sparse4d_head, "temporal", False):
        if not getattr(val_dataset, "temporal", False):
            raise ValueError("A temporal checkpoint requires data.val.temporal=True")
        sparse4d_head.reset_temporal("inference")
    # SensorDataset sorts by scene and timestamp. Iterate directly with B=1;
    # scene/frame metadata invalidates history when crossing a scene boundary.
    print(f"Evaluating {len(val_dataset)} validation samples with DINOv3 backbone inference.")

    for i in tqdm(range(len(val_dataset))):
        batch = val_dataset[i]
        real_token = resolve_sensor_token(batch, sensor_dataset)
        sample_data = nusc.get("sample_data", real_token)
        true_sample_token = sample_data["sample_token"]

        ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
        ego2global_rotation = ego_pose["rotation"]
        ego2global_translation = ego_pose["translation"]
        ego2global_quat = Quaternion(ego2global_rotation)

        x_img = batch["imgs"].to(device)
        img_metas = make_img_metas(batch, x_img, device)

        if task == "tracking":
            with inference_autocast(device, precision):
                img_feats = model.extract_image_features(x_img)
                features = model.scene_encoder.encode_image_features(img_feats, img_metas)
                tracks = sparse4d_head.get_tracks(features, img_metas, score_thr, max_num)
            pd_bboxes = tracks["bboxes"]
            track_ids = tracks["instance_ids"][0].cpu()
        else:
            pd_bboxes = predict_bboxes(
                model=model,
                x_img=x_img,
                img_metas=img_metas,
                score_thr=score_thr,
                max_num=max_num,
                precision=precision,
                head=head,
            )

        pd_boxes_flat = pd_bboxes[0].detach().cpu()
        valid_mask = pd_boxes_flat[..., 11] > 0.0
        valid_pds = pd_boxes_flat[valid_mask]

        sample_results = []
        valid_ids = track_ids[valid_mask] if task == "tracking" else None
        for box_index, box in enumerate(valid_pds):
            x_c, y_c, z_c, l_log, w_log, h_log, sin, cos, vx, vy, label_id, score = box.tolist()
            detection_name = CLASSES[int(label_id)]
            if task == "tracking" and detection_name not in {
                    "bicycle", "bus", "car", "motorcycle", "pedestrian", "trailer", "truck"}:
                continue

            length = float(np.exp(l_log))
            width = float(np.exp(w_log))
            height = float(np.exp(h_log))
            nusc_size = [width, length, height]

            yaw = float(np.arctan2(sin, cos))
            ego_quat = Quaternion(axis=[0, 0, 1], radians=yaw)
            global_quat = ego2global_quat * ego_quat

            ego_translation = np.array([x_c, y_c, z_c])
            global_translation = ego2global_quat.rotate(ego_translation) + np.array(
                ego2global_translation
            )

            ego_vel = np.array([vx, vy, 0.0])
            global_vel = ego2global_quat.rotate(ego_vel)[:2]

            sample_results.append(
                {
                    "sample_token": true_sample_token,
                    "translation": global_translation.tolist(),
                    "size": nusc_size,
                    "rotation": global_quat.elements.tolist(),
                    "velocity": global_vel.tolist(),
                    "detection_name": detection_name,
                    "detection_score": score,
                    "attribute_name": infer_attribute(detection_name, global_vel),
                }
            )
            if task == "tracking":
                record = sample_results[-1]
                record["tracking_name"] = record.pop("detection_name")
                record["tracking_score"] = record.pop("detection_score")
                record.pop("attribute_name")
                scene = nusc.get("sample", true_sample_token)["scene_token"]
                record["tracking_id"] = f"{scene}:{int(valid_ids[box_index])}"

        if true_sample_token in results_dict:
            raise ValueError(f"Duplicate nuScenes sample token: {true_sample_token}")
        results_dict[true_sample_token] = sample_results

    if task == "tracking":
        from bev_vae.evaluation.tracking import validate_tracking_submission
        validate_tracking_submission(results_dict, nusc)

    return {
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": True,
        },
        "results": results_dict,
    }


def make_overrides(args: argparse.Namespace) -> List[str]:
    overrides = [f"experiment={args.experiment}"]
    if args.dinov3_repo_dir is not None:
        overrides.append(f"model.backbone.repo_or_dir={args.dinov3_repo_dir}")
    if args.backbone_weight_path is not None:
        overrides.append(f"model.backbone.weight_path={args.backbone_weight_path}")
    if args.no_pretrained_backbone:
        overrides.append("model.backbone.weight_path=null")
    return overrides


def compose_config(args: argparse.Namespace):
    if getattr(args, "run_config", None):
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(args.run_config)
        # Only inference initialization paths may differ from the saved run.
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(make_overrides(args)[1:]))
        cfg.batch_size = 1
        cfg.data.train = None
        for key in ("with_bboxes", "with_map_vectors"):
            if key in cfg.data.val:
                cfg.data.val[key] = False
        cfg.model.ckpt_path = None
        cfg.model.load_ema = False
        if getattr(args, "task", "detection") == "tracking":
            raise ValueError("--run-config currently supports detection/map only; use --experiment for tracking")
        return cfg
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    config_path = Path(args.config_path)
    if config_path.is_absolute():
        hydra.initialize_config_dir(config_dir=str(config_path), job_name="det_dinov3_eval")
    else:
        hydra.initialize(config_path=str(config_path), job_name="det_dinov3_eval")

    cfg = hydra.compose(
        config_name=args.config_name,
        overrides=make_overrides(args),
    )
    cfg.batch_size = 1
    cfg.data.train = None
    if "with_bboxes" in cfg.data.val:
        cfg.data.val.with_bboxes = False
    if "with_map_vectors" in cfg.data.val:
        cfg.data.val.with_map_vectors = False
    # Restore below with strict=True so a mismatched experiment config cannot
    # silently evaluate a partially loaded checkpoint.
    cfg.model.ckpt_path = None
    cfg.model.load_ema = False
    if getattr(args, "task", "detection") == "tracking":
        if cfg.model.sparse4d_head is None or not cfg.model.sparse4d_head.get("temporal", False):
            raise ValueError("Tracking needs a temporal Sparse4D experiment")
        from omegaconf import open_dict
        with open_dict(cfg):
            cfg.model.sparse4d_head.tracking = True
            cfg.model.sparse4d_head.tracking_threshold = args.tracking_threshold
            cfg.data.val.tracking = True
    return cfg


def load_checkpoint_strict(model, ckpt_path: Path) -> None:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(checkpoint).__name__}")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint state_dict is not a mapping")

    model.load_state_dict(state_dict, strict=True)
    print(f"Strictly restored {len(state_dict)} tensors from {ckpt_path}.")


def run_nuscenes_eval(data_root: str, result_path: Path, output_dir: Path) -> None:
    from nuscnes_eval import NuScenesEval_custom
    nusc = NuScenes(version="v1.0-trainval", dataroot=data_root, verbose=True)
    eval_config = config_factory("detection_cvpr_2019")
    nusc_eval = NuScenesEval_custom(
        nusc=nusc,
        config=eval_config,
        result_path=str(result_path),
        eval_set="val",
        output_dir=str(output_dir),
        verbose=True,
        overlap_test=False,
    )
    nusc_eval.main(plot_examples=0, render_curves=False)


def infer_output_names(ckpt_path: Path) -> tuple[str, str]:
    ckpt_name = ckpt_path.stem
    if ckpt_path.parent.name == "checkpoints" and len(ckpt_path.parents) > 1:
        return ckpt_path.parents[1].name, ckpt_name
    return ckpt_path.parent.name, ckpt_name


def main(default_task="detection") -> None:
    args = parse_args(default_task)
    set_default_env(args.data_root)
    if args.task == "tracking" and not args.no_run_nuscenes_eval:
        from bev_vae.evaluation.tracking import get_tracking_evaluator
        get_tracking_evaluator()  # Fail before expensive checkpoint inference.

    ckpt_path = Path(args.ckpt_path)
    exp_name, ckpt_name = infer_output_names(ckpt_path)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = "val_tracking_results" if args.task == "tracking" else "val_results"
    result_path = results_dir / f"{exp_name}_{ckpt_name}_{suffix}.json"

    output_dir = Path(args.eval_output_dir) / f"{exp_name}_{ckpt_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = compose_config(args)
    datamodule = hydra.utils.instantiate(cfg.data)
    val_dataset = datamodule.hparams.val

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    model = hydra.utils.instantiate(cfg.model)
    load_checkpoint_strict(model, ckpt_path)
    model = model.to(device)

    submission = build_submission(
        model=model,
        val_dataset=val_dataset,
        data_root=args.data_root,
        device=device,
        score_thr=args.score_thr,
        max_num=args.max_num,
        precision=args.precision,
        head=args.head,
        task=args.task,
    )

    with result_path.open("w") as f:
        json.dump(submission, f, indent=4)
    print(f"Saved predictions to {result_path} with {len(submission['results'])} samples.")

    if not args.no_run_nuscenes_eval:
        print(f"Running nuScenes {args.task} evaluation.")
        if args.task == "tracking":
            from bev_vae.evaluation.tracking import run_tracking_eval
            run_tracking_eval(args.data_root, result_path, output_dir / "tracking")
        else:
            run_nuscenes_eval(args.data_root, result_path, output_dir)
        print(f"Saved nuScenes evaluation outputs to {output_dir}.")


if __name__ == "__main__":
    main()
