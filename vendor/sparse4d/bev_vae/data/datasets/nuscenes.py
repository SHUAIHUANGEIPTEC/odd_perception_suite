from dataclasses import dataclass
from pyquaternion import Quaternion
from bev_vae.data.datasets.utils import read_json_file
import torchvision.transforms as T
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from bev_vae.data.datasets.nuscenes_sensor import SensorDataset
from bev_vae.data.temporal_metadata import nuscenes_timestamp_to_ns
from bev_vae.data.transforms import (
    GlobalRotScaleTransImage,
    RCrop,
    RecordCompose,
    RResize,
    RRandomHorizontalFlip,
    RandomGaussianBlur,
    normalize,
)
import random


@dataclass
class nuScenes(Dataset):
    data_dir: str
    split: str
    cam_names: List[str]
    other_sensors: List[str]
    class_num: int 
    scene_size: List[int] 
    pc: List[float] 
    with_condition: bool = False
    return_bev: bool = False
    with_bboxes: bool = False
    with_map_vectors: bool = False
    cam_res: Tuple[int, int] = (256, 256)
    crop_box: Optional[Tuple[int, int, int, int]] = None
    augment: bool = False
    augmentation_mode: str = "bevvae"
    sparse4d_resize_lim: Tuple[float, float] = (0.40, 0.47)
    sparse4d_bot_pct_lim: Tuple[float, float] = (0.0, 0.0)
    sparse4d_rot_lim: Tuple[float, float] = (-5.4, 5.4)
    sparse4d_rand_flip: bool = True
    sparse4d_rot3d_range: Tuple[float, float] = (-0.3925, 0.3925)
    temporal: bool = False
    tracking: bool = False
    maptr_v2: bool = False
    map_vertex_num: int = 20
    map_max_instances: int = 128
    map_pv_stride: int = 32
    map_ground_height: float = 0.0
    map_aux_targets: bool = True
    # Independent map target extent; detection and camera calibration keep pc.
    map_pc: Optional[List[float]] = None
    map_scene_size: Optional[List[int]] = None
    temporal_timestamp_unit: str = "auto"
    sequences_split_num: int = 2
    extrinsic_augment: bool = False
    extrinsic_rot_range: Tuple[float, float] = (-22.5, 22.5)
    extrinsic_scale_ratio_range: Tuple[float, float] = (0.95, 1.05)
    extrinsic_flip_dx_ratio: float = 0.5
    extrinsic_flip_dy_ratio: float = 0.5
    extrinsic_aug_only_gt: bool = True
    def __post_init__(self) -> None:
        self.map_pc = list(self.pc if self.map_pc is None else self.map_pc)
        self.map_scene_size = list(self.scene_size if self.map_scene_size is None else self.map_scene_size)
        if len(self.map_pc) != 6 or not np.isfinite(self.map_pc).all() or any(
            self.map_pc[i] >= self.map_pc[i + 3] for i in range(3)
        ):
            raise ValueError("map_pc must contain six finite, increasing xyz bounds")
        if len(self.map_scene_size) != 3 or any(int(v) != v or v < 1 for v in self.map_scene_size):
            raise ValueError("map_scene_size must contain three positive integer dimensions")
        if not self.maptr_v2 and (self.map_pc != list(self.pc) or self.map_scene_size != list(self.scene_size)):
            raise ValueError("Independent map range/grid requires maptr_v2=True")
        if self.tracking and not self.temporal:
            raise ValueError("Tracking requires temporal=True and chronological scene sampling")
        if self.maptr_v2 and (self.augmentation_mode != "sparse4d" or self.extrinsic_augment):
            raise ValueError("MapTRv2 requires sparse4d image/ego augmentation with extrinsic_augment=False")
        if self.augmentation_mode not in {"bevvae", "sparse4d"}:
            raise ValueError(
                "augmentation_mode must be 'bevvae' or 'sparse4d', got "
                f"{self.augmentation_mode!r}"
            )
        if self.augmentation_mode == "sparse4d" and self.crop_box is not None:
            raise ValueError(
                "Sparse4D samples resize/crop jointly; crop_box must be null"
            )
        if self.augmentation_mode == "sparse4d" and self.return_bev:
            raise ValueError("Sparse4D image augmentation requires return_bev=False")
        self.data_dir = Path(self.data_dir)
        self.dataset = SensorDataset(
            data_dir=self.data_dir, 
            split=self.split, 
            cam_names=self.cam_names, 
            other_sensors=self.other_sensors, 
            class_num=self.class_num, 
            scene_size=self.scene_size, 
            pc=self.pc, 
            with_condition=self.with_condition,
            return_bev=self.return_bev,
            with_bboxes=self.with_bboxes,
            with_map_vectors=self.with_map_vectors,
            maptr_v2=self.maptr_v2,
            temporal=self.temporal)
        if self.temporal:
            if self.augmentation_mode != "sparse4d" or self.return_bev:
                raise ValueError("Temporal detection requires sparse4d image augmentation and return_bev=False")
            if self.dataset.refer_sensor != "LIDAR_TOP":
                raise ValueError("Temporal nuScenes requires LIDAR_TOP keyframe indexing")
            if self.sequences_split_num < 1:
                raise ValueError("sequences_split_num must be positive")
            if not hasattr(self.dataset, "token2ego"):
                self.dataset.token2ego = read_json_file(self.data_dir / "nusc/token2ego.json")
            records = self.dataset.sensor_cache.xs(key=self.dataset.refer_sensor, level=2)
            self.sequence_groups = []
            scene_ids = records.index.get_level_values("log_id").to_numpy()
            for scene_id in dict.fromkeys(scene_ids):
                indices = np.flatnonzero(scene_ids == scene_id)
                # Preserve the SensorDataset index mapping; never index raw scenes independently.
                self.sequence_groups.extend([chunk.tolist() for chunk in
                    np.array_split(indices, self.sequences_split_num) if len(chunk)])
        self.preprocess = T.ToPILImage() 
        image_geometry = []
        if self.crop_box is not None:
            image_geometry.append(RCrop(self.crop_box))
        image_geometry.append(RResize(self.cam_res))
        self.resize = RecordCompose(image_geometry)
        self.geometric_augmentation = RecordCompose([RRandomHorizontalFlip(p=0.5)])
        self.color_jittering = T.Compose([T.RandomApply([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8), 
            T.RandomGrayscale(p=0.2)])
        # self.local_transfo_extra = RandomGaussianBlur(p=0.5)
        self.normalize = T.Compose([
            T.ToTensor(), normalize()])
        self.tensor_normalize = normalize()
        self.extrinsic_augmentation = None
        if self.augmentation_mode != "sparse4d" and self.extrinsic_augment:
            self.extrinsic_augmentation = GlobalRotScaleTransImage(
                rot_range=self.extrinsic_rot_range,
                scale_ratio_range=self.extrinsic_scale_ratio_range,
                flip_dx_ratio=self.extrinsic_flip_dx_ratio,
                flip_dy_ratio=self.extrinsic_flip_dy_ratio,
                only_gt=self.extrinsic_aug_only_gt,
            )
        print(f'nuScenes {self.split} has {len(self)} samples.')

    def _sample_sparse4d_augmentation(self, rng=None) -> dict:
        rng = np.random if rng is None else rng
        source_h, source_w = 900, 1600
        final_h, final_w = (int(value) for value in self.cam_res)
        if self.augment:
            resize = rng.uniform(*self.sparse4d_resize_lim)
            resized_w, resized_h = int(source_w * resize), int(source_h * resize)
            crop_h = (
                int(
                    (1.0 - rng.uniform(*self.sparse4d_bot_pct_lim))
                    * resized_h
                )
                - final_h
            )
            crop_w = int(rng.uniform(0, max(0, resized_w - final_w)))
            flip = bool(
                self.sparse4d_rand_flip and rng.choice([0, 1])
            )
            rotate = float(rng.uniform(*self.sparse4d_rot_lim))
            rotate_3d = float(rng.uniform(*self.sparse4d_rot3d_range))
        else:
            resize = max(final_h / source_h, final_w / source_w)
            resized_w, resized_h = int(source_w * resize), int(source_h * resize)
            crop_h = (
                int(
                    (1.0 - np.mean(self.sparse4d_bot_pct_lim)) * resized_h
                )
                - final_h
            )
            crop_w = int(max(0, resized_w - final_w) / 2)
            flip = False
            rotate = 0.0
            rotate_3d = 0.0
        return {
            "resize": float(resize),
            "resize_dims": (resized_w, resized_h),
            "crop": (crop_w, crop_h, crop_w + final_w, crop_h + final_h),
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }

    @staticmethod
    def _apply_sparse4d_3d_rotation(results: dict, angle: float) -> None:
        """Apply Sparse4D BBoxRotation to BEV-VAE boxes and calibration."""
        cosine = np.cos(angle)
        sine = np.sin(angle)
        rotation = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        GlobalRotScaleTransImage._transform_bboxes(results, rotation, 1.0)
        GlobalRotScaleTransImage._transform_extrinsics(results, rotation)
        rotation_inverse = np.linalg.inv(rotation).astype(np.float32)
        results.setdefault("aug_param", {})["GlobalRotScaleTransImage_param"] = (
            np.float32(angle),
            np.float32(1.0),
            np.bool_(False),
            np.bool_(False),
            rotation_inverse,
            np.bool_(False),
        )

    @staticmethod
    def _sparse4d_photometric_distortion(img: np.ndarray) -> np.ndarray:
        """RGB equivalent of Sparse4D's per-view MMCV photometric pipeline."""
        img = img.astype(np.float32, copy=True)
        if np.random.randint(2):
            img += np.random.uniform(-32.0, 32.0)

        contrast_first = np.random.randint(2) == 1
        if contrast_first and np.random.randint(2):
            img *= np.random.uniform(0.5, 1.5)

        # mmcv.bgr2hsv scales S/V to [0, 255]. The input in BEV-VAE is
        # already RGB, so use the equivalent OpenCV RGB conversion.
        hsv = cv2.cvtColor(img / 255.0, cv2.COLOR_RGB2HSV)
        hsv[..., 1:] *= 255.0
        if np.random.randint(2):
            hsv[..., 1] *= np.random.uniform(0.5, 1.5)
        if np.random.randint(2):
            hsv[..., 0] += np.random.uniform(-18.0, 18.0)
            hsv[..., 0][hsv[..., 0] > 360.0] -= 360.0
            hsv[..., 0][hsv[..., 0] < 0.0] += 360.0
        hsv_for_rgb = hsv.copy()
        hsv_for_rgb[..., 1:] /= 255.0
        img = cv2.cvtColor(hsv_for_rgb, cv2.COLOR_HSV2RGB) * 255.0

        if not contrast_first and np.random.randint(2):
            img *= np.random.uniform(0.5, 1.5)
        if np.random.randint(2):
            img = img[..., np.random.permutation(3)]
        return img

    def _process_sparse4d_img(
        self,
        img: np.ndarray,
        aug_config: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.preprocess(img)
        image = image.resize(aug_config["resize_dims"]).crop(aug_config["crop"])
        if aug_config["flip"]:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = image.rotate(aug_config["rotate"])
        image = np.array(image, dtype=np.float32, copy=True)
        if self.augment:
            image = self._sparse4d_photometric_distortion(image)
        image = np.ascontiguousarray(image)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        image_tensor = self.tensor_normalize(image_tensor)

        resize = aug_config["resize"]
        crop = aug_config["crop"]
        transform_matrix = np.eye(3, dtype=np.float32)
        transform_matrix[:2, :2] *= resize
        transform_matrix[:2, 2] -= np.asarray(crop[:2], dtype=np.float32)
        if aug_config["flip"]:
            flip_matrix = np.asarray(
                [[-1.0, 0.0, crop[2] - crop[0]], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            transform_matrix = flip_matrix @ transform_matrix
        rotate = aug_config["rotate"] / 180.0 * np.pi
        rotation_matrix = np.asarray(
            [
                [np.cos(rotate), np.sin(rotate), 0.0],
                [-np.sin(rotate), np.cos(rotate), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        rotation_center = np.asarray(
            [crop[2] - crop[0], crop[3] - crop[1]], dtype=np.float32
        ) / 2.0
        rotation_matrix[:2, 2] = (
            -rotation_matrix[:2, :2] @ rotation_center + rotation_center
        )
        transform_matrix = rotation_matrix @ transform_matrix
        return image_tensor, torch.from_numpy(transform_matrix)

    def process_img(self, img, sparse4d_aug_config: Optional[dict] = None):
        if self.augmentation_mode == "sparse4d":
            if sparse4d_aug_config is None:
                raise ValueError("Sparse4D image augmentation parameters are missing")
            return self._process_sparse4d_img(img, sparse4d_aug_config)
        img, geometric = self.resize(self.preprocess(img)) 
        if self.augment:
            img_aug, geometric_aug = self.geometric_augmentation(img.copy())
            geometric = geometric_aug @ geometric
            img_aug = self.color_jittering(img_aug)
            return torch.cat([self.normalize(img), self.normalize(img_aug)]), geometric
        return self.normalize(img), geometric
    
    def get_geometric(self, w, h):
        if self.crop_box is None:
            return torch.diag(
                torch.tensor([self.cam_res[1] / w, self.cam_res[0] / h, 1.])
            )
        left, top, right, bottom = self.crop_box
        crop = torch.tensor(
            [[1.0, 0.0, -float(left)], [0.0, 1.0, -float(top)], [0.0, 0.0, 1.0]]
        )
        resize = torch.diag(
            torch.tensor(
                [
                    self.cam_res[1] / (right - left),
                    self.cam_res[0] / (bottom - top),
                    1.0,
                ]
            )
        )
        return resize @ crop

    def __getitem__(self, idx: int):
        request = idx if isinstance(idx, dict) else None
        if request is not None:
            idx = int(request["idx"])
        if self.temporal and self.split == "train" and request is None:
            raise ValueError("Temporal training must use SequenceBatchSampler, not a shuffled frame loader")
        while True:
            try:
                datum = self.dataset[idx]
                sparse4d_aug_config = (
                    self._sample_sparse4d_augmentation(
                        np.random.RandomState(request["aug_seed"]) if request is not None else None
                    )
                    if self.augmentation_mode == "sparse4d"
                    else None
                )
                if self.return_bev:
                    img_sizes = [(1600, 900)] * len(self.cam_names) # nusc
                    geometric = [self.get_geometric(w, h) for w, h in img_sizes]
                    intrinsic = [datum["latent"][cam]["intrinsic"] for cam in self.cam_names]
                    extrinsic = [datum["latent"][cam]["extrinsic"] for cam in self.cam_names]
                    datum["path"] =  datum["latent"]["path"]
                    datum["bev"] =  datum["latent"]["bev"]
                    del datum["latent"]
                else:
                    if not self.temporal:
                        targets = [self.normalize(self.preprocess(datum["synchronized_imagery"][cam]["img"])) for cam in self.cam_names]
                        datum["targets"] = torch.stack(targets)
                    imgs, geometric = zip(*[
                        self.process_img(
                            datum["synchronized_imagery"][cam]["img"],
                            sparse4d_aug_config,
                        )
                        for cam in self.cam_names
                    ])
                    intrinsic = [datum["synchronized_imagery"][cam]["intrinsic"] for cam in self.cam_names]
                    extrinsic = [datum["synchronized_imagery"][cam]["extrinsic"] for cam in self.cam_names]
                    datum["paths"] =  [datum["synchronized_imagery"][cam]["path"] for cam in self.cam_names]
                    datum["imgs"] = torch.stack(imgs)
                    del datum["synchronized_imagery"]

                datum["img_shape"] = torch.tensor(self.cam_res)
                datum["geometric"] = torch.stack(geometric)
                datum["intrinsic"] = np.stack(intrinsic).astype(np.float32) # float64 -> float32
                datum["extrinsic"] = np.stack(extrinsic).astype(np.float32) # float64 -> float32
                if sparse4d_aug_config is not None:
                    self._apply_sparse4d_3d_rotation(
                        datum, sparse4d_aug_config["rotate_3d"]
                    )
                elif self.extrinsic_augmentation is not None:
                    datum = self.extrinsic_augmentation(datum)
                if self.maptr_v2 and self.with_map_vectors:
                    from bev_vae.data.maptr_targets import transform_instances, build_map_targets, extraction_range
                    query_pc = extraction_range(self.map_pc, sparse4d_aug_config["rotate_3d"])
                    instances = self.dataset._load_map_instances(datum["log_id"], datum["timestamp_ns"], query_pc)
                    instances = transform_instances(instances, self.map_pc, sparse4d_aug_config["rotate_3d"])
                    projection = {}
                    if self.map_aux_targets:
                        projection = dict(intrinsic=datum["intrinsic"], extrinsic=datum["extrinsic"],
                                          geometric=datum["geometric"].numpy(), image_shape=self.cam_res,
                                          pv_stride=self.map_pv_stride, ground_height=self.map_ground_height)
                    datum.update(build_map_targets(instances, self.map_pc, self.map_scene_size,
                                                  self.map_vertex_num, self.map_max_instances, **projection))
                # Preserve the cache key used by paths/JSON and score callbacks.
                datum["sample_token"] = f'{datum["log_id"]}_{datum["timestamp_ns"]}'
                if self.temporal:
                    scene_id = datum["log_id"]
                    timestamp = int(datum["timestamp_ns"])
                    token = self.dataset.scene2sensor2stamp2token[scene_id][self.dataset.refer_sensor][str(timestamp)]
                    params = self.dataset.token2ego[token]
                    ego_to_global = np.eye(4, dtype=np.float64)
                    ego_to_global[:3, :3] = Quaternion(params[:4]).rotation_matrix
                    ego_to_global[:3, 3] = params[4:]
                    angle = sparse4d_aug_config["rotate_3d"]
                    c, s = np.cos(angle), np.sin(angle)
                    # Boxes are in augmented ego coordinates: global = T_ego_global @ inv(A) @ box_aug.
                    inverse_aug = np.eye(4, dtype=np.float64)
                    inverse_aug[:2, :2] = [[c, s], [-s, c]]
                    datum["ego_to_global"] = ego_to_global @ inverse_aug
                    datum["timestamp_ns"] = nuscenes_timestamp_to_ns(
                        timestamp, self.temporal_timestamp_unit)
                    datum["frame_index"] = np.int64(datum["idx_in_log"])
                    datum["sequence_id"] = f'{self.split}:{scene_id}:' + (request["sequence_id"] if request else "inference")
                    datum["sequence_start"] = bool(request["sequence_start"] if request else datum["idx_in_log"] == 0)
                datum["cam_names"] = self.cam_names
                return datum
            except Exception as e:
                if self.split != "train" or self.temporal or self.maptr_v2:
                    raise
                print(f"Error accessing index {idx}: {e}. Picking a new index.")
                idx = random.randint(0, len(self.dataset) - 1)
    
    def __len__(self):
        return len(self.dataset)
