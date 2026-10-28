import math
from typing import Sequence

import numpy as np

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import (Compose, GaussianBlur, Normalize,
                                    RandomApply, RandomHorizontalFlip, Resize)

# Use timm's names
# IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_DEFAULT_MEAN = (0.5, 0.5, 0.5)
IMAGENET_DEFAULT_STD = (0.5, 0.5, 0.5)


def normalize(type: str = "IMAGENET") -> Normalize:
    if type == "IMAGENET":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif type == "SD":
        mean = std = (0.5, 0.5, 0.5)
    else:
        raise ValueError("Only support IMAGENET and SD")
    return Normalize(mean=mean, std=std)

def denormalize(tensor: torch.Tensor, type: str = "IMAGENET") -> torch.Tensor:
    if type == "IMAGENET":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif type == "SD":
        mean = std = (0.5, 0.5, 0.5)
    else:
        raise ValueError("Only support IMAGENET and SD")
    tensor = tensor.clone() 
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    mean, std = mean.view(-1, 1, 1), std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean).clamp_(0, 1)
    return tensor

def denormalize_z(
    tensor: torch.Tensor,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> torch.Tensor:
    tensor = tensor.clone() 
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    mean, std = mean.view(-1, 1, 1), std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


class RecordCompose(Compose):
    def __call__(self, img):
        matrix = torch.eye(3)
        for t in self.transforms:
            img = t(img)
            matrix = t.matrix @ matrix
        return img, matrix


class RCrop:
    """Crop an image and record the matching pixel-coordinate transform.

    ``crop_box`` follows PIL's ``(left, top, right, bottom)`` convention.
    """

    def __init__(self, crop_box: Sequence[int]) -> None:
        if len(crop_box) != 4:
            raise ValueError("crop_box must be (left, top, right, bottom)")
        left, top, right, bottom = (int(value) for value in crop_box)
        if left < 0 or top < 0 or right <= left or bottom <= top:
            raise ValueError(f"Invalid crop_box: {tuple(crop_box)}")
        self.crop_box = (left, top, right, bottom)
        self.matrix = torch.tensor(
            [[1.0, 0.0, -float(left)], [0.0, 1.0, -float(top)], [0.0, 0.0, 1.0]]
        )

    def __call__(self, img):
        left, top, right, bottom = self.crop_box
        width, height = img.size
        if right > width or bottom > height:
            raise ValueError(
                f"crop_box {self.crop_box} exceeds image size {(width, height)}"
            )
        return F.crop(
            img,
            top=top,
            left=left,
            height=bottom - top,
            width=right - left,
        )


class RResize(Resize):
    matrix = torch.eye(3)
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        # self.size(h, w)
        w, h = img.size # PIL
        self.matrix = torch.diag(torch.tensor([self.size[1]/w, self.size[0]/h, 1.]))
        return F.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
    
class RRandomHorizontalFlip(RandomHorizontalFlip):
    matrix = torch.eye(3)
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            self.matrix = torch.tensor([[-1., 0., img.size[0]], [0., 1., 0.], [0., 0., 1.]])
            return F.hflip(img)
        self.matrix = torch.eye(3)
        return img

class RandomGaussianBlur(RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class GlobalRotScaleTransImage:
    """BEV coordinate augmentation adapted from BEVFormer v2.

    This project stores 3D boxes as padded arrays:
    [x, y, z, log(l), log(w), log(h), sin(yaw), cos(yaw), vx, vy, label].
    The transform applies the sampled BEV coordinate transform directly to that
    representation and records the inverse matrix for BEV feature warping.
    """

    def __init__(
        self,
        rot_range=(-22.5, 22.5),
        scale_ratio_range=(0.95, 1.05),
        translation_std=(0.0, 0.0, 0.0),
        reverse_angle=True,
        flip_dx_ratio=0.5,
        flip_dy_ratio=0.5,
        only_gt=True,
    ) -> None:
        if any(abs(float(v)) > 0 for v in translation_std):
            raise ValueError("GlobalRotScaleTransImage only supports zero translation in this project")
        self.rot_range = tuple(float(v) for v in rot_range)
        self.scale_ratio_range = tuple(float(v) for v in scale_ratio_range)
        self.reverse_angle = reverse_angle
        self.flip_dx_ratio = float(flip_dx_ratio)
        self.flip_dy_ratio = float(flip_dy_ratio)
        self.only_gt = bool(only_gt)

    def __call__(self, results):
        results.setdefault("aug_param", {})
        param_key = "GlobalRotScaleTransImage_param"
        if param_key in results["aug_param"]:
            rot_angle, scale_ratio, flip_dx, flip_dy, bda_mat, only_gt = results["aug_param"][param_key]
            aug_mat = np.linalg.inv(np.asarray(bda_mat, dtype=np.float32))
        else:
            rot_angle, scale_ratio, flip_dx, flip_dy, aug_mat, bda_mat = self._sample_augmentation()
            only_gt = self.only_gt
            results["aug_param"][param_key] = (
                np.float32(rot_angle),
                np.float32(scale_ratio),
                np.bool_(flip_dx),
                np.bool_(flip_dy),
                bda_mat.astype(np.float32),
                np.bool_(only_gt),
            )

        self._transform_bboxes(results, aug_mat, scale_ratio)
        if not only_gt:
            self._transform_extrinsics(results, aug_mat)
        return results

    def _sample_augmentation(self):
        rot_angle = np.random.uniform(*self.rot_range) / 180.0 * math.pi
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        flip_dx = np.random.uniform() < self.flip_dx_ratio
        flip_dy = np.random.uniform() < self.flip_dy_ratio

        cos_theta = math.cos(rot_angle)
        sin_theta = math.sin(rot_angle)
        rot_mat = np.array(
            [
                [cos_theta, -sin_theta, 0.0],
                [sin_theta, cos_theta, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        scale_mat = np.diag([scale_ratio, scale_ratio, scale_ratio]).astype(np.float32)
        flip_mat = np.diag(
            [
                -1.0 if flip_dx else 1.0,
                -1.0 if flip_dy else 1.0,
                1.0,
            ]
        ).astype(np.float32)
        aug_mat = flip_mat @ (scale_mat @ rot_mat)
        bda_mat = np.linalg.inv(aug_mat)
        return rot_angle, scale_ratio, flip_dx, flip_dy, aug_mat, bda_mat

    @staticmethod
    def _transform_bboxes(results, aug_mat, scale_ratio):
        bboxes = results.get("gt_bboxes")
        if bboxes is None:
            return
        valid = bboxes[..., -1] != -1
        if not np.any(valid):
            return

        linear_xy = aug_mat[:2, :2]
        bboxes[valid, 0:3] = bboxes[valid, 0:3] @ aug_mat.T
        bboxes[valid, 3:6] += np.asarray(np.log(scale_ratio), dtype=bboxes.dtype)

        direction = np.stack([bboxes[valid, 7], bboxes[valid, 6]], axis=-1)
        direction = direction @ linear_xy.T
        direction_norm = np.linalg.norm(direction, axis=-1, keepdims=True).clip(min=1e-6)
        direction = direction / direction_norm
        bboxes[valid, 6] = direction[:, 1]
        bboxes[valid, 7] = direction[:, 0]

        bboxes[valid, 8:10] = bboxes[valid, 8:10] @ linear_xy.T

    @staticmethod
    def _transform_extrinsics(results, aug_mat):
        extrinsic = results.get("extrinsic")
        if extrinsic is None:
            return
        aug_4x4 = np.eye(4, dtype=np.float32)
        aug_4x4[:3, :3] = aug_mat.astype(np.float32)
        results["extrinsic"] = (aug_4x4[None] @ extrinsic).astype(np.float32)
