"""Native GridMask used by the Sparse4D comparison experiment."""

import torch
import torch.nn as nn


class Sparse4DGridMask(nn.Module):
    """Sparse4D's official GridMask parameters without an MMCV dependency.

    The reference configuration uses ``rotate=1``, so its sampled rotation is
    always zero degrees. A single mask is shared across the full flattened
    batch/camera/channel tensor, matching the original implementation.
    """

    def __init__(
        self,
        use_h: bool = True,
        use_w: bool = True,
        rotate: int = 1,
        offset: bool = False,
        ratio: float = 0.5,
        mode: int = 1,
        prob: float = 0.7,
    ) -> None:
        super().__init__()
        if rotate != 1:
            raise ValueError(
                "The native Sparse4D GridMask currently implements the "
                "official rotate=1 setting only"
            )
        if offset:
            raise ValueError(
                "The official Sparse4Dv3 config uses offset=False; other "
                "GridMask offset modes are not enabled in this integration"
            )
        if mode not in (0, 1):
            raise ValueError("GridMask mode must be 0 or 1")
        self.use_h = bool(use_h)
        self.use_w = bool(use_w)
        self.ratio = float(ratio)
        self.mode = int(mode)
        self.prob = float(prob)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand((), device=image.device) > self.prob:
            return image
        if image.ndim != 4:
            raise ValueError(
                f"GridMask expects [N,C,H,W], got {tuple(image.shape)}"
            )
        batch_size, channels, height, width = image.shape
        if height <= 2:
            return image

        mask_height = int(1.5 * height)
        mask_width = int(1.5 * width)
        distance = int(
            torch.randint(2, height, (), device=image.device).item()
        )
        line_width = min(
            max(int(distance * self.ratio + 0.5), 1), distance - 1
        )
        start_h = int(
            torch.randint(distance, (), device=image.device).item()
        )
        start_w = int(
            torch.randint(distance, (), device=image.device).item()
        )
        mask = image.new_ones(mask_height, mask_width)
        if self.use_h:
            for index in range(mask_height // distance):
                start = distance * index + start_h
                mask[start : min(start + line_width, mask_height), :] = 0
        if self.use_w:
            for index in range(mask_width // distance):
                start = distance * index + start_w
                mask[:, start : min(start + line_width, mask_width)] = 0
        top = (mask_height - height) // 2
        left = (mask_width - width) // 2
        mask = mask[top : top + height, left : left + width]
        if self.mode == 1:
            mask = 1.0 - mask
        mask = mask.reshape(1, 1, height, width)
        return image * mask.expand(batch_size, channels, -1, -1)


__all__ = ["Sparse4DGridMask"]
