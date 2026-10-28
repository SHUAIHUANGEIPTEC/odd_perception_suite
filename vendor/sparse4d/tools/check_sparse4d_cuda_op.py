"""Compare the native Sparse4D CUDA operator against PyTorch grid_sample."""

import argparse
from typing import List, Sequence

import torch
import torch.nn.functional as F

from bev_vae.models.ops.deformable_aggregation import (
    CUDA_DEFORMABLE_AGGREGATION_AVAILABLE,
    deformable_aggregation,
    format_feature_maps,
)


def reference_aggregation(
    feature_maps: Sequence[torch.Tensor],
    sampling_location: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Reference for locations [B,A,P,V,2], weights [B,A,P,V,L,G]."""
    batch_size, num_anchor, num_points, num_cams = sampling_location.shape[:4]
    num_groups = weights.shape[-1]
    embed_dims = feature_maps[0].shape[2]
    group_dims = embed_dims // num_groups
    camera_grid = sampling_location.permute(0, 3, 1, 2, 4)
    camera_grid = (camera_grid * 2.0 - 1.0).flatten(0, 1)

    sampled_levels: List[torch.Tensor] = []
    for feature_map in feature_maps:
        sampled = F.grid_sample(
            feature_map.flatten(0, 1),
            camera_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.reshape(
            batch_size,
            num_cams,
            embed_dims,
            num_anchor,
            num_points,
        ).permute(0, 3, 1, 4, 2)
        sampled_levels.append(sampled)
    sampled = torch.stack(sampled_levels, dim=3)
    sampled = sampled.reshape(
        batch_size,
        num_anchor,
        num_cams,
        len(feature_maps),
        num_points,
        num_groups,
        group_dims,
    )
    reference_weights = weights.permute(0, 1, 3, 4, 2, 5)
    return (sampled * reference_weights[..., None]).sum(dim=(2, 3, 4)).flatten(2)


def max_error(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float((lhs - rhs).abs().max().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--grad-atol", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not CUDA_DEFORMABLE_AGGREGATION_AVAILABLE:
        raise RuntimeError(
            "Sparse4D CUDA extension is not importable. Build it with: python "
            "bev_vae/models/ops/deformable_aggregation/setup.py build_ext --inplace"
        )

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    batch_size, num_cams, embed_dims = 2, 3, 16
    num_anchor, num_points, num_groups = 7, 5, 4
    spatial_shapes = ((11, 17), (6, 9))

    base_features = [
        torch.randn(
            batch_size, num_cams, embed_dims, height, width, device=device
        )
        for height, width in spatial_shapes
    ]
    base_location = torch.empty(
        batch_size, num_anchor, num_points, num_cams, 2, device=device
    ).uniform_(0.1, 0.9)
    base_weights = torch.randn(
        batch_size,
        num_anchor,
        num_points,
        num_cams,
        len(spatial_shapes),
        num_groups,
        device=device,
    )

    reference_features = [x.clone().requires_grad_() for x in base_features]
    cuda_features = [x.clone().requires_grad_() for x in base_features]
    reference_location = base_location.clone().requires_grad_()
    cuda_location = base_location.clone().requires_grad_()
    reference_weights = base_weights.clone().requires_grad_()
    cuda_weights = base_weights.clone().requires_grad_()

    reference_output = reference_aggregation(
        reference_features, reference_location, reference_weights
    )
    formatted = format_feature_maps(cuda_features)
    cuda_output = deformable_aggregation(
        formatted[0],
        formatted[1],
        formatted[2],
        cuda_location,
        cuda_weights,
    )
    output_error = max_error(reference_output, cuda_output)

    grad_seed = torch.randn_like(reference_output)
    reference_grads = torch.autograd.grad(
        reference_output,
        [*reference_features, reference_location, reference_weights],
        grad_seed,
    )
    cuda_grads = torch.autograd.grad(
        cuda_output,
        [*cuda_features, cuda_location, cuda_weights],
        grad_seed,
    )
    grad_errors = [
        max_error(reference_grad, cuda_grad)
        for reference_grad, cuda_grad in zip(reference_grads, cuda_grads)
    ]

    print(f"forward max_abs_error: {output_error:.8g}")
    for level_idx in range(len(spatial_shapes)):
        print(
            f"feature_level_{level_idx} grad max_abs_error: "
            f"{grad_errors[level_idx]:.8g}"
        )
    print(f"sampling_location grad max_abs_error: {grad_errors[-2]:.8g}")
    print(f"weights grad max_abs_error: {grad_errors[-1]:.8g}")

    if output_error > args.atol:
        raise AssertionError(
            f"CUDA forward error {output_error} exceeds atol={args.atol}"
        )
    if max(grad_errors) > args.grad_atol:
        raise AssertionError(
            f"CUDA backward error {max(grad_errors)} exceeds "
            f"grad_atol={args.grad_atol}"
        )
    print("Sparse4D CUDA forward/backward parity check passed.")


if __name__ == "__main__":
    main()
