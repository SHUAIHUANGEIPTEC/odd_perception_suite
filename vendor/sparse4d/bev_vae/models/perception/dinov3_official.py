import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from bev_vae.models.layers_cuda import MSDATransformer2D
from bev_vae.models.perception.resnet import SceneEncoder as _BaseSceneEncoder
from bev_vae.models.pos_encoding import PosEncoding2D, PosEncoding3D


_ARCH_TO_MODEL = {
    "vitb16": "dinov3_vitb16",
    "vitl16": "dinov3_vitl16",
}

_ARCH_TO_CHANNELS = {
    "vitb16": 768,
    "vitl16": 1024,
}

_ARCH_TO_DEPTH = {
    "vitb16": 12,
    "vitl16": 24,
}

_CONVNEXT_ARCH_TO_MODEL = {
    "convnext_base": "dinov3_convnext_base",
    "convnext_large": "dinov3_convnext_large",
}

_CONVNEXT_ARCH_TO_CHANNELS = {
    "convnext_base": (128, 256, 512, 1024),
    "convnext_large": (192, 384, 768, 1536),
}


def _infer_arch(arch: str, weight_path: Optional[str]) -> str:
    if arch != "auto":
        if arch not in _ARCH_TO_MODEL:
            raise ValueError(f"Unsupported DINOv3 arch: {arch}")
        return arch
    if weight_path is None:
        raise ValueError("arch='auto' requires a DINOv3 weight_path")
    name = os.path.basename(weight_path).lower()
    if "vitl16" in name:
        return "vitl16"
    if "vitb16" in name:
        return "vitb16"
    raise ValueError(f"Could not infer DINOv3 arch from weight_path: {weight_path}")


def _default_out_indices(depth: int, level_num: int) -> tuple[int, ...]:
    if level_num <= 0:
        raise ValueError("level_num must be positive")
    return tuple(max(0, round(depth * (i + 1) / level_num) - 1) for i in range(level_num))


def _resolve_weight_path(weight_path: Optional[str]) -> Optional[str]:
    if weight_path is None:
        return None
    candidates = [weight_path]
    _, ext = os.path.splitext(weight_path)
    if ext == "":
        candidates.append(weight_path + ".pth")
    elif ext == ".pth":
        candidates.append(weight_path[:-4])
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return weight_path


def _is_local_repo(repo_or_dir: Optional[str]) -> bool:
    return repo_or_dir is not None and Path(repo_or_dir).expanduser().exists()


def _load_official_model(
    model_name: str,
    weight_path: Optional[str],
    repo_or_dir: Optional[str],
    check_hash: bool,
) -> nn.Module:
    manual_local_load = weight_path is not None and os.path.exists(weight_path)
    loader_kwargs = {}
    # DINOv3 ViT hub functions consume check_hash themselves.  ConvNeXt hub
    # functions forward unknown kwargs into ConvNeXt.__init__, where
    # check_hash is invalid.
    if not model_name.startswith("dinov3_convnext_"):
        loader_kwargs["check_hash"] = check_hash

    if repo_or_dir is not None:
        if _is_local_repo(repo_or_dir):
            return _load_official_model_from_local_repo(
                model_name=model_name,
                weight_path=weight_path,
                repo_or_dir=repo_or_dir,
                check_hash=check_hash,
            )
        model = torch.hub.load(
            repo_or_dir,
            model_name,
            pretrained=weight_path is not None,
            weights=weight_path,
            **loader_kwargs,
        )
        return model

    try:
        from dinov3.hub import backbones

        model_fn = getattr(backbones, model_name)
        model = model_fn(
            pretrained=weight_path is not None and not manual_local_load,
            weights=None if manual_local_load else weight_path,
            **loader_kwargs,
        )
        if manual_local_load:
            _load_local_weights(model, weight_path)
        return model
    except ImportError:
        model = torch.hub.load(
            "facebookresearch/dinov3",
            model_name,
            pretrained=weight_path is not None and not manual_local_load,
            weights=None if manual_local_load else weight_path,
            **loader_kwargs,
        )
        if manual_local_load:
            _load_local_weights(model, weight_path)
        return model


def _load_official_model_from_local_repo(
    model_name: str,
    weight_path: Optional[str],
    repo_or_dir: str,
    check_hash: bool,
) -> nn.Module:
    repo = str(Path(repo_or_dir).expanduser().resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)

    _patch_torch_amp_for_dinov3()
    _patch_torch_dynamo_config_for_dinov3()

    from dinov3.hub import backbones

    model_fn = getattr(backbones, model_name)
    manual_local_load = weight_path is not None and os.path.exists(weight_path)
    loader_kwargs = {}
    if not model_name.startswith("dinov3_convnext_"):
        loader_kwargs["check_hash"] = check_hash
    model = model_fn(
        pretrained=weight_path is not None and not manual_local_load,
        weights=None if manual_local_load else weight_path,
        **loader_kwargs,
    )
    if manual_local_load:
        _load_local_weights(model, weight_path)
    return model


def _patch_torch_amp_for_dinov3() -> None:
    try:
        from torch.amp import custom_bwd, custom_fwd  # noqa: F401
        return
    except (ImportError, AttributeError):
        pass

    try:
        from torch.cuda.amp import custom_bwd as cuda_custom_bwd
        from torch.cuda.amp import custom_fwd as cuda_custom_fwd
    except (ImportError, AttributeError):
        return

    import importlib
    import types

    def custom_fwd(*args, **kwargs):
        kwargs.pop("device_type", None)
        return cuda_custom_fwd(*args, **kwargs)

    def custom_bwd(*args, **kwargs):
        kwargs.pop("device_type", None)
        return cuda_custom_bwd(*args, **kwargs)

    try:
        amp_module = importlib.import_module("torch.amp")
    except ImportError:
        amp_module = types.ModuleType("torch.amp")
        sys.modules["torch.amp"] = amp_module
        torch.amp = amp_module

    amp_module.custom_fwd = custom_fwd
    amp_module.custom_bwd = custom_bwd


def _patch_torch_dynamo_config_for_dinov3() -> None:
    try:
        import torch._dynamo.config as dynamo_config
    except Exception:
        return

    if hasattr(dynamo_config, "accumulated_cache_size_limit"):
        return

    for attr in ("_config", "_default"):
        cfg = getattr(dynamo_config, attr, None)
        if isinstance(cfg, dict):
            cfg["accumulated_cache_size_limit"] = 1024

    allowed_keys = getattr(dynamo_config, "_allowed_keys", None)
    if isinstance(allowed_keys, set):
        allowed_keys.add("accumulated_cache_size_limit")

    try:
        dynamo_config.accumulated_cache_size_limit = 1024
    except AttributeError:
        pass


def _load_local_weights(model: nn.Module, weight_path: str) -> None:
    ckpt = torch.load(weight_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)


def _group_count(channels: int) -> int:
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


def _make_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int,
    norm: Optional[str],
    activation: Optional[str],
    stride: int = 1,
) -> nn.Module:
    use_bias = norm is None or str(norm).lower() in ("none", "identity")
    layers: List[nn.Module] = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=use_bias,
        )
    ]
    if norm is not None and str(norm).lower() not in ("none", "identity"):
        norm_name = str(norm).upper()
        if norm_name == "GN":
            layers.append(nn.GroupNorm(_group_count(out_channels), out_channels))
        elif norm_name in ("BN", "SYNCBN"):
            layers.append(nn.BatchNorm2d(out_channels))
        else:
            raise ValueError(f"Unsupported FPN norm: {norm}")
    if activation is not None and str(activation).lower() not in ("none", "identity"):
        act_name = str(activation).upper()
        if act_name == "GELU":
            layers.append(nn.GELU())
        elif act_name == "RELU":
            layers.append(nn.ReLU(inplace=True))
        else:
            raise ValueError(f"Unsupported FPN activation: {activation}")
    return layers[0] if len(layers) == 1 else nn.Sequential(*layers)


class DINOv3OfficialBackbone(nn.Module):
    """Official DINOv3 ViT wrapper for BEV-VAE perception.

    The ViT internals are provided by the official DINOv3 model. This class only
    selects intermediate patch maps and returns them in the dict format expected
    by the existing BEV scene encoder.
    """

    def __init__(
        self,
        weight_path: Optional[str] = "/data/ckpt/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        arch: str = "auto",
        out_indices: Optional[Sequence[int]] = None,
        level_num: int = 3,
        repo_or_dir: Optional[str] = None,
        check_hash: bool = False,
        freeze: bool = False,
        checkpoint_blocks: bool = False,
    ) -> None:
        super().__init__()
        arch = _infer_arch(arch, weight_path)
        self.arch = arch
        self.out_channels = _ARCH_TO_CHANNELS[arch]
        self.out_indices = tuple(out_indices) if out_indices is not None else _default_out_indices(
            _ARCH_TO_DEPTH[arch],
            level_num,
        )
        if any(idx < 0 or idx >= _ARCH_TO_DEPTH[arch] for idx in self.out_indices):
            raise ValueError(f"out_indices must be in [0, {_ARCH_TO_DEPTH[arch] - 1}], got {self.out_indices}")
        self.freeze = freeze
        self.checkpoint_blocks = bool(checkpoint_blocks)

        resolved_weight_path = _resolve_weight_path(weight_path)
        self.model = _load_official_model(
            model_name=_ARCH_TO_MODEL[arch],
            weight_path=resolved_weight_path,
            repo_or_dir=repo_or_dir,
            check_hash=check_hash,
        )

        if self.checkpoint_blocks:
            if freeze:
                raise ValueError("checkpoint_blocks is unnecessary and unsupported when the backbone is frozen")
            try:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    CheckpointImpl,
                    checkpoint_wrapper,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "DINOv3 block checkpointing requires PyTorch checkpoint_wrapper support"
                ) from exc
            for block_idx, block in enumerate(self.model.blocks):
                self.model.blocks[block_idx] = checkpoint_wrapper(
                    block,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )

        if freeze:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True) -> "DINOv3OfficialBackbone":
        if self.freeze:
            return super().train(False)
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.model.get_intermediate_layers(
            x,
            n=self.out_indices,
            reshape=True,
            norm=True,
        )
        return {f"c{i + 2}": feat for i, feat in enumerate(outputs)}


class DINOv3OfficialConvNeXtBackbone(nn.Module):
    """Official DINOv3 ConvNeXt wrapper with native hierarchical outputs.

    Unlike the ViT/16 wrapper, the returned C2-C5 maps are produced by actual
    stride-4/8/16/32 stages.  ``patch_size`` must remain ``None`` in the
    official ConvNeXt implementation; setting it to 16 would resize every
    stage back to the ViT token grid and recreate the pseudo-pyramid problem.
    """

    def __init__(
        self,
        weight_path: Optional[str] = "/data/ckpt/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
        arch: str = "convnext_base",
        out_indices: Optional[Sequence[int]] = None,
        level_num: int = 4,
        repo_or_dir: Optional[str] = None,
        check_hash: bool = False,
        freeze: bool = False,
        checkpoint_blocks: bool = False,
    ) -> None:
        super().__init__()
        if arch not in _CONVNEXT_ARCH_TO_MODEL:
            raise ValueError(f"Unsupported DINOv3 ConvNeXt arch: {arch}")
        if level_num != 4:
            raise ValueError("DINOv3 ConvNeXt true FPN requires all four native stages")

        self.arch = arch
        self.out_channels = _CONVNEXT_ARCH_TO_CHANNELS[arch]
        self.out_indices = tuple(out_indices) if out_indices is not None else (0, 1, 2, 3)
        if self.out_indices != (0, 1, 2, 3):
            raise ValueError(
                "DINOv3 ConvNeXt true FPN expects out_indices=(0, 1, 2, 3), "
                f"got {self.out_indices}"
            )
        self.freeze = bool(freeze)
        self.checkpoint_blocks = bool(checkpoint_blocks)

        self.model = _load_official_model(
            model_name=_CONVNEXT_ARCH_TO_MODEL[arch],
            weight_path=_resolve_weight_path(weight_path),
            repo_or_dir=repo_or_dir,
            check_hash=check_hash,
        )
        if getattr(self.model, "patch_size", None) is not None:
            raise ValueError(
                "DINOv3 ConvNeXt patch_size must be None so intermediate stages keep "
                "their native spatial resolutions"
            )

        if self.checkpoint_blocks:
            if self.freeze:
                raise ValueError("checkpoint_blocks is unnecessary and unsupported when the backbone is frozen")
            try:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    CheckpointImpl,
                    checkpoint_wrapper,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "DINOv3 ConvNeXt block checkpointing requires PyTorch checkpoint_wrapper support"
                ) from exc
            for stage in self.model.stages:
                for block_idx, block in enumerate(stage):
                    stage[block_idx] = checkpoint_wrapper(
                        block,
                        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                    )

        if self.freeze:
            self.eval()
            self.requires_grad_(False)

    def train(self, mode: bool = True) -> "DINOv3OfficialConvNeXtBackbone":
        if self.freeze:
            return super().train(False)
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.model.get_intermediate_layers(
            x,
            n=self.out_indices,
            reshape=True,
            norm=True,
        )
        return {f"c{i + 2}": feat for i, feat in enumerate(outputs)}


class _SpatialDetailStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DINOv3OfficialSTABackbone(DINOv3OfficialBackbone):
    """DINOv3 ViT plus a BEV-oriented Spatial Tuning Adapter (STA-BEV).

    It follows DEIMv2 STA's central design: a convolutional spatial-prior branch
    runs in parallel with DINOv3, then each real-resolution detail map is
    concatenated with an aligned ViT semantic map and projected.  The original
    detector-oriented STA emits strides 8/16/32.  This BEV variant deliberately
    emits strides 4/8/16 to preserve the scene encoder and DD3D contracts and
    retain more small/distant-object evidence at 768x432.
    """

    def __init__(
        self,
        *args,
        sta_out_channels: int = 96,
        detail_channels: Sequence[int] = (24, 48, 96),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if len(self.out_indices) != 3:
            raise ValueError("STA-BEV expects exactly three ViT intermediate layers")
        if len(detail_channels) != 3 or any(int(ch) <= 0 for ch in detail_channels):
            raise ValueError("detail_channels must contain three positive channel counts")
        if sta_out_channels <= 0:
            raise ValueError("sta_out_channels must be positive")

        vit_channels = self.out_channels
        detail_channels = tuple(int(ch) for ch in detail_channels)
        self.sta_out_channels = int(sta_out_channels)
        self.out_channels = self.sta_out_channels

        self.detail_stem = nn.Sequential(
            nn.Conv2d(3, detail_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(detail_channels[0]), detail_channels[0]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.detail_stages = nn.ModuleList(
            [
                _SpatialDetailStage(detail_channels[0], detail_channels[1], stride=2),
                _SpatialDetailStage(detail_channels[1], detail_channels[2], stride=2),
            ]
        )
        self.fusion_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        vit_channels + channels,
                        self.sta_out_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.GroupNorm(_group_count(self.sta_out_channels), self.sta_out_channels),
                )
                for channels in detail_channels
            ]
        )
        if self.freeze:
            self.eval()
            self.requires_grad_(False)

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        """Return newly initialized parameters so they can use the neck LR."""
        modules = (
            self.detail_stem,
            self.detail_stages,
            self.fusion_projections,
        )
        for module in modules:
            yield from module.parameters()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        semantic_maps = self.model.get_intermediate_layers(
            x,
            n=self.out_indices,
            reshape=True,
            norm=True,
        )

        detail_maps = [self.detail_stem(x)]
        for stage in self.detail_stages:
            detail_maps.append(stage(detail_maps[-1]))

        outputs = {}
        for idx, (semantic, detail, fusion_proj) in enumerate(
            zip(semantic_maps, detail_maps, self.fusion_projections)
        ):
            semantic = F.interpolate(
                semantic,
                size=detail.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            outputs[f"c{idx + 2}"] = fusion_proj(torch.cat([semantic, detail], dim=1))
        return outputs


class CornerPool2D(nn.Module):
    """Directional context pooling used by the BEVFormer++ image neck.

    The two branches aggregate the top-left and bottom-right directional
    contexts used by CornerNet-style corner pooling.  The residual projection
    is zero-initialized so enabling the module starts from the original FPN
    rather than perturbing pretrained features on the first update.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
        )
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    @staticmethod
    def _reverse_cummax(x: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.flip(torch.cummax(torch.flip(x, dims=(dim,)), dim=dim).values, dims=(dim,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        top = torch.cummax(x, dim=-2).values
        bottom = self._reverse_cummax(x, dim=-2)
        left = torch.cummax(x, dim=-1).values
        right = self._reverse_cummax(x, dim=-1)
        pooled = torch.cat((top + left, bottom + right), dim=1)
        return x + self.fuse(pooled)


class SimpleDINOv3FPN(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int = 96,
        in_keys: Sequence[str] = ("c2", "c3", "c4"),
        scale_factors: Sequence[float] = (4.0, 2.0, 1.0),
        top_down: bool = True,
        norm: Optional[str] = None,
        activation: Optional[str] = None,
        output_levels: Optional[int] = None,
        corner_pool: bool = False,
    ) -> None:
        super().__init__()
        if len(in_keys) != len(scale_factors):
            raise ValueError("in_keys and scale_factors must have the same length")
        self.in_keys = tuple(in_keys)
        self.scale_factors = tuple(float(v) for v in scale_factors)
        self.top_down = top_down
        if isinstance(in_channels, int):
            lateral_in_channels = [in_channels] * len(self.in_keys)
        else:
            lateral_in_channels = [int(ch) for ch in in_channels]
            if len(lateral_in_channels) != len(self.in_keys):
                raise ValueError("in_channels and in_keys must have the same length")
        self.output_levels = len(self.in_keys) if output_levels is None else int(output_levels)
        if self.output_levels < 1:
            raise ValueError("output_levels must be positive")
        self.base_output_levels = min(self.output_levels, len(self.in_keys))
        self.lateral_convs = nn.ModuleList(
            [
                _make_conv_block(channels, out_channels, kernel_size=1, padding=0, norm=norm, activation=activation)
                for channels in lateral_in_channels
            ]
        )
        self.output_convs = nn.ModuleList(
            [
                _make_conv_block(out_channels, out_channels, kernel_size=3, padding=1, norm=norm, activation=activation)
                for _ in range(self.base_output_levels)
            ]
        )
        self.extra_level_convs = nn.ModuleList(
            [
                _make_conv_block(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm=norm,
                    activation=activation,
                )
                for _ in range(self.output_levels - self.base_output_levels)
            ]
        )
        self.corner_pool_blocks = nn.ModuleList(
            [
                CornerPool2D(out_channels) if corner_pool else nn.Identity()
                for _ in range(self.output_levels)
            ]
        )

    def _resize(self, x: torch.Tensor, scale_factor: float) -> torch.Tensor:
        if scale_factor == 1.0:
            return x
        size = (
            max(1, int(round(x.shape[-2] * scale_factor))),
            max(1, int(round(x.shape[-1] * scale_factor))),
        )
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, feats: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        for key, scale, lateral in zip(self.in_keys, self.scale_factors, self.lateral_convs):
            outputs.append(self._resize(lateral(feats[key]), scale))

        if self.top_down:
            for idx in range(len(outputs) - 2, -1, -1):
                outputs[idx] = outputs[idx] + F.interpolate(
                    outputs[idx + 1],
                    size=outputs[idx].shape[-2:],
                    mode="nearest",
                )

        outputs = [
            conv(feat)
            for conv, feat in zip(
                self.output_convs, outputs[: self.base_output_levels]
            )
        ]
        for extra_conv in self.extra_level_convs:
            outputs.append(extra_conv(outputs[-1]))
        return [
            corner_pool(feat)
            for corner_pool, feat in zip(self.corner_pool_blocks, outputs)
        ]


class SceneEncoder(_BaseSceneEncoder):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        block_num: int,
        head_num: int,
        level_num: int,
        point_num: int,
        dropout: int,
        bias: bool,
        ckpt: bool,
        image_size: List[int],
        scene_size: List[int],
        pc: List[int],
        fpn_in_keys: Optional[Sequence[str]] = None,
        fpn_scale_factors: Optional[Sequence[float]] = None,
        fpn_top_down: bool = True,
        fpn_norm: Optional[str] = None,
        fpn_activation: Optional[str] = None,
        fpn_in_channels: Optional[Sequence[int]] = None,
        fpn_output_levels: Optional[int] = None,
        corner_pool: bool = False,
    ) -> None:
        super().__init__(
            in_ch=in_ch,
            out_ch=out_ch,
            block_num=block_num,
            head_num=head_num,
            level_num=level_num,
            point_num=point_num,
            dropout=dropout,
            bias=bias,
            ckpt=ckpt,
            image_size=image_size,
            scene_size=scene_size,
            pc=pc,
        )
        if fpn_in_keys is None:
            fpn_in_keys = tuple(f"c{i + 2}" for i in range(level_num))
        if fpn_scale_factors is None:
            fpn_scale_factors = tuple(2.0 ** (level_num - i - 1) for i in range(level_num))
        self.fpn = SimpleDINOv3FPN(
            in_channels=in_ch if fpn_in_channels is None else fpn_in_channels,
            out_channels=out_ch,
            in_keys=fpn_in_keys,
            scale_factors=fpn_scale_factors,
            top_down=fpn_top_down,
            norm=fpn_norm,
            activation=fpn_activation,
            output_levels=fpn_output_levels,
            corner_pool=corner_pool,
        )


class SceneEncoder2D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        block_num: int,
        head_num: int,
        level_num: int,
        point_num: int,
        dropout: int,
        bias: bool,
        ckpt: bool,
        image_size: List[int],
        scene_size: List[int],
        pc: List[int],
        z_anchor_num: int = 4,
        z_anchors: Optional[Sequence[float]] = None,
        fpn_in_keys: Optional[Sequence[str]] = None,
        fpn_scale_factors: Optional[Sequence[float]] = None,
        fpn_top_down: bool = True,
        fpn_norm: Optional[str] = None,
        fpn_activation: Optional[str] = None,
        fpn_in_channels: Optional[Sequence[int]] = None,
        fpn_output_levels: Optional[int] = None,
        corner_pool: bool = False,
    ) -> None:
        super().__init__()
        if z_anchor_num < 1:
            raise ValueError("z_anchor_num must be positive")

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.image_size = image_size
        self.scene_size = scene_size
        self.pc = pc
        self.z_anchor_num = int(z_anchor_num)
        if point_num % self.z_anchor_num != 0:
            raise ValueError(
                "point_num must be divisible by z_anchor_num for BEV pillar "
                f"sampling, got {point_num} and {self.z_anchor_num}"
            )

        _, h, w = scene_size
        self.scene_embedding = nn.Embedding(h * w, out_ch)
        self.pos_embedding = PosEncoding2D(out_ch)
        self.register_buffer("scene_3d", self.get_scene_point(self.z_anchor_num, h, w, pc, z_anchors))

        if fpn_in_keys is None:
            fpn_in_keys = tuple(f"c{i + 2}" for i in range(level_num))
        if fpn_scale_factors is None:
            fpn_scale_factors = tuple(2.0 ** (level_num - i - 1) for i in range(level_num))
        self.fpn = SimpleDINOv3FPN(
            in_channels=in_ch if fpn_in_channels is None else fpn_in_channels,
            out_channels=out_ch,
            in_keys=fpn_in_keys,
            scale_factors=fpn_scale_factors,
            top_down=fpn_top_down,
            norm=fpn_norm,
            activation=fpn_activation,
            output_levels=fpn_output_levels,
            corner_pool=corner_pool,
        )
        self.transformer = MSDATransformer2D(
            out_ch, block_num, head_num, level_num, point_num, dropout, bias, ckpt)

    @staticmethod
    def get_scene_point(D, H, W, pc, z_anchors: Optional[Sequence[float]] = None):
        y = torch.linspace(0.5, H - 0.5, H).view(H, 1).repeat(1, W) / H
        x = torch.linspace(0.5, W - 0.5, W).view(1, W).repeat(H, 1) / W
        if z_anchors is None:
            z = torch.linspace(0.5, D - 0.5, D) / D
        else:
            if len(z_anchors) != D:
                raise ValueError(f"z_anchors length {len(z_anchors)} must match z_anchor_num={D}")
            z = torch.tensor(list(z_anchors), dtype=torch.float32)
            pc_tensor = torch.tensor(pc, dtype=torch.float32)
            z = (z - pc_tensor[2]) / (pc_tensor[5] - pc_tensor[2]).clamp(min=1e-6)
            z = z.clamp(1e-4, 1.0 - 1e-4)
        xy = torch.stack([x, y], dim=-1).reshape(H * W, 1, 2).expand(-1, D, -1)
        z = z.reshape(1, D, 1).expand(H * W, -1, -1)
        return torch.cat([xy, z], dim=-1).contiguous()

    def encode_image_features(self, x, img_metas):
        img_feats = self.fpn(x)
        img_feats = [rearrange(x, "(b v) ... -> b v ...", v=img_metas["cam_num"]) for x in img_feats]
        return img_feats

    def forward(self, x, img_metas, return_img_feats: bool = False):
        img_feats = self.encode_image_features(x, img_metas)
        b = img_feats[0].shape[0]
        _, h, w = self.scene_size
        d = self.z_anchor_num

        query = self.scene_embedding.weight.reshape(h, w, -1)[None].repeat(b, 1, 1, 1)
        query = query + self.pos_embedding(query)
        query = query.flatten(1, 2)

        shape, value = [], []
        for feat in img_feats:
            shape.append(feat.shape[-2:])
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            value.append(feat)
        shape = torch.tensor(shape, device=query.device)
        value = torch.cat(value, 2).permute(0, 2, 1, 3)

        # point_sampling converts normalized coordinates to metric coordinates
        # in-place, so each batch item needs independent storage.
        point = self.scene_3d[None].repeat(b, 1, 1, 1)
        point, valid = self.point_sampling(
            point.reshape(b, h * w * d, 3), img_metas
        )
        point = point.reshape(point.shape[0], b, h * w, d, 2)
        valid = valid.reshape(valid.shape[0], b, h * w, d)
        scene = self.transformer(query, value, point, valid, shape)
        scene = scene.reshape(b, h, w, self.out_ch)
        scene = scene.permute(0, 3, 1, 2).unsqueeze(2).contiguous()
        if return_img_feats:
            return scene, img_feats
        return scene

    def point_sampling(self, point, img_metas, eps=1e-6):
        return _BaseSceneEncoder.point_sampling(self, point, img_metas, eps=eps)
