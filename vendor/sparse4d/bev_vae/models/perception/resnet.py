from typing import List


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import DeformConv2d
from einops import rearrange
from bev_vae.models.layers_cuda import MSDATransformer2D
from bev_vae.models.pos_encoding import PosEncoding3D


def normalize_resnet_state_dict(state_dict, target_state_dict):
    """Keep only backbone keys and adapt common MMDet/DCN checkpoint names."""
    target_keys = set(target_state_dict.keys())
    normalized = {}
    prefixes = ("module.", "model.", "img_backbone.", "backbone.")

    for key, value in state_dict.items():
        name = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    changed = True

        candidates = [name]
        if ".conv2.weight" in name:
            candidates.append(name.replace(".conv2.weight", ".conv2.conv.weight"))
        if ".conv2.bias" in name:
            candidates.append(name.replace(".conv2.bias", ".conv2.conv.bias"))
        if ".conv2.conv_offset.weight" in name:
            candidates.append(name.replace(".conv2.conv_offset.weight", ".conv2.offset_mask.weight"))
        if ".conv2.conv_offset.bias" in name:
            candidates.append(name.replace(".conv2.conv_offset.bias", ".conv2.offset_mask.bias"))

        for candidate in candidates:
            if candidate not in target_keys:
                continue
            if target_state_dict[candidate].shape != value.shape:
                continue
            normalized[candidate] = value
            break

    return normalized


class DeformConv2dPack(nn.Module):
    def __init__(self, conv: nn.Conv2d, deform_groups: int = 1):
        super().__init__()
        self.conv = DeformConv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
        )
        self.conv.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            self.conv.bias.data.copy_(conv.bias.data)
        kernel_h, kernel_w = conv.kernel_size
        offset_channels = deform_groups * 2 * kernel_h * kernel_w
        mask_channels = deform_groups * kernel_h * kernel_w
        self.offset_mask = nn.Conv2d(
            conv.in_channels,
            offset_channels + mask_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
        )
        nn.init.constant_(self.offset_mask.weight, 0.0)
        nn.init.constant_(self.offset_mask.bias, 0.0)
        with torch.no_grad():
            self.offset_mask.bias[offset_channels:].fill_(10.0)
        self.offset_channels = offset_channels

    def forward(self, x):
        offset_mask = self.offset_mask(x)
        offset = offset_mask[:, :self.offset_channels]
        mask = offset_mask[:, self.offset_channels:].sigmoid()
        return self.conv(x, offset, mask)


def replace_bottleneck_conv2_with_dcn(layer: nn.Module, deform_groups: int) -> None:
    for block in layer:
        if hasattr(block, "conv2") and isinstance(block.conv2, nn.Conv2d):
            block.conv2 = DeformConv2dPack(block.conv2, deform_groups=deform_groups)


class ResNetBackbone(nn.Module):
    def __init__(
        self,
        weight_path="/data/ckpt/resnet50-0676ba61.pth",
        depth: int = 50,
        dcn_stages=None,
        dcn_deform_groups: int = 1,
    ):
        super().__init__()

        if depth == 50:
            model = models.resnet50(weights=None)
        elif depth == 101:
            model = models.resnet101(weights=None)
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")

        # ===== 拆 backbone（方便多尺度输出）=====
        dcn_stages = dcn_stages or [False, False, False, False]
        if len(dcn_stages) != 4:
            raise ValueError("dcn_stages must have four booleans for layer1-layer4")
        for use_dcn, layer in zip(dcn_stages, [model.layer1, model.layer2, model.layer3, model.layer4]):
            if use_dcn:
                replace_bottleneck_conv2_with_dcn(layer, dcn_deform_groups)

        if weight_path is not None:
            ckpt = torch.load(weight_path, map_location="cpu")
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]

            ckpt = normalize_resnet_state_dict(ckpt, model.state_dict())
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print(
                f"[ResNetBackbone] Loaded {len(ckpt)} tensors from: {weight_path}; "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1  # C2
        self.layer2 = model.layer2  # C3
        self.layer3 = model.layer3  # C4
        self.layer4 = model.layer4  # C5

    def forward(self, x):
        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 多尺度特征
        c2 = self.layer1(x)  # stride 4
        c3 = self.layer2(c2) # stride 8
        c4 = self.layer3(c3) # stride 16
        c5 = self.layer4(c4) # stride 32

        return {
            "c2": c2,
            "c3": c3,
            "c4": c4,
            "c5": c5,
        }
    

class SimpleFPN(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024, 2048], out_channels=96):
        super().__init__()

        # lateral conv（统一通道）
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels[i], out_channels, kernel_size=1)
            for i in range(4)
        ])

        # output conv（平滑）
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(3)
        ])

    def forward(self, feats):
        """
        feats: dict with keys c2, c3, c4, c5
        """

        c2, c3, c4, c5 = feats["c2"], feats["c3"], feats["c4"], feats["c5"]

        # ===== lateral =====
        l2 = self.lateral_convs[0](c2)
        l3 = self.lateral_convs[1](c3)
        l4 = self.lateral_convs[2](c4)
        l5 = self.lateral_convs[3](c5)

        # ===== top-down =====
        p4 = l4 + F.interpolate(l5, size=l4.shape[-2:], mode='nearest')
        p3 = l3 + F.interpolate(p4, size=l3.shape[-2:], mode='nearest')
        p2 = l2 + F.interpolate(p3, size=l2.shape[-2:], mode='nearest')

        # ===== smooth =====
        p2 = self.output_convs[0](p2)
        p3 = self.output_convs[1](p3)
        p4 = self.output_convs[2](p4)
        return [p2, p3, p4]


class SceneEncoder(nn.Module):
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
        ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.image_size = image_size
        self.scene_size = scene_size
        self.pc = pc
        self.scene_embedding = nn.Embedding(scene_size[1] * scene_size[2], out_ch)
        self.pos_embedding = PosEncoding3D(out_ch)
        self.register_buffer("scene_3d", self.get_scene_point(*scene_size))
        self.fpn = SimpleFPN(out_channels=out_ch)
        self.transformer = MSDATransformer2D(
            out_ch, block_num, head_num, level_num, point_num, dropout, bias, ckpt)
        
    @staticmethod
    def get_scene_point(D, H, W):
        z = torch.linspace(0.5, D - 0.5, D).view(D, 1, 1).repeat(1, H, W) / D
        y = torch.linspace(0.5, H - 0.5, H).view(1, H, 1).repeat(D, 1, W) / H
        x = torch.linspace(0.5, W - 0.5, W).view(1, 1, W).repeat(D, H, 1) / W
        point = torch.stack([x, y, z]).flatten(1).permute(1, 0).contiguous() # (DHW) xyz
        return point 

    def encode_image_features(self, x, img_metas):
        img_feats = self.fpn(x)
        img_feats = [rearrange(x, "(b v) ... -> b v ...", v=img_metas["cam_num"]) for x in img_feats]
        return img_feats

    def forward(self, x, img_metas, return_img_feats: bool = False):
        img_feats = self.encode_image_features(x, img_metas)
        b = img_feats[0].shape[0]
        d, h, w = self.scene_size
        query = self.scene_embedding.weight.reshape(h, w, -1)[None, None].repeat(b, d, 1, 1, 1)
        query += self.pos_embedding(query) # B D H W E
        query = query.flatten(1, 3) # B (DHW) E

        shape, value = [], []
        for lvl, feat in enumerate(img_feats):
            shape.append(feat.shape[-2:])
            feat = feat.flatten(3).permute(1, 0, 3, 2) # B V E H W -> B V E (HW) -> V B (HW) E
            value.append(feat)
        shape = torch.tensor(shape, device=query.device)
        value = torch.cat(value, 2).permute(0, 2, 1, 3) # V B sum(HW) E -> V sum(HW) B E 

        point = self.scene_3d[None].repeat(b, 1, 1) # B (DHW) xyz
        point, valid = self.point_sampling(point, img_metas) # point: V B (DHW) xy , valid: V B (DHW)
        scene = self.transformer(query, value, point, valid, shape)

        scene = scene.permute(0, 2, 1).reshape(b, self.out_ch, *self.scene_size) # B E D H W
        if return_img_feats:
            return scene, img_feats
        return scene

    def point_sampling(self, point, img_metas, eps=1e-6):
        _, dhw, _ = point.shape # B (DHW) 3
        point[..., 0:1] = point[..., 0:1] * (self.pc[3] - self.pc[0]) + self.pc[0]
        point[..., 1:2] = point[..., 1:2] * (self.pc[4] - self.pc[1]) + self.pc[1]
        point[..., 2:3] = point[..., 2:3] * (self.pc[5] - self.pc[2]) + self.pc[2]
        point = torch.cat([point, torch.ones_like(point[..., :1])], -1) # B (DHW) 4
        point = point[:, None].repeat(1, img_metas["cam_num"], 1, 1)[..., None] # B V (DHW) 4 1
        point = torch.inverse(img_metas["extrinsic"].float())[:, :, None].repeat(1, 1, dhw, 1, 1).to(point.dtype) @ point
        point = img_metas["intrinsic"][:, :, None].repeat(1, 1, dhw, 1, 1) @ point[..., :3, :]
        valid_z = point.squeeze(-1)[..., -1] > eps
        # Same geometrics in a batch is necessary, for valid points is same between different samples.
        point = img_metas["geometric"][:, :, None].repeat(1, 1, dhw, 1, 1) @ point
        point = point.squeeze(-1) # B V (DHW) 3
        # B V (DHW) 2 by x/z and y/z
        point = point[..., :2] / torch.maximum(point[..., -1:], torch.ones_like(point[..., -1:]) * eps)
        point[..., 0] /=  self.image_size[1] # x / W
        point[..., 1] /=  self.image_size[0] # y / H
        valid_x = (0. < point[..., 0]) & (point[..., 0] < 1.)
        valid_y = (0. < point[..., 1]) & (point[..., 1] < 1.)
        valid = valid_x & valid_y & valid_z
        point = point.permute(1, 0, 2, 3) # V B (DHW) 2
        valid = valid.permute(1, 0, 2) # V B (DHW)
        return point, valid # xy    
