import math
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F

from bev_vae.models.msda.multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp16, MultiScaleDeformableAttnFunction_fp32, MultiScaleDeformableAttnFunction3D_fp16, MultiScaleDeformableAttnFunction3D_fp32

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, embed_dim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-6)


class MLP(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        dropout: float,
        bias: bool) -> None:
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class MoEFFN(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dropout: float,
        bias: bool,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_mult: int = 4,
        route_noise: float = 0.0,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")

        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.route_noise = float(route_noise)
        hidden_dim = int(hidden_mult) * embed_dim
        self.gate = nn.Linear(embed_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim, bias=bias),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim, bias=bias),
                nn.Dropout(dropout),
            )
            for _ in range(self.num_experts)
        ])
        self.last_aux_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(x)
        if self.training and self.route_noise > 0:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * self.route_noise

        gate_probs = gate_logits.softmax(-1)
        top_probs, top_indices = gate_probs.topk(self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(-1, keepdim=True).clamp_min(1e-6)

        out = x.new_zeros(x.shape)
        for expert_idx, expert in enumerate(self.experts):
            expert_weight = (top_probs * (top_indices == expert_idx).to(top_probs.dtype)).sum(-1, keepdim=True)
            out = out + expert(x) * expert_weight

        importance = gate_probs.mean(dim=(0, 1))
        load = F.one_hot(top_indices, num_classes=self.num_experts).to(gate_probs.dtype).sum(-2)
        load = load.mean(dim=(0, 1)) / float(self.top_k)
        self.last_aux_loss = self.num_experts * torch.sum(importance * load)
        return out


class RearrangeContiguous(nn.Module):
    def __init__(self, pattern, **kwargs):
        super().__init__()
        self.rearrange = Rearrange(pattern, **kwargs)

    def forward(self, x):
        return self.rearrange(x).contiguous()


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=patch_size, stride=patch_size),
            RearrangeContiguous('b e h w -> b h w e'))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
    

class DePatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            RearrangeContiguous('b h w e -> b e h w'),
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=patch_size, stride=patch_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PatchSceneEmbed(nn.Module):
    def __init__(self, height: int, patch_size: int, in_ch: int, ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_ch, ch, 1), nn.GELU(),
            RearrangeContiguous('b e d h w -> (b d) e h w'),
            nn.Conv2d(ch, out_ch, kernel_size=patch_size, stride=patch_size),
            RearrangeContiguous('(b d) e h w -> b d h w e', d=height))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
    

class DePatchSceneEmbed(nn.Module):
    def __init__(self, height: int, patch_size: int, in_ch: int, ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            RearrangeContiguous('b d h w e -> (b d) e h w'),
            nn.ConvTranspose2d(in_ch, ch, kernel_size=patch_size, stride=patch_size),
            RearrangeContiguous('(b d) e h w -> b e d h w', d=height),
            nn.GELU(), nn.Conv3d(ch, out_ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)  


class MultiScaleDefAttn2D(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_num: int,
        level_num: int,
        point_num: int,
        dropout: float,
        bias: bool) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.head_num = head_num
        self.level_num = level_num
        self.point_num = point_num
        self.dropout = nn.Dropout(dropout)
        self.offset = nn.Linear(embed_dim, head_num * level_num * point_num *  2, bias)
        self.weight = nn.Linear(embed_dim, head_num * level_num * point_num, bias)
        # self.value_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias)

    def forward(
        self, 
        query: torch.Tensor, 
        value: torch.Tensor, 
        point: torch.Tensor, 
        valid: torch.Tensor, 
        shape: List[torch.Tensor]) -> torch.Tensor:
        cam_num = valid.shape[0]
        # query: B Q E | value: V sum(imgHW) B E
        # point: V B Q xy or V B Q R xy, where R is the number of z anchors.
        # valid follows point without the final xy dimension.
        multi_anchor = point.ndim == 5
        query_valid = valid.any(dim=-1) if multi_anchor else valid
        # Every sample has its own ego pose and image/3D augmentation. Rebatch
        # the union of visible queries, then mask each sample independently.
        indexes = [
            valid_per_view.any(dim=0).nonzero().squeeze(-1)
            for valid_per_view in query_valid
        ]
        query_num = max([len(valid_per_view) for valid_per_view in indexes]) 
        if query_num == 0:
            return query * 0 + value.sum() * 0
        b, _, _ = query.shape # B (DHW) E
        query_rebatch = query.new_zeros([b, cam_num, query_num, self.embed_dim])
        if multi_anchor:
            anchor_num = point.shape[-2]
            point_rebatch = point.new_full(
                [b, cam_num, query_num, anchor_num, 2], -2.0
            )
        else:
            point_rebatch = point.new_zeros([b, cam_num, query_num, 2])
        for v in range(cam_num):
            query_rebatch[:, v, :len(indexes[v])] = query[:, indexes[v]]
            selected_point = point[v, :, indexes[v]]
            if multi_anchor:
                selected_valid = valid[v, :, indexes[v]]
                selected_point = torch.where(
                    selected_valid[..., None],
                    selected_point,
                    selected_point.new_full((), -2.0),
                )
            point_rebatch[:, v, :len(indexes[v])] = selected_point
        _, value_num, _, _ = value.shape # V sum(imgHW) B C 
        output = self.cross_attn(
            query_rebatch.reshape(b * cam_num, query_num, self.embed_dim), # B V (DHW) E -> BV (DHW) E
            value.permute(2, 0, 1, 3).reshape(b * cam_num, value_num, self.embed_dim), #  B V sum(imgHW) E -> BV sum(imgHW) E
            point_rebatch.reshape(
                b * cam_num, query_num, *point_rebatch.shape[3:]
            ),
            shape).reshape(b, cam_num, query_num, self.embed_dim)
        slots = torch.zeros_like(query) # B (DHW) E
        for v in range(cam_num):
            slots[:, indexes[v]] += output[:, v, :len(indexes[v])] * query_valid[v, :, indexes[v], None]
        count = query_valid.permute(1, 2, 0).sum(-1).clamp(min=1.0).to(slots.dtype)
        slots = self.proj(slots / count[..., None])
        return self.dropout(slots) 

    def cross_attn(
        self, 
        query: torch.Tensor, 
        value: torch.Tensor, 
        point: torch.Tensor, 
        shape: torch.Tensor) -> torch.Tensor: # 注意：原类型提示写的是 List，但实际传入的是 Tensor
        
        # 1. 提取基础维度
        b_cam, query_num, _ = query.shape
        _, value_num, _ = value.shape
        e = self.embed_dim // self.head_num
        assert shape.prod(1).sum() == value_num

        # 2. 预测 offset 和 weight
        offset = self.offset(query).reshape(
            b_cam, query_num, self.head_num, self.level_num, self.point_num, 2)
        weight = self.weight(query).reshape(
            b_cam, query_num, self.head_num, self.level_num * self.point_num).softmax(-1)
        weight = weight.reshape(
            b_cam, query_num, self.head_num, self.level_num, self.point_num)
            
        shape_xyz = shape.flip(-1).contiguous()
        
        if point.ndim == 3:
            location = (
                point[:, :, None, None, None, :]
                + offset / shape_xyz[None, None, None, :, None, :]
            )
        elif point.ndim == 4:
            anchor_num = point.shape[-2]
            if self.point_num % anchor_num != 0:
                raise ValueError(
                    "point_num must be divisible by the number of z anchors: "
                    f"{self.point_num} vs {anchor_num}"
                )
            points_per_anchor = self.point_num // anchor_num
            offset = offset.reshape(
                b_cam,
                query_num,
                self.head_num,
                self.level_num,
                anchor_num,
                points_per_anchor,
                2,
            )
            location = (
                point[:, :, None, None, :, None, :]
                + offset
                / shape_xyz[None, None, None, :, None, None, :]
            )
            location = location.reshape(
                b_cam,
                query_num,
                self.head_num,
                self.level_num,
                self.point_num,
                2,
            )
        else:
            raise ValueError(
                "point must have shape (B,Q,2) or (B,Q,R,2), "
                f"got {tuple(point.shape)}"
            )
        '''
        (N, Q, 1, 1, 1, 2)
                │  │  │
                │  │  point_num
                │  level_num
                head_num
        location: (N, Q, M, L, P, 2)
       '''

        # ---------------------------------------------------------
        # 以下为 CUDA 接口准备数据
        # ---------------------------------------------------------

        # A. 准备 value: (b_cam, value_num, head_num, e)
        value_cuda = value.view(b_cam, value_num, self.head_num, e)

        # B. 准备 spatial_shapes: (num_levels, 2)
        spatial_shapes_cuda = shape.long()

        # C. 准备 level_start_index: (num_levels,) 
        # 原生没有这个参数，需要用累加算出来
        level_start_index_cuda = torch.cat((
            spatial_shapes_cuda.new_zeros((1, )),
            spatial_shapes_cuda.prod(1).cumsum(0)[:-1]
        ))
        
        # 动态判断传入数据的精度类型！
        if value.dtype == torch.float16:
            msda_func = MultiScaleDeformableAttnFunction_fp16
        elif value.dtype == torch.float32:
            msda_func = MultiScaleDeformableAttnFunction_fp32
        else:
            raise ValueError(f"不支持的数据精度: {value.dtype}，必须为 FP16 或 FP32")

        # 4. 调用优化的 CUDA 算子
        if not value_cuda.is_cuda:
            from bev_vae.models.msda.multi_scale_deformable_attn_function import multi_scale_deformable_attn_reference
            return multi_scale_deformable_attn_reference(
                value_cuda, spatial_shapes_cuda, location, weight)
        output = msda_func.apply(
            value_cuda.contiguous(),
            spatial_shapes_cuda.contiguous(),
            level_start_index_cuda.contiguous(),
            location.contiguous(),
            weight.contiguous(),
            min(b_cam, 64) # im2col_step: 设定一个安全的截断值防溢出
        )
        '''
        value_cuda:          (B*V, S, M, C)
        spatial_shapes_cuda: (L, 2)
        level_start_index:   (L,)
        location:            (B*V, Qv, M, L, P, 2)
        weight:              (B*V, Qv, M, L, P)
        output:              (B*V, Qv, E)
        '''

        return output
    

class MSDABlock2D(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        head_num: int,
        level_num: int,
        point_num: int,
        dropout: float,
        bias: bool) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(embed_dim, bias)
        self.ln_v = LayerNorm(embed_dim, bias)
        self.attn = MultiScaleDefAttn2D(embed_dim, head_num, level_num, point_num, dropout, bias)
        self.ln_2 = LayerNorm(embed_dim, bias)
        self.mlp = MLP(embed_dim, dropout, bias)

    def forward(
        self, 
        query: torch.Tensor, 
        value: torch.Tensor, 
        point: torch.Tensor, 
        valid: torch.Tensor, 
        shape: List[torch.Tensor]) -> torch.Tensor:
        query = query + self.attn(self.ln_1(query), self.ln_v(value), point, valid, shape)
        query = query + self.mlp(self.ln_2(query))
        return query
    

class MSDATransformer2D(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        block_num: int,
        head_num: int,
        level_num: int,
        point_num: int,
        dropout: float,
        bias: bool,
        ckpt: bool) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            MSDABlock2D(embed_dim, head_num, level_num, point_num, dropout, bias) for _ in range(block_num)])
        self.norm = LayerNorm(embed_dim, bias)
        self.embed_dim = embed_dim
        self.ckpt = ckpt

    def forward(
        self, 
        query: torch.Tensor, 
        value: torch.Tensor, 
        point: torch.Tensor, 
        valid: torch.Tensor, 
        shape: List[torch.Tensor]) -> torch.Tensor:
        for block in self.blocks:
            if self.ckpt and self.training:
                query = checkpoint.checkpoint(block, query, value, point, valid, shape, use_reentrant=False) # query : (B, DHW, E) | value: (V, sum(HW), B, E) | point: (V, B, DHW, 2) | valid: (V, B, DHW) | shape: (L, 2)
            else:
                query = block(query, value, point, valid, shape)
        return self.norm(query)


class MultiScaleDefAttn3D(nn.Module):
    def __init__(self,
        embed_dim: int,
        head_num: int,
        level_num: int,
        point_num: int,
        depth_num: int,
        dropout: float,
        bias: bool) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.head_num = head_num
        self.level_num = level_num
        self.point_num = point_num
        self.depth_num = depth_num
        self.dropout = nn.Dropout(dropout)
        self.offset = nn.Linear(embed_dim, head_num * level_num * point_num * 3, bias)
        self.weight = nn.Linear(embed_dim, head_num * level_num * point_num, bias)
        self.depth = nn.Linear(embed_dim, 1, bias)
        # self.value_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias)

    def forward(
        self, 
        query: torch.Tensor, 
        value: torch.Tensor, 
        point: torch.Tensor, 
        valid: torch.Tensor, 
        shape: List[torch.Tensor]) -> torch.Tensor: 
        # query: B V (DHW) E | value: sum(scnDHW) B E | point: V B (DHW) xyz | valid: V B (DHW)
        # different sample in a batch must be same geom augmented in the same view 
        cam_num = valid.shape[0]
        indexes = [valid_per_view[0].nonzero().squeeze(-1) for valid_per_view in valid]
        query_num = max([len(valid_per_view) for valid_per_view in indexes])
        b, _, _, _ = query.shape # B V (DHW) E
        query_rebatch = query.new_zeros([b, cam_num, query_num, self.embed_dim])
        point_rebatch = point.new_zeros([b, cam_num, query_num, 3])
        for v in range(cam_num):
            query_rebatch[:, v, :len(indexes[v])] = query[:, v, indexes[v]]
            point_rebatch[:, v, :len(indexes[v])] = point[v, :, indexes[v]]
        output = self.cross_attn(
            query_rebatch.reshape(b * cam_num, query_num, self.embed_dim),
            value.permute(1, 0, 2),
            point_rebatch.reshape(b * cam_num, query_num, 3),
            shape).reshape(b, cam_num, query_num, self.embed_dim)
        slots = torch.zeros_like(query) # B V (DHW) E
        for v in range(cam_num):
            slots[:, v, indexes[v]] += output[:, v, :len(indexes[v])]
        slots = rearrange(slots, "b v (d q) e -> b v q e d", d = self.depth_num)
        depth = self.depth(query) # B V (DHW) 1
        depth = depth.masked_fill(~valid.permute(1, 0, 2)[..., None], -torch.finfo(depth.dtype).max)
        depth = rearrange(depth, "b v (d q) e -> b v q e d ", d=self.depth_num)
        depth = depth.softmax(-1)
        slots = self.proj((slots * depth).sum(-1)) # B V Q E
        slots = slots[:, :, None].repeat(1, 1, self.depth_num, 1, 1).flatten(2, 3)
        return self.dropout(slots) 

    def cross_attn(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        point: torch.Tensor,
        shape: List[torch.Tensor]) -> torch.Tensor:
        
        # 1. 提取基础维度
        b_cam, query_num, _ = query.shape
        b, value_num, _ = value.shape
        cam_num = b_cam // b
        e = self.embed_dim // self.head_num
        assert shape.prod(1).sum() == value_num

        # 2. 预测 offset 和 weight (原作者逻辑)
        offset = self.offset(query).reshape(
            b_cam, query_num, self.head_num, self.level_num, self.point_num, 3)
        weight = self.weight(query).reshape(
            b_cam, query_num, self.head_num, self.level_num * self.point_num).softmax(-1)
        weight = weight.reshape(
            b_cam, query_num, self.head_num, self.level_num, self.point_num)
            
        # 3. 计算绝对采样坐标 (x, y, z -> w, h, d 映射)
        # location = point[:, :, None, None, None, :] + offset / shape[None, None, None, :, None, :].flip(-1)
        # 先对极小的 2D shape 进行 flip 并强制内存连续！
        # shape_xyz 的大小仅仅是 (level_num, 3)，连续化开销几乎为 0
        shape_xyz = shape.flip(-1).contiguous()
        
        # 然后再进行高维广播和数学运算，此时 GPU 的访存将是极其顺滑的线性读取！
        location = point[:, :, None, None, None, :] + offset / shape_xyz[None, None, None, :, None, :]
        # ---------------------------------------------------------
        # 以下为 3D CUDA 接口的“数据翻译层 (Bridging)”
        # ---------------------------------------------------------

        # A. 维度折叠：巧妙化解 cam_num 不匹配问题
        # 将 query 侧的变量从 (b_cam, Q, ...) 变为 (b, cam_num * Q, ...)
        location = location.view(b, cam_num * query_num, self.head_num, self.level_num, self.point_num, 3)
        weight = weight.view(b, cam_num * query_num, self.head_num, self.level_num, self.point_num)
        
        # B. 准备 value: (b, value_num, head_num, e)
        value_cuda = value.view(b, value_num, self.head_num, e)

        # C. 准备 spatial_shapes: (num_levels, 3) 对应 [d, h, w]
        spatial_shapes_cuda = shape.long()
        level_start_index_cuda = torch.cat((
            spatial_shapes_cuda.new_zeros((1, )),
            spatial_shapes_cuda.prod(1).cumsum(0)[:-1]
        ))

        # 动态判断传入数据的精度类型！
        if value.dtype == torch.float16:
            msda_func = MultiScaleDeformableAttnFunction3D_fp16
        elif value.dtype == torch.float32:
            msda_func = MultiScaleDeformableAttnFunction3D_fp32
        else:
            raise ValueError(f"不支持的数据精度: {value.dtype}，必须为 FP16 或 FP32")

        # 调用 3D CUDA 算子
        output = msda_func.apply(
            value_cuda.contiguous(),
            spatial_shapes_cuda.contiguous(),
            level_start_index_cuda.contiguous(),
            location.contiguous(),
            weight.contiguous(),
            min(b, 64) 
        )

        # E. 恢复输出维度: 从 (b, cam_num * Q, E) 变回 (BV, Q, E)
        output = output.view(b_cam, query_num, self.embed_dim)
        return output
    

class MSDABlock3D(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        head_num: int,
        level_num: int,
        point_num: int,
        depth_num: int,
        dropout: float,
        bias: bool) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(embed_dim, bias)
        self.ln_v = LayerNorm(embed_dim, bias)
        self.attn = MultiScaleDefAttn3D(embed_dim, head_num, level_num, point_num, depth_num, dropout, bias)
        self.ln_2 = LayerNorm(embed_dim, bias)
        self.mlp = MLP(embed_dim, dropout, bias)

    def forward(
        self, 
        query: torch.Tensor, 
        value: torch.Tensor, 
        point: torch.Tensor, 
        valid: torch.Tensor, 
        shape: List[torch.Tensor]) -> torch.Tensor:
        query = query + self.attn(self.ln_1(query), self.ln_v(value), point, valid, shape)
        query = query + self.mlp(self.ln_2(query))
        return query


class MSDATransformer3D(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        block_num: int,
        head_num: int,
        level_num: int,
        point_num: int,
        depth_num: int,
        dropout: float,
        bias: bool,
        ckpt: bool) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            MSDABlock3D(embed_dim, head_num, level_num, point_num, depth_num, dropout, bias) for _ in range(block_num)])
        self.depth_num = depth_num
        self.norm = LayerNorm(embed_dim, bias)
        self.embed_dim = embed_dim
        self.ckpt = ckpt
        
    def forward(
        self, 
        query: torch.Tensor, 
        value: torch.Tensor, 
        point: torch.Tensor, 
        valid: torch.Tensor, 
        shape: List[torch.Tensor]) -> torch.Tensor:
        for block in self.blocks:
            if self.ckpt and self.training:
                query = checkpoint.checkpoint(block, query, value, point, valid, shape, use_reentrant=False)
            else:
                query = block(query, value, point, valid, shape)
        query = rearrange(query, "b v (d q) e -> b v q e d", d=self.depth_num)
        return self.norm(query.mean(-1))
    

class SelfAttn(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        head_num: int,
        dropout: float,
        bias: bool,
        flash_attn: bool,
        group_detr: int = 1) -> None:
        super().__init__()

        assert embed_dim % head_num == 0
        if group_detr < 1:
            raise ValueError("group_detr must be >= 1")
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
        self.head_num = head_num
        self.dropout = dropout
        self.flash_attn = flash_attn
        self.group_detr = group_detr
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (embed_dim)
        grouped = self.training and self.group_detr > 1
        if grouped:
            if T % self.group_detr != 0:
                raise ValueError(f"query count {T} must be divisible by group_detr={self.group_detr}")
            query_per_group = T // self.group_detr
            x = x.reshape(B, self.group_detr, query_per_group, C).reshape(B * self.group_detr, query_per_group, C)
            B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.head_num, C // self.head_num).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.head_num, C // self.head_num).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.head_num, C // self.head_num).transpose(1, 2) # (B, nh, T, hs)

        if self.flash_attn:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        if grouped:
            y = y.reshape(-1, self.group_detr, query_per_group, C).reshape(-1, self.group_detr * query_per_group, C)
        return y



class DefAttn2D(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_num: int,
        level_num: int,
        point_num: int,
        dropout: float,
        bias: bool,
        init_mode: str = "default") -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.head_num = head_num
        self.level_num = level_num
        self.point_num = point_num
        self.dropout = nn.Dropout(dropout)
        self.offset = nn.Linear(embed_dim, head_num * level_num * point_num * 2, bias)
        self.weight = nn.Linear(embed_dim, head_num * level_num * point_num, bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias)
        self.init_mode = str(init_mode).lower()
        if self.init_mode not in {"default", "radial"}:
            raise ValueError(f"Unsupported deformable attention init_mode: {init_mode}")
        if self.init_mode == "radial":
            if self.offset.bias is None or self.weight.bias is None:
                raise ValueError("radial deformable attention initialization requires bias=True")
            self._init_radial_sampling()

    def _init_radial_sampling(self) -> None:
        """Match the structured sampling initialization from Deformable DETR."""
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.weight.weight)
        nn.init.zeros_(self.weight.bias)

        theta = torch.arange(self.head_num, dtype=torch.float32) * (
            2.0 * math.pi / self.head_num
        )
        directions = torch.stack((theta.cos(), theta.sin()), dim=-1)
        directions = directions / directions.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        directions = directions[:, None, None, :].repeat(
            1, self.level_num, self.point_num, 1
        )
        radii = torch.arange(1, self.point_num + 1, dtype=torch.float32)
        directions = directions * radii[None, None, :, None]
        with torch.no_grad():
            self.offset.bias.copy_(directions.reshape(-1))

    def forward(
        self, 
        query, 
        value, 
        point, 
        shape):
        b, query_num, _ = query.shape
        _, value_num, _ = value.shape
        value = value.view(b, value_num, self.head_num, -1)
        offset = self.offset(query).view(b, query_num, self.head_num, self.level_num, self.point_num, 2)
        weight = self.weight(query).view(b, query_num, self.head_num, self.level_num * self.point_num)
        weight = weight.softmax(-1).view(b, query_num, self.head_num, self.level_num, self.point_num)
        location = point[:, :, None, :, None, :] + offset / shape[None, None, None, :, None, :]
        level_start_index = torch.cat([shape.new_zeros((1, )), shape.prod(1).cumsum(0)[:-1]])

        if not value.is_cuda:
            from bev_vae.models.msda.multi_scale_deformable_attn_function import multi_scale_deformable_attn_reference
            # The mathematical CPU reference keeps joint-head tests independent
            # of nvcc; the CUDA dispatch below is unchanged.
            with torch.autocast(device_type=value.device.type, enabled=False):
                output = multi_scale_deformable_attn_reference(
                    value.float(), shape, location.float(), weight.float())
            return self.dropout(self.proj(output.to(query.dtype)))

        if value.dtype == torch.float16:
            msda_func = MultiScaleDeformableAttnFunction_fp16
        elif value.dtype == torch.float32:
            msda_func = MultiScaleDeformableAttnFunction_fp32
        else:
            raise ValueError(f"不支持的数据精度: {value.dtype}，必须为 FP16 或 FP32")
        
        output = msda_func.apply(value, shape, level_start_index, location, weight, min(b, 64))
        return self.dropout(self.proj(output))


class QueryMoEExpert(nn.Module):
    """One detection-query expert with its own BEV sampler and nonlinear update."""

    def __init__(
        self,
        embed_dim: int,
        head_num: int,
        level_num: int,
        point_num: int,
        dropout: float,
        bias: bool,
        hidden_mult: int,
        deformable_init: str = "default",
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_mult) * embed_dim
        self.query_norm = LayerNorm(embed_dim, bias)
        self.cross_attn = DefAttn2D(
            embed_dim,
            head_num,
            level_num,
            point_num,
            dropout,
            bias,
            init_mode=deformable_init,
        )
        self.ffn_norm = LayerNorm(embed_dim, bias)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        point: torch.Tensor,
        shape: torch.Tensor,
    ) -> torch.Tensor:
        expert_state = query + self.cross_attn(
            self.query_norm(query), value, point[..., :2], shape
        )
        expert_state = expert_state + self.ffn(self.ffn_norm(expert_state))
        return expert_state - query


class QueryMoE(nn.Module):
    """Sparse instance/query-level routing over complete decoder expert branches.

    The router observes both the current query content and its refined 3D
    reference point.  Only top-k selected experts execute for each query; each
    expert owns its deformable BEV cross-attention and FFN parameters.
    """

    def __init__(
        self,
        embed_dim: int,
        head_num: int,
        level_num: int,
        point_num: int,
        dropout: float,
        bias: bool,
        num_experts: int,
        top_k: int,
        hidden_mult: int,
        route_noise: float,
        deformable_init: str = "default",
    ) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError("query-level MoE requires at least two experts")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if hidden_mult < 1:
            raise ValueError("hidden_mult must be positive")
        if route_noise < 0:
            raise ValueError("route_noise must be non-negative")

        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.route_noise = float(route_noise)
        self.router_norm = LayerNorm(embed_dim, bias)
        self.reference_proj = nn.Sequential(
            nn.Linear(3, embed_dim, bias=bias),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim, bias=bias),
        )
        self.gate = nn.Linear(embed_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                QueryMoEExpert(
                    embed_dim=embed_dim,
                    head_num=head_num,
                    level_num=level_num,
                    point_num=point_num,
                    dropout=dropout,
                    bias=bias,
                    hidden_mult=hidden_mult,
                    deformable_init=deformable_init,
                )
                for _ in range(self.num_experts)
            ]
        )
        self.last_aux_log = None

    def forward(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        point: torch.Tensor,
        shape: torch.Tensor,
    ) -> torch.Tensor:
        if point.shape[-1] != 3:
            raise ValueError(
                f"query-level MoE router requires xyz reference points, got {point.shape[-1]}D"
            )
        route_input = self.router_norm(query) + self.reference_proj(
            point.squeeze(-2).to(dtype=query.dtype)
        )
        gate_logits = self.gate(route_input)
        if self.training and self.route_noise > 0:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * self.route_noise

        gate_probs = gate_logits.softmax(dim=-1)
        top_logits, top_indices = gate_logits.topk(self.top_k, dim=-1)
        top_weights = top_logits.softmax(dim=-1)
        mixed_delta = torch.zeros_like(query)

        # Dispatch only selected queries.  Per-sample dispatch avoids padding a
        # large number of inactive tokens through the custom MSDA kernel.
        for expert_idx, expert in enumerate(self.experts):
            for batch_idx in range(query.shape[0]):
                assignments = torch.nonzero(
                    top_indices[batch_idx] == expert_idx, as_tuple=False
                )
                if assignments.numel() == 0:
                    continue
                query_indices = assignments[:, 0]
                route_slots = assignments[:, 1]
                expert_query = query[batch_idx : batch_idx + 1, query_indices]
                expert_point = point[batch_idx : batch_idx + 1, query_indices]
                expert_delta = expert(
                    expert_query,
                    value[batch_idx : batch_idx + 1],
                    expert_point,
                    shape,
                ).squeeze(0)
                expert_weights = top_weights[
                    batch_idx, query_indices, route_slots
                ].unsqueeze(-1)
                mixed_delta[batch_idx].index_add_(
                    0, query_indices, expert_delta * expert_weights
                )

        importance = gate_probs.mean(dim=(0, 1))
        load = F.one_hot(top_indices, num_classes=self.num_experts).to(
            gate_probs.dtype
        ).mean(dim=(0, 1, 2))
        balance_loss = self.num_experts * torch.sum(importance * load)
        router_z_loss = torch.logsumexp(gate_logits, dim=-1).square().mean()
        router_entropy = -(
            gate_probs * gate_probs.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        self.last_aux_log = {
            "moe_balance_loss": balance_loss,
            "moe_router_z_loss": router_z_loss,
            "moe_router_entropy": router_entropy,
        }
        for expert_idx in range(self.num_experts):
            self.last_aux_log[f"moe_expert_{expert_idx}_load"] = load[expert_idx]
            self.last_aux_log[f"moe_expert_{expert_idx}_importance"] = importance[
                expert_idx
            ]
        return query + mixed_delta
    

class SDBlock2D(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        head_num: int,
        level_num: int,
        point_num: int,
        dropout: float,
        bias: bool,
        flash_attn: bool,
        group_detr: int = 1,
        moe_config: Any = None,
        deformable_init: str = "default") -> None:
        super().__init__()
        self.ln_0 = LayerNorm(embed_dim, bias)
        self.ln_v = LayerNorm(embed_dim, bias)
        self.sa = SelfAttn(embed_dim, head_num, dropout, bias, flash_attn, group_detr=group_detr)
        moe_enabled = bool(_config_get(moe_config, "enabled", False))
        self.moe_mode = str(_config_get(moe_config, "mode", "ffn")).lower()
        if self.moe_mode not in {"ffn", "query"}:
            raise ValueError(f"Unsupported MoE mode: {self.moe_mode}")
        if moe_enabled and self.moe_mode == "query":
            if group_detr != 1:
                raise ValueError("query-level MoE requires group_detr=1")
            self.query_moe = QueryMoE(
                embed_dim=embed_dim,
                head_num=head_num,
                level_num=level_num,
                point_num=point_num,
                dropout=dropout,
                bias=bias,
                num_experts=int(_config_get(moe_config, "num_experts", 4)),
                top_k=int(_config_get(moe_config, "top_k", 2)),
                hidden_mult=int(_config_get(moe_config, "hidden_mult", 4)),
                route_noise=float(_config_get(moe_config, "route_noise", 0.0)),
                deformable_init=deformable_init,
            )
            self.ln_1 = None
            self.da = None
            self.ln_2 = None
            self.mlp = None
        else:
            self.query_moe = None
            self.ln_1 = LayerNorm(embed_dim, bias)
            self.da = DefAttn2D(
                embed_dim,
                head_num,
                level_num,
                point_num,
                dropout,
                bias,
                init_mode=deformable_init,
            )
            self.ln_2 = LayerNorm(embed_dim, bias)
        if moe_enabled and self.moe_mode == "ffn":
            self.mlp = MoEFFN(
                embed_dim=embed_dim,
                dropout=dropout,
                bias=bias,
                num_experts=int(_config_get(moe_config, "num_experts", 4)),
                top_k=int(_config_get(moe_config, "top_k", 2)),
                hidden_mult=int(_config_get(moe_config, "hidden_mult", 4)),
                route_noise=float(_config_get(moe_config, "route_noise", 0.0)),
            )
        elif self.moe_mode != "query":
            self.mlp = MLP(embed_dim, dropout, bias)

    def forward(
        self, 
        query: torch.Tensor, 
        value: torch.Tensor, 
        point: torch.Tensor, 
        shape: List[torch.Tensor]) -> torch.Tensor:
        query = query + self.sa(self.ln_0(query))
        if self.query_moe is not None:
            query = self.query_moe(query, self.ln_v(value), point, shape)
        else:
            query = query + self.da(
                self.ln_1(query), self.ln_v(value), point[..., :2], shape
            )
            query = query + self.mlp(self.ln_2(query))
        return query

    def get_aux_log(self):
        if self.query_moe is not None:
            return self.query_moe.last_aux_log
        aux_loss = getattr(self.mlp, "last_aux_loss", None)
        if aux_loss is None:
            return None
        return {"moe_balance_loss": aux_loss}


class SDTransformer2D(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        block_num: int,
        head_num: int,
        level_num: int,
        point_num: int,
        cls_num: int,
        reg_num: int,
        dropout: float,
        bias: bool,
        flash_attn: bool,
        ckpt: bool, 
        eps: float = 1e-6,
        vertex_num: int = -1,
        group_detr: int = 1,
        moe_config: Any = None,
        deformable_init: str = "default",
        cls_prior_prob: Optional[float] = None) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            SDBlock2D(
                embed_dim,
                head_num,
                level_num,
                point_num,
                dropout,
                bias,
                flash_attn,
                group_detr=group_detr,
                moe_config=moe_config,
                deformable_init=deformable_init,
            )
            for _ in range(block_num)])
        self.cls_proj = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, embed_dim, bias=bias), nn.GELU(), 
            nn.Linear(embed_dim, embed_dim, bias=bias), nn.GELU(), nn.Linear(embed_dim, cls_num, bias=bias)) for _ in range(block_num)])
        self.reg_proj = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, embed_dim, bias=bias), nn.GELU(), 
            nn.Linear(embed_dim, embed_dim, bias=bias), nn.GELU(), nn.Linear(embed_dim, reg_num, bias=bias)) for _ in range(block_num)])
        if cls_prior_prob is not None:
            cls_prior_prob = float(cls_prior_prob)
            if not 0.0 < cls_prior_prob < 1.0:
                raise ValueError("cls_prior_prob must be in (0, 1)")
            cls_bias = math.log(cls_prior_prob / (1.0 - cls_prior_prob))
            for cls_branch in self.cls_proj:
                if cls_branch[-1].bias is None:
                    raise ValueError(
                        "cls_prior_prob requires bias=True in classification heads"
                    )
                nn.init.constant_(cls_branch[-1].bias, cls_bias)
        self.embed_dim = embed_dim
        self.ckpt = ckpt
        self.eps = eps
        self.vertex_num = vertex_num
        self.moe_enabled = bool(_config_get(moe_config, "enabled", False))

    def forward(
        self, 
        query: torch.Tensor, 
        value: torch.Tensor, 
        point: torch.Tensor, 
        shape: List[torch.Tensor],
        return_aux: bool = False,
        return_motion_features: bool = False):
        cls_preds, reg_preds, aux_logs = [], [], []
        for i, block in enumerate(self.blocks):
            if self.ckpt and self.training:
                query = checkpoint.checkpoint(block, query, value, point, shape, use_reentrant=False)
                reg_pred = checkpoint.checkpoint(self.reg_proj[i], query, use_reentrant=False)
                if self.vertex_num < 0:
                    cls_pred = checkpoint.checkpoint(self.cls_proj[i], query, use_reentrant=False)
                else:
                    cls_pred = checkpoint.checkpoint(
                        self.cls_proj[i], query.view(query.size(0), -1, self.vertex_num, self.embed_dim).mean(-2), use_reentrant=False)
                
            else:
                query = block(query, value, point, shape)
                reg_pred = self.reg_proj[i](query)
                if self.vertex_num < 0:
                    cls_pred = self.cls_proj[i](query)
                else:
                    cls_pred = self.cls_proj[i](query.view(query.size(0), -1, self.vertex_num, self.embed_dim).mean(-2))
                
            # Keep iterative reference refinement out-of-place.  With activation
            # checkpointing, ``point`` is saved for recomputation.  Making it a
            # detached view of ``reg_pred`` allowed later coordinate conversion
            # to mutate the saved storage and fail backward with a version
            # mismatch (most visible with Group DETR's 9900 queries).
            refined_center = (
                reg_pred[..., :3]
                + point.clamp(self.eps, 1 - self.eps).logit().squeeze(-2)
            ).sigmoid()
            reg_pred = torch.cat([refined_center, reg_pred[..., 3:]], dim=-1)
            point = refined_center.detach().clone().unsqueeze(-2)
            cls_preds.append(cls_pred)
            reg_preds.append(reg_pred)
            aux_log = block.get_aux_log()
            if aux_log is not None:
                aux_logs.append(aux_log)

        if return_aux:
            if aux_logs:
                keys = set().union(*(log.keys() for log in aux_logs))
                aux_log = {
                    key: torch.stack([log[key] for log in aux_logs if key in log]).mean()
                    for key in keys
                }
            else:
                aux_log = {"moe_balance_loss": query.new_zeros(())}
            if return_motion_features:
                return cls_preds, reg_preds, aux_log, query
            return cls_preds, reg_preds, aux_log
        if return_motion_features:
            return cls_preds, reg_preds, query
        return cls_preds, reg_preds
