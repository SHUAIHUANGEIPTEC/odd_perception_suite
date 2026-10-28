from typing import List

import torch.nn as nn
from bev_vae.models.rope import RoFormerPE2D


class Identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    def __init__(
        self, 
        in_ch: int,
        out_ch: int,
        dropout: float,
        bias: bool) -> None:
        super().__init__()
        self.c_fc = nn.Linear(in_ch, out_ch, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(out_ch, out_ch, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

class Connector(nn.Module):
    def __init__(
        self, 
        state_size: List[int], 
        patch_size: int,
        in_ch: int,
        out_ch: int,
        block_num: int,
        head_num: int,
        dropout: float,
        bias: bool,
        flash_attn: bool,
        ckpt: bool) -> None:
        super().__init__()
        self.grid_size = (state_size[0]//patch_size, state_size[1]//patch_size)
        self.proj = nn.Linear(in_ch, out_ch, bias=bias)
        self.transformer = RoFormerPE2D(self.grid_size, out_ch, block_num, head_num, dropout, bias, flash_attn, ckpt)

    def forward(self, x):
        x = self.proj(x) # B H W E
        x = self.transformer(x)
        return x
