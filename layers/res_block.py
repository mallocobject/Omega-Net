import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional
from einops import rearrange

from .basic_block import BasicBlock


class ResBlock(nn.Module):
    def __init__(self, in_dims, out_dims, time_emb_dim=None, groups=8):
        super(ResBlock, self).__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_dims * 2))
            if time_emb_dim is not None
            else None
        )
        self.block1 = BasicBlock(in_dims, out_dims, groups=groups)
        self.block2 = BasicBlock(out_dims, out_dims, groups=groups)
        self.res_conv = (
            nn.Conv1d(in_dims, out_dims, kernel_size=1)
            if in_dims != out_dims
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1")
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


if __name__ == "__main__":

    block = ResBlock(32, 64, time_emb_dim=128)
    x = torch.randn(8, 32, 128)
    t = torch.randn(8, 128)
    y = block(x, t)
    print(y.shape)  # should be (8, 64, 128)
    block = ResBlock(32, 32)
    y = block(x)
    print(y.shape)  # should be (8, 32, 128)
