import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from basic_block import BasicBlock


class ResnetBlock(nn.Module):
    def __init__(self, in_dims, out_dims, time_emb_dim=None, groups=8):
        super(ResnetBlock, self).__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_dims * 2))
            if time_emb_dim is not None
            else None
        )
        self.conv1 = BasicBlock(in_dims, out_dims, groups=groups)
        self.conv2 = BasicBlock(out_dims, out_dims, groups=groups)
        self.res_conv = (
            nn.Conv1d(in_dims, out_dims, kernel_size=1)
            if in_dims != out_dims
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)

        if self.residual_conv is not None:
            residual = self.residual_conv(residual)

        x += residual
        x = self.act2(x)

        return x
