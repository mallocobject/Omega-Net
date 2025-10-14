import torch
import torch.nn as nn
import torch.nn.functional as F

# import os
# import sys

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wstd_conv import WeightStandardizedConv1d
from typing import Optional, Tuple


class BasicBlock(nn.Module):
    def __init__(self, in_dims, out_dims, groups=8):
        super(BasicBlock, self).__init__()
        self.proj = WeightStandardizedConv1d(
            in_dims, out_dims, kernel_size=3, padding=1
        )
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=out_dims)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        scale_shift: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        out = self.act(x)
        return out


if __name__ == "__main__":
    block = BasicBlock(32, 64)
    x = torch.randn(8, 32, 128)
    y = block(x)
    print(y.shape)  # should be (8, 64, 128)
    scale = torch.randn(8, 64, 1)
    shift = torch.randn(8, 64, 1)
    y = block(x, (scale, shift))
    print(y.shape)  # should be (8, 64, 128)
