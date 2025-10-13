import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops.layers.torch import Rearrange

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import default


class DownSample(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(DownSample, self).__init__()
        self.layers = nn.Sequential(
            Rearrange("b c (h p1) -> b (c p1) h", p1=2),
            nn.Conv1d(in_dims * 2, default(out_dims, in_dims * 2), kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        out = self.layers(x)
        return out


if __name__ == "__main__":
    x = torch.randn(100, 2, 400)
    downsample = DownSample(2, 4)
    y = downsample(x)
    print(y.shape)
