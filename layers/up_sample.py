import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import default


class UpSample(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(UpSample, self).__init__()
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(in_dims, default(out_dims, in_dims), kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor):
        out = self.layers(x)
        return out


if __name__ == "__main__":
    x = torch.randn(100, 2, 400)
    upsample = UpSample(2, 1)
    y = upsample(x)
    print(y.shape)
