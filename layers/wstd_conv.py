import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import reduce
from functools import partial


# 对卷积核进行权重标准化
# Reference: https://arxiv.org/abs/1903.10520
class WeightStandardizedConv1d(nn.Conv1d):
    def forward(self, x: torch.Tensor):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1", partial(torch.var, unbiased=False))
        normalized = (weight - mean) * (var + eps).rsqrt()
        return F.conv1d(
            x,
            normalized,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


if __name__ == "__main__":
    conv = WeightStandardizedConv1d(32, 64, 3, padding=1)
    x = torch.randn(8, 32, 128)
    y = conv(x)
    print(y.shape)  # should be (8, 64, 128)
