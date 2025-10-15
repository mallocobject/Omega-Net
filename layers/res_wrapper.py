import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResWrapper:
    def __init__(self, fn: nn.Module):
        self.fn = fn

    def __call__(self, x: torch.Tensor, *arg, **kargs) -> torch.Tensor:
        return x + self.fn(x, *arg, **kargs)


if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
    res_conv = ResWrapper(conv)
    y = res_conv(x)
    print(y.shape)  # torch.Size([2, 3, 32, 32])
