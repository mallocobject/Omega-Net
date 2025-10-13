import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResWrapper(nn.Module):
    def __init__(self, fn):
        super(ResWrapper, self).__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *arg, **kargs) -> torch.Tensor:
        return x + self.fn(x, *arg, **kargs)


if __name__ == "__main__":
    x = torch.randn(2, 3, 4, 4)
    conv = nn.Conv2d(3, 3, 3, padding=1)
    res_block = ResWrapper(conv)
    y = res_block(x)
    print(y.shape)
