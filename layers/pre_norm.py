import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 在自注意力机制或其他层之前进行归一化
class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super(PreNorm, self).__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.norm(x)
        return self.fn(x, **kwargs)


if __name__ == "__main__":
    from atten_block import AttenBlock

    x = torch.randn(2, 32, 128)
    attn = AttenBlock(32)
    prenorm_attn = PreNorm(32, attn)
    y = prenorm_attn(x)
    print(y.shape)  # should be (2, 32, 128)
