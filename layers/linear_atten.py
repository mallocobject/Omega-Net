import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange


class LinearAtten(nn.Module):
    def __init__(self, in_dims, heads=4, dim_head=32):
        super(LinearAtten, self).__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dims = dim_head * heads
        self.to_qkv = nn.Conv1d(in_dims, hidden_dims * 3, kernel_size=1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dims, in_dims, kernel_size=1), nn.GroupNorm(1, in_dims)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.shape
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) l -> b h c l", h=h), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("b h c j, b h d j -> b h c d", k, v)
        out = torch.einsum("b h c d, b h d i -> b h c i", context, q)
        out = rearrange(out, "b h c l -> b (h c) l")
        out = self.to_out(out)
        return out


if __name__ == "__main__":
    x = torch.randn(2, 64, 128)
    model = LinearAttention(64)
    y = model(x)
    print(y.shape)  # torch.Size([2, 64, 128])
