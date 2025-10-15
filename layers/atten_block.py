import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange


class AttenBlock(nn.Module):
    def __init__(self, in_dims, heads=4, dim_head=32):
        super(AttenBlock, self).__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dims = dim_head * heads
        self.to_qkv = nn.Conv1d(in_dims, hidden_dims * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv1d(hidden_dims, in_dims, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.shape
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) l -> b h c l", h=h), qkv)
        q = q * self.scale
        attn = torch.einsum("b h c i, b h c j -> b h i j", q, k)
        attn = attn - attn.amax(dim=-1, keepdim=True).detach()
        attn = attn.softmax(dim=-1)
        out = torch.einsum("b h i j, b h c j -> b h c i", attn, v)
        out = rearrange(out, "b h c l -> b (h c) l")
        out = self.to_out(out)
        return out


if __name__ == "__main__":
    block = AttentionBlock(32, heads=4, dim_head=8)
    x = torch.randn(8, 32, 128)
    y = block(x)
    print(y.shape)  # should be (8, 32, 128)
