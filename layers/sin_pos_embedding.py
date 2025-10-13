import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 神经网络擅长处理高维数据，但时间步t是一个标量。
# 我们需要将t嵌入到一个高维空间中，才能与神经网络的其他输入一起使用。
# 这里我们使用正弦位置编码（sinusoidal positional encoding），
# 这种编码方式最早在Transformer论文中提出，并被广泛应用于各种任务中。
# 这种编码方式的优点是它能够捕捉不同时间步之间的相对关系，
# 并且具有良好的平滑性和周期性。
class SinPosEmbedding(nn.Module):
    def __init__(self, dim: int):

        super(SinPosEmbedding, self).__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: shape (b,) or (b,1). Values are scalars (time steps / noise level).
        returns (b, dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t * emb  # (b, half_dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb  # (b, dim)


if __name__ == "__main__":
    pe = SinPosEmbedding(128)
    t = torch.tensor([1.0, 2.0, 3.0])
    print(pe(t).shape)  # should be (3, 128)

    print(pe(t)[:3, :3])
