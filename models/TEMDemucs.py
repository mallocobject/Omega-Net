from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from functools import partial

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bid=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            num_layers=layers,
            batch_first=False,
            bidirectional=bid,
        )
        self.linear = None
        if bid:
            self.linear = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        x, hidden = self.lstm(x, hidden)
        if self.linear is not None:
            x = self.linear(x)
        return x, hidden


class TEMDemucs(nn.Module):
    def __init__(
        self,
        chin=1,
        chout=1,
        hidden=32,
        depth=4,
        kernel_size=3,
        stride=2,
        padding=1,
        causal=False,
        growth=2,
        max_hidden=1000,
        glu=True,
        stddev=None,
    ):
        super().__init__()

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.causal = causal
        self.growth = growth
        self.max_hidden = max_hidden
        self.stddev = stddev

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        activation = nn.GLU(dim=1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        for idx in range(depth):
            encode = [
                nn.Conv1d(chin, hidden, kernel_size, stride, padding),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1),
                activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = [
                nn.Conv1d(hidden, hidden * ch_scale, 1),
                activation,
                nn.ConvTranspose1d(
                    hidden,
                    chout,
                    kernel_size,
                    stride,
                    padding,
                    output_padding=stride - 1,
                ),
            ]
            if idx > 0:
                decode.append(nn.ReLU())
            self.decoder.append(nn.Sequential(*decode))
            chin = hidden
            chout = hidden
            hidden = min(int(hidden * growth), max_hidden)

        self.lstm = BLSTM(chout, bid=not causal)

    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, T)

        length = x.shape[-1]
        skips = []
        for encoder in self.encoder:
            x = encoder(x)
            skips.append(x)
        x = x.permute(2, 0, 1)  # (T, B, C)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)  # (B, C, T)
        for decoder in reversed(self.decoder):
            skip = skips.pop()
            x = x + skip[..., : x.shape[-1]]  # 对齐长度
            x = decoder(x)
        x = x[..., :length]  # 裁剪到原始长度
        return x.squeeze(1)  # (B, T)


if __name__ == "__main__":
    model = TEMDemucs()
    print(model)
    x = torch.randn(100, 400)
    y = model(x)
    print(y.shape)  # 应该输出 (100, 400)
