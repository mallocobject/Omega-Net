import torch
import torch.nn as nn
import torch.nn.functional as F


import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import layers
from utils import seq2img, img2seq


class TEMDnet(nn.Module):
    def __init__(self, in_channels=1, stddev=None):
        super(TEMDnet, self).__init__()
        self.in_channels = in_channels
        self.stddev = stddev

        # Dilated convolutions
        dilated_conv1 = layers.DilatedConv(
            in_channels=in_channels,
            out_channels=32,
            use_bn=False,
            kernel_size=(3, 3),
            dilation=1,
            padding="SAME",
            activation=nn.ReLU(),
            stddev=stddev,
        )
        dilated_conv2 = layers.DilatedConv(
            in_channels=32,
            out_channels=64,
            use_bn=False,
            kernel_size=(3, 3),
            dilation=2,
            padding="SAME",
            activation=nn.ReLU(),
            stddev=stddev,
        )

        # Residual blocks
        res_block_1 = layers.ResBlockV1(
            in_channels=64,
            out_channels_list=[64, 128],
            change_dimension=True,
            block_stride=1,
            use_bn=True,
            stddev=stddev,
            activate=True,
        )
        res_block_2 = layers.ResBlockV2(
            in_channels=128,
            out_channels=128,
            block_stride=1,
            use_bn=True,
            stddev=stddev,
        )
        res_block_3 = layers.ResBlockV2(
            in_channels=128,
            out_channels=128,
            block_stride=1,
            use_bn=True,
            stddev=stddev,
        )
        res_block_4 = layers.ResBlockV2(
            in_channels=128,
            out_channels=128,
            block_stride=1,
            use_bn=True,
            stddev=stddev,
        )
        res_block_5 = layers.ResBlockV1(
            in_channels=128,
            out_channels_list=[128, 64],
            change_dimension=True,
            block_stride=1,
            use_bn=True,
            stddev=stddev,
            activate=True,
        )

        # Final dilated convolutions
        dilated_conv3 = layers.DilatedConv(
            in_channels=64,
            out_channels=32,
            use_bn=False,
            kernel_size=(3, 3),
            dilation=2,
            padding="SAME",
            activation=nn.ReLU(),
            stddev=stddev,
        )
        dilated_conv4 = layers.DilatedConv(
            in_channels=32,
            out_channels=1,
            use_bn=False,
            kernel_size=(3, 3),
            dilation=1,
            padding="SAME",
            activation=None,
            stddev=stddev,
        )

        self.network = nn.Sequential(
            dilated_conv1,
            dilated_conv2,
            res_block_1,
            res_block_2,
            res_block_3,
            res_block_4,
            res_block_5,
            dilated_conv3,
            dilated_conv4,
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor = None) -> torch.Tensor:

        x_length = x.shape[-1]
        img = seq2img(x)  # (B, L) -> (B, H, W), H*W=L
        img = img.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

        out = self.network(img)  # (B, 1, H, W)

        out = out.squeeze(1)  # (B, 1, H, W) -> (B, H, W)

        out = img2seq(out, x_length)  # (B, H, W) -> (B, L), L=H*W

        return out


if __name__ == "__main__":
    x = torch.randn(100, 400)  # (B, L)
    model = TEMDnet(in_channels=1)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
