import torch
import torch.nn as nn
import torch.nn.functional as F


import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers import DilatedConv, ResBlockV1, ResBlockV2
from utils import seq2img, img2seq


class TEMDnet(nn.Module):
    def __init__(self, in_channels=1, stddev=None):
        super(TEMDnet, self).__init__()
        self.in_channels = in_channels
        self.stddev = stddev

        # Dilated convolutions
        self.dilated_conv1 = DilatedConv(
            in_channels=in_channels,
            out_channels=32,
            use_bn=False,
            kernel_size=(3, 3),
            dilation=1,
            padding="SAME",
            activation=nn.ReLU(),
            stddev=stddev,
        )
        self.dilated_conv2 = DilatedConv(
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
        self.res_block_1 = ResBlockV1(
            in_channels=64,
            out_channels_list=[64, 128],
            change_dimension=True,
            block_stride=1,
            use_bn=True,
            stddev=stddev,
            activate=True,
        )
        self.res_block_2 = ResBlockV2(
            in_channels=128,
            out_channels=128,
            block_stride=1,
            use_bn=True,
            stddev=stddev,
        )
        self.res_block_3 = ResBlockV2(
            in_channels=128,
            out_channels=128,
            block_stride=1,
            use_bn=True,
            stddev=stddev,
        )
        self.res_block_4 = ResBlockV2(
            in_channels=128,
            out_channels=128,
            block_stride=1,
            use_bn=True,
            stddev=stddev,
        )
        self.res_block_5 = ResBlockV1(
            in_channels=128,
            out_channels_list=[128, 64],
            change_dimension=True,
            block_stride=1,
            use_bn=True,
            stddev=stddev,
            activate=True,
        )

        # Final dilated convolutions
        self.dilated_conv3 = DilatedConv(
            in_channels=64,
            out_channels=32,
            use_bn=False,
            kernel_size=(3, 3),
            dilation=2,
            padding="SAME",
            activation=nn.ReLU(),
            stddev=stddev,
        )
        self.dilated_conv4 = DilatedConv(
            in_channels=32,
            out_channels=1,
            use_bn=False,
            kernel_size=(3, 3),
            dilation=1,
            padding="SAME",
            activation=None,
            stddev=stddev,
        )

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:

        x_length = x.shape[-1]
        img = seq2img(x)  # (B, L) -> (B, H, W), H*W=L
        img = img.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

        out = self.dilated_conv1(img, training=training)
        out = self.dilated_conv2(out, training=training)
        out = self.res_block_1(out, training=training)
        out = self.res_block_2(out, training=training)
        out = self.res_block_3(out, training=training)
        out = self.res_block_4(out, training=training)
        out = self.res_block_5(out, training=training)
        out = self.dilated_conv3(out, training=training)
        out = self.dilated_conv4(out, training=training)

        out = out.squeeze(1)  # (B, 1, H, W) -> (B, H, W)

        out = img2seq(out, x_length)  # (B, H, W) -> (B, 1, L), L=H*W

        return out


if __name__ == "__main__":
    x = torch.randn(100, 400)  # (B, L)
    model = TEMDnet(in_channels=1)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
