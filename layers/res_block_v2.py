import torch
import torch.nn as nn
import torch.nn.functional as F

from conv import Conv


class ResBlockV2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        block_stride=1,
        use_bn=True,
        stddev=None,
        activation=nn.ReLU(),
    ):
        super(ResBlockV2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_stride = block_stride
        self.use_bn = use_bn
        self.stddev = stddev
        self.activation = activation

        # Shortcut path (identity)
        self.shortcut = nn.Identity()

        # Main path: two convolutions
        self.conv1 = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            use_bn=True,
            kernel_size=(3, 3),
            stride=block_stride,
            padding="SAME",
            activation=F.relu,
            stddev=stddev,
        )
        self.conv2 = Conv(
            in_channels=out_channels,
            out_channels=out_channels,
            use_bn=True,
            kernel_size=(3, 3),
            stride=block_stride,
            padding="SAME",
            activation=None,
            stddev=stddev,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Shortcut path
        shortcut = self.shortcut(x)

        # Main path
        out = self.conv1(x)
        out = self.conv2(out)

        # Residual connection
        out = out + shortcut

        # Final ReLU activation
        out = self.activation(out)

        return out


if __name__ == "__main__":
    # Create sample input tensor (batch=10, channels=128, height=28, width=28)
    x = torch.randn(10, 128, 28, 28)
    module = ResBlockV2(
        in_channels=128,
        out_channels=128,
        block_stride=1,
        use_bn=True,
        stddev=None,
    )
    output = module(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
