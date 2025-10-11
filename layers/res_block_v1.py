import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class ResBlockV1(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels_list: list,
        change_dimension=False,
        block_stride=1,
        use_bn=True,
        stddev=None,
        activate=True,
    ):
        super(ResBlockV1, self).__init__()
        self.in_channels = in_channels
        self.out_channels_list = out_channels_list
        self.change_dimension = change_dimension
        self.block_stride = block_stride
        self.use_bn = use_bn
        self.stddev = stddev
        self.activate = activate

        if len(out_channels_list) != 2:
            raise ValueError(
                "n_out_list must contain exactly two values [n_out_1, n_out_2]"
            )

        # Shortcut path
        if change_dimension:
            self.shortcut = Conv(
                in_channels=in_channels,
                out_channels=out_channels_list[1],
                use_bn=True,
                kernel_size=(1, 1),
                stride=block_stride,
                padding="SAME",
                activation=None,
                stddev=stddev,
            )
        else:
            self.shortcut = nn.Identity()

        # Main path: three convolutions
        self.conv1 = Conv(
            in_channels=in_channels,
            out_channels=out_channels_list[0],
            use_bn=True,
            kernel_size=(1, 1),
            stride=block_stride,
            padding="SAME",
            activation=F.relu,
            stddev=stddev,
        )
        self.conv2 = Conv(
            in_channels=out_channels_list[0],
            out_channels=out_channels_list[0],
            use_bn=True,
            kernel_size=(3, 3),
            stride=1,
            padding="SAME",
            activation=F.relu,
            stddev=stddev,
        )
        self.conv3 = Conv(
            in_channels=out_channels_list[0],
            out_channels=out_channels_list[1],
            use_bn=True,
            kernel_size=(1, 1),
            stride=1,
            padding="SAME",
            activation=None,
            stddev=stddev,
        )

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:

        # Shortcut path
        shortcut = self.shortcut(x, training=training)

        # Main path
        out = self.conv1(x, training=training)
        out = self.conv2(out, training=training)
        out = self.conv3(out, training=training)

        # Residual connection
        out = out + shortcut

        # Final activation
        if self.activate:
            out = F.relu(out)

        return out


if __name__ == "__main__":
    # Create sample input tensor (batch=10, channels=64, height=28, width=28)
    x = torch.randn(10, 64, 28, 28)
    module = ResBlockV1(
        in_channels=64,
        out_channels_list=[64, 128],
        change_dimension=True,
        block_stride=1,
        use_bn=True,
        stddev=None,
        activate=True,
    )
    output = module(x, training=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
