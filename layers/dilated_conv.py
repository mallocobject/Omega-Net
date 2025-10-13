import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DilatedConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_bn=False,
        kernel_size=(3, 3),
        dilation=1,
        padding="SAME",
        activation=nn.ReLU(),
        stddev=None,
    ):
        super(DilatedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.dilation = dilation
        self.padding = padding.upper()
        self.activation = activation
        self.stddev = (
            stddev
            if stddev is not None
            else math.sqrt(
                2.0 / (self.kernel_size[0] * self.kernel_size[1] * out_channels)
            )
        )  # He initialization

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=0 if self.padding == "VALID" else self._compute_same_padding(),
            bias=True,
        )

        nn.init.trunc_normal_(self.conv.weight, mean=0.0, std=self.stddev)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)

        # Batch normalization
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None

    def _compute_same_padding(self):
        kh, kw = self.kernel_size
        dilation = self.dilation
        # Effective kernel size with dilation
        effective_kh = kh + (kh - 1) * (dilation - 1)
        effective_kw = kw + (kw - 1) * (dilation - 1)
        # Padding to ensure output size matches input size
        pad_h = (effective_kh - 1) // 2
        pad_w = (effective_kw - 1) // 2
        return (pad_h, pad_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.bn:
            out = self.bn(out)
        if self.activation:
            out = self.activation(out)

        return out


if __name__ == "__main__":
    x = torch.randn(10, 1, 28, 28)
    module = DilatedConv(1, 32, False)
    output = module(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
