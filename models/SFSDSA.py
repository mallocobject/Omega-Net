import torch
import torch.nn as nn
import torch.nn.functional as F


import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers import StackedFC


# Autoencoder with Stacked Fully Connected layers for Speech Denoising and Source Separation
class SFSDSA(nn.Module):
    def __init__(
        self,
        in_features=400,
        hidden_features=[256, 128, 64],
        use_bn=False,
        stddev=None,
    ):
        super(SFSDSA, self).__init__()
        self.metric = nn.MSELoss()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.use_bn = use_bn
        self.stddev = stddev

        # encoder
        self.encoder = StackedFC(
            layers=[in_features] + hidden_features,
            use_bn=use_bn,
            activate=True,
        )

        # decoder
        self.decoder = StackedFC(
            layers=hidden_features[::-1] + [in_features],
            use_bn=use_bn,
            activate=False,
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor = None) -> torch.Tensor:

        # x: (B, L)
        x = (
            x + torch.randn_like(x) * self.stddev
            if self.stddev
            else torch.zeros_like(x)
        )
        encoded = self.encoder(x)  # (B, out_features)
        out = self.decoder(encoded)  # (B, in_features)

        return out

    def criterion(self, x: torch.Tensor, outputs: torch.Tensor, label: torch.Tensor):
        """
        x: noisy signal
        outputs: estimate signal
        label: clean signal
        """
        return self.metric(outputs, label)


if __name__ == "__main__":
    x = torch.randn(100, 400)  # (B, L)
    model = SFSDSA(in_features=400)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
