import torch
import torch.nn as nn


class SFSDSALoss(nn.MSELoss):
    def __init__(self):
        super(SFSDSALoss, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        outputs: torch.Tensor,
        label: torch.Tensor,
    ):
        """
        x: noisy signal
        outputs: estimate signal
        label: clean signal
        """
        return super().forward(outputs, label)
