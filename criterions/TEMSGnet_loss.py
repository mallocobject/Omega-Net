import torch
import torch.nn as nn


class TEMSGnetLoss(nn.MSELoss):
    def __init__(self):
        super(TEMSGnetLoss, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        outputs: torch.Tensor,
        label: torch.Tensor,
    ):
        """
        x: noisy signal
        outputs: noise
        label: clean signal
        """
        x = x.detach()
        outputs = x - outputs
        return super().forward(outputs, label)
