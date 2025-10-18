import torch
import torch.nn as nn


class MSECriterion(nn.MSELoss):
    def __init__(self):
        super(MSECriterion, self).__init__()

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


class MSECriterionWithNoise(nn.MSELoss):
    def __init__(self):
        super(MSECriterionWithNoise, self).__init__()

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
