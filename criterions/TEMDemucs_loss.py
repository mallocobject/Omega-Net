import torch
import torch.nn as nn
import torch.nn.functional as F


class TEMDemucsLoss(nn.MSELoss):
    def __init__(self):
        super(TEMDemucsLoss, self).__init__()

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
        mse_loss = super().forward(outputs, label)
        diff_out = outputs[:, 1:] - outputs[:, :-1]
        Lmono_out = F.relu(-diff_out).mean()

        total_loss = mse_loss + Lmono_out
        return total_loss
