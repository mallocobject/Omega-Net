import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedFC(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_features=[],
        use_bn=False,
        activate=True,
        activation=nn.SELU(),
    ):
        super(StackedFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.use_bn = use_bn
        self.activation = activation

        layers = []
        all_features = [in_features] + hidden_features + [out_features]
        for i in range(len(all_features) - 1):
            layers.append(nn.Linear(all_features[i], all_features[i + 1]))
            if i < len(all_features) - 2:  # No BN after the last layer
                if use_bn:
                    layers.append(nn.BatchNorm1d(all_features[i + 1]))
            if activate:
                layers.append(activation)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
