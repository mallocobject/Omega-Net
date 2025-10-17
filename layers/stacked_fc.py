import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedFC(nn.Module):
    def __init__(
        self,
        layers=[],
        use_bn=False,
        activate=True,
        activation=nn.ReLU(),
    ):
        super(StackedFC, self).__init__()
        self.layers = layers
        self.use_bn = use_bn
        self.activation = activation

        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            if use_bn:
                net.append(nn.BatchNorm1d(layers[i + 1]))
            if activate and i < len(layers) - 1:
                net.append(activation)

        self.network = nn.Sequential(*net)

    def forward(self, x):
        return self.network(x)
