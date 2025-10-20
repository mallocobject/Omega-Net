import torch
import torch.nn as nn
import torch.nn.functional as F


# 多层全连接堆叠模块
class StackedFC(nn.Module):
    def __init__(
        self,
        layers=[],
        use_bn=False,
        activate=True,
        activation=nn.ReLU(),
    ):
        super(StackedFC, self).__init__()

        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            if use_bn:
                net.append(nn.BatchNorm1d(layers[i + 1]))
            # 最后一层通常不加激活
            if activate and i < len(layers) - 2:
                net.append(activation)

        self.network = nn.Sequential(*net)

    def forward(self, x):
        return self.network(x)
