import torch
import torch.nn as nn
import torch.nn.functional as F


class TEMSGnetLoss(nn.Module):
    """
    TEMSGnet 扩散损失函数:
    计算模型预测噪声与真实噪声之间的误差
    """

    def __init__(self, loss_type="huber"):
        super(TEMSGnetLoss, self).__init__()
        if loss_type == "l2":
            self.criterion = F.mse_loss
        elif loss_type == "huber":
            self.criterion = F.smooth_l1_loss
        else:
            raise NotImplementedError(f"Unknown loss type: {loss_type}")

    def forward(self, model, x_0, condition, t):
        """
        model: TEMSGnet 模型
        x_0: 干净信号
        condition: 条件（如含噪信号）
        t: 时间步 tensor (batch,)
        """
        betas = model.betas
        alphas = model.alphas
        alpha_hats = model.alpha_hats

        # 当前时间步对应的系数
        sqrt_alpha_hat = torch.sqrt(alpha_hats[t]).to(x_0.device)
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - alpha_hats[t]).to(x_0.device)

        # 添加噪声
        noise = torch.randn_like(x_0)
        x_t = sqrt_alpha_hat[:, None] * x_0 + sqrt_one_minus_alpha_hat[:, None] * noise

        # 模型预测噪声
        output = model.model(x_t, t, x_self_cond=condition)

        # 损失函数
        loss = self.criterion(output, noise)
        return loss
