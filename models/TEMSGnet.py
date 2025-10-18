from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from functools import partial

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import layers
from utils import default


# duffusion model with 1D U-Net architecture
class UNet1D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        num_features=32,
        num_levels=4,
        is_cond: bool = False,
        res_block_groups=4,
        stddev=None,
    ):
        super(UNet1D, self).__init__()
        if is_cond:
            in_channels = in_channels * 2  # if conditional, attach condition to input

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.num_levels = num_levels
        self.is_cond = is_cond
        self.stddev = stddev

        # Initial convolution
        self.init_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_features,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        block_fn = partial(layers.ResBlock, groups=res_block_groups)
        time_dim = num_features * 4

        self.time_mlp = nn.Sequential(
            layers.SinusoidalPosEmb(num_features),
            nn.Linear(num_features, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder path
        self.encoders = nn.ModuleList()
        for i in range(num_levels - 1):
            input_channels = num_features * (2**i)
            output_channels = num_features * (2 ** (i + 1))
            encoder = nn.ModuleList(
                [
                    block_fn(input_channels, input_channels, time_emb_dim=time_dim),
                    block_fn(input_channels, input_channels, time_emb_dim=time_dim),
                    layers.ResWrapper(
                        layers.PreNorm(
                            input_channels, layers.AttenBlock(input_channels)
                        )
                    ),
                    layers.DownSample(input_channels, output_channels),
                ]
            )

            self.encoders.append(encoder)

        # Bottleneck
        # bottleneck 的通道数应该是经过 (num_levels-1) 次下采样后的通道数
        bottleneck_channels = num_features * (2 ** (num_levels - 1))
        # 用 ModuleList 而不是 Sequential，因为每个 block 需要 time embedding 参数
        self.bottleneck = nn.ModuleList(
            [
                block_fn(
                    bottleneck_channels, bottleneck_channels, time_emb_dim=time_dim
                ),
                block_fn(
                    bottleneck_channels, bottleneck_channels, time_emb_dim=time_dim
                ),
                layers.ResWrapper(
                    layers.PreNorm(
                        bottleneck_channels, layers.AttenBlock(bottleneck_channels)
                    )
                ),
                layers.DownSample(bottleneck_channels, bottleneck_channels * 2),
                block_fn(
                    bottleneck_channels * 2,
                    bottleneck_channels * 2,
                    time_emb_dim=time_dim,
                ),
                block_fn(
                    bottleneck_channels * 2,
                    bottleneck_channels * 2,
                    time_emb_dim=time_dim,
                ),
                layers.ResWrapper(
                    layers.PreNorm(
                        bottleneck_channels * 2,
                        layers.AttenBlock(bottleneck_channels * 2),
                    )
                ),
                layers.UpSample(bottleneck_channels * 2, bottleneck_channels),
            ]
        )

        # Decoder path
        self.decoders = nn.ModuleList()
        for i in range(num_levels - 2, -1, -1):  # 这个倒序写法6
            input_channels = num_features * (2 ** (i + 1))
            output_channels = num_features * (2**i)
            decoder = nn.ModuleList(
                [  # skip connection, so input channels is doubled
                    block_fn(input_channels * 2, input_channels, time_emb_dim=time_dim),
                    block_fn(input_channels, input_channels, time_emb_dim=time_dim),
                    layers.ResWrapper(
                        layers.PreNorm(
                            input_channels, layers.AttenBlock(input_channels)
                        )
                    ),
                    layers.UpSample(input_channels, output_channels),
                ]
            )
            self.decoders.append(decoder)

        # Final convolution
        # self.final_res_block = block_fn(num_features * 2, num_features)

        self.final_conv = nn.Conv1d(
            in_channels=num_features,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        x_self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, L) -> (B, C, L)

        if self.is_cond:
            y = default(x_self_cond, lambda: torch.zeros_like(x))

        if y.dim() == 2:
            y = y.unsqueeze(1)  # (B, L) -> (B, C, L)

        if self.stddev is not None:
            noise = torch.randn_like(x) * self.stddev
            x = x + noise

        if self.is_cond:
            # split input into two parts along channel dimension
            x = torch.cat([x, y], dim=1)

        # time embedding
        t = self.time_mlp(time)

        # initial conv
        x = self.init_conv(x)

        # encoder path
        skip_connections = []
        for block1, block2, attn, downsample in self.encoders:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = downsample(x)
            skip_connections.append(x)

        # bottleneck
        for block in self.bottleneck:
            if isinstance(block, layers.ResBlock):
                x = block(x, t)
            else:
                x = block(x)

        # decoder path
        skip_connections = skip_connections[::-1]  # reverse to start from last encoder
        for i, (block1, block2, attn, upsample) in enumerate(self.decoders):
            skip_connection = skip_connections[i]
            x = torch.cat((skip_connection, x), dim=1)  # concatenate along channel dim
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        # final conv
        # x = self.final_res_block(x)
        x = self.final_conv(x)

        x = x.squeeze(1) if x.shape[1] == 1 else x  # (B, 1, L) -> (B, L)
        return x


def extract(a, t, x_shape):
    """
    Extract values from a 1-D tensor `a` for a batch of indices `t`,
    and reshape to `x_shape`.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t).float()
    return out.view(batch_size, *((1,) * (len(x_shape) - 1)))


class TEMSGnet(nn.Module):
    def __init__(
        self,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        stddev=None,
    ):
        super(TEMSGnet, self).__init__()
        self.model = UNet1D(
            in_channels=1,
            out_channels=1,
            num_features=32,
            num_levels=4,
            is_cond=True,
            res_block_groups=4,
        )
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_hats = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_hats", alpha_hats)
        self.register_buffer("sqrt_recip_alphas_hat", torch.sqrt(1.0 / alpha_hats))
        self.register_buffer("sqrt_one_minus_alpha_hats", torch.sqrt(1.0 - alpha_hats))

    @torch.no_grad()
    def p_denoise_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform one denoising step.
        x_t: Noisy input at time step t
        t: Time step tensor
        x_self_cond: Optional self-conditioning input
        """
        beta_t = extract(self.betas, t, x_t.shape)
        sqrt_recip_alpha_hat_t = extract(self.sqrt_recip_alphas_hat, t, x_t.shape)
        sqrt_one_minus_alpha_hat_t = extract(
            self.sqrt_one_minus_alpha_hats, t, x_t.shape
        )

        # Predict the noise
        model_output = self.model(x_t, t, x_self_cond)

        # Compute the denoised output
        x_0_pred = sqrt_recip_alpha_hat_t * (
            x_t - beta_t / sqrt_one_minus_alpha_hat_t * model_output
        )

        # 不加噪声
        return x_0_pred

    @torch.no_grad()
    def denoise_from_noisy(self, x_noisy, condition=None, start_t=None, steps=None):
        """
        从含噪信号开始去噪
        - x_noisy: 含噪输入 (B, L)
        - condition: 条件输入 (B, L)
        - start_t: 起始时间步（可根据噪声程度设定）
        - steps: 限制反扩散步数（默认全程）
        """
        self.model.eval()
        B = x_noisy.shape[0]
        if start_t is None:
            start_t = self.timesteps - 1
        if steps is None:
            steps = start_t + 1

        # 反向扩散过程
        x_t = x_noisy
        for t in reversed(range(start_t, steps)):
            t_batch = torch.full((B,), t, dtype=torch.long, device=x_t.device)
            x_t = self.p_denoise_step(x_t, t_batch, x_self_cond=condition)
        return x_t  # 返回去噪结果(esimate signal)

    def forward(
        self,
        x_self_cond: torch.Tensor,
        x: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        betas = self.betas
        alphas = self.alphas
        alpha_hats = self.alpha_hats

        # 当前时间步对应的系数
        sqrt_alpha_hat = torch.sqrt(alpha_hats[time])
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - alpha_hats[time])

        # 添加噪声
        noise = torch.randn_like(x)
        x_t = sqrt_alpha_hat[:, None] * x + sqrt_one_minus_alpha_hat[:, None] * noise

        # 模型预测噪声
        out = self.model(x_t, time, x_self_cond=x_self_cond)

        return out


if __name__ == "__main__":
    model = UNet1D(
        in_channels=1,
        out_channels=1,
        num_features=32,
        num_levels=4,
        is_cond=True,
        res_block_groups=4,
        stddev=None,
    )
    x = torch.randn(100, 400)
    t = torch.randint(0, 1000, (100,))
    print(t.shape)
    y = model(x, t)
    print(y.shape)  # should be (100, 400)
