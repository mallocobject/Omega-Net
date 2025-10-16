import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm.rich import tqdm  # 导入 tqdm 库
import numpy as np

import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import dataset
from models import TEMDnet, SFSDSA, UNet1D
from utils import plot

NPY_DIR = "dataset/"

BATCH_SIZE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

npy_files = glob.glob(os.path.join(NPY_DIR, "raw_tem_data_batch_*.npy"))
dataset = dataset.TEMDataset(npy_files, split="test")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


def test(model_name="temdnet"):
    if model_name == "temdnet":
        model = TEMDnet(in_channels=1).to(DEVICE)
    elif model_name == "sfsdsa":
        model = SFSDSA(in_features=400).to(DEVICE)
    elif model_name == "unet1d":
        model = UNet1D(in_channels=1, out_channels=1, num_features=32, num_levels=4).to(
            DEVICE
        )
    else:
        raise ValueError("Invalid model name. Choose 'temdnet', 'sfsdsa', or 'unet1d'.")

    # 加载模型权重
    model_path = f"checkpoints/{model_name}_best.pth"
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"Loaded model weights from {model_path}")

    model.eval()

    total_loss = 0  # 用于累计每个 epoch 的损失
    total_batches = 0  # 用于统计每个 epoch 中的 batch 数量
    criterion = nn.MSELoss()

    with torch.no_grad():
        for t, x, label in tqdm(
            dataloader,
            desc=f"[bold cyan]Testing {model_name}",
            colour="magenta",
            unit="batch",
        ):
            x, label = x.to(DEVICE), label.to(DEVICE)
            time_emb = torch.randint(0, 1000, (x.size(0),)).to(DEVICE)  # 随机时间步
            estimate_noise = model(x) if model_name != "unet1d" else model(x, time_emb)
            real_noise = x - label

            loss = criterion(estimate_noise, real_noise)

            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    print(f"Average Loss on Test Set: {avg_loss:.6f}")

    with torch.no_grad():
        # 获取第一批次的第一个样本进行可视化
        t, x, label = next(iter(dataloader))
        x = x[0:1].to(DEVICE)
        label = label[0:1].to(DEVICE)
        time_emb = torch.randint(0, 1000, (x.size(0),)).to(DEVICE)
        estimate_noise = model(x) if model_name != "unet1d" else model(x, time_emb)
        denoised_signal = x - estimate_noise

        # 将张量移回 CPU 并转换为 NumPy 数组
        t = t[0].cpu().numpy()
        noisy_signal = x[0].cpu().numpy()
        clean_signal = label[0].cpu().numpy()
        denoised_signal = denoised_signal[0].cpu().numpy()

        # 使用 plot 函数进行可视化
        plot(
            t,
            noisy_signal,
            clean_signal,
            denoised_signal,
            x_axis="time (ms)",
            y_axis="B (nT)",
            title=f"{model_name} Denoising Result",
        )

    return avg_loss


if __name__ == "__main__":
    test("temdnet")
    # test("sfsdsa")
