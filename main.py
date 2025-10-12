import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import snr, mse
import data.dataset

from models import TEMDnet, SFSDSA

npy_dir = "data/raw_data/"
batch_size = 100
epochs = 200

# 获取目录下所有的 .npy 文件
npy_files = glob.glob(os.path.join(npy_dir, "raw_tem_data_batch_*.npy"))

dataset = data.dataset.TEMDataset(npy_files)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(model_name="temdnet"):
    if model_name == "temdnet":
        model = TEMDnet(in_channels=1)
    elif model_name == "sfsdsa":
        model = SFSDSA(in_features=400)
    else:
        raise ValueError("Invalid model name. Choose 'temdnet' or 'sfsdsa'.")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()

    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")

        total_loss = 0  # 用于累计每个 epoch 的损失
        total_batches = 0  # 用于统计每个 epoch 中的 batch 数量

        for t, x, label in tqdm(
            dataloader, desc=f"Training Epoch {epoch+1}", unit="batch"
        ):
            estimate_noise = model(x)
            real_noise = x - label

            loss = criterion(estimate_noise, real_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累积损失
            total_loss += loss.item()
            total_batches += 1

        # 计算每个 epoch 的平均损失
        avg_loss = total_loss / total_batches
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), f"checkpoints/{model_name}_model.pth")


# # test
# model.eval()
# with torch.no_grad():
#     for t, x, label in dataloader:
#         estimate_noise = model(x, training=False)
#         denoise_data = x - estimate_noise
#         break  # 只测试一个 batch


def test():
    t, x, label = next(iter(dataloader))

    model = TEMDnet()
    model.load_state_dict(
        torch.load("checkpoints/temdnet_model.pth", weights_only=True)
    )
    model.eval()
    with torch.no_grad():
        estimate_noise = model(x, training=False)
        denoised_data = x - estimate_noise

    time = t[0].numpy()
    clean_data = label[0].numpy()
    noisy_data = x[0].numpy()
    denoised_data = denoised_data[0].numpy()

    plt.figure(figsize=(12, 8))

    plt.plot(time * 1e3, np.abs(clean_data), "g-", linewidth=2, label="Clean Signal")
    plt.plot(time * 1e3, np.abs(noisy_data), "r-", linewidth=1, label="Noisy Signal")
    # plt.plot(time * 1e3, np.abs(denoised_data), "b-", linewidth=2, label="Denoised Signal")
    plt.xlabel("Time (ms)")
    plt.ylabel("dBz/dt (V/m²)")
    plt.title("WEF Denoising Result")
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    plt.show()

    print(f"SNR Improvement: {snr(clean_data, denoised_data):.2f} dB")
    print(f"MSE: {mse(clean_data, denoised_data):.6f}")


if __name__ == "__main__":
    train("sfsdsa")
