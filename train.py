import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm.rich import tqdm  # 导入 tqdm 库
import numpy as np

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import dataset
from models import TEMDnet, SFSDSA, UNet1D

NPY_DIR = "dataset/"
EPOCHS = 200
BATCH_SIZE = 128
LR = 1e-3
STDDEV = 0.01
LEARN_WEIGHT_DECAY = 0.02
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = dataset.TEMDDateset(data_dir=NPY_DIR, split="train")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def train(model_name="temdnet"):
    if model_name == "temdnet":
        model = TEMDnet(in_channels=1, stddev=STDDEV).to(DEVICE)
    elif model_name == "sfsdsa":
        model = SFSDSA(in_features=400).to(DEVICE)
    elif model_name == "unet1d":
        model = UNet1D(in_channels=1, out_channels=1, num_features=32, num_levels=4).to(
            DEVICE
        )
    else:
        raise ValueError("Invalid model name. Choose 'temdnet', 'sfsdsa', or 'unet1d'.")

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=LEARN_WEIGHT_DECAY)

    model.train()

    for epoch in range(EPOCHS):
        print(f"Epoch [{epoch+1}/{EPOCHS}]")

        total_loss = 0  # 用于累计每个 epoch 的损失
        total_batches = 0  # 用于统计每个 epoch 中的 batch 数量

        for x, label in tqdm(
            dataloader,
            desc=f"[bold cyan]Training Epoch {epoch+1}",
            colour="magenta",
            unit="batch",
        ):
            x, label = x.to(DEVICE), label.to(DEVICE)
            time_emb = torch.randint(0, 1000, (x.size(0),)).to(DEVICE)  # 随机时间步
            estimate_noise = model(x) if model_name != "unet1d" else model(x, time_emb)
            real_noise = x - label

            loss = criterion(estimate_noise, real_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # 累加损失
            total_batches += 1  # 累加 batch 数量

        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        print(f"Average Loss for Epoch {epoch+1}: {avg_loss:.6f}")

    # 保存模型
    model_save_path = f"checkpoints/{model_name}_best.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    print(torch.cuda.is_available())
    train("temdnet")
