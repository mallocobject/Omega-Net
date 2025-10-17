import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm.rich import tqdm  # 导入 tqdm 库
import numpy as np

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import TEMDataset
from models import TEMDnet, SFSDSA, TEMSGnet
from utils import plot

NPY_DIR = "data/raw_data/"

BATCH_SIZE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = TEMDataset(NPY_DIR, split="test")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


def test(model_name="temdnet"):
    # ======================
    # 1️⃣ 初始化模型
    # ======================
    if model_name == "temdnet":
        model = TEMDnet(in_channels=1, stddev=0.01).to(DEVICE)
    elif model_name == "sfsdsa":
        model = SFSDSA(in_features=400).to(DEVICE)
    elif model_name == "temsgnet":
        model = TEMSGnet(in_channels=1).to(DEVICE)
    else:
        raise ValueError(
            "Invalid model name. Choose 'temdnet', 'sfsdsa', or 'temsgnet'."
        )

    # ======================
    # 2️⃣ 加载模型参数
    # ======================
    model_path = f"checkpoints/{model_name}_best.pth"
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)

    print(f"✅ Loaded model weights from {model_path}")

    model.eval()
    criterion = nn.MSELoss()

    # ======================
    # 4️⃣ 测试阶段
    # ======================
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for x, label in tqdm(
            dataloader,
            desc=f"[bold cyan]Testing {model_name}",
            colour="magenta",
            unit="batch",
        ):
            x, label = x.to(DEVICE), label.to(DEVICE)
            time_emb = torch.randint(0, 1000, (x.size(0),)).to(DEVICE)

            estimate_noise = model(x) if model_name != "unet1d" else model(x, time_emb)
            real_noise = x - label
            loss = criterion(estimate_noise, real_noise)

            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    print(f"📉 Average Loss on Test Set: {avg_loss:.6f}")

    return avg_loss


if __name__ == "__main__":
    test("temdnet")
    # test("sfsdsa")
