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
from models import TEMDnet, SFSDSA, TEMSGnet
from utils import plot

NPY_DIR = "data/raw_data"

BATCH_SIZE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = dataset.TEMDataset(NPY_DIR, split="test")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


model_name = "sfsdsa"  # 可选 "temdnet", "sfsdsa", "temsgnet"
if model_name == "temdnet":
    model = TEMDnet(stddev=0.01).to(DEVICE)
elif model_name == "sfsdsa":
    model = SFSDSA().to(DEVICE)
elif model_name == "temsgnet":
    model = TEMSGnet().to(DEVICE)

# ======================
# 2️⃣ 加载模型参数
# ======================
model_path = f"checkpoints/{model_name}_best.pth"
state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict, strict=False)

print(f"✅ Loaded model weights from {model_path}")

model.eval()
criterion = nn.MSELoss()

# ======================
# 3️⃣ 提前取出一批数据用于可视化
# ======================
vis_x, vis_label = next(iter(dataloader))  # 只取第一批数据
vis_x, vis_label = vis_x[0:1].to(DEVICE), vis_label[0:1].to(DEVICE)

with torch.no_grad():
    time_emb = torch.randint(0, 200, (vis_x.size(0),)).to(DEVICE)
    estimate_noise = model(vis_x, time_emb)
    denoised_signal = vis_x - estimate_noise

    noisy_signal = vis_x[0].cpu().numpy()
    clean_signal = vis_label[0].cpu().numpy()
    denoised_signal = denoised_signal[0].cpu().numpy()

    t = np.linspace(0, 400, 400)  # 时间轴（ms）

    plot(
        t,
        clean_signal,
        noisy_signal,
        denoised_signal,
        x_axis="time (ms)",
        y_axis="B (nT)",
        title=f"{model_name} Denoising Result",
    )
