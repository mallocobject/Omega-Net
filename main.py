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

NPY_DIR = "data/raw_data"

BATCH_SIZE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = TEMDataset(
    NPY_DIR,
    split="test",
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


model_name = "temdnet"  # 可选 "temdnet", "sfsdsa", "temsgnet"
if model_name == "temdnet":
    model = TEMDnet(stddev=0.05).to(DEVICE)
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
# vis_x, vis_label = next(iter(dataloader))  # 只取第一批数据
# 随机取一条数据进行可视化
vis_idx = np.random.randint(0, len(dataset))
vis_x, vis_label = dataset[vis_idx]
vis_x = vis_x.unsqueeze(0).to(DEVICE)  # 增加批次维度
vis_label = vis_label.unsqueeze(0).to(DEVICE)

signal_data = np.load("data/raw_data/test_data.npy", allow_pickle=True)
ori_sig = np.array([item["response_with_noise"] for item in signal_data])[vis_idx]

# 加载归一化参数
stats_path = os.path.join(NPY_DIR, "norm_stats.npy")
stats = np.load(stats_path)
min, max = stats[0], stats[1]


with torch.no_grad():

    if model_name != "temsgnet":
        estimate_noise = model(vis_x)
        denoised_signal = vis_x - estimate_noise
    else:
        estimate_noise = model.denoise_from_noisy(vis_x, vis_x, 100)
        denoised_signal = estimate_noise  # TEMSGnet 直接输出去噪结果

    # 转回 CPU 并反标准化
    noisy_signal = vis_x[0].cpu().numpy()
    clean_signal = vis_label[0].cpu().numpy()
    denoised_signal = denoised_signal[0].cpu().numpy()

    t = np.linspace(0, 400, 400)  # 时间轴（ms）

    plot(
        t,
        clean_signal,
        ori_sig,
        denoised_signal,
        x_axis="time (ms)",
        y_axis="Amp (mV)",
        title=f"{model_name} Denoising Result",
    )
