import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import warnings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

from tqdm.rich import tqdm
from accelerate import Accelerator

# ====== 项目路径 ======
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import TEMDataset
from models import TEMDnet, SFSDSA, TEMSGnet


# ==================== 训练函数 ====================
def train(args):
    accelerator = Accelerator()
    device = accelerator.device
    if accelerator.is_local_main_process:
        print(f"Using device: {device}, mixed_precision: {accelerator.mixed_precision}")
        print(f"Training model: {args.model}")

    # ------- 数据加载 -------
    train_dataset = TEMDataset(data_dir=args.data_dir, split="train")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    # ------- 模型选择 -------
    if args.model == "temdnet":
        model = TEMDnet(in_channels=1, stddev=args.stddev)
    elif args.model == "sfsdsa":
        model = SFSDSA(in_features=400, stddev=args.stddev)
    elif args.model == "temsgnet":
        model = TEMSGnet(in_channels=1, stddev=args.stddev)
    else:
        raise ValueError(f"Invalid model name: {args.model}")

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.regularizer)
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_decay
    )

    # ------- accelerate 包装 -------
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    model.train()

    # ------- 训练循环 -------
    for epoch in range(args.epochs):
        if accelerator.is_local_main_process:
            print(
                f"\nEpoch [{epoch+1}/{args.epochs}] | LR: {scheduler.get_last_lr()[0]:.6f}"
            )

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"[bold cyan]Training Epoch {epoch+1}",
            unit="batch",
            colour="magenta",
            disable=not accelerator.is_local_main_process,
        )

        for x, label in progress_bar:
            x, label = x.to(device), label.to(device)
            time_emb = torch.randint(0, args.time_steps, (x.size(0),), device=device)

            estimate_noise = model(x, time_emb)

            real_noise = x - label
            loss = criterion(estimate_noise, real_noise)

            optimizer.zero_grad()
            accelerator.backward(loss)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / num_batches
        if accelerator.is_local_main_process:
            print(f"Average Loss: {avg_loss:.6f}")

    # ------- 保存模型 -------
    if accelerator.is_local_main_process:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        model_to_save = accelerator.unwrap_model(model)
        save_path = os.path.join(args.ckpt_dir, f"{args.model}_best.pth")
        torch.save(model_to_save.state_dict(), save_path)
        print(f"✅ Model saved to {save_path}")


# ==================== 主入口 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 1D Signal Denoising Models")

    # 数据 & 模型
    parser.add_argument(
        "--data_dir", type=str, default="data/raw_data/", help="训练数据路径"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="temdnet",
        choices=["temdnet", "sfsdsa", "temsgnet"],
        help="模型类型",
    )

    # 训练参数
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--regularizer", type=float, default=0, help="正则化系数")
    parser.add_argument("--lr_decay", type=float, default=1.0, help="权重衰减")
    parser.add_argument(
        "--lr_step", type=int, default=100000, help="每隔多少个 epoch 衰减一次学习率"
    )
    parser.add_argument("--time_steps", type=int, default=200, help="时间步数")
    parser.add_argument("--stddev", type=float, default=None, help="噪声标准差")

    # 其他
    parser.add_argument(
        "--ckpt_dir", type=str, default="checkpoints", help="模型保存路径"
    )

    args = parser.parse_args()
    train(args)
