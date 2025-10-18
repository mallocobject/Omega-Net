import argparse
import torch

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exps import DenoisingExperiment


import argparse


def get_args():
    parser = argparse.ArgumentParser(description="TEM 一维信号去噪实验")

    # ===============================
    # 数据与模型相关参数
    # ===============================
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/raw_data/",
        help="数据集路径(train/test 通用)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="temdnet",
        choices=["temdnet", "sfsdsa", "temsgnet"],
        help="选择使用的模型结构",
    )
    parser.add_argument(
        "--time_steps", type=int, default=200, help="输入信号的时间步长"
    )
    parser.add_argument(
        "--stddev", type=float, default=None, help="噪声标准差（仅用于训练数据生成）"
    )

    # ===============================
    # 模式选择
    # ===============================
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="运行模式:train 或 test",
    )

    # ===============================
    # 训练参数（仅 train 模式使用）
    # ===============================
    parser.add_argument_group("Training Parameters")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="训练批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--regularizer", type=float, default=0.0, help="L2 正则化系数")
    parser.add_argument("--lr_decay", type=float, default=1.0, help="学习率衰减系数")
    parser.add_argument(
        "--lr_step", type=int, default=100000, help="每隔多少个 epoch 衰减一次学习率"
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="./checkpoints", help="模型权重保存目录"
    )

    # ===============================
    # 测试参数（仅 test 模式使用）
    # ===============================
    parser.add_argument_group("Testing Parameters")
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="测试时加载的模型权重文件路径 (.pt/.pth)",
    )

    # ===============================
    # 其他设置
    # ===============================
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # ===============================
    # 自动模式修正逻辑
    # ===============================
    if args.mode == "test":
        # 测试时不需要训练相关参数
        args.epochs = None
        args.batch_size = None
        args.lr = None
        args.lr_decay = None
        args.lr_step = None
        args.regularizer = None
        # 必须提供加载路径
        if args.load_checkpoint is None:
            parser.error("--load_checkpoint 必须在测试模式下提供。")

    return args


def set_seed(seed: int):
    import random
    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = get_args()

    # 固定随机种子，保证可复现性
    set_seed(args.seed)

    # 创建权重保存路径
    os.makedirs(args.ckpt_dir, exist_ok=True)

    task = DenoisingExperiment(args)

    if args.mode == "train":
        task.train()
    elif args.mode == "test":
        task.test()
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
