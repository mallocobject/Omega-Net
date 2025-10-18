import argparse
import torch

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exps import DenoisingExperiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/Test 1D Signal Denoising Models"
    )

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
    parser.add_argument("--lr_decay", type=float, default=1.0, help="学习率衰减系数")
    parser.add_argument(
        "--lr_step", type=int, default=100000, help="每隔多少个 epoch 衰减一次学习率"
    )
    parser.add_argument("--time_steps", type=int, default=200, help="时间步数")
    parser.add_argument("--stddev", type=float, default=None, help="噪声标准差")

    # 模式选择
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="运行模式:train 或 test",
    )

    # 其他
    parser.add_argument(
        "--ckpt_dir", type=str, default="checkpoints", help="模型保存路径"
    )

    args = parser.parse_args()

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
    args = parse_args()

    # 固定随机种子，保证可复现性
    set_seed(args.seed)

    # 创建权重保存路径
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # 初始化实验类
    experiment = DenoisingExperiment(args)

    if args.mode == "train":
        experiment.train()
    elif args.mode == "test":
        experiment.test()
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
