import torch
from torch.utils.data import Dataset
import numpy as np

import scipy.io as sio

import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
import numpy as np
import torch
from torch.utils.data import Dataset


import os
import numpy as np
import torch
from torch.utils.data import Dataset


class TEMDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        stats_file: str = "norm_stats.npy",
    ):
        """
        Args:
            data_dir: 数据所在目录
            split: 'train' | 'valid' | 'test'
            对数归一化
        """
        if split not in ["train", "valid", "test"]:
            raise ValueError("split must be 'train' or 'valid' or 'test'")

        # 文件路径映射
        file_map = {
            "train": "train_data.npy",
            "valid": "valid_data.npy",
            "test": "test_data.npy",
        }
        file_path = os.path.join(data_dir, file_map[split])
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")

        # 加载数据
        self.signal_data = np.load(file_path, allow_pickle=True)
        self.noisy_signal = np.array(
            [item["response_with_noise"] for item in self.signal_data]
        )
        self.clean_signal = np.array([item["response"] for item in self.signal_data])

        # 对数归一化
        self.noisy_signal = np.sign(self.noisy_signal) * np.log1p(
            np.abs(self.noisy_signal)
        )
        self.clean_signal = np.log1p(self.clean_signal)

        if split in ["train", "valid"]:
            min = self.noisy_signal.min()
            max = self.noisy_signal.max()
            if split == "train":
                np.save(os.path.join(data_dir, stats_file), np.array([min, max]))
        elif split == "test":
            min = np.load(os.path.join(data_dir, stats_file))[0]
            max = np.load(os.path.join(data_dir, stats_file))[1]

        else:
            raise ValueError("split must be 'train' or 'valid' or 'test'")

        self.noisy_signal = (self.noisy_signal - min) / (max - min)
        self.clean_signal = (self.clean_signal - min) / (max - min)

    def __len__(self):
        return len(self.signal_data)

    def __getitem__(self, idx):
        noisy_signal = torch.tensor(self.noisy_signal[idx], dtype=torch.float32)
        clean_signal = torch.tensor(self.clean_signal[idx], dtype=torch.float32)
        return noisy_signal, clean_signal

    @staticmethod
    def denormalize(
        signal: torch.Tensor,
        data_dir: str = "data/raw_data",
        stats_file="norm_stats.npy",
    ) -> torch.Tensor:
        """
        反归一化
        Args:
            signal: 归一化后的信号
            data_dir: 数据目录
        Returns:
            反归一化后的信号
        """
        import numpy as np
        import torch, os

        min_, max_ = np.load(os.path.join(data_dir, stats_file))
        # 转为 tensor 并与输入对齐设备与类型
        min_ = torch.tensor(min_, device=signal.device, dtype=signal.dtype)
        max_ = torch.tensor(max_, device=signal.device, dtype=signal.dtype)

        signal = signal * (max_ - min_) + min_
        signal = torch.sign(signal) * torch.expm1(torch.abs(signal))
        return signal


# class TEMDDateset(Dataset):
#     """
#     已废弃
#     """

#     def __init__(self, data_dir: str = "dataset", split: str = "train"):
#         if split not in ["train", "test"]:
#             raise ValueError("split must be 'train' or 'test'")
#         if split == "train":
#             self.clean_signals = sio.loadmat(f"{data_dir}/clean_signal.mat")[
#                 "clean_sig"
#             ]
#             self.noise_signals = sio.loadmat(f"{data_dir}/noise_signal.mat")[
#                 "noise_sig"
#             ]
#         else:
#             self.test_signals = sio.loadmat(f"{data_dir}/test_signal.mat")["test"]

#     def __len__(self):
#         return (
#             len(self.clean_signals)
#             if hasattr(self, "clean_signals")
#             else len(self.test_signals)
#         )

#     def __getitem__(self, idx):
#         if hasattr(self, "clean_signals"):
#             clean_signal = torch.tensor(self.clean_signals[idx], dtype=torch.float32)
#             noise_signal = torch.tensor(self.noise_signals[idx], dtype=torch.float32)
#             noisy_signal = clean_signal + noise_signal
#             return noisy_signal, clean_signal
#         else:
#             test_signal = torch.tensor(self.test_signals[idx], dtype=torch.float32)
#             time = torch.linspace(0, 1, steps=test_signal.shape[0])
#             return test_signal


if __name__ == "__main__":
    dataset = TEMDataset(data_dir="data/raw_data", split="train")
    print(len(dataset))
    x, label = dataset[0]
    print(x.shape, label.shape)
