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
        normalize: bool = True,
        method: str = "zscore",
        mean: float = None,
        std: float = None,
        x_min: float = None,
        x_max: float = None,
    ):
        """
        Args:
            data_dir: 数据所在目录
            split: 'train' | 'valid' | 'test'
            normalize: 是否进行标准化
            method: 'zscore' 或 'minmax'
            mean/std/x_min/x_max: 用于验证/测试集的标准化参数
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

        # 归一化选项
        self.normalize = normalize
        self.method = method

        # ====== 仅在 train 阶段计算参数 ======
        if self.normalize and split == "train":
            if method == "zscore":
                self.mean = self.clean_signal.mean()
                self.std = self.clean_signal.std()
                self.x_min = self.x_max = None
            elif method == "minmax":
                self.x_min = self.clean_signal.min()
                self.x_max = self.clean_signal.max()
                self.mean = self.std = None
        # ====== valid/test 直接使用传入的参数 ======
        elif self.normalize:
            self.mean = mean
            self.std = std
            self.x_min = x_min
            self.x_max = x_max

        # ====== 统一标准化 ======
        if self.normalize:
            if method == "zscore":
                self.noisy_signal = (self.noisy_signal - self.mean) / (self.std + 1e-8)
                self.clean_signal = (self.clean_signal - self.mean) / (self.std + 1e-8)
            elif method == "minmax":
                self.noisy_signal = (
                    2
                    * (
                        (self.noisy_signal - self.x_min)
                        / (self.x_max - self.x_min + 1e-8)
                    )
                    - 1
                )
                self.clean_signal = (
                    2
                    * (
                        (self.clean_signal - self.x_min)
                        / (self.x_max - self.x_min + 1e-8)
                    )
                    - 1
                )

    def __len__(self):
        return len(self.signal_data)

    def __getitem__(self, idx):
        noisy_signal = torch.tensor(self.noisy_signal[idx], dtype=torch.float32)
        clean_signal = torch.tensor(self.clean_signal[idx], dtype=torch.float32)
        return noisy_signal, clean_signal

    # ===== 静态方法：反标准化 =====
    @staticmethod
    def denormalize_signal(x_norm: torch.Tensor, params: dict, method: str = "zscore"):
        """反标准化（从标准化后信号恢复原始尺度）"""
        if method == "zscore":
            return x_norm * params["std"] + params["mean"]
        elif method == "minmax":
            return (x_norm + 1) / 2 * (params["x_max"] - params["x_min"]) + params[
                "x_min"
            ]
        else:
            raise ValueError("method must be 'zscore' or 'minmax'")


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
