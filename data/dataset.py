import torch
from torch.utils.data import Dataset
import numpy as np

import scipy.io as sio

import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TEMDataset(Dataset):
    def __init__(self, data_dir: str | list, split: str = "train"):
        data_path = glob.glob(os.path.join(data_dir, "raw_tem_data_batch_*.npy"))
        if isinstance(data_path, str):
            pack_data = np.load(data_path, allow_pickle=True)
            self.signal_data = pack_data.tolist()
        elif isinstance(data_path, list):
            self.signal_data = []

            # 遍历每个文件路径，加载数据
            if split not in ["train", "test"]:
                raise ValueError("split must be 'train' or 'test'")
            if split == "train":
                data_path = data_path[:-2]  # 取前面的文件作为训练集
            else:
                data_path = data_path[-2:]  # 取最后一个文件作为测试集
            for data_path in data_path:
                pack_data = np.load(data_path, allow_pickle=True)
                self.signal_data.extend(pack_data.tolist())  # 合并数据

        self.time = np.array([item["time"] for item in self.signal_data])
        self.noisy_signal = np.array(
            [item["response_with_noise_and_impulse"] for item in self.signal_data]
        )
        self.clean_signal = np.array([item["response"] for item in self.signal_data])

    def __len__(self):
        return len(self.signal_data)

    def __getitem__(self, idx):
        noisy_signal = torch.tensor(self.noisy_signal[idx], dtype=torch.float32)
        time = torch.tensor(self.time[idx], dtype=torch.float32)
        clean_signal = torch.tensor(self.clean_signal[idx], dtype=torch.float32)

        return time, noisy_signal, clean_signal


class TEMDDateset(Dataset):
    def __init__(self, data_dir: str = "dataset", split: str = "train"):
        if split not in ["train", "test"]:
            raise ValueError("split must be 'train' or 'test'")
        if split == "train":
            self.clean_signals = sio.loadmat(f"{data_dir}/clean_signal.mat")[
                "clean_sig"
            ]
            self.noise_signals = sio.loadmat(f"{data_dir}/noise_signal.mat")[
                "noise_sig"
            ]
        else:
            self.test_signals = sio.loadmat(f"{data_dir}/test_signal.mat")["test"]

    def __len__(self):
        return (
            len(self.clean_signals)
            if hasattr(self, "clean_signals")
            else len(self.test_signals)
        )

    def __getitem__(self, idx):
        if hasattr(self, "clean_signals"):
            clean_signal = torch.tensor(self.clean_signals[idx], dtype=torch.float32)
            noise_signal = torch.tensor(self.noise_signals[idx], dtype=torch.float32)
            noisy_signal = clean_signal + noise_signal
            return noisy_signal, clean_signal
        else:
            test_signal = torch.tensor(self.test_signals[idx], dtype=torch.float32)
            time = torch.linspace(0, 1, steps=test_signal.shape[0])
            return test_signal


if __name__ == "__main__":
    dataset = TEMDataset(data_dir="data/raw_data", split="train")
    print(len(dataset))
    t, x, label = dataset[0]
    print(t.shape, x.shape, label.shape)

    dataset = TEMDDateset(data_dir="dataset", split="train")
    print(len(dataset))
    x, label = dataset[0]
    print(x.shape, label.shape)
