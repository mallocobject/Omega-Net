import torch
from torch.utils.data import Dataset
import numpy as np


class TEMDataset(Dataset):
    def __init__(self, data_path: str | list):
        if isinstance(data_path, str):
            pack_data = np.load(data_path, allow_pickle=True)
            self.signal_data = pack_data.tolist()
        elif isinstance(data_path, list):
            self.signal_data = []

            # 遍历每个文件路径，加载数据
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


if __name__ == "__main__":
    dataset_1 = TEMDataset(data_path="./data/raw_data/raw_tem_data_batch_10.npy")
    dataset_2 = TEMDataset(data_path="./data/raw_data/raw_tem_data_batch_9.npy")

    time, noisy_signal, clean_signal = dataset_1[0]
    print(noisy_signal[:10])
    print(clean_signal[:10])

    time, noisy_signal, clean_signal = dataset_2[0]
    print(noisy_signal[:10])
    print(clean_signal[:10])
