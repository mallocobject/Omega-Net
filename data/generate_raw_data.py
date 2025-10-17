import numpy as np
import os
import sys
from tqdm.rich import tqdm  # 导入 tqdm 库

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_simple_tem_signal


def save_data_to_npy(num_samples: int, file_name: str):
    assert num_samples > 0, "Number of samples must be greater than 0"

    current_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(current_dir, "raw_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    all_signals = []

    # 使用 tqdm 包装 for 循环，显示进度条
    for i in tqdm(
        range(num_samples),
        desc=f"[bold cyan]Generating samples",
        unit="sample",
        colour="magenta",
    ):
        _, response, response_with_noise = get_simple_tem_signal()
        signal_data = {
            "response": response,
            "response_with_noise": response_with_noise,
            "x axis": "time (ms)",
            "y axis": "B (nT)",
        }

        all_signals.append(signal_data)

    file_path = os.path.join(data_dir, f"{file_name}.npy")
    np.save(file_path, all_signals)
    print(f"✅ Saved {len(all_signals)} samples to {file_path}")


if __name__ == "__main__":
    np.random.seed(None)  # 确保每次运行生成不同的数据
    save_data_to_npy(num_samples=10000, file_name="train_data")
    save_data_to_npy(num_samples=100, file_name="test_data")
