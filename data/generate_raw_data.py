import numpy as np
import os
import sys
from tqdm.rich import tqdm  # 导入 tqdm 库

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import tem_generator


def save_data_to_npy(num_samples: int, file_name: str):
    assert num_samples > 0, "Number of samples must be greater than 0"
    assert num_samples % 100 == 0, "Number of samples must be a multiple of 100"

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
        response, _, response_with_noise_and_impulse = tem_generator.get_tem_signal()
        signal_data = {
            "response": response,
            "response_with_noise_and_impulse": response_with_noise_and_impulse,
            "x axis": "time (ms)",
            "y axis": "B (nT)",
        }

        all_signals.append(signal_data)

        if (i + 1) % 100 == 0:
            # 在文件名中添加批次号，避免文件被覆盖
            batch_file_name = f"{file_name}_batch_{(i + 1) // 100}.npy"

            # 将信号数据保存到当前脚本同级文件夹
            np.save(os.path.join(data_dir, batch_file_name), all_signals)
            # print(f"Saved {i + 1} samples to {batch_file_name}")

            # 重置 all_signals 列表，准备保存下一批次的数据
            all_signals = []


if __name__ == "__main__":
    save_data_to_npy(num_samples=1200, file_name="raw_tem_data")
