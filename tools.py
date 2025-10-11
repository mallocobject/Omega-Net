import numpy as np
import torch
import math


# SNR
def snr(clean, denoised_signal):
    noise = clean - denoised_signal
    snr_value = 10 * np.log10(np.mean(clean**2) / np.mean(noise**2))
    return snr_value


# MSE
def mse(clean, denoised_signal):
    return np.mean((clean - denoised_signal) ** 2)


def seq2img(seq: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    # 1D to 2D
    if isinstance(seq, np.ndarray):
        if seq.ndim == 1:
            length = len(seq)
            sqrt_n = int(math.ceil(math.sqrt(length)))
            padded_length = sqrt_n**2
            if length < padded_length:
                seq = np.pad(seq, (0, padded_length - length), mode="edge")
            img = seq.reshape(sqrt_n, sqrt_n)
            # Z字形填充
            img[1::2] = img[1::2, ::-1]
        elif seq.ndim == 2:
            batch_size = seq.shape[0]
            length = seq.shape[1]
            sqrt_n = int(math.ceil(math.sqrt(length)))
            padded_length = sqrt_n**2
            if length < padded_length:
                seq = np.pad(seq, ((0, 0), (0, padded_length - length)), mode="edge")

            # 转换为三维张量 [batch_size, sqrt_n, sqrt_n]
            img = seq.reshape(batch_size, sqrt_n, sqrt_n)
            img[:, 1::2] = img[:, 1::2, ::-1]
        else:
            raise ValueError("Input signal must be 1D or 2D.")
    elif isinstance(seq, torch.Tensor):
        if seq.dim() == 1:
            length = seq.size(0)
            sqrt_n = int(math.ceil(math.sqrt(length)))
            padded_length = sqrt_n**2
            if length < padded_length:
                padding = (0, padded_length - length)
                seq = torch.nn.functional.pad(seq, padding, "constant", value=seq[-1])

            img = seq.reshape(sqrt_n, sqrt_n)
            img[1::2] = img[1::2].flip(dims=(1,))
        elif seq.dim() == 2:
            batch_size = seq.size(0)
            length = seq.size(1)
            sqrt_n = int(math.ceil(math.sqrt(length)))
            padded_length = sqrt_n**2
            if length < padded_length:
                pad_amount = padded_length - length
                padding = (0, pad_amount)
                fill_values = seq[:, -1]
                seq = torch.nn.functional.pad(seq, padding, "constant", value=0)
                seq[:, length:padded_length] = fill_values[:, None].expand(
                    -1, pad_amount
                )
            img = seq.reshape(batch_size, sqrt_n, sqrt_n)
            img[:, 1::2] = img[:, 1::2].flip(dims=(2,))
    else:
        raise TypeError("Input must be a numpy array or a torch tensor.")

    return img


def img2seq(img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    # 2D to 1D
    pass


if __name__ == "__main__":
    # 测试 1: 二维 NumPy 输入，完美平方长度 (batch_size=3, length=4)
    a_np = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    print("测试 1: 二维 NumPy 输入 (batch_size=3, length=4)")
    print("输入:")
    print(a_np)
    b_np = seq2img(a_np)
    print("输出:")
    print(b_np)

    # 测试 2: 二维 NumPy 输入，非完美平方长度 (batch_size=2, length=3)
    a_np_non_square = np.array([[0, 1, 2], [3, 4, 5]])
    print("\n测试 2: 二维 NumPy 输入 (batch_size=2, length=3)")
    print("输入:")
    print(a_np_non_square)
    b_np_non_square = seq2img(a_np_non_square)
    print("输出:")
    print(b_np_non_square)

    # 测试 3: 二维 PyTorch 输入，完美平方长度 (batch_size=2, length=4)
    a_torch = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    print("\n测试 3: 二维 PyTorch 输入 (batch_size=2, length=4)")
    print("输入:")
    print(a_torch)
    b_torch = seq2img(a_torch)
    print("输出:")
    print(b_torch)

    # 测试 4: 二维 PyTorch 输入，非完美平方长度 (batch_size=2, length=5)
    a_torch_non_square = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    print("\n测试 4: 二维 PyTorch 输入 (batch_size=2, length=5)")
    print("输入:")
    print(a_torch_non_square)
    b_torch_non_square = seq2img(a_torch_non_square)
    print("输出:")
    print(b_torch_non_square)
