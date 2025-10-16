import numpy as np
import torch
import math
import matplotlib.pyplot as plt


def exists(x):
    return x is not None


def default(val, fn):
    if exists(val):
        return val
    return fn() if callable(fn) else fn


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

    if img is None:
        raise ValueError("Image could not be generated from input.")

    return img


def img2seq(
    img: np.ndarray | torch.Tensor, origin_len: int = None
) -> np.ndarray | torch.Tensor:
    # 2D to 1D
    origin_len = origin_len if origin_len else img.shape[-1] * img.shape[-2]
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            img[1::2] = img[1::2, ::-1]
            seq = img.flatten()
            seq = seq[:origin_len]
        elif img.ndim == 3:
            img[:, 1::2] = img[:, 1::2, ::-1]
            seq = img.reshape(img.shape[0], -1)
            seq = seq[:, :origin_len]
        else:
            raise ValueError("Input image must be 2D or 3D.")
    elif isinstance(img, torch.Tensor):
        if img.dim() == 2:
            img[1::2] = img[1::2].flip(dims=(1,))
            seq = img.flatten()
            seq = seq[:origin_len]
        elif img.dim() == 3:
            img[:, 1::2] = img[:, 1::2].flip(dims=(2,))
            seq = img.reshape(img.size(0), -1)
            seq = seq[:, :origin_len]
        else:
            raise ValueError("Input image must be 2D or 3D.")
    else:
        raise TypeError("Input must be a numpy array or a torch tensor.")

    if seq is None:
        raise ValueError("Sequence could not be generated from image.")

    return seq


def plot(
    t,
    *sig,
    x_axis="time (ms)",
    y_axis="B (nT)",
    title="Signal",
):
    plt.figure(figsize=(12, 8))

    # sig[0]: clean signal
    # sig[1]: noisy signal
    # sig[2]: denoised signal
    labels = ["Clean Signal", "Noisy Signal", "Denoised Signal"]
    colors = ["g-", "r-", "b-"]
    linewidths = [2, 1, 2]
    for i in range(len(sig)):
        plt.plot(
            t * 1e3, np.abs(sig[i]), colors[i], linewidth=linewidths[i], label=labels[i]
        )

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # 测试 1: 二维 NumPy 输入，完美平方长度 (batch_size=3, length=4)
    a_np = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    print("测试 1: 二维 NumPy 输入 (batch_size=3, length=4)")
    print("输入:")
    print(a_np)
    b_np = seq2img(a_np)
    print("输出:")
    print(b_np)

    c_np = img2seq(b_np)

    print("复原输出:")
    print(c_np)

    # 测试 2: 二维 NumPy 输入，非完美平方长度 (batch_size=2, length=3)
    a_np_non_square = np.array([[0, 1, 2], [3, 4, 5]])
    print("\n测试 2: 二维 NumPy 输入 (batch_size=2, length=3)")
    print("输入:")
    print(a_np_non_square)
    b_np_non_square = seq2img(a_np_non_square)
    print("输出:")
    print(b_np_non_square)

    c_np_non_square = img2seq(b_np_non_square, a_np_non_square.shape[-1])
    print("复原输出:")
    print(c_np_non_square)

    # 测试 3: 二维 PyTorch 输入，完美平方长度 (batch_size=2, length=4)
    a_torch = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    print("\n测试 3: 二维 PyTorch 输入 (batch_size=2, length=4)")
    print("输入:")
    print(a_torch)
    b_torch = seq2img(a_torch)
    print("输出:")
    print(b_torch)

    c_torch = img2seq(b_torch)
    print("复原输出:")
    print(c_torch)

    # 测试 4: 二维 PyTorch 输入，非完美平方长度 (batch_size=2, length=5)
    a_torch_non_square = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    print("\n测试 4: 二维 PyTorch 输入 (batch_size=2, length=5)")
    print("输入:")
    print(a_torch_non_square)
    b_torch_non_square = seq2img(a_torch_non_square)
    print("输出:")
    print(b_torch_non_square)

    c_torch_non_square = img2seq(b_torch_non_square, a_torch_non_square.shape[-1])
    print("复原输出:")
    print(c_torch_non_square)
