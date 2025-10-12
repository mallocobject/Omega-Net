import numpy as np
from matplotlib import pyplot as plt

WINDOW = 5


def smooth(x: np.ndarray, win_size: int = WINDOW):
    if len(x.shape) > 1 and x.shape[0] == 1:
        x = x.flatten()

    if win_size > len(x):
        win_size = len(x)
    if win_size < 1:
        win_size = 1

    y = np.convolve(x, np.ones(win_size, dtype=int), mode="valid") / win_size

    r = np.arange(1, win_size - 1, 2)
    start = np.cumsum(x[: win_size - 1])[::2] / r
    end = (np.cumsum(x[:-win_size:-1])[::2] / r)[::-1]

    return np.concatenate([start, y, end])


def get_noise(noise_level: float, noise_size: int):
    b = np.random.rand(noise_size) * noise_level
    return b


def add_noise_stddev(signal: np.ndarray, stddev: float):
    noise = np.random.normal(0, stddev, signal.shape)
    return signal + noise


def add_noise_snr(signal: np.ndarray, snr_db: float):
    signal_power = np.mean(signal**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise


def get_abrupt_part(x: np.ndarray):
    if len(x.shape) > 1:
        out = np.zeros_like(x)
        for i in range(x.shape[0]):
            y = x[i] - smooth(x[i])
            out[i] = y
    else:
        out = x - smooth(x)
    return out


def simulation_data(
    sigma=True,
    noise_level=0.1,
    signal_size=400,
    batch_size=1,
    k1_min=50000,
    k1_max=120000,
    k2_min=10,
    k2_max=40,
    b_min=1500,
    b_max=2000,
):
    t = np.linspace(0, 4, signal_size)
    if batch_size == 1:
        k1 = np.random.randint(k1_min, k1_max)
        k2 = np.random.randint(k2_min, k2_max)
        b = np.random.randint(b_min, b_max)

        y = k1 * np.exp(-k2 * t) + b
        if sigma:
            out = y + get_noise(noise_level=noise_level, noise_size=len(y))
        else:
            out = y

    else:
        out = np.zeros((batch_size, signal_size))
        for i in range(batch_size):
            k1 = np.random.randint(k1_min, k1_max)
            k2 = np.random.randint(k2_min, k2_max)
            b = np.random.randint(b_min, b_max)

            y = k1 * np.exp(-k2 * t) + b
            if sigma:
                out[i] = y + get_noise(noise_level=noise_level, noise_size=len(y))
            else:
                out[i] = y
    return out


def pair_simulation_data(noise_level, data_size, batch_size=1):
    if batch_size > 1:
        clean = simulation_data(
            sigma=False,
            noise_level=noise_level,
            signal_size=data_size,
            batch_size=batch_size,
        )
        noisy = np.zeros_like(clean)
        for i in range(batch_size):
            noisy[i] = clean[i] + get_noise(noise_level, data_size)
    else:
        clean = simulation_data(
            sigma=False, noise_level=noise_level, signal_size=data_size
        )
        noisy = clean + get_noise(noise_level=noise_level, noise_size=data_size)
    return clean, noisy


if __name__ == "__main__":
    # 测试1: 基本功能测试
    print("=== 测试1: 基本功能测试 ===")

    # 生成测试数据
    clean_data, noisy_data = pair_simulation_data(
        noise_level=1000, data_size=400, batch_size=1
    )

    print(f"Clean data shape: {clean_data.shape}")
    print(f"Noisy data shape: {noisy_data.shape}")

    # 平滑处理
    smoothed_data = smooth(noisy_data)
    print(f"Smoothed data shape: {smoothed_data.shape}")

    # 提取突变成分
    abrupt_part = get_abrupt_part(noisy_data)
    print(f"Abrupt part shape: {abrupt_part.shape}")

    # 测试2: 批量数据处理
    print("\n=== 测试2: 批量数据处理 ===")
    batch_clean, batch_noisy = pair_simulation_data(
        noise_level=1000, data_size=400, batch_size=3
    )
    print(f"Batch clean shape: {batch_clean.shape}")
    print(f"Batch noisy shape: {batch_noisy.shape}")

    batch_abrupt = get_abrupt_part(batch_noisy)
    print(f"Batch abrupt shape: {batch_abrupt.shape}")

    # 可视化结果
    print("\n=== 可视化结果 ===")
    t = np.linspace(0, 4, 400)

    # 单个信号的可视化
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(t, clean_data, "g-", label="Clean Signal", linewidth=2)
    plt.plot(t, noisy_data, "b-", label="Noisy Signal", alpha=0.7)
    plt.title("Original Signals")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(t, noisy_data, "b-", label="Noisy Signal", alpha=0.7)
    plt.plot(t, smoothed_data, "r-", label="Smoothed Signal", linewidth=2)
    plt.title("Smoothing Effect")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(t, abrupt_part, "m-", label="Abrupt Part", linewidth=2)
    plt.title("Abrupt Component")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    # 显示批量数据中的第一个样本
    plt.plot(t, batch_clean[0], "g-", label="Clean (Batch 0)", linewidth=2)
    plt.plot(t, batch_noisy[0], "b-", label="Noisy (Batch 0)", alpha=0.7)
    plt.plot(t, batch_abrupt[0], "m-", label="Abrupt (Batch 0)", linewidth=1)
    plt.title("Batch Processing Example")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 测试3: 参数范围测试
    print("\n=== 测试3: 参数范围测试 ===")

    # 测试不同噪声水平
    noise_levels = [500, 1000, 2000]
    plt.figure(figsize=(15, 5))

    for i, noise_level in enumerate(noise_levels):
        _, test_noisy = pair_simulation_data(noise_level=noise_level, data_size=400)
        test_smooth = smooth(test_noisy)
        test_abrupt = get_abrupt_part(test_noisy)

        plt.subplot(1, 3, i + 1)
        plt.plot(t, test_noisy, "b-", alpha=0.7, label="Noisy")
        plt.plot(t, test_smooth, "r-", label="Smooth")
        plt.plot(t, test_abrupt, "m-", label="Abrupt")
        plt.title(f"Noise Level: {noise_level}")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("所有测试完成!")
