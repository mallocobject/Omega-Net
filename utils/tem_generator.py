import numpy as np
import empymod
import matplotlib.pyplot as plt
import os
import sys


def add_noise_stddev(signal: np.ndarray, stddev: float):
    noise = np.random.normal(0, stddev, signal.shape)
    return signal + noise


def add_noise_snr(signal: np.ndarray, snr_db: float):
    signal_power = np.mean(signal**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise


def add_colored_noise(signal, beta=1.0, scale=0.05):
    """添加 1/f^beta 噪声,beta=1为粉噪声"""
    N = len(signal)
    freqs = np.fft.rfftfreq(N)
    freqs[0] = 1e-6
    amplitude = 1 / (freqs ** (beta / 2))
    amplitude = amplitude / np.sqrt(np.mean(amplitude**2))  # RMS归一化

    noise_fft = np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs))
    noise = np.fft.irfft(noise_fft * amplitude)
    noise = noise / np.max(np.abs(noise)) * np.std(signal) * scale
    return signal + noise


# https://ieeexplore.ieee.org/document/9698089
def get_simple_tem_signal(
    noise_stddev: float = 500,
    k1: tuple = (5e4, 12e4),
    k2: tuple = (10, 40),
    b: tuple = (1500, 2000),
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成简单的一维瞬变电磁(TEM)信号
    使用指数衰减模型模拟1D地层的瞬变电磁响应
    k1, k2, b: 控制指数衰减模型的参数
    noise_stddev: 高斯噪声的标准差
    time: 100ms
    response: nT
    """
    time = np.linspace(0, 4, 400)  # 时间采样点
    k1 = np.random.randint(*k1)
    k2 = np.random.randint(*k2)
    b = np.random.randint(*b)
    response = k1 * np.exp(-k2 * time) + b  # 指数衰减模型

    response_with_noise = add_noise_stddev(response, noise_stddev)

    time = time * 100

    return time, response, response_with_noise


# https://arxiv.org/html/2510.13160
def get_simple_tem_signal_v2(
    snr_db: float = 20,
    Q1: tuple = (100, 1500),
    Q2: tuple = (0.5, 4.0),
    B: tuple = (2.0, 6.0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成简单的一维瞬变电磁(TEM)信号（改进版）
    使用双指数衰减模型模拟1D地层的瞬变电磁响应
    Q1, Q2, B: 控制双指数衰减模型的参数
    snr_db: 20~25dB 信噪比
    return:
    time: 0~400ms
    """
    time = np.linspace(0, 4, 400)  # 时间采样点
    Q1 = np.random.uniform(*Q1)
    Q2 = np.random.uniform(*Q2)
    B = np.random.uniform(*B)
    # Q1 = 1300
    # Q2 = 2.5
    # B = 4.0
    response = Q1 * np.exp(-Q2 * time) + B  # 指数衰减模型

    response_with_noise = add_noise_snr(response, snr_db)

    time = time * 100

    return time, response, response_with_noise


# deprecated
def get_tem_signal(
    noise_stddev: float = 500,
    min_impulse: float = -500,
    max_impulse: float = 1000,
    num_impulse: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    (已废弃)
    生成一维瞬变电磁(TEM)信号
    使用empymod库模拟更接近真实的1D地层的瞬变电磁响应
    """
    # 地层参数（基于常见地质条件），加上随机扰动
    resistivity = [
        1e12,
        100.0 * np.random.uniform(0.8, 1.2),
        10.0 * np.random.uniform(0.8, 1.2),
        500.0 * np.random.uniform(0.8, 1.2),
        200.0 * np.random.uniform(0.8, 1.2),
    ]
    thickness = [
        10.0 * np.random.uniform(0.9, 1.1),
        50.0 * np.random.uniform(0.9, 1.1),
        100.0 * np.random.uniform(0.9, 1.1),
    ]

    depth = [0.0] + list(np.cumsum(thickness))  # 地层深度

    # 回线源（矩形回线）
    coil_length = 30.0
    coil_width = 25.0
    src = [
        -coil_length / 2,  # 回线左下角 x 坐标
        -coil_width / 2,  # 回线左下角 y 坐标
        0.0,  # 回线 z 坐标（地面）
        coil_length / 2,  # 回线右上角 x 坐标
        coil_width / 2,  # 回线右上角 y 坐标
        0.0,  # 回线 z 坐标（地面）
    ]

    # 接收器（点接收器，位置设为回线源中心，略高于地面以避免数值问题）
    rec = [
        0.0,  # x坐标
        0.0,  # y坐标
        0.1,  # z坐标（略高于地面以避免数值问题）
        0.0,  # x方向接收器方向（默认为0）
        0.0,  # y方向接收器方向（默认为0）
    ]

    # 时间采样
    offset = 10  # 偏移量
    time = np.linspace(1e-20, 0.05, 400 + offset)

    # 计算源强度（假设电流为10安，回线面积为 300m * 250m）
    strength = 10.0 * coil_length * coil_width  # 10安 × 回线面积

    # 计算TEM响应
    response = empymod.loop(
        src=src,  # 回线源位置
        rec=rec,  # 接收器位置
        depth=depth,  # 地层深度
        res=resistivity,  # 地层电阻率
        freqtime=time,  # 时间域采样点
        signal=-1,  # 断开信号（步进关断，典型TEM设置）
        mrec=False,  # 计算磁感应强度 B
        recpts=1,  # 单个接收点
        strength=strength,  # 设置源强度
        verb=1,  # 输出较少信息（提高计算效率）
        htarg={"dlf": "key_201_2012"},  # Hankel变换滤波器，确保数值稳定性
    )

    response = np.abs(response) * 1e9  # 转换为 nT
    # response = response - 2000  # 增加直流

    response_with_noise = add_noise_stddev(response, noise_stddev)

    # 加入脉冲噪声（脉冲噪声具有短时间突发性）
    pulse_noise = np.zeros_like(response_with_noise)
    pulse_times = np.random.choice(
        len(time), size=5, replace=False
    )  # 随机选择5个时刻加入脉冲
    pulse_magnitude = np.random.uniform(
        min_impulse, max_impulse, size=num_impulse
    )  # 随机脉冲幅度
    for pt, mag in zip(pulse_times, pulse_magnitude):
        pulse_noise[pt] = mag  # 在这些时刻加入脉冲噪声

    # 将脉冲噪声加入到TEM响应信号中
    response_with_noise_and_impulse = response_with_noise + pulse_noise

    time = time * 1000  # 转换为毫秒

    time = time[offset:]  # 去除前offset个采样点
    response = response[offset:]
    response_with_noise = response_with_noise[offset:]
    response_with_noise_and_impulse = response_with_noise_and_impulse[offset:]

    return (
        time,
        response,
        response_with_noise,
        response_with_noise_and_impulse,
    )


def get_tem_signal_realistic(
    noise_stddev: float = 500,
    min_impulse: float = -500,
    max_impulse: float = 1000,
    num_impulse: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成“更真实”的一维瞬变电磁(TEM)信号
    模拟地层扰动、采样抖动、1/f噪声、慢漂移和脉冲干扰
    """

    # ====== 地层参数（带随机扰动） ======
    thickness = [
        10.0 * np.random.uniform(0.9, 1.1),
        50.0 * np.random.uniform(0.9, 1.1),
        100.0 * np.random.uniform(0.9, 1.1),
    ]
    resistivity = [
        1e12,
        100.0 * np.random.uniform(0.8, 1.2),
        10.0 * np.random.uniform(0.8, 1.2),
        500.0 * np.random.uniform(0.8, 1.2),
        200.0 * np.random.uniform(0.8, 1.2),
    ]
    depth = [0.0] + list(np.cumsum(thickness))

    # ====== 源与接收参数 ======
    coil_length, coil_width = 30.0, 25.0
    src = [-coil_length / 2, -coil_width / 2, 0.0, coil_length / 2, coil_width / 2, 0.0]
    rec = [0.0, 0.0, 0.1, 0.0, 0.0]  # 点接收器略高于地面
    strength = 10.0 * coil_length * coil_width  # 电流×回线面积

    # ====== 时间采样（带采样抖动） ======
    offset = 10
    time = np.linspace(1e-20, 0.05, 400 + offset)
    time_jitter = time * (1 + np.random.normal(0, 0.002, size=time.shape))

    # ====== TEM响应计算 ======
    response = empymod.loop(
        src=src,
        rec=rec,
        depth=depth,
        res=resistivity,
        freqtime=time_jitter,
        signal=-1,  # 关断信号
        mrec=False,
        recpts=1,
        strength=strength,
        verb=0,
        htarg={"dlf": "key_201_2012"},
    )

    response = np.abs(response) * 1e9  # nT

    # ====== 加噪声 ======
    response_noisy = add_noise_stddev(response, noise_stddev)

    # ====== 输出整合 ======
    time_ms = time_jitter[offset:] * 1000
    response = response[offset:]
    response_noisy = response_noisy[offset:]

    return time_ms, response, response_noisy


if __name__ == "__main__":
    np.random.seed(None)
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import plot

    time, clean, noisy = get_tem_signal_realistic()

    print(len(time), len(clean), len(noisy))

    plot(
        time,
        clean,
        noisy,
        x_axis="time (ms)",
        y_axis="B (nT)",
    )
