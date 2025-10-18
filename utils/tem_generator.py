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
    thickness = [
        10.0,  # 覆盖层厚度：10米（如沙土）
        50.0,  # 导电层厚度：50米（如沉积层）
        100.0,  # 基底层厚度：100米（如花岗岩）
    ]
    resistivity = [
        1e12,  # 空气层（电阻率极高）
        100.0 * np.random.uniform(0.95, 1.05),  # 覆盖层（100 Ω·m，增加随机变化）
        10.0 * np.random.uniform(0.95, 1.05),  # 导电层（10 Ω·m，增加随机变化）
        500.0 * np.random.uniform(0.95, 1.05),  # 基底层（500 Ω·m，增加随机变化）
        200.0 * np.random.uniform(0.95, 1.05),  # 额外层（200 Ω·m，增加随机变化）
    ]
    depth = [0.0] + list(np.cumsum(thickness))  # 地层深度

    # 回线源（矩形回线）
    coil_length = 250.0
    coil_width = 250.0
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
    offset = 10
    time = np.linspace(1e-3, 0.4, 400 + offset)

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

    response = response[offset:]
    time = time[offset:]

    response = response * 1e9  # 转换为 nT
    response = response - 2000  # 增加直流

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

    return response, response_with_noise, response_with_noise_and_impulse


if __name__ == "__main__":
    np.random.seed(None)
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import plot

    time, response, response_with_noise = get_simple_tem_signal()
