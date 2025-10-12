import numpy as np
import empymod
import matplotlib.pyplot as plt
import os
import sys

from .data_provider import add_noise_stddev


def get_tem_signal(
    noise_stddev: float,
    min_impulse: float,
    max_impulse: float,
    num_impulse: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成一维瞬变电磁(TEM)信号
    使用empymod库模拟更接近真实的1D地层的瞬变电磁响应
    返回:
        time (np.ndarray): 时间采样点(秒)
        response (np.ndarray): TEM信号(dB/dt,磁场变化率)
        response_with_noise (np.ndarray): 含噪声的TEM信号(dB/dt,磁场变化率)
        response_with_noise_and_impulse (np.ndarray): 含噪声和脉冲干扰的TEM信号(dB/dt,磁场变化率)
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
    coil_length = 300.0  # 回线长度：300米
    coil_width = 250.0  # 回线宽度：250米
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
        mrec=True,  # 计算磁场变化率（dB/dt）
        recpts=1,  # 单个接收点
        strength=strength,  # 设置源强度
        verb=1,  # 输出较少信息（提高计算效率）
        htarg={"dlf": "key_201_2012"},  # Hankel变换滤波器，确保数值稳定性
    )

    response = response[offset:]
    time = time[offset:]

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

    return time, response, response_with_noise, response_with_noise_and_impulse


def plot_tem_signal(time: np.ndarray, signal: np.ndarray, ax: plt.Axes, label: str):
    """
    绘制瞬变电磁信号,单位转换为nT
    """
    # 将信号从 V/m² 转换为 nT
    scaling_factor = 1e5  # 1e4 * nT

    ax.plot(
        time * 1e3,
        np.abs(signal) * scaling_factor,
        label=label,
        linewidth=2,
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("dBz/dt (nT * 1e4)")
    ax.set_title(label)
    ax.grid(True, which="both", ls="--", alpha=0.7)


if __name__ == "__main__":
    np.random.seed(None)
    time, response, response_with_noise, response_with_noise_and_impulse = (
        get_tem_signal(5e-7, -5e-6, 5e-6)
    )

    # 创建一个包含三个子图的窗口
    fig, axs = plt.subplots(3, 1, figsize=(6, 12))  # 三行一列的子图

    # 在不同的子图中绘制不同的信号
    plot_tem_signal(time, response, axs[0], "TEM Response")
    plot_tem_signal(time, response_with_noise, axs[1], "TEM Response with Noise")
    plot_tem_signal(
        time,
        response_with_noise_and_impulse,
        axs[2],
        "TEM Response with Noise and Impulse",
    )

    # 显示图形
    plt.tight_layout()  # 调整子图间距
    plt.show()
