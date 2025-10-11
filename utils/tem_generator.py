import numpy as np
import empymod
import matplotlib.pyplot as plt


def get_tem_signal() -> np.ndarray:
    """
    生成一维瞬变电磁(TEM)信号
    使用empymod库模拟1D地层的瞬变电磁响应
    """
    # 地层参数
    thickness = [100.0, 50.0]
    resistivity = [1e12, 100.0, 10.0, 100.0]
    depth = [0.0] + list(np.cumsum(thickness))  # [0.0, 100.0, 150.0]

    # 回线源
    coil_length = 300
    coil_width = 280
    src = [
        -coil_length // 2,
        -coil_width // 2,
        0.0,
        coil_length // 2,
        coil_width // 2,
        0.0,
    ]  # 矩形回线，面积 300 x 280 m² (TEM-NLnet)

    # 接收器
    rec = [0.0, 0.0, 0.0, 0.0, 0.0]  # 中心点接收器

    # 时间采样
    time = np.linspace(1e-5, 0.3, 400)  # 1e-5 到 0.3 秒，400 个采样点

    # 计算响应
    response = empymod.loop(
        src=src,  # 源位置（回线）
        rec=rec,  # 接收器位置
        depth=depth,  # 地层深度
        res=resistivity,  # 电阻率
        freqtime=time,  # 时间域
        signal=-1,  # 断开信号
        mrec=True,  # 磁场变化率
        recpts=1,
        strength=300 * 280.0,  # 考虑实际回线面积
        verb=3,  # 输出详细信息
    )
    return response, time


def plot_tem_signal(signal: np.ndarray, time: np.ndarray):
    """
    绘制瞬变电磁信号
    """
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.semilogy(
        time * 1e3, np.abs(signal), "b-", linewidth=2, label="Simulated TEM Response"
    )
    plt.xlabel("Time (ms)")
    plt.ylabel("dBz/dt (V/m²)")
    plt.title("1D Transient Electromagnetic Response")
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    signal, time = get_tem_signal()
    plot_tem_signal(signal, time)
