from typing import List, Optional, Tuple
import numpy as np
import pywt
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


class SixLevelWaveletTransform:
    """
    六层小波变换类

    支持的小波基函数:
    - 'db1' 到 'db20': Daubechies 小波
    - 'sym2' 到 'sym20': Symlets 小波
    - 'coif1' 到 'coif5': Coiflets 小波
    - 'bior1.1' 到 'bior6.8': 双正交小波
    - 'rbio1.1' 到 'rbio6.8': 反向双正交小波
    """

    def __init__(self, wavelet: str = "db4", mode: str = "symmetric"):
        self.wavelet = wavelet
        self.mode = mode
        self.max_levels = 6

        # 存储变换结果
        self.coeffs = None
        self.original_signal = None
        self.reconstructed_signal = None

    def _validate_signal(self, signal: np.ndarray) -> np.ndarray:
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)
            logger.info("输入信号已转换为numpy数组")

        if signal.size == 0:
            raise ValueError("输入信号不能为空")

        # 确保是一维信号
        original_shape = signal.shape
        if signal.ndim > 1:
            signal = signal.flatten()

        # 检查信号长度是否足够
        min_length = 2**self.max_levels
        if len(signal) < min_length:
            # 选择调整层数
            self.max_levels = int(np.log2(len(signal)))

        # 检查数值有效性
        if np.any(np.isnan(signal)):
            signal = np.nan_to_num(signal)

        if np.any(np.isinf(signal)):
            signal = np.clip(signal, -1e10, 1e10)

        return signal

    def decompose(self, signal: np.ndarray, level: int = 6) -> List[np.ndarray]:
        signal = self._validate_signal(signal)
        self.original_signal = signal.copy()

        # 验证分解层数
        if not isinstance(level, int) or level < 1 or level > self.max_levels:
            level = self.max_levels

        # 执行小波分解
        self.coeffs = pywt.wavedec(signal, self.wavelet, level=level, mode=self.mode)

        return self.coeffs

    def reconstruct(
        self, coeffs: Optional[List[np.ndarray]] = None, level: Optional[int] = None
    ) -> np.ndarray:
        if coeffs is None:
            if self.coeffs is None:
                raise ValueError("没有可用的系数，请先执行分解")
            coeffs = self.coeffs

        # 验证系数
        if not isinstance(coeffs, list) or len(coeffs) == 0:
            raise ValueError("无效的小波系数")

        # 部分重构（只使用前level+1个系数）
        if level is not None and 0 < level < len(coeffs):
            reconstruct_coeffs = coeffs[: level + 1]
            # 将不需要的细节系数置零
            for i in range(level + 1, len(coeffs)):
                reconstruct_coeffs.append(np.zeros_like(coeffs[i]))
        else:
            reconstruct_coeffs = coeffs

        # 执行重构
        self.reconstructed_signal = pywt.waverec(
            reconstruct_coeffs, self.wavelet, mode=self.mode
        )

        # 确保重构信号长度与原始信号一致
        if self.original_signal is not None:
            if len(self.reconstructed_signal) != len(self.original_signal):
                # 截断或填充以匹配原始长度
                if len(self.reconstructed_signal) > len(self.original_signal):
                    self.reconstructed_signal = self.reconstructed_signal[
                        : len(self.original_signal)
                    ]
                else:
                    pad_length = len(self.original_signal) - len(
                        self.reconstructed_signal
                    )
                    self.reconstructed_signal = np.pad(
                        self.reconstructed_signal, (0, pad_length), "constant"
                    )

        return self.reconstructed_signal

    def denoise(
        self, signal: np.ndarray, threshold: float = 0.1, method: str = "soft"
    ) -> np.ndarray:
        # 执行分解
        coeffs = self.decompose(signal)

        # 计算阈值（使用通用阈值）
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # 噪声估计
        universal_threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        actual_threshold = threshold * universal_threshold

        # 应用阈值（保留近似系数，对细节系数应用阈值）
        thresholded_coeffs = [coeffs[0]]  # 保留近似系数
        for i in range(1, len(coeffs)):
            detail_coeff = coeffs[i]
            if method == "soft":
                # 软阈值
                thresholded_coeff = np.sign(detail_coeff) * np.maximum(
                    np.abs(detail_coeff) - actual_threshold, 0
                )
            else:
                # 硬阈值
                thresholded_coeff = detail_coeff * (
                    np.abs(detail_coeff) > actual_threshold
                )

            thresholded_coeffs.append(thresholded_coeff)

        # 重构去噪信号
        denoised_signal = self.reconstruct(thresholded_coeffs)
        return denoised_signal


class ExponentialFitterWithAbruptDetection:
    """带突变点检测的指数拟合器"""

    def __init__(self, base_line: str = "non_linear"):
        self.abrupt_indices = None
        self.fitted_parameters = None
        self.fitted_curve = None
        if base_line == "linear" or base_line == "non_linear":
            self.base_line = base_line
        else:
            raise ValueError("base_line 参数必须是 'linear' 或 'non-linear'")

    @staticmethod
    def exponential_func(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """指数函数: y = a * exp(-b * x) + c"""
        return a * np.exp(-b * x) + c

    def detect_abrupt_changes(
        self,
        x: np.ndarray,
        threshold_factor: float = 0.2,
    ) -> np.ndarray:
        # 创建线性参考线
        x_points = [0, len(x) - 1]
        y_points = [x[0], x[-1]]
        if self.base_line == "linear":
            y_numpy = np.interp(np.arange(len(x)), x_points, y_points)
            # 计算与线性参考的偏差
            x_datum = x - y_numpy
            p = np.ones_like(x_datum) * (x_datum[0] ** 2)
            for i in range(1, len(x_datum)):
                p[i] = p[i - 1] + threshold_factor * (x_datum[i] ** 2 - p[i - 1])

        elif self.base_line == "non_linear":  # 非线性参考线
            # k[n] = (ln x[n] - ln x[n-1]) / (n - (n-1)) = ln(x[n]/x[n-1])
            k = np.zeros(len(x) - 1, dtype=x.dtype)
            for i in range(1, len(x)):
                k[i - 1] = np.log(x[i] / x[i - 1])  # x[n] > 0

        if self.base_line == "linear":
            # 计算平均能量
            P = np.mean(np.abs(x_datum) ** 2)
            abrupt_index = p > P
        else:
            P = 0.1 * np.mean(np.abs(k))
            abrupt_index = k > P

        return abrupt_index

    def fit_exponential(
        self, x: np.ndarray, init_size: int = 10, increase_by: float = 10**0.1
    ) -> np.ndarray:

        current_size = init_size
        start_index = 0

        while start_index <= len(x):
            # 获取当前窗口数据
            end_index = min(start_index + current_size, len(x))
            win_data = x[start_index:end_index]

            if len(win_data) < 2:  # 窗口太小无法检测
                break

            # 检测当前窗口的突变点
            win_abrupt = self.detect_abrupt_changes(win_data)

            # 将窗口内的突变点映射到全局索引
            global_indices = np.arange(start_index, start_index + len(win_abrupt))
            win_abrupt_indices = global_indices[win_abrupt]
            win_non_abrupt_indices = global_indices[~win_abrupt]

            t = win_non_abrupt_indices
            y = x[win_non_abrupt_indices]

            p0 = [np.max(y), np.median(t), np.min(y)]  # 初始猜测参数
            params, _ = curve_fit(self.exponential_func, t, y, p0=p0, maxfev=10000)
            self.fitted_parameters = params
            self.fitted_curve = self.exponential_func(np.arange(len(x)), *params)

            for idx in win_abrupt_indices:
                x[idx] = self.fitted_curve[idx]

            start_index += int(current_size)
            current_size = int(current_size * increase_by)

        return x

    def predict(self, x_new: np.ndarray) -> np.ndarray:
        """使用拟合参数进行预测"""
        if self.fitted_parameters is None:
            raise ValueError("没有可用的拟合参数，请先执行拟合")

        return self.exponential_func(x_new, *self.fitted_parameters)


class WEF:
    def __init__(
        self,
        wavelet: str = "db4",
        mode: str = "symmetric",
        base_line: str = "non_linear",
    ):
        self.swt = SixLevelWaveletTransform(wavelet=wavelet, mode=mode)
        self.ekf = ExponentialFitterWithAbruptDetection(base_line)

    def denoise(self, signal: np.ndarray, threshold: float = 0.1, method: str = "soft"):
        x = self.swt.denoise(signal, threshold=threshold, method=method)
        x = self.ekf.fit_exponential(x)
        return x


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import utils.data_provider as dp
    import utils.tem_generator as tg

    np.random.seed(24)
    print("=== WEF去噪测试 ===")
    print("使用小波基函数: db4, 对瞬变电磁信号进行去噪")
    print("小波阈值: 0.1, 软阈值处理")

    # 生成测试数据
    clean_data, time = tg.get_tem_signal()
    noisy_data = clean_data + dp.get_noise(noise_level=1e-7, noise_size=len(clean_data))

    wef = WEF(wavelet="db4", mode="symmetric", base_line="non_linear")
    denoised_data = wef.denoise(noisy_data, threshold=0.1, method="soft")
    print(f"Clean data shape: {clean_data.shape}")
    print(f"Noisy data shape: {noisy_data.shape}")
    print(f"Denoised data shape: {denoised_data.shape}")
    print("\n=== 可视化结果 ===")

    # 可视化结果
    plt.figure(figsize=(12, 8))

    plt.semilogy(
        time * 1e3, np.abs(clean_data), "g-", linewidth=2, label="Clean Signal"
    )
    plt.semilogy(
        time * 1e3, np.abs(noisy_data), "r-", linewidth=1, label="Noisy Signal"
    )
    plt.semilogy(
        time * 1e3, np.abs(denoised_data), "b-", linewidth=2, label="Denoised Signal"
    )
    plt.xlabel("Time (ms)")
    plt.ylabel("dBz/dt (V/m²)")
    plt.title("WEF Denoising Result")
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    plt.show()

    from tools import snr, mse

    print(f"SNR Improvement: {snr(clean_data, denoised_data):.2f} dB")
    print(f"MSE: {mse(clean_data, denoised_data):.6f}")
