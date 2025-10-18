import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from criterions import snr, mse
from utils import get_simple_tem_signal


class AdaptiveExtendedKalmanFilter:
    """
    自适应扩展卡尔曼滤波器
    """

    def __init__(self, n_states=3, n_obs=1):
        self.n_states = n_states
        self.n_obs = n_obs
        self.x = np.zeros(n_states)
        self.P = np.eye(n_states) * 0.1
        self.Q = np.eye(n_states) * 1e-6
        self.R = np.eye(n_obs) * 1e-7
        self.lambda_ = 0.98

    def state_transition_function(self, x, dt=1.0):
        A, beta, C = x
        A_new = A * np.exp(-beta * dt)
        beta_new = beta
        C_new = C
        return np.array([A_new, beta_new, C_new])

    def observation_function(self, x, t):
        A, beta, C = x
        return np.array([A * np.exp(-beta * t) + C])

    def state_jacobian(self, x, dt=1.0):
        A, beta, C = x
        jac = np.array(
            [
                [np.exp(-beta * dt), -A * dt * np.exp(-beta * dt), 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
        return jac

    def observation_jacobian(self, x, t):
        A, beta, C = x
        jac = np.array([[np.exp(-beta * t), -A * t * np.exp(-beta * t), 1]])
        return jac

    def initialize_parameters(self, initial_observations, initial_times):
        if len(initial_observations) < 3:
            self.x = np.array(
                [np.max(initial_observations), 0.1, np.min(initial_observations)]
            )
            return

        try:

            def exponential_model(t, A, beta, C):
                return A * np.exp(-beta * t) + C

            A0 = initial_observations[0]
            beta0 = 0.5
            C0 = initial_observations[-1]

            popt, _ = curve_fit(
                exponential_model,
                initial_times,
                initial_observations,
                p0=[A0, beta0, C0],
                maxfev=5000,
            )
            self.x = popt

        except Exception:
            A_guess = initial_observations[0]
            if len(initial_observations) > 4 and initial_observations[4] > 0:
                beta_guess = -np.log(
                    initial_observations[4] / initial_observations[0]
                ) / (initial_times[4] - initial_times[0])
            else:
                beta_guess = 0.1
            C_guess = (
                np.mean(initial_observations[-5:])
                if len(initial_observations) >= 5
                else initial_observations[-1]
            )
            self.x = np.array([A_guess, beta_guess, C_guess])

    def adapt_noise_covariances(self, innovation, innovation_covariance):
        innovation = innovation.flatten()
        actual_covariance = np.outer(innovation, innovation)
        innovation_diff = actual_covariance - innovation_covariance
        self.R = self.lambda_ * self.R + (1 - self.lambda_) * np.clip(
            innovation_diff, 1e-12, 1e-5
        )
        self.R = np.maximum(self.R, 1e-12)

    def filter(self, observations, time_steps):
        n_steps = len(observations)
        filtered_observations = np.zeros(n_steps)

        init_obs = observations[: min(10, len(observations))]
        init_time = time_steps[: len(init_obs)]
        self.initialize_parameters(init_obs, init_time)

        for k in range(n_steps):
            dt = (
                time_steps[k] - time_steps[k - 1]
                if k > 0
                else time_steps[1] - time_steps[0]
            )

            self.x = self.state_transition_function(self.x, dt)
            F = self.state_jacobian(self.x, dt)
            self.P = F @ self.P @ F.T + self.Q

            H = self.observation_jacobian(self.x, time_steps[k])
            z_pred = self.observation_function(self.x, time_steps[k])
            innovation = observations[k] - z_pred

            S = H @ self.P @ H.T + self.R
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ innovation
            self.P = (np.eye(self.n_states) - K @ H) @ self.P

            self.adapt_noise_covariances(innovation, S)
            filtered_observations[k] = self.observation_function(self.x, time_steps[k])[
                0
            ]

        return filtered_observations


class AKEKFTEMDenoiser:
    def __init__(self, n_states=3):
        self.n_states = n_states
        self.akef = AdaptiveExtendedKalmanFilter(n_states=n_states)

    def denoise(self, signal, time=None):
        if time is None:
            time = np.arange(len(signal))
        denoised_signal = self.akef.filter(signal, time)
        return denoised_signal


if __name__ == "__main__":
    np.random.seed(24)

    time, clean_sig, noisy_sig = get_simple_tem_signal()

    # AK-EKF去噪
    akekf_denoiser = AKEKFTEMDenoiser(n_states=3)
    denoised_data = akekf_denoiser.denoise(noisy_sig, time)

    # 显示AK-EKF去噪图
    plt.figure(figsize=(12, 8))

    plt.semilogy(time * 1e3, np.abs(clean_sig), "g-", linewidth=2, label="Clean Signal")
    plt.semilogy(
        time * 1e3,
        np.abs(noisy_sig),
        "r-",
        alpha=0.7,
        linewidth=1,
        label="Noisy Signal",
    )
    plt.semilogy(
        time * 1e3, np.abs(denoised_data), "b-", linewidth=2, label="AK-EKF Denoised"
    )

    plt.xlabel("Time (ms)")
    plt.ylabel("|dBz/dt| (V/m²)")
    plt.title("AK-EKF Denoising Result")
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    plt.show()
