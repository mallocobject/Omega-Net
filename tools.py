import numpy as np


# SNR
def snr(clean, denoised_signal):
    noise = clean - denoised_signal
    snr_value = 10 * np.log10(np.mean(clean**2) / np.mean(noise**2))
    return snr_value


# MSE
def mse(clean, denoised_signal):
    return np.mean((clean - denoised_signal) ** 2)
