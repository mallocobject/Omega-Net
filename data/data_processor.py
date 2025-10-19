import torch


def normalize_signal(x, method="zscore"):
    """
    对TEM信号标准化
    Args:
        x: torch.Tensor [B, L] 或 [L]
        method: "zscore" 或 "minmax"
    Returns:
        x_norm: 归一化信号
        params: dict(保存反标准化所需参数)
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)  # 转为 [1, L]

    if method == "zscore":
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_norm = (x - mean) / (std + 1e-8)
        params = {"mean": mean, "std": std}

    elif method == "minmax":
        x_min = x.min(dim=-1, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0]
        x_norm = 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1
        params = {"x_min": x_min, "x_max": x_max}

    else:
        raise ValueError("method must be 'zscore' or 'minmax'")

    return x_norm, params


def denormalize_signal(x_norm, params, method="zscore"):
    """
    反标准化（恢复真实幅值）
    """
    if method == "zscore":
        return x_norm * params["std"] + params["mean"]
    elif method == "minmax":
        return (x_norm + 1) / 2 * (params["x_max"] - params["x_min"]) + params["x_min"]
