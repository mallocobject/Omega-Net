import numpy as np
import torch
import os


class EarlyStopping:
    """
    TEM一维信号去噪任务的早停机制(兼容 accelerate + 大尺度 loss)
    """

    def __init__(
        self,
        accelerator,
        patience=10,
        delta=0.0,
        save_mode=True,
        save_path=None,
        verbose=True,
    ):
        """
        参数:
            accelerator: accelerate.Accelerator 对象，用于分布式控制
            patience (int): 容忍验证集loss不改善的最大epoch数;若设为 -1 则禁用早停
            delta (float): 最小改善阈值（越大，早停越严格）
            save_mode (bool): 是否保存模型
            save_dir (str): 模型保存目录
            verbose (bool): 是否打印日志
        """
        self.accelerator = accelerator
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_mode = save_mode
        self.save_path = save_path
        self.verbose = verbose

        save_dir = os.path.dirname(save_path)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

    def __call__(self, val_loss, model):
        """每个验证阶段后调用"""
        # 检查NaN
        if np.isnan(val_loss):
            self.early_stop = True
            if self.verbose:
                self.accelerator.print(
                    "⚠️ Validation loss is NaN. Early stopping immediately."
                )
            return

        score = -val_loss  # 越低越好

        if self.best_score is None:
            # 第一次记录
            self.best_score = score
            if self.save_mode:
                self._save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            # 没有明显改善
            self.counter += 1
            if self.verbose:
                self.accelerator.print(
                    f"⚠️ No improvement for {self.counter}/{self.patience} epochs."
                )
            if self.patience != -1 and self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.accelerator.print("⏹ Early stopping triggered.")
        else:
            # 改善了
            self.best_score = score
            if self.save_mode:
                self._save_checkpoint(val_loss, model)
            self.counter = 0

    def _save_checkpoint(self, val_loss, model):
        """保存模型（仅主进程执行）"""
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                self.accelerator.print(
                    f"✅ Validation loss improved to {val_loss:.6f}. Model saved to {self.save_path}."
                )
        self.val_loss_min = val_loss
