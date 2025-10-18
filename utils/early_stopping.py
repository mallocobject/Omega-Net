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
        save_interval=1,
        verbose=True,
    ):
        """
        参数:
            accelerator: accelerate 管理器
            patience: 容忍验证集 loss 不改善的最大 epoch 数
            min_delta: 改善幅度阈值（小于此幅度视为无改善）
            save_mode: 是否保存模型
            save_path: 最优模型保存路径
            save_interval: 最少间隔多少个 epoch 才能再次保存模型
            verbose: 是否打印日志
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
        self.save_interval = save_interval
        self.verbose = verbose

        self.last_save_epoch = -save_interval  # 确保第一次能保存模型

        save_dir = os.path.dirname(save_path)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

    def __call__(self, val_loss, model, epoch):
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
            if self.save_mode and (epoch - self.last_save_epoch >= self.save_interval):
                self._save_checkpoint(val_loss, model)
                self.last_save_epoch = epoch
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
