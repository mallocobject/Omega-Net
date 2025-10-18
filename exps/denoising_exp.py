import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.rich import tqdm
from accelerate import Accelerator
import torch.distributed as dist
import numpy as np
import time
import warnings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TEMDnet, SFSDSA, TEMSGnet
from data import TEMDataset
from utils import EarlyStopping
from criterions import MSECriterion, MSECriterionWithNoise


class DenoisingExperiment:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.accelerator = Accelerator()

        self.model_dict = {
            "temdnet": TEMDnet,
            "sfsdsa": SFSDSA,
            "temsgnet": TEMSGnet,
        }

    def _build_model(self):
        model = self.model_dict[self.args.model](stddev=self.args.stddev)
        return model

    def _get_dataloader(self, split: str):
        dataset = TEMDataset(data_dir=self.args.data_dir, split=split)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=(split == "train"),
            num_workers=2,
        )
        return dataloader

    def _select_criterion(self):
        if self.args.model == "temsgnet" or self.args.model == "temdnet":
            criterion = MSECriterionWithNoise()
        elif self.args.model == "sfsdsa":
            criterion = MSECriterion()
        else:
            raise ValueError(f"Unknown model type: {self.args.model}")
        return criterion

    def _select_optimizer(self, model: nn.Module):
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.regularizer,
        )
        return optimizer

    def _select_scheduler(self, optimizer: optim.Optimizer):
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.args.lr_step, gamma=self.args.lr_decay
        )
        return scheduler

    def _check_early_stop(self, early_stopping):
        stop_tensor = torch.tensor(0, device=self.accelerator.device)
        if self.accelerator.is_main_process:
            if early_stopping.early_stop:
                stop_tensor.fill_(1)
        # 同步所有进程
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(stop_tensor, src=0)
        return stop_tensor.item() == 1

    def train(self):
        train_dataloader = self._get_dataloader("train")
        valid_dataloader = self._get_dataloader("valid")

        model = self._build_model()

        train_criterion = self._select_criterion()
        valid_criterion = self._select_criterion()

        optimizer = self._select_optimizer(model)
        scheduler = self._select_scheduler(optimizer)

        model, train_criterion, optimizer, train_dataloader, valid_dataloader = (
            self.accelerator.prepare(
                model, train_criterion, optimizer, train_dataloader, valid_dataloader
            )
        )

        early_stopping = EarlyStopping(
            accelerator=self.accelerator,
            patience=15,
            delta=0.0,
            save_mode=False,
            save_path=os.path.join(self.args.ckpt_dir, f"{self.args.model}_best.pth"),
            save_interval=10,
            verbose=True,
        )

        vali_loss = 0.0  # 保存最后一次验证集损失
        for epoch in range(self.args.epochs):
            start_time = time.time()
            model.train()
            progress_bar = tqdm(
                train_dataloader,
                desc=f"[bold cyan]Training Epoch {epoch+1}",
                unit="batch",
                colour="magenta",
                disable=not self.accelerator.is_local_main_process,
            )
            for x, label in progress_bar:
                x, label = x.to(self.accelerator.device), label.to(
                    self.accelerator.device
                )
                time_emb = torch.randint(
                    0,
                    self.args.time_steps,
                    (x.size(0),),
                    device=self.accelerator.device,
                )

                optimizer.zero_grad()
                outputs = model(x, time_emb)
                loss = train_criterion(x.detach(), outputs, label)
                self.accelerator.backward(loss)
                optimizer.step()

            scheduler.step()

            vali_loss = self.validate(model, valid_dataloader, valid_criterion)
            early_stopping(vali_loss, model, epoch)
            end_time = time.time()
            self.accelerator.print(
                f"Epoch [{epoch+1}/{self.args.epochs}] | Validation Loss: {vali_loss:.6f} | Time: {end_time - start_time:.2f}s"
            )

            if self._check_early_stop(early_stopping):
                self.accelerator.print(
                    "⏹ Early stopping triggered — stopping training on all devices."
                )
                break

        early_stopping._save_checkpoint(vali_loss, model)

    def validate(self, model, valid_dataloader, valid_criterion):
        total_loss = []
        with torch.no_grad():
            model.eval()
            for x, label in valid_dataloader:
                x, label = x.to(self.accelerator.device), label.to(
                    self.accelerator.device
                )
                time_emb = torch.randint(
                    0,
                    self.args.time_steps,
                    (x.size(0),),
                    device=self.accelerator.device,
                )

                outputs = model(x, time_emb)
                loss = valid_criterion(x.detach(), outputs, label)
                total_loss.append(loss.item())

        vali_loss = np.average(total_loss)
        return vali_loss

    def test(self):
        test_dataloader = self._get_dataloader("test")

        model = self._build_model()
        test_criterion = self._select_criterion()

        # ====== 加载 checkpoint ======
        model.load_state_dict(
            torch.load(self.args.load_checkpoint, weights_only=True, map_location="cpu")
        )

        # prepare（保证设备、DDP兼容）
        model, test_criterion, test_dataloader = self.accelerator.prepare(
            model, test_criterion, test_dataloader
        )

        # ====== 测试阶段 ======
        model.eval()
        total_loss = []

        with torch.no_grad():
            for x, label in tqdm(
                test_dataloader,
                desc="[bold cyan]Testing",
                disable=not self.accelerator.is_local_main_process,
            ):
                x, label = x.to(self.accelerator.device), label.to(
                    self.accelerator.device
                )
                time_emb = torch.randint(
                    0,
                    self.args.time_steps,
                    (x.size(0),),
                    device=self.accelerator.device,
                )

                outputs = model(x, time_emb)
                loss = test_criterion(x, outputs, label)
                total_loss.append(loss.item())

        test_loss = np.mean(total_loss)
        self.accelerator.print(f"✅ Test Loss: {test_loss:.6f}")
