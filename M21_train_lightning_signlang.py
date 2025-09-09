# -*- coding: utf-8 -*-
"""
train_lightning_signlang.py

Перенос обучения свёрточной сети для Sign Language MNIST на PyTorch Lightning
с поддержкой fast_dev_run, ClearML и визуализации метрик.
"""

import os
import argparse
import sys
import time
import zipfile
import urllib.request
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.utils.data as data
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

import matplotlib.pyplot as plt

from clearml import Task, Logger
from getpass import getpass


# ===================== CFG =====================
@dataclass
class CFG:
    project_name: str = "SignLanguage Project"
    experiment_name: str = "Lightning CNN"
    seed: int = 127
    batch_size: int = 512
    max_epochs: int = 20
    num_workers: int = 16
    data_url_train: str = "https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/sign_mnist_train.csv.zip"
    data_url_test: str = "https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/sign_mnist_test.csv.zip"
    data_dir: Path = Path("data")


cfg = CFG()
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ===================== ClearML =====================
save_cfg = cfg.__dict__.copy()
save_cfg['data_dir'] = str(save_cfg['data_dir'])
task = Task.init(project_name=cfg.project_name, task_name=cfg.experiment_name)
task.add_tags(["PyTorch-Lightning", "CNN", "SignLanguage"])
task.connect(save_cfg)
logger_clearml = Logger.current_logger()


# ===================== Dataset & DataModule =====================
class SignLanguageDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row.iloc[0])
        pixels = row.iloc[1:].values.astype(np.float32).reshape(28, 28)
        img = torch.tensor(pixels).unsqueeze(0)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class SignLanguageDataModule(pl.LightningDataModule):
    def __init__(self, train_csv: str, test_csv: str, batch_size: int, num_workers: int = 4):
        super().__init__()
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.1),
             transforms.RandomApply([transforms.RandomRotation(degrees=(-180, 180))], p=0.2),
             ]
        )
        self.test_transform = None

    def setup(self, stage=None):
        train_df = pd.read_csv(self.train_csv)
        test_df = pd.read_csv(self.test_csv)
        self.train_dataset = SignLanguageDataset(train_df, transform=self.train_transform)
        self.valid_dataset = SignLanguageDataset(test_df, transform=self.test_transform)
        self.test_dataset = self.valid_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True)


# ===================== Model =====================
class MyConvNet(nn.Module):
    def __init__(self, n_classes=25):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.AvgPool2d(2),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(2),
            nn.ReLU(),
        )
        self.lin1 = nn.Linear(16 * 7 * 7, 100)
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(0.3)
        self.lin2 = nn.Linear(100, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.lin2(x)
        return x


class LitSignLang(pl.LightningModule):
    def __init__(self, lr=1e-3, n_classes=25):
        super().__init__()
        self.save_hyperparameters()
        self.model = MyConvNet(n_classes=n_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        logger_clearml.report_scalar("train_loss", "epoch", loss.item(), self.current_epoch)
        logger_clearml.report_scalar("train_acc", "epoch", acc.item(), self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("valid_loss", loss, on_epoch=True, prog_bar=True)
        self.log("valid_acc", acc, on_epoch=True, prog_bar=True)
        logger_clearml.report_scalar("valid_loss", "epoch", loss.item(), self.current_epoch)
        logger_clearml.report_scalar("valid_acc", "epoch", acc.item(), self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ===================== Dataset Loader =====================
def download_and_extract(url: str, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / Path(url).name
    if not filename.exists():
        print(f"Скачиваю {url}...")
        urllib.request.urlretrieve(url, filename)
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(save_dir)
    print(f"Файл {filename} распакован в {save_dir}")


def maybe_load_dataset(load_dataset: bool):
    train_csv = cfg.data_dir / "sign_mnist_train.csv"
    test_csv = cfg.data_dir / "sign_mnist_test.csv"
    if load_dataset:
        download_and_extract(cfg.data_url_train, cfg.data_dir)
        download_and_extract(cfg.data_url_test, cfg.data_dir)
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("Не найдены датасеты. Запустите с --load_dataset True")
    return str(train_csv), str(test_csv)


# ===================== Training =====================
def run_training(args):
    train_csv, test_csv = maybe_load_dataset(args.load_dataset)
    dm = SignLanguageDataModule(train_csv, test_csv, batch_size=cfg.batch_size)
    dm.setup()

    try:
        torch.set_float32_matmul_precision('high')
    except:
        pass

    model = LitSignLang()

    csv_logger = CSVLogger("lightning_logs", name="signlang_experiment")

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # Fast dev run
    if args.fast_dev_run:
        print("Тестовый прогон...")
        trainer = pl.Trainer(accelerator=accelerator, fast_dev_run=True, logger=csv_logger)
        try:
            trainer.fit(model, dm)
            print("Тестовый прогон успешно пройден")
        except Exception as e:
            print("Тестовый прогон завершился с ошибкой:", e)
            sys.exit(1)

    # Основное обучение

    trainer = pl.Trainer(max_epochs=cfg.max_epochs,
                         accelerator=accelerator,
                         logger=csv_logger,
                         deterministic=True,
                         )
    start = time.time()
    trainer.fit(model, dm)
    print(f"Обучение завершено за {time.time() - start:.1f} секунд")

    # Сохраняем веса
    ckpt_path = MODEL_DIR / "myconvnet_sign_lang.ckpt"
    trainer.save_checkpoint(str(ckpt_path))
    print(f"Модель сохранена: {ckpt_path}")

    # Визуализация кривых обучения
    metrics = pd.read_csv(csv_logger.log_dir + "/metrics.csv")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(metrics["train_loss_epoch"].dropna(), label="train_loss")
    plt.plot(metrics["valid_loss"].dropna(), label="valid_loss")
    plt.legend()
    plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(metrics["train_acc_epoch"].dropna(), label="train_acc")
    plt.plot(metrics["valid_acc"].dropna(), label="valid_acc")
    plt.legend()
    plt.title("Accuracy")
    fig_path = MODEL_DIR / "training_curves.png"
    plt.savefig(fig_path)
    plt.close()

    # Отправка в ClearML
    logger = Logger.current_logger()
    logger.report_image(
        title="Learning Curves",
        series="Logloss",
        local_path=str(fig_path),
    )

    # Инференс на одном примере
    test_loader = dm.test_dataloader()
    imgs, labels = next(iter(test_loader))
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(imgs), dim=1)
    print(f"Пример инференса: true={labels[0]}, pred={preds[0]}")


# ===================== Main =====================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast_dev_run", type=lambda x: str(x).lower() in ["true", "1"],
                        default=False)
    parser.add_argument("--load_dataset", type=lambda x: str(x).lower() in ["true", "1"],
                        default=False)
    return parser.parse_args()


def check_clearml_env():
    required_env_vars = [
        "CLEARML_WEB_HOST",
        "CLEARML_API_HOST",
        "CLEARML_FILES_HOST",
        "CLEARML_API_ACCESS_KEY",
        "CLEARML_API_SECRET_KEY"
    ]
    env_vars = dict(CLEARML_WEB_HOST="https://app.clear.ml/",
                    CLEARML_API_HOST="https://api.clear.ml",
                    CLEARML_FILES_HOST="https://files.clear.ml"
                    )

    for var in required_env_vars[:3]:
        if os.getenv(var) is None:
            os.environ[var] = env_vars[var]

    missing_vars = [var for var in required_env_vars if os.getenv(var) is None]

    if missing_vars:
        print("⚠️  Некоторые переменные среды ClearML отсутствуют.")
        for var in missing_vars:
            os.environ[var] = getpass(f"Введите значение для {var}: ")
        print("✅ Все переменные ClearML установлены.\n")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


if __name__ == "__main__":
    args = parse_args()
    check_clearml_env()
    run_training(args)
    # Не забываем завершить Task'у
    print("✅ Завершаю Task'у...")
    task.close()
