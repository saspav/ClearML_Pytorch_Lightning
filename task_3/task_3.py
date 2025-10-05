# -*- coding: utf-8 -*-
"""
task_3.py

"""

import os
import argparse
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
import torchvision.utils as vutils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt

from clearml import Task, Logger
from getpass import getpass


# ===================== CFG ClearML =====================
@dataclass
class CFG:
    """
    Класс для хранения всех параметров конфигурации эксперимента.

    Атрибуты:
        project_name (str): Название проекта в ClearML.
        experiment_name (str): Название конкретного эксперимента в ClearML.
        data_dir (Path): Путь к папке с данными (скачивание/хранение).
        seed (int): Фиксированный seed для воспроизводимости экспериментов.
        batch_size (int): Размер батча для обучения и валидации.
        max_epochs (int): Количество эпох для обучения модели.
        num_workers (int): Количество потоков (воркеров) для загрузки данных.
        learning_rate (float): Скорость обучения оптимизатора.
        noise_dim (int): Размер входного вектора шума для генератора.
        debug_samples_epoch (int): Частота сохранения примеров сгенерированных изображений.
    """
    project_name: str = "GAN Project"
    experiment_name: str = "Lightning CV"
    data_dir: Path = Path("data")
    seed: int = 127
    batch_size: int = 512
    max_epochs: int = 30
    num_workers: int = 16
    learning_rate: float = 1e-5
    noise_dim: int = 100
    debug_samples_epoch: int = 1


# ===================== DataModule =====================
class MNISTDataModule(pl.LightningDataModule):
    """
    DataModule для загрузки и предобработки данных MNIST.

    Этот класс управляет всеми стадиями работы с данными:
    - prepare_data: скачивание датасета (вызывается 1 раз на всех процессах)
    - setup: разделение данных на train/val/test
    - предоставление DataLoader'ов для каждой стадии обучения
    """

    def __init__(self, _cfg: CFG):
        """
        Инициализация DataModule с конфигурацией и трансформациями.

        Args:
            _cfg (CFG): Конфигурация эксперимента.
        """
        super().__init__()
        self.cfg = _cfg

        # Последовательность преобразований изображений MNIST:
        # 1) Конвертируем изображение в тензор
        # 2) Нормализуем к диапазону [-1, 1], чтобы ускорить обучение GAN
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Переменные для хранения датасетов на разных стадиях
        self.mnist_train = self.mnist_valid = self.mnist_test = None

    def prepare_data(self):
        """
        Скачивает датасет MNIST, если он ещё не загружен.
        Этот метод вызывается один раз и не загружает данные в память.
        """
        datasets.MNIST(root=self.cfg.data_dir, train=True, download=True)
        datasets.MNIST(root=self.cfg.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """
        Разделяет датасет MNIST на train, validation и test наборы.

        Args:
            stage (str, optional): Стадия, для которой происходит подготовка данных.
                                   Может быть 'fit', 'validate', 'test' или None.
        """
        # Загружаем тренировочные данные
        if stage == "fit" or stage is None:
            self.mnist_train = datasets.MNIST(
                root=self.cfg.data_dir,
                train=True,
                transform=self.transform
            )

            # Используем весь тестовый набор в качестве валидационного
            self.mnist_valid = datasets.MNIST(
                root=self.cfg.data_dir,
                train=False,
                transform=self.transform
            )

        # Загружаем данные для теста
        if stage == "test":
            self.mnist_test = datasets.MNIST(
                root=self.cfg.data_dir,
                train=False,
                transform=self.transform
            )

    def train_dataloader(self):
        """
        Создаёт DataLoader для тренировочного набора.

        Returns:
            DataLoader: DataLoader для обучения модели.
        """
        return DataLoader(
            self.mnist_train,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,  # Обычно shuffle=True для обучения, можно включить
            persistent_workers=True
        )

    def val_dataloader(self):
        """
        Создаёт DataLoader для валидационного набора.

        Returns:
            DataLoader: DataLoader для валидации модели.
        """
        return DataLoader(
            self.mnist_valid,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            persistent_workers=True
        )

    def test_dataloader(self):
        """
        Создаёт DataLoader для тестового набора.

        Returns:
            DataLoader: DataLoader для финального тестирования модели.
        """
        return DataLoader(
            self.mnist_test,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            persistent_workers=True
        )


# ===================== Model =====================
class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.main = nn.Sequential(
            nn.Linear(noise_dim, 256 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            # -> (128, 14, 14)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, bias=False),
            # -> (1, 28, 28)
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            # -> (64, 14, 14)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            # -> (128, 7, 7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class GAN(pl.LightningModule):
    """
    GAN-модель на PyTorch Lightning для генерации изображений MNIST.

    Состоит из двух сетей:
        - Generator: генерирует изображения из случайного шума
        - Discriminator: различает реальные и сгенерированные изображения

    Модель использует ручную оптимизацию (manual optimization),
    чтобы отдельно обновлять веса дискриминатора и генератора.
    """

    def __init__(self, noise_dim=100, lr=0.0002, betas=(0.5, 0.999), debug_samples_epoch=1):
        """
        Инициализация GAN.

        Args:
            noise_dim (int): Размер входного вектора шума для генератора.
            lr (float): Скорость обучения для обоих оптимизаторов.
            betas (tuple): Параметры beta1 и beta2 для Adam (рекомендованы для GAN).
            debug_samples_epoch (int): Частота логирования изображений на валидации.
        """
        super(GAN, self).__init__()

        # Сохраняем гиперпараметры для логирования в Lightning
        self.save_hyperparameters()

        # Параметры модели
        self.noise_dim = noise_dim
        self.lr = lr
        self.betas = betas
        self.debug_samples_epoch = debug_samples_epoch

        # Определяем архитектуру GAN
        self.generator = Generator(noise_dim)  # Генератор
        self.discriminator = Discriminator()  # Дискриминатор

        # Функция потерь — бинарная кросс-энтропия
        self.criterion = nn.BCELoss()

        # Отключаем автоматическую оптимизацию в Lightning
        self.automatic_optimization = False

        # Логгер для ClearML
        self.log_clrml = Logger.current_logger()

    def forward(self, z):
        """
        Прямой проход через генератор.

        Args:
            z (Tensor): Входной шум размерности [batch_size, noise_dim].

        Returns:
            Tensor: Сгенерированные изображения.
        """
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        """
        Логика одной итерации обучения GAN.
        Вручную обновляет веса дискриминатора и генератора.

        Args:
            batch (tuple): Батч данных (реальные изображения и метки).
            batch_idx (int): Индекс батча.

        Returns:
            dict: Словарь с потерями дискриминатора и генератора.
        """
        # Получаем оптимизаторы для дискриминатора и генератора
        opt_d, opt_g = self.optimizers()

        # Разделяем изображения и метки (метки не нужны)
        real_images, _ = batch
        batch_size = real_images.size(0)
        device = real_images.device

        # Создаём тензоры меток для реальных и фейковых изображений
        real_labels = torch.full((batch_size,), 1.0, device=device)
        fake_labels = torch.full((batch_size,), 0.0, device=device)

        # ================== Шаг 1: обучение дискриминатора ==================
        opt_d.zero_grad()

        # 1.1. Предсказания для реальных изображений
        output_real = self.discriminator(real_images).view(-1)
        loss_real = self.criterion(output_real, real_labels)

        # 1.2. Предсказания для фейковых изображений
        noise = torch.randn(batch_size, self.noise_dim, device=device)
        fake_images = self.generator(noise)
        output_fake = self.discriminator(fake_images.detach()).view(-1)
        loss_fake = self.criterion(output_fake, fake_labels)

        # Общая потеря дискриминатора
        loss_D = loss_real + loss_fake

        # Обратное распространение ошибки
        self.manual_backward(loss_D)
        opt_d.step()

        # ================== Шаг 2: обучение генератора ======================
        opt_g.zero_grad()

        # Генерируем новый шум
        noise = torch.randn(batch_size, self.noise_dim, device=device)
        fake_images = self.generator(noise)
        output = self.discriminator(fake_images).view(-1)

        # Потеря генератора (хотим, чтобы дискриминатор считал фейки реальными)
        loss_G = self.criterion(output, real_labels)

        # Обратное распространение ошибки
        self.manual_backward(loss_G)
        opt_g.step()

        # ================== Логирование ==================
        self.log('loss_D', loss_D, prog_bar=True, on_epoch=True)
        self.log('loss_G', loss_G, prog_bar=True, on_epoch=True)

        # Логирование в ClearML
        self.log_clrml.report_scalar("Loss", "Discriminator", loss_D.item(),
                                     iteration=self.global_step)
        self.log_clrml.report_scalar("Loss", "Generator", loss_G.item(),
                                     iteration=self.global_step)

        return {"loss_D": loss_D, "loss_G": loss_G}

    def validation_step(self, batch, batch_idx):
        """
        Вызывается на каждом батче валидационного датасета.
        Здесь логируются сгенерированные изображения каждые N эпох.

        Args:
            batch (tuple): Батч данных (не используется для оценки качества).
            batch_idx (int): Индекс батча.
        """
        # Логируем изображения только на определённых эпохах
        if not self.current_epoch % self.hparams.debug_samples_epoch:
            device = self.device

            # Генерируем фиксированный шум для сравнения между эпохами
            fixed_noise = torch.randn(16, self.noise_dim, device=device)
            fake_images = self.generator(fixed_noise)

            # Превращаем батч изображений в сетку
            grid = vutils.make_grid(fake_images, normalize=True)

            # Конвертируем в PIL-формат для ClearML
            pil_img = transforms.ToPILImage()(grid.cpu())

            # Отправляем изображение в ClearML
            self.log_clrml.report_image(
                "Validation Images",
                f"Epoch {self.current_epoch}",
                iteration=self.global_step,
                image=pil_img
            )

        # Логирование фиктивной метрики для Lightning (чтобы не было ошибок)
        self.log("valid_none", 0, prog_bar=False)

    def configure_optimizers(self):
        """
        Определяет оптимизаторы для генератора и дискриминатора.

        Returns:
            tuple: Список оптимизаторов и пустой список планировщиков.
        """
        optimizerD = Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)
        optimizerG = Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        return [optimizerD, optimizerG], []


# ===================== Training =====================
def run_training(cfg: CFG, datamodule: MNISTDataModule):
    """
    Основной процесс обучения модели:

    Аргументы:
        cfg: конфигурация
        datamodule: экземпляр класса MNISTDataModule
    """

    # Оптимизация умножений матриц (PyTorch 2.0+)
    try:
        torch.set_float32_matmul_precision('high')
    except:
        pass

    # Инициализация модели
    model = GAN(noise_dim=cfg.noise_dim,
                lr=cfg.learning_rate,
                debug_samples_epoch=cfg.debug_samples_epoch
                )

    # Логгер Lightning (сохраняет метрики в CSV)
    csv_logger = CSVLogger("lightning_logs", name="GAN_experiment")

    # Определение устройства
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # ===================== Основное обучение =====================
    # чекпоинт для сохранения лучшей модели
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_DIR,
        filename="best_gan",
        save_top_k=1,
        monitor="loss_G",  # метрика для раннего стопа
        mode="min",  # минимизируем метрику
        save_weights_only=True  # сохраняем только веса
    )

    # ранний стоп
    early_stopping_callback = EarlyStopping(
        patience=5,  # если не улучшается X эпох подряд — останавливаем
        monitor="loss_G",
        mode="min",
        verbose=True
    )

    trainer = pl.Trainer(max_epochs=cfg.max_epochs,
                         accelerator=accelerator,
                         logger=csv_logger,
                         deterministic=True,
                         # добавляем колбэки
                         callbacks=[checkpoint_callback, early_stopping_callback]
                         )

    start = time.time()
    trainer.fit(model, datamodule)
    print(f"Обучение завершено за {time.time() - start:.1f} секунд")

    # ===================== Визуализация кривых обучения =====================
    metrics = pd.read_csv(csv_logger.log_dir + "/metrics.csv")

    plt.figure(figsize=(12, 5))

    # Loss curves
    # plt.subplot(1, 2, 1)
    plt.plot(metrics["loss_G_epoch"].dropna(), label="loss_G")
    plt.plot(metrics["loss_D_epoch"].dropna(), label="loss_D")
    plt.legend()
    plt.title("Loss")

    # Сохраняем график
    fig_path = MODEL_DIR / "training_curves.png"
    plt.savefig(fig_path)
    plt.close()

    # ===================== Отправка графика в ClearML =====================
    logger = Logger.current_logger()
    logger.report_image(
        title="Learning Curves",
        series="loss",
        local_path=str(fig_path),
    )


# ===================== Main =====================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10, help="Количество эпох обучения")
    parser.add_argument("--debug_samples_epoch", type=int, default=1,
                        help="Частота логирования (1 - каждую эпоху, 2 - каждую вторую и ...)")
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


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    args = parse_args()  # парсим аргументы

    check_clearml_env()  # читаем/устанавливаем переменные среды для clearml

    # ===================== Init ClearML =====================
    cml_cfg = CFG()

    if args.epoch:
        cml_cfg.max_epochs = args.epoch
    if args.debug_samples_epoch:
        cml_cfg.debug_samples_epoch = args.debug_samples_epoch

    save_cfg = cml_cfg.__dict__.copy()
    save_cfg['data_dir'] = str(save_cfg['data_dir'])

    # создаём Task (важно, чтобы он был виден глобально)
    task = Task.init(project_name=cml_cfg.project_name,
                     task_name=cml_cfg.experiment_name,
                     task_type=Task.TaskTypes.training)
    task.add_tags(["PyTorch-Lightning", "CV", "GAN"])
    task.connect(save_cfg)
    logger_clearml = Logger.current_logger()

    mnist_dm = MNISTDataModule(cml_cfg)  # готовим датасеты
    mnist_dm.prepare_data()
    mnist_dm.setup()

    run_training(cml_cfg, mnist_dm)  # обучаем модель

    print("✅ Завершаю Task'у...")
    task.close()
